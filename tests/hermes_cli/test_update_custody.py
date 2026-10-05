"""R2: every updater git runner shares ONE custody policy (hermes_cli.update_custody).

Real git, real processes, no mocks:
* every runner passes ``-c gc.autoDetach=false -c maintenance.auto=false`` (git reports the
  effective config it was started with), so no detached gc/maintenance child is ever forked;
* POSIX: the checkout lock fd reaches only local mutators — a ``git fetch``'s upload-pack (where a
  credential-cache daemon would hang) never holds it, a ``git stash push``'s clean filter does.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import update_lock as ul

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", env=_env(cwd)).stdout.strip()


def _env(base: Path) -> dict:
    return dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for key, value in _env(tmp_path).items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    (root / "f.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "f.txt")
    _git(root, "commit", "-qm", "one")
    return root


def _runners(repo: Path):
    """Every updater git runner, as ``args -> stdout``."""
    from hermes_cli import gitlock, update_cmd, update_cmd_check, update_cmd_git, update_cmd_stash

    yield "update_cmd._git_run", lambda a: update_cmd._git_run(["git"], a, cwd=repo).stdout
    yield "update_cmd_git._git_run", lambda a: update_cmd_git._git_run(["git"], a, cwd=repo).stdout
    yield "update_cmd_stash._git_quiet", lambda a: update_cmd_stash._git_quiet(["git"], a, repo, text=True).stdout
    yield "update_cmd_check._git", lambda a: update_cmd_check._git(["git"], repo, a).stdout
    yield "gitlock._git_stdout_lines", lambda a: "\n".join(gitlock._git_stdout_lines(repo, a))


def test_every_updater_git_runner_forbids_detached_children(repo):
    missing = {}
    for name, run in _runners(repo):
        seen = {key: run(["config", "--get", key]).strip() for key in ("gc.autoDetach", "maintenance.auto")}
        if seen != {"gc.autoDetach": "false", "maintenance.auto": "false"}:
            missing[name] = seen
    assert not missing, f"updater git runners that may fork a detached gc/maintenance child: {missing}"


# Records whether the checkout lock file is among this process's open fds, then hands over.
_RECORDER = textwrap.dedent("""\
    #!/bin/sh
    lock="$HERMES_TEST_LOCK"; out="$HERMES_TEST_OUT"
    held=no
    for fd in /proc/$$/fd/*; do
      [ "$(readlink "$fd")" = "$lock" ] && held=yes
    done
    echo "$held" >> "$out"
    exec "$@"
""")


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd listing")
def test_lock_fd_reaches_local_mutators_only(repo, tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    recorder = tmp_path / "recorder.sh"
    recorder.write_text(_RECORDER, encoding="utf-8")
    recorder.chmod(0o755)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(remote)], check=True, env=_env(tmp_path))
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "config", "filter.rec.clean", f"{recorder} cat")
    (repo / ".git" / "info" / "attributes").write_text("f.txt filter=rec\n", encoding="utf-8")
    out_fetch, out_stash = tmp_path / "fetch.out", tmp_path / "stash.out"

    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=repo)
    assert lock.acquire()
    try:
        monkeypatch.setenv("HERMES_TEST_LOCK", os.path.realpath(ul.checkout_lock_path(repo)))
        monkeypatch.setenv("HERMES_TEST_OUT", str(out_fetch))
        fetched = update_cmd._git_run(["git"], ["fetch", f"--upload-pack={recorder} git-upload-pack", "origin"],
                                      cwd=repo, network=True)
        assert fetched.returncode == 0, fetched.stderr
        monkeypatch.setenv("HERMES_TEST_OUT", str(out_stash))
        (repo / "f.txt").write_text("two\n", encoding="utf-8")
        stashed = update_cmd._git_run(["git"], ["stash", "push", "-m", "custody"], cwd=repo)
        assert stashed.returncode == 0, stashed.stderr
    finally:
        lock.release()
    assert out_fetch.read_text(encoding="utf-8-sig").split() == ["no"], "a fetch child (credential/daemon territory) held the lock"
    assert "yes" in out_stash.read_text(encoding="utf-8-sig").split(), "a local mutator's child lost the checkout lock"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only")
def test_network_git_dies_with_its_killed_owner(repo, tmp_path):
    """The fd-less fetch must not keep rewriting refs after the owner died (the next owner's lease)."""
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from pathlib import Path
        from hermes_cli import update_lock as ul, update_cmd
        lock = ul.UpdateLock(path=Path(sys.argv[2]), install_root=sys.argv[1])
        assert lock.acquire()
        update_cmd._git_run(["git"], ["fetch", "--upload-pack=" + sys.argv[3], "origin"], cwd=sys.argv[1], network=True)
    """)
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(remote)], check=True, env=_env(tmp_path))
    _git(repo, "remote", "add", "origin", str(remote))
    started = tmp_path / "upload-pack.pid"
    blocker = tmp_path / "block.sh"
    blocker.write_text(f"#!/bin/sh\necho $$ > {started}.self\necho $PPID > {started}\nsleep 60\nexec git-upload-pack \"$@\"\n", encoding="utf-8")
    blocker.chmod(0o755)
    owner = subprocess.Popen([sys.executable, "-c", script, str(repo), str(tmp_path / "marker"), str(blocker)],
                             env=_env(tmp_path))
    try:
        for _ in range(300):
            if started.exists() and started.read_text(encoding="utf-8-sig").strip():
                break
            time.sleep(0.05)
        # git runs the upload-pack command through `sh -c`: the fetch is the shell's parent.
        shell = int(started.read_text(encoding="utf-8-sig"))
        fetch_pid = next(int(line.split()[1]) for line in Path(f"/proc/{shell}/status").read_text(encoding="utf-8-sig").splitlines()
                         if line.startswith("PPid:"))
        assert "git" in Path(f"/proc/{fetch_pid}/cmdline").read_text(encoding="utf-8-sig", errors="replace")
        owner.kill()
        owner.wait(timeout=10)
        for _ in range(100):
            if not Path(f"/proc/{fetch_pid}").exists() or "Z" in _state(fetch_pid):
                break
            time.sleep(0.05)
        assert not Path(f"/proc/{fetch_pid}").exists() or "Z" in _state(fetch_pid), \
            "git fetch outlived its killed update owner"
    finally:
        if owner.poll() is None:
            owner.kill()
        blocker_pid = Path(f"{started}.self")
        if blocker_pid.exists():
            subprocess.run(["kill", "-9", blocker_pid.read_text(encoding="utf-8-sig").strip()], capture_output=True)


def _state(pid: int) -> str:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8-sig")
    except OSError:
        return "Z"
    return stat[stat.rfind(")") + 2:][:1]


def test_a_refused_build_join_is_logged_and_noted_in_the_receipt(tmp_path, monkeypatch, caplog):
    """m1: the job-joining launcher's notice used to reach only the build's captured stderr
    (run_contained merges it into the output it drops on success). The report it leaves is now
    a warning (errors.log) and a receipt step; the build's outcome is untouched."""
    import json
    import logging

    import hermes_cli.update_receipt as ur
    from hermes_cli import update_custody

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    report = tmp_path / "report.txt"
    note = f"{update_custody._CUSTODY_UNAVAILABLE} (could not join the update job: 5); the command was not run"
    report.write_text(note, encoding="utf-8")
    with ur.update_receipt_scope():
        ur.begin_update_receipt()
        with caplog.at_level(logging.WARNING, logger="hermes_cli.update_custody"):
            assert update_custody._report_refused_join(str(report), ["node", "scripts/build/tui.mjs"]) == note
        payload = json.loads(ur.finalize_update_receipt("success").read_text(encoding="utf-8-sig"))
    assert any(note in rec.getMessage() and "tui.mjs" in rec.getMessage() for rec in caplog.records)
    steps = [s for s in payload["steps"] if s["name"] == "update_custody"]
    assert steps and steps[0]["ok"] is False and note in steps[0]["detail"], payload["steps"]
    assert payload["outcome"] == "success"
    assert not report.exists()
    empty = tmp_path / "joined.txt"
    empty.write_text("", encoding="utf-8")
    assert update_custody._report_refused_join(str(empty), ["node"]) is None  # joined: silent


def test_a_swallowed_refusal_is_what_the_update_reports(tmp_path, monkeypatch, capsys):
    """R8 m2: readers swallow an OSError (``_git_stdout``, ``_capture_head_sha``), so a refused
    child used to end ``hermes update`` as whatever error followed downstream. The refusal is
    receipted and kept for the run, and the command's failure path prints it, actionably."""
    import contextlib
    from types import SimpleNamespace

    import hermes_cli.update_receipt as ur
    from hermes_cli import main, update_cmd, update_custody, update_owning_install

    monkeypatch.setattr(update_custody, "_RUN", {"ran": False, "refused": None})
    monkeypatch.setattr(update_owning_install, "retarget_to_owning_install", lambda root: None)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / "checkout").mkdir()
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path / "checkout")
    monkeypatch.setattr(main, "_update_preflight_handled", lambda args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **kw: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda state: None)

    def impl(args, gateway_mode):
        ur.begin_update_receipt()
        with contextlib.suppress(OSError):  # a reader that swallows it, as _git_stdout does
            raise update_custody._refuse(["git", "rev-parse"], OSError(6, "The handle is invalid"))
        print("✗ Could not resolve HEAD")  # the downstream error it turned into
        raise SystemExit(1)

    monkeypatch.setattr(update_cmd, "_cmd_update_impl", impl)
    with ur.update_receipt_scope(), pytest.raises(SystemExit) as stop:
        main.cmd_update(SimpleNamespace(gateway=False))
    assert stop.value.code == 1
    out = capsys.readouterr().out
    assert "`hermes update` stopped: Windows would not put `git` in this update's process job" in out, out
    assert "Nothing was changed" in out and "Run `hermes update` again from a regular terminal" in out, out
    payload = ur.read_latest_receipt()
    steps = [s for s in payload["steps"] if s["name"] == "update_custody"]
    assert steps and steps[0]["ok"] is False and "rev-parse" in steps[0]["detail"], payload["steps"]


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd listing")
def test_a_partial_clone_move_never_fetches_under_the_lock_fd(repo, tmp_path, monkeypatch):
    """m3: installer checkouts are partial clones, so `reset --hard <target>` used to fetch the
    target's missing objects itself — as a descendant holding the lock fd, with the user's
    credential helper (a `git credential-cache--daemon` then held the checkout for 900 s). The
    objects are fetched first without the fd; the mutator runs with no helper and fetches nothing."""
    recorder = tmp_path / "recorder.sh"
    recorder.write_text(_RECORDER, encoding="utf-8")
    recorder.chmod(0o755)
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(repo), str(origin)], check=True, env=_env(tmp_path))
    _git(origin, "config", "uploadpack.allowFilter", "true")
    _git(origin, "config", "uploadpack.allowAnySHA1InWant", "true")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", f"file://{origin}", str(clone)],
                   check=True, env=_env(tmp_path))
    (repo / "new.txt").write_text("only on the remote\n", encoding="utf-8")
    _git(repo, "add", "new.txt")
    _git(repo, "commit", "-qm", "two")
    _git(repo, "push", "-q", str(origin), "main")
    _git(clone, "fetch", "-q", "origin")
    _git(clone, "config", "remote.origin.uploadpack", f"{recorder} git-upload-pack")
    missing = [o for o in _git(clone, "rev-list", "--objects", "--missing=print", "origin/main").split() if o[0] == "?"]
    assert missing, "fixture: the new blob must be missing locally"
    out = tmp_path / "upload-pack.out"
    monkeypatch.setenv("HERMES_TEST_LOCK", os.path.realpath(ul.checkout_lock_path(clone)))
    monkeypatch.setenv("HERMES_TEST_OUT", str(out))

    from hermes_cli.update_custody import run_git

    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=clone)
    assert lock.acquire()
    try:
        moved = run_git(["git"], ["reset", "--hard", "origin/main"], cwd=clone, capture_output=True, text=True)
        assert moved.returncode == 0, moved.stderr
    finally:
        lock.release()
    assert (clone / "new.txt").read_text(encoding="utf-8-sig") == "only on the remote\n"
    served = out.read_text(encoding="utf-8-sig").split()
    assert served and "yes" not in served, f"an upload-pack (a lazy fetch) ran holding the checkout lock: {served}"
    assert "credential.helper=" in moved.args, moved.args


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd listing")
@pytest.mark.parametrize(("args", "holds"), [(["gc", "--auto"], True), (["rev-parse", "HEAD"], False)])
def test_run_git_hands_the_lock_fd_to_local_mutators_only(repo, tmp_path, monkeypatch, args, holds):
    """Through the production runner: `gc` (packs refs, repacks) is a local mutator and runs
    holding the checkout lock; a reader does not. `pull` is never one (its network half)."""
    from hermes_cli.update_custody import LOCAL_MUTATORS, run_git

    assert "pull" not in LOCAL_MUTATORS
    git = tmp_path / "git-recorder.sh"
    git.write_text(_RECORDER.replace('exec "$@"', 'exec git "$@"'), encoding="utf-8")
    git.chmod(0o755)
    out = tmp_path / "git.out"
    monkeypatch.setenv("HERMES_TEST_LOCK", os.path.realpath(ul.checkout_lock_path(repo)))
    monkeypatch.setenv("HERMES_TEST_OUT", str(out))
    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=repo)
    assert lock.acquire()
    try:
        ran = run_git([str(git)], args, cwd=repo, capture_output=True, text=True)
    finally:
        lock.release()
    assert ran.returncode == 0, ran.stderr
    assert out.read_text(encoding="utf-8-sig").split() == (["yes"] if holds else ["no"]), args


@pytest.mark.platforms("posix")
def test_a_timed_out_child_never_waits_on_a_grandchild_holding_its_pipes(monkeypatch):
    """N14: the in-update (Windows) run() killed only the direct child on a timeout, then waited
    unbounded for the pipes a grandchild still held (git-remote-https under a stalled fetch).
    Driven here through that branch with a POSIX grandchild the tree kill (taskkill) cannot reach."""
    from hermes_cli import update_custody

    monkeypatch.setattr(update_custody.sys, "platform", "win32")
    monkeypatch.setattr(update_custody, "_held", lambda: {"fd": None})
    monkeypatch.setattr(update_custody, "popen", lambda argv, **kw: subprocess.Popen(list(argv), **kw))
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        update_custody.run(["sh", "-c", "sleep 20 & sleep 20"], capture_output=True, timeout=0.5)
    assert time.monotonic() - started < 15, "run() waited on a grandchild after the timeout"


def _held_lock_recorder(tmp_path: Path, name: str, body: str) -> Path:
    """An executable on a fresh PATH dir that records whether it holds the checkout lock fd."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / name
    exe.write_text(_RECORDER.replace('exec "$@"', body), encoding="utf-8")
    exe.chmod(0o755)
    return exe


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd listing")
def test_no_build_descendant_outlives_the_build_call(repo, tmp_path, monkeypatch):
    """N13: node marks the donated lock fd close-on-exec, so npm/esbuild under it never hold the
    checkout; custody rested on node alone. A node that dies leaving a writer behind must not
    leave it running once the build call returns: the next lock owner would share the checkout."""
    from hermes_cli.source_build import run_source_script

    late = tmp_path / "late"
    writer = f"import os, time; os.closerange(3, 4096); time.sleep(2); __import__('pathlib').Path({str(late)!r}).touch()"
    node = _held_lock_recorder(tmp_path, "node", f"{sys.executable} -c \"{writer}\" >/dev/null 2>&1 &\nexit 1")
    monkeypatch.setenv("HERMES_TEST_LOCK", os.path.realpath(ul.checkout_lock_path(repo)))
    monkeypatch.setenv("HERMES_TEST_OUT", str(tmp_path / "node.out"))
    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=repo)
    assert lock.acquire()
    try:
        with pytest.raises(subprocess.CalledProcessError):
            run_source_script(repo, "build.mjs", env={**os.environ, "PATH": f"{node.parent}{os.pathsep}{os.environ['PATH']}"}, label="probe")
    finally:
        lock.release()
    time.sleep(3)
    assert not late.exists(), "a build descendant kept writing after the build call returned"


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd listing")
def test_the_desktop_build_runs_in_checkout_custody(repo, tmp_path, monkeypatch):
    """F04: the desktop app build (npm run build / builder) writes the checkout like the other
    product builds, so it holds the checkout lock while it runs."""
    from hermes_cli import main_desktop

    npm = _held_lock_recorder(tmp_path, "npm", "exit 0")
    out = tmp_path / "npm.out"
    monkeypatch.setenv("HERMES_TEST_LOCK", os.path.realpath(ul.checkout_lock_path(repo)))
    monkeypatch.setenv("HERMES_TEST_OUT", str(out))
    desktop = repo / "apps" / "desktop"
    desktop.mkdir(parents=True)
    lock = ul.UpdateLock(path=tmp_path / "marker", install_root=repo)
    assert lock.acquire()
    try:
        main_desktop.build_prepared_desktop(desktop, source_mode=True, npm=str(npm), env=dict(os.environ))
    finally:
        lock.release()
    assert out.read_text(encoding="utf-8-sig").split() == ["yes"], "the desktop build ran without the checkout lock"
