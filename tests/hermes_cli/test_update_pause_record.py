"""Paused-gateway record and update-claim adoption, with real processes and a real checkout."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import update_pause_record as pause_record

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows cells live in wine2e")


# Every child (and its children) locks a private file for THIS repo's checkout, like the
# conftest fixture does in-process: the real one lives in the git common dir that parallel test
# files and, from a linked worktree, the live install's `hermes update` share. A held one reads
# as "another update is live" and recovery correctly defers to it.
_PRIVATE_CHECKOUT_LOCK = """
import os
from pathlib import Path
from hermes_cli import update_lock as _lock
_repo, _real = Path(_lock.__file__).resolve().parents[1], _lock.checkout_lock_path
def checkout_lock_path(install_root=None):
    root = Path(install_root) if install_root else _repo
    return Path(os.environ["HERMES_TEST_CHECKOUT_LOCK"]) if root.resolve() == _repo else _real(install_root)
_lock.checkout_lock_path = checkout_lock_path
"""


def _child(code: str, *argv: str, env: dict) -> subprocess.Popen:
    shim = Path(env["HERMES_HOME"]) / "checkout-lock-shim"
    shim.mkdir(exist_ok=True)
    (shim / "sitecustomize.py").write_text(_PRIVATE_CHECKOUT_LOCK, encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(shim), str(REPO))),
           "HERMES_TEST_CHECKOUT_LOCK": str(shim / "hermes-update.lock"), **env}
    return subprocess.Popen([sys.executable, "-c", textwrap.dedent(code), *argv], cwd=REPO, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, text=True, env=env)


def _orphaned_profiles(home: Path) -> dict | None:
    """``orphaned_record()`` read by a fresh process (the liveness probe reads the checkout's git dir)."""
    probe = _child("""
        import json
        from hermes_cli import update_pause_record as r
        body = r.orphaned_record()
        print(json.dumps(None if body is None else body["token"]["profiles"]))
    """, env={"HERMES_HOME": str(home)})
    out, _ = probe.communicate(timeout=60)
    return json.loads(out.strip().splitlines()[-1])


def test_record_written_by_a_killed_updater_is_orphaned_only_after_its_death(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    owner = _child("""
        import sys, time
        from hermes_cli import update_pause_record as r
        r.write(r.stamp_tree({"resume_needed": True, "profiles": {"default": 4242}}), owner=r.identity())
        print("written", flush=True)
        time.sleep(120)
    """, env={"HERMES_HOME": str(tmp_path)})
    try:
        assert owner.stdout.readline().strip() == "written"
        body = pause_record.read()
        assert body["owner"]["pid"] == owner.pid and body["owner"]["ct"].startswith("ct:")
        assert body["token"]["profiles"] == {"default": 4242}
        assert _orphaned_profiles(tmp_path) is None  # live owner: its own resume owns the set
    finally:
        owner.send_signal(signal.SIGKILL)  # windows-footgun: ok — module skips on Windows
        owner.wait(timeout=10)
    assert _orphaned_profiles(tmp_path) == {"default": 4242}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}).stdout.strip()


def test_resume_waits_for_a_whole_tree(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    for name in ("a.py", "b.py", "local.txt"):
        (root / name).write_text("v1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "v1")
    (root / "local.txt").write_text("user edit\n", encoding="utf-8")  # dirty before the update: not git's doing
    token = pause_record.stamp_tree({"resume_needed": True}, root)
    assert pause_record.tree_is_whole(token, root) == (True, "")

    (root / "a.py").write_text("v2\n", encoding="utf-8")  # git wrote a.py, then died before b.py and HEAD
    whole, why = pause_record.tree_is_whole(token, root)
    assert not whole and "a.py" in why

    (root / "a.py").write_text("v1\n", encoding="utf-8")
    marker = root / ".git" / "hermes-update-pull"
    marker.write_text("pid\n", encoding="utf-8")
    assert pause_record.tree_is_whole(token, root)[0] is False
    marker.unlink()

    _git(root, "commit", "-qam", "v2")  # HEAD moved: only a dependency sync for it makes it whole
    (root / "b.py").write_text("rewritten by a build step\n", encoding="utf-8")
    whole, why = pause_record.tree_is_whole(token, root)
    # A committed update is judged on its dependencies alone; a tracked file the build rewrote
    # must not keep the gateways stopped on every later launch.
    assert not whole and "dependencies" in why, why


def test_an_update_adopting_an_orphan_never_certifies_the_tree_it_left_torn(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.py").write_text("v1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "v1")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pause_record, "install_root", lambda: root)
    pause_record.write(pause_record.stamp_tree({"resume_needed": True, "profiles": {"default": 4242}}),
                       owner=pause_record.UNOWNED)
    (root / "a.py").write_text("half-written\n", encoding="utf-8")  # the killed update's git died mid-checkout
    assert not pause_record.tree_is_whole(pause_record.read()["token"], root)[0], "premise: the orphan's gate refuses"

    adopted, claims = pause_record.adopt_orphans()
    token = pause_record.record_pause({"resume_needed": True, "profiles": {"beta": 99}}, adopted, claims)
    whole, why = pause_record.tree_is_whole(token, root)
    assert not whole and "a.py" in why, "adoption certified the torn tree as the pre-update baseline"


def test_an_adopted_baseline_never_replaces_this_runs_own(tmp_path, monkeypatch):
    """Review W1: the killed run moved HEAD X->Y; this run starts at Y and its git dies mid-checkout
    without moving HEAD. The orphan's X baseline says nothing about Y — this run's own must still gate."""
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.py").write_text("v1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "X")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pause_record, "install_root", lambda: root)
    pause_record.write(pause_record.stamp_tree({"resume_needed": True, "profiles": {"default": 4242}}),
                       owner=pause_record.UNOWNED)
    (root / "a.py").write_text("v2\n", encoding="utf-8")
    _git(root, "commit", "-qam", "Y")  # the killed update moved HEAD, synced Y's dependencies, then died
    monkeypatch.setattr(pause_record, "_venv_is_current", lambda root: True)

    adopted, claims = pause_record.adopt_orphans()
    token = pause_record.record_pause({"resume_needed": True, "profiles": {"beta": 99}}, adopted, claims)
    assert pause_record.tree_is_whole(token, root) == (True, ""), "a clean tree at this run's own HEAD must pass"
    (root / "a.py").write_text("half-written\n", encoding="utf-8")  # this run's git died at HEAD Y
    whole, why = pause_record.tree_is_whole(pause_record.read()["token"], root)
    assert not whole and "a.py" in why, "this run's own baseline was dropped for the adopted one"


def _orphan(tmp_path: Path, profiles: dict) -> None:
    """A record whose owner — a real ``hermes update`` stand-in — was SIGKILLed after writing it."""
    owner = _child("""
        import time
        from hermes_cli import update_pause_record as r
        r.write(r.stamp_tree({"resume_needed": True, "profiles": %r}), owner=r.identity())
        print("written", flush=True)
        time.sleep(120)
    """ % profiles, env={"HERMES_HOME": str(tmp_path)})
    assert owner.stdout.readline().strip() == "written"
    owner.send_signal(signal.SIGKILL)  # windows-footgun: ok — module skips on Windows
    owner.wait(timeout=10)


# The Windows resume cannot run here: the stand-in for it prints the set it was handed and
# (mode "hang") parks like a resume waiting on relaunch verification.
_RECOVER = """
    import sys, time
    import hermes_cli.update_cmd_windows as w
    def resume(token):
        print("resume", sorted(token.get("profiles") or {}), flush=True)
        if sys.argv[1] == "hang":
            time.sleep(120)
        token["resume_needed"] = False
    w._resume_windows_gateways_after_update = resume
    from hermes_cli import update_pause_record as r
    r.recover(["status"])
    print("done", flush=True)
"""


@pytest.mark.live_system_guard_bypass
def test_a_launch_killed_mid_recovery_leaves_the_set_to_the_next_launch(tmp_path):
    _orphan(tmp_path, {"default": 4242})
    env = {"HERMES_HOME": str(tmp_path)}
    first = _child(_RECOVER, "hang", env=env)
    try:
        assert first.stdout.readline().strip() == "resume ['default']"
    finally:
        first.send_signal(signal.SIGKILL)  # windows-footgun: ok — taskkill / console close mid-resume
        first.wait(timeout=10)
    second = _child(_RECOVER, "ok", env=env)
    out, _ = second.communicate(timeout=60)
    assert out.splitlines()[:1] == ["resume ['default']"], out
    third = _child(_RECOVER, "ok", env=env)
    out, _ = third.communicate(timeout=60)
    assert out.strip() == "done", f"a resumed set was resumed again: {out}"


# An update takes the checkout right after this launch claimed the set (the race window).
_UPDATE_AFTER_CLAIM = """
    import subprocess, sys
    from hermes_cli import update_pause_record as r
    hold = ("import time; from hermes_cli import update_lock as l; from hermes_cli.update_pause_record import install_root;"
            "print(l.UpdateLock().acquire_checkout(install_root()), flush=True); time.sleep(120)")
    real_claim, updates = r.claim, []
    def claim(src):
        won = real_claim(src)
        updates.append(subprocess.Popen([sys.executable, "-c", hold], stdout=subprocess.PIPE, text=True))
        assert updates[-1].stdout.readline().strip() == "True"
        return won
    r.claim = claim
""" + _RECOVER + """
    for update in updates:
        update.kill()
        update.wait()
"""


@pytest.mark.live_system_guard_bypass
def test_recovery_never_starts_gateways_under_an_update_that_took_the_checkout(tmp_path):
    _orphan(tmp_path, {"default": 4242})
    env = {"HERMES_HOME": str(tmp_path)}
    out, _ = _child(_UPDATE_AFTER_CLAIM, "ok", env=env).communicate(timeout=60)
    assert out.strip() == "done", f"gateways restarted while an update owned the checkout: {out}"
    out, _ = _child(_RECOVER, "ok", env=env).communicate(timeout=60)
    assert out.splitlines()[:1] == ["resume ['default']"], f"the handed-back set was stranded: {out}"


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("argv, recovers", [
    (["--resume", "update", "--version"], True),  # "update" is a session name here, not the command
    (["update", "--help"], False),  # the update adopts the set itself
])
def test_startup_recovery_follows_the_parsed_command_not_raw_argv(tmp_path, argv, recovers):
    _orphan(tmp_path, {})  # nothing to start: a recovering launch just retires it
    cli = _child("""
        import sys
        from hermes_cli.main import main
        sys.argv = ["hermes", *sys.argv[1:]]
        try:
            main()
        except SystemExit:
            pass
    """, *argv, env={"HERMES_HOME": str(tmp_path), "HOME": str(tmp_path / "home")})
    out, _ = cli.communicate(timeout=120)
    assert (_record_files(tmp_path) == []) is recovers, out


_HOLDER = """
    import subprocess, sys, time
    from hermes_cli.update_lock import UpdateLock
    lock = UpdateLock()
    assert lock.acquire() and lock.acquired
    host = subprocess.Popen([sys.executable, *sys.argv[1:]], stdout=subprocess.PIPE, text=True)
    print(host.stdout.read().strip(), flush=True)
    host.wait()
"""
# A stand-in for the host the update relaunched: its argv names the host command, and the
# ``hermes update`` its agent starts is its child.
_HOST = """
import subprocess, sys
probe = "from hermes_cli.update_lock import UpdateLock; l = UpdateLock(); print(l.acquire(), l.holder is not None)"
print(subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True).stdout.strip())
"""


# The intermediate is a stand-in script, not a real gateway/update: nothing touches the checkout.
@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("host_argv, adopts", [
    (("hermes_cli.main", "gateway", "run"), False),
    (("hermes_cli.main", "--profile", "work", "gateway", "run"), False),
    (("/opt/hermes/hermes_cli/main.py", "-p", "work", "gateway", "run"), False),
    (("-m", "hermes_cli.main", "--profile", "work", "dashboard"), False),
    (("hermes_cli.main", "status"), True),
])
def test_update_started_from_a_relaunched_gateway_does_not_share_the_claim(tmp_path, host_argv, adopts):
    script = tmp_path / "host.py"
    script.write_text(_HOST, encoding="utf-8")
    holder = _child(_HOLDER, str(script), *host_argv, env={"HERMES_HOME": str(tmp_path)})
    out, _ = holder.communicate(timeout=60)
    assert holder.returncode == 0, out
    assert out.strip() == ("True False" if adopts else "False True"), out


# --- R7: restart debt is conserved across custody transfers ------------------------------------
# A child stops ITSELF at the named line of the real module (settrace), so a kill or a rival lands
# exactly on the transfer boundary; nothing in the module under test is replaced.
_AT_LINE = """
    import inspect, json, os, sys, time
    from pathlib import Path
    from hermes_cli import update_pause_record as r
    def stop_at(fn, text, action, flag=None):
        lines, start = inspect.getsourcelines(fn)
        line = start + next(i for i, t in enumerate(lines) if text in t)
        def trace(frame, event, arg):  # traces only fn's own frames: the rest runs at full speed
            if frame.f_code is not fn.__code__:
                return None
            if event == "line" and frame.f_lineno == line:
                if action == "kill":
                    os._exit(71)
                print("paused", flush=True)
                while not Path(flag).exists():
                    time.sleep(0.01)
            return trace
        sys.settrace(trace)
"""
_RESUMES = """
    import hermes_cli.update_cmd_windows as w
    def resume(token):
        print("resume", sorted(token.get("profiles") or {}), flush=True)
        token["resume_needed"] = False
    w._resume_windows_gateways_after_update = resume
    r.recover(["status"])
    print("done", flush=True)
"""


def _launches(tmp_path: Path, n: int) -> list[list[str]]:
    """What each of *n* successive fresh launches resumed."""
    seen = []
    for _ in range(n):
        out, _ = _child(_AT_LINE + _RESUMES, env={"HERMES_HOME": str(tmp_path)}).communicate(timeout=60)
        seen.append([line for line in out.splitlines() if line.startswith("resume")])
    return seen


def _record_files(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(pause_record.RECORD_STEM)
                  and p.suffix in (".json", ".claim"))


@pytest.mark.live_system_guard_bypass
def test_a_claim_in_transfer_cannot_be_taken_by_a_second_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pause_record, "_MUTEX_WAIT_S", 0.5)
    _orphan(tmp_path, {"default": 4242})
    src = pause_record.record_path()
    flag = tmp_path / "go"
    first = _child(_AT_LINE + """
    stop_at(r.claim, "src.unlink()", "pause", sys.argv[2])
    won = r.claim(Path(sys.argv[1]))
    sys.settrace(None)
    print(json.dumps(won and str(won[0])), flush=True)
    """, str(src), str(flag), env={"HERMES_HOME": str(tmp_path)})
    try:
        assert first.stdout.readline().strip() == "paused"
        rivals = [pause_record.claim(p) for p in (src, *pause_record._claims(src))]
        assert rivals == [None] * len(rivals), "a second launch took a claim whose transfer was in flight"
    finally:
        flag.touch()
    won = json.loads(first.stdout.readline())
    first.wait(timeout=30)
    assert _record_files(tmp_path) == [Path(won).name], "the paused set is now carried by two files"


@pytest.mark.live_system_guard_bypass
def test_a_launch_killed_between_claim_and_retire_resumes_the_set_once(tmp_path):
    _orphan(tmp_path, {"default": 4242})
    killed = _child(_AT_LINE + """
    stop_at(r.claim, "src.unlink()", "kill")
    r.recover(["status"])
    """, env={"HERMES_HOME": str(tmp_path)})
    killed.wait(timeout=60)
    assert killed.returncode == 71 and len(_record_files(tmp_path)) == 2, "premise: killed after publishing the claim"
    assert _launches(tmp_path, 2) == [["resume ['default']"], []]
    assert _record_files(tmp_path) == []


@pytest.mark.live_system_guard_bypass
def test_an_update_killed_after_publishing_its_record_never_restarts_a_set_twice(tmp_path):
    _orphan(tmp_path, {"default": 4242})
    killed = _child(_AT_LINE + """
    adopted, claims = r.adopt_orphans()
    stop_at(r.record_pause, "release_claims(claims)", "kill")
    r.record_pause({"resume_needed": True, "profiles": {"beta": 99}}, adopted, claims)
    """, env={"HERMES_HOME": str(tmp_path)})
    killed.wait(timeout=60)
    assert killed.returncode == 71 and len(_record_files(tmp_path)) == 2, "premise: killed before retiring the claim"
    assert _launches(tmp_path, 2) == [["resume ['beta', 'default']"], []]
    assert _record_files(tmp_path) == []


_DRAINING = """
    import os, signal, sys, time
    from pathlib import Path
    flag = Path(sys.argv[1])
    def stop(*_):
        print("stopping", flush=True)  # acknowledged; drains until the flag appears
        while not flag.exists():
            time.sleep(0.01)
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    print("up", flush=True)
    while True:
        time.sleep(0.1)
"""


@pytest.mark.live_system_guard_bypass
def test_a_gateway_draining_after_the_stop_request_keeps_its_restart_debt(tmp_path):
    flag = tmp_path / "drained"
    gateway = _child(_DRAINING, str(flag), env={"HERMES_HOME": str(tmp_path)})
    assert gateway.stdout.readline().strip() == "up"
    updater = _child(_AT_LINE + """
    pid = int(sys.argv[1])
    token = r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                            "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
    r.mark_stop_requested(token, [pid])
    r.mark_stop_sent(token, pid)
    os.kill(pid, 15)
    print("asked", flush=True)
    time.sleep(120)
    """, str(gateway.pid), env={"HERMES_HOME": str(tmp_path)})
    try:
        assert updater.stdout.readline().strip() == "asked"
        assert gateway.stdout.readline().strip() == "stopping"
    finally:
        updater.send_signal(signal.SIGKILL)  # windows-footgun: ok — module skips on Windows
        updater.wait(timeout=10)
    try:
        assert gateway.poll() is None, "premise: the gateway is still draining"
        assert _launches(tmp_path, 1) == [[]], "a draining gateway was restarted before it exited"
        assert len(_record_files(tmp_path)) == 1, "a draining gateway's restart debt was dropped"
    finally:
        flag.touch()
        gateway.wait(timeout=10)
    assert _launches(tmp_path, 2) == [["resume ['default']"], []]


@pytest.mark.live_system_guard_bypass
def test_a_gateway_never_asked_to_stop_is_not_restarted(tmp_path):
    gateway = _child(_DRAINING, str(tmp_path / "unused"), env={"HERMES_HOME": str(tmp_path)})
    assert gateway.stdout.readline().strip() == "up"
    updater = _child(_AT_LINE + """
    pid = int(sys.argv[1])
    r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                    "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
    print("recorded", flush=True)
    time.sleep(120)
    """, str(gateway.pid), env={"HERMES_HOME": str(tmp_path)})
    try:
        assert updater.stdout.readline().strip() == "recorded"
        updater.send_signal(signal.SIGKILL)  # windows-footgun: ok — killed before its first stop request
        updater.wait(timeout=10)
        assert _launches(tmp_path, 1) == [[]]
        assert _record_files(tmp_path) == [], "a still-serving gateway's entry stayed owed"
    finally:
        gateway.kill()
        gateway.wait(timeout=10)


@pytest.mark.live_system_guard_bypass
def test_a_stop_intended_but_never_issued_leaves_a_serving_gateway_alone_and_a_later_user_stop_final(tmp_path):
    """The updater recorded its intent, then died before the request (N1): the gateway was never
    asked, keeps serving and owes nothing; when the user stops it later, no launch restarts it."""
    gateway = _child(_DRAINING, str(tmp_path / "unused"), env={"HERMES_HOME": str(tmp_path)})
    assert gateway.stdout.readline().strip() == "up"
    updater = _child(_AT_LINE + """
    pid = int(sys.argv[1])
    token = r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                            "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
    r.mark_stop_requested(token, [pid])
    print("intended", flush=True)
    time.sleep(120)
    """, str(gateway.pid), env={"HERMES_HOME": str(tmp_path)})
    try:
        assert updater.stdout.readline().strip() == "intended"
        updater.send_signal(signal.SIGKILL)  # windows-footgun: ok — killed between the intent and the request
        updater.wait(timeout=10)
        assert _launches(tmp_path, 1) == [[]]
        assert _record_files(tmp_path) == [], "a gateway never asked to stop is held as draining"
    finally:
        gateway.kill()  # the user's stop, after the update's obligation was retired
        gateway.wait(timeout=10)
    assert _launches(tmp_path, 1) == [[]], "a gateway the user stopped was restarted for an update that never stopped it"


@pytest.mark.live_system_guard_bypass
def test_a_request_on_disk_is_owed_even_before_it_is_recorded_sent(tmp_path):
    """Killed after writing the planned-stop marker (the request), before recording it sent: the
    gateway acts on it, so it is owed; a marker the USER's stop wrote is no evidence."""
    marker = tmp_path / "profile" / ".gateway-planned-stop.json"
    marker.parent.mkdir()
    gateway = _child(_DRAINING, str(tmp_path / "unused"), env={"HERMES_HOME": str(tmp_path)})
    assert gateway.stdout.readline().strip() == "up"
    updater = _child(_AT_LINE + """
    pid, marker = int(sys.argv[1]), Path(sys.argv[2])
    token = r.record_pause({"resume_needed": True, "profiles": {"default": pid},
                            "identities": {str(pid): r.identity(pid)["ct"]}}, None, [])
    r.mark_stop_requested(token, [pid], markers={pid: marker})
    from hermes_cli.update_cmd_windows import _write_update_planned_stop_marker
    assert _write_update_planned_stop_marker(marker.parent, pid)
    print("requested", flush=True)
    time.sleep(120)
    """, str(gateway.pid), str(marker), env={"HERMES_HOME": str(tmp_path)})
    try:
        assert updater.stdout.readline().strip() == "requested"
        updater.send_signal(signal.SIGKILL)  # windows-footgun: ok — killed before mark_stop_sent
        updater.wait(timeout=10)
        assert _launches(tmp_path, 1) == [[]] and len(_record_files(tmp_path)) == 1, "an issued request lost its debt"
        marker.write_text(json.dumps({"target_pid": gateway.pid, "stopper_pid": 1}), encoding="utf-8")  # a user's stop replaced it
        assert _launches(tmp_path, 1) == [[]] and len(_record_files(tmp_path)) == 1, "resolved evidence was re-judged"
    finally:
        gateway.kill()
        gateway.wait(timeout=10)
    assert _launches(tmp_path, 2) == [["resume ['default']"], []]


# --- R13: a checkout sharing the home never takes another checkout's paused set --------------
def _checkout_copy(root: Path) -> Path:
    """A second checkout: the real module file under another install root."""
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "update_pause_record.py").write_bytes(Path(pause_record.__file__).read_bytes())
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "b")
    return root


@pytest.mark.live_system_guard_bypass
def test_another_checkouts_writer_never_imports_or_relabels_this_checkouts_debt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _orphan(tmp_path, {"alpha": 4242})
    other = _checkout_copy(tmp_path / "other")
    writer = _child("""
        import importlib.util, json, sys
        spec = importlib.util.spec_from_file_location("other_pause", sys.argv[1])
        b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
        b.write(b.stamp_tree({"resume_needed": True, "profiles": {"beta": 1}}), owner=b.UNOWNED)
        print(json.dumps([len(b.orphans()), b.read()]), flush=True)
    """, str(other / "hermes_cli" / "update_pause_record.py"), env={"HERMES_HOME": str(tmp_path)})
    out, _ = writer.communicate(timeout=60)
    seen, theirs = json.loads(out.strip().splitlines()[-1])
    assert seen == 1 and sorted(theirs["token"]["profiles"]) == ["beta"], f"imported this checkout's set: {theirs}"
    ours = pause_record.read()
    assert ours["install_root"] == str(REPO) and sorted(ours["token"]["profiles"]) == ["alpha"], ours


# --- R12: the pause reader judges incarnations by the update marker's rule ---------------------
def test_our_own_pid_is_ours_only_at_our_exact_creation_time():
    """A record left by a killed update whose pid this launch now has (fresh pid namespace) is dead."""
    me = pause_record.identity()
    assert me["ct"].startswith("ct:") and pause_record.identity_is_live(me)
    created = float(me["ct"][3:])
    reused = {"pid": os.getpid(), "ct": f"ct:{created - 0.5:.3f}"}  # inside the cross-writer skew
    assert not pause_record.identity_is_live(reused), "a killed update's record became ours by pid reuse"


def test_a_record_naming_our_pid_without_a_creation_time_is_a_previous_incarnation():
    assert not pause_record.identity_is_live({"pid": os.getpid(), "ct": None})


# --- R9: the retired list never fails open ------------------------------------------------------
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads a mode-000 file")
def test_an_unreadable_retired_list_claims_nothing_and_forgets_nothing(tmp_path, monkeypatch):
    """Mode 000 on the real list stands in for a Windows read refusal (a sharing violation, an AV
    scanner): which obligations are complete is then unknown, never "none"."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    src = pause_record.record_path()
    retired = src.with_suffix(".retired")
    done = "a" * 32  # completed earlier; Windows kept its record file
    pause_record.write({"pause_id": done, "resume_needed": True, "profiles": {"default": 4242}},
                       owner=pause_record.UNOWNED)
    pause_record._atomic_write(retired, {"schema": 1, "ids": [done]})
    assert pause_record.orphans() == []
    other = src.with_name(f"{src.name}.999.deadbeef.claim")  # an obligation completing meanwhile
    pause_record._atomic_write(other, {"install_root": str(REPO), "claimer": pause_record.UNOWNED,
                                       "token": {"pause_id": "b" * 32}})

    def owed() -> list[str]:
        try:
            return [body["token"]["pause_id"] for _src, body in pause_record.orphans()]
        except OSError:  # unknown: recovery reports it and claims nothing
            return []

    retired.chmod(0)
    try:
        assert done not in owed(), "a completed obligation is owed again while its list is unreadable"
        assert pause_record.claim(src) is None, "a completed obligation was claimed again"
        with pytest.raises(OSError), pause_record._mutex():
            pause_record._retire(src, [(other, json.loads(other.read_text(encoding="utf-8-sig")))])
        assert other.exists(), "a retirement that could not be recorded deleted its carrier"
    finally:
        retired.chmod(0o644)
    assert json.loads(retired.read_text(encoding="utf-8-sig"))["ids"] == [done], "the history was rewritten"
    assert done not in owed()



# --- Review W3: record hygiene --------------------------------------------------------------------
def test_a_refused_publish_leaves_no_temp_beside_the_record(tmp_path, monkeypatch):
    target = tmp_path / "records" / "record.json"

    def refuse(src, dst):
        raise PermissionError(13, "Access is denied", str(dst))  # a reader without FILE_SHARE_DELETE

    monkeypatch.setattr(pause_record.os, "replace", refuse)
    with pytest.raises(PermissionError):
        pause_record._atomic_write(target, {"schema": 1})
    assert sorted(p.name for p in target.parent.iterdir()) == [], "the half-published temp was left behind"


def test_the_record_mutex_excludes_other_threads_of_this_process(tmp_path, monkeypatch):
    import threading

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    held, release = threading.Event(), threading.Event()

    def holder():
        with pause_record._mutex():
            held.set()
            release.wait(30)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(30)
        with pytest.raises(pause_record.RecordBusy), pause_record._mutex(wait_s=0.3):
            pass  # rode on the other thread's hold
    finally:
        release.set()
        thread.join(30)
    with pause_record._mutex(), pause_record._mutex():  # re-entry on one thread still works
        pass


def test_a_malformed_retired_list_is_set_aside_only_when_nothing_refers_to_it(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    retired = pause_record._retired_path(pause_record.record_path())
    retired.write_text("{trunc", encoding="utf-8")
    assert pause_record.adopt_orphans() == (None, []), "a corrupt history aborted the update"
    assert not retired.exists() and retired.with_suffix(".corrupt").read_text(encoding="utf-8-sig") == "{trunc"

    # With a record on disk the list may be the only proof it was completed: still unknown.
    pause_record.write({"pause_id": "c" * 32, "resume_needed": True, "profiles": {"default": 4242}},
                       owner=pause_record.UNOWNED)
    retired.write_text("{trunc", encoding="utf-8")
    with pytest.raises(pause_record.RetiredUnknown):
        pause_record.orphans()
    assert retired.read_text(encoding="utf-8-sig") == "{trunc"


def test_a_host_without_psutil_records_an_identity_the_liveness_rule_can_prove(monkeypatch):
    """Review 5411136378 fix 2: the record's creation time comes from the same probe the reader
    judges with, so a psutil-less host never writes ``ct=None`` (unprovable, so never orphaned)."""
    from hermes_cli import update_lock
    monkeypatch.setitem(sys.modules, "psutil", None)  # `import psutil` raises ImportError
    ident = pause_record.identity()
    assert ident["ct"] is not None, "no creation time without psutil"
    assert update_lock.incarnation_live(ident["pid"], ident["ct"]) is True


def test_the_tree_gate_finds_pm_git_and_runs_it_in_the_updaters_custody(tmp_path, monkeypatch):
    """Review 5411136378 fix 3: on an install whose only git is PM's staged copy (no git on PATH),
    a bare ``git`` made ``head_sha`` None, so ``tree_is_whole`` refused forever."""
    import shutil
    from types import SimpleNamespace

    import pm
    from hermes_cli import update_custody
    root = tmp_path / "checkout"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "a.py").write_text("v1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "v1")
    head, pm_git = _git(root, "rev-parse", "HEAD"), shutil.which("git")
    ran, real_run_git = [], update_custody.run_git
    monkeypatch.setattr(update_custody, "run_git", lambda cmd, args, **kw: ran.append(cmd[0]) or real_run_git(cmd, args, **kw))
    monkeypatch.setattr(pm, "installed_package",
                        lambda name, allow_outdated=False: SimpleNamespace(binary=Path(pm_git)) if name == "git" else None)
    (tmp_path / "no-git").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "no-git"))
    assert pause_record.head_sha(root) == head
    assert ran == [pm_git], "the gate's git bypassed update_custody.run_git"
