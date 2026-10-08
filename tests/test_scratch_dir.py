"""Scratch dir contract: TMPDIR/TMP/TEMP follow HERMES_HOME/cache/scratch unless the user set them."""

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_constants import (apply_scratch_tmp_env, get_scratch_dir, is_scratch_path,
                              prune_scratch_dir, session_scratch_dir, _scratch_session_component,
                              _SCRATCH_SESSION_COMPONENT_MAX, SCRATCH_SESSION_PREFIX)


def test_scratch_env_follows_home_and_respects_user_tmpdir(tmp_path):
    """Unset temp vars → scratch of env HERMES_HOME; a Hermes-exported value re-derives for a routed
    home; a user/OS-set value (macOS ``/var/folders``, ``%TEMP%``) is never touched."""
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    env = {"HERMES_HOME": str(home_a)}
    assert apply_scratch_tmp_env(env) is True
    scratch_a = str(home_a / "cache" / "scratch")
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HERMES_SCRATCH_DIR"] == scratch_a
    assert os.path.isdir(scratch_a)

    env["HERMES_HOME"] = str(home_b)  # child served under another profile's home
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(home_b / "cache" / "scratch")

    user_env = {"HERMES_HOME": str(home_a), "TMPDIR": "/var/folders/zz"}
    assert apply_scratch_tmp_env(user_env) is False
    assert user_env["TMPDIR"] == "/var/folders/zz" and "HERMES_SCRATCH_DIR" not in user_env
    assert "TMP" not in user_env  # a partially user-set triple is left exactly as found


def test_bootstrap_import_exports_scratch_to_process_and_children(tmp_path):
    """``import hermes_bootstrap`` alone makes ``tempfile`` (this process AND a child) land in scratch.

    Unscoped on purpose (no session in the child env): a bound session gets its own lane, which
    ``test_bootstrap_export_follows_the_bound_session`` covers.
    """
    env = {k: v for k, v in os.environ.items()
           if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR", "HERMES_SESSION_ID")}
    env["HERMES_HOME"] = str(tmp_path)
    code = ("import tempfile, os, subprocess, sys; import hermes_bootstrap; "
            "print(tempfile.gettempdir()); "
            "print(subprocess.run([sys.executable, '-c', 'import tempfile;print(tempfile.gettempdir())'],"
            " capture_output=True, text=True).stdout.strip())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(tmp_path / "cache" / "scratch")
    assert out.stdout.split() == [expected, expected]


def test_bootstrap_export_follows_the_bound_session(tmp_path):
    """With a session bound, the bootstrap exports the session lane -- in this process and in a
    child that inherits the env -- so ``tempfile`` writes land where the session can claim them."""
    env = {k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TMP", "TEMP", "HERMES_SCRATCH_DIR")}
    env["HERMES_HOME"] = str(tmp_path)
    env["HERMES_SESSION_ID"] = "api_1_abc"
    code = ("import tempfile, os, subprocess, sys; import hermes_bootstrap; "
            "print(tempfile.gettempdir()); "
            "print(subprocess.run([sys.executable, '-c', 'import tempfile;print(tempfile.gettempdir())'],"
            " capture_output=True, text=True).stdout.strip())")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, encoding="utf-8",
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), check=True)
    expected = str(_lane_dir(tmp_path, "api_1_abc"))
    assert out.stdout.split() == [expected, expected]
    assert os.path.isdir(expected)


def test_prune_removes_idle_entries_and_keeps_trees_written_deep_inside(tmp_path):
    """Idle retention: an entry goes when nothing in its subtree was written within the window;
    a tree whose only recent write is three levels down is still in use and stays, even though
    its top-level mtime is ancient (a directory's mtime ignores writes below its children)."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live, fresh = scratch / "idle", scratch / "live", scratch / "fresh.txt"
    deep = live / "lane" / "wt"
    deep.mkdir(parents=True)
    idle.mkdir()
    (idle / "f").write_text("x", encoding="utf-8")
    (deep / "log").write_text("x", encoding="utf-8")
    fresh.write_text("y", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    for path in (idle, idle / "f", live, live / "lane", deep):
        os.utime(path, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    assert not idle.exists() and live.exists() and fresh.exists()


@pytest.mark.platforms("posix")  # POSIX directory modes
class TestScratchDirPermissionPolicy:
    """get_scratch_dir must honor the home permission policy instead of a blanket 0700:
    an explicit HERMES_HOME_MODE and a managed/shared home win (#117347)."""

    @staticmethod
    def _native_mode_after_chmod(path, requested=0o2770):
        """Return what this host/filesystem preserves from a real chmod request.

        macOS may clear setgid when the directory group is not one of the caller's groups.
        The portable contract is to request the operator's mode and preserve every bit the
        host accepts, not to pretend all POSIX filesystems retain identical special bits.
        """
        os.chmod(path, requested)
        return stat.S_IMODE(os.stat(path).st_mode)

    def _isolate_env(self, monkeypatch, tmp_path):
        # A known, marker-free effective home: each case below plants `.managed` in the home whose
        # scratch dir it exercises (`get_scratch_dir(home)` reads the marker there), and the real
        # home's own marker must not leak into callers that still resolve the effective home.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
        for var in ("HERMES_HOME_MODE", "HERMES_MANAGED", "HERMES_CONTAINER",
                    "HERMES_SKIP_CHMOD", "HERMES_UID", "HERMES_GID"):
            monkeypatch.delenv(var, raising=False)

    def test_default_is_owner_only(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o700

    def test_explicit_home_mode_uses_native_chmod_semantics(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_managed_env_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_managed_marker_file_leaves_preexisting_mode_untouched(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("nixos", encoding="utf-8")
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_empty_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # Legacy NixOS module wrote an empty marker; config.get_managed_system treats it as
        # managed, so the parity twin must too (not re-enable chmod on managed installs).
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").write_text("", encoding="utf-8")
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_unreadable_managed_marker_counts_as_managed(self, tmp_path, monkeypatch):
        # A marker that exists but cannot be read (OSError -> "") still counts as managed.
        home = tmp_path / "home"
        home.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (home / ".managed").mkdir()
        pre = home / "cache" / "scratch"
        pre.mkdir(parents=True)
        expected = self._native_mode_after_chmod(pre)
        scratch = get_scratch_dir(home, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == expected

    def test_effective_home_marker_does_not_govern_another_homes_scratch(self, tmp_path, monkeypatch):
        # The caller's home decides the policy. A boot caller (``export_scratch_tmp_env``) and
        # ``hermes doctor`` have already resolved theirs, so the marker lookup must not fall back
        # to ``get_hermes_home()``: for a sticky profile with HERMES_HOME unset that lookup warns
        # about the default profile on every command, and it is the wrong home to judge by anyway.
        selected = tmp_path / "home"
        selected.mkdir()
        self._isolate_env(monkeypatch, tmp_path)
        (selected / ".managed").write_text("nixos", encoding="utf-8")
        other = tmp_path / "other"
        pre = other / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        assert stat.S_IMODE(os.stat(get_scratch_dir(other, prune=False)).st_mode) == 0o700

    def test_canonical_container_signal_without_env_override_keeps_operator_mode(self, tmp_path, monkeypatch):
        # Podman/containerd/K8s runtimes often don't export HERMES_CONTAINER; the canonical
        # _detect_container breadth (not the narrower legacy signal set) must skip the chmod.
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        scratch = get_scratch_dir(tmp_path, prune=False)
        assert stat.S_IMODE(os.stat(scratch).st_mode) == 0o750

    def test_repeated_calls_do_not_strip_filesystem_supported_mode_bits(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        first = get_scratch_dir(tmp_path, prune=False)
        second = get_scratch_dir(tmp_path, prune=False)
        assert first == second
        assert stat.S_IMODE(os.stat(second).st_mode) == expected

    def test_container_keeps_operator_mode_but_honors_explicit(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_CONTAINER", "1")
        pre = tmp_path / "cache" / "scratch"
        pre.mkdir(parents=True)
        os.chmod(pre, 0o750)
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == 0o750
        probe = tmp_path / "mode-probe"
        probe.mkdir()
        expected = self._native_mode_after_chmod(probe)
        monkeypatch.setenv("HERMES_HOME_MODE", "2770")
        assert stat.S_IMODE(os.stat(get_scratch_dir(tmp_path, prune=False)).st_mode) == expected

    def test_hermes_uid_gid_applied_to_scratch(self, tmp_path, monkeypatch):
        self._isolate_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_UID", "1000")
        monkeypatch.setenv("HERMES_GID", "911")
        with patch.object(os, "chown") as mock_chown:
            get_scratch_dir(tmp_path, prune=False)
        mock_chown.assert_called_once_with(tmp_path / "cache" / "scratch", 1000, 911)


@pytest.mark.platforms("posix")  # POSIX file modes
def test_secure_file_skips_chmod_on_canonical_container_signal(tmp_path, monkeypatch):
    """_secure_file skips on the same canonical container signal that apply_secure_dir_policy /
    get_scratch_dir already honor (one policy implementation in hermes_constants)."""
    from hermes_cli import config

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    for var in ("HERMES_MANAGED", "HERMES_CONTAINER", "HERMES_SKIP_CHMOD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("hermes_constants._detect_container", lambda: True)
    f = tmp_path / "config.yaml"
    f.write_text("", encoding="utf-8")
    os.chmod(f, 0o640)
    config._secure_file(f)
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o640


def test_prune_reaps_process_living_in_idle_entry_and_spares_live_tree(tmp_path):
    """A process whose cwd is inside an idle entry is gone by the time the entry is
    (a lane's headless browsers survived for days with a deleted cwd); one living in a
    tree that is still being written is not touched."""
    import subprocess

    scratch = get_scratch_dir(tmp_path, prune=False)
    idle, live = scratch / "idle-lane" / "wt", scratch / "live-lane" / "wt"
    idle.mkdir(parents=True)
    live.mkdir(parents=True)
    ancient = time.time() - 30 * 3600
    for path in (idle.parent, idle, live.parent, live):
        os.utime(path, (ancient, ancient))
    (live / "log").write_text("still writing", encoding="utf-8")
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    doomed = subprocess.Popen(sleeper, cwd=str(idle), stdin=subprocess.DEVNULL)
    spared = subprocess.Popen(sleeper, cwd=str(live), stdin=subprocess.DEVNULL)
    try:
        assert prune_scratch_dir(scratch) == 1
        assert doomed.wait(timeout=10) is not None
        assert spared.poll() is None
        assert not idle.parent.exists() and live.exists()
    finally:
        for proc in (doomed, spared):
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)


def test_prune_records_removals_and_kills_in_scratch_prune_log(tmp_path: Path) -> None:
    """Every removed entry and every signalled process leaves a line in
    ``<home>/logs/scratch-prune.log``: the prune's own handler writes it, so the boot-time
    prune (before ``setup_logging()``) is not silent. An entry rmtree could not delete is
    neither counted nor recorded as removed (#132401)."""
    import subprocess

    scratch = get_scratch_dir(tmp_path, prune=False)
    gone, stuck = scratch / "idle-lane", scratch / "stuck-lane"
    ancient = time.time() - 30 * 3600
    for entry in (gone, stuck):
        entry.mkdir()
        (entry / "out.md").write_text("deliverable", encoding="utf-8")
        for path in (entry / "out.md", entry):
            os.utime(path, (ancient, ancient))
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=str(gone),
                              stdin=subprocess.DEVNULL)
    real_rmtree = shutil.rmtree

    def rmtree(path: str | os.PathLike[str], *args: object, **kwargs: object) -> None:
        if os.path.realpath(path) != os.path.realpath(stuck):
            real_rmtree(path, *args, **kwargs)

    try:
        with patch("hermes_constants_scratch.shutil.rmtree", rmtree):
            assert prune_scratch_dir(scratch) == 1
        assert worker.wait(timeout=10) is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait(timeout=10)
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert f"removed {str(gone)!r} (" in log
    assert f"sent TERM pid={worker.pid} " in log
    assert f"could not fully remove {str(stuck)!r}" in log and f"removed {str(stuck)!r}" not in log


@pytest.mark.platforms("linux")  # macOS and Windows filesystems reject non-UTF-8 names
def test_prune_log_keeps_record_of_non_utf8_name(tmp_path: Path) -> None:
    """A legal POSIX name that is not UTF-8 is written escaped, not dropped with its record."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    raw = os.fsencode(scratch) + b"/old-\xff"
    with open(raw, "wb") as out:
        out.write(b"x")
    ancient = time.time() - 30 * 3600
    os.utime(raw, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    log = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig")
    assert "removed " in log and "old-\\udcff" in log


@pytest.mark.platforms("linux")  # Windows rejects newlines in names
def test_prune_log_escapes_newline_in_name(tmp_path: Path) -> None:
    """A newline in a legal POSIX name is written escaped, so the record stays one line and
    the rest of the name cannot pass for a record of its own."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "old\nscratch prune: removed fake-entry"
    entry.write_text("x", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    os.utime(entry, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    lines = (tmp_path / "logs" / "scratch-prune.log").read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 1
    assert "old\\nscratch prune: removed fake-entry" in lines[0]


@pytest.mark.platforms("posix")  # POSIX rename semantics
def test_prune_log_keeps_record_when_another_prune_rotated_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two prunes of one home can both reach rollover: the other one renames
    ``scratch-prune.log`` between this one's existence check and its rename. The rotation is
    already done, so the record goes to the new file instead of being dropped."""
    import hermes_constants_scratch

    scratch = get_scratch_dir(tmp_path, prune=False)
    entry = scratch / "idle-lane"
    entry.write_text("x", encoding="utf-8")
    ancient = time.time() - 30 * 3600
    os.utime(entry, (ancient, ancient))
    log = tmp_path / "logs" / "scratch-prune.log"
    log.parent.mkdir()
    log.write_bytes(b"x" * 200)
    monkeypatch.setattr(hermes_constants_scratch, "_PRUNE_LOG_MAX_BYTES", 100)
    real_rename = os.rename
    raced: list[str] = []

    def rename(src: str, dst: str, *args: object, **kwargs: object) -> None:
        if os.fspath(src) == str(log) and not raced:
            raced.append(dst)
            real_rename(src, dst)  # the other prune's rename lands first
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    assert prune_scratch_dir(scratch) == 1
    assert raced
    records = "".join(p.read_text(encoding="utf-8-sig", errors="replace")
                      for p in log.parent.glob("scratch-prune.log*"))
    assert f"removed {str(entry)!r} (" in records


def test_prune_releases_git_worktree_registration_of_idle_entry(tmp_path):
    """Deleting a scratch entry that held a linked worktree leaves the repo with no
    dangling registration (10 sat in one repo's ``git worktree list`` after cleanup)."""
    import subprocess

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull,
                                                     "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
                                                     "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"})

    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", cwd=repo)
    (repo / "f").write_text("x", encoding="utf-8")
    git("add", "f", cwd=repo)
    git("commit", "-q", "-m", "init", cwd=repo)
    scratch = get_scratch_dir(tmp_path, prune=False)
    tree = scratch / "lane" / "abwt"
    tree.parent.mkdir()
    git("worktree", "add", "-q", "--detach", str(tree), cwd=repo)
    ancient = time.time() - 30 * 3600
    for dirpath, dirnames, filenames in os.walk(tree.parent):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), (ancient, ancient), follow_symlinks=False)
    os.utime(tree.parent, (ancient, ancient))
    assert prune_scratch_dir(scratch) == 1
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=repo, capture_output=True,
                             text=True, stdin=subprocess.DEVNULL, check=True).stdout
    assert str(tree) not in listing and not tree.exists()


# ── Session-scoped lanes ─────────────────────────────────────────────────────

def _lane_dir(home, session_id: str):
    """A session lane under *home*: ``<scratch>/session-<component>`` with the current
    sanitizer (readable prefix + stable hash), so assertions stay in sync with the
    component contract instead of hard-coding the hash value."""
    return get_scratch_dir(home, prune=False) / (SCRATCH_SESSION_PREFIX + _scratch_session_component(session_id))


def test_session_scratch_lane_is_per_session(tmp_path, monkeypatch):
    """A bound session gets its own lane, so a concurrent session's temp files never share one
    tree (and nothing has to guess which session left a file behind)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "api_1_abc")
    lane_a = session_scratch_dir()
    assert lane_a == _lane_dir(tmp_path, "api_1_abc")
    assert lane_a.is_dir()

    monkeypatch.setenv("HERMES_SESSION_ID", "cli_2_def")
    lane_b = session_scratch_dir()
    assert lane_b == _lane_dir(tmp_path, "cli_2_def")
    assert lane_a != lane_b and lane_a.is_dir() and lane_b.is_dir()


def test_session_scratch_env_points_tmpdir_at_the_lane(tmp_path):
    """TMPDIR follows the session lane, and a Hermes-set value is re-derived per session, while a
    user/OS-set TMPDIR still wins."""
    env = {"HERMES_HOME": str(tmp_path), "HERMES_SESSION_ID": "api_1_abc"}
    assert apply_scratch_tmp_env(env) is True
    lane = str(_lane_dir(tmp_path, "api_1_abc"))
    assert env["TMPDIR"] == env["TMP"] == env["TEMP"] == env["HERMES_SCRATCH_DIR"] == lane
    assert os.path.isdir(lane)

    env["HERMES_SESSION_ID"] = "cli_2_def"
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(_lane_dir(tmp_path, "cli_2_def"))

    user_env = {"HERMES_HOME": str(tmp_path), "HERMES_SESSION_ID": "api_1_abc",
                "TMPDIR": "/var/folders/zz"}
    assert apply_scratch_tmp_env(user_env) is False
    assert user_env["TMPDIR"] == "/var/folders/zz" and "HERMES_SCRATCH_DIR" not in user_env


def test_no_session_bound_keeps_the_shared_scratch_dir(tmp_path, monkeypatch):
    """CLI/cron without a bound session must not invent one: TMPDIR stays on the shared dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    assert session_scratch_dir() is None
    env = {"HERMES_HOME": str(tmp_path)}
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(tmp_path / "cache" / "scratch")


def test_scratch_scope_can_be_pinned_to_shared(tmp_path, monkeypatch):
    """HERMES_SCRATCH_SCOPE=shared restores the unscoped layout for operators who script against
    one fixed temp dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "api_1_abc")
    monkeypatch.setenv("HERMES_SCRATCH_SCOPE", "shared")
    assert session_scratch_dir() is None
    env = {"HERMES_HOME": str(tmp_path), "HERMES_SESSION_ID": "api_1_abc"}
    assert apply_scratch_tmp_env(env) is True
    assert env["TMPDIR"] == str(tmp_path / "cache" / "scratch")


def test_session_lane_component_cannot_escape_the_scratch_root(tmp_path, monkeypatch):
    """A custom/imported session id is sanitized: separators and ``..`` never relocate the lane."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = get_scratch_dir(prune=False)
    lane = session_scratch_dir(session_id="../../etc/evil")
    assert lane is not None
    assert lane.parent == root and root in lane.parents
    # An id that sanitizes to nothing still gets a lane (never the bare root).
    assert session_scratch_dir(session_id="///").parent == root


def test_lane_component_separator_collision_is_eliminated():
    """``a/b`` and ``a_b`` sanitize to the same prefix but must not share one lane: the lossy
    separator replacement used to fuse them onto ``session-a_b``, silently merging the scratch
    trees of two different sessions."""
    assert _scratch_session_component("a/b") != _scratch_session_component("a_b")
    assert _scratch_session_component("api/1_abc") != _scratch_session_component("api_1_abc")


def test_lane_component_truncation_collision_is_eliminated():
    """Two distinct ids sharing their first 64 characters used to collapse onto one lane after
    truncation (the old sanitizer kept ``cleaned[:64]``, so the tail never mattered); the
    appended hash of the FULL id keeps them apart."""
    long_id_a = "s" * 80 + "A"
    long_id_b = "s" * 80 + "B"
    # Both ids sanitize to a prefix > 64 chars, so the old code truncated both to the same
    # 64-char string; today the hash suffix separates them.
    assert _scratch_session_component(long_id_a) != _scratch_session_component(long_id_b)
    # The readable prefix is still truncated to fit the component budget (64 = prefix + 1 + hash).
    assert len(_scratch_session_component(long_id_a)) == _SCRATCH_SESSION_COMPONENT_MAX


def test_lane_component_empty_sanitize_collision_is_eliminated():
    """Ids that sanitize to nothing (``///``, ``___``) all used to fall back to the literal
    ``session`` component, fusing every such id onto one lane; the hash now separates them."""
    assert _scratch_session_component("///") != _scratch_session_component("___")
    assert _scratch_session_component("///") != _scratch_session_component("!!!")


def test_session_lanes_are_pruned_individually(tmp_path):
    """Idle retention keeps working per lane: an old session's lane goes, a live one stays."""
    scratch = get_scratch_dir(tmp_path, prune=False)
    old = scratch / "session-old_session"
    live = scratch / "session-live_session"
    old.mkdir(parents=True)
    live.mkdir(parents=True)
    (old / "helper.sh").write_text("x", encoding="utf-8")
    (live / "helper.sh").write_text("x", encoding="utf-8")
    stale = time.time() - 48 * 3600
    os.utime(old / "helper.sh", (stale, stale))
    os.utime(old, (stale, stale))
    assert prune_scratch_dir(scratch, max_idle_hours=24) == 1
    assert not old.exists() and live.is_dir()


def test_is_scratch_path_covers_the_root_and_every_lane(tmp_path, monkeypatch):
    """The predicate the checkpoint manager uses: the scratch root and anything inside it, never
    a sibling directory that merely lives under the same cache dir."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    scratch = get_scratch_dir(prune=False)
    lane = session_scratch_dir(session_id="api_1_abc")
    assert is_scratch_path(scratch) is True
    assert is_scratch_path(lane) is True
    assert is_scratch_path(lane / "nested" / "deep.sh") is True
    assert is_scratch_path(tmp_path / "cache" / "other") is False
    assert is_scratch_path(tmp_path) is False
    assert is_scratch_path("/tmp") is False
    assert is_scratch_path(None) is False
