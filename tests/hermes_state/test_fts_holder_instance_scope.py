"""Instance-scoping of the uninspectable-holder fallback.

Field-verified 2026-09-07 on a production host running TWO independent
Hermes instances: a main gateway (user ``ubuntu``,
HERMES_HOME=/home/ubuntu/.hermes) and a demo gateway (user ``demo``,
HERMES_HOME=/home/demo/.hermes).  The demo gateway runs as another user, so
its ``/proc/<pid>/fd`` table is unreadable from the main instance and the
holder scan falls back to ``/proc/<pid>/cmdline`` + ``_looks_like_hermes``.
The demo process's argv matches the Hermes patterns exactly, so the fallback
flagged it as an uninspectable holder of the MAIN instance's state.db even
though ``lsof`` proved zero open handles on it.  Consequence: the stale-FTS
rebuild in ``hermes_state_schema._recover_stale_fts`` was deferred 42 times
across 6 gateway restarts, the ``fts_stale`` breadcrumb never cleared, and
FTS self-repair stayed permanently disabled.

PR #92419 fixed substring false positives (journalctl/grep mentioning
hermes); a genuine second instance with a DIFFERENT HERMES_HOME is still
misjudged on current main (issue #92401).

Behavior contract: an uninspectable holder identified only by argv must be
counted unless its own argv proves it is scoped to a *different* Hermes
home / state.db and never references ours.  Ambiguous argv (no absolute
paths at all) must remain fail-closed, exactly as before — the conservative
intent of the fallback is preserved.
"""

import os

import pytest

import hermes_state_holders

# Capture the pristine stdlib functions at import time: monkeypatched calls
# re-enter these closures, and re-capturing ``os.listdir`` after a previous
# patch would compose the fakes into a double path-rewrite.
_REAL_LISTDIR = os.listdir
_REAL_READLINK = os.readlink


# Representative demo-gateway argv on the two-instance host: every absolute
# token lives under /home/demo/.hermes, the binary name matches the Hermes
# patterns, and nothing references the main instance's home or state.db.
DEMO_HOME_ARGV = [
    "/home/demo/.hermes/hermes-agent/hermes",
    "gateway",
    "run",
]

# Same, spelled through the venv interpreter + hermes launcher script.
DEMO_VENV_ARGV = [
    "/home/demo/.hermes/hermes-agent/venv/bin/python",
    "/home/demo/.hermes/hermes-agent/hermes_cli/main.py",
    "gateway",
]

# A Hermes-shaped argv with no absolute paths: cannot disprove that this
# process touches our state.db, so it must stay fail-closed.
AMBIGUOUS_ARGV = ["hermes", "gateway", "run"]


def _install_fake_proc(monkeypatch, tmp_path, unreadable_pids=(), fd_pids=()):
    """Redirect the module's /proc access to an inert fake tree.

    PIDs in ``unreadable_pids`` raise PermissionError on their fd dir
    (cross-user process); PIDs in ``fd_pids`` expose an empty-but-listable
    fd dir whose single descriptor fails readlink with EACCES
    (uninspectable-descriptor branch).
    """
    proc_root = tmp_path / "proc"
    for pid in set(unreadable_pids) | set(fd_pids):
        (proc_root / str(pid)).mkdir(parents=True, exist_ok=True)
    for pid in fd_pids:
        (proc_root / str(pid) / "fd").mkdir(exist_ok=True)
        (proc_root / str(pid) / "fd" / "3").touch(exist_ok=True)

    monkeypatch.setattr(hermes_state_holders.os, "getpid", lambda: 111)

    def _listdir(path):
        if isinstance(path, str):
            for pid in unreadable_pids:
                if path == f"/proc/{pid}/fd":
                    raise PermissionError(errno_value("EACCES"), path)
            path = path.replace("/proc", str(proc_root))
        return _REAL_LISTDIR(path)

    monkeypatch.setattr(hermes_state_holders.os, "listdir", _listdir)

    def _readlink(path):
        if "222/fd/3" in str(path):
            raise PermissionError(errno_value("EACCES"), str(path))
        return _REAL_READLINK(str(path).replace("/proc", str(proc_root)))

    monkeypatch.setattr(hermes_state_holders.os, "readlink", _readlink)


def errno_value(name):
    import errno

    return getattr(errno, name)


def _install_fake_argv(monkeypatch, argv_by_pid):
    monkeypatch.setattr(
        hermes_state_holders,
        "_read_proc_argv",
        lambda pid: list(argv_by_pid.get(pid)) if pid in argv_by_pid else None,
    )


@pytest.mark.linux_only
class TestUninspectableHolderInstanceScope:
    def test_other_instance_argv_is_not_a_holder_of_our_db(self, tmp_path, monkeypatch):
        """RED: fd dir unreadable + argv proves the process belongs to a
        DIFFERENT Hermes home → not a holder of our state.db."""
        db_path = tmp_path / "state.db"
        _install_fake_proc(monkeypatch, tmp_path, unreadable_pids=(222,))
        _install_fake_argv(monkeypatch, {222: DEMO_HOME_ARGV})

        holders = hermes_state_holders.foreign_state_db_holders(db_path)
        assert holders == []

    def test_argv_referencing_our_db_stays_flagged(self, tmp_path, monkeypatch):
        """A (possibly second) instance whose argv names OUR state.db, our
        sidecars, or our home must still be fail-closed flagged."""
        db_path = tmp_path / "state.db"
        our_home = str(tmp_path)

        for argv in (
            ["hermes", f"--db={db_path}", "gateway"],
            ["hermes", "checkpoint", f"{db_path}-wal"],
            ["hermes", "--home", our_home, "gateway"],
        ):
            assert hermes_state_holders._looks_like_hermes(argv) or argv[0] == "hermes"
            _install_fake_proc(monkeypatch, tmp_path, unreadable_pids=(222,))
            _install_fake_argv(monkeypatch, {222: argv})

            holders = hermes_state_holders.foreign_state_db_holders(db_path)
            assert [pid for pid, _ in holders] == [222], argv
            assert holders[0][1].startswith("uninspectable holder:"), argv


class TestIsAbsoluteArgvToken:
    """Contract for the cross-host absolute-path detector.

    ``os.path.isabs`` only recognises the current host's form; the helper
    additionally accepts Windows absolutes everywhere so holder argv can be
    inspected for a home segment regardless of where the scan runs.
    """

    def test_posix_absolute(self):
        assert hermes_state_holders._is_absolute_argv_token("/home/demo/.hermes/state.db")

    def test_windows_drive_absolute_backslash(self):
        assert hermes_state_holders._is_absolute_argv_token("C:\\Users\\a\\.hermes\\state.db")

    def test_windows_drive_absolute_forward_slash(self):
        assert hermes_state_holders._is_absolute_argv_token("C:/Users/a/.hermes/state.db")

    def test_windows_unc_absolute(self):
        assert hermes_state_holders._is_absolute_argv_token("\\\\wsl$\\Ubuntu\\home\\a\\.hermes")

    def test_relative_tokens_rejected(self):
        assert not hermes_state_holders._is_absolute_argv_token("hermes")
        assert not hermes_state_holders._is_absolute_argv_token("gateway")
        assert not hermes_state_holders._is_absolute_argv_token(".hermes/state.db")
        assert not hermes_state_holders._is_absolute_argv_token("--db=state.db")


@pytest.mark.linux_only
class TestArgvScopePosixPaths:
    """POSIX regression arm: other-home argv scopes away, ours stays flagged."""

    def test_other_posix_home_scopes_to_other(self, tmp_path):
        db_path = tmp_path / "state.db"
        argv = ["/home/demo/.hermes/hermes-agent/hermes", "gateway", "run"]
        assert hermes_state_holders._argv_scoped_to_other_home(argv, db_path)

    def test_option_value_other_home_scopes_to_other(self, tmp_path):
        db_path = tmp_path / "state.db"
        argv = ["hermes", "--db=/home/demo/.hermes/state.db", "gateway"]
        assert hermes_state_holders._argv_scoped_to_other_home(argv, db_path)

    def test_our_db_reference_is_not_other(self, tmp_path):
        db_path = tmp_path / "state.db"
        argv = ["hermes", f"--db={db_path}", "gateway"]
        assert not hermes_state_holders._argv_scoped_to_other_home(argv, db_path)

    def test_ambiguous_argv_stays_fail_closed(self, tmp_path):
        db_path = tmp_path / "state.db"
        assert not hermes_state_holders._argv_scoped_to_other_home(AMBIGUOUS_ARGV, db_path)


@pytest.mark.windows_only
class TestArgvScopeWindowsPaths:
    """Windows arm: drive-letter / UNC argv must scope to the other home.

    On Windows the previous ``startswith("/")`` gate never matched, so a
    second instance under a different ``HERMES_HOME`` stayed fail-closed
    suspected and deferred this instance's stale-FTS rebuild forever.
    """

    def test_other_windows_home_scopes_to_other(self, tmp_path):
        db_path = tmp_path / "state.db"
        argv = ["C:\\Users\\a\\.hermes\\hermes-agent\\hermes.exe", "gateway", "run"]
        assert hermes_state_holders._argv_scoped_to_other_home(argv, db_path)

    def test_option_value_windows_home_scopes_to_other(self, tmp_path):
        db_path = tmp_path / "state.db"
        argv = ["hermes.exe", "--db=C:\\Users\\a\\.hermes\\state.db", "gateway"]
        assert hermes_state_holders._argv_scoped_to_other_home(argv, db_path)
