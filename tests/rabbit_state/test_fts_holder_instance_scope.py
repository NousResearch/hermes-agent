"""Instance-scoping of the uninspectable-holder fallback.

Field-verified 2026-09-07 on a production host running TWO independent
Rabbit instances: a main gateway (user ``ubuntu``,
RABBIT_HOME=/home/ubuntu/.rabbit) and a demo gateway (user ``demo``,
RABBIT_HOME=/home/demo/.rabbit).  The demo gateway runs as another user, so
its ``/proc/<pid>/fd`` table is unreadable from the main instance and the
holder scan falls back to ``/proc/<pid>/cmdline`` + ``_looks_like_rabbit``.
The demo process's argv matches the Rabbit patterns exactly, so the fallback
flagged it as an uninspectable holder of the MAIN instance's state.db even
though ``lsof`` proved zero open handles on it.  Consequence: the stale-FTS
rebuild in ``rabbit_state_schema._recover_stale_fts`` was deferred 42 times
across 6 gateway restarts, the ``fts_stale`` breadcrumb never cleared, and
FTS self-repair stayed permanently disabled.

PR #92419 fixed substring false positives (journalctl/grep mentioning
rabbit); a genuine second instance with a DIFFERENT RABBIT_HOME is still
misjudged on current main (issue #92401).

Behavior contract: an uninspectable holder identified only by argv must be
counted unless its own argv proves it is scoped to a *different* Rabbit
home / state.db and never references ours.  Ambiguous argv (no absolute
paths at all) must remain fail-closed, exactly as before — the conservative
intent of the fallback is preserved.
"""

import os

import pytest

import rabbit_state_holders

# Capture the pristine stdlib functions at import time: monkeypatched calls
# re-enter these closures, and re-capturing ``os.listdir`` after a previous
# patch would compose the fakes into a double path-rewrite.
_REAL_LISTDIR = os.listdir
_REAL_READLINK = os.readlink


# Representative demo-gateway argv on the two-instance host: every absolute
# token lives under /home/demo/.rabbit, the binary name matches the Rabbit
# patterns, and nothing references the main instance's home or state.db.
DEMO_HOME_ARGV = [
    "/home/demo/.rabbit/rabbit-agent/rabbit",
    "gateway",
    "run",
]

# Same, spelled through the venv interpreter + rabbit launcher script.
DEMO_VENV_ARGV = [
    "/home/demo/.rabbit/rabbit-agent/venv/bin/python",
    "/home/demo/.rabbit/rabbit-agent/rabbit_cli/main.py",
    "gateway",
]

# A Rabbit-shaped argv with no absolute paths: cannot disprove that this
# process touches our state.db, so it must stay fail-closed.
AMBIGUOUS_ARGV = ["rabbit", "gateway", "run"]


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

    monkeypatch.setattr(rabbit_state_holders.os, "getpid", lambda: 111)

    def _listdir(path):
        if isinstance(path, str):
            for pid in unreadable_pids:
                if path == f"/proc/{pid}/fd":
                    raise PermissionError(errno_value("EACCES"), path)
            path = path.replace("/proc", str(proc_root))
        return _REAL_LISTDIR(path)

    monkeypatch.setattr(rabbit_state_holders.os, "listdir", _listdir)

    def _readlink(path):
        if "222/fd/3" in str(path):
            raise PermissionError(errno_value("EACCES"), str(path))
        return _REAL_READLINK(str(path).replace("/proc", str(proc_root)))

    monkeypatch.setattr(rabbit_state_holders.os, "readlink", _readlink)


def errno_value(name):
    import errno

    return getattr(errno, name)


def _install_fake_argv(monkeypatch, argv_by_pid):
    monkeypatch.setattr(
        rabbit_state_holders,
        "_read_proc_argv",
        lambda pid: list(argv_by_pid.get(pid)) if pid in argv_by_pid else None,
    )


@pytest.mark.platforms("linux")
class TestUninspectableHolderInstanceScope:
    def test_other_instance_argv_is_not_a_holder_of_our_db(self, tmp_path, monkeypatch):
        """RED: fd dir unreadable + argv proves the process belongs to a
        DIFFERENT Rabbit home → not a holder of our state.db."""
        # A real ``.rabbit`` home, as on the field host this test is drawn from: the install
        # location can only identify a home that is itself part of an install layout.
        db_path = tmp_path / ".rabbit" / "state.db"
        _install_fake_proc(monkeypatch, tmp_path, unreadable_pids=(222,))
        _install_fake_argv(monkeypatch, {222: DEMO_HOME_ARGV})

        holders = rabbit_state_holders.foreign_state_db_holders(db_path)
        assert holders == []

    def test_argv_referencing_our_db_stays_flagged(self, tmp_path, monkeypatch):
        """A (possibly second) instance whose argv names OUR state.db, our
        sidecars, or our home must still be fail-closed flagged."""
        db_path = tmp_path / "state.db"
        our_home = str(tmp_path)

        for argv in (
            ["rabbit", f"--db={db_path}", "gateway"],
            ["rabbit", "checkpoint", f"{db_path}-wal"],
            ["rabbit", "--home", our_home, "gateway"],
        ):
            assert rabbit_state_holders._looks_like_rabbit(argv) or argv[0] == "rabbit"
            _install_fake_proc(monkeypatch, tmp_path, unreadable_pids=(222,))
            _install_fake_argv(monkeypatch, {222: argv})

            holders = rabbit_state_holders.foreign_state_db_holders(db_path)
            assert [pid for pid, _ in holders] == [222], argv
            assert holders[0][1].startswith("uninspectable holder:"), argv

    def test_install_root_argv_still_holds_a_profile_store_under_it(self, tmp_path, monkeypatch):
        """One process per host serves EVERY profile, so argv naming only the install root does
        not prove the process is another instance: it holds ``<root>/profiles/<name>/state.db``
        too. A DIFFERENT install root is still proof."""
        root = tmp_path / ".rabbit"
        db_path = root / "profiles" / "b" / "state.db"
        db_path.parent.mkdir(parents=True)
        multiplexer = ["rabbit", "--home", str(root), "gateway", "run"]
        other_install = ["rabbit", "--home", "/home/demo/.rabbit", "gateway", "run"]

        _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
        _install_fake_argv(monkeypatch, {222: multiplexer})
        assert [pid for pid, _ in rabbit_state_holders.foreign_state_db_holders(db_path)] == [222]

        _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
        _install_fake_argv(monkeypatch, {222: other_install})
        assert rabbit_state_holders.foreign_state_db_holders(db_path) == []

    def test_shared_binary_plus_another_profile_selection_is_dismissed(self, tmp_path, monkeypatch):
        """argv[0] is the SHARED install binary, so it cannot prove a hold of profile b's store.

        Every rabbit process on a normal host runs ``<root>/venv/bin/rabbit``; counting that token
        as proof made ``rabbit -p other chat -q`` an uninspectable holder of every OTHER profile's
        state.db, deferring its FTS rebuild and auto-VACUUM for as long as the sibling lived
        (#92401, inside a single install). The process's own ``-p``/``--profile`` selection decides.
        """
        root = tmp_path / ".rabbit"
        db_path = root / "profiles" / "b" / "state.db"
        db_path.parent.mkdir(parents=True)
        shared_binary = str(root / "venv" / "bin" / "rabbit")

        for argv, expected in (
            ([shared_binary, "-p", "other", "chat", "-q"], []),
            ([shared_binary, "--profile=other", "gateway", "run"], []),
            # Control: the sibling that DOES serve profile b is still a fail-closed holder.
            ([shared_binary, "-p", "b", "gateway", "run"], [222]),
            # Control: no selection at all stays fail-closed (it may be the multiplexer).
            ([shared_binary, "gateway", "run"], [222]),
        ):
            _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
            _install_fake_argv(monkeypatch, {222: argv})
            holders = rabbit_state_holders.foreign_state_db_holders(db_path)
            assert [pid for pid, _ in holders] == expected, argv

    def test_token_naming_another_profiles_store_is_dismissal_evidence(self, tmp_path, monkeypatch):
        """A token positively naming ANOTHER profile's store is the strongest dismissal there is.

        It sits under our install root, so a blanket own-prefix test read the strongest evidence of
        a different scope as proof of ours.
        """
        root = tmp_path / ".rabbit"
        db_path = root / "profiles" / "b" / "state.db"
        db_path.parent.mkdir(parents=True)
        other_store = root / "profiles" / "other" / "state.db"

        for argv in (
            ["rabbit", f"--db={other_store}", "sessions", "optimize"],
            [str(root / "venv" / "bin" / "rabbit"), "sessions", str(other_store)],
        ):
            _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
            _install_fake_argv(monkeypatch, {222: argv})
            assert rabbit_state_holders.foreign_state_db_holders(db_path) == [], argv

    def test_unrelated_install_under_a_non_rabbit_profiles_tree_stays_dismissed(
        self, tmp_path, monkeypatch
    ):
        """``<X>/profiles/<n>/state.db`` does not make all of ``<X>`` ours.

        A raw ``basename == "profiles"`` test promoted any such parent to the install root, so an
        unrelated Rabbit install living under it was counted as a holder — the literal two-instance
        shape #92401 was filed about. The canonical ``named_profile_home`` predicate requires the
        parent to be a real Rabbit home.
        """
        work = tmp_path / "work"
        db_path = work / "profiles" / "b" / "state.db"
        db_path.parent.mkdir(parents=True)
        unrelated_home = work / "demo" / ".rabbit"
        unrelated_home.mkdir(parents=True)

        _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
        _install_fake_argv(
            monkeypatch, {222: ["rabbit", "--rabbit-home", str(unrelated_home), "gateway", "run"]})
        assert rabbit_state_holders.foreign_state_db_holders(db_path) == []

    def test_custom_home_is_not_dismissed_by_the_install_location(self, tmp_path, monkeypatch):
        """A store at a custom RABBIT_HOME is SERVED BY the binary under ``~/.rabbit``.

        Dismissing on that argv[0] admitted auto-VACUUM and the FTS rebuild under the live
        multiplexer that holds the store. argv[0] locates the install, never the home.
        """
        db_path = tmp_path / "custom-store" / "state.db"
        db_path.parent.mkdir(parents=True)

        _install_fake_proc(monkeypatch, db_path.parent, unreadable_pids=(222,))
        _install_fake_argv(
            monkeypatch, {222: ["/home/u/.rabbit/venv/bin/rabbit", "gateway", "run"]})
        holders = rabbit_state_holders.foreign_state_db_holders(db_path)
        assert [pid for pid, _ in holders] == [222]
