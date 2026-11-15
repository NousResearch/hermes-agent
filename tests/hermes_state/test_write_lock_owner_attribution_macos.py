"""Lock-owner attribution on non-Linux hosts (macOS) — 2026-09-28 RCA follow-up.

Incident receipt: during the 10:17–10:21 lock storm, the exhausted-patience
logger emitted 38 lines of "no write-class lock held at the deadline (the holder
released just before it)" because ``state_db_write_lock_holders`` returned [] on
any non-Linux platform (hermes_state_lockowners gated on sys.platform — macOS
has no /proc/locks, and fcntl byte-range lock owners are not enumerable there
at all).

That empty fallback reads as "nobody held it", which is false-by-omission: the
holder existed and was later named by the in-process holder warning. macOS
cannot name the lock OWNER, but the tree's holder authority names every process
with the db (or a WAL sidecar) open — the candidate set that contains the
holder.

The real cross-process psutil scan is probe-verified on this machine (skeptic
2026-09-28, /tmp probe) but the live scan can hit unrelated processes touching
the real hermes home, which upstream's anti-test-fixture guard turns into a
pid=-1 "unproven" sentinel. So these tests drive the scan through a controlled
fake psutil and own the contract that matters here: the FALLBACK WIRING and its
honest wording. The Linux /proc/locks path keeps its exact semantics
(test_write_lock_owner_attribution.py).
"""

import sqlite3
import sys
from types import SimpleNamespace

import pytest

import hermes_state_holders
from hermes_state_lockowners import state_db_write_lock_holders


class _FakePsutil:
    """Controlled stand-in: process_iter + Process(pid).cmdline(), nothing else."""

    def __init__(self, open_by_pid):
        self._open_by_pid = open_by_pid

    def process_iter(self, fields):
        return iter([
            SimpleNamespace(info={
                "pid": pid,
                "open_files": [SimpleNamespace(path=p) for p in paths],
            })
            for pid, paths in self._open_by_pid.items()
        ])

    def Process(self, pid):
        return SimpleNamespace(cmdline=lambda: [sys.executable, "-c", "hold-state-db"],
                               name=lambda: "python")


@pytest.fixture()
def quiet_db(tmp_path):
    db_path = tmp_path / "state.db"
    setup = sqlite3.connect(str(db_path))
    setup.execute("CREATE TABLE t(x)")
    setup.commit()
    setup.close()
    return db_path


@pytest.mark.platforms("macos", "windows")
class TestNonLinuxAttribution:
    def test_open_file_holder_is_named_as_candidate(self, quiet_db, monkeypatch):
        """A foreign process holding the db open must be named in the attribution
        lines when /proc/locks is unavailable — with wording that says CANDIDATE,
        so an operator never reads a descriptor match as lock-proof."""
        monkeypatch.setattr(hermes_state_holders, "psutil",
                            _FakePsutil({999999: [str(quiet_db)]}))
        lines = state_db_write_lock_holders(quiet_db)
        joined = "\n".join(lines)
        assert "999999" in joined, f"holder pid missing from: {joined!r}"
        assert "candidate" in joined.lower(), (
            "non-Linux attribution is descriptor-based, not lock-proof — the "
            f"wording must say so: {joined!r}"
        )

    def test_quiet_db_names_no_foreign_candidates(self, quiet_db, monkeypatch):
        """Polarity: with no foreign process holding the db open, attribution must
        not invent candidates — and must say the honest thing: the holder, if any,
        is likely in THIS process (the 2026-09-28 incident class — the exhausted-
        patience logger runs inside the very process that held the lock)."""
        monkeypatch.setattr(hermes_state_holders, "psutil", _FakePsutil({}))
        lines = state_db_write_lock_holders(quiet_db)
        joined = "\n".join(lines)
        assert "open-file candidate" not in joined, f"no foreign holder — no candidates: {lines!r}"
        assert "this process" in joined.lower(), (
            f"the in-process hint is the honest fallback when the scan is blind to "
            f"own-process handles: {lines!r}"
        )

    def test_unproven_sentinel_is_never_rendered_as_a_named_holder(self, quiet_db, monkeypatch):
        """foreign_state_db_holders returns pid=-1 *unproven* rows when the scan
        itself fails (fixture-guard trips, psutil unavailable). The target of such
        a row is a reason string, NOT a path — it must never read as "has X open".
        Honesty here is the entire point of the fallback (38 false "no holder"
        lines in the 2026-09-28 incident)."""
        monkeypatch.setattr(
            hermes_state_holders, "foreign_state_db_holders",
            lambda db_path: [(-1, "open-file scan failed: reason")],
        )
        lines = state_db_write_lock_holders(quiet_db)
        joined = "\n".join(lines)
        assert "unproven: open-file scan failed: reason" in joined, f"sentinel lost its reason: {lines!r}"
        assert "has" not in joined, f"a scan failure must never read as a named holder: {lines!r}"
        assert "candidate" not in joined.lower()

    def test_duplicate_pid_sidecar_descriptors_collapse_per_pid_and_subsecond_patience_logged(
        self, quiet_db, monkeypatch, caplog
    ):
        """A single process opening state.db, state.db-wal, and state.db-shm across multiple
        connections must collapse to one candidate line per PID rather than filling all 6 slots,
        and sub-second patience (0.5s) must log as '0.5s' rather than rounding to '0s'."""
        from hermes_state_lockowners import log_write_lock_holders

        wal_path = str(quiet_db) + "-wal"
        shm_path = str(quiet_db) + "-shm"
        monkeypatch.setattr(
            hermes_state_holders,
            "psutil",
            _FakePsutil(
                {
                    4143: [str(quiet_db), wal_path, shm_path, str(quiet_db), str(quiet_db), wal_path] * 5,
                    21820: [str(quiet_db)],
                }
            ),
        )
        lines = state_db_write_lock_holders(quiet_db)
        assert len(lines) == 2, lines
        assert "4143" in lines[0] and "state.db, state.db-wal, state.db-shm" in lines[0]
        assert "21820" in lines[1]

        with caplog.at_level("WARNING"):
            log_write_lock_holders(quiet_db, 0.5)
        assert "state.db write lock unavailable for 0.5s (state.db):" in caplog.text
        assert "unavailable for 0s" not in caplog.text


def test_linux_branch_semantics_unchanged(monkeypatch, tmp_path):
    """The Linux path keeps /proc/locks as its authority: on Linux the open-file
    fallback must NOT run (the descriptor scan would add every reader as a false
    candidate). We pin the routing: with sys.platform faked to linux and
    /proc/locks unreadable, the function returns [] — never descriptor
    candidates."""
    import hermes_state_lockowners as lo

    monkeypatch.setattr(lo.sys, "platform", "linux")
    monkeypatch.setattr(hermes_state_holders, "psutil",
                        _FakePsutil({999999: [str(tmp_path / "state.db")]}))
    with monkeypatch.context() as m:
        m.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("no /proc here")))
        lines = lo.state_db_write_lock_holders(tmp_path / "state.db")
    assert lines == [], "Linux branch must fail closed via /proc/locks, not descriptor scan"
