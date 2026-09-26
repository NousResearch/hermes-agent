"""Automatic in-place recovery for a lost WAL/SHM generation (#109687).

A short-lived reader's clean close unlinks the ``-wal``/``-shm`` sidecars
while a long-lived gateway writer still holds them, leaving the writer
committing into an unlinked inode. The fail-closed halt (#105670 capture +
sticky refusal) is the default; these tests cover the opt-in
``database.wal_self_heal`` path that lets a flagged writer recover without
operator intervention: capture the retired frames, drop the orphaned
descriptors (close-time checkpoint disabled), reopen through the refuse
guard, adopt the current generation.

The real unlink/adopt flow is Linux-only (Windows cannot unlink a held
sidecar; the /proc deleted-fd scan is Linux-only) — same gate as the guard
tests in test_deleted_wal_generation_guard.py. The gating, refusal and
read-pool-epoch logic is exercised cross-platform by arming a faked loss.
"""

import contextlib
import os
import sys
from pathlib import Path

import pytest

import hermes_state
from hermes_state import DeletedWalGenerationError, SessionDB
from tests.hermes_state._wal_generation_harness import (
    integrity_ok_path, lose_sidecars, make_db, message_count,
    pin_wal, require_wal, write_second_generation,
)


@pytest.fixture
def force_wal(monkeypatch):
    pin_wal(monkeypatch)


@pytest.fixture
def self_heal_enabled(monkeypatch):
    """Opt the runtime into the self-heal without a config file."""
    monkeypatch.setattr(SessionDB, "_wal_self_heal_enabled", lambda self: True)


@pytest.fixture
def armable_loss(monkeypatch):
    """Lose the generation only once the test ARMS it (``db._armed_loss = True``),
    so seeding writes through the same handle still succeed. Works on Windows,
    where unlinking an open -wal raises PermissionError."""
    monkeypatch.setattr(
        SessionDB, "_wal_generation_was_lost",
        lambda self: bool(getattr(self, "_armed_loss", False)),
    )


def test_flag_defaults_off(tmp_path, force_wal):
    db = make_db(tmp_path / "state.db", "s", "x")
    try:
        assert db._wal_self_heal_enabled() is False
    finally:
        db.close()


def test_heal_off_keeps_fail_closed_halt(tmp_path, force_wal, armable_loss, monkeypatch):
    """Unflagged: the halt stays exactly as shipped — capture + sticky refusal.
    The heal is not even attempted: the flag gates the call itself."""
    attempted = []
    monkeypatch.setattr(
        SessionDB, "_try_heal_lost_wal_generation",
        lambda self: attempted.append(1) or False,
    )
    db = make_db(tmp_path / "state.db", "s", "held")
    try:
        db._armed_loss = True
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("s", role="user", content="post-loss")
        assert attempted == []
        assert db._db_wal_generation_lost is True
    finally:
        with contextlib.suppress(Exception):
            db.close()


def test_heal_attempted_then_sticky_when_it_cannot_heal(
    tmp_path, force_wal, self_heal_enabled, armable_loss,
):
    """Flagged: the heal IS attempted; when it cannot complete (capture fails
    here — this process holds no retired inode to preserve) the halt is exactly
    the shipped sticky refusal."""
    db = make_db(tmp_path / "state.db", "s", "held")
    try:
        db._armed_loss = True
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("s", role="user", content="post-loss")
        assert db._db_wal_generation_lost is True  # heal failed closed, sticky
    finally:
        with contextlib.suppress(Exception):
            db.close()


def test_heal_refuses_when_guard_detects_foreign_holders(
    tmp_path, force_wal, self_heal_enabled, armable_loss, monkeypatch,
):
    """_try_heal_lost_wal_generation raises DeletedWalGenerationError when the
    pre-reopen guard finds a foreign holder: the heal never mints a second WAL
    while another live process holds the orphan. Capture is stubbed (the frames
    are faked-preserved) so the flow reaches the guard."""
    monkeypatch.setattr(
        SessionDB, "_capture_retired_generation",
        lambda self, trigger: Path(self.db_path).parent / "stubbed-retired-wal",
    )
    db = make_db(tmp_path / "state.db", "s", "held")
    try:
        db._armed_loss = True
        # Patch the guard AFTER open: _connect_and_init calls it too.
        def _fake_refuse(db_path):
            raise DeletedWalGenerationError("foreign holder")
        monkeypatch.setattr(hermes_state, "refuse_deleted_wal_generation", _fake_refuse)
        with pytest.raises(DeletedWalGenerationError):
            db.append_message("s", role="user", content="must not land")
        assert db._db_wal_generation_lost is True  # sticky: still halted
        assert db._conn is None  # the heal closed the orphaned writer descriptor
    finally:
        with contextlib.suppress(Exception):
            db.close()


def test_heal_adopts_current_generation_and_resumes(
    tmp_path, force_wal, self_heal_enabled, armable_loss, monkeypatch,
):
    """The full in-place heal: capture (stubbed), close the orphaned
    descriptors, reopen through the real guard (clean — no holders), adopt the
    CURRENT generation, clear the sticky halt, resume writes."""
    monkeypatch.setattr(
        SessionDB, "_capture_retired_generation",
        lambda self, trigger: Path(self.db_path).parent / "stubbed-retired-wal",
    )
    path = tmp_path / "state.db"
    db = make_db(path, "s", "held")
    try:
        db._armed_loss = True
        db.append_message("s", role="user", content="post-heal turn")
        # Writes resumed: sticky flag cleared, identity re-adopted.
        assert db._db_wal_generation_lost is False
        assert db._db_sidecar_identity is not None
        contents = [m["content"] for m in db.get_messages("s")]
        assert "post-heal turn" in contents
        # Subsequent writes keep landing (no one-shot).
        db.append_message("s", role="user", content="post-heal turn 2")
        assert "post-heal turn 2" in [m["content"] for m in db.get_messages("s")]
    finally:
        db.close()


def test_pool_evicts_pre_heal_read_conns(tmp_path, force_wal, self_heal_enabled):
    """A read connection minted BEFORE the heal never serves reads afterwards:
    the epoch marker closes it at checkout instead of reusing the orphaned fd."""
    path = tmp_path / "state.db"
    db = make_db(path, "s", "held")
    require_wal(db)
    try:
        # Mint a read conn under the pre-heal epoch, return it to the pool.
        stale = db._checkout_read_conn()
        assert stale is not None
        with db._read_conns_lock:
            db._read_pool.put_nowait(stale)
            assert db._read_conns_closed is False
        # Simulate the heal having adopted a new generation.
        db._fresh_read_conns = set()
        conn = db._checkout_read_conn()
        assert conn is not None
        assert conn is not stale  # stale evicted, a fresh one minted
    finally:
        db.close()


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="real sidecar unlink + /proc deleted-fd scan is Linux-only",
)
def test_heal_recovers_writer_without_operator(tmp_path, force_wal, self_heal_enabled):
    """Flagged, real flow: after the sidecars are unlinked and a second
    generation exists on the path, the next write captures the retired frames,
    drops the orphaned descriptors and resumes on the CURRENT generation."""
    path = tmp_path / "state.db"
    db = make_db(path, "s", "held")
    wal = require_wal(db)
    inode_before = wal.stat().st_ino
    try:
        lose_sidecars(path, rename=False)
        # A separate (non-hermes) opener mints a fresh generation on the path,
        # exactly like the field incident's short-lived CLI reader.
        write_second_generation(path, 1)

        # First write after the loss: the heal must run end to end.
        db.append_message("s", role="user", content="post-heal turn")

        # Writes resumed: sticky flag cleared, retired frames preserved.
        assert db._db_wal_generation_lost is False
        assert db._retired_generation_capture is not None
        artifact = Path(db._retired_generation_capture)
        assert artifact.exists()
        wal_now = Path(os.fspath(path) + "-wal")
        if wal_now.exists():
            assert wal_now.stat().st_ino != inode_before
        # The healed write is visible on the current generation.
        contents = [m["content"] for m in db.get_messages("s")]
        assert "post-heal turn" in contents
        # And it survives a fresh process reading the file.
        rows = message_count(path)
        assert rows >= 2  # held + gen2 row + post-heal turn (>= the originals)
        assert integrity_ok_path(path)
    finally:
        db.close()
