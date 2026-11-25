"""The boot heal for handed-off local receipts scans the store without the state.db write lock.

``release_handed_off_receipts`` joined every local receipt to its session row INSIDE the write
transaction, so every gateway start held the write lock for as long as that scan took (seconds on a
large cold store) and every other writer waited. Now the scan is a read; a boot that finds nothing
opens no write transaction, and one that finds receipts re-checks only those inside the write.

The first two cases fail with the source change reverted (nothing found still opens a write, and
the whole scan runs inside it); the third pins the re-probe of a receipt that stopped being moved.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from persistence.core.identity import SessionId
from persistence.sqlite import _holder
from persistence.sqlite.authority import _local
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved, seed

LANES = frozenset({"telegram"})


def _hand_off(store: Any, sid: str) -> None:
    """A ``/handoff``: the receipt's current target row now carries a Telegram lane's route."""
    from tests.persistence_sqlite.authority.test_runtime_picks import build_seed

    def step(db: Any) -> None:
        db.store.holder()._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET session_key='agent:main:telegram:dm:42', source='telegram', chat_id='42' WHERE id=?",
            (sid,)))
    build_seed(store, step)


@pytest.fixture
def tracked(monkeypatch):
    """Record every write transaction and every ``moved_receipts`` probe, with its timing."""
    events: list[tuple[str, Any]] = []
    real_execute = _holder.StoreHolder._execute_write
    real_moved = _local.moved_receipts

    def execute(holder_self, fn, *a, **k):
        events.append(("write", None))
        return real_execute(holder_self, fn, *a, **k)

    def moved(conn, *a, **k):
        events.append(("probe", k.get("owners")))
        return real_moved(conn, *a, **k)

    monkeypatch.setattr(_holder.StoreHolder, "_execute_write", execute)
    monkeypatch.setattr(_local, "moved_receipts", moved)
    return events


def _chats(store: Any, count: int) -> None:
    seed(store, *[reserved(make_binding(f"chat{n}", request_id=f"chat{n}")) for n in range(count)])


def _keys(store: Any) -> set[str]:
    with store.holder()._read_ctx() as conn:
        return {row[0] for row in conn.execute("SELECT key FROM state_meta WHERE key GLOB ?", (POLICY_PREFIX + "*",))}


def test_a_boot_with_nothing_to_release_opens_no_write_transaction(store, tracked):
    _chats(store, 20)  # their rows carry no lane: nothing was handed off
    tracked.clear()  # the seeding writes are not the boot's
    store.bindings.release_handed_off_receipts(epoch=1, lanes=LANES)
    assert tracked == [("probe", None)], tracked


def test_the_scan_runs_outside_the_write_and_the_write_probes_only_what_it_found(store, tracked):
    seed(store, *[reserved(make_binding(f"chat{n}", request_id=f"chat{n}")) for n in range(5)],
         reserved(make_binding("moved", request_id="moved")))
    _hand_off(store, "moved")
    tracked.clear()  # the seeding writes are not the boot's

    store.bindings.release_handed_off_receipts(epoch=1, lanes=LANES)

    assert tracked[0] == ("probe", None), tracked  # the whole-store scan, before any write
    assert tracked[1] == ("write", None), tracked  # then the write
    assert tracked[2:] == [("probe", ["moved"])], tracked  # which re-checks the one receipt it found, by key
    keys = _keys(store)
    assert POLICY_PREFIX + "moved" not in keys and POLICY_PREFIX + "chat0" in keys, keys


def test_a_receipt_that_stopped_being_moved_between_the_scan_and_the_write_is_kept(store, monkeypatch):
    seed(store, reserved(make_binding("moved", request_id="moved")))
    _hand_off(store, "moved")
    real = _local.moved_receipts

    def raced(conn, *a, **k):
        if k.get("owners") is not None:  # between the read and the write the lane let go of the row
            conn.execute("UPDATE sessions SET session_key='agent:main:local:dm:moved' WHERE id='moved'")
        return real(conn, *a, **k)

    monkeypatch.setattr(_local, "moved_receipts", raced)
    store.bindings.release_handed_off_receipts(epoch=1, lanes=LANES)

    with store.holder()._read_ctx() as conn:
        kept = conn.execute("SELECT count(*) FROM state_meta WHERE key=?", (POLICY_PREFIX + "moved",)).fetchone()[0]
    assert kept == 1


def test_a_point_probe_and_the_whole_scan_agree(store):
    seed(store, reserved(make_binding("a", request_id="a")), reserved(make_binding("b", request_id="b")),
         reserved(make_binding("c", request_id="c")))
    _hand_off(store, "a")
    _hand_off(store, "b")
    with store.holder()._read_ctx() as conn:
        whole = _local.moved_receipts(conn)
        assert set(whole) == {"a", "b"}, whole
        assert _local.moved_receipts(conn, owners=["a", "c", "missing"]) == {"a": whole["a"]}
        assert _local.moved_receipts(conn, owners=[]) == {}
