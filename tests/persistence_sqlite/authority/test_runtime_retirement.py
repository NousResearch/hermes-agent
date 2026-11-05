"""``SessionRuntimeRepository`` on SQLite against main: the retirement reads, the generation walks, the viewer closes,
the policy picks and refresh, and the two ledger-gated ends.

Each verb runs main's function (or the gateway, TUI or webhook SQL it absorbed) and the backend verb on copies of one
seed (same outcome, same rows) and opens exactly one transaction; a write that fails right after its first mutating
statement leaves every table unchanged. ``first_generation`` is checked for the three ``GenerationChain`` kinds against
main's ``first_generation(db, derive, pinned=..., spent=...)``: the verb returns the row and the caller applies the spent
rule, calling again from the next generation.
"""

from __future__ import annotations

import json
import time
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest

from persistence.authority.bindings import api_generation_id, generation_request_id, local_session_id, room_session_id
from persistence.authority.runtime import ApiGenerations, LocalGenerations, PinnedIdentity, RoomGenerations
from tests.persistence_sqlite.authority._admission_support import admit, begin_epoch, claim, settle
from tests.persistence_sqlite.authority._support import assert_same, run_both
from tests.persistence_sqlite.authority.test_binding_local import make_binding, reserved as reserving
from tests.persistence_sqlite.authority.test_runtime_picks import assert_one_read, assert_one_write, build_seed

PROFILE = "default"
PRINCIPAL = "user-1"
OWNER = "owner-a"
SID = str(local_session_id(PROFILE, PRINCIPAL, "req-1"))


@pytest.fixture
def frozen_clock(monkeypatch):
    """Both runs stamp the same ``time.time()``, so the rows they end compare exactly."""
    import time

    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)


def seed(store: Any, *steps: Any, legacy: bool = True) -> None:
    def run(db: Any) -> None:
        begin_epoch(db, OWNER)
        for step in steps:
            step(db)
    build_seed(store, run, legacy=legacy)


def meta(key: str, value: Any):
    return lambda db: db.set_meta(key, value if isinstance(value, str) else json.dumps(value))


def retired(sid: str, *, expires: bool = False):
    from hermes_state_mutation_retirement import RETIRED_PREFIX

    return meta(RETIRED_PREFIX + sid, {"retired_at": 1.0, **({"expires": True} if expires else {})})


def reserved(db: Any) -> None:
    """Reserve the local session ``SID`` with the policy ``_POLICY``."""
    reserving(make_binding(SID, request_json="{}"))(db)


# ── retired_session / expiring_retirements ────────────────────────────────────


@pytest.mark.parametrize("sid", ["s-gone", "s-swept", "s-live"])
def test_retired_session_matches_m1(store, legacy_store, sid):
    from hermes_state_mutation_retirement import retired_session

    seed(store, retired("s-gone"), retired("s-swept", expires=True))
    legacy, ours = run_both(store.store_path, lambda db: retired_session(db, sid), lambda: store.runtime.retired_session(sid))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.runtime.retired_session(sid))


def test_expiring_retirements_matches_m1(store, legacy_store):
    from hermes_state_mutation_retirement import expiring_retirements

    ids = ["s-gone", "s-swept", "s-live"]
    seed(store, retired("s-gone"), retired("s-swept", expires=True))
    legacy, ours = run_both(store.store_path, lambda db: expiring_retirements(db, ids), lambda: store.runtime.expiring_retirements(ids))
    assert_same(legacy, ours)
    assert ours[1] == {"s-swept"}
    assert_one_read(store, lambda: store.runtime.expiring_retirements(ids))


# ── first_generation ──────────────────────────────────────────────────────────


IDENTITY = ("home-1", "room-1", "member-1", "default")
_CHAINS = {
    "api": (ApiGenerations("api-root"), partial(api_generation_id, "api-root")),
    "room": (RoomGenerations(IDENTITY), lambda n: room_session_id(IDENTITY, n)),
    "local": (LocalGenerations(PROFILE, PRINCIPAL, "bot-chat:default"),
              lambda n: local_session_id(PROFILE, PRINCIPAL, generation_request_id("bot-chat:default", n))),
}


def _spent(conn: Any, sid: str) -> bool:
    """Main's Bot Chat spent rule (gateway/session_bot.py)."""
    from gateway.session_route_recovery import _messaging_platform

    row = conn.execute('SELECT title, source FROM sessions WHERE id=?', (sid,)).fetchone()
    return row is not None and (bool(row[0]) or _messaging_platform(row[1]) is not None)


def _row_spent(row: Any) -> bool:
    from gateway.session_route_recovery import _messaging_platform

    return row is not None and (bool(row.title) or _messaging_platform(row.source) is not None)


def _generations(derive: Any) -> list[Any]:
    """Generation 0 retired, 1 titled (spent), 2 handed off to Telegram (spent), 3 retired with the pinned identity's
    tombstone, 4 free."""
    from hermes_state_terminal import identity_key

    def titled(db: Any) -> None:
        db.create_session(str(derive(1)), source="cli")
        db.set_session_title(str(derive(1)), "Renamed")

    return [retired(str(derive(0))), titled, lambda db: db.create_session(str(derive(2)), source="telegram"),
            retired(str(derive(3))), meta(identity_key(PRINCIPAL, str(derive(3)), "req-pinned"), json.dumps("adm-1"))]


@pytest.mark.parametrize("kind", sorted(_CHAINS))
def test_first_generation_matches_m1_without_the_spent_rule(store, legacy_store, kind):
    from hermes_state_mutation_retirement import first_generation

    chain, derive = _CHAINS[kind]
    seed(store, *_generations(derive))
    legacy, ours = run_both(store.store_path, lambda db: first_generation(db, derive),
                            lambda: store.runtime.first_generation(chain))
    assert_same(legacy, ours, lambda pick: (pick.generation, pick.session_id))
    assert ours[1].generation == 1 and ours[1].row is not None and ours[1].row.title == "Renamed"
    assert_one_read(store, lambda: store.runtime.first_generation(chain))


@pytest.mark.parametrize("kind", sorted(_CHAINS))
def test_first_generation_with_the_callers_spent_rule_matches_m1(store, legacy_store, kind):
    from hermes_state_mutation_retirement import first_generation

    chain, derive = _CHAINS[kind]
    seed(store, *_generations(derive))

    def walk() -> Any:
        pick = store.runtime.first_generation(chain)
        while _row_spent(pick.row):
            pick = store.runtime.first_generation(chain, start=pick.generation + 1)
        return pick

    legacy, ours = run_both(store.store_path, lambda db: first_generation(db, derive, spent=_spent), walk)
    assert_same(legacy, ours, lambda pick: (pick.generation, pick.session_id))
    assert ours[1].generation == 4 and ours[1].row is None


@pytest.mark.parametrize("kind", sorted(_CHAINS))
@pytest.mark.parametrize("request_id", ["req-pinned", "req-other"])
def test_first_generation_pinned_matches_m1(store, legacy_store, kind, request_id):
    from hermes_state_mutation_retirement import first_generation
    from hermes_state_terminal import identity_key

    chain, derive = _CHAINS[kind]
    seed(store, *_generations(derive))

    def pinned(conn: Any, sid: str) -> bool:
        # main's pinned checks in gateway/session_api.py and gateway/session_a2a.py
        return conn.execute('SELECT 1 FROM state_meta WHERE key=?', (identity_key(PRINCIPAL, sid, request_id),)).fetchone() is not None

    legacy, ours = run_both(
        store.store_path, lambda db: first_generation(db, derive, pinned=pinned, spent=_spent),
        lambda: _walk_pinned(store, chain, PinnedIdentity(PRINCIPAL, request_id)))
    assert_same(legacy, ours, lambda pick: (pick.generation, pick.session_id))


def _walk_pinned(store: Any, chain: Any, pinned: PinnedIdentity) -> Any:
    pick = store.runtime.first_generation(chain, pinned=pinned)
    while _row_spent(pick.row):
        pick = store.runtime.first_generation(chain, pinned=pinned, start=pick.generation + 1)
    return pick


# ── viewer closes ─────────────────────────────────────────────────────────────


def _authority(db: Any, epoch: int = 1) -> Any:
    return SimpleNamespace(db=db, epoch=epoch)


def _m2_finalize_webhook(authority: Any, receipt: Any) -> bool:
    """Main's webhook finalize check in gateway/platforms/webhook_ingress.py."""
    if receipt.status != "terminal" or receipt.execution_generation is None:
        return False
    sid = receipt.ref.session_id
    def write(conn):
        return authority.db._end_and_bump(conn, """
            UPDATE sessions SET ended_at=?, end_reason=?
            WHERE id=? AND ended_at IS NULL AND runtime_generation=?
              AND EXISTS (SELECT 1 FROM runtime_epoch WHERE singleton=1 AND epoch=?)
              AND EXISTS (SELECT 1 FROM session_admissions WHERE admission_id=?
                AND target_session_id=? AND status='terminal' AND generation=? AND owner_epoch=?)
              AND NOT EXISTS (SELECT 1 FROM session_admissions WHERE target_session_id=? AND status!='terminal')
            """, (time.time(), "webhook_complete", sid, receipt.execution_generation,
                  authority.epoch, receipt.admission_id, sid, receipt.execution_generation,
                  receipt.authority_epoch, sid), sid, "webhook_complete")
    return bool(authority.db._execute_write(write))


@pytest.mark.parametrize("epoch", [1, 2])
def test_store_viewer_close_matches_m1(store, legacy_store, epoch):
    from tests.persistence_sqlite._main_callers.session_viewer_close import _persist_close

    seed(store, meta("gateway.viewer_close.v1:s-1", "old"))
    legacy, ours = run_both(store.store_path, lambda db: _persist_close(_authority(db, epoch), "s-1", "record-1"),
                            lambda: store.runtime.store_viewer_close(epoch, session_id="s-1", record="record-1"))
    assert_same(legacy, ours)


@pytest.mark.parametrize("record", ["record-1", "record-other"])
def test_forget_viewer_close_matches_m1(store, legacy_store, record):
    from tests.persistence_sqlite._main_callers.session_viewer_close import forget_persisted_close

    seed(store, meta("gateway.viewer_close.v1:s-1", "record-1"))
    legacy, ours = run_both(store.store_path, lambda db: forget_persisted_close(_authority(db), "s-1", record),
                            lambda: store.runtime.forget_viewer_close(1, session_id="s-1", record=record))
    assert_same(legacy, ours)


def test_forget_viewer_close_refuses_a_stale_epoch_where_m1_logged_false(store, legacy_store, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    from tests.persistence_sqlite._main_callers import session_viewer_close
    from tests.persistence_sqlite._main_callers.session_viewer_close import forget_persisted_close

    from persistence import FenceStale

    # main's function catches its legacy store's refusal class
    monkeypatch.setattr(session_viewer_close, "AuthorityRefusal", RuntimeStoreError)
    seed(store, meta("gateway.viewer_close.v1:s-1", "record-1"))
    legacy, ours = run_both(store.store_path, lambda db: forget_persisted_close(_authority(db, 2), "s-1", "record-1"),
                            lambda: store.runtime.forget_viewer_close(2, session_id="s-1", record="record-1"))
    # CONTRACT: the verb raises the fence; the caller logs and answers False, as main's function does.
    assert legacy == ("ok", False)
    assert ours[0] == "error" and isinstance(ours[1], FenceStale)


def test_viewer_close_writes_are_one_write(store):
    seed(store, legacy=False)
    assert_one_write(store, lambda: store.runtime.store_viewer_close(1, session_id="s-1", record="record-1"))
    assert_one_write(store, lambda: store.runtime.forget_viewer_close(1, session_id="s-1", record="record-1"))


def test_viewer_close_reads_match_m1(store, legacy_store):
    from tests.persistence_sqlite._main_callers.session_viewer_close import CLOSE_PREFIX

    seed(store, meta(CLOSE_PREFIX + "s-1", "record-1"), meta(CLOSE_PREFIX + "s-2", "record-2"))
    legacy, ours = run_both(store.store_path, lambda db: db.get_meta(CLOSE_PREFIX + "s-1"), lambda: store.runtime.viewer_close("s-1"))
    assert_same(legacy, ours)

    def stored(db: Any) -> Any:
        # main's read in gateway/session_viewer_close.py
        with db._read_ctx() as conn:
            rows = conn.execute('SELECT key, value FROM state_meta WHERE key LIKE ?', (CLOSE_PREFIX + '%',)).fetchall()
        return [(key[len(CLOSE_PREFIX):], record) for key, record in rows]

    legacy, ours = run_both(store.store_path, stored, store.runtime.viewer_closes)
    assert_same(legacy, ours)
    assert sorted(ours[1]) == [("s-1", "record-1"), ("s-2", "record-2")]
    assert_one_read(store, store.runtime.viewer_closes)


# ── policy picks and the policy refresh ───────────────────────────────────────


def _receipt(name: str, result: dict) -> Any:
    return meta("gateway.mutation.v1." + name, {"digest": "d-" + name, "result": result})


@pytest.mark.parametrize("session_id", [SID, "s-branch", "s-branch-picks", "s-none"])
def test_policy_picks_matches_m1(store, legacy_store, session_id):
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_mutation_prepared import policy_picks

    seed(store, reserved,
         _receipt("a", {"session_id": SID, "operation": "model", "revision": 1}),
         _receipt("b", {"session_id": SID, "operation": "reasoning", "status": "deferred", "revision": 2}),
         _receipt("c", {"session_id": SID, "operation": "workspace", "revision": 3}),
         _receipt("d", {"session_id": SID, "operation": "branch", "revision": 2, "branched_session_id": "s-branch"}),
         meta(POLICY_PREFIX + "s-branch", {"session_id": "s-branch"}),
         meta(POLICY_PREFIX + "s-branch-picks", {"session_id": "s-branch-picks", "picks": ["reasoning"]}))
    legacy, ours = run_both(store.store_path, lambda db: policy_picks(db, session_id), lambda: store.runtime.policy_picks(session_id))
    assert_same(legacy, ours)
    assert_one_read(store, lambda: store.runtime.policy_picks(session_id))


_POLICY = {"source": "cli", "model": "model-a", "cwd": "/workspace", "request_json": "{}"}


def _queued(db: Any) -> None:
    from hermes_state_runtime import admit_session_input

    admit_session_input(db, epoch=1, principal_id=PRINCIPAL, session_id=SID, request_id="in-1", payload={"text": "hi"})


@pytest.mark.parametrize("case", ["refresh", "old_changed", "generation_moved", "busy", "no_receipt", "stale_epoch"])
def test_commit_refreshed_policy_matches_m1(store, legacy_store, case):
    from hermes_state_mutation_prepared import commit_refreshed_policy

    steps = [] if case == "no_receipt" else [reserved]
    if case == "no_receipt":
        steps.append(lambda db: db.create_session(SID, source="cli"))
    if case == "busy":
        steps.append(_queued)
    seed(store, *steps)
    old = {**_POLICY, "model": "model-x"} if case == "old_changed" else _POLICY
    generation = 5 if case == "generation_moved" else 0
    epoch = 2 if case == "stale_epoch" else 1
    new = {**_POLICY, "model": "model-b"}
    legacy, ours = run_both(
        store.store_path, lambda db: commit_refreshed_policy(db, epoch=epoch, session_id=SID, old=old, new=new, generation=generation),
        lambda: store.runtime.commit_refreshed_policy(epoch, session_id=SID, old=old, new=new, generation=generation))
    assert_same(legacy, ours)


def test_commit_refreshed_policy_is_one_write(store):
    seed(store, reserved, legacy=False)
    assert_one_write(store, lambda: store.runtime.commit_refreshed_policy(
        1, session_id=SID, old=_POLICY, new={**_POLICY, "model": "model-b"}, generation=0))


# ── end_unless_ledger_busy ────────────────────────────────────────────────────

# Main's TUI caller body (tui_gateway/session_lifecycle.py), the legacy side of the parity check.
_M1_CANONICAL_PENDING_SQL = """SELECT status FROM session_admissions WHERE target_session_id=? AND status!='terminal'
    UNION ALL SELECT status FROM worker_executions WHERE session_id=? AND status!='terminal' LIMIT 1"""


def _m1_end_unless_ledger_busy(db, session_id: str, session_key, end_reason: str) -> bool:
    """End *session_id*'s row unless the gateway's durable ledger still holds work for it; True when
    the row may end (ended now, or already ended). The check and the stamp share one write txn.

    A viewer process (a dashboard, a second Desktop window) runs this reaper for a session the
    owning gateway is still executing; stamping it ended mid-turn made the owner treat its own live
    route as stale (#54878) and replace the conversation (#140642). The ledger names a local session
    by its logical id (the row's ``chat_id``), so the physical id, the key and that id are checked: a
    ``local-`` id, or the lineage owner its receipt names (an adopted conversation keeps its legacy id)."""
    execute_write = getattr(db, "_execute_write", None)
    if execute_write is None:
        db.end_session(session_id, end_reason)
        return True

    def write(conn):
        row = conn.execute("SELECT chat_id FROM sessions WHERE id=?", (session_id,)).fetchone()
        ids = {session_id, str(session_key or "")}
        if row is not None and str(row[0] or "").startswith("local-"):
            ids.add(str(row[0]))
        from hermes_state_local import local_lineage_owner
        ids.add(local_lineage_owner(conn, session_id))
        for sid in filter(None, ids):
            if conn.execute(_M1_CANONICAL_PENDING_SQL, (sid, sid)).fetchone() is not None:
                return False
        db._end_and_bump(conn, "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                         (time.time(), end_reason, session_id), session_id, end_reason)
        return True

    if execute_write(write):
        return True
    return False



@pytest.mark.parametrize("case", ["idle", "pending_on_id", "pending_on_key", "pending_on_owner", "already_ended", "missing"])
def test_end_unless_ledger_busy_matches_m1(store, legacy_store, frozen_clock, case):
    from hermes_state_runtime import admit_session_input
    steps = [lambda db: db.create_session("s-1", source="cli"), lambda db: db.create_session("key-1", source="cli"), reserved]
    if case == "pending_on_id":
        steps.append(lambda db: admit_session_input(db, epoch=1, principal_id=PRINCIPAL, session_id="s-1", request_id="in-1", payload={"text": "hi"}))
    if case == "pending_on_key":
        steps.append(lambda db: admit_session_input(db, epoch=1, principal_id=PRINCIPAL, session_id="key-1", request_id="in-1", payload={"text": "hi"}))
    if case == "already_ended":
        steps.append(lambda db: db.end_session("s-1", "user_exit"))
    target = SID if case == "pending_on_owner" else ("s-none" if case == "missing" else "s-1")
    if case == "pending_on_owner":
        steps.append(_queued)
    seed(store, *steps)
    legacy, ours = run_both(store.store_path, lambda db: _m1_end_unless_ledger_busy(db, target, "key-1", "tui_close"),
                            lambda: store.runtime.end_unless_ledger_busy(target, session_key="key-1", end_reason="tui_close"))
    assert_same(legacy, ours)


def test_end_unless_ledger_busy_is_one_write(store):
    seed(store, lambda db: db.create_session("s-1", source="cli"), legacy=False)
    assert_one_write(store, lambda: store.runtime.end_unless_ledger_busy("s-1", session_key=None, end_reason="tui_close"))


# ── end_after_settlement ──────────────────────────────────────────────────────


def _settled(db: Any, *, pending: bool = False) -> dict:
    db.create_session("s-hook", source="webhook")
    admitted = admit(db, "s-hook", "in-1", payload={"text": "hi"}, principal_id="webhook")
    claimed = claim(db, "s-hook")
    settle(db, claimed, result={"ok": True})
    if pending:
        admit(db, "s-hook", "in-2", payload={"text": "again"}, principal_id="webhook")
    return {"admission_id": admitted["admission_id"], "generation": claimed["generation"]}


@pytest.mark.parametrize("case", ["settled", "pending", "stale_epoch", "moved_generation", "other_admission_epoch"])
def test_end_after_settlement_matches_m1(store, legacy_store, frozen_clock, case):
    found: dict = {}
    seed(store, lambda db: found.update(_settled(db, pending=case == "pending")))
    epoch = 2 if case == "stale_epoch" else 1
    generation = found["generation"] + (1 if case == "moved_generation" else 0)
    admission_epoch = 2 if case == "other_admission_epoch" else 1
    receipt = SimpleNamespace(status="terminal", execution_generation=generation, ref=SimpleNamespace(session_id="s-hook"),
                              admission_id=found["admission_id"], authority_epoch=admission_epoch)
    legacy, ours = run_both(
        store.store_path, lambda db: _m2_finalize_webhook(_authority(db, epoch), receipt),
        lambda: store.runtime.end_after_settlement(epoch, session_id="s-hook", admission_id=found["admission_id"],
                                                   generation=generation, admission_epoch=admission_epoch, reason="webhook_complete"))
    assert_same(legacy, ours)
    assert ours[1] is (case == "settled")


def test_end_after_settlement_is_one_write(store):
    found: dict = {}
    seed(store, lambda db: found.update(_settled(db)), legacy=False)
    assert_one_write(store, lambda: store.runtime.end_after_settlement(
        1, session_id="s-hook", admission_id=found["admission_id"], generation=found["generation"], admission_epoch=1,
        reason="webhook_complete"))
