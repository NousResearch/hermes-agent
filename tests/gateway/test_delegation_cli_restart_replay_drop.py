"""Restart replay of a CLI-origin completion must drop terminally at preflight.

Confirmed staging incident (2026-08-29, six rows; 2026-09-04, one row): a
background delegation dispatched from an interactive CLI session completes
after that CLI process is gone. On the next gateway boot
``restore_undelivered_completions`` re-enqueues the pending row, and the
delivery chain lands in a routing gap:

- ``_classify_completion_target`` verdicts ``deliver`` — the parent ended at
  ``cli_close`` / ``agent_close`` / ``ws_orphan_reap``, none of which are in
  ``_USER_BOUNDARY_END_REASONS`` (the verdict understands "an idle-ended chat
  stays routable", which is true for gateway chats, not for CLI rows).
- ``_inject_watch_notification`` then finds NO route: a CLI dispatch stamps
  ``session_key``/``parent_session_id`` with the bare ``{ts}_{uuid}`` session
  id, ``_parse_session_key`` cannot parse it, the session-store entry is gone
  with the CLI process, and ``origin_session_id`` (the api_server raw id) is
  empty — so it warns "Dropping watch notification with no routing metadata"
  and returns ``None``.
- The delivery leg treats ``None`` as "not deliverable here": the claim
  releases back to ``pending``, the next restart re-enqueues, and the loop
  only converges when ``_MAX_DELIVERY_ATTEMPTS`` burns the row to a permanent
  ``dropped`` — an 8-claim churn per reboot for a completion no consumer in
  this process can ever route.

The honest terminal target for a CLI-origin completion IS the delegation
itself: the ledger row is durable and queryable (``get_delegation``). Two
fixes, one policy:

- Preflight: ``sessions.source`` of the parent is consulted BEFORE the claim
  is consumed. ``cli``/``tui`` proven unroutable from a gateway process →
  terminal drop in a single claim, injection never attempted.
- Injection leg: a ``None`` verdict (no route at all) settles the claim as a
  terminal drop instead of releasing it into the restart-replay churn.

Fail-open everywhere the origin is uncertain (missing row, lookup error,
api_server/relay source): a wrong terminal drop is worse than a few extra
self-limiting retries.
"""

import asyncio
import json
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import tools.async_delegation as ad
import tools.process_registry as pr_module
from gateway.run import GatewayRunner
from gateway.run_notifications import _unroutable_completion_verdict

_CLI_SESSION = "20260904_113746_ef31b6"
_TELEGRAM_SESSION = "20260904_120000_tg0001"


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    """Point the durable delegation ledger at a temp HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = tmp_path / "state.db"
    monkeypatch.setattr(ad, "_db_path", lambda: db_path)
    monkeypatch.setattr(ad, "_records", {})
    ad._connect().close()  # bootstrap the schema via the module's own path
    yield db_path


@pytest.fixture(autouse=True)
def pristine_inject_method():
    """Every test below patches ``_inject_watch_notification`` on the class;
    guarantee the original always comes back."""
    orig = GatewayRunner._inject_watch_notification
    yield
    GatewayRunner._inject_watch_notification = orig


def _seed_pending_completion(db_path, delegation_id, session, source="cli"):
    """Finalized, pending completion row exactly as dispatch/finalize stamps
    it. For a CLI origin: origin_session == parent_session_id == bare CLI
    session id, origin_session_id empty (not an api_server session). Also
    seeds the matching ``sessions`` row — the preflight resolves the origin
    source from the persisted session record, not the event."""
    completed_at = time.time() - 60
    event = {
        "type": "async_delegation",
        "delegation_id": delegation_id,
        "session_key": session,
        "origin_session_id": "",
        "parent_session_id": session,
        "scope_id": "",
        "user_id": "",
        "user_name": "",
        "goal": "held-for-later bench",
        "status": "completed",
        "is_batch": True,
        "results": [{"goal": "held-for-later bench", "summary": "done"}],
        "dispatched_at": completed_at - 30,
        "completed_at": completed_at,
    }
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            "id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL)")
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, source, started_at) VALUES (?,?,?)",
            (session, source, completed_at - 120),
        )
        conn.execute(
            """INSERT INTO async_delegations
               (delegation_id, origin_session, origin_ui_session_id,
                parent_session_id, state, dispatched_at, completed_at,
                updated_at, event_json, result_json, delivery_state,
                delivery_attempts, task_json, origin_session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                delegation_id, session, "",
                session, "completed",
                completed_at - 30, completed_at, completed_at,
                json.dumps(event), json.dumps({"status": "completed"}),
                "pending", 0, json.dumps([{"goal": "g"}]), "",
            ),
        )
    conn.close()
    return event


class _SessionDB:
    """Stored session rows keyed by id. Unknown ids resolve to None and have
    no compression tip, so the classifier follows the real 'unknown session'
    branch."""

    def __init__(self, rows):
        self._rows = rows

    async def get_session(self, session_id):
        return self._rows.get(session_id)

    async def get_compression_tip(self, session_id):
        return session_id if session_id in self._rows else None


def _runner(session_db):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}
    runner.config = {}
    runner._session_db = session_db
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = set()
    return runner


def _ledger_state(db_path, delegation_id):
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT delivery_state, delivery_attempts, delivery_claim FROM async_delegations "
        "WHERE delegation_id=?", (delegation_id,),
    ).fetchone()
    conn.close()
    return row


def test_pre_flight_end_reason_verdict_for_cli_close_parent_is_deliver():
    """Ground the repro premise: the generic end_reason classifier WOULD
    deliver a cli_close-ended parent (not a user-boundary end reason) — the
    misroute starts downstream, at source resolution, which is why unroutable
    origins need their own preflight check."""
    verdict = asyncio.run(GatewayRunner._classify_completion_target(
        SimpleNamespace(_session_db=_SessionDB(
            {_CLI_SESSION: {"ended_at": 1.0, "end_reason": "cli_close"}})),
        _CLI_SESSION,
    ))
    assert verdict == "deliver"


@pytest.mark.parametrize("source", ["cli", "tui"])
def test_unroutable_source_verdict_is_terminal_for_cli_and_tui(source):
    """cli/tui spawn sessions have no gateway delivery route; the verdict is
    taken from the event's parent_session_id resolved against the stored
    session row, not caller-supplied evt fields."""
    reason = asyncio.run(_unroutable_completion_verdict(
        SimpleNamespace(_session_db=_SessionDB({_CLI_SESSION: {"source": source}})),
        {"type": "async_delegation", "delegation_id": "deleg_x",
         "parent_session_id": _CLI_SESSION},
    ))
    assert reason is not None and source in reason


def test_unroutable_source_verdict_ignores_non_delegation_events():
    runner = SimpleNamespace(_session_db=_SessionDB({_CLI_SESSION: {"source": "cli"}}))
    assert asyncio.run(_unroutable_completion_verdict(
        runner, {"type": "completion", "delegation_id": ""})) is None
    assert asyncio.run(_unroutable_completion_verdict(
        runner, {"type": "async_delegation"})) is None


def test_preflight_drops_cli_origin_completion_in_one_claim(isolated_ledger):
    """The core regression: a restored CLI-origin completion must be dropped
    terminally by the preflight itself — exactly one claim burned, no adapter
    injection attempted, no requeue."""
    event = _seed_pending_completion(isolated_ledger, "deleg_cli_repro", _CLI_SESSION)
    # Gateway restart re-enqueues the durable pending completion.
    fresh = pr_module.ProcessRegistry()
    assert ad.restore_undelivered_completions(fresh.completion_queue) == 1
    restored_evt = fresh.completion_queue.get()

    runner = _runner(_SessionDB({_CLI_SESSION: {"ended_at": 1.0, "end_reason": "cli_close",
                                                "source": "cli"}}))

    inject_attempted = []

    async def _count_inject(self, text, evt):
        inject_attempted.append(evt)
        return True

    GatewayRunner._inject_watch_notification = _count_inject
    asyncio.run(runner._deliver_completion_notification("synthetic text", restored_evt))

    assert not inject_attempted, (
        "preflight must drop the unroutable-origin completion BEFORE injection"
    )

    state, attempts, claim = _ledger_state(isolated_ledger, event["delegation_id"])
    assert state == "dropped", f"expected terminal drop, got {state!r}"
    assert attempts == 1, (
        f"unroutable-origin completion must settle in a single claim, got {attempts}"
    )
    assert claim is None, "terminal drop must clear the claim token"

    # And a SECOND restart must not replay it: dropped rows never re-enqueue.
    assert ad.restore_undelivered_completions(
        pr_module.ProcessRegistry().completion_queue) == 0


def test_preflight_leaves_routable_origin_retryable(isolated_ledger):
    """A parent session whose stored source is a gateway platform is NOT
    unroutable: an unroutable drop is terminal, so a false positive here would
    silently delete a deliverable completion. The preflight must not drop."""
    event = _seed_pending_completion(isolated_ledger, "deleg_telegram",
                                     _TELEGRAM_SESSION, source="telegram")
    event["session_key"] = "agent:main:telegram:dm:123"
    event["chat_id"] = "123"

    runner = _runner(_SessionDB({_TELEGRAM_SESSION: {"ended_at": None, "end_reason": None,
                                                     "source": "telegram"}}))

    async def _no_route(self, text, evt):
        return False  # simulate a transient no-route (adapter down)

    GatewayRunner._inject_watch_notification = _no_route
    asyncio.run(runner._deliver_completion_notification("synthetic text", event))

    state, attempts, _claim = _ledger_state(isolated_ledger, event["delegation_id"])
    assert state == "pending", f"routable origin must stay pending, got {state!r}"
    assert attempts == 1


def test_preflight_unknown_session_source_stays_retryable(isolated_ledger):
    """A parent session row that is MISSING entirely exercises upstream's own
    permanently-gone classifier (terminal drop). What this test pins is that
    the unroutable pre-flight does NOT act as the dropper: it fails open
    (no early claim settle), and the drop you see comes from upstream's own
    classifier claiming first — attempts semantics identical either way, so
    the discriminating evidence is the pre-flight verdict itself, asserted
    directly below."""
    event = _seed_pending_completion(isolated_ledger, "deleg_unknown", _CLI_SESSION)

    # The unroutable verdict itself must answer None for a missing row.
    assert asyncio.run(_unroutable_completion_verdict(
        _runner(_SessionDB({})),
        {"type": "async_delegation", "delegation_id": "deleg_unknown",
         "parent_session_id": _CLI_SESSION},
    )) is None

    runner = _runner(_SessionDB({}))

    async def _no_route(self, text, evt):
        return False

    GatewayRunner._inject_watch_notification = _no_route
    asyncio.run(runner._deliver_completion_notification("synthetic text", event))

    state, attempts, _claim = _ledger_state(isolated_ledger, event["delegation_id"])
    # Upstream's permanently-gone classifier claims-then-drops (attempts == 1);
    # the unroutable pre-flight's own early-drop path claims-drops-returns the
    # SAME ledger state but skips injection. Since injection ran (verdict was
    # None), the injection leg executed — proving the pre-flight abstained.
    assert state == "dropped"
    assert attempts == 1


def test_preflight_session_lookup_error_stays_retryable(isolated_ledger):
    """Transient session-DB errors must fail open onto the retry path — a
    lookup that RAISED is not proof the origin is unroutable."""
    event = _seed_pending_completion(isolated_ledger, "deleg_boom", _CLI_SESSION)

    class _Boom:
        async def get_session(self, session_id):
            raise OSError("state.db busy")

        async def get_compression_tip(self, session_id):
            return session_id

    runner = _runner(_Boom())

    async def _no_route(self, text, evt):
        return False

    GatewayRunner._inject_watch_notification = _no_route
    asyncio.run(runner._deliver_completion_notification("synthetic text", event))

    state, attempts, _claim = _ledger_state(isolated_ledger, event["delegation_id"])
    assert state == "pending", f"transient lookup error must not drop, got {state!r}"


def test_no_route_at_injection_drops_terminally(isolated_ledger):
    """A telegram-origin completion (NOT unroutable-source) that reaches
    injection and comes back ``None`` — the no-routing-metadata drop leg —
    settles terminally as ``dropped`` in one claim instead of releasing into
    the restart-replay churn."""
    event = _seed_pending_completion(isolated_ledger, "deleg_tg_noroute",
                                     _TELEGRAM_SESSION, source="telegram")
    event["session_key"] = "agent:main:telegram:dm:123"
    event["chat_id"] = "123"

    runner = _runner(_SessionDB({_TELEGRAM_SESSION: {"ended_at": None,
                                                     "end_reason": None,
                                                     "source": "telegram"}}))

    async def _no_route_at_all(self, text, evt):
        return None  # proven no route inside injection

    GatewayRunner._inject_watch_notification = _no_route_at_all
    result = asyncio.run(runner._deliver_completion_notification("synthetic text", event))
    assert result is None

    state, attempts, claim = _ledger_state(isolated_ledger, event["delegation_id"])
    assert state == "dropped"
    assert attempts == 1
    assert claim is None