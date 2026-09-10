"""Regression: relay-plane delivery defects (staging incident 2026-08-09).

Defect A — the "skip redundant final edit" branch records ``self._accumulated``
as the delivered turn-final payload even when the last edit that actually
reached the platform was an earlier throttled preview snapshot. The recorded
payload then satisfies ``delivered_final_matches`` and the gateway suppresses
the corrective final send, leaving the user staring at a frozen preview ending
in the streaming cursor.

Contract under test (end-to-end): the payload recorded as *delivered* must be
what was last acknowledged on the wire, so a preview/final mismatch yields
``delivered_final_matches(final) is False`` and the normal final send fires.

Defect B — ``_classify_completion_target`` treats every ended parent session
as terminal unless it ended by compression. Relay-plane sessions end on idle
by design (scale-to-zero); the chat route remains valid, so async delegation
completions must classify "deliver", not be terminally dropped. Explicit user
boundaries (/new -> session_reset / user_exit) stay terminal.
"""

import asyncio
import concurrent.futures
import threading
from collections import OrderedDict

import pytest

from gateway.run import GatewayRunner
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


# ---------------------------------------------------------------------------
# Defect A: stale preview recorded as delivered final
# ---------------------------------------------------------------------------

class _EditAdapter:
    """Adapter stub: every send/edit succeeds and remembers the last payload."""

    name = "stub"

    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, content, **kw):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id="m1")

    async def edit_message(self, chat_id, message_id, content, **kw):
        self.sent.append(content)
        return SimpleNamespace(success=True, message_id=message_id)


def _consumer(adapter):
    cfg = StreamConsumerConfig(edit_interval=0.01)
    return GatewayStreamConsumer(adapter, "C1", config=cfg)


@ pytest.mark.asyncio
async def test_skip_redundant_finalize_records_acked_payload_not_accumulated():
    """The delivered record must reflect the last ACKED edit, so a stale
    preview cannot masquerade as the delivered final (incident class:
    'It launched but ▉')."""
    adapter = _EditAdapter()
    sc = _consumer(adapter)

    preview = "It launched but"
    final = (
        "It launched but the worker had no credentials, so the check did "
        "not run. The delegates completed; results follow."
    )

    # Simulate: a mid-stream edit delivered the throttled preview snapshot,
    # then the turn finished with more content accumulated and the consumer
    # took the skip-redundant-finalize branch (no further edit issued).
    sc._message_id = "m1"
    sc._last_sent_text = preview + sc.cfg.cursor
    sc._accumulated = final
    sc._mark_skip_redundant_finalize()

    verdict = sc.delivered_final_matches(final)
    assert verdict is False, (
        "stale preview snapshot must NOT be reconciled as the delivered "
        f"final (got verdict={verdict!r}); the normal final send would be "
        "suppressed and the user left with a frozen preview"
    )


@ pytest.mark.asyncio
async def test_finalize_edit_success_still_reconciles_true():
    """Control: when the finalize edit actually delivered the full final
    text, reconciliation must remain True (no dup sends regression)."""
    adapter = _EditAdapter()
    sc = _consumer(adapter)
    final = "Complete final answer."
    sc._message_id = "m1"
    sc._last_sent_text = final
    sc._accumulated = final
    sc._mark_skip_redundant_finalize()
    assert sc.delivered_final_matches(final) is True


# ---------------------------------------------------------------------------
# Defect B: idle-ended relay session terminally drops completions
# ---------------------------------------------------------------------------

class _SessionDB:
    def __init__(self, row):
        self._row = row

    async def get_session(self, session_id):
        return self._row

    async def get_compression_tip(self, session_id):
        return None


def _classify_runner(row):
    runner = object.__new__(GatewayRunner)
    runner._session_db = _SessionDB(row)
    return runner


@ pytest.mark.asyncio
@ pytest.mark.parametrize("end_reason", ["idle_timeout", "timeout", None, ""])
async def test_idle_ended_parent_classifies_deliver(end_reason):
    """Relay-plane norm: session ended on idle, chat still routable ->
    the completion must be deliverable, not terminally dropped."""
    runner = _classify_runner(
        {"ended_at": 1786288000.0, "end_reason": end_reason}
    )
    verdict = await runner._classify_completion_target("sess-idle")
    assert verdict == "deliver", (
        f"end_reason={end_reason!r} must classify 'deliver' "
        f"(got {verdict!r}); completed delegation work was dropped in "
        "staging because idle-ended sessions classified terminal"
    )


@ pytest.mark.asyncio
@ pytest.mark.parametrize("end_reason", ["session_reset", "user_exit", "session_switch"])
async def test_user_boundary_still_terminal(end_reason):
    """Explicit user boundaries remain terminal — /new means the user
    closed the thread of work on purpose."""
    runner = _classify_runner(
        {"ended_at": 1786288000.0, "end_reason": end_reason}
    )
    verdict = await runner._classify_completion_target("sess-reset")
    assert verdict == "terminal"


@ pytest.mark.asyncio
async def test_unknown_session_still_terminal():
    runner = _classify_runner(None)
    assert await runner._classify_completion_target("gone") == "terminal"


# ---------------------------------------------------------------------------
# Issue #82703: gateway should not claim async_delegation/completion
# when it has no adapter route. Claiming burns a delivery attempt;
# if no route exists (e.g. raw CLI session with no api_server adapter),
# the row should stay pending for a CLI/TUI/api_server consumer.
# ---------------------------------------------------------------------------

def _make_no_route_runner():
    """Runner with no adapters and no session store entries -> no route."""
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner._draining = False
    runner._restart_requested = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._stop_task = None
    runner._exit_cleanly = False
    runner._exit_with_failure = False
    runner._exit_reason = None
    runner._exit_code = None
    runner._restart_drain_timeout = 0.01
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    runner.adapters = {}  # NO adapters -> no route
    runner._background_tasks = set()
    runner._failed_platforms = []
    runner._shutdown_event = asyncio.Event()
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._busy_ack_ts = {}
    runner._executor_lock = threading.Lock()
    runner._executor_closing = False
    runner._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    runner._session_db = None
    runner.session_store = type("Store", (), {"_entries": {}})()
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 100
    runner._completion_notification_batch_window = 0.01
    runner._completion_notification_batches = {}
    runner._completion_notification_batch_tasks = {}
    return runner


@ pytest.mark.asyncio
async def test_async_delegation_no_route_does_not_claim():
    """Gateway with no route should return None without claiming the durable row."""
    from tools.async_delegation import claim_completion_delivery

    runner = _make_no_route_runner()
    calls = []

    original_claim = claim_completion_delivery

    def tracked_claim(delegation_id, claim_id):
        calls.append((delegation_id, claim_id))
        return original_claim(delegation_id, claim_id)

    import tools.async_delegation as ad_mod
    ad_mod.claim_completion_delivery = tracked_claim

    try:
        evt = {
            "type": "async_delegation",
            "delegation_id": "test-delegation-123",
            "session_key": "raw-session-no-route",  # not in session_store
            "platform": "",
            "chat_type": "",
            "chat_id": "",
        }
        result = await runner._deliver_completion_notification("test output", evt)
        assert result is None, f"expected None (no route), got {result}"
        assert calls == [], f"claim_completion_delivery was called: {calls}"
    finally:
        ad_mod.claim_completion_delivery = original_claim


@ pytest.mark.asyncio
async def test_completion_no_route_does_not_claim():
    """Completion events with no route should also not be claimed."""
    from tools.async_delegation import claim_completion_delivery

    runner = _make_no_route_runner()
    calls = []

    original_claim = claim_completion_delivery

    def tracked_claim(delegation_id, claim_id):
        calls.append((delegation_id, claim_id))
        return original_claim(delegation_id, claim_id)

    import tools.async_delegation as ad_mod
    ad_mod.claim_completion_delivery = tracked_claim

    try:
        evt = {
            "type": "completion",
            "delegation_id": "",
            "session_id": "raw-session-no-route",
            "session_key": "raw-session-no-route",
            "platform": "",
            "chat_type": "",
            "chat_id": "",
        }
        result = await runner._deliver_completion_notification("test output", evt)
        assert result is None, f"expected None (no route), got {result}"
        assert calls == [], f"claim_completion_delivery was called: {calls}"
    finally:
        ad_mod.claim_completion_delivery = original_claim


# Need imports for the new tests
import asyncio
import concurrent.futures
import threading
from collections import OrderedDict
from types import SimpleNamespace