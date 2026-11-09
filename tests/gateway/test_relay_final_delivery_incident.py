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
from types import SimpleNamespace

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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
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


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", ["idle_timeout", "timeout", None, ""])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("end_reason", ["session_reset", "user_exit", "session_switch"])
async def test_user_boundary_still_terminal(end_reason):
    """Explicit user boundaries remain terminal — /new means the user
    closed the thread of work on purpose."""
    runner = _classify_runner(
        {"ended_at": 1786288000.0, "end_reason": end_reason}
    )
    verdict = await runner._classify_completion_target("sess-reset")
    assert verdict == "terminal"


@pytest.mark.asyncio
async def test_unknown_session_still_terminal():
    runner = _classify_runner(None)
    assert await runner._classify_completion_target("gone") == "terminal"
