"""Pending-setup watchdog for the pre-agent turn phase.

Regression for #138358: a turn wedged in session-resolve / turn-prepare /
agent:start hooks holds the _AGENT_PENDING_SENTINEL with no timeout, so every
later message queues silently forever.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session_state import (
    TURN_STAGE_CLAIMED,
    TURN_STAGE_PREPARE_TURN,
    claim_turn_slot,
)

KEY = "agent:main:telegram:dm:138358"


def _build_gateway(monkeypatch):
    gateway = object.__new__(GatewayRunner)
    gateway._persist_active_agents = lambda: None
    gateway._agent_cache_lock = None
    gateway._agent_cache = {}
    monkeypatch.setenv("HERMES_GATEWAY_PENDING_SETUP_TIMEOUT", "600")
    state = gateway._session_state(KEY)
    state.turn.agent = _AGENT_PENDING_SENTINEL
    state.turn.started_ts = time.time() - 3600
    state.turn.event = SimpleNamespace(text="stuck hello", message_id=7)
    state.turn.stage = TURN_STAGE_PREPARE_TURN
    return gateway


def test_claim_turn_slot_sets_triple():
    from gateway.session_state import TurnState

    turn = TurnState()
    before = time.time()
    claim_turn_slot(turn, event=None, sentinel=_AGENT_PENDING_SENTINEL)
    assert turn.agent is _AGENT_PENDING_SENTINEL
    assert turn.event is None
    assert turn.started_ts >= before
    assert turn.stage == TURN_STAGE_CLAIMED


def test_stale_pending_sentinel_is_evicted(monkeypatch, caplog):
    gateway = _build_gateway(monkeypatch)
    with caplog.at_level("WARNING", logger="gateway.run_inbound"):
        gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is None
    assert "Evicting wedged pre-agent turn" in caplog.text
    assert TURN_STAGE_PREPARE_TURN in caplog.text
    assert "stuck hello" in caplog.text


def test_evict_stages_resend_notice(monkeypatch):
    gateway = _build_gateway(monkeypatch)
    gateway._hm_evict_idle_stale_agent(KEY)
    notes = gateway._consume_pending_turn_sidecar_notes(KEY)
    assert len(notes) == 1
    assert "please resend it" in notes[0]
    assert TURN_STAGE_PREPARE_TURN in notes[0]
    # One-shot: consumed.
    assert gateway._consume_pending_turn_sidecar_notes(KEY) == []


def test_append_sidecar_notes_merges():
    gateway = object.__new__(GatewayRunner)
    gateway._session_state(KEY)
    gateway._set_pending_turn_sidecar_notes(KEY, ["first"])
    gateway._append_turn_sidecar_notes(KEY, ["second"])
    assert gateway._consume_pending_turn_sidecar_notes(KEY) == ["first", "second"]


def test_fresh_pending_sentinel_is_kept(monkeypatch):
    gateway = _build_gateway(monkeypatch)
    gateway._peek_session_state(KEY).turn.started_ts = time.time() - 5
    gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is _AGENT_PENDING_SENTINEL
    assert gateway._consume_pending_turn_sidecar_notes(KEY) == []


def test_pending_evict_disabled_at_zero(monkeypatch):
    gateway = _build_gateway(monkeypatch)
    monkeypatch.setenv("HERMES_GATEWAY_PENDING_SETUP_TIMEOUT", "0")
    gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is _AGENT_PENDING_SENTINEL


def test_oldest_turn_age_covers_pending_sentinel():
    gateway = object.__new__(GatewayRunner)
    assert gateway._oldest_turn_age_s() is None
    now = time.time()
    stale = gateway._session_state("k-stale")
    stale.turn.agent = _AGENT_PENDING_SENTINEL
    stale.turn.started_ts = now - 500
    live = gateway._session_state("k-live")
    live.turn.agent = object()
    live.turn.started_ts = now - 50
    age = gateway._oldest_turn_age_s(now=now)
    assert age is not None and 499 <= age <= 501


def test_status_update_carries_oldest_turn_age():
    from gateway import status as status_mod

    _, payload, _ = status_mod._prepare_runtime_status_update(
        oldest_turn_age_s=12.5, load_existing=False)
    assert payload["oldest_turn_age_s"] == 12.5
    assert status_mod._build_runtime_status_record()["oldest_turn_age_s"] is None


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self.sent = []

    async def send(self, chat_id, content, metadata=None):
        self.sent.append(
            {"chat_id": chat_id, "content": content, "metadata": metadata}
        )


@pytest.mark.asyncio
async def test_watcher_evicts_stale_sentinel_without_followup(monkeypatch):
    from gateway.session import SessionSource
    from gateway.config import Platform

    adapter = _FakeAdapter()
    gateway = object.__new__(GatewayRunner)
    gateway._persist_active_agents = lambda: None
    gateway._agent_cache_lock = None
    gateway._agent_cache = {}
    gateway.adapters = {"fake": adapter}
    gateway._profile_adapters = {}
    gateway._running_agents = {}
    monkeypatch.setenv("HERMES_GATEWAY_PENDING_SETUP_TIMEOUT", "600")
    source = SessionSource(chat_id="chat-1", thread_id=None, platform=Platform.TELEGRAM)
    adapter._pending_messages[KEY] = SimpleNamespace(
        text="follow-up", source=source, timestamp=time.time())
    gateway._running_agents[KEY] = _AGENT_PENDING_SENTINEL
    state = gateway._session_state(KEY)
    state.turn.agent = _AGENT_PENDING_SENTINEL
    state.turn.started_ts = time.time() - 3600
    state.turn.event = SimpleNamespace(text="stuck hello", message_id=7)

    sent = await gateway._check_session_stalls(60)
    assert sent == 0
    assert gateway._peek_session_state(KEY).turn.agent is None
    assert adapter.sent == []
