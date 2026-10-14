"""Pending-setup watchdog for the pre-agent turn phase (#138358).

A turn wedged in session-resolve / turn-prepare / agent:start hooks holds the
_AGENT_PENDING_SENTINEL with no timeout; the stale-evictor explicitly never
evicts it, so every later message queues silently forever.
"""
from __future__ import annotations

import time

from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL

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
    state.turn.event = None
    return gateway


def test_stale_pending_sentinel_is_evicted(monkeypatch, caplog):
    gateway = _build_gateway(monkeypatch)
    gateway._peek_session_state(KEY).turn.stage = "prepare_turn"
    with caplog.at_level("WARNING", logger="gateway.run_inbound"):
        gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is None
    assert "Evicting wedged pre-agent turn" in caplog.text
    assert "prepare_turn" in caplog.text


def test_fresh_pending_sentinel_is_kept(monkeypatch):
    gateway = _build_gateway(monkeypatch)
    gateway._peek_session_state(KEY).turn.started_ts = time.time() - 5
    gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is _AGENT_PENDING_SENTINEL


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
