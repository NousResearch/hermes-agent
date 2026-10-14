"""Pending-setup watchdog for the pre-agent turn phase (#138358).

A turn wedged in session-resolve / turn-prepare / agent:start hooks holds the
_AGET_PENDING_SENTINEL with no timeout; the stale-evictor explicitly never
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


def test_stale_pending_sentinel_is_evicted(monkeypatch):
    gateway = _build_gateway(monkeypatch)
    gateway._hm_evict_idle_stale_agent(KEY)
    assert gateway._peek_session_state(KEY).turn.agent is None
