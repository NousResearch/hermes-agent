"""Changing verbosity must rebuild the gateway agent; unchanged config must reuse it."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner


@pytest.mark.parametrize("before,after", [("low", "high"), ("low", ""), ("", "medium")])
def test_verbosity_change_rebuilds_cached_agent(monkeypatch, before, after):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_db = None
    runner._refresh_fallback_model = lambda: None
    runner._apply_fallback_chain_to_agent = lambda *args: None
    runner._enforce_agent_cache_cap = lambda: None
    ctx = SimpleNamespace(
        user_config={"agent": {"text_verbosity": before}}, enabled_toolsets=[],
        source=SimpleNamespace(user_id="test-user", user_id_alt=None),
        session_key="test-session", session_id="test-id", _interrupt_depth=0,
    )
    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = ctx
    build = Mock(side_effect=lambda *args: SimpleNamespace())
    monkeypatch.setattr(turn, "_build_fresh_agent", build)
    route = {"model": "gpt-5.5", "runtime": {"provider": "copilot", "api_mode": "codex_responses"}}

    first, reused = turn._resolve_turn_agent(route, "telegram", "", 5, None, {})
    assert not reused
    same, reused = turn._resolve_turn_agent(route, "telegram", "", 5, None, {})
    assert reused and same is first
    ctx.user_config = {"agent": {"text_verbosity": after}}
    changed, reused = turn._resolve_turn_agent(route, "telegram", "", 5, None, {})
    assert not reused and changed is not first
    assert build.call_count == 2
    assert runner._agent_cache[ctx.session_key][0] is changed
