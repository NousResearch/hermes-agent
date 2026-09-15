"""Both extracted manager factories preserve context and stay off the event loop."""

import asyncio
from contextvars import ContextVar
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import GatewayRunner
from gateway.run_goals import GatewayGoalsMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("post_turn", [False, True])
async def test_manager_factory_runs_off_loop_with_profile_context(post_turn, monkeypatch):
    profile = ContextVar("test_manager_profile")
    token = profile.set("isolated-profile")
    loop_thread = threading.get_ident()
    entry = SimpleNamespace(session_id="test-session")
    manager = object()
    runner = GatewayRunner.__new__(GatewayRunner)
    monkeypatch.setattr(runner, "_session_entry_for_manager", AsyncMock(return_value=entry))
    monkeypatch.setattr(runner, "_warm_goals_session_db", AsyncMock())

    def factory(sid):
        assert threading.get_ident() != loop_thread
        assert profile.get() == "isolated-profile"
        assert sid == entry.session_id
        return manager

    try:
        if post_turn:
            result = await GatewayGoalsMixin._post_turn_manager(
                runner, entry, "goal continuation", "goals", lambda: factory,
            )
            assert result is manager
        else:
            result = await GatewayGoalsMixin._manager_for_event(
                runner, object(), "goal", lambda: factory,
            )
            assert result == (manager, entry)
    finally:
        profile.reset(token)
        if getattr(runner, "_executor", None) is not None:
            await asyncio.to_thread(runner._executor.shutdown, wait=True)
