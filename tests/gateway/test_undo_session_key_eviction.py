""" /undo must evict the agent cache under the real routing session key.

Bare ``build_session_key(source)`` ignores gateway config
(``group_sessions_per_user``, multiplex profile namespace, etc.). When that
diverges from ``_session_key_for_source`` / SessionStore, /undo would leave
the live cache entry in place and the next turn would reuse a stale agent.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, build_session_key


def _group_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="g1",
        user_name="alice",
        chat_type="group",
    )


def _make_event() -> MessageEvent:
    return MessageEvent(text="/undo", source=_group_source(), message_id="m1")


def _make_runner(*, group_sessions_per_user: bool) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        group_sessions_per_user=group_sessions_per_user,
    )
    # Bind the real resolver so config kwargs are honored (no session_store
    # short-circuit path needed — fall through to build_session_key with config).
    runner.session_store = None
    runner._session_key_for_source = GatewayRunner._session_key_for_source.__get__(
        runner, GatewayRunner
    )

    source = _group_source()
    real_key = runner._session_key_for_source(source)
    session_entry = SessionEntry(
        session_key=real_key,
        session_id="sess-undo-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
        last_prompt_tokens=1234,
    )

    # async_session_store is a property; install the mock behind the private
    # cache attribute and keep facade._store identical to session_store so the
    # property does not rebuild over None.
    store = MagicMock()
    store._store = None
    store.get_or_create_session = AsyncMock(return_value=session_entry)
    store.rewind_session = AsyncMock(
        return_value={
            "rewound_count": 2,
            "turns_undone": 1,
            "target_text": "previous user turn",
        }
    )
    runner._async_session_store = store
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._running_agents = {}
    runner._evict_cached_agent = MagicMock(
        wraps=GatewayRunner._evict_cached_agent.__get__(runner, GatewayRunner)
    )
    return runner


@pytest.mark.asyncio
async def test_undo_evicts_store_key_when_group_sessions_shared():
    """With group_sessions_per_user=False, bare build_session_key diverges."""
    runner = _make_runner(group_sessions_per_user=False)
    source = _group_source()
    bare_key = build_session_key(source)
    real_key = runner._session_key_for_source(source)
    assert bare_key != real_key

    stale_agent = MagicMock(name="stale-agent")
    runner._agent_cache[real_key] = (stale_agent, "sig")
    # Also plant a decoy under the bare key so a wrong eviction would "succeed"
    # at the mock call site while leaving the live entry.
    runner._agent_cache[bare_key] = (MagicMock(name="decoy"), "sig")

    result = await runner._handle_undo_command(_make_event())

    assert isinstance(result, str) and result
    runner._evict_cached_agent.assert_called_once_with(real_key)
    # Live cache entry under the routing key must be gone; the bare-key decoy
    # must remain (proves we did not only evict the wrong key).
    assert real_key not in runner._agent_cache
    assert bare_key in runner._agent_cache
