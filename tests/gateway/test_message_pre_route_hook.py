"""Tests for the message:pre_route hook block in gateway/run.py.

Rather than attempting to drive _handle_message_with_agent end-to-end (which
requires stubbing a large LLM + transcript stack), these tests isolate the
pre_route block by reproducing its exact logic in a thin async helper and
testing that helper against the same mocked objects that production code uses.

Covers:
- emit_collect called with the correct context keys
- switch_session called when hook returns decision=switch_session with different id
- switch_session NOT called when hook returns same session_id
- switch_session NOT called when hook returns None
- switch_session NOT called when hook returns {} (empty dict)
- exception in emit_collect is caught, logged, processing continues
- break fires after first switch_session decision; subsequent results are ignored
- non-dict results in the list are skipped gracefully
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource, build_session_key
from gateway.platforms.base import MessageEvent, MessageType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    user_id: str = "u1",
    chat_id: str = "c1",
    platform: Platform = Platform.TELEGRAM,
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
        chat_type=chat_type,
        thread_id=thread_id,
    )


def _make_event(text: str = "deploy to prod", source: SessionSource | None = None) -> MessageEvent:
    if source is None:
        source = _make_source()
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _make_session_entry(
    session_id: str = "sess-1",
    source: SessionSource | None = None,
) -> SessionEntry:
    if source is None:
        source = _make_source()
    return SessionEntry(
        session_key=build_session_key(source),
        session_id=session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


# ---------------------------------------------------------------------------
# Thin async replica of the pre_route block from _handle_message_with_agent.
#
# Mirrors gateway/run.py pre_route block exactly.  When that block is
# refactored the tests here will need a parallel update.
# ---------------------------------------------------------------------------

async def _run_pre_route_block(
    hooks,
    async_session_store,
    event: MessageEvent,
    source: SessionSource,
    session_entry: SessionEntry,
    evict_cached_agent=None,
) -> SessionEntry:
    """Execute only the message:pre_route hook dispatch block.

    Returns the (possibly switched) session_entry, mirroring the in-place
    reassignment that production code performs.

    ``evict_cached_agent`` is an optional callable(session_key) that mirrors
    the gateway's ``self._evict_cached_agent`` call.  Pass a MagicMock to
    assert it was called after a successful session swap.
    """
    session_key = session_entry.session_key

    _pre_route_ctx = {
        "platform": source.platform.value if hasattr(source.platform, "value") else str(source.platform),
        "user_id": str(source.user_id),
        "chat_id": str(source.chat_id),
        "thread_id": str(source.thread_id) if source.thread_id else None,
        "chat_type": source.chat_type or "",
        "session_id": session_entry.session_id,
        "session_key": session_key,
        "message": event.text or "",
    }
    try:
        _pre_route_results = await hooks.emit_collect("message:pre_route", _pre_route_ctx)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("message:pre_route hook emit failed", exc_info=True)
        _pre_route_results = []
    for _pr in _pre_route_results:
        if not isinstance(_pr, dict):
            continue
        if _pr.get("decision") == "switch_session" and _pr.get("session_id"):
            _target_id = str(_pr["session_id"])
            if _target_id != session_entry.session_id:
                _switched = await async_session_store.switch_session(
                    session_key, _target_id
                )
                if _switched:
                    session_entry = _switched
                    # Evict the cached agent so the next turn starts with a
                    # clean context bound to the new session_id — mirrors
                    # /resume (slash_commands.py:4430-4442).
                    if evict_cached_agent is not None:
                        evict_cached_agent(session_key)
            break

    return session_entry


# ---------------------------------------------------------------------------
# TestPreRouteHookEmit
# ---------------------------------------------------------------------------


class TestPreRouteHookEmit:
    """Tests for the message:pre_route hook invocation block."""

    @pytest.mark.asyncio
    async def test_emit_collect_called_with_correct_context_keys(self):
        """emit_collect must receive the expected context dictionary."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source(user_id="user42", chat_id="chat99", chat_type="group", thread_id="t1")
        event = _make_event(text="train a new model", source=source)
        entry = _make_session_entry(session_id="sess-1", source=source)

        await _run_pre_route_block(hooks, async_store, event, source, entry)

        hooks.emit_collect.assert_awaited_once()
        call_args = hooks.emit_collect.call_args
        event_name = call_args.args[0]
        ctx = call_args.args[1]

        assert event_name == "message:pre_route"
        assert ctx["platform"] == "telegram"
        assert ctx["user_id"] == "user42"
        assert ctx["chat_id"] == "chat99"
        assert ctx["thread_id"] == "t1"
        assert ctx["chat_type"] == "group"
        assert ctx["session_id"] == "sess-1"
        assert ctx["session_key"] == entry.session_key
        assert ctx["message"] == "train a new model"

    @pytest.mark.asyncio
    async def test_emit_collect_thread_id_none_when_source_has_no_thread(self):
        """thread_id must be None in context when source.thread_id is None/empty."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source(thread_id=None)
        event = _make_event(source=source)
        entry = _make_session_entry(source=source)

        await _run_pre_route_block(hooks, async_store, event, source, entry)

        ctx = hooks.emit_collect.call_args.args[1]
        assert ctx["thread_id"] is None

    @pytest.mark.asyncio
    async def test_switch_session_called_for_different_session_id(self):
        """When hook returns switch_session with a different id, switch_session is called."""
        target_entry = _make_session_entry(session_id="sess-target")
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[{"decision": "switch_session", "session_id": "sess-target"}]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock(return_value=target_entry)

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)
        event = _make_event(text="fix the bug", source=source)

        result = await _run_pre_route_block(hooks, async_store, event, source, entry)

        async_store.switch_session.assert_awaited_once_with(
            entry.session_key, "sess-target"
        )
        assert result.session_id == "sess-target"

    @pytest.mark.asyncio
    async def test_switch_session_not_called_when_hook_returns_same_session_id(self):
        """Hook returns switch_session with the SAME id → no actual switch call."""
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[{"decision": "switch_session", "session_id": "sess-current"}]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)
        event = _make_event(source=source)

        await _run_pre_route_block(hooks, async_store, event, source, entry)

        async_store.switch_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_switch_session_not_called_when_hook_returns_none(self):
        """None result in hook list → no switch_session call."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[None]))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-1", source=source)

        await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        async_store.switch_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_switch_session_not_called_when_hook_returns_empty_dict(self):
        """Empty dict result → no switch_session call (no 'decision' key)."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[{}]))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-1", source=source)

        await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        async_store.switch_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_exception_in_emit_collect_is_caught_and_processing_continues(self):
        """If emit_collect raises, the error is swallowed and switch_session is not called."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(side_effect=RuntimeError("network down")))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-1", source=source)

        # Must not raise
        result = await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        async_store.switch_session.assert_not_awaited()
        # Entry unchanged — no session switch happened
        assert result.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_break_fires_after_first_switch_session_decision(self):
        """Only the FIRST switch_session result acts; subsequent results are ignored."""
        target_entry = _make_session_entry(session_id="sess-target")
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[
                    {"decision": "switch_session", "session_id": "sess-target"},
                    {"decision": "switch_session", "session_id": "sess-other"},
                ]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock(return_value=target_entry)

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)

        await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        # Called exactly once, for "sess-target"
        assert async_store.switch_session.await_count == 1
        assert async_store.switch_session.call_args.args[1] == "sess-target"

    @pytest.mark.asyncio
    async def test_non_dict_results_skipped_gracefully(self):
        """Non-dict values in results list are skipped without error."""
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(return_value=["string", None, 42, True])
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-1", source=source)

        result = await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        async_store.switch_session.assert_not_awaited()
        assert result.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_switch_session_none_return_does_not_replace_entry(self):
        """If switch_session returns None (store rejected it), session_entry stays unchanged."""
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[{"decision": "switch_session", "session_id": "sess-target"}]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock(return_value=None)  # store rejected

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)

        result = await _run_pre_route_block(hooks, async_store, _make_event(source=source), source, entry)

        async_store.switch_session.assert_awaited_once()
        # Entry should remain the original because switch returned None
        assert result.session_id == "sess-current"

    @pytest.mark.asyncio
    async def test_agent_cache_evicted_after_successful_session_switch(self):
        """_evict_cached_agent is called with the session_key after a successful switch."""
        target_entry = _make_session_entry(session_id="sess-target")
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[{"decision": "switch_session", "session_id": "sess-target"}]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock(return_value=target_entry)
        evict_mock = MagicMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)

        await _run_pre_route_block(
            hooks, async_store, _make_event(source=source), source, entry,
            evict_cached_agent=evict_mock,
        )

        evict_mock.assert_called_once_with(entry.session_key)

    @pytest.mark.asyncio
    async def test_agent_cache_not_evicted_when_switch_session_returns_none(self):
        """_evict_cached_agent is NOT called when switch_session returns None (rejected)."""
        hooks = SimpleNamespace(
            emit_collect=AsyncMock(
                return_value=[{"decision": "switch_session", "session_id": "sess-target"}]
            )
        )
        async_store = MagicMock()
        async_store.switch_session = AsyncMock(return_value=None)
        evict_mock = MagicMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-current", source=source)

        await _run_pre_route_block(
            hooks, async_store, _make_event(source=source), source, entry,
            evict_cached_agent=evict_mock,
        )

        evict_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_agent_cache_not_evicted_when_hook_returns_none(self):
        """_evict_cached_agent is NOT called when hook returns None (no switch)."""
        hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[None]))
        async_store = MagicMock()
        async_store.switch_session = AsyncMock()
        evict_mock = MagicMock()

        source = _make_source()
        entry = _make_session_entry(session_id="sess-1", source=source)

        await _run_pre_route_block(
            hooks, async_store, _make_event(source=source), source, entry,
            evict_cached_agent=evict_mock,
        )

        evict_mock.assert_not_called()


# ---------------------------------------------------------------------------
# TestPreRouteTimeoutResolution
# ---------------------------------------------------------------------------


class TestPreRouteTimeoutResolution:
    """Tests for gateway.run._resolve_pre_route_timeout().

    The pre_route hook timeout is configurable via config.yaml
    ``hooks.pre_route_timeout`` (seconds), defaults to 30s, and is clamped
    to a 120s maximum so a wedged hook cannot stall the gateway loop.
    """

    def _resolve(self, hooks_cfg=None, *, load_config_raises: bool = False) -> float:
        """Call _resolve_pre_route_timeout with a monkeypatched load_config."""
        from unittest.mock import patch

        import gateway.run as gr

        # The resolver imports load_config from hermes_cli.config at call
        # time, so the patch must target the source module, not gateway.run.
        if load_config_raises:
            patcher = patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom"))
        else:
            patcher = patch("hermes_cli.config.load_config", return_value={"hooks": hooks_cfg} if hooks_cfg is not None else {})
        with patcher:
            return gr._resolve_pre_route_timeout()

    def test_default_when_config_absent(self):
        """No hooks config → 30s default."""
        assert self._resolve() == 30.0

    def test_custom_value_respected(self):
        """hooks.pre_route_timeout: 45 → 45s."""
        assert self._resolve({"pre_route_timeout": 45}) == 45.0

    def test_clamped_to_max_120(self):
        """Value above 120s is clamped to the 120s bound."""
        assert self._resolve({"pre_route_timeout": 999}) == 120.0

    def test_clamped_to_min_1(self):
        """Non-positive values are clamped to a 1s floor."""
        assert self._resolve({"pre_route_timeout": 0}) == 1.0
        assert self._resolve({"pre_route_timeout": -5}) == 1.0

    def test_invalid_value_falls_back_to_default(self):
        """Non-numeric value → 30s default."""
        assert self._resolve({"pre_route_timeout": "way-too-long"}) == 30.0

    def test_load_config_failure_falls_back_to_default(self):
        """Config load failure → 30s default, no crash."""
        assert self._resolve(load_config_raises=True) == 30.0
