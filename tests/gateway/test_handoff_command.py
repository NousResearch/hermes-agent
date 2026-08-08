from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _FakeSessionDB:
    def __init__(self, messages):
        self._messages = messages

    async def get_messages(self, session_id: str):
        return list(self._messages)


class _FakeAsyncSessionStore:
    def __init__(self, store, session_id: str = "sess-gw-1"):
        self._store = store
        self.get_or_create_session = AsyncMock(return_value=SimpleNamespace(session_id=session_id))


def _make_event(text: str = "/handoff inline review repo", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        thread_id=None,
    )
    return MessageEvent(text=text, source=source)


def _make_runner(messages=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = _FakeSessionDB(messages or [
        {"role": "user", "content": "Please review /srv/app and capture a clean handoff."},
        {"role": "assistant", "content": "I checked /srv/app/config.yaml and https://example.com/docs."},
    ])
    runner.session_store = SimpleNamespace()
    runner._async_session_store = _FakeAsyncSessionStore(runner.session_store)
    runner._resolve_profile_home_for_source = MagicMock(return_value=Path("/tmp/default-profile-home"))
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_run_generation = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.emit_collect = AsyncMock(return_value=[])
    runner.delivery_router = MagicMock()
    runner._check_slash_access = lambda source, canonical: None
    runner._claim_active_session_slot = lambda *_a, **_k: (None, None)
    runner._begin_session_run_generation = lambda *_a, **_k: 1
    runner._release_running_agent_state = lambda *_a, **_k: None
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    return runner


class TestGatewayHandoffCommand:
    @pytest.mark.asyncio
    async def test_inline_returns_markdown(self):
        runner = _make_runner()
        event = _make_event("/handoff inline review repo")

        result = await runner._handle_handoff_command(event)

        assert getattr(result, "agent_seed", None) is None
        assert "# Handoff:" in result.text
        assert "Suggested filename:" in result.text

    @pytest.mark.asyncio
    async def test_consume_returns_agent_seed(self, monkeypatch, tmp_path):
        handoff = tmp_path / "gateway-handoff.md"
        handoff.write_text(
            "# Handoff: repo review\n\n"
            "## Purpose of next session\nReview the repo.\n\n"
            "## Current status\n- scope defined\n\n"
            "## Relevant artifacts\n- workdir: /srv/app\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nRead the key files and continue the review.\n\n"
            "## Success criteria\n- [ ] review completed\n",
            encoding="utf-8",
        )
        runner = _make_runner()
        runner._resolve_profile_home_for_source.return_value = tmp_path / ".hermes-profile"
        event = _make_event(f"/handoff consume {handoff}")

        result = await runner._handle_handoff_command(event)

        assert result.agent_seed is not None
        assert str(handoff) in result.agent_seed
        assert "Loaded handoff from:" in result.text
        runner._resolve_profile_home_for_source.assert_called_once_with(event.source)

    @pytest.mark.asyncio
    async def test_gateway_handoff_uses_async_session_store(self, tmp_path):
        handoff = tmp_path / "gateway-handoff.md"
        handoff.write_text(
            "# Handoff: repo review\n\n"
            "## Purpose of next session\nReview the repo.\n\n"
            "## Current status\n- scope defined\n\n"
            "## Relevant artifacts\n- workdir: /srv/app\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nRead the key files and continue the review.\n\n"
            "## Success criteria\n- [ ] review completed\n",
            encoding="utf-8",
        )
        runner = _make_runner()
        event = _make_event(f"/handoff consume {handoff}")

        await runner._handle_handoff_command(event)

        runner.async_session_store.get_or_create_session.assert_awaited_once_with(event.source)

    @pytest.mark.asyncio
    async def test_platform_arg_on_gateway_returns_guidance(self):
        runner = _make_runner()
        event = _make_event("/handoff telegram", platform=Platform.DISCORD)

        result = await runner._handle_handoff_command(event)

        assert isinstance(result, str)
        assert "CLI-only" in result

    @pytest.mark.asyncio
    async def test_help_lists_handoff(self):
        runner = _make_runner()
        event = _make_event("/help")

        result = await runner._handle_help_command(event)

        assert "/handoff" in result

    @pytest.mark.asyncio
    async def test_pasted_handoff_auto_consumes_in_gateway_message_flow(self):
        runner = _make_runner()
        pasted = (
            "# Handoff: repo review\n\n"
            "## Purpose of next session\nReview the repo.\n\n"
            "## Current status\n- scope defined\n\n"
            "## Relevant artifacts\n- workdir: /srv/app\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nRead the key files and continue the review.\n\n"
            "## Success criteria\n- [ ] review completed\n"
        )
        event = _make_event(pasted)

        async def _fake_agent(self_runner, ev, src, key, generation):
            assert "Handoff payload:" in ev.text
            assert "<pasted handoff from gateway>" in ev.text
            return "ok"

        with patch("gateway.run.GatewayRunner._handle_message_with_agent", _fake_agent):
            result = await runner._handle_message(event)

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_oversized_pasted_handoff_is_rejected_before_agent(self):
        from hermes_cli.handoff_doc_cmd import _MAX_HANDOFF_BYTES

        runner = _make_runner()
        adapter = SimpleNamespace(send=AsyncMock())
        runner.adapters = {Platform.TELEGRAM: adapter}
        oversized = (
            "# Handoff: repo review\n\n"
            "## Purpose of next session\n" + ("A" * (_MAX_HANDOFF_BYTES + 128)) + "\n\n"
            "## Current status\n- scope defined\n\n"
            "## Relevant artifacts\n- workdir: /srv/app\n\n"
            "## Constraints and non-goals\n- stay narrow\n\n"
            "## Exact first prompt\nRead the key files and continue the review.\n\n"
            "## Success criteria\n- [ ] review completed\n"
        )
        event = _make_event(oversized)

        with patch("gateway.run.GatewayRunner._handle_message_with_agent", AsyncMock(return_value="should-not-run")) as agent_mock:
            result = await runner._handle_message(event)

        assert result is None
        agent_mock.assert_not_awaited()
        adapter.send.assert_awaited_once()
        sent_text = adapter.send.await_args.args[1]
        assert "Handoff too large to consume" in sent_text
