from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class _FakeSessionDB:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self, session_id: str):
        return list(self._messages)


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
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="sess-gw-1")
    )
    runner._session_db = _FakeSessionDB(messages or [
        {"role": "user", "content": "Please review /srv/app and capture a clean handoff."},
        {"role": "assistant", "content": "I checked /srv/app/config.yaml and https://example.com/docs."},
    ])
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
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path / ".hermes")
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

        result = await runner._handle_handoff_command(event)

        assert result.agent_seed is not None
        assert str(handoff) in result.agent_seed
        assert "Loaded handoff from:" in result.text

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
