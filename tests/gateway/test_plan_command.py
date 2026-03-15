"""Tests for the /plan gateway slash command."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.skill_commands import scan_skill_commands
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:c1:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "planned",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )
    return runner


def _make_event(text="/plan"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="tester",
            chat_type="dm",
        ),
        message_id="m1",
    )


def _make_plan_skill(skills_dir):
    skill_dir = skills_dir / "plan"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: plan
description: Plan mode skill.
---

# Plan

Use the current conversation context when no explicit instruction is provided.
Save plans under $HERMES_HOME/plans.
"""
    )


class TestGatewayPlanCommand:
    @pytest.mark.asyncio
    async def test_plan_command_loads_skill_and_runs_agent(self, monkeypatch, tmp_path):
        import gateway.run as gateway_run

        runner = _make_runner()
        event = _make_event("/plan Add OAuth login")

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
        monkeypatch.setattr(
            "agent.model_metadata.get_model_context_length",
            lambda *_args, **_kwargs: 100_000,
        )

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_plan_skill(tmp_path)
            scan_skill_commands()
            result = await runner._handle_message(event)

        assert result == "planned"
        forwarded = runner._run_agent.call_args.kwargs["message"]
        assert "Plan mode skill" in forwarded
        assert "Add OAuth login" in forwarded
        assert str(tmp_path / "plans") in forwarded
        assert "Runtime note:" in forwarded

    @pytest.mark.asyncio
    async def test_plan_command_appears_in_help_output_via_skill_listing(self, tmp_path):
        runner = _make_runner()
        event = _make_event("/help")

        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            _make_plan_skill(tmp_path)
            scan_skill_commands()
            result = await runner._handle_help_command(event)

        assert "/plan" in result
