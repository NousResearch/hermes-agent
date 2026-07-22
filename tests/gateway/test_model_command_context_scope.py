"""Regression tests for context-length display during gateway /model switches."""

from __future__ import annotations

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    return runner


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.DISCORD, chat_id="12345", chat_type="thread", thread_id="12345"),
    )


def _codex_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.5",
        target_provider="openai-codex",
        provider_changed=True,
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        provider_label="OpenAI Codex",
        is_global=False,
    )


@pytest.mark.asyncio
async def test_session_model_switch_does_not_reuse_global_context_override(tmp_path, monkeypatch):
    """A session-only Codex switch must display Codex's real 272K cap.

    The profile can be globally configured for a different 1M-token model.
    Before the fix, the confirmation message reused that stale global
    ``model.context_length`` and claimed Codex gpt-5.5 had 1,000,000 tokens.
    """
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "glm-5.2",
                    "provider": "zai",
                    "context_length": 1_000_000,
                },
                "providers": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kw: _codex_switch_result())
    monkeypatch.setattr("hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **kw: None)

    result = await _make_runner()._handle_model_command(_make_event("/model gpt-5.5 --provider openai-codex"))

    assert result is not None
    assert "gpt-5.5" in result
    assert "272,000" in result
    assert "1,000,000" not in result
