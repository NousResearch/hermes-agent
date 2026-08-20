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
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="12345",
            chat_type="thread",
            thread_id="12345",
        ),
    )


def _codex_switch_result():
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model="gpt-5.6-sol",
        target_provider="openai-codex",
        provider_changed=True,
        api_key="codex-token",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        provider_label="OpenAI Codex",
        is_global=False,
    )


def _write_pinned_profile(hermes_home) -> None:
    """Globally pin the profile to a 1M-token model on a different route."""
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {
                "default": "glm-5.2",
                "provider": "zai",
                "context_length": 1_000_000,
            },
            "providers": {},
        }),
        encoding="utf-8",
    )


def _apply_common_mocks(monkeypatch, hermes_home) -> None:
    """Point every hermes-home lookup at *hermes_home* and stub the switch."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model", lambda **kw: _codex_switch_result()
    )
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning", lambda *a, **kw: None
    )


def _read_profile(hermes_home) -> dict:
    return yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_session_model_switch_does_not_reuse_global_context_override(
    tmp_path, monkeypatch
):
    """A session-only Codex switch must display Codex's real 272K cap.

    The profile can be globally configured for a different 1M-token model.
    Before the fix, the confirmation message reused that stale global
    ``model.context_length`` and claimed Codex gpt-5.6-sol had 1,000,000 tokens.

    ``--session`` opts out of persistence, so the originally configured route
    must also survive the switch unchanged on disk.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_pinned_profile(hermes_home)
    _apply_common_mocks(monkeypatch, hermes_home)

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.6-sol --provider openai-codex --session")
    )

    assert result is not None
    assert "gpt-5.6-sol" in result
    assert "272,000" in result
    assert "1,000,000" not in result

    # Session-only: the configured route is left untouched on disk.
    persisted = _read_profile(hermes_home)["model"]
    assert persisted["default"] == "glm-5.2"
    assert persisted["provider"] == "zai"
    assert persisted["context_length"] == 1_000_000


@pytest.mark.asyncio
async def test_global_model_switch_drops_stale_context_pin(tmp_path, monkeypatch):
    """A persistent (--global) switch to a new route must not inherit the pin.

    hermes-sweeper (#48187): the original guard reloaded config after the
    switch had already written ``model.default``/``model.provider``, saw the
    new target route, and accepted the inherited 1M ``context_length`` for
    persistent switches. The confirmation must show Codex's real 272K cap
    (the session-only check uses pre-switch route state), and the persisted
    pin must be dropped because its route no longer matches — the global
    switch stays explicitly scoped instead of inheriting the stale override.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    _write_pinned_profile(hermes_home)
    _apply_common_mocks(monkeypatch, hermes_home)

    result = await _make_runner()._handle_model_command(
        _make_event("/model gpt-5.6-sol --provider openai-codex --global")
    )

    assert result is not None
    assert "gpt-5.6-sol" in result
    # Display resolves against the pre-switch route, so the stale 1M pin is cleared.
    assert "272,000" in result
    assert "1,000,000" not in result

    # Persisted route moved to Codex; the route-mismatched pin is dropped, not inherited.
    persisted = _read_profile(hermes_home)["model"]
    assert persisted["default"] == "gpt-5.6-sol"
    assert persisted["provider"] == "openai-codex"
    assert "context_length" not in persisted
