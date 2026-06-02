"""Tests that gateway /model switch persists across messages.

The gateway /model command stores session overrides in
``_session_model_overrides``.  These must:

1. Be applied in ``run_sync()`` so the next agent uses the switched model.
2. Not be mistaken for fallback activation (which evicts the cached agent).
3. Survive across multiple messages until /reset clears them.

Tests exercise the real ``_apply_session_model_override()`` and
``_is_intentional_model_switch()`` methods on ``GatewayRunner``.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    """Create a minimal GatewayRunner with stubbed internals."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._effective_model = None
    runner._effective_provider = None
    runner.session_store = MagicMock()
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    return runner


# ---------------------------------------------------------------------------
# Tests: _apply_session_model_override
# ---------------------------------------------------------------------------


class TestApplySessionModelOverride:
    """Verify _apply_session_model_override replaces config defaults."""

    def test_override_replaces_all_fields(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4-turbo",
            "provider": "openrouter",
            "api_key": "or-key-123",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4-turbo"
        assert rt["provider"] == "openrouter"
        assert rt["api_key"] == "or-key-123"
        assert rt["base_url"] == "https://openrouter.ai/api/v1"
        assert rt["api_mode"] == "chat_completions"

    def test_no_override_returns_originals(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        orig_model = "anthropic/claude-sonnet-4"
        orig_rt = {"provider": "anthropic", "api_key": "key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"}

        model, rt = runner._apply_session_model_override(sk, orig_model, dict(orig_rt))

        assert model == orig_model
        assert rt == orig_rt

    def test_none_values_do_not_overwrite(self):
        """Override with None api_key/base_url should preserve config defaults."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": None,
            "base_url": None,
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert model == "gpt-5.4"
        assert rt["provider"] == "openai"
        assert rt["api_key"] == "ant-key"  # preserved — None didn't overwrite
        assert rt["base_url"] == "https://api.anthropic.com"  # preserved
        assert rt["api_mode"] == "chat_completions"  # overwritten (not None)

    def test_empty_string_overwrites(self):
        """Empty string is not None — it should overwrite the config value."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "local-model",
            "provider": "custom",
            "api_key": "local-key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        _, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "https://api.anthropic.com", "api_mode": "anthropic_messages"},
        )

        assert rt["base_url"] == ""  # empty string overwrites

    def test_different_session_key_not_affected(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        other_sk = "other_session"

        runner._session_model_overrides[other_sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        model, rt = runner._apply_session_model_override(
            sk,
            "anthropic/claude-sonnet-4",
            {"provider": "anthropic", "api_key": "ant-key", "base_url": "url", "api_mode": "anthropic_messages"},
        )

        assert model == "anthropic/claude-sonnet-4"  # unchanged — wrong session key


# ---------------------------------------------------------------------------
# Tests: _is_intentional_model_switch
# ---------------------------------------------------------------------------


class TestIsIntentionalModelSwitch:
    """Verify fallback detection respects intentional /model overrides."""

    def test_matches_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is True

    def test_no_override_returns_false(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False

    def test_different_model_returns_false(self):
        """Agent fell back to a different model than the override."""
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides[sk] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4-mini") is False

    def test_wrong_session_key(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())

        runner._session_model_overrides["other_session"] = {
            "model": "gpt-5.4",
            "provider": "openai",
            "api_key": "key",
            "base_url": "",
            "api_mode": "chat_completions",
        }

        assert runner._is_intentional_model_switch(sk, "gpt-5.4") is False


@pytest.mark.asyncio
async def test_model_picker_refuses_switch_while_agent_running(monkeypatch):
    """Interactive gateway pickers must match typed /model's mid-turn guard."""
    import gateway.run as gateway_run

    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    runner._running_agents[session_key] = object()

    captured = {}

    class _PickerAdapter:
        async def send_model_picker(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(success=True)

    runner.adapters = {Platform.TELEGRAM: _PickerAdapter()}

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {"default": "old-model", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_picker_providers",
        lambda **_kwargs: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["new-model"],
                "total_models": 1,
                "is_current": True,
            }
        ],
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: pytest.fail("switch_model must not run mid-turn"),
    )

    event = SimpleNamespace(source=source, get_command_args=lambda: "")

    result = await runner._handle_model_command(event)

    assert result is None
    callback = captured["on_model_selected"]
    callback_result = await callback(source.chat_id, "new-model", "openrouter")

    assert callback_result == "Agent is running — wait or /stop first, then switch models."
    assert session_key not in runner._session_model_overrides
