"""Tests for Telegram /model switch propagating to the live agent session.

Regression for issue #37621 — when a user sends `/model claude-opus-4-5` in
Telegram, the bot acknowledges the switch but the running session continued
using the old model because the in-place ``switch_model()`` call was not
verified to reach the cached agent.

Covers:
* ``_handle_model_command`` (direct ``/model <name>`` path) — verifies that
  ``switch_model()`` is called on the cached agent AND the session override is
  stored so subsequent fresh-agent creations also use the new model.
* Picker path (``_on_model_selected`` callback) — same assertions for the
  Telegram inline-keyboard picker flow.
* Session override applied on next turn — verifies that after the command the
  next agent resolved by ``_resolve_session_agent_runtime`` uses the new model.
"""

from __future__ import annotations

import sys
import threading
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_telegram_mock():
    """Install a minimal telegram stub so telegram.py can be imported."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()


def _make_source(chat_id: str = "123456", user_id: str = "u1") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id=user_id,
        chat_id=chat_id,
        user_name="tester",
        chat_type="dm",
    )


def _make_runner():
    """Create a minimal GatewayRunner with stubbed internals."""
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = False
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._effective_model = None
    runner._effective_provider = None

    source = _make_source()
    session_key = build_session_key(source)
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-tg-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    store._generate_session_key.return_value = session_key
    store._entries = {session_key: session_entry}
    runner.session_store = store
    return runner


def _make_event(text: str = "/model claude-opus-4-5") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=_make_source(),
    )


def _stub_switch_model_result(
    new_model: str = "claude-opus-4-5",
    provider: str = "anthropic",
    api_key: str = "sk-ant-test",
    base_url: str = "",
    api_mode: str = "anthropic_messages",
):
    """Return a minimal ModelSwitchResult-like object."""
    r = MagicMock()
    r.success = True
    r.new_model = new_model
    r.target_provider = provider
    r.api_key = api_key
    r.base_url = base_url
    r.api_mode = api_mode
    r.provider_label = provider
    r.model_info = None
    r.warning_message = None
    r.error_message = ""
    return r


# ---------------------------------------------------------------------------
# Test: direct /model <name> command updates cached agent in-place
# ---------------------------------------------------------------------------


class TestDirectModelCommandUpdatesLiveAgent:
    """Verify that /model <name> calls switch_model() on the live cached agent."""

    @pytest.mark.asyncio
    async def test_switch_model_called_on_cached_agent(self, monkeypatch, tmp_path):
        """switch_model() must be invoked on the cached agent when one exists."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Seed the agent cache with a fake live agent
        fake_agent = MagicMock()
        fake_agent.switch_model = MagicMock()
        runner._agent_cache[session_key] = (fake_agent, "old-sig")

        switch_result = _stub_switch_model_result()

        # Patch hermes_cli.model_switch so we control the result
        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("claude-opus-4-5", "", False, False)
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.list_picker_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        # Minimal config
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        event = _make_event("/model claude-opus-4-5")
        result = await runner._handle_model_command(event)

        # switch_model() must have been called on the live cached agent
        fake_agent.switch_model.assert_called_once_with(
            new_model="claude-opus-4-5",
            new_provider="anthropic",
            api_key="sk-ant-test",
            base_url="",
            api_mode="anthropic_messages",
        )

    @pytest.mark.asyncio
    async def test_session_override_stored_after_switch(self, monkeypatch, tmp_path):
        """Session override must be stored so the next fresh-agent creation uses the new model."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        switch_result = _stub_switch_model_result(
            new_model="claude-opus-4-5",
            provider="anthropic",
            api_key="sk-ant-test",
            api_mode="anthropic_messages",
        )

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("claude-opus-4-5", "", False, False)
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.list_picker_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        event = _make_event("/model claude-opus-4-5")
        await runner._handle_model_command(event)

        # The session override must have been stored
        assert session_key in runner._session_model_overrides
        override = runner._session_model_overrides[session_key]
        assert override["model"] == "claude-opus-4-5"
        assert override["provider"] == "anthropic"
        assert override["api_mode"] == "anthropic_messages"

    @pytest.mark.asyncio
    async def test_cached_agent_evicted_after_switch(self, monkeypatch, tmp_path):
        """Cached agent must be evicted so the next turn creates a fresh agent from the override."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        fake_agent = MagicMock()
        fake_agent.switch_model = MagicMock()
        runner._agent_cache[session_key] = (fake_agent, "old-sig")

        switch_result = _stub_switch_model_result()

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("claude-opus-4-5", "", False, False)
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.list_picker_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        event = _make_event("/model claude-opus-4-5")
        await runner._handle_model_command(event)

        # Cached agent must have been evicted
        assert session_key not in runner._agent_cache

    @pytest.mark.asyncio
    async def test_switch_model_failure_still_stores_override(self, monkeypatch, tmp_path):
        """If switch_model() raises on the cached agent, the session override must still be stored."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        fake_agent = MagicMock()
        fake_agent.switch_model = MagicMock(
            side_effect=RuntimeError("client build failed")
        )
        runner._agent_cache[session_key] = (fake_agent, "old-sig")

        switch_result = _stub_switch_model_result()

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("claude-opus-4-5", "", False, False)
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.list_picker_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        event = _make_event("/model claude-opus-4-5")
        # Must not raise even though switch_model() on the cached agent failed
        result = await runner._handle_model_command(event)

        # Override must still be stored for the next turn's fresh-agent creation
        assert session_key in runner._session_model_overrides
        override = runner._session_model_overrides[session_key]
        assert override["model"] == "claude-opus-4-5"

    @pytest.mark.asyncio
    async def test_no_cached_agent_still_stores_override(self, monkeypatch, tmp_path):
        """When there is no cached agent, the session override must be stored for the next turn."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # No cached agent
        assert session_key not in runner._agent_cache

        switch_result = _stub_switch_model_result()

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("claude-opus-4-5", "", False, False)
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.list_picker_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        event = _make_event("/model claude-opus-4-5")
        await runner._handle_model_command(event)

        assert session_key in runner._session_model_overrides
        assert runner._session_model_overrides[session_key]["model"] == "claude-opus-4-5"


# ---------------------------------------------------------------------------
# Test: session override applied on next turn
# ---------------------------------------------------------------------------


class TestSessionOverrideAppliedOnNextTurn:
    """Verify _apply_session_model_override returns the switched model for the next turn."""

    def test_override_applied_to_next_turn_model(self):
        """After /model, _apply_session_model_override must return the new model."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        runner._session_model_overrides[session_key] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": "sk-ant-new",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }

        model, rt = runner._apply_session_model_override(
            session_key,
            "claude-sonnet-4-6",  # old model from config
            {"provider": "anthropic", "api_key": "sk-ant-old", "base_url": "", "api_mode": "anthropic_messages"},
        )

        assert model == "claude-opus-4-5"
        assert rt["api_key"] == "sk-ant-new"
        assert rt["provider"] == "anthropic"
        assert rt["api_mode"] == "anthropic_messages"

    def test_override_not_applied_for_different_session_key(self):
        """Override stored for session A must not bleed into session B."""
        runner = _make_runner()

        runner._session_model_overrides["session-A"] = {
            "model": "claude-opus-4-5",
            "provider": "anthropic",
            "api_key": "sk-ant-a",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }

        model, rt = runner._apply_session_model_override(
            "session-B",
            "claude-sonnet-4-6",
            {"provider": "openrouter", "api_key": "sk-or-b", "base_url": "https://openrouter.ai/api/v1", "api_mode": "chat_completions"},
        )

        assert model == "claude-sonnet-4-6"
        assert rt["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# Test: Telegram picker callback calls switch_model() via on_model_selected
# ---------------------------------------------------------------------------


class TestPickerCallbackAppliesSwitch:
    """Verify the Telegram picker callback propagates the switch to the live agent.

    These tests exercise the ``_on_model_selected`` closure created inside
    ``_handle_model_command`` when Telegram's inline-keyboard picker is used.
    Rather than routing through the full picker UI, we call the closure
    directly to verify the core switch+override logic.
    """

    def _build_runner_with_picker_adapter(self):
        """Runner whose Telegram adapter has ``send_model_picker`` on its type."""
        runner = _make_runner()

        # The has_picker check in _handle_model_command uses getattr(type(adapter), ...)
        # A plain MagicMock's type doesn't expose send_model_picker, so we need a
        # class-based stub.
        class _FakeAdapterWithPicker:
            platform = Platform.TELEGRAM

            async def send_model_picker(self, **kwargs):
                raise NotImplementedError  # overridden per test

        adapter = _FakeAdapterWithPicker()
        runner.adapters[Platform.TELEGRAM] = adapter
        return runner, adapter

    @pytest.mark.asyncio
    async def test_on_model_selected_calls_switch_model_on_agent(self, monkeypatch, tmp_path):
        """The _on_model_selected closure must call switch_model() on the cached agent."""
        runner, adapter = self._build_runner_with_picker_adapter()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        # Seed the agent cache with a fake live agent
        fake_agent = MagicMock()
        fake_agent.switch_model = MagicMock()
        runner._agent_cache[session_key] = (fake_agent, "old-sig-picker")

        switch_result = _stub_switch_model_result(
            new_model="claude-opus-4-5",
            provider="anthropic",
            api_key="sk-ant-picker",
            api_mode="anthropic_messages",
        )

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("", "", False, False)
        ms_mod.list_picker_providers = lambda **kw: [
            {"slug": "anthropic", "name": "Anthropic", "models": ["claude-opus-4-5"],
             "total_models": 1, "is_current": False}
        ]
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        # Capture the on_model_selected callback by intercepting send_model_picker
        captured_callback = []

        async def fake_send_model_picker(**kwargs):
            cb = kwargs.get("on_model_selected")
            if cb:
                captured_callback.append(cb)
            from gateway.platforms.base import SendResult
            return SendResult(success=True, message_id="42")

        adapter.send_model_picker = fake_send_model_picker

        event = _make_event("/model")
        await runner._handle_model_command(event)

        # Picker should have been sent (no model_input → picker path)
        assert len(captured_callback) == 1, "Expected on_model_selected callback to be captured"

        # Simulate user picking a model from the picker
        cb = captured_callback[0]
        await cb(source.chat_id, "claude-opus-4-5", "anthropic")

        # switch_model() must have been called on the live cached agent
        fake_agent.switch_model.assert_called_once_with(
            new_model="claude-opus-4-5",
            new_provider="anthropic",
            api_key="sk-ant-picker",
            base_url="",
            api_mode="anthropic_messages",
        )

    @pytest.mark.asyncio
    async def test_on_model_selected_stores_session_override(self, monkeypatch, tmp_path):
        """After picker selection, session override must be stored for next-turn fresh-agent."""
        runner, adapter = self._build_runner_with_picker_adapter()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        switch_result = _stub_switch_model_result(
            new_model="claude-opus-4-5",
            provider="anthropic",
            api_key="sk-ant-picker",
            api_mode="anthropic_messages",
        )

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("", "", False, False)
        ms_mod.list_picker_providers = lambda **kw: [
            {"slug": "anthropic", "name": "Anthropic", "models": ["claude-opus-4-5"],
             "total_models": 1, "is_current": False}
        ]
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        captured_callback = []

        async def fake_send_model_picker(**kwargs):
            cb = kwargs.get("on_model_selected")
            if cb:
                captured_callback.append(cb)
            from gateway.platforms.base import SendResult
            return SendResult(success=True, message_id="42")

        adapter.send_model_picker = fake_send_model_picker

        event = _make_event("/model")
        await runner._handle_model_command(event)

        assert len(captured_callback) == 1

        cb = captured_callback[0]
        await cb(source.chat_id, "claude-opus-4-5", "anthropic")

        # Session override must be stored
        assert session_key in runner._session_model_overrides
        override = runner._session_model_overrides[session_key]
        assert override["model"] == "claude-opus-4-5"
        assert override["provider"] == "anthropic"
        assert override["api_mode"] == "anthropic_messages"

    @pytest.mark.asyncio
    async def test_on_model_selected_evicts_cached_agent(self, monkeypatch, tmp_path):
        """After picker selection, the cached agent must be evicted."""
        runner, adapter = self._build_runner_with_picker_adapter()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        fake_agent = MagicMock()
        fake_agent.switch_model = MagicMock()
        runner._agent_cache[session_key] = (fake_agent, "old-sig-picker")

        switch_result = _stub_switch_model_result()

        ms_mod = types.ModuleType("hermes_cli.model_switch")
        ms_mod.switch_model = lambda **kw: switch_result
        ms_mod.parse_model_flags = lambda raw: ("", "", False, False)
        ms_mod.list_picker_providers = lambda **kw: [
            {"slug": "anthropic", "name": "Anthropic", "models": ["claude-opus-4-5"],
             "total_models": 1, "is_current": False}
        ]
        ms_mod.list_authenticated_providers = lambda **kw: []
        ms_mod.resolve_display_context_length = lambda *a, **kw: None

        monkeypatch.setitem(sys.modules, "hermes_cli.model_switch", ms_mod)

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "model:\n  default: claude-sonnet-4-6\n  provider: anthropic\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        captured_callback = []

        async def fake_send_model_picker(**kwargs):
            cb = kwargs.get("on_model_selected")
            if cb:
                captured_callback.append(cb)
            from gateway.platforms.base import SendResult
            return SendResult(success=True, message_id="42")

        adapter.send_model_picker = fake_send_model_picker

        event = _make_event("/model")
        await runner._handle_model_command(event)

        assert len(captured_callback) == 1
        cb = captured_callback[0]
        await cb(source.chat_id, "claude-opus-4-5", "anthropic")

        assert session_key not in runner._agent_cache
