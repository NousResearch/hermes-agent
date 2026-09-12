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
    runner._pending_one_turn_model_restores = {}
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


# ---------------------------------------------------------------------------
# Tests: _is_intentional_model_switch
# ---------------------------------------------------------------------------


class TestIsIntentionalModelSwitch:
    """The fallback-eviction check must not evict a session whose model differs from the config
    default for a reason the system produced: a /model override, or the Nous gateway moving the
    session off the ``nous/welcome`` alias the config still carries."""

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

        agent = SimpleNamespace(model="gpt-5.4")
        assert runner._is_intentional_model_switch(sk, agent, "openai/gpt-5") is True

    def test_server_model_switch_off_the_welcome_alias_is_intentional(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        # apply_model_switch moved the session and recorded the move (alias -> backing).
        agent = SimpleNamespace(model="z-ai/glm-5.3-flash", _nous_model_switch=("nous/welcome", "z-ai/glm-5.3-flash"))
        assert runner._is_intentional_model_switch(sk, agent, "nous/welcome") is True
        # A config that names something else is real drift, not the server's move.
        assert runner._is_intentional_model_switch(sk, agent, "openai/gpt-5") is False
        # A later fallback onto a third model is drift too, even with the config still on the alias.
        agent.model = "fallback/model"
        assert runner._is_intentional_model_switch(sk, agent, "nous/welcome") is False

    def test_plain_drift_is_not_intentional(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        agent = SimpleNamespace(model="fallback/model")
        assert runner._is_intentional_model_switch(sk, agent, "primary/model") is False


class TestOneTurnModelOverrideRestore:
    """Verify gateway one-turn overrides restore previous session state."""

    def test_restores_previous_override(self):
        runner = _make_runner()
        sk = build_session_key(_make_source())
        previous = {
            "model": "old/model",
            "provider": "openrouter",
            "api_key": "old-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        }
        runner._session_model_overrides[sk] = previous

        snapshot = runner._snapshot_session_model_override(sk)
        runner._session_model_overrides[sk] = {
            "model": "temp/model",
            "provider": "anthropic",
        }

        runner._restore_session_model_override(sk, snapshot)

        assert runner._session_model_overrides[sk] == previous


@pytest.mark.asyncio
async def test_model_switch_note_preserves_session_language_and_is_consumed_once():
    from gateway.run_turn_runner import TurnRunner
    from gateway.slash_commands_model import _ModelSwitchContext

    runner = _make_runner()
    source = _make_source()
    session_key = build_session_key(source)
    other_key = "agent:main:telegram:dm:other-chat"
    runner._async_session_store = SimpleNamespace(set_model_override=AsyncMock())
    runner._pending_model_notes[other_key] = "[Note: other session.]"
    ctx = _ModelSwitchContext(
        session_key=session_key,
        source=source,
        config_path=None,
        persist_global=False,
        current_model="old/model",
    )
    result = SimpleNamespace(
        new_model="new/model",
        target_provider="openrouter",
        provider_label="OpenRouter",
        api_key="sk-test",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        request_overrides={},
        runtime_capabilities={},
    )

    await runner._record_model_switch(
        result, ctx, source=source, one_turn=False, picker=False
    )

    note = runner._pending_model_notes[session_key]
    assert (
        "Continue responding in the language already established in this conversation."
        in note
    )

    first_turn = SimpleNamespace(session_key=session_key, message="Next question")
    TurnRunner(runner, first_turn)._prepend_pending_note("_pending_model_notes")
    assert first_turn.message == f"{note}\n\nNext question"
    assert session_key not in runner._pending_model_notes
    assert runner._pending_model_notes[other_key] == "[Note: other session.]"

    second_turn = SimpleNamespace(session_key=session_key, message="Later question")
    TurnRunner(runner, second_turn)._prepend_pending_note("_pending_model_notes")
    assert second_turn.message == "Later question"


class TestOneTurnNeverPersisted:
    """/model --once must never write through to the session store.

    Regression guard for the #29923 review defect: the original
    implementation wrote the once-override through set_model_override, so a
    gateway restart before the finally-restore rehydrated a supposedly
    one-turn model permanently. Drives the real _handle_model_command with
    a mocked switch pipeline and asserts on the store boundary.
    """

    @staticmethod
    def _runner_with_store(tmp_path, monkeypatch):
        import yaml as _yaml

        import gateway.run as gateway_run
        from gateway.run import GatewayRunner
        from hermes_cli.model_switch import ModelSwitchResult

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            _yaml.safe_dump(
                {"model": {"default": "old-model", "provider": "openrouter"}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.5",
                target_provider="openrouter",
                provider_changed=False,
                api_key="sk-test",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
                runtime_capabilities={"openai_native_compaction": True},
                provider_label="OpenRouter",
            ),
        )
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._pending_one_turn_model_restores = {}
        runner._running_agents = {}
        # async_session_store is a property over session_store; install the
        # mock behind the private cache attribute it reads.
        _store = MagicMock()
        _store.set_model_override = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _event(text):
        from gateway.platforms.event import MessageEvent, MessageType

        return MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=_make_source(),
        )

    @pytest.mark.asyncio
    async def test_once_skips_session_store_write_through(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())

        result = await runner._handle_model_command(
            self._event("/model gpt-5.5 --once")
        )

        assert result is not None and "gpt-5.5" in result
        # In-memory override installed for the next turn + restore queued...
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        assert runner._session_model_overrides[sk]["capabilities"] == {
            "openai_native_compaction": True
        }
        assert sk in runner._pending_one_turn_model_restores
        # ...but NEVER written through to the persistent session store.
        runner.async_session_store.set_model_override.assert_not_awaited()

