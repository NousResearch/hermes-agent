"""Tests that gateway /model switch persists across messages.

The gateway /model command stores session overrides in
``_session_model_overrides``.  These must:

1. Be applied in ``run_sync()`` so the next agent uses the switched model.
2. Not be mistaken for fallback activation (which evicts the cached agent).
3. Survive across multiple messages until /reset clears them.

Tests exercise the real ``_apply_session_model_override()`` and
``_is_intentional_model_switch()`` methods on ``GatewayRunner``.
"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

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
        previous_reasoning = {"enabled": True, "effort": "low"}
        runner._session_model_overrides[sk] = previous
        runner._session_reasoning_overrides[sk] = previous_reasoning

        snapshot = runner._snapshot_session_model_override(sk)
        runner._claim_one_turn_restore(sk, snapshot)
        runner._claim_one_turn_reasoning_restore(sk)
        runner._session_model_overrides[sk] = {
            "model": "temp/model",
            "provider": "anthropic",
        }
        runner._session_reasoning_overrides[sk] = {"enabled": True, "effort": "high"}

        runner._restore_pending_one_turn_model_override(sk)

        assert runner._session_model_overrides[sk] == previous
        assert runner._session_reasoning_overrides[sk] == previous_reasoning

        model_only_snapshot = runner._snapshot_session_model_override(sk)
        later_reasoning = {"enabled": True, "effort": "xhigh"}
        runner._session_reasoning_overrides[sk] = later_reasoning

        runner._restore_session_model_override(sk, model_only_snapshot)

        assert runner._session_reasoning_overrides[sk] == later_reasoning


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
        import gateway.run as gateway_run
        from gateway.run import GatewayRunner
        from hermes_cli.model_switch import ModelSwitchResult

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "agent": {"reasoning_effort": "low"},
                    "model": {"default": "old-model", "provider": "openrouter"},
                }
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
    async def test_once_keeps_model_and_reasoning_non_persistent(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())

        result = await runner._handle_model_command(
            self._event("/model gpt-5.5 --reasoning high --once")
        )

        assert result is not None and "gpt-5.5" in result
        # In-memory override installed for the next turn + restore queued...
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        assert runner._session_model_overrides[sk]["capabilities"] == {
            "openai_native_compaction": True
        }
        assert runner._session_reasoning_overrides[sk] == {"enabled": True, "effort": "high"}
        assert sk in runner._pending_one_turn_model_restores
        assert runner._pending_one_turn_model_restores[sk]["had_reasoning_override"] is False
        # ...but NEVER written through to the persistent session store.
        runner.async_session_store.set_model_override.assert_not_awaited()
        config = yaml.safe_load((tmp_path / ".hermes" / "config.yaml").read_text(encoding="utf-8"))
        assert config["agent"]["reasoning_effort"] == "low"

        runner._restore_pending_one_turn_model_override(sk)

        assert sk not in runner._session_reasoning_overrides
        assert runner._resolve_session_reasoning_config(session_key=sk, model="old-model") == {
            "enabled": True, "effort": "low"
        }

    @pytest.mark.asyncio
    async def test_repeated_once_keeps_the_earliest_restore_target(self, tmp_path, monkeypatch):
        """`/model X --once` then `/model Y --once` before any turn: the pending snapshot must still
        be the user's standing override (none here), not X — otherwise slot cleanup would make the
        first temporary model permanent."""
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        prior_reasoning = {"enabled": True, "effort": "medium"}
        runner._session_reasoning_overrides[sk] = prior_reasoning

        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        await runner._handle_model_command(self._event("/model gpt-5.5 --reasoning xhigh --once"))

        # The second producer call snapshotted the live gpt-5.5 override; the pending restore
        # must still be the ORIGINAL "no override" state.
        assert runner._pending_one_turn_model_restores[sk]["had_override"] is False
        assert runner._pending_one_turn_model_restores[sk]["restore_reasoning"] is True
        assert runner._pending_one_turn_model_restores[sk]["reasoning_override"] == prior_reasoning
        assert runner._session_reasoning_overrides[sk] == {"enabled": True, "effort": "xhigh"}

    @pytest.mark.asyncio
    async def test_reasoning_restore_starts_when_first_reasoning_once_is_queued(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        runner._session_reasoning_overrides[sk] = {"enabled": True, "effort": "medium"}

        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))
        await runner._handle_reasoning_command(self._event("/reasoning high"))
        await runner._handle_model_command(
            self._event("/model gpt-5.5 --reasoning xhigh --once")
        )

        runner._restore_pending_one_turn_model_override(sk)

        assert runner._session_reasoning_overrides[sk] == {"enabled": True, "effort": "high"}

    @pytest.mark.asyncio
    async def test_reasoning_restore_uses_state_at_one_turn_commit(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        runner._session_reasoning_overrides[sk] = {"enabled": True, "effort": "medium"}
        switch_started = asyncio.Event()
        continue_switch = asyncio.Event()
        perform_switch = runner._perform_model_switch

        async def paused_switch(*args, **kwargs):
            switch_started.set()
            await continue_switch.wait()
            return await perform_switch(*args, **kwargs)

        runner._perform_model_switch = paused_switch
        model_command = asyncio.create_task(
            runner._handle_model_command(
                self._event("/model gpt-5.5 --reasoning xhigh --once")
            )
        )
        await switch_started.wait()
        await runner._handle_reasoning_command(self._event("/reasoning high"))
        continue_switch.set()
        await model_command

        runner._restore_pending_one_turn_model_override(sk)

        assert runner._session_reasoning_overrides[sk] == {"enabled": True, "effort": "high"}

    @pytest.mark.asyncio
    async def test_superseded_one_turn_switch_does_not_apply_reasoning(
        self, tmp_path, monkeypatch
    ):
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        runner._session_reasoning_overrides[sk] = {"enabled": True, "effort": "medium"}
        confirmation_started = asyncio.Event()
        continue_confirmation = asyncio.Event()
        build_confirmation = runner._model_switch_confirmation
        confirmation_count = 0

        async def paused_first_confirmation(*args, **kwargs):
            nonlocal confirmation_count
            confirmation_count += 1
            if confirmation_count == 1:
                confirmation_started.set()
                await continue_confirmation.wait()
            return await build_confirmation(*args, **kwargs)

        runner._model_switch_confirmation = paused_first_confirmation
        one_turn_command = asyncio.create_task(
            runner._handle_model_command(
                self._event("/model gpt-5.5 --reasoning xhigh --once")
            )
        )
        await confirmation_started.wait()
        await runner._handle_model_command(self._event("/model gpt-5.5 --session"))
        continue_confirmation.set()
        await one_turn_command

        assert sk not in runner._pending_one_turn_model_restores
        assert runner._session_reasoning_overrides[sk] == {"enabled": True, "effort": "medium"}

