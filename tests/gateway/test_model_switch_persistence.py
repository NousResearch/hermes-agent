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

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
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


class TestFallbackEvictionHonorsChannelOverrides:
    """A ``channel_overrides`` model is the configured model for that chat, so an agent still running it
    after a successful turn is not a fallback; comparing it with the global default evicted (and rebuilt)
    the cached agent after every turn."""

    @staticmethod
    def _evicted_after_turn(
        monkeypatch, source, agent_model, *, global_model="default/model", channel_model="chan/model",
        session_model=None, session_provider="nous", **agent_attrs,
    ):
        from gateway.turn_context import TurnContext

        monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda *a, **k: global_model)
        runner = _make_runner()
        runner.config.platforms[Platform.TELEGRAM].channel_overrides = {"c1": ChannelOverride(model=channel_model)}
        if session_model:
            runner._session_model_overrides[build_session_key(source)] = {
                "model": session_model, "provider": session_provider,
            }
        runner._evict_cached_agent = MagicMock()
        ctx = TurnContext(source=source, session_key=build_session_key(source))
        ctx.agent_holder[0] = SimpleNamespace(model=agent_model, **({"provider": "openrouter"} | agent_attrs))
        ctx.result_holder[0] = {"final_response": "ok"}
        runner._run_agent_evict_on_fallback(ctx)
        return runner._evict_cached_agent.called

    @pytest.mark.parametrize("thread_of_the_channel", [False, True])
    def test_channel_override_model_keeps_the_cached_agent(self, monkeypatch, thread_of_the_channel):
        source = _make_source()
        if thread_of_the_channel:
            source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="t9", parent_chat_id="c1",
                                   user_name="tester", chat_type="thread")
        assert self._evicted_after_turn(monkeypatch, source, "chan/model") is False

    def test_fallback_off_the_channel_model_still_evicts(self, monkeypatch):
        assert self._evicted_after_turn(monkeypatch, _make_source(), "fallback/model") is True

    _WELCOME = "https://welcome-api.nousresearch.com/v1"

    @pytest.mark.parametrize("global_model", ["nous/welcome", "default/model"])
    def test_the_welcome_host_pins_its_model_over_the_channel_model(self, monkeypatch, global_model):
        """AIAgent.__init__ runs a welcome-host agent on ``nous/welcome`` whatever the chat configured."""
        from hermes_cli.anon_auth import pin_model_for_route

        agent_model = pin_model_for_route("nous", self._WELCOME, "chan/model")
        assert agent_model == "nous/welcome"
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), agent_model, global_model=global_model,
            provider="nous", base_url=self._WELCOME,
        ) is False

    def test_the_recorded_welcome_alias_switch_keeps_the_cached_agent(self, monkeypatch):
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), "backing/model", provider="nous", base_url=self._WELCOME,
            _nous_model_switch=("nous/welcome", "backing/model"),
        ) is False

    def test_a_fallback_onto_the_welcome_host_still_evicts(self, monkeypatch):
        """The pin applies to the agent's primary route, not the fallback route it is on now."""
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), "nous/welcome", provider="nous", base_url=self._WELCOME,
            _primary_runtime={"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"},
        ) is True

    @pytest.mark.parametrize("agent_model, session_provider, switch, evicted", [
        ("backing/model", "nous", ("nous/welcome", "backing/model"), False),  # the server's recorded move
        ("chan/model", "nous", None, True),  # a fallback that happens to land on the channel's model
        # The override's provider was unavailable, so this turn ran on the channel's model.
        ("chan/model", "openai-codex", None, False),
    ], ids=["recorded-switch-off-the-session-model", "fallback-onto-the-channel-model", "override-unavailable"])
    def test_a_session_model_override_outranks_the_channel_model(
        self, monkeypatch, agent_model, session_provider, switch, evicted,
    ):
        """The agent was built on the session's /model choice, so drift is measured from it."""
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), agent_model, global_model="backing/model", session_model="nous/welcome",
            session_provider=session_provider, provider="nous", base_url="https://inference-api.nousresearch.com/v1",
            **({"_nous_model_switch": switch} if switch else {}),
        ) is evicted

    @pytest.mark.parametrize("session_model", [None, "chan/model"], ids=["channel-model", "session-model"])
    def test_a_provider_fallback_serving_the_configured_model_still_evicts(self, monkeypatch, session_model):
        """Same model name, different endpoint: the model strings match, but the turn ran on a fallback."""
        route = {"provider": "custom", "base_url": "https://primary.example/v1"}
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), "chan/model", session_model=session_model, session_provider="custom",
            provider="custom", base_url="https://fallback.example/v1", _primary_runtime=route,
            _provider_fallback_active=True,
        ) is True

    @pytest.mark.parametrize("primary_provider, evicted", [
        ("anthropic", False),  # native primary: the agent runs the prefix-stripped channel model
        ("openrouter", True),  # OpenRouter primary that fell back to native Anthropic
    ])
    def test_the_channel_model_is_normalized_for_the_primary_provider(self, monkeypatch, primary_provider, evicted):
        assert self._evicted_after_turn(
            monkeypatch, _make_source(), "claude-sonnet-4-6", channel_model="anthropic/claude-sonnet-4.6",
            provider="anthropic", base_url="https://api.anthropic.com",
            _primary_runtime={"provider": primary_provider},
        ) is evicted


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
        import hermes_yaml as _yaml

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
        monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
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

    @pytest.mark.asyncio
    async def test_repeated_once_keeps_the_earliest_restore_target(self, tmp_path, monkeypatch):
        """`/model X --once` then `/model Y --once` before any turn: the pending snapshot must still
        be the user's standing override (none here), not X — otherwise slot cleanup would make the
        first temporary model permanent."""
        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())

        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.5"
        await runner._handle_model_command(self._event("/model gpt-5.5 --once"))

        # The second producer call snapshotted the live gpt-5.5 override; the pending restore
        # must still be the ORIGINAL "no override" state.
        assert runner._pending_one_turn_model_restores[sk]["had_override"] is False

