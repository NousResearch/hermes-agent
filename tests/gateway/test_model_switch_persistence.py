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
        from gateway.platforms.base import MessageEvent, MessageType

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
    async def test_global_900k_switch_clears_stale_session_override(
        self, tmp_path, monkeypatch
    ):
        """A global context-variant selection must become the live authority.

        Keeping the previous per-session base slug makes it outrank the newly
        persisted global ``-900k`` alias after restart, shrinking the effective
        window back to 272K and firing compression at 231,200 tokens.
        """
        from hermes_cli.model_switch import ModelSwitchResult

        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        events = []
        from hermes_cli.config import save_config as real_save_config

        def recording_save(config):
            events.append("config")
            return real_save_config(config)

        monkeypatch.setattr("hermes_cli.config.save_config", recording_save)
        runner.async_session_store.set_model_override.side_effect = (
            lambda _key, value: events.append(("session", value))
        )
        runner._session_model_overrides[sk] = {
            "model": "gpt-5.6-sol",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.6-sol-900k",
                target_provider="openai-codex",
                provider_changed=False,
                api_key="oauth-token",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                provider_label="OpenAI Codex",
            ),
        )

        result = await runner._handle_model_command(
            self._event("/model gpt-5.6-sol-900k --global")
        )

        assert result is not None and "gpt-5.6-sol-900k" in result
        assert sk not in runner._session_model_overrides
        runner.async_session_store.set_model_override.assert_awaited_with(sk, None)
        assert events[-2:] == ["config", ("session", None)]

    @pytest.mark.asyncio
    async def test_global_save_failure_keeps_900k_session_override(
        self, tmp_path, monkeypatch
    ):
        """If config persistence fails, the successful switch stays session-local."""
        from hermes_cli.model_switch import ModelSwitchResult

        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.6-sol-900k",
                target_provider="openai-codex",
                api_key="oauth-token",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                provider_label="OpenAI Codex",
            ),
        )
        monkeypatch.setattr(
            "hermes_cli.config.save_config",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )

        result = await runner._handle_model_command(
            self._event("/model gpt-5.6-sol-900k --global")
        )

        assert result is not None
        assert "global config write failed" in result.lower()
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.6-sol-900k"
        runner.async_session_store.set_model_override.assert_awaited_with(
            sk, runner._session_model_overrides[sk]
        )

    @pytest.mark.asyncio
    async def test_global_clear_failure_rolls_back_config_and_keeps_900k(
        self, tmp_path, monkeypatch
    ):
        """A failed durable clear rolls config back before session fallback."""
        import yaml as _yaml
        from hermes_cli.model_switch import ModelSwitchResult

        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        durable = {}

        async def write_override(key, value):
            if value is None:
                raise OSError("state db locked")
            durable[key] = value

        runner.async_session_store.set_model_override.side_effect = write_override
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model="gpt-5.6-sol-900k",
                target_provider="openai-codex",
                api_key="oauth-token",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                provider_label="OpenAI Codex",
            ),
        )

        result = await runner._handle_model_command(
            self._event("/model gpt-5.6-sol-900k --global")
        )

        cfg = _yaml.safe_load(
            (tmp_path / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        )
        assert cfg["model"]["default"] == "gpt-5.6-sol-900k"
        assert "override cleanup failed" in result.lower()
        assert runner._session_model_overrides[sk]["model"] == "gpt-5.6-sol-900k"
        assert durable[sk]["model"] == "gpt-5.6-sol-900k"

    @pytest.mark.asyncio
    async def test_concurrent_later_session_switch_wins_durably(
        self, tmp_path, monkeypatch
    ):
        """One session's model-switch commit is serialized end to end."""
        from hermes_cli.model_switch import ModelSwitchResult

        runner = self._runner_with_store(tmp_path, monkeypatch)
        sk = build_session_key(_make_source())
        clear_started = asyncio.Event()
        release_clear = asyncio.Event()
        durable = {}

        async def write_override(key, value):
            if value is None:
                clear_started.set()
                await release_clear.wait()
            durable[key] = value

        runner.async_session_store.set_model_override.side_effect = write_override

        def switch_model(**kwargs):
            raw = kwargs["raw_input"]
            is_large = raw.endswith("-900k")
            return ModelSwitchResult(
                success=True,
                new_model=raw,
                target_provider="openai-codex",
                api_key="oauth-token",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                provider_label="OpenAI Codex",
                provider_changed=False,
                resolved_via_alias="large" if is_large else "small",
            )

        monkeypatch.setattr("hermes_cli.model_switch.switch_model", switch_model)

        global_task = asyncio.create_task(
            runner._handle_model_command(
                self._event("/model gpt-5.6-sol-900k --global")
            )
        )
        await asyncio.wait_for(clear_started.wait(), timeout=2)
        session_task = asyncio.create_task(
            runner._handle_model_command(
                self._event("/model gpt-5.6-sol --session")
            )
        )
        await asyncio.sleep(0.05)

        # The later command must wait for the global command's config+override
        # commit; otherwise its durable write can be erased by the older clear.
        assert not session_task.done()

        release_clear.set()
        await asyncio.gather(global_task, session_task)

        assert runner._session_model_overrides[sk]["model"] == "gpt-5.6-sol"
        assert durable[sk]["model"] == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_concurrent_global_switches_share_config_lock(
        self, tmp_path, monkeypatch
    ):
        """Global config commits serialize across different conversation keys."""
        import yaml as _yaml
        from gateway.platforms.base import MessageEvent, MessageType
        from hermes_cli.model_switch import ModelSwitchResult

        runner = self._runner_with_store(tmp_path, monkeypatch)
        clear_a_started = asyncio.Event()
        release_a = asyncio.Event()

        async def write_override(key, value):
            if value is None and key.endswith(":c1"):
                clear_a_started.set()
                await release_a.wait()
                raise OSError("session A clear failed")

        runner.async_session_store.set_model_override.side_effect = write_override
        monkeypatch.setattr(
            "hermes_cli.model_switch.switch_model",
            lambda **kw: ModelSwitchResult(
                success=True,
                new_model=kw["raw_input"],
                target_provider="openai-codex",
                api_key="oauth-token",
                base_url="https://chatgpt.com/backend-api/codex",
                api_mode="codex_responses",
                provider_label="OpenAI Codex",
            ),
        )

        def event(chat_id, model):
            return MessageEvent(
                text=f"/model {model} --global",
                message_type=MessageType.TEXT,
                source=SessionSource(
                    platform=Platform.TELEGRAM,
                    chat_id=chat_id,
                    chat_type="dm",
                ),
            )

        first = asyncio.create_task(
            runner._handle_model_command(event("c1", "gpt-5.6-sol-900k"))
        )
        await asyncio.wait_for(clear_a_started.wait(), timeout=2)
        second = asyncio.create_task(
            runner._handle_model_command(event("c2", "gpt-5.6-terra-900k"))
        )
        await asyncio.sleep(0.05)
        assert not second.done()

        release_a.set()
        await asyncio.gather(first, second)

        cfg = _yaml.safe_load(
            (tmp_path / ".hermes" / "config.yaml").read_text(encoding="utf-8")
        )
        assert cfg["model"]["default"] == "gpt-5.6-terra-900k"

