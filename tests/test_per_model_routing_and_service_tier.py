"""Per-model provider_routing overlay and service_tier_overrides.

Covers resolvers on the three config loaders plus CLI / gateway / TUI
construction and switch_model pin preservation. Real YAML against the
isolated HERMES_HOME from tests/conftest.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from hermes_constants import (
    get_hermes_home,
    resolve_provider_routing_for_model,
)

_MATCH_MODEL = "google/gemini-3.7-flash"
_OTHER_MODEL = "openai/gpt-5"

_CONFIG = {
    "model": {"default": _MATCH_MODEL, "provider": "openrouter"},
    "agent": {
        "service_tier": "priority",
        "service_tier_overrides": {
            _MATCH_MODEL: "flex",
        },
    },
    "provider_routing": {
        "sort": "throughput",
        "only": ["foo"],
        "models": {
            _MATCH_MODEL: {"only": ["google-ai-studio"]},
            "qwen/qwen3.8-27b": {"order": ["reka/fp8"]},
        },
    },
}


def _write_config(extra=None):
    payload = dict(_CONFIG)
    if extra:
        payload = {**payload, **extra}
    path = get_hermes_home() / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestThreeLoadersAcceptNewKeys:
    def test_load_config_keeps_models_and_overrides(self):
        _write_config()
        from hermes_cli.config import load_config

        cfg = load_config()
        assert cfg["provider_routing"]["models"][_MATCH_MODEL]["only"] == [
            "google-ai-studio"
        ]
        assert cfg["agent"]["service_tier_overrides"][_MATCH_MODEL] == "flex"

    def test_load_cli_config_keeps_models_and_overrides(self):
        _write_config()
        import cli as cli_mod

        with patch.object(cli_mod, "_hermes_home", get_hermes_home()):
            cfg = cli_mod.load_cli_config()
        assert cfg["provider_routing"]["models"][_MATCH_MODEL]["only"] == [
            "google-ai-studio"
        ]
        assert cfg["agent"]["service_tier_overrides"][_MATCH_MODEL] == "flex"

    def test_gateway_loader_keeps_models_and_overrides(self):
        _write_config()
        import gateway.run as gateway_run

        with patch.object(gateway_run, "_hermes_home", get_hermes_home()):
            raw = gateway_run.GatewayRunner._load_provider_routing()
        assert "models" in raw
        resolved = resolve_provider_routing_for_model(raw, _MATCH_MODEL)
        assert "models" not in resolved
        assert resolved["only"] == ["google-ai-studio"]
        assert resolved["sort"] == "throughput"

    def test_tui_loader_keeps_models_and_overrides(self):
        _write_config()
        import tui_gateway.server as server

        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None
        with patch.object(server, "_hermes_home", get_hermes_home()):
            raw = server._load_provider_routing()
            tier = server._load_service_tier(_MATCH_MODEL)
        assert "models" in raw
        resolved = resolve_provider_routing_for_model(raw, _MATCH_MODEL)
        assert "models" not in resolved
        assert resolved["only"] == ["google-ai-studio"]
        assert tier == "flex"


class TestCliAgentConstruction:
    def test_matching_model_gets_overlay_and_flex(self, monkeypatch):
        _write_config()
        import cli as cli_mod

        monkeypatch.setattr(cli_mod, "_hermes_home", get_hermes_home())
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", cli_mod.load_cli_config())
        captured = {}

        def _fake_agent(*_a, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)
        shell = cli_mod.HermesCLI(model=_MATCH_MODEL, compact=True, max_turns=1)
        shell._session_db = object()
        shell._resumed = False
        shell.conversation_history = []
        shell._install_tool_callbacks = lambda: None
        shell._ensure_tirith_security = lambda: None
        shell._ensure_runtime_credentials = lambda: True
        from hermes_cli import mcp_startup

        monkeypatch.setattr(
            mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **_k: None
        )
        assert shell._init_agent() is True
        assert captured["providers_allowed"] == ["google-ai-studio"]
        assert captured["provider_sort"] == "throughput"
        assert captured["service_tier"] == "flex"
        assert shell.agent._config_managed_routing_tier is True
        assert shell.agent._provider_routing_config["models"][_MATCH_MODEL]["only"] == [
            "google-ai-studio"
        ]

    def test_unrelated_model_keeps_flat_routing_and_global_tier(self, monkeypatch):
        _write_config()
        import cli as cli_mod

        monkeypatch.setattr(cli_mod, "_hermes_home", get_hermes_home())
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", cli_mod.load_cli_config())
        captured = {}

        def _fake_agent(*_a, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(cli_mod, "AIAgent", _fake_agent)
        shell = cli_mod.HermesCLI(model=_OTHER_MODEL, compact=True, max_turns=1)
        shell._session_db = object()
        shell._resumed = False
        shell.conversation_history = []
        shell._install_tool_callbacks = lambda: None
        shell._ensure_tirith_security = lambda: None
        shell._ensure_runtime_credentials = lambda: True
        from hermes_cli import mcp_startup

        monkeypatch.setattr(
            mcp_startup, "ensure_mcp_discovery_before_agent_build", lambda **_k: None
        )
        assert shell._init_agent() is True
        assert captured["providers_allowed"] == ["foo"]
        assert captured["service_tier"] == "priority"


class TestGatewaySessionResolve:
    def test_pin_beats_per_model_beats_global(self, monkeypatch):
        import gateway.run as gateway_run
        from gateway.session_state import SERVICE_TIER_UNSET

        runner = object.__new__(gateway_run.GatewayRunner)
        pinned = SimpleNamespace(
            conversation=SimpleNamespace(service_tier_override="priority")
        )
        unpinned = SimpleNamespace(
            conversation=SimpleNamespace(service_tier_override=SERVICE_TIER_UNSET)
        )
        runner._peek_session_state = (
            lambda key: pinned if key == "pinned" else unpinned
        )
        monkeypatch.setattr(
            gateway_run,
            "_load_gateway_runtime_config",
            lambda: {
                "agent": {
                    "service_tier": "default",
                    "service_tier_overrides": {_MATCH_MODEL: "flex"},
                }
            },
        )
        assert (
            runner._resolve_session_service_tier(
                session_key="pinned", model=_MATCH_MODEL
            )
            == "priority"
        )
        assert (
            runner._resolve_session_service_tier(
                session_key="other", model=_MATCH_MODEL
            )
            == "flex"
        )
        assert (
            runner._resolve_session_service_tier(
                session_key="other", model=_OTHER_MODEL
            )
            is None
        )

    def test_session_routing_uses_effective_model(self):
        import gateway.run as gateway_run

        runner = object.__new__(gateway_run.GatewayRunner)
        runner._provider_routing = _CONFIG["provider_routing"]
        matched = resolve_provider_routing_for_model(
            runner._provider_routing, _MATCH_MODEL
        )
        other = resolve_provider_routing_for_model(
            runner._provider_routing, _OTHER_MODEL
        )
        assert matched["only"] == ["google-ai-studio"]
        assert "models" not in matched
        assert other["only"] == ["foo"]
        assert "order" not in other


class TestTuiFactory:
    def test_factory_applies_overlay_and_pin_wins(self, monkeypatch):
        fake_runtime = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
            "credential_pool": None,
        }
        captured = {}

        def _fake_agent(*_a, **kwargs):
            ns = SimpleNamespace(**kwargs)
            captured.update(kwargs)
            captured["_agent"] = ns
            return ns

        import tui_gateway.server as server

        monkeypatch.setattr(
            server,
            "_load_cfg",
            lambda: {
                "model": {"default": _MATCH_MODEL, "provider": "openrouter"},
                "agent": {
                    "service_tier": "priority",
                    "service_tier_overrides": {_MATCH_MODEL: "flex"},
                    "system_prompt": "test",
                },
                "provider_routing": _CONFIG["provider_routing"],
            },
        )
        monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
        monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "compact")
        monkeypatch.setattr(server, "_load_enabled_toolsets", lambda _p=None: None)
        monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
        monkeypatch.setattr(server, "_load_fallback_model", lambda: None)
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_k: fake_runtime,
        )
        monkeypatch.setattr("run_agent.AIAgent", _fake_agent)

        agent = server._make_agent("sid-1", "key-1")
        assert captured["providers_allowed"] == ["google-ai-studio"]
        assert captured["service_tier"] == "flex"
        assert agent._service_tier_session_pinned is False
        assert agent._config_managed_routing_tier is True

        pinned = server._make_agent(
            "sid-2", "key-2", service_tier_override="priority"
        )
        assert captured["service_tier"] == "priority"
        assert pinned._service_tier_session_pinned is True

        explicit_normal = server._make_agent(
            "sid-3", "key-3", service_tier_override=""
        )
        # * "" is the TUI explicit-normal sentinel (not inherit / not None).
        assert captured["service_tier"] == ""
        assert explicit_normal._service_tier_session_pinned is True


class TestSwitchModelReResolve:
    def _make_fake_agent(self, model="openai/gpt-5", provider="openrouter"):
        agent = MagicMock()
        agent.model = model
        agent.provider = provider
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "test-key"
        agent._client_kwargs = {
            "api_key": "test-key",
            "base_url": "https://openrouter.ai/api/v1",
        }
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.reasoning_config = None
        agent.service_tier = "priority"
        agent._service_tier_session_pinned = False
        agent._config_managed_routing_tier = True
        agent._provider_routing_config = _CONFIG["provider_routing"]
        agent.providers_allowed = ["foo"]
        agent.providers_ignored = None
        agent.providers_order = None
        agent.provider_sort = "throughput"
        agent.provider_require_parameters = False
        agent.provider_data_collection = None
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._config_context_length = None
        agent._transport_cache = {}
        agent.context_compressor = None
        agent._cached_system_prompt = None
        agent._anthropic_api_key = ""
        agent._anthropic_base_url = None
        agent._is_anthropic_oauth = False
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._ensure_lmstudio_runtime_loaded = MagicMock(return_value=None)
        agent._lmstudio_load_was_unverified = MagicMock(return_value=False)
        agent._effective_lmstudio_context_length = MagicMock(return_value=None)
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._read_reasoning_echo_from_config = MagicMock(return_value=False)
        return agent

    def test_switch_recomputes_routing_and_tier(self):
        from agent.agent_runtime_helpers import switch_model

        agent = self._make_fake_agent()
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.providers_allowed == ["google-ai-studio"]
        assert agent.provider_sort == "throughput"
        assert agent.service_tier == "flex"

    def test_switch_does_not_clobber_session_pin(self):
        from agent.agent_runtime_helpers import switch_model

        agent = self._make_fake_agent()
        agent._service_tier_session_pinned = True
        agent.service_tier = "priority"
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier == "priority"
        assert agent._service_tier_session_pinned is True
        assert agent.providers_allowed == ["google-ai-studio"]


def _switch_result(new_model: str):
    from hermes_cli.model_switch import ModelSwitchResult

    return ModelSwitchResult(
        success=True,
        new_model=new_model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        api_mode="chat_completions",
        warning_message="",
        provider_label="OpenRouter",
        resolved_via_alias="",
        capabilities=None,
        model_info=None,
        is_global=False,
    )


def _bind_real_switch(agent):
    from agent.agent_runtime_helpers import switch_model as real_switch

    def _sm(
        new_model,
        new_provider,
        api_key="",
        base_url="",
        api_mode="",
        capabilities=None,
        **_kwargs,
    ):
        real_switch(
            agent,
            new_model,
            new_provider,
            api_key=api_key,
            base_url=base_url,
            api_mode=api_mode,
            capabilities=capabilities,
        )

    agent.switch_model = _sm
    return agent


class TestTuiLiveFastPinSurvivesSwitchModel:
    """Live config.set /fast must pin so a later /model does not clobber tier."""

    def test_live_fast_then_switch_keeps_pin_and_recomputes_routing(self, monkeypatch):
        from agent.agent_runtime_helpers import switch_model
        import tui_gateway.server as server

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        session = {"session_key": "k1", "agent": agent}
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}), \
                patch(
                    "hermes_cli.models.resolve_fast_mode_overrides",
                    return_value={"service_tier": "priority"},
                ):
            resp = server._methods["config.set"](
                "rid-1",
                {"key": "fast", "session_id": "s1", "value": "fast"},
            )
        assert resp["result"]["value"] == "fast"
        write_key.assert_not_called()
        assert agent._service_tier_session_pinned is True
        assert agent.service_tier == "priority"

        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier == "priority"
        assert agent._service_tier_session_pinned is True
        assert agent.providers_allowed == ["google-ai-studio"]

    def test_live_normal_pins_and_blocks_per_model_override(self):
        from agent.agent_runtime_helpers import switch_model
        import tui_gateway.server as server

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        session = {"session_key": "k1", "agent": agent}
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key") as write_key, \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}):
            resp = server._methods["config.set"](
                "rid-1",
                {"key": "fast", "session_id": "s1", "value": "normal"},
            )
        assert resp["result"]["value"] == "normal"
        write_key.assert_not_called()
        assert session["create_service_tier_override"] == ""
        assert agent._service_tier_session_pinned is True
        assert agent.service_tier is None

        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier is None
        assert agent._service_tier_session_pinned is True
        assert agent.providers_allowed == ["google-ai-studio"]

    def _slash_fast(self, command: str):
        import tui_gateway.server as server

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        session = {"session_key": "k1", "agent": agent}
        with patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}):
            warning = server._mirror_slash_side_effects("s1", session, command)
        assert warning == ""
        return agent, session

    def test_slash_fast_pins_priority_across_switch(self):
        from agent.agent_runtime_helpers import switch_model

        agent, _session = self._slash_fast("/fast on")
        assert agent.service_tier == "priority"
        assert agent._service_tier_session_pinned is True
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier == "priority"
        assert agent._service_tier_session_pinned is True

    def test_slash_normal_pins_and_blocks_per_model_override(self):
        from agent.agent_runtime_helpers import switch_model

        agent, _session = self._slash_fast("/fast normal")
        assert agent.service_tier is None
        assert agent._service_tier_session_pinned is True
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model=_MATCH_MODEL,
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier is None
        assert agent._service_tier_session_pinned is True


class TestCliModelSwitchSyncsShellForBackground:
    """After /model, shell routing/tier feed background AIAgent construction."""

    def _shell(self, agent):
        return SimpleNamespace(
            model=_OTHER_MODEL,
            provider="openrouter",
            requested_provider="openrouter",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            _explicit_api_key="test-key",
            _explicit_base_url="https://openrouter.ai/api/v1",
            conversation_history=[],
            _pending_model_switch_note="",
            _provider_routing_raw=_CONFIG["provider_routing"],
            _providers_only=["foo"],
            _providers_ignore=None,
            _providers_order=None,
            _provider_sort="throughput",
            _provider_require_params=False,
            _provider_data_collection=None,
            service_tier="priority",
            _service_tier_session_pinned=False,
            config={
                "agent": _CONFIG["agent"],
                "provider_routing": _CONFIG["provider_routing"],
            },
            agent=agent,
        )

    def test_model_switch_updates_shell_routing_for_background(self, monkeypatch):
        import cli as cli_mod

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        shell = self._shell(agent)
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
        monkeypatch.setattr(cli_mod, "save_config_value", lambda *_a, **_k: None)
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            cli_mod.HermesCLI._apply_model_switch_result(
                shell, _switch_result(_MATCH_MODEL), False
            )
        assert shell._providers_only == ["google-ai-studio"]
        assert shell._provider_sort == "throughput"
        assert agent.providers_allowed == ["google-ai-studio"]

    def test_model_switch_syncs_shell_service_tier(self, monkeypatch):
        import cli as cli_mod

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        shell = self._shell(agent)
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
        monkeypatch.setattr(cli_mod, "save_config_value", lambda *_a, **_k: None)
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            cli_mod.HermesCLI._apply_model_switch_result(
                shell, _switch_result(_MATCH_MODEL), False
            )
        assert shell.service_tier == "flex"
        assert agent.service_tier == "flex"

    def test_pinned_shell_tier_survives_model_switch(self, monkeypatch):
        import cli as cli_mod

        agent = _bind_real_switch(TestSwitchModelReResolve()._make_fake_agent())
        agent._service_tier_session_pinned = True
        agent.service_tier = "priority"
        shell = self._shell(agent)
        shell._service_tier_session_pinned = True
        shell.service_tier = "priority"
        fake_cfg = {
            "agent": _CONFIG["agent"],
            "provider_routing": _CONFIG["provider_routing"],
        }
        monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
        monkeypatch.setattr(cli_mod, "save_config_value", lambda *_a, **_k: None)
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            cli_mod.HermesCLI._apply_model_switch_result(
                shell, _switch_result(_MATCH_MODEL), False
            )
        assert shell.service_tier == "priority"
        assert agent.service_tier == "priority"
        assert shell._providers_only == ["google-ai-studio"]


class TestGatewayProfileScopedConfig:
    """Multiplex boundary: loaders honor ``_profile_runtime_scope``, not launch YAML.

    Full gateway multiplex e2e (worker pool + adapters) is not exercised here.
    """

    def test_two_profiles_resolve_distinct_routing_tier_and_escalation(self, tmp_path):
        from gateway.run import GatewayRunner, _profile_runtime_scope

        def _write(home, *, only, global_tier, override_tier, esc_enabled, threshold):
            home.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": {"default": _MATCH_MODEL, "provider": "openrouter"},
                "agent": {
                    "service_tier": global_tier,
                    "service_tier_overrides": {_MATCH_MODEL: override_tier},
                    "service_tier_escalation": {
                        "enabled": esc_enabled,
                        "ttft_threshold_seconds": threshold,
                        "consecutive_slow_requests": 2 if esc_enabled else 1,
                    },
                },
                "provider_routing": {
                    "sort": "throughput",
                    "only": [only],
                    "models": {_MATCH_MODEL: {"only": [only + "-studio"]}},
                },
            }
            (home / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

        profile_a = tmp_path / "alpha"
        profile_b = tmp_path / "beta"
        _write(
            profile_a,
            only="alpha-only",
            global_tier="priority",
            override_tier="flex",
            esc_enabled=True,
            threshold=3.5,
        )
        _write(
            profile_b,
            only="beta-only",
            global_tier="flex",
            override_tier="normal",
            esc_enabled=False,
            threshold=9.0,
        )

        with _profile_runtime_scope(profile_a):
            routing_a = GatewayRunner._load_provider_routing()
            tier_a = GatewayRunner._load_service_tier(_MATCH_MODEL)
            esc_a = GatewayRunner._load_service_tier_escalation()
        with _profile_runtime_scope(profile_b):
            routing_b = GatewayRunner._load_provider_routing()
            tier_b = GatewayRunner._load_service_tier(_MATCH_MODEL)
            esc_b = GatewayRunner._load_service_tier_escalation()

        resolved_a = resolve_provider_routing_for_model(routing_a, _MATCH_MODEL)
        resolved_b = resolve_provider_routing_for_model(routing_b, _MATCH_MODEL)
        assert resolved_a["only"] == ["alpha-only-studio"]
        assert resolved_b["only"] == ["beta-only-studio"]
        assert tier_a == "flex"
        assert tier_b is None
        assert esc_a.enabled is True
        assert esc_a.ttft_threshold_seconds == 3.5
        assert esc_a.consecutive_slow_requests == 2
        assert esc_b.enabled is False

    def test_multiplex_flag_uses_is_true(self):
        import gateway.run as gateway_run

        on = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True))
        off = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
        missing = SimpleNamespace(config=SimpleNamespace())
        assert gateway_run._multiplex_profiles_active(on) is True
        assert gateway_run._multiplex_profiles_active(off) is False
        assert gateway_run._multiplex_profiles_active(missing) is False


def _new_session_cli_stub(agent, *, model="gpt-5.4", provider="openai"):
    return SimpleNamespace(
        agent=agent,
        conversation_history=[],
        session_id="old-session",
        _session_db=None,
        _pending_title=None,
        _resumed=False,
        reasoning_config=None,
        _notify_session_boundary=MagicMock(),
        service_tier="priority",
        _service_tier_session_pinned=True,
        _pending_one_turn_model_restore=None,
        model=model,
        provider=provider,
        requested_provider=provider,
        api_key="k",
        base_url="https://api.openai.com/v1",
        api_mode="chat_completions",
    )


class TestRequestOverridesLifecycleSync:
    """Tier keys in request_overrides must follow pin/new/switch, not stick."""

    def test_cli_new_drops_session_pin_from_wire(self):
        from agent.chat_completion_helpers import _effective_request_overrides
        from cli import CLI_CONFIG, HermesCLI

        agent = SimpleNamespace(
            model="gpt-5.4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            service_tier="priority",
            request_overrides={"service_tier": "priority", "keep_me": True},
            _service_tier_session_pinned=True,
            reasoning_config=None,
            reset_session_state=MagicMock(),
        )
        stub = _new_session_cli_stub(agent)
        with patch.dict(
            CLI_CONFIG.setdefault("agent", {}),
            {"service_tier": "", "service_tier_overrides": {}},
        ), patch.dict(
            CLI_CONFIG,
            {"model": {"default": "gpt-5.4", "provider": "openai"}},
        ):
            HermesCLI.new_session(stub, silent=True)
        wire = _effective_request_overrides(agent)
        assert "service_tier" not in wire
        assert "speed" not in wire
        assert wire.get("keep_me") is True
        assert stub._service_tier_session_pinned is False
        assert agent._service_tier_session_pinned is False

    def test_cli_new_applies_global_flex_on_wire(self):
        from agent.chat_completion_helpers import _effective_request_overrides
        from cli import CLI_CONFIG, HermesCLI

        agent = SimpleNamespace(
            model="gpt-5.4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            service_tier="priority",
            request_overrides={"service_tier": "priority"},
            _service_tier_session_pinned=True,
            reasoning_config=None,
            reset_session_state=MagicMock(),
        )
        stub = _new_session_cli_stub(agent)
        with patch.dict(
            CLI_CONFIG.setdefault("agent", {}),
            {"service_tier": "flex", "service_tier_overrides": {}},
        ), patch.dict(
            CLI_CONFIG,
            {"model": {"default": "gpt-5.4", "provider": "openai"}},
        ):
            HermesCLI.new_session(stub, silent=True)
        wire = _effective_request_overrides(agent)
        assert wire.get("service_tier") == "flex"
        assert "speed" not in wire

    def test_tui_config_set_fast_then_slash_normal_clears_wire(self):
        from agent.chat_completion_helpers import _effective_request_overrides
        import tui_gateway.server as server

        agent = SimpleNamespace(
            reasoning_config=None,
            service_tier=None,
            request_overrides={"keep_me": 1},
            model="gpt-5.4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            session_id="sess-key",
        )
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key"), \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}):
            server._methods["config.set"](
                "rid-1",
                {"key": "fast", "session_id": "s1", "value": "fast"},
            )
            assert _effective_request_overrides(agent).get("service_tier") == "priority"
            server._mirror_slash_side_effects("s1", session, "/fast normal")
        wire = _effective_request_overrides(agent)
        assert "service_tier" not in wire
        assert "speed" not in wire
        assert wire.get("keep_me") == 1
        assert agent._service_tier_session_pinned is True

    def test_tui_slash_fast_after_config_set_normal_sets_wire(self):
        from agent.chat_completion_helpers import _effective_request_overrides
        import tui_gateway.server as server

        agent = SimpleNamespace(
            reasoning_config=None,
            service_tier="priority",
            request_overrides={"service_tier": "priority"},
            model="gpt-5.4",
            provider="openai",
            base_url="https://api.openai.com/v1",
            session_id="sess-key",
        )
        session = {"session_key": "k1", "agent": agent}
        with patch.dict(server._sessions, {"s1": session}, clear=False), \
                patch.object(server, "_write_config_key"), \
                patch.object(server, "_persist_live_session_runtime"), \
                patch.object(server, "_emit"), \
                patch.object(server, "_session_info", return_value={}):
            server._methods["config.set"](
                "rid-1",
                {"key": "fast", "session_id": "s1", "value": "normal"},
            )
            assert "service_tier" not in _effective_request_overrides(agent)
            server._mirror_slash_side_effects("s1", session, "/fast fast")
        wire = _effective_request_overrides(agent)
        assert wire.get("service_tier") == "priority"
        assert "speed" not in wire

    def test_switch_model_rewrites_anthropic_speed_to_openrouter_service_tier(self):
        from agent.agent_runtime_helpers import switch_model
        from agent.chat_completion_helpers import _effective_request_overrides

        agent = TestSwitchModelReResolve()._make_fake_agent(
            model="claude-opus-4-6",
            provider="anthropic",
        )
        agent.base_url = "https://api.anthropic.com"
        agent.api_mode = "anthropic_messages"
        agent.service_tier = "priority"
        agent._service_tier_session_pinned = True
        agent.request_overrides = {"speed": "fast", "keep_me": True}
        fake_cfg = {
            "agent": {"service_tier": "flex"},
            "provider_routing": {},
        }
        with patch("hermes_cli.config.load_config", return_value=fake_cfg):
            switch_model(
                agent,
                new_model="openai/gpt-5",
                new_provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_mode="chat_completions",
            )
        assert agent.service_tier == "priority"
        wire = _effective_request_overrides(agent)
        assert wire.get("service_tier") == "priority"
        assert "speed" not in wire
        assert wire.get("keep_me") is True


def _resync_agent(**kwargs):
    """Plain agent object for ``resync_per_model_routing_and_tier`` (not MagicMock)."""
    from agent.service_tier_escalation import bind_service_tier_escalation

    agent = SimpleNamespace(
        model=kwargs.get("model", _MATCH_MODEL),
        provider=kwargs.get("provider", "openrouter"),
        base_url=kwargs.get("base_url", "https://openrouter.ai/api/v1"),
        api_mode=kwargs.get("api_mode", "chat_completions"),
        service_tier=kwargs.get("service_tier", "priority"),
        _service_tier_session_pinned=kwargs.get("pinned", False),
        _config_managed_routing_tier=kwargs.get("managed", True),
        _provider_routing_config=kwargs.get(
            "routing", _CONFIG["provider_routing"]
        ),
        _agent_config=kwargs.get("agent_cfg", _CONFIG["agent"]),
        request_overrides=dict(kwargs.get("request_overrides", {})),
        providers_allowed=["stale"],
        providers_ignored=None,
        providers_order=None,
        provider_sort="latency",
        provider_require_parameters=False,
        provider_data_collection=None,
    )
    bind_service_tier_escalation(
        agent,
        {
            "enabled": True,
            "ttft_threshold_seconds": 8.0,
            "consecutive_slow_requests": 2,
        },
    )
    agent._service_tier_escalation.streak = 3
    agent._service_tier_escalation.effective_tier = None
    return agent


class TestFallbackResyncPerModel:
    """Automatic fallback/restore re-apply overlays without switch_model."""

    def test_resync_applies_fallback_overlay_and_keeps_escalation(self):
        from agent.agent_runtime_helpers import resync_per_model_routing_and_tier

        agent = _resync_agent(model=_MATCH_MODEL)
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == ["google-ai-studio"]
        assert agent.service_tier == "flex"

        state = agent._service_tier_escalation
        state.base_tier = "flex"
        state.effective_tier = None
        state.climbed_rungs = 1
        state.streak = 3

        agent.model = _OTHER_MODEL
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == ["foo"]
        assert agent.service_tier == "priority"
        assert state.streak == 3
        assert state.base_tier == "priority"
        assert state.effective_tier == "priority"
        assert state.climbed_rungs == 1

        agent.model = _MATCH_MODEL
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == ["google-ai-studio"]
        assert agent.service_tier == "flex"
        assert state.streak == 3
        assert state.base_tier == "flex"
        assert state.effective_tier is None
        assert state.climbed_rungs == 1

    def test_pinned_tier_survives_fallback_wire_form_rewritten(self):
        from agent.agent_runtime_helpers import resync_per_model_routing_and_tier
        from agent.chat_completion_helpers import _effective_request_overrides

        agent = _resync_agent(
            model=_MATCH_MODEL,
            pinned=True,
            service_tier="priority",
            request_overrides={"service_tier": "priority", "keep_me": True},
        )
        agent.model = _OTHER_MODEL
        resync_per_model_routing_and_tier(agent)
        assert agent.service_tier == "priority"
        assert agent._service_tier_session_pinned is True
        assert agent.providers_allowed == ["foo"]
        wire = _effective_request_overrides(agent)
        assert wire.get("service_tier") == "priority"
        assert wire.get("keep_me") is True

        agent.model = "claude-opus-4-6"
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"
        agent.api_mode = "anthropic_messages"
        resync_per_model_routing_and_tier(agent)
        assert agent.service_tier == "priority"
        wire = _effective_request_overrides(agent)
        assert "service_tier" not in wire or wire.get("service_tier") != "flex"
        assert wire.get("keep_me") is True

    def test_non_openrouter_fallback_does_not_emit_provider_prefs(self):
        from agent.agent_runtime_helpers import resync_per_model_routing_and_tier
        from agent.chat_completion_helpers import _provider_preferences_for_agent
        from providers import get_provider_profile

        agent = _resync_agent(model=_MATCH_MODEL)
        resync_per_model_routing_and_tier(agent)
        agent.model = "claude-opus-4-6"
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"
        resync_per_model_routing_and_tier(agent)
        prefs = _provider_preferences_for_agent(agent)
        profile = get_provider_profile("anthropic")
        extra = {}
        if profile is not None:
            extra = profile.build_extra_body(
                session_id="s",
                provider_preferences=prefs or None,
                model=agent.model,
                base_url=agent.base_url,
                reasoning_config=None,
            ) or {}
        assert "provider" not in extra
        assert extra.get("only") is None

    def test_try_activate_fallback_and_restore_swap_overlays(self):
        from agent.agent_runtime_helpers import (
            resync_per_model_routing_and_tier,
            restore_primary_runtime,
        )
        from agent.chat_completion_helpers import try_activate_fallback
        from run_agent import AIAgent

        _write_config()
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                model=_MATCH_MODEL,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[
                    {
                        "provider": "openrouter",
                        "model": _OTHER_MODEL,
                        "base_url": "https://openrouter.ai/api/v1",
                    }
                ],
            )
        agent._config_managed_routing_tier = True
        agent.client = MagicMock()
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == ["google-ai-studio"]
        assert agent.service_tier == "flex"
        assert isinstance(agent._agent_config, dict)
        agent._service_tier_escalation.streak = 4
        agent._service_tier_escalation.effective_tier = None
        agent._service_tier_escalation.climbed_rungs = 1
        mock_client = MagicMock()
        mock_client.base_url = "https://openrouter.ai/api/v1"
        mock_client.api_key = "fb-key"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, _OTHER_MODEL),
        ):
            assert try_activate_fallback(agent) is True
        assert agent.model == _OTHER_MODEL
        assert agent.providers_allowed == ["foo"]
        assert agent.service_tier == "priority"
        assert agent._service_tier_escalation.streak == 4
        assert agent._service_tier_escalation.effective_tier == "priority"
        assert agent._service_tier_escalation.climbed_rungs == 1

        assert restore_primary_runtime(agent) is True
        assert agent.model == _MATCH_MODEL
        assert agent.providers_allowed == ["google-ai-studio"]
        assert agent.service_tier == "flex"
        assert agent._service_tier_escalation.streak == 4
        assert agent._service_tier_escalation.effective_tier is None
        assert agent._service_tier_escalation.climbed_rungs == 1

    def test_resync_skips_agent_without_provenance_opt_in(self):
        from agent.agent_runtime_helpers import resync_per_model_routing_and_tier

        agent = _resync_agent(managed=False, service_tier="priority")
        agent.providers_allowed = ["constructor-only"]
        agent.request_overrides = {"service_tier": "priority", "keep_me": True}
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == ["constructor-only"]
        assert agent.service_tier == "priority"
        assert agent.request_overrides == {"service_tier": "priority", "keep_me": True}

    def test_programmatic_agent_keeps_constructor_overrides_on_fallback(self):
        from agent.agent_runtime_helpers import (
            resync_per_model_routing_and_tier,
            restore_primary_runtime,
        )
        from agent.chat_completion_helpers import try_activate_fallback
        from run_agent import AIAgent

        _write_config()
        explicit_allowed = ["constructor-only"]
        explicit_overrides = {"service_tier": "priority", "keep_me": True}
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                model=_MATCH_MODEL,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                service_tier="priority",
                providers_allowed=list(explicit_allowed),
                request_overrides=dict(explicit_overrides),
                fallback_model=[
                    {
                        "provider": "openrouter",
                        "model": _OTHER_MODEL,
                        "base_url": "https://openrouter.ai/api/v1",
                    }
                ],
            )
        assert agent._config_managed_routing_tier is False
        agent.client = MagicMock()
        resync_per_model_routing_and_tier(agent)
        assert agent.providers_allowed == explicit_allowed
        assert agent.service_tier == "priority"
        assert agent.request_overrides.get("keep_me") is True
        assert agent.request_overrides.get("service_tier") == "priority"

        mock_client = MagicMock()
        mock_client.base_url = "https://openrouter.ai/api/v1"
        mock_client.api_key = "fb-key"
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(mock_client, _OTHER_MODEL),
        ):
            assert try_activate_fallback(agent) is True
        assert agent.model == _OTHER_MODEL
        assert agent.providers_allowed == explicit_allowed
        assert agent.service_tier == "priority"
        assert agent.request_overrides.get("keep_me") is True
        assert agent.request_overrides.get("service_tier") == "priority"

        assert restore_primary_runtime(agent) is True
        assert agent.model == _MATCH_MODEL
        assert agent.providers_allowed == explicit_allowed
        assert agent.service_tier == "priority"
        assert agent.request_overrides.get("keep_me") is True
        assert agent.request_overrides.get("service_tier") == "priority"


class _RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


def _conversation_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


class TestFallbackRebaseInRunConversation:
    """Primary flex + fallback per-model priority must rebase the wire snapshot."""

    def test_fallback_request_carries_priority_restore_returns_to_flex(self):
        from run_agent import AIAgent

        # * Stay on chat_completions — openai/gpt-5 would flip the
        # fallback onto the Codex Responses path.
        fallback_model = "qwen/qwen3.8-27b"
        _write_config(
            {
                "agent": {
                    "service_tier": "flex",
                    "service_tier_overrides": {
                        _MATCH_MODEL: "flex",
                        fallback_model: "priority",
                    },
                    "service_tier_escalation": {
                        "enabled": True,
                        "ttft_threshold_seconds": 8.0,
                        "consecutive_slow_requests": 1,
                    },
                },
            }
        )
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                provider="openrouter",
                model=_MATCH_MODEL,
                service_tier="flex",
                service_tier_escalation={
                    "enabled": True,
                    "ttft_threshold_seconds": 8.0,
                    "consecutive_slow_requests": 1,
                },
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                fallback_model=[
                    {
                        "provider": "openrouter",
                        "model": fallback_model,
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_mode": "chat_completions",
                    }
                ],
            )
        agent._config_managed_routing_tier = True
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._api_max_retries = 1

        captured = []

        def fake_api_call(api_kwargs, on_first_delta=None, **_kwargs):
            captured.append(
                {
                    "model": agent.model,
                    "service_tier": api_kwargs.get("service_tier"),
                }
            )
            if agent.model == _MATCH_MODEL and len(captured) == 1:
                raise _RateLimitError()
            return _conversation_response("ok")

        mock_fb = MagicMock()
        mock_fb.base_url = "https://openrouter.ai/api/v1"
        mock_fb.api_key = "fb-key"
        mock_fb._custom_headers = None
        mock_fb.default_headers = None

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_interruptible_streaming_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch("agent.conversation_loop.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb, fallback_model),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            first = agent.run_conversation("hello")
            assert first["completed"] is True
            assert len(captured) >= 2
            assert captured[0]["model"] == _MATCH_MODEL
            assert captured[0]["service_tier"] == "flex"
            assert captured[1]["model"] == fallback_model
            assert captured[1]["service_tier"] == "priority"

            # * 429 arms ``_rate_limited_until`` so the next turn stays on
            # fallback until the cooldown elapses. Expire it without sleep.
            agent._rate_limited_until = 0
            second = agent.run_conversation("hello again")
            assert second["completed"] is True
            assert captured[-1]["model"] == _MATCH_MODEL
            assert captured[-1]["service_tier"] == "flex"


_OR_FALLBACK_MODEL = "qwen/qwen3.8-27b"

_BG_OR_CONFIG = {
    "agent": {
        "service_tier": "priority",
        "service_tier_overrides": {
            _MATCH_MODEL: "flex",
            _OR_FALLBACK_MODEL: "priority",
        },
        "service_tier_escalation": {
            "enabled": True,
            "ttft_threshold_seconds": 8.0,
            "consecutive_slow_requests": 1,
        },
    },
    "provider_routing": {
        "sort": "throughput",
        "only": ["foo"],
        "models": {
            _MATCH_MODEL: {"only": ["google-ai-studio"]},
            _OR_FALLBACK_MODEL: {"only": ["reka"]},
        },
    },
}

_BG_FALLBACK_CHAIN = [
    {
        "provider": "openrouter",
        "model": _OR_FALLBACK_MODEL,
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
]


def _assert_bg_provenance_and_snapshot(agent):
    assert agent._config_managed_routing_tier is True
    assert agent._block_service_tier_escalation is True
    assert isinstance(agent._agent_config, dict)
    assert agent._agent_config["service_tier_overrides"][_MATCH_MODEL] == "flex"
    assert isinstance(agent._provider_routing_config, dict)
    assert agent._provider_routing_config["models"][_MATCH_MODEL]["only"] == [
        "google-ai-studio"
    ]
    assert agent._service_tier_escalation.enabled is True


def _assert_request_routing_tier(agent, *, only, tier, model=None):
    from agent.chat_completion_helpers import (
        _effective_request_overrides,
        _provider_preferences_for_agent,
    )

    if model is not None:
        assert agent.model == model
    assert agent.providers_allowed == only
    assert agent.service_tier == tier
    assert _provider_preferences_for_agent(agent).get("only") == only
    assert _effective_request_overrides(agent).get("service_tier") == tier


def _force_fallback_restore_and_block_escalation(agent):
    from agent.agent_runtime_helpers import restore_primary_runtime
    from agent.chat_completion_helpers import try_activate_fallback
    from agent.service_tier_escalation import (
        TtftObservation,
        accept_logical_request,
        begin_logical_request,
        escalation_is_active,
        finish_request_ttft,
    )

    agent.client = MagicMock()
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "fb-key"
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(mock_client, _OR_FALLBACK_MODEL),
    ):
        assert try_activate_fallback(agent) is True
    _assert_request_routing_tier(
        agent,
        only=["reka"],
        tier="priority",
        model=_OR_FALLBACK_MODEL,
    )
    assert restore_primary_runtime(agent) is True
    _assert_request_routing_tier(
        agent,
        only=["google-ai-studio"],
        tier="flex",
        model=_MATCH_MODEL,
    )

    assert escalation_is_active(agent) is False
    obs = TtftObservation(clock=lambda: 0.0)
    obs.t_send = 0.0
    obs.t_first = 20.0
    obs.open_count = 1
    begin_logical_request(agent)
    finish_request_ttft(agent, obs)
    accept_logical_request(agent)
    assert agent._service_tier_escalation.effective_tier == "flex"
    _assert_request_routing_tier(
        agent,
        only=["google-ai-studio"],
        tier="flex",
        model=_MATCH_MODEL,
    )


class TestCliBackgroundManagedRoutingAndHardGate:
    """CLI /bg agents are config-managed and hard-gated off the TTFT ladder."""

    def _cli_stub(self):
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli._background_task_counter = 0
        cli._background_tasks = {}
        cli._ensure_runtime_credentials = MagicMock(return_value=True)
        cli._resolve_turn_agent_config = MagicMock(
            return_value={
                "model": _MATCH_MODEL,
                "runtime": {
                    "api_key": "test-key",
                    "base_url": "https://openrouter.ai/api/v1",
                    "provider": "openrouter",
                    "api_mode": "chat_completions",
                },
                "request_overrides": None,
            }
        )
        cli.max_turns = 2
        cli.enabled_toolsets = []
        cli._session_db = None
        cli.reasoning_config = {}
        cli.service_tier = "flex"
        cli._service_tier_escalation_cfg = {
            "enabled": True,
            "ttft_threshold_seconds": 8.0,
            "consecutive_slow_requests": 1,
        }
        cli._providers_only = ["google-ai-studio"]
        cli._providers_ignore = None
        cli._providers_order = None
        cli._provider_sort = "throughput"
        cli._provider_require_params = False
        cli._provider_data_collection = None
        cli._openrouter_min_coding_score = None
        cli._fallback_model = list(_BG_FALLBACK_CHAIN)
        cli._agent_running = False
        cli._spinner_text = ""
        cli.bell_on_complete = False
        cli.final_response_markdown = "strip"
        cli._app = None
        return cli

    def test_bg_agent_resyncs_fallback_and_blocks_escalation(self, monkeypatch):
        import cli as cli_mod
        from run_agent import AIAgent

        _write_config(_BG_OR_CONFIG)
        captured = {}

        def _capture_run(self, *args, **kwargs):
            captured["agent"] = self
            return {"final_response": "", "completed": True, "messages": []}

        cli = self._cli_stub()
        monkeypatch.setattr(cli_mod, "_cprint", lambda *_a, **_k: None)
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch.object(AIAgent, "run_conversation", _capture_run),
            patch.object(cli_mod, "ChatConsole"),
        ):
            cli._handle_background_command("/bg summarize the inbox")
            for thread in list(cli._background_tasks.values()):
                thread.join(timeout=10)

        agent = captured.get("agent")
        assert agent is not None
        _assert_bg_provenance_and_snapshot(agent)
        _assert_request_routing_tier(
            agent,
            only=["google-ai-studio"],
            tier="flex",
            model=_MATCH_MODEL,
        )
        _force_fallback_restore_and_block_escalation(agent)


class TestTuiBackgroundManagedRoutingAndHardGate:
    """TUI prompt.background agents are config-managed and hard-gated."""

    def test_prompt_background_resyncs_fallback_and_blocks_escalation(
        self, monkeypatch
    ):
        import tui_gateway.server as server
        from run_agent import AIAgent

        _write_config(_BG_OR_CONFIG)
        server._cfg_cache = None
        server._cfg_mtime = None
        server._cfg_path = None
        monkeypatch.setattr(server, "_hermes_home", get_hermes_home())

        parent = SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=None,
            model=_MATCH_MODEL,
            enabled_toolsets=[],
            ephemeral_system_prompt=None,
            providers_allowed=["google-ai-studio"],
            providers_ignored=None,
            providers_order=None,
            provider_sort="throughput",
            provider_require_parameters=False,
            provider_data_collection=None,
            openrouter_min_coding_score=None,
            reasoning_config={},
            service_tier="flex",
            request_overrides={},
            _fallback_chain=list(_BG_FALLBACK_CHAIN),
        )
        session = {
            "agent": parent,
            "session_key": "k1",
            "profile_home": None,
            "cwd": str(get_hermes_home()),
        }
        captured = {}

        def _capture_run(self, *args, **kwargs):
            captured["agent"] = self
            return {"final_response": "ok", "completed": True, "messages": []}

        class _InlineThread:
            def __init__(self, target=None, daemon=None, **_kwargs):
                self._target = target

            def start(self):
                if self._target is not None:
                    self._target()

        monkeypatch.setattr(server, "_get_db", lambda: MagicMock())
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch.object(AIAgent, "run_conversation", _capture_run),
            patch("tui_gateway.server.threading.Thread", _InlineThread),
            patch.object(server, "_sess", return_value=(session, None)),
            patch.object(server, "_emit"),
        ):
            resp = server._methods["prompt.background"](
                "rid-1",
                {"text": "summarize the inbox", "session_id": "s1"},
            )

        assert resp.get("result", {}).get("task_id")
        agent = captured.get("agent")
        assert agent is not None
        _assert_bg_provenance_and_snapshot(agent)
        _assert_request_routing_tier(
            agent,
            only=["google-ai-studio"],
            tier="flex",
            model=_MATCH_MODEL,
        )
        _force_fallback_restore_and_block_escalation(agent)
