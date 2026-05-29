"""Regression tests for hermes -z oneshot mode."""

import sys
from types import SimpleNamespace


class _FakeAgent:
    captured_kwargs = {}

    def __init__(self, **kwargs):
        type(self).captured_kwargs = kwargs
        self.suppress_status_output = False
        self.stream_delta_callback = object()
        self.tool_gen_callback = object()

    def chat(self, prompt):
        return f"response to {prompt}"


def _patch_oneshot_dependencies(monkeypatch, cfg):
    import hermes_cli.config as config_mod
    import hermes_cli.models as models_mod
    import hermes_cli.oneshot as oneshot_mod
    import hermes_cli.runtime_provider as runtime_provider_mod
    import hermes_cli.tools_config as tools_config_mod

    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(models_mod, "detect_provider_for_model", lambda model, provider: None)
    monkeypatch.setattr(
        runtime_provider_mod,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "test-provider",
            "api_mode": "chat",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(tools_config_mod, "_get_platform_tools", lambda cfg, platform: set())
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setattr(oneshot_mod, "get_fallback_chain", lambda cfg: [])
    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=_FakeAgent))

    return oneshot_mod


def test_oneshot_passes_configured_system_prompt(monkeypatch):
    cfg = {
        "model": {"default": "test-model", "provider": "test-provider"},
        "agent": {"system_prompt": "You are Alt Brian."},
    }
    monkeypatch.delenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", raising=False)
    oneshot_mod = _patch_oneshot_dependencies(monkeypatch, cfg)

    assert oneshot_mod._run_agent("hello") == "response to hello"

    assert _FakeAgent.captured_kwargs["ephemeral_system_prompt"] == "You are Alt Brian."


def test_oneshot_env_system_prompt_overrides_config(monkeypatch):
    cfg = {
        "model": {"default": "test-model", "provider": "test-provider"},
        "agent": {"system_prompt": "Config prompt"},
    }
    monkeypatch.setenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "Env prompt")
    oneshot_mod = _patch_oneshot_dependencies(monkeypatch, cfg)

    oneshot_mod._run_agent("hello")

    assert _FakeAgent.captured_kwargs["ephemeral_system_prompt"] == "Env prompt"
