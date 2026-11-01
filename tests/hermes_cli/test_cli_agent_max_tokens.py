"""The CLI agent honours ``model.max_tokens`` from config.

The CLI build path passed no output cap to ``AIAgent``, so a configured
``model.max_tokens`` only applied when the provider happened to enforce its own
limit (and never on custom OpenAI-compatible endpoints). ``_init_agent`` now
reads ``model.max_tokens`` and forwards it, matching the gateway/desktop paths.
"""

import cli
from hermes_cli import config as hermes_config


class _DummyAgent:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


def _runtime(**kwargs):
    return {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "source": "env/config",
    }


def _init_shell_with_model_cfg(monkeypatch, model_cfg):
    monkeypatch.setattr("run_agent.AIAgent", _DummyAgent)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )
    real_load = hermes_config.load_config

    def _load(*args, **kwargs):
        cfg = dict(real_load(*args, **kwargs) or {})
        model = dict(cfg.get("model") or {})
        model.update(model_cfg)
        cfg["model"] = model
        return cfg

    monkeypatch.setattr(hermes_config, "load_config", _load)

    shell = cli.HermesCLI(model="gpt-5", compact=True, max_turns=1)
    assert shell._init_agent() is True
    return shell


def test_init_agent_forwards_configured_max_tokens(monkeypatch):
    shell = _init_shell_with_model_cfg(monkeypatch, {"max_tokens": 4242})

    assert shell.agent.kwargs["max_tokens"] == 4242


def test_init_agent_passes_none_when_max_tokens_unset(monkeypatch):
    shell = _init_shell_with_model_cfg(monkeypatch, {})

    assert shell.agent.kwargs["max_tokens"] is None
