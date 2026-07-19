from types import SimpleNamespace
from typing import Any, cast

from run_agent import AIAgent


def _agent(load_mode="explicit"):
    return SimpleNamespace(
        provider="lmstudio",
        model="test/model",
        base_url="http://127.0.0.1:1234/v1",
        api_key="",
        lmstudio_load_mode=load_mode,
        _config_context_length=None,
        context_compressor=None,
        api_mode="chat_completions",
    )


def test_lmstudio_jit_load_mode_skips_explicit_preload(monkeypatch):
    calls = []

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return 64000

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, _agent("jit")))

    assert calls == []


def test_lmstudio_explicit_load_mode_preserves_preload(monkeypatch):
    calls = []

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return 64000

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, _agent("explicit")))

    assert len(calls) == 1
    assert calls[0][0][:3] == ("test/model", "http://127.0.0.1:1234/v1", "")
    # No _config_context_length set → pass None so LM Studio uses its stored
    # per-model config / global default. The previous behavior (64000 floor)
    # was the root cause of the 64K override bug.
    assert calls[0][0][3] is None


def test_lmstudio_explicit_load_mode_honors_explicit_config_override(monkeypatch):
    # When the user has set model.context_length in config.yaml, that value
    # IS forwarded to ensure_lmstudio_model_loaded — the explicit override
    # path still works.
    calls = []

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return 262144

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    agent = _agent("explicit")
    agent._config_context_length = 262144
    AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, agent))

    assert len(calls) == 1
    assert calls[0][0][3] == 262144


def test_missing_lmstudio_load_mode_defaults_to_explicit(monkeypatch):
    calls = []
    agent = _agent()
    delattr(agent, "lmstudio_load_mode")

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return 64000

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, agent))

    assert len(calls) == 1
    # Same contract: no _config_context_length → pass None
    assert calls[0][0][3] is None
