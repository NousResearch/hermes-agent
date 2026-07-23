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
    assert calls[0][0][3] == 64000


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
