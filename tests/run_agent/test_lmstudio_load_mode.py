from types import SimpleNamespace
from typing import Any, cast

from hermes_cli.models import LMStudioLoadResult
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
        return LMStudioLoadResult(64_000)

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    result = AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, _agent("jit")))

    assert result is None
    assert calls == []


def test_lmstudio_explicit_load_mode_passes_no_override_as_none(monkeypatch):
    calls = []

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return LMStudioLoadResult(96_000, load_attempted=True)

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    result = AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, _agent("explicit")))

    assert result.context_length == 96_000
    assert len(calls) == 1
    assert calls[0][0][:3] == ("test/model", "http://127.0.0.1:1234/v1", "")
    assert calls[0][0][3] is None
    assert calls[0][1]["return_load_result"] is True


def test_missing_lmstudio_load_mode_defaults_to_explicit(monkeypatch):
    calls = []
    agent = _agent()
    delattr(agent, "lmstudio_load_mode")

    def fake_ensure(*args, **kwargs):
        calls.append((args, kwargs))
        return LMStudioLoadResult(64_000)

    monkeypatch.setattr("hermes_cli.models.ensure_lmstudio_model_loaded", fake_ensure)

    AIAgent._ensure_lmstudio_runtime_loaded(cast(Any, agent))

    assert len(calls) == 1


def test_explicit_budget_below_loaded_runtime_limits_effective_context():
    result = AIAgent._effective_lmstudio_context_length(
        80_000,
        LMStudioLoadResult(120_000),
    )

    assert result == 80_000


def test_attempted_unverified_load_has_no_effective_context():
    result = AIAgent._effective_lmstudio_context_length(
        100_000,
        LMStudioLoadResult(None, load_attempted=True),
    )

    assert result is None
