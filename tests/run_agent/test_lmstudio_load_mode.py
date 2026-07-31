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






def test_explicit_budget_below_loaded_runtime_limits_effective_context():
    result = AIAgent._effective_lmstudio_context_length(
        80_000,
        LMStudioLoadResult(120_000),
    )

    assert result == 80_000


