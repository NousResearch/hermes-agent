"""Contract-тесты безопасного MoA benchmark."""

import json
from types import SimpleNamespace

import pytest

import agent.moa_loop as moa_loop
from scripts import moa_benchmark


def _slot(provider: str, model: str) -> dict[str, str]:
    return {"provider": provider, "model": model}


def _preset(
    *,
    references: list[dict[str, str]] | None = None,
    budget: float | None = None,
    auto_routes: dict[str, str] | None = None,
) -> dict:
    return {
        "enabled": True,
        "reference_models": references or [],
        "aggregator": _slot("openai-codex", "gpt-test"),
        "fallback_aggregators": [],
        "max_reference_cost_usd": budget,
        "auto_routes": auto_routes or {},
    }


def _config(*, save_traces: bool = False) -> dict:
    grok = _slot("xai-oauth", "grok-test")
    glm = _slot("openrouter", "glm-test")
    routes = {
        "fast": "fast",
        "balanced": "balanced",
        "research": "research",
        "code_heavy": "code-heavy",
        "max": "max",
    }
    return {
        "default_preset": "auto",
        "active_preset": "auto",
        "save_traces": save_traces,
        "presets": {
            "auto": _preset(auto_routes=routes),
            "fast": _preset(),
            "balanced": _preset(references=[grok], budget=0.10),
            "research": _preset(references=[grok, glm], budget=0.40),
            "code-heavy": _preset(references=[glm, grok], budget=0.35),
            "max": _preset(references=[grok, glm], budget=0.75),
        },
    }


def test_live_preflight_accepts_three_cases_at_budget_limit():
    result = moa_benchmark.validate_live_preflight(
        _config(),
        "auto",
        3,
        reference_budget_usd=0.75,
    )
    assert result["planned_reference_budget_usd"] == 0.75
    assert [case["resolved_preset"] for case in result["cases"]] == [
        "fast",
        "research",
        "code-heavy",
    ]


def test_live_preflight_rejects_total_budget_before_provider_call():
    with pytest.raises(ValueError, match="exceeds limit"):
        moa_benchmark.validate_live_preflight(
            _config(),
            "auto",
            4,
            reference_budget_usd=0.75,
        )


def test_live_preflight_rejects_enabled_traces():
    with pytest.raises(ValueError, match="save_traces=false"):
        moa_benchmark.validate_live_preflight(_config(save_traces=True), "auto", 3)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -0.01])
def test_live_preflight_rejects_invalid_budget_limit(limit):
    with pytest.raises(ValueError, match="finite and non-negative"):
        moa_benchmark.validate_live_preflight(
            _config(),
            "auto",
            3,
            reference_budget_usd=limit,
        )


def test_live_preflight_rejects_unbounded_reference_preset():
    config = _config()
    config["presets"]["research"]["max_reference_cost_usd"] = None
    with pytest.raises(ValueError, match="no max_reference_cost_usd"):
        moa_benchmark.validate_live_preflight(config, "auto", 3)


def test_live_result_does_not_contain_prompt_or_response(monkeypatch):
    secret_response = "response-body-must-not-be-persisted"

    class FakeCompletions:
        def __init__(self, _preset_name: str):
            self.last_runtime_status = {"degraded": False}

        def create(self, **_kwargs):
            message = SimpleNamespace(content=secret_response)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(moa_loop, "MoAChatCompletions", FakeCompletions)
    result = moa_benchmark.run_live(_config(), "auto", 1)
    rendered = json.dumps(result)
    assert secret_response not in rendered
    assert moa_benchmark.CASES[0][1] not in rendered
    assert result["results"][0]["response_chars"] == len(secret_response)
    assert result["results"][0]["response_sha256"]


def test_main_budget_failure_does_not_start_live_calls(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider path must not start")

    monkeypatch.setattr(moa_benchmark, "reload_env", lambda: None)
    monkeypatch.setattr(moa_benchmark, "load_config", lambda: {"moa": _config()})
    monkeypatch.setattr(moa_benchmark, "run_live", fail_if_called)
    with pytest.raises(SystemExit) as exc_info:
        moa_benchmark.main([
            "--live",
            "--confirm-live",
            "--max-cases",
            "4",
        ])
    assert exc_info.value.code == 2
    assert called is False
