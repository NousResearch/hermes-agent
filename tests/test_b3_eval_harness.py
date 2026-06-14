import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_harness.b3 as b3
from eval_harness.b3 import (
    EvalAttempt,
    EvalTask,
    VeraFinding,
    load_fixture,
    main,
    score_eval_suite,
)


def test_scores_pass_metrics_cost_latency_and_vera_precision_recall():
    tasks = [
        EvalTask(
            task_id="bob-fix-1",
            agent="bob",
            attempts=[
                EvalAttempt(passed=True, cost_usd=0.10, latency_ms=1200),
                EvalAttempt(passed=False, cost_usd=0.20, latency_ms=1800),
            ],
        ),
        EvalTask(
            task_id="steve-fix-1",
            agent="steve",
            attempts=[
                EvalAttempt(passed=False, cost_usd=0.05, latency_ms=900),
                EvalAttempt(passed=True, cost_usd=0.15, latency_ms=1100),
            ],
        ),
        EvalTask(
            task_id="vera-review-1",
            agent="vera",
            attempts=[EvalAttempt(passed=True, cost_usd=0.03, latency_ms=600)],
            expected_findings=[
                VeraFinding(id="missing-test", severity="high"),
                VeraFinding(id="secret-log", severity="critical"),
            ],
            reported_findings=[
                VeraFinding(id="missing-test", severity="high"),
                VeraFinding(id="style-nit", severity="low"),
            ],
        ),
    ]

    summary = score_eval_suite(tasks, k=2)

    assert summary["overall"]["task_count"] == 3
    assert summary["overall"]["pass@1"] == pytest.approx(2 / 3)
    assert summary["overall"]["pass^2"] == 1.0
    assert summary["overall"]["cost_usd"] == 0.53
    assert summary["overall"]["cost"] == {
        "usd": 0.53,
        "known_usd": 0.53,
        "known_attempts": 5,
        "unknown_attempts": 0,
        "actual_attempts": 5,
        "estimated_attempts": 0,
        "included_attempts": 0,
    }
    assert summary["overall"]["latency_ms"]["mean"] == 1120
    assert summary["agents"]["bob"]["pass@1"] == 1.0
    assert summary["agents"]["steve"]["pass@1"] == 0.0
    assert summary["agents"]["vera"]["vera"]["precision"] == 0.5
    assert summary["agents"]["vera"]["vera"]["recall"] == 0.5
    assert summary["agents"]["vera"]["vera"]["tp"] == 1
    assert summary["agents"]["vera"]["vera"]["fp"] == 1
    assert summary["agents"]["vera"]["vera"]["fn"] == 1


def test_load_fixture_accepts_mini_swe_runner_shape_and_usage_telemetry(tmp_path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "mini-1",
                "agent": "steve",
                "k": 2,
                "runs": [
                    {
                        "completed": False,
                        "api_calls": 2,
                        "metadata": {"model": "anthropic/claude-sonnet-4-20250514"},
                        "usage": {
                            "cost_usd": 0.12,
                            "latency_ms": 1500,
                            "input_tokens": 1000,
                            "output_tokens": 200,
                        },
                    },
                    {
                        "completed": True,
                        "api_calls": 3,
                        "metadata": {"model": "anthropic/claude-sonnet-4-20250514"},
                        "usage": {"cost_usd": 0.34, "latency_ms": 2500},
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks, k=2)

    assert tasks[0].task_id == "mini-1"
    assert tasks[0].attempts[0].passed is False
    assert tasks[0].attempts[1].passed is True
    assert tasks[0].attempts[0].api_calls == 2
    assert summary["overall"]["pass@1"] == 0.0
    assert summary["overall"]["pass^2"] == 1.0
    assert summary["overall"]["cost_usd"] == 0.46
    assert summary["overall"]["latency_ms"]["mean"] == 2000


def test_repository_fixture_example_scores_without_network():
    fixture = Path(__file__).parent / "fixtures" / "b3_eval_fixture.json"

    summary = score_eval_suite(load_fixture(fixture), k=2)

    assert summary["overall"]["task_count"] == 3
    assert summary["overall"]["pass^2"] == 1.0
    assert summary["agents"]["vera"]["vera"]["precision"] == 0.5


def test_explicit_fixture_cost_wins_over_estimated_cost(tmp_path, monkeypatch):
    def _unexpected_estimator(*args, **kwargs):
        raise AssertionError("explicit fixture cost should bypass estimation")

    monkeypatch.setattr(b3, "estimate_usage_cost_from_static_pricing", _unexpected_estimator)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "explicit-cost",
                "agent": "steve",
                "attempts": [
                    {
                        "passed": True,
                        "cost_usd": 0.99,
                        "usage": {
                            "input_tokens": 1000,
                            "output_tokens": 200,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)

    assert tasks[0].attempts[0].cost_usd == 0.99
    assert tasks[0].attempts[0].cost_status == "actual"
    assert tasks[0].attempts[0].cost_source == "fixture"


def test_load_fixture_estimates_cost_from_known_static_provider_when_cost_missing(tmp_path, monkeypatch):
    seen = {}

    def _fake_estimator(model_name, usage, *, provider=None, base_url=None):
        seen["model_name"] = model_name
        seen["provider"] = provider
        seen["base_url"] = base_url
        seen["usage"] = usage
        return SimpleNamespace(
            amount_usd=Decimal("1.2345674"),
            status="estimated",
            source="official_docs_snapshot",
        )

    monkeypatch.setattr(b3, "estimate_usage_cost_from_static_pricing", _fake_estimator)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "steve-live-1",
                "agent": "steve",
                "runs": [
                    {
                        "completed": True,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "usage": {"input_tokens": 1000, "output_tokens": 200},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert tasks[0].attempts[0].cost_usd == 1.234567
    assert tasks[0].attempts[0].cost_status == "estimated"
    assert tasks[0].attempts[0].cost_source == "official_docs_snapshot"
    assert summary["overall"]["cost_usd"] == 1.234567
    assert summary["overall"]["cost"]["estimated_attempts"] == 1
    assert summary["overall"]["cost"]["unknown_attempts"] == 0
    assert seen["model_name"] == "claude-sonnet-4-6"
    assert seen["provider"] == "anthropic"
    assert seen["usage"].input_tokens == 1000
    assert seen["usage"].output_tokens == 200


def test_remote_pricing_routes_do_not_fetch_metadata_and_stay_unknown(tmp_path, monkeypatch):
    def _forbid_remote_fetch(*args, **kwargs):
        raise AssertionError("offline B3 fixtures must not fetch remote pricing metadata")

    monkeypatch.setattr("agent.usage_pricing.fetch_model_metadata", _forbid_remote_fetch)
    monkeypatch.setattr("agent.usage_pricing.fetch_endpoint_model_metadata", _forbid_remote_fetch)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "openrouter-route",
                        "agent": "steve",
                        "attempts": [
                            {
                                "passed": True,
                                "provider": "openrouter",
                                "model": "anthropic/claude-sonnet-4-20250514",
                                "usage": {"input_tokens": 1000, "output_tokens": 200},
                            }
                        ],
                    },
                    {
                        "task_id": "nous-route",
                        "agent": "steve",
                        "attempts": [
                            {
                                "passed": True,
                                "provider": "nous",
                                "model": "openai/gpt-5.5-pro",
                                "usage": {"input_tokens": 1000, "output_tokens": 200},
                            }
                        ],
                    },
                    {
                        "task_id": "base-url-route",
                        "agent": "steve",
                        "attempts": [
                            {
                                "passed": True,
                                "model": "zai-org/GLM-5-TEE",
                                "base_url": "https://llm.chutes.ai/v1",
                                "usage": {"input_tokens": 1000, "output_tokens": 200},
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert [task.attempts[0].cost_usd for task in tasks] == [None, None, None]
    assert [task.attempts[0].cost_status for task in tasks] == ["unknown", "unknown", "unknown"]
    assert summary["overall"]["cost_usd"] == 0.0
    assert summary["overall"]["cost"] == {
        "usd": 0.0,
        "known_usd": 0.0,
        "known_attempts": 0,
        "unknown_attempts": 3,
        "actual_attempts": 0,
        "estimated_attempts": 0,
        "included_attempts": 0,
    }


def test_metadata_model_and_provider_are_accepted(tmp_path, monkeypatch):
    seen = {}

    def _fake_estimator(model_name, usage, *, provider=None, base_url=None):
        seen["model_name"] = model_name
        seen["provider"] = provider
        return SimpleNamespace(
            amount_usd=Decimal("0.42"),
            status="estimated",
            source="official_docs_snapshot",
        )

    monkeypatch.setattr(b3, "estimate_usage_cost_from_static_pricing", _fake_estimator)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "metadata-route",
                "agent": "steve",
                "attempts": [
                    {
                        "passed": True,
                        "metadata": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                        "usage": {"input_tokens": 1000, "output_tokens": 200},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)

    assert tasks[0].attempts[0].cost_usd == 0.42
    assert tasks[0].attempts[0].cost_status == "estimated"
    assert seen == {"model_name": "claude-sonnet-4-6", "provider": "anthropic"}


def test_vendor_prefixed_model_without_explicit_provider_stays_unknown(tmp_path, monkeypatch):
    def _unexpected_estimator(*args, **kwargs):
        raise AssertionError("vendor-prefixed model alone is not a trusted provider route")

    monkeypatch.setattr(b3, "estimate_usage_cost_from_static_pricing", _unexpected_estimator)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "vendor-prefixed-only",
                "agent": "steve",
                "attempts": [
                    {
                        "passed": True,
                        "metadata": {"model": "anthropic/claude-sonnet-4-20250514"},
                        "usage": {"input_tokens": 1000, "output_tokens": 200},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert tasks[0].attempts[0].cost_usd is None
    assert tasks[0].attempts[0].cost_status == "unknown"
    assert tasks[0].attempts[0].cost_source == "none"
    assert summary["overall"]["cost_usd"] == 0.0
    assert summary["overall"]["cost"]["unknown_attempts"] == 1


def test_subscription_included_route_counts_as_known_included(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "codex-included",
                "agent": "steve",
                "attempts": [
                    {
                        "passed": True,
                        "provider": "openai-codex",
                        "model": "gpt-5.1-codex",
                        "usage": {"input_tokens": 1000, "output_tokens": 200},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert tasks[0].attempts[0].cost_usd == 0.0
    assert tasks[0].attempts[0].cost_status == "included"
    assert summary["overall"]["cost"]["known_attempts"] == 1
    assert summary["overall"]["cost"]["included_attempts"] == 1
    assert summary["overall"]["cost"]["unknown_attempts"] == 0


@pytest.mark.parametrize(
    "attempt, expected_status",
    [
        ({"passed": True, "usage": {"input_tokens": 1000, "output_tokens": 200}}, "unknown"),
        ({"passed": True, "provider": "anthropic", "model": "claude-sonnet-4-6"}, "no_usage"),
    ],
)
def test_missing_model_or_zero_usage_is_not_reported_as_exact_free(tmp_path, attempt, expected_status):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps({"task_id": "missing-cost-input", "agent": "steve", "attempts": [attempt]}),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert tasks[0].attempts[0].cost_usd is None
    assert tasks[0].attempts[0].cost_status == expected_status
    assert summary["overall"]["cost_usd"] == 0.0
    assert summary["overall"]["cost"]["unknown_attempts"] == 1


def test_cache_read_and_write_token_fields_are_passed_to_static_estimator(tmp_path, monkeypatch):
    seen = {}

    def _fake_estimator(model_name, usage, *, provider=None, base_url=None):
        seen["usage"] = usage
        return SimpleNamespace(amount_usd=Decimal("0.01"))

    monkeypatch.setattr(b3, "estimate_usage_cost_from_static_pricing", _fake_estimator)
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "cache-tokens",
                "agent": "steve",
                "attempts": [
                    {
                        "passed": True,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "cache_read_input_tokens": 30,
                            "cache_creation_input_tokens": 40,
                            "request_count": 2,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    load_fixture(fixture)

    assert seen["usage"].input_tokens == 100
    assert seen["usage"].output_tokens == 20
    assert seen["usage"].cache_read_tokens == 30
    assert seen["usage"].cache_write_tokens == 40
    assert seen["usage"].request_count == 2


def test_load_fixture_estimates_cost_from_static_pricing_when_cost_missing(tmp_path):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(
        json.dumps(
            {
                "task_id": "steve-live-1",
                "agent": "steve",
                "runs": [
                    {
                        "completed": True,
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    tasks = load_fixture(fixture)
    summary = score_eval_suite(tasks)

    assert tasks[0].attempts[0].cost_usd > 0.0
    assert summary["overall"]["cost_usd"] == tasks[0].attempts[0].cost_usd


def test_cli_writes_json_summary_for_fixture(tmp_path, capsys):
    fixture = tmp_path / "fixture.json"
    output = tmp_path / "summary.json"
    fixture.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "vera-1",
                        "agent": "vera",
                        "attempts": [{"passed": True, "cost_usd": 0.01, "latency_ms": 100}],
                        "expected_findings": [{"id": "bug-a"}],
                        "reported_findings": [{"id": "bug-a"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--fixture", str(fixture), "--k", "3", "--output", str(output)])

    assert exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["overall"]["pass^3"] == 1.0
    assert saved["overall"]["task_count"] == 1
    assert "pass^3" in capsys.readouterr().out
