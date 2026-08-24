from __future__ import annotations

from typing import Any

import pytest

from hermes_cli.config import (
    build_cron_model_impact,
    cron_model_drift_axes,
    resolve_cron_model_drift_defaults,
)


def _job(**overrides: Any) -> dict[str, Any]:
    job = {
        "id": "job-1",
        "name": "Morning summary",
        "enabled": True,
        "no_agent": False,
        "provider_snapshot": "openrouter",
        "model_snapshot": "old/model",
    }
    job.update(overrides)
    return job


def _impact(jobs: object, **config: Any) -> dict[str, Any]:
    return build_cron_model_impact(
        current_provider="nous",
        current_model="new/model",
        config=config,
        jobs=jobs,
    )


def test_drift_axes_match_unpinned_guard_semantics() -> None:
    assert cron_model_drift_axes(
        _job(), current_provider=" NOUS ", current_model="NEW/MODEL", config={}
    ) == ["provider", "model"]
    assert cron_model_drift_axes(
        _job(provider="openrouter"),
        current_provider="nous",
        current_model="new/model",
        config={},
    ) == ["model"]
    assert cron_model_drift_axes(
        _job(model="old/model", provider="openrouter"),
        current_provider="nous",
        current_model="new/model",
        config={},
    ) == []


def test_fleet_defaults_and_literal_false_guard_suppress_only_intended_axes() -> None:
    assert cron_model_drift_axes(
        _job(),
        current_provider="nous",
        current_model="new/model",
        config={"cron": {"model": "fleet/model"}},
    ) == ["provider"]
    assert cron_model_drift_axes(
        _job(),
        current_provider="nous",
        current_model="new/model",
        config={"cron": {"model_provider": "openrouter"}},
    ) == ["model"]
    assert cron_model_drift_axes(
        _job(),
        current_provider="nous",
        current_model="new/model",
        config={"cron": {"model_drift_guard": False}},
    ) == []
    assert cron_model_drift_axes(
        _job(),
        current_provider="nous",
        current_model="new/model",
        config={"cron": {"model_drift_guard": "false"}},
    ) == ["provider", "model"]


def test_summary_filters_disabled_no_agent_and_mixed_pins() -> None:
    jobs = [
        _job(id="disabled", enabled=False),
        _job(id="script", no_agent=True),
        _job(id="paused-state", state="paused"),
        _job(id="paused-marker", paused_at="2026-08-11T00:00:00Z"),
        _job(id="fully-pinned", model="old/model", provider="openrouter"),
        _job(id="model-pinned", model="old/model"),
        _job(id="provider-pinned", provider="openrouter"),
    ]

    impact = _impact(jobs)

    assert impact == {
        "available": True,
        "guard_enabled": True,
        "affected_count": 2,
        "truncated": False,
        "jobs": [
            {"id": "model-pinned", "name": "Morning summary", "drifted_axes": ["provider"]},
            {"id": "provider-pinned", "name": "Morning summary", "drifted_axes": ["model"]},
        ],
    }


def test_summary_handles_legacy_and_malformed_records_without_hiding_valid_siblings() -> None:
    impact = _impact(
        [
            None,
            "bad",
            {"id": ["not-a-string"]},
            _job(id="job-1", name="  Morning\u0000   summary  "),
            _job(id="job-1", name="duplicate"),
            _job(id="job-2", name=42),
            _job(id="missing-snapshot", provider_snapshot=None, model_snapshot={}),
        ]
    )

    assert impact["affected_count"] == 2
    assert impact["jobs"] == [
        {"id": "job-1", "name": "Morning summary", "drifted_axes": ["provider", "model"]},
        {"id": "job-2", "name": "Job job-2", "drifted_axes": ["provider", "model"]},
    ]


@pytest.mark.parametrize("jobs", [{"jobs": []}, "jobs", 7])
def test_non_list_job_collection_is_unavailable(jobs: object) -> None:
    assert _impact(jobs) == {
        "available": False,
        "guard_enabled": True,
        "affected_count": 0,
        "truncated": False,
        "jobs": [],
    }


def test_summary_bounds_id_name_and_payload() -> None:
    valid = [_job(id=f"job-{index:03}", name=f" Job {index} ") for index in range(51)]
    jobs = [
        _job(id="x" * 257),
        _job(id="bad\u200bid"),
        _job(id="edge", name="x" * 121),
        *valid,
    ]

    impact = _impact(jobs)

    assert impact["affected_count"] == 52
    assert len(impact["jobs"]) == 50
    assert impact["truncated"] is True
    assert impact["jobs"][0]["id"] == "edge"
    assert impact["jobs"][0]["name"] == "x" * 120
    assert impact["jobs"][-1]["id"] == "job-048"


def test_fallback_name_respects_the_desktop_code_point_limit() -> None:
    job_id = "x" * 256

    impact = _impact([_job(id=job_id, name=42)])

    assert impact["affected_count"] == 1
    assert impact["jobs"][0]["name"] == (f"Job {job_id}")[:120]


def test_guard_disabled_summary_is_available_but_empty() -> None:
    assert _impact([_job()], cron={"model_drift_guard": False}) == {
        "available": True,
        "guard_enabled": False,
        "affected_count": 0,
        "truncated": False,
        "jobs": [],
    }


def test_effective_defaults_match_scheduler_config_over_env_precedence() -> None:
    assert resolve_cron_model_drift_defaults(
        {"model": {"provider": "managed", "default": "managed/model"}},
        environ={"HERMES_MODEL": "env/model"},
    ) == ("managed", "managed/model")
    assert resolve_cron_model_drift_defaults(
        {"model": {}}, environ={"HERMES_MODEL": "env/model"}
    ) == ("", "env/model")
    assert resolve_cron_model_drift_defaults(
        {"model": "legacy/model"}, environ={"HERMES_MODEL": "env/model"}
    ) == ("", "legacy/model")


def test_loader_exception_returns_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import cron.jobs

    def fail() -> list[dict[str, Any]]:
        raise RuntimeError("broken store")

    monkeypatch.setattr(cron.jobs, "load_jobs", fail)

    impact = build_cron_model_impact(
        current_provider="nous", current_model="new/model", config={}
    )

    assert impact["available"] is False
    assert impact["affected_count"] == 0
