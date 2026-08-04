from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.llm_benchmark_measurements import collect
from scripts.llm_benchmark_weekly import (
    INSUFFICIENT_EVIDENCE,
    apply_evidence_gate,
    build_preflight,
    detect_config_drift,
    sanitise_measurements,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_preflight_collects_effective_config_routes_catalogue_hardware_sources_and_previous(tmp_path: Path) -> None:
    home = tmp_path / ".hermes"
    (home / "config.yaml").parent.mkdir(parents=True)
    (home / "config.yaml").write_text("model: root-model\nfallback_models: [root-fallback]\n", encoding="utf-8")
    profile = home / "profiles" / "coder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("model: coder-model\n", encoding="utf-8")
    _write_json(home / "cron" / "jobs.json", {"jobs": [{"name": "llm-benchmark-weekly", "enabled": True, "profile": "coder", "model": "pinned"}]})
    catalogue = home / "model_catalogue.json"
    _write_json(catalogue, {"models": ["available-only"]})
    previous = tmp_path / "previous.json"
    _write_json(previous, {"generated_at": "2026-07-24T10:00:00Z"})
    sources = tmp_path / "sources.json"
    _write_json(sources, [{"name": "LiveBench", "url": "https://livebench.ai", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30", "status": "success"}])

    result = build_preflight(home, catalogue, sources, previous, now=datetime(2026, 7, 31, 10, tzinfo=timezone.utc), hardware_reader=lambda: {"memory_bytes": 42})

    assert result["effective_configs"]["root"]["model"] == "root-model"
    assert result["effective_configs"]["profiles"]["coder"]["model"] == "coder-model"
    assert result["active_cron_routes"][0]["model"] == "pinned"
    assert result["catalogue"]["evidence_role"] == "availability_only"
    assert result["catalogue"]["modified_at"]
    assert result["hardware"] == {"memory_bytes": 42}
    assert result["source_retrievals"][0]["source_date"] == "2026-07-30"
    assert result["previous_snapshot"]["generated_at"] == "2026-07-24T10:00:00Z"


def test_config_drift_is_derived_from_effective_config_not_hardcoded_fleet() -> None:
    configs = {"root": {"model": "new"}, "profiles": {"coder": {"model": "actual", "fallback_models": ["b", "a"]}}}
    declared = {"root": {"model": "old"}, "profiles": {"coder": {"model": "claimed", "fallback_models": ["a", "b"]}}}

    drift = detect_config_drift(configs, declared)

    assert {item["path"] for item in drift} == {"root.model", "profiles.coder.model", "profiles.coder.fallback_models"}


def test_fresh_independent_sources_and_row_provenance_allow_rankings() -> None:
    now = datetime(2026, 7, 31, 10, tzinfo=timezone.utc)
    sources = [
        {"name": "LiveBench", "status": "success", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30"},
        {"name": "LMSYS", "status": "success", "retrieved_at": "2026-07-31T09:05:00Z", "source_date": "2026-07-29"},
    ]
    rows = [{"rank": 1, "model": "Example", "provenance": [{"source": "LiveBench", "source_date": "2026-07-30"}, {"source": "LMSYS", "source_date": "2026-07-29"}]}]

    gated = apply_evidence_gate(rows, sources, now=now, max_age_days=31)

    assert gated["status"] == "current"
    assert gated["rows"] == rows


@pytest.mark.parametrize(
    "sources,rows",
    [
        ([{"name": "LiveBench", "status": "success", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30"}], [{"model": "X", "provenance": [{"source": "LiveBench", "source_date": "2026-07-30"}]}]),
        ([{"name": "A", "status": "success", "retrieved_at": "2026-01-01T00:00:00Z", "source_date": "2026-01-01"}, {"name": "B", "status": "success", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30"}], [{"model": "X", "provenance": []}]),
        ([{"name": "A", "status": "success", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30"}, {"name": "B", "status": "success", "retrieved_at": "2026-07-31T09:00:00Z", "source_date": "2026-07-30"}], [{"model": "X"}]),
    ],
)
def test_insufficient_or_stale_sources_and_missing_provenance_fail_closed(sources: list[dict], rows: list[dict]) -> None:
    gated = apply_evidence_gate(rows, sources, now=datetime(2026, 7, 31, 10, tzinfo=timezone.utc), max_age_days=31)

    assert gated["status"] == "insufficient_current_evidence"
    assert gated["message"] == INSUFFICIENT_EVIDENCE
    assert gated["rows"] == []


def test_evidence_gate_does_not_mutate_html_structure() -> None:
    html = '<!doctype html><html><head><style>.x{color:#fff}</style></head><body><section id="overall"><table><thead><tr><th>Model</th></tr></thead><tbody>{{OVERALL_ROWS}}</tbody></table></section><section id="open">{{OPEN_ROWS}}</section></body></html>'
    before = re.findall(r"</?[a-zA-Z][^>]*>", html)

    apply_evidence_gate([], [], now=datetime(2026, 7, 31, tzinfo=timezone.utc))

    assert re.findall(r"</?[a-zA-Z][^>]*>", html) == before


def test_measurements_are_bounded_and_secrets_are_removed() -> None:
    raw = [{"provider": "p", "model": "m", "task": "coding", "latency_ms": 12, "char_count": 900, "content_preview": "x" * 500, "api_key": "SECRET", "authorization": "Bearer SECRET"}] * 5

    clean = sanitise_measurements(raw, max_records=2, max_preview_chars=40)

    assert len(clean) == 2
    assert len(clean[0]["content_preview"]) == 40
    assert "SECRET" not in json.dumps(clean)
    assert set(clean[0]) == {"provider", "model", "task", "latency_ms", "char_count", "content_preview", "status", "measured_at"}


def test_separate_collector_is_bounded_and_does_not_serialise_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_TEST_KEY", "TOP-SECRET")
    config = {"providers": [
        {"name": "first", "model": "m1", "endpoint": "http://127.0.0.1:1", "key_env": "MISSING_KEY"},
        {"name": "second", "model": "m2", "endpoint": "http://127.0.0.1:1", "key_env": "BENCHMARK_TEST_KEY"},
    ]}

    records = collect(config, timeout_seconds=0.01, max_providers=1)

    assert len(records) == 3
    assert {record["provider"] for record in records} == {"first"}
    assert {record["status"] for record in records} == {"credential_unavailable"}
    assert "TOP-SECRET" not in json.dumps(records)


def test_owned_snapshot_prompt_has_no_hardcoded_fleet_and_defines_fail_closed_contract() -> None:
    snapshot = json.loads((Path(__file__).parents[2] / "cron" / "jobs.snapshot.json").read_text(encoding="utf-8"))
    job = next(job for job in snapshot["jobs"] if job["name"] == "llm-benchmark-weekly")

    assert "Insufficient current evidence" in job["prompt"]
    assert "two current independent" in job["prompt"]
    assert "availability evidence only" in job["prompt"]
    assert "Preserve the current report HTML, CSS, layout, sections" in job["prompt"]
    assert "deepseek-v4-flash" not in job["prompt"]
