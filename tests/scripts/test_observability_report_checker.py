from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_checker():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "observability_report_checker.py"
    spec = importlib.util.spec_from_file_location("observability_report_checker", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["observability_report_checker"] = module
    spec.loader.exec_module(module)
    return module


def write_report(path: Path, *, start: str, end: str, agent_turns: int, api_calls: int = 20, dataset: str = "hermes-agent") -> None:
    path.write_text(
        f"""---
type: observability-report
system: hermes-agent
source: honeycomb
team: hermes
environment: prod
dataset: {dataset}
date_ran: 2026-07-01
period_analyzed_start: {start}
period_analyzed_end: {end}
period_analyzed: 7d
trace_count: {agent_turns}
agent_turn_count: {agent_turns}
api_call_count: {api_calls}
tool_call_count: 30
error_ratio: 0.02
tool_error_ratio: 0.03
p95_agent_latency_ms: 1000.0
p95_api_latency_ms: 200.0
p95_tool_latency_ms: 50.0
cache_read_ratio: 0.80
tags: [observability, hermes, honeycomb, weekly]
---

# Hermes Observability Weekly — 2026-07-01

## Query notes
""",
        encoding="utf-8",
    )


def test_comparability_labels_overlapping_weekly_windows(tmp_path):
    checker = load_checker()
    current_path = tmp_path / "2026-06-30 Hermes Observability Weekly.md"
    prior_path = tmp_path / "2026-06-29 Hermes Observability Weekly.md"
    write_report(
        current_path,
        start="2026-06-24T00:00:55Z",
        end="2026-07-01T00:00:55Z",
        agent_turns=159,
        api_calls=1882,
    )
    write_report(
        prior_path,
        start="2026-06-23T00:00:54Z",
        end="2026-06-30T00:00:54Z",
        agent_turns=236,
        api_calls=2916,
    )

    result = checker.build_result(current_path)

    assert result["prior_report"] == str(prior_path)
    assert result["comparability"] == "overlapping"
    assert "overlap" in result["comparability_reason"]
    assert result["deltas"]["agent_turn_count"]["delta_percent"] == pytest_approx(-32.627, abs=0.01)
    assert "Trend comparability label: `overlapping`" in result["query_notes"]
    assert "api_token_spend_by_session" in result["query_notes"]


def test_comparability_labels_independent_matching_windows(tmp_path):
    checker = load_checker()
    current_path = tmp_path / "2026-07-08 Hermes Observability Weekly.md"
    prior_path = tmp_path / "2026-07-01 Hermes Observability Weekly.md"
    write_report(current_path, start="2026-07-01T00:00:00Z", end="2026-07-08T00:00:00Z", agent_turns=10)
    write_report(prior_path, start="2026-06-24T00:00:00Z", end="2026-07-01T00:00:00Z", agent_turns=20)

    result = checker.build_result(current_path)

    assert result["comparability"] == "independent"
    assert result["deltas"]["agent_turn_count"]["delta_percent"] == -50.0


def test_comparability_labels_incomparable_when_scope_differs(tmp_path):
    checker = load_checker()
    current_path = tmp_path / "2026-07-08 Hermes Observability Weekly.md"
    prior_path = tmp_path / "2026-07-01 Hermes Observability Weekly.md"
    write_report(current_path, start="2026-07-01T00:00:00Z", end="2026-07-08T00:00:00Z", agent_turns=10)
    write_report(
        prior_path,
        start="2026-06-24T00:00:00Z",
        end="2026-07-01T00:00:00Z",
        agent_turns=20,
        dataset="different-dataset",
    )

    result = checker.build_result(current_path)

    assert result["comparability"] == "incomparable"
    assert "dataset" in result["comparability_reason"]


def test_comparability_labels_incomparable_without_prior_report(tmp_path):
    checker = load_checker()
    current_path = tmp_path / "2026-07-08 Hermes Observability Weekly.md"
    write_report(current_path, start="2026-07-01T00:00:00Z", end="2026-07-08T00:00:00Z", agent_turns=10)

    result = checker.build_result(current_path)

    assert result["prior_report"] is None
    assert result["comparability"] == "incomparable"
    assert "No prior report" in result["comparability_reason"]


def test_parse_simple_frontmatter_without_pyyaml(tmp_path):
    checker = load_checker()
    path = tmp_path / "2026-07-01 Hermes Observability Weekly.md"
    write_report(path, start="2026-06-24T00:00:00Z", end="2026-07-01T00:00:00Z", agent_turns=42)

    report = checker.load_report(path)

    assert report.frontmatter["tags"] == ["observability", "hermes", "honeycomb", "weekly"]
    assert report.metric("agent_turn_count") == 42
    assert int(report.duration_seconds) == 7 * 24 * 60 * 60


def pytest_approx(*args, **kwargs):
    import pytest

    return pytest.approx(*args, **kwargs)
