"""``hermes.compression.count`` keeps a committed fallback's class instead of reading as a clean success.

A committed fallback is either a deterministic summary (the summary model failed, was skipped or benched,
or the attempt stalled) or a main-model summary written after a separate summary model failed. Both commit
and break the prompt cache; reporting them as ``success`` / ``none`` hid them from the metric.
"""

import json
from pathlib import Path

import pytest

from hermes_cli.observability import shared_metrics_events as events
from hermes_cli.observability.shared_metrics import SharedMetricsStore
from hermes_cli.observability.shared_metrics_fields import compression_fields

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "hermes_cli" / "observability" / "schemas"
    / "hermes.shared_metrics.v4.schema.json"
)
RESOURCE = {"architecture": "x86_64", "hermes_version": "test", "install_method": "git", "os_family": "linux"}


def _finish(monkeypatch, commit_status, failure_class):
    rows = []
    monkeypatch.setattr(events, "record_compression", lambda **kw: rows.append(compression_fields(**kw)))
    events.begin_compression_attempt("auto", 80_000)
    events.finish_compression_attempt(commit_status, failure_class, 100_000)
    return [(row["outcome"], row["failure_class"]) for row in rows]


@pytest.mark.parametrize(
    ("failure_class", "expected"),
    [
        (None, ("success", "none")),
        ("summary_generation_failed", ("success", "summary_generation_failed")),
        ("feasibility_skip", ("success", "feasibility_skip")),
        ("stall_deterministic_fallback", ("success", "stall_deterministic_fallback")),
        ("aux_model_fallback", ("success", "aux_model_fallback")),
        # A benched summary model is a pre-LLM skip to the deterministic summary.
        ("summary_model_benched", ("success", "feasibility_skip")),
        # A class outside the closed set still never reads as a clean success.
        ("plugin_specific_fallback", ("success", "other")),
    ],
)
def test_committed_attempt_keeps_its_fallback_class(monkeypatch, failure_class, expected):
    assert _finish(monkeypatch, "committed", failure_class) == [expected]


def test_aborted_attempts_keep_their_skip_and_failure_verdicts(monkeypatch):
    assert _finish(monkeypatch, "aborted", "lock_contended") == [("skipped", "lock_contended")]
    assert _finish(monkeypatch, "aborted", "summary_auth_failure") == [("failed", "summary_auth_failure")]


def test_committed_fallback_row_validates_against_the_package_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    store = SharedMetricsStore(tmp_path / "metrics.sqlite3", tmp_path / "outbox")
    fields = compression_fields(
        trigger="auto", outcome="success", tokens_before=80_000, context_length=100_000,
        failure_class="summary_generation_failed",
    )
    store.record_counter("hermes.compression.count", fields, RESOURCE)

    [package_path] = store.create_and_export_package()
    package = json.loads(package_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(package)
    [row] = [m for m in package["metrics"] if m["name"] == "hermes.compression.count"]
    assert row["dimensions"]["failure_class"] == "summary_generation_failed"
