from __future__ import annotations

import importlib.util
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import urllib.error

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "timings_report.py"
_SPEC = importlib.util.spec_from_file_location("timings_report", SCRIPT_PATH)
assert _SPEC and _SPEC.loader
TIMINGS_REPORT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(TIMINGS_REPORT)


def _http_403(url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, 403, "Forbidden", hdrs=Message(), fp=None)


def test_collect_timings_returns_warning_instead_of_raising_for_orchestrator_jobs_403(monkeypatch: pytest.MonkeyPatch):
    def fake_api_get(path: str, token: str, params=None, list_key=None):
        if path == "/repos/acme/hermes/actions/runs/123":
            return {"created_at": "2026-07-04T00:00:00Z"}
        if path == "/repos/acme/hermes/actions/runs/123/jobs":
            raise _http_403(f"https://api.github.com{path}")
        if path == "/repos/acme/hermes/actions/runs":
            return []
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(TIMINGS_REPORT, "api_get", fake_api_get)

    timings = TIMINGS_REPORT.collect_timings(
        token="token",
        repo="acme/hermes",
        run_id="123",
        head_sha="deadbeef",
    )

    assert timings["run_id"] == "123"
    assert timings["jobs"] == []
    assert timings["warnings"] == [
        "Could not list orchestrator jobs for run 123 (HTTP 403 Forbidden); generated a partial report without those jobs."
    ]


def test_collect_timings_skips_only_the_forbidden_subworkflow_runs_listing(monkeypatch: pytest.MonkeyPatch):
    direct_job = {
        "id": 1,
        "name": "direct",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-07-04T00:00:00Z",
        "completed_at": "2026-07-04T00:01:00Z",
        "steps": [],
    }

    def fake_api_get(path: str, token: str, params=None, list_key=None):
        if path == "/repos/acme/hermes/actions/runs/123":
            return {"created_at": "2026-07-04T00:00:00Z"}
        if path == "/repos/acme/hermes/actions/runs/123/jobs":
            return [direct_job]
        if path == "/repos/acme/hermes/actions/runs":
            raise _http_403(f"https://api.github.com{path}")
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(TIMINGS_REPORT, "api_get", fake_api_get)

    timings = TIMINGS_REPORT.collect_timings(
        token="token",
        repo="acme/hermes",
        run_id="123",
        head_sha="deadbeef",
    )

    assert [job["name"] for job in timings["jobs"]] == ["direct"]
    assert timings["warnings"] == [
        "Could not list sub-workflow runs for head deadbeef (HTTP 403 Forbidden); generated a partial report without those jobs."
    ]


def test_collect_timings_skips_only_the_forbidden_subworkflow_jobs(monkeypatch: pytest.MonkeyPatch):
    direct_job = {
        "id": 1,
        "name": "direct",
        "status": "completed",
        "conclusion": "success",
        "started_at": "2026-07-04T00:00:00Z",
        "completed_at": "2026-07-04T00:01:00Z",
        "steps": [],
    }

    def fake_api_get(path: str, token: str, params=None, list_key=None):
        if path == "/repos/acme/hermes/actions/runs/123":
            return {"created_at": "2026-07-04T00:00:00Z"}
        if path == "/repos/acme/hermes/actions/runs/123/jobs":
            return [direct_job]
        if path == "/repos/acme/hermes/actions/runs":
            return [{"id": 456, "name": "Collect timings", "created_at": "2026-07-04T00:00:01Z"}]
        if path == "/repos/acme/hermes/actions/runs/456/jobs":
            raise _http_403(f"https://api.github.com{path}")
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(TIMINGS_REPORT, "api_get", fake_api_get)

    timings = TIMINGS_REPORT.collect_timings(
        token="token",
        repo="acme/hermes",
        run_id="123",
        head_sha="deadbeef",
    )

    assert [job["name"] for job in timings["jobs"]] == ["direct"]
    assert timings["warnings"] == [
        "Could not list sub-workflow jobs for Collect timings for run 456 (HTTP 403 Forbidden); generated a partial report without those jobs."
    ]


def test_collect_timings_reraises_non_403_http_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_api_get(path: str, token: str, params=None, list_key=None):
        if path == "/repos/acme/hermes/actions/runs/123":
            return {"created_at": "2026-07-04T00:00:00Z"}
        if path == "/repos/acme/hermes/actions/runs/123/jobs":
            raise urllib.error.HTTPError(
                f"https://api.github.com{path}",
                500,
                "Internal Server Error",
                hdrs=Message(),
                fp=None,
            )
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(TIMINGS_REPORT, "api_get", fake_api_get)

    with pytest.raises(urllib.error.HTTPError):
        TIMINGS_REPORT.collect_timings(
            token="token",
            repo="acme/hermes",
            run_id="123",
            head_sha="deadbeef",
        )


def test_generate_summary_includes_warnings_section():
    summary = TIMINGS_REPORT.generate_summary(
        {
            "run_id": "123",
            "head_sha": "deadbeef",
            "created_at": "2026-07-04T00:00:00Z",
            "jobs": [],
            "warnings": [
                "Could not list orchestrator jobs for run 123 (HTTP 403 Forbidden); generated a partial report without those jobs."
            ],
        }
    )

    assert "### Warnings" in summary
    assert "HTTP 403 Forbidden" in summary


def test_generate_html_includes_warning_banner():
    html = TIMINGS_REPORT.generate_html(
        {
            "run_id": "123",
            "head_sha": "deadbeef",
            "created_at": "2026-07-04T00:00:00Z",
            "jobs": [],
            "warnings": ["partial report"],
        }
    )

    assert "Warnings:" in html
    assert "partial report" in html


def test_format_http_warning_omits_missing_reason():
    warning = TIMINGS_REPORT._format_http_warning(
        label="orchestrator jobs",
        resource_id="run 123",
        err=cast(urllib.error.HTTPError, SimpleNamespace(code=403, reason=None)),
    )

    assert warning == (
        "Could not list orchestrator jobs for run 123 "
        "(HTTP 403); generated a partial report without those jobs."
    )


def test_main_honors_from_json_without_calling_collect_timings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_json = tmp_path / "input.json"
    baseline_json = tmp_path / "baseline.json"
    html_out = tmp_path / "report.html"
    json_out = tmp_path / "out.json"
    summary_out = tmp_path / "summary.md"

    payload = {
        "run_id": "123",
        "head_sha": "deadbeef",
        "created_at": "2026-07-04T00:00:00Z",
        "jobs": [],
        "warnings": ["partial"],
    }
    input_json.write_text(__import__("json").dumps(payload), encoding="utf-8")
    baseline_json.write_text(__import__("json").dumps({"jobs": []}), encoding="utf-8")

    def fail_collect(*args, **kwargs):
        raise AssertionError("collect_timings should not be called for --from-json")

    monkeypatch.setattr(TIMINGS_REPORT, "collect_timings", fail_collect)
    monkeypatch.setattr(
        "sys.argv",
        [
            "timings_report.py",
            "--from-json",
            str(input_json),
            "--baseline",
            str(baseline_json),
            "--output",
            str(html_out),
            "--json-out",
            str(json_out),
            "--summary-out",
            str(summary_out),
        ],
    )

    TIMINGS_REPORT.main()

    assert json_out.exists()
    assert html_out.exists()
    assert summary_out.exists()
    assert "### Warnings" in summary_out.read_text(encoding="utf-8")
