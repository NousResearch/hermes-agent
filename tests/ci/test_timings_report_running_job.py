"""The timing report counts the job it runs in.

The report runs as trailing steps of the `all-checks-pass` gate job, which the jobs API still lists
as in_progress while the report collects. Other unfinished jobs stay out; the report's own job
counts up to its last finished step, so the gate appears in every report and in main's baseline.
"""
from __future__ import annotations

from scripts.ci import timings_report

GATE = "All required checks pass (v2)"


def _step(name: str, start: str, end: str | None, conclusion: str | None = "success") -> dict:
    return {"name": name, "number": 1, "status": "completed" if end else "in_progress",
            "conclusion": conclusion if end else None, "started_at": start, "completed_at": end}


def _jobs() -> list[dict]:
    return [
        {"id": 1, "name": "Detect affected areas", "status": "completed", "conclusion": "success",
         "started_at": "2026-01-01T00:00:00Z", "completed_at": "2026-01-01T00:00:10Z",
         "runner_name": "runner-a", "steps": []},
        {"id": 2, "name": GATE, "status": "in_progress", "conclusion": None,
         "started_at": "2026-01-01T00:05:00Z", "completed_at": None, "runner_name": "runner-gate",
         "steps": [_step("Set up job", "2026-01-01T00:05:00Z", "2026-01-01T00:05:01Z"),
                   _step("Evaluate job results", "2026-01-01T00:05:01Z", "2026-01-01T00:05:04Z"),
                   _step("Collect timings and generate report", "2026-01-01T00:05:04Z", None)]},
        {"id": 3, "name": "Still running elsewhere", "status": "in_progress", "conclusion": None,
         "started_at": "2026-01-01T00:04:00Z", "completed_at": None, "runner_name": "runner-other",
         "steps": [_step("Set up job", "2026-01-01T00:04:00Z", "2026-01-01T00:04:01Z")]},
    ]


def _collect(monkeypatch, current_runner: str) -> dict:
    def fake_api_get(path, token, params=None, list_key=None):
        if path.endswith("/runs/9"):
            return {"created_at": "2026-01-01T00:00:00Z", "head_branch": "main", "event": "push"}
        if path.endswith("/runs/9/jobs"):
            return _jobs()
        return []

    monkeypatch.setattr(timings_report, "api_get", fake_api_get)
    monkeypatch.setattr(timings_report, "_collect_cache_data", lambda jobs, repo, token: {})
    return timings_report.collect_timings("t", "o/r", "9", "sha", current_runner=current_runner)


def test_the_running_gate_counts_up_to_its_last_finished_step(monkeypatch):
    jobs = {job["name"]: job for job in _collect(monkeypatch, "runner-gate")["jobs"]}
    assert set(jobs) == {"Detect affected areas", GATE}, "only this job's in-progress entry counts"
    gate = jobs[GATE]
    assert gate["completed_at"] == "2026-01-01T00:05:04Z"
    assert gate["duration_s"] == 4
    assert gate["conclusion"] == "success"
    assert [step["name"] for step in gate["steps"]] == ["Set up job", "Evaluate job results"]
    stats = timings_report.compute_stats({"jobs": list(jobs.values())})
    assert stats["wall"] == 304, "wall time runs to the gate's verdict, not the last worker"


def test_a_failed_verdict_stays_failed(monkeypatch):
    jobs = _jobs()
    jobs[1]["steps"][1]["conclusion"] = "failure"
    closed = timings_report.close_running_job(jobs[1])
    assert closed is not None and closed["conclusion"] == "failure"


def test_without_a_runner_name_no_unfinished_job_counts(monkeypatch):
    names = [job["name"] for job in _collect(monkeypatch, "")["jobs"]]
    assert names == ["Detect affected areas"]


def test_the_running_job_is_not_asked_for_its_log(monkeypatch):
    """The jobs API has no log for a job that is still running: fetching it marks the gate
    "unavailable" in every report's cache section."""
    fetched = []

    def fake_job_log(job_id, repo, token):
        fetched.append(job_id)
        if job_id == 2:
            raise OSError("job log HTTP 404")
        return ""

    def fake_api_get(path, token, params=None, list_key=None):
        if path.endswith("/runs/9"):
            return {"created_at": "2026-01-01T00:00:00Z"}
        return _jobs() if path.endswith("/runs/9/jobs") else []

    monkeypatch.setattr(timings_report, "api_get", fake_api_get)
    monkeypatch.setattr(timings_report, "_job_log", fake_job_log)
    data = timings_report.collect_timings("t", "o/r", "9", "sha", current_runner="runner-gate")
    assert GATE in {job["name"] for job in data["jobs"]}
    assert data["cache_data"]["unavailable_jobs"] == []
    assert fetched == [1]
