"""Tests for the cron → workflow-dispatch branch (hermes-cron-e2e-005).

Covers:

* ``extract_workflow_ids`` tolerates missing metadata, string scalars,
  empty/whitespace entries, and non-list metadata.
* ``resolve_dispatch_url`` honours the explicit env var ahead of the
  base-url fallback.
* ``_run_job_impl`` short-circuits to ``dispatch_workflow_runs`` when
  ``metadata.workflow_ids`` is non-empty AND a dispatch URL is set.
* When ``workflow_ids`` is set but no dispatch URL exists, the
  legacy prompt-spawn path is preserved (backward compatibility).
* Plain-prompt jobs (no metadata) never touch the dispatch branch.

The dispatch HTTP call is patched so tests don't open sockets.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_WORKFLOW_DISPATCH_URL", raising=False)
    monkeypatch.delenv("AGENTE_DESKTOP_BASE_URL", raising=False)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.workflow_dispatch
    importlib.reload(cron.workflow_dispatch)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# extract_workflow_ids
# ---------------------------------------------------------------------------


def test_extract_workflow_ids_missing(hermes_env):
    from cron.workflow_dispatch import extract_workflow_ids
    assert extract_workflow_ids({}) == []
    assert extract_workflow_ids({"workflow_ids": None}) == []
    assert extract_workflow_ids({"metadata": None}) == []
    assert extract_workflow_ids({"metadata": "not-a-dict"}) == []


def test_extract_workflow_ids_top_level_canonical(hermes_env):
    """Canonical shape — what create_job / /api/jobs actually stores."""
    from cron.workflow_dispatch import extract_workflow_ids
    assert extract_workflow_ids({"workflow_ids": ["wf-1", "wf-2"]}) == ["wf-1", "wf-2"]
    assert extract_workflow_ids({"workflow_ids": "wf-abc"}) == ["wf-abc"]
    assert extract_workflow_ids({"workflow_ids": ["wf-1", " wf-2 ", "", "wf-3"]}) == [
        "wf-1", "wf-2", "wf-3",
    ]


def test_extract_workflow_ids_metadata_fallback(hermes_env):
    """Forward-compat: jobs that stash workflow_ids under metadata still work."""
    from cron.workflow_dispatch import extract_workflow_ids
    assert extract_workflow_ids({"metadata": {"workflow_ids": ["wf-meta"]}}) == ["wf-meta"]
    assert extract_workflow_ids({"metadata": {"workflow_ids": "wf-meta"}}) == ["wf-meta"]


def test_extract_workflow_ids_empty(hermes_env):
    from cron.workflow_dispatch import extract_workflow_ids
    assert extract_workflow_ids({"workflow_ids": []}) == []
    assert extract_workflow_ids({"workflow_ids": ["", "  "]}) == []
    assert extract_workflow_ids({"metadata": {"workflow_ids": []}}) == []


def test_extract_workflow_ids_top_level_wins_over_metadata(hermes_env):
    """If both are set, top-level (canonical) wins — keeps shape unambiguous."""
    from cron.workflow_dispatch import extract_workflow_ids
    job = {"workflow_ids": ["canonical"], "metadata": {"workflow_ids": ["fallback"]}}
    assert extract_workflow_ids(job) == ["canonical"]


# ---------------------------------------------------------------------------
# resolve_dispatch_url
# ---------------------------------------------------------------------------


def test_resolve_dispatch_url_explicit(hermes_env, monkeypatch):
    monkeypatch.setenv("HERMES_WORKFLOW_DISPATCH_URL", "http://example/dispatch")
    from cron.workflow_dispatch import resolve_dispatch_url
    assert resolve_dispatch_url() == "http://example/dispatch"


def test_resolve_dispatch_url_base_fallback(hermes_env, monkeypatch):
    monkeypatch.setenv("AGENTE_DESKTOP_BASE_URL", "http://desktop:9090/")
    # Companion AGENTE support is provided by adapter patch (core resolve is HERMES-only).
    from gateway.agente_desktop_adapter import workflow_routine_bridge as br
    br._patch_cron_workflow_dispatch()
    from cron.workflow_dispatch import resolve_dispatch_url
    assert resolve_dispatch_url() == "http://desktop:9090/api/workflow-runs"


def test_resolve_dispatch_url_none(hermes_env):
    from cron.workflow_dispatch import resolve_dispatch_url
    assert resolve_dispatch_url() is None


# ---------------------------------------------------------------------------
# _run_job_impl branch
# ---------------------------------------------------------------------------


def _make_workflow_job(**overrides):
    base = {
        "id": "job-abc123",
        "name": "morning-triage",
        "prompt": "should be ignored when workflow_ids set",
        "schedule": "every 5m",
        "deliver": "local",
        # Top-level shape mirrors what ``cron.jobs.create_job`` actually
        # writes (see ``_normalize_workflow_ids`` + the persisted job
        # dict at ``cron/jobs.py``).
        "workflow_ids": ["wf-triage-001"],
    }
    base.update(overrides)
    return base


def test_run_job_dispatches_workflow_when_url_configured(hermes_env, monkeypatch):
    monkeypatch.setenv("HERMES_WORKFLOW_DISPATCH_URL", "http://desktop/api/workflow-runs")

    import cron.scheduler as scheduler
    import cron.workflow_dispatch as wf

    captured = []

    def fake_post(url, payload, timeout):
        captured.append((url, payload, timeout))
        return True, '{"run_id": "run-1"}'

    monkeypatch.setattr(wf, "_post_json", fake_post)

    success, doc, final_response, err = scheduler.run_job(_make_workflow_job())

    assert success is True
    assert err is None
    assert "workflow-dispatch" in doc
    assert "wf-triage-001" in doc
    assert "Workflow run(s) started" in final_response
    assert len(captured) == 1
    url, payload, _ = captured[0]
    assert url == "http://desktop/api/workflow-runs"
    assert payload["workflow_id"] == "wf-triage-001"
    assert payload["trigger"]["source"] == "hermes_cron"
    assert payload["trigger"]["job_id"] == "job-abc123"


def test_run_job_dispatch_reports_partial_failure(hermes_env, monkeypatch):
    monkeypatch.setenv("HERMES_WORKFLOW_DISPATCH_URL", "http://desktop/api/workflow-runs")

    import cron.scheduler as scheduler
    import cron.workflow_dispatch as wf

    calls = {"n": 0}

    def fake_post(url, payload, timeout):
        calls["n"] += 1
        if payload["workflow_id"] == "wf-good":
            return True, '{"run_id": "ok"}'
        return False, "HTTP 500: boom"

    monkeypatch.setattr(wf, "_post_json", fake_post)

    job = _make_workflow_job(workflow_ids=["wf-good", "wf-bad"])
    success, doc, _, err = scheduler.run_job(job)

    assert success is False
    assert "wf-bad" in err
    assert "boom" in err
    assert calls["n"] == 2  # both dispatched, neither short-circuits


def test_run_job_falls_back_to_legacy_when_no_dispatch_url(hermes_env, caplog):
    # Workflow_ids set, but no dispatch URL → must NOT short-circuit.
    # The fallback path enters the legacy LLM machinery; in test envs
    # that machinery raises an auth error (no provider configured). The
    # contract we assert is: (a) the fallback warning is emitted, and
    # (b) execution proceeds past the dispatch branch (i.e. raises a
    # downstream error, not silently succeeds).
    import cron.scheduler as scheduler

    with caplog.at_level("WARNING"):
        success = None
        try:
            success, _, _, _ = scheduler.run_job(_make_workflow_job())
        except Exception:
            success = False

    assert any(
        "workflow_ids" in rec.message and "falling back" in rec.message
        for rec in caplog.records
    ), "expected fallback warning when workflow_ids set but no dispatch URL"
    # Either an exception bubbled (legacy auth path) or the job reported
    # failure — what must NOT happen is silent success via the dispatch
    # branch (which would mean the workflow ran without ever hitting the
    # workflow engine).
    assert success is not True


def test_run_job_plain_prompt_unaffected(hermes_env, monkeypatch):
    """Backward compat: jobs without workflow_ids must never touch dispatch."""
    import cron.scheduler as scheduler
    import cron.workflow_dispatch as wf

    def fail_post(*a, **kw):  # pragma: no cover - must not be called
        raise AssertionError("dispatch must not fire for plain-prompt jobs")

    monkeypatch.setattr(wf, "_post_json", fail_post)
    monkeypatch.setenv("HERMES_WORKFLOW_DISPATCH_URL", "http://should-not-be-used")

    class FakeAIAgent:
        def __init__(self, *a, **kw):
            raise RuntimeError("expected: plain-prompt path attempted")

    monkeypatch.setattr("run_agent.AIAgent", FakeAIAgent)

    plain = {
        "id": "job-plain",
        "name": "plain",
        "prompt": "hi",
        "schedule": "every 5m",
        "deliver": "local",
    }
    try:
        scheduler.run_job(plain)
    except Exception as exc:
        assert "plain-prompt path attempted" in str(exc)


# ---------------------------------------------------------------------------
# /api/jobs end-to-end regression — exercises the real stored job shape.
# ---------------------------------------------------------------------------


def test_api_jobs_top_level_workflow_ids_short_circuits_dispatch(hermes_env, monkeypatch):
    """End-to-end regression for hermes-cron-e2e-005.

    Create a job via the same code path ``/api/jobs`` calls
    (``cron.jobs.create_job``) with the top-level ``workflow_ids``
    argument the Desktop ScheduleDispatcher actually sends, then run
    it through ``scheduler.run_job`` and prove the dispatch branch
    fires — i.e. the cron → workflow connection works against the
    real persisted shape, not a synthetic ``metadata`` payload.
    """
    monkeypatch.setenv("HERMES_WORKFLOW_DISPATCH_URL", "http://desktop/api/workflow-runs")

    from cron.jobs import create_job
    import cron.scheduler as scheduler
    import cron.workflow_dispatch as wf

    captured = []

    def fake_post(url, payload, timeout):
        captured.append(payload["workflow_id"])
        return True, '{"run_id": "ok"}'

    monkeypatch.setattr(wf, "_post_json", fake_post)

    job = create_job(
        prompt="never reached",
        schedule="every 5m",
        name="routine-morning-triage",
        deliver="local",
        workflow_ids=["wf-triage-a", "wf-triage-b"],
    )
    assert job["workflow_ids"] == ["wf-triage-a", "wf-triage-b"]
    assert job.get("metadata") is None  # confirm canonical shape

    success, _, final, err = scheduler.run_job(job)
    assert success is True
    assert err is None
    assert captured == ["wf-triage-a", "wf-triage-b"]
    assert "wf-triage-a" in final and "wf-triage-b" in final


def test_api_server_forwards_workflow_ids_to_cron_create(hermes_env):
    """``POST /api/jobs`` body workflow_ids[] must reach ``create_job``.

    Confirms the api_server.py passthrough — without it, the Desktop
    ScheduleDispatcher's request lands a job with workflow_ids=None
    even when the body had a populated list.
    """
    import inspect
    from gateway.platforms import api_server

    src = inspect.getsource(api_server)
    assert 'body.get("workflow_ids")' in src, (
        "api_server.py must forward request body workflow_ids[] to _cron_create — "
        "otherwise routine → cron → workflow dispatch is broken end-to-end."
    )
