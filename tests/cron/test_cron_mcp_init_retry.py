"""Cron MCP-init race (reported 2026-10-16): bounded retry + doctor stamp.

A cold cron worker registers MCP servers before building the agent; its first registration can
race the MCP loop thread it just started — ``_ensure_mcp_loop()`` returns before ``run_forever``
is actually on the thread, so ``_run_on_mcp_loop`` raises "MCP event loop is not running". The
init failure is non-fatal, so the run still books green and the whole tick runs with no MCP tools.

Fix under test: ``_init_cron_mcp_tools`` retries that race once, and every init outcome is stamped
on the job record (``last_mcp_init_error``), which ``hermes cron doctor`` reports until a later
successful init clears it.
"""
from unittest.mock import patch

import pytest

from cron import jobs as cron_jobs

RACE = "MCP event loop is not running"


@pytest.fixture()
def cron_store(tmp_path):
    with cron_jobs.use_cron_store(tmp_path):
        yield tmp_path


def _seed_job(job_id="mcpjob"):
    cron_jobs.save_jobs([{"id": job_id, "name": job_id, "prompt": "probe",
                          "schedule": "every 15m", "enabled": True}])


def _init(job_id="mcpjob"):
    from cron.scheduler import _init_cron_mcp_tools
    _init_cron_mcp_tools(job_id)


def test_loop_start_race_is_retried_once(cron_store):
    _seed_job()
    calls = []

    def discover():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(RACE)
        return ["mcp__demo__get_portfolio"]

    with patch("tools.mcp_tool_discovery.discover_mcp_tools", side_effect=discover), \
         patch("cron.scheduler._CRON_MCP_INIT_RETRY_DELAY_S", 0):
        _init()
    assert len(calls) == 2
    assert "last_mcp_init_error" not in cron_jobs.get_job("mcpjob")


def test_persistent_race_failure_is_stamped_for_the_doctor(cron_store):
    _seed_job()
    calls = []

    def discover():
        calls.append(1)
        raise RuntimeError(RACE)

    with patch("tools.mcp_tool_discovery.discover_mcp_tools", side_effect=discover), \
         patch("cron.scheduler._CRON_MCP_INIT_RETRY_DELAY_S", 0):
        _init()
    assert len(calls) == 2  # one retry, then the failure is recorded
    err = cron_jobs.get_job("mcpjob")["last_mcp_init_error"]
    assert RACE in err["detail"] and err["at"]


def test_other_failures_are_not_retried(cron_store):
    _seed_job()
    calls = []

    def discover():
        calls.append(1)
        raise RuntimeError("demo: connection refused")

    with patch("tools.mcp_tool_discovery.discover_mcp_tools", side_effect=discover), \
         patch("cron.scheduler._CRON_MCP_INIT_RETRY_DELAY_S", 0):
        _init()
    assert len(calls) == 1
    assert "connection refused" in cron_jobs.get_job("mcpjob")["last_mcp_init_error"]["detail"]


def test_successful_init_clears_the_stamp(cron_store):
    _seed_job()
    cron_jobs.note_mcp_init_failure("mcpjob", RACE)
    assert cron_jobs.get_job("mcpjob")["last_mcp_init_error"]
    with patch("tools.mcp_tool_discovery.discover_mcp_tools",
               return_value=["mcp__demo__get_portfolio"]):
        _init()
    assert "last_mcp_init_error" not in cron_jobs.get_job("mcpjob")


def test_doctor_reports_the_stamp():
    from hermes_cli.cron import _cron_doctor_issues_for_job
    job = {"id": "mcpjob", "name": "mcpjob", "enabled": True, "state": "scheduled",
           "last_mcp_init_error": {"at": "2026-10-16T10:15:49+00:00", "detail": RACE}}
    issues = _cron_doctor_issues_for_job(job)
    assert any("MCP init failed at 2026-10-16T10:15:49+00:00" in i and RACE in i for i in issues)
