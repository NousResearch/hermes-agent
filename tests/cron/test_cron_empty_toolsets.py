"""Test that an empty enabled_toolsets list ([]) is respected and not overwritten
by default platform toolsets.
"""

from __future__ import annotations

import cron.jobs as cron_jobs
from cron.scheduler import _resolve_cron_enabled_toolsets


def test_cron_create_and_update_preserves_empty_enabled_toolsets(tmp_path):
    with cron_jobs.use_cron_store(tmp_path):
        job = cron_jobs.create_job(
            name="No Tools Job",
            prompt="Simple summary",
            schedule="every 1h",
            enabled_toolsets=[],
        )
        assert job["enabled_toolsets"] == []

        loaded = cron_jobs.get_job(job["id"])
        assert loaded is not None
        assert loaded["enabled_toolsets"] == []

        updated = cron_jobs.update_job(job["id"], {"enabled_toolsets": []})
        assert updated is not None
        assert updated["enabled_toolsets"] == []


def test_resolve_cron_enabled_toolsets_with_empty_list():
    job = {"enabled_toolsets": []}
    cfg = {"platform_toolsets": {"cron": ["terminal", "web_search", "file"]}}
    
    # An explicit empty list should return empty list (plus any MCP if configured),
    # NOT the default platform tools
    result = _resolve_cron_enabled_toolsets(job, cfg)
    assert result == []


def test_resolve_cron_enabled_toolsets_with_none_uses_platform_defaults():
    job = {"enabled_toolsets": None}
    cfg = {"platform_toolsets": {"cron": ["terminal", "file"]}}
    
    result = _resolve_cron_enabled_toolsets(job, cfg)
    assert "terminal" in result
    assert "file" in result


def test_cron_string_list_and_dashboard_normalization_preserves_empty_list(tmp_path):
    from hermes_cli.web_server_cron import _cron_string_list
    from hermes_cli.web_routers.cron import _normalize_dashboard_cron_updates
    
    # allow_empty=True preserves explicit empty list []
    assert _cron_string_list([], allow_empty=True) == []
    # absent or default allow_empty=False still coalesces to None
    assert _cron_string_list([]) is None
    
    # Dashboard normalization preserves enabled_toolsets=[]
    normalized = _normalize_dashboard_cron_updates({"enabled_toolsets": []}, tmp_path)
    assert normalized.get("enabled_toolsets") == []


def test_cronjob_tool_create_and_update_preserves_empty_enabled_toolsets(tmp_path):
    import json
    from tools.cronjob_tools import cronjob
    
    with cron_jobs.use_cron_store(tmp_path):
        res_raw = cronjob(
            action="create",
            name="Tool-less Job",
            prompt="Do nothing",
            schedule="every 2h",
            enabled_toolsets=[],
        )
        res = json.loads(res_raw)
        assert res.get("success") is True
        job_id = res["job_id"]
        
        stored = cron_jobs.get_job(job_id)
        assert stored is not None
        assert stored.get("enabled_toolsets") == []
        
        # Updating with empty list still preserves []
        res_up_raw = cronjob(
            action="update",
            job_id=job_id,
            enabled_toolsets=[],
        )
        res_up = json.loads(res_up_raw)
        assert res_up.get("success") is True
        
        stored_up = cron_jobs.get_job(job_id)
        assert stored_up.get("enabled_toolsets") == []
