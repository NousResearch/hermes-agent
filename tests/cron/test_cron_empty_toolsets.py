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
