"""Tests for cron jobs that intentionally need the messaging toolset."""

from __future__ import annotations


def test_cron_disables_messaging_by_default():
    from cron.scheduler import _resolve_cron_disabled_toolsets

    disabled = _resolve_cron_disabled_toolsets({})

    assert "cronjob" in disabled
    assert "clarify" in disabled
    assert "messaging" in disabled


def test_cron_allows_messaging_when_job_explicitly_requests_it():
    from cron.scheduler import _resolve_cron_disabled_toolsets

    disabled = _resolve_cron_disabled_toolsets({}, {"enabled_toolsets": ["web", "messaging"]})

    assert "cronjob" in disabled
    assert "clarify" in disabled
    assert "messaging" not in disabled


def test_cron_allows_messaging_when_legacy_single_arg_contains_enabled_toolsets():
    from cron.scheduler import _resolve_cron_disabled_toolsets

    disabled = _resolve_cron_disabled_toolsets({"enabled_toolsets": ["web", "messaging"]})

    assert "messaging" not in disabled


def test_cron_global_config_enabled_toolsets_does_not_grant_messaging_to_all_jobs():
    from cron.scheduler import _resolve_cron_disabled_toolsets

    disabled = _resolve_cron_disabled_toolsets(
        {"enabled_toolsets": ["web", "messaging"]},
        {"enabled_toolsets": ["web"]},
    )

    assert "messaging" in disabled


def test_cron_prompt_allows_explicit_router_jobs_to_use_send_message():
    from cron.scheduler import _build_job_prompt

    prompt = _build_job_prompt({
        "prompt": "Route Ziv items with send_message, then return [SILENT].",
        "enabled_toolsets": ["web", "messaging"],
    })

    assert "You may use send_message only when this job's prompt explicitly asks" in prompt
    assert "do NOT use send_message" not in prompt
