"""Regression tests for #109050: a per-job ``enabled_toolsets`` entry naming an MCP server that
resolves to zero tools must refuse the run through the blocked_config path instead of letting the
agent run tool-less while the scheduler records success.

Covers the guard added in ``cron.scheduler._verify_cron_mcp_toolsets`` (wired at the end of
``_resolve_cron_agent_setup``, after MCP discovery): fail-closed on a named server with zero
tools, silent re-block via ``last_status`` dedup, opt-out via ``cron.preflight``, fail-open on
verifier errors, and the unknown-name / no-per-job-list exclusions.
"""

from __future__ import annotations

import cron.scheduler as scheduler


def _job(**overrides):
    job = {
        "id": "job-1",
        "name": "notion sync",
        "prompt": "sync notes",
        "enabled_toolsets": ["file", "terminal", "notion"],
    }
    job.update(overrides)
    return job


def _alias_mcp_server(monkeypatch, mapping):
    from tools.registry import registry

    monkeypatch.setattr(
        registry, "get_toolset_alias_target", lambda alias: mapping.get(alias)
    )


def _resolve_tools(monkeypatch, mapping):
    import toolsets

    monkeypatch.setattr(
        toolsets, "resolve_toolset", lambda name, *a, **k: list(mapping.get(name, []))
    )


def test_named_mcp_server_with_zero_tools_blocks(monkeypatch):
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(
        monkeypatch, {"file": ["read_file"], "terminal": ["terminal"], "notion": []}
    )

    result = scheduler._verify_cron_mcp_toolsets(_job(), "job-1", "notion sync", {})

    assert result is not None
    success, blocked_doc, final_response, error = result
    assert success is False
    assert final_response == ""
    assert error.startswith("[blocked_config] ")
    assert "'notion'" in error
    assert "BLOCKED (configuration)" in blocked_doc
    assert "'notion'" in blocked_doc


def test_named_mcp_server_with_tools_passes(monkeypatch):
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(
        monkeypatch, {"notion": ["mcp__notion__search", "mcp__notion__create"]}
    )

    assert (
        scheduler._verify_cron_mcp_toolsets(_job(), "job-1", "notion sync", {}) is None
    )


def test_unknown_name_without_alias_is_not_guarded(monkeypatch):
    # A typo'd or unknown toolset stays the -t/quiet-warning domain (#94151): only names whose
    # registry alias targets an mcp-* toolset are MCP servers.
    _alias_mcp_server(monkeypatch, {})
    _resolve_tools(monkeypatch, {})

    assert (
        scheduler._verify_cron_mcp_toolsets(_job(), "job-1", "notion sync", {}) is None
    )


def test_no_per_job_toolsets_skips_check(monkeypatch):
    # Servers layered in by the global MCP merge are not the job's intent statement.
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(monkeypatch, {"notion": []})

    job = _job(enabled_toolsets=None)
    assert scheduler._verify_cron_mcp_toolsets(job, "job-1", "notion sync", {}) is None


def test_verifier_errors_fail_open(monkeypatch):
    from tools.registry import registry

    def _boom(alias):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry, "get_toolset_alias_target", _boom)

    assert (
        scheduler._verify_cron_mcp_toolsets(_job(), "job-1", "notion sync", {}) is None
    )


def test_reblock_after_blocked_config_is_silent(monkeypatch):
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(monkeypatch, {"notion": []})

    result = scheduler._verify_cron_mcp_toolsets(
        _job(last_status="blocked_config"), "job-1", "notion sync", {}
    )

    assert result is not None
    assert result[3].startswith("[blocked_config:silent] ")


def test_preflight_opt_out_disables_guard(monkeypatch):
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(monkeypatch, {"notion": []})

    cfg = {"cron": {"preflight": False}}
    assert (
        scheduler._verify_cron_mcp_toolsets(_job(), "job-1", "notion sync", cfg) is None
    )


def test_setup_wiring_blocks_run_after_mcp_discovery(monkeypatch):
    # End of _resolve_cron_agent_setup: after _init_cron_mcp_tools runs, a named MCP server with
    # zero tools must surface as setup.blocked so run_job refuses the dispatch (no LLM call).
    _alias_mcp_server(monkeypatch, {"notion": "mcp-notion"})
    _resolve_tools(monkeypatch, {"file": ["read_file"], "notion": []})

    init_calls: list[str] = []
    monkeypatch.setattr(scheduler, "_load_prefill_messages", lambda cfg, job_id: None)
    monkeypatch.setattr(scheduler, "_guard_job_credential_exfil", lambda job: None)
    monkeypatch.setattr(
        scheduler, "_preflight_or_block", lambda job, job_id, job_name, cfg: None
    )
    monkeypatch.setattr(
        scheduler,
        "_resolve_job_runtime",
        lambda job, job_id, jc: ({"provider": "test"}, "m"),
    )
    monkeypatch.setattr(
        scheduler, "_resolve_job_reasoning_config", lambda job, cfg, model: None
    )
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    monkeypatch.setattr(
        scheduler, "_load_credential_pool", lambda runtime, job_id: None
    )
    monkeypatch.setattr(
        scheduler, "_init_cron_mcp_tools", lambda job_id: init_calls.append(job_id)
    )

    jc = scheduler._CronJobConfig(
        cfg={}, model="m", model_cfg={}, cron_default_provider=""
    )
    setup = scheduler._resolve_cron_agent_setup(_job(), "job-1", "notion sync", jc)

    assert init_calls == [
        "job-1"
    ]  # verification happens after MCP discovery, not before
    assert setup.blocked is not None
    assert setup.blocked[0] is False
    assert "'notion'" in setup.blocked[3]
