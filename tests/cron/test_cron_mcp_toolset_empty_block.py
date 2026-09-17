"""A cron job whose ``enabled_toolsets`` names an MCP server that resolves to zero tools is
blocked as ``blocked_config`` instead of running tool-less and booking success (#109050).

Under a multiplexer the server's toolset alias is process-global while its tools live in the
discovering profile's registry overlay, so another profile's job sees the name but no tools.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cron.jobs as cron_jobs


def _run_owned_job(job, tmp_path):
    """Run through the production owner bridge: cron turns execute inside the gateway owner."""
    from gateway.session_contract import SessionRef
    from gateway.session_cron import current_execution, execute
    from hermes_state_registry import acquire, release

    ref = SessionRef("test-profile", "cron-owner-session")
    admission_id = "cron-owner-admission"
    request_id = "cron-owner-request"
    db = acquire(tmp_path / "state.db")
    db.create_session(ref.session_id, source="cron")
    authority = SimpleNamespace(
        db=db,
        sessions={ref.session_id: SimpleNamespace(source=SimpleNamespace(user_id="cron-owner"))},
        pending_results={},
    )
    row = {"admission_id": admission_id, "request_id": request_id,
           "principal_id": "cron-owner", "payload": {"text": ""}}
    policy = SimpleNamespace(request_json=json.dumps({
        "cron_job": job, "extra_prompt": None, "request_id": request_id}))

    async def run():
        previous = current_execution()
        try:
            await execute(authority, ref, row, policy)
        except RuntimeError as exc:
            saved = authority.pending_results.get(admission_id)
            if saved is None:
                raise
            assert str(exc) == saved["result"]["cron_result"][3]
        assert current_execution() is previous
        return tuple(authority.pending_results[admission_id]["result"]["cron_result"])

    try:
        return asyncio.run(run())
    finally:
        release(db)

_RUNTIME = {"api_key": "k", "base_url": "https://example.invalid/v1", "provider": "openrouter",
            "api_mode": "chat_completions"}


def _job(**overrides):
    job = {
        "id": "mcpjob", "name": "mcp job", "prompt": "hello", "enabled": True, "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"}, "deliver": "local",
        "model": None, "provider": None, "base_url": None,
    }
    job.update(overrides)
    return job


def _run(job, tmp_path):
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: test-model\nmcp_servers:\n  notion:\n    url: https://mcp.invalid\n", encoding="utf-8")
    with patch("run_agent.AIAgent") as agent_cls, \
         patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=dict(_RUNTIME)):
        agent_cls.return_value.run_conversation.return_value = {"final_response": "ok"}
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            result = _run_owned_job(job, tmp_path)
        return result, agent_cls.called


def _register_notion_in_scope(scope):
    from tools.registry import registry
    registry.register(
        name="mcp__notion__search", toolset="mcp-notion",
        schema={"name": "mcp__notion__search", "description": "x",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda a, **k: "{}", scope=scope)
    registry.register_toolset_alias("notion", "mcp-notion")
    return lambda: registry.deregister("mcp__notion__search", scope=scope)


def test_requested_mcp_server_owned_by_other_profile_blocks_run(tmp_path):
    from agent.secret_scope import set_multiplex_active
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    set_multiplex_active(True)
    token = set_hermes_home_override(tmp_path / "other")
    try:
        undo = _register_notion_in_scope(hermes_home_key())
    finally:
        reset_hermes_home_override(token)
    try:
        (success, _output, _final, error), agent_built = _run(
            _job(enabled_toolsets=["terminal", "notion"]), tmp_path)
    finally:
        undo()
        set_multiplex_active(False)

    assert agent_built is False
    assert success is False
    assert error is not None and "[blocked_config]" in error and "notion" in error
    # The reason is operator-facing copy: it must say the block re-evaluates itself so a
    # transient outage is not mistaken for a config error to repair by hand (#112871).
    assert "clears by itself" in error


def test_requested_mcp_server_with_tools_runs(tmp_path):
    undo = _register_notion_in_scope(None)
    try:
        (success, _output, _final, error), agent_built = _run(
            _job(enabled_toolsets=["terminal", "notion"]), tmp_path)
    finally:
        undo()

    assert agent_built is True
    assert success is True and error is None
