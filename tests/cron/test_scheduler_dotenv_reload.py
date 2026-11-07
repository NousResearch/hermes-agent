"""Cron dotenv reload must expose newly available tools (#82912)."""

import asyncio
import contextlib
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model_tools


_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


def _run_owned_job(job, tmp_path, db=None):
    """Run the production owner bridge, preserving each test's agent/DB probes."""
    from gateway.session_contract import SessionRef
    from gateway.session_cron import current_execution, execute
    from hermes_state_registry import acquire, release

    ref = SessionRef("test-profile", "canonical-cron-session")
    admission_id = "cron-owner-admission"
    request_id = "cron-owner-request"
    owns_db = db is None
    if owns_db:
        db = acquire(tmp_path / "state.db")
        db.create_session(ref.session_id, source="cron")
    else:
        db.db_path = str(tmp_path / "state.db")
    authority = SimpleNamespace(
        db=db,
        sessions={ref.session_id: SimpleNamespace(source=SimpleNamespace(user_id="cron-owner"))},
        pending_results={},
    )
    row = {
        "admission_id": admission_id, "request_id": request_id,
        "principal_id": "cron-owner", "payload": {"text": ""},
    }
    policy = SimpleNamespace(request_json=json.dumps({
        "cron_job": job, "extra_prompt": None, "request_id": request_id,
    }))

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
        assert authority._cron_cancellations == {}
        return tuple(authority.pending_results[admission_id]["result"]["cron_result"])

    try:
        return asyncio.run(run())
    finally:
        if owns_db:
            release(db)


def test_tool_definitions_cache_is_cleared_after_dotenv_reload(tmp_path, monkeypatch):
    """A cron dotenv reload must expose newly available tools (#82912).

    Seed the real quiet-mode cache before ``TAVILY_API_KEY`` exists, then
    run a cron job whose real tool-definition lookup happens after the
    reload. This verifies the user-visible agent toolset rather than only
    inspecting the private cache dictionary.
    """
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: gpt-4o-mini-cron-test\n"
        "web:\n  backend: tavily\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    from tools import web_tools
    from tools.registry import invalidate_check_fn_cache

    # Keep the test independent of packages or credentials installed on
    # the developer machine. Tavily remains the only availability signal.
    def tavily_only_backend_available(backend):
        return backend == "tavily" and bool(os.environ.get("TAVILY_API_KEY"))

    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    with contextlib.ExitStack() as cleanup, patch.object(
        web_tools,
        "_is_backend_available",
        side_effect=tavily_only_backend_available,
    ), patch(
        "agent.web_search_registry.get_active_search_provider",
        return_value=None,
    ), patch(
        "agent.web_search_registry.get_active_extract_provider",
        return_value=None,
    ):
        cleanup.callback(invalidate_check_fn_cache)
        cleanup.callback(model_tools._clear_tool_defs_cache)
        stale_tools = model_tools.get_tool_definitions(
            enabled_toolsets=["web", "file"], quiet_mode=True
        )
        stale_names = {tool["function"]["name"] for tool in stale_tools}
        assert not {"web_search", "web_extract"} & stale_names
        file_names = {"patch", "read_file", "search_files", "write_file"}
        assert file_names <= stale_names
        (tmp_path / ".env").write_text("TAVILY_API_KEY=cron-test-key\n")

        created_agents = []

        def build_agent(**kwargs):
            agent = MagicMock()
            agent.tools = model_tools.get_tool_definitions(
                enabled_toolsets=kwargs["enabled_toolsets"],
                disabled_toolsets=kwargs["disabled_toolsets"],
                quiet_mode=kwargs["quiet_mode"],
            )
            agent.run_conversation.return_value = {"final_response": "ok"}
            created_agents.append(agent)
            return agent

        job = {
            "id": "cache-clear-job",
            "name": "cache test",
            "prompt": "hi",
            "enabled_toolsets": ["web", "file"],
        }
        fake_db = MagicMock()

        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler._resolve_delivery_target", return_value=None), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=fake_db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value=_RUNTIME), \
             patch("run_agent.AIAgent", side_effect=build_agent) as mock_agent_cls:
            success, _, _, error = _run_owned_job(job, tmp_path, fake_db)

    assert success is True
    assert error is None
    mock_agent_cls.assert_called_once()
    assert len(created_agents) == 1
    tool_names = {tool["function"]["name"] for tool in created_agents[0].tools}
    assert {"web_search", "web_extract"} | file_names <= tool_names
