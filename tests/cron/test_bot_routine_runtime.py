from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_cli.bot_catalog import BotRoutine, resolve_bot_catalog_entry
from hermes_cli.bot_routines import activate_bot_routine


def test_bot_routine_runs_through_scheduler_with_workflow_tools(tmp_path: Path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "timezone: UTC\n"
        "model:\n  provider: openrouter\n  default: test/model\n"
        "platform_toolsets:\n  cron: [hermes-cron]\n",
        encoding="utf-8",
    )
    (home / "profile.yaml").write_text("routines: []\n", encoding="utf-8")
    entry = resolve_bot_catalog_entry("research-analyst", include_live=False).model_copy(update={
        "routines": [BotRoutine(
            id="safe-probe", name="Safe probe", prompt="Return the word ok.", schedule="1h",
        )]
    })

    from cron import jobs
    from cron.scheduler import run_job
    from model_tools import get_tool_definitions

    with jobs.use_cron_store(home):
        activated = activate_bot_routine(
            entry, "safe-probe", schedule="1h", timezone_name="UTC",
            destination="local", setup_ready=True,
        )
        job = jobs.get_job(activated["job_id"])
        assert job is not None
        exposed: set[str] = set()

        def agent_factory(**kwargs):
            definitions = get_tool_definitions(
                enabled_toolsets=kwargs["enabled_toolsets"],
                disabled_toolsets=kwargs["disabled_toolsets"],
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
            exposed.update(
                str(item.get("function", {}).get("name") or "")
                for item in definitions
                if isinstance(item, dict)
            )
            agent = MagicMock()
            agent.run_conversation.return_value = {"final_response": "ok"}
            return agent

        fake_db = MagicMock()
        patches = [
            patch("cron.scheduler._hermes_home", home),
            patch("cron.scheduler_delivery._resolve_origin", return_value=None),
            patch("hermes_cli.env_loader.load_hermes_dotenv"),
            patch("hermes_cli.env_loader.reset_secret_source_cache"),
            patch("hermes_state_registry.acquire", return_value=fake_db),
            patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]),
            patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": "openrouter",
                "api_mode": "chat_completions",
            }),
            patch("run_agent.AIAgent", side_effect=agent_factory),
        ]
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            success, _output, final_response, error = run_job(job)

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert {"web_search", "skill_view", "read_file"} <= exposed
