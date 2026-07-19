"""Regression coverage for model-aware cron runtime resolution.

OpenCode Go serves models through multiple API surfaces. The cron scheduler
must resolve the runtime from the model the job will run, rather than an
unrelated ``model.default`` value.
"""

from unittest.mock import MagicMock, patch


def test_cron_agent_uses_pinned_model_for_opencode_runtime(tmp_path, monkeypatch):
    """A DeepSeek cron job must not inherit MiniMax's Anthropic API mode."""
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: opencode-go\n  default: minimax-m3\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "OPENCODE_GO_API_KEY=test-opencode-go-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    fake_db = MagicMock()
    fake_agent = MagicMock()
    fake_agent.run_conversation.return_value = {
        "completed": True,
        "final_response": "ok",
        "messages": [],
    }
    job = {
        "id": "opencode-go-model-routing",
        "name": "OpenCode Go model routing",
        "prompt": "ping",
        "model": "deepseek-v4-flash",
        "provider": "opencode-go",
        "deliver": "local",
    }

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._get_hermes_home", return_value=tmp_path),
        patch("cron.scheduler.claim_dispatch", return_value=True),
        patch("cron.scheduler.save_job_output", return_value=tmp_path / "output.md"),
        patch("cron.scheduler._deliver_result", return_value=None),
        patch("cron.scheduler.mark_job_run") as mark_job_run,
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("cron.scheduler._build_job_prompt", return_value="ping"),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch("agent.auxiliary_client.cleanup_stale_async_clients"),
        patch("run_agent.AIAgent", return_value=fake_agent) as agent_cls,
    ):
        from cron.scheduler import run_one_job

        processed = run_one_job(job)

    assert processed is True
    agent_cls.assert_called_once()
    mark_job_run.assert_called_once()
    assert mark_job_run.call_args.args[1] is True

    agent_kwargs = agent_cls.call_args.kwargs
    assert agent_kwargs["model"] == "deepseek-v4-flash"
    assert agent_kwargs["provider"] == "opencode-go"
    assert agent_kwargs["api_mode"] == "chat_completions"
    assert agent_kwargs["base_url"] == "https://opencode.ai/zen/go/v1"
