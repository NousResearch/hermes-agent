"""OpenRouter Analytics attribution for cron jobs (#66117)."""

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock, patch

from cron.scheduler import run_job


def _run_job_with_provider(tmp_path, provider: str) -> Mapping[str, Any]:
    job = {
        "id": "a1b2c3d4e5f6",
        "name": "daily cost report",
        "prompt": "summarize costs",
    }
    fake_db = MagicMock()

    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB", return_value=fake_db),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "provider": provider,
                "api_mode": "chat_completions",
            },
        ),
        patch("run_agent.AIAgent") as agent_cls,
    ):
        agent_cls.return_value.run_conversation.return_value = {"final_response": "ok"}

        success, _, _, error = run_job(job)

    assert success is True
    assert error is None
    return agent_cls.call_args.kwargs


def test_openrouter_cron_job_sets_stable_external_user(tmp_path):
    kwargs = _run_job_with_provider(tmp_path, "openrouter")

    assert kwargs["request_overrides"] == {"user": "cron:a1b2c3d4e5f6"}


def test_non_openrouter_cron_job_does_not_set_external_user(tmp_path):
    kwargs = _run_job_with_provider(tmp_path, "nous")

    assert kwargs["request_overrides"] is None
