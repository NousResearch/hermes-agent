"""Regression tests for cron honoring configured priority service tier."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    from cron import scheduler

    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    monkeypatch.setattr(scheduler, "_hermes_home", hermes_home)

    return hermes_home


def _run_job_and_capture_agent_kwargs(cron_env: Path, service_tier_value: str):
    """Run a cron job with a fake agent and return its constructor kwargs."""
    (cron_env / "config.yaml").write_text(
        "model:\n"
        "  default: gpt-5.5\n"
        "agent:\n"
        f"  service_tier: {service_tier_value}\n",
        encoding="utf-8",
    )

    captured: dict = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, prompt):
            assert "cron service tier regression" in prompt
            return {"completed": True, "final_response": "ok"}

        def close(self):
            return None

    class FakeSessionDB:
        def set_session_title(self, *args, **kwargs):
            return None

        def end_session(self, *args, **kwargs):
            return None

        def close(self):
            return None

    job = {
        "id": "service-tier-regression",
        "name": "service-tier-regression",
        "prompt": "cron service tier regression",
        "schedule_display": "manual",
    }

    with (
        patch("run_agent.AIAgent", FakeAIAgent),
        patch("hermes_state.SessionDB", FakeSessionDB),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openai-codex",
                "api_key": None,
                "base_url": None,
                "api_mode": None,
                "command": None,
                "args": None,
            },
        ),
    ):
        from cron.scheduler import run_job

        success, _output, final_response, error = run_job(job)

    assert success is True, error
    assert final_response == "ok"
    return captured


@pytest.mark.parametrize("configured", ["fast", "priority", "on"])
def test_cron_agent_receives_priority_service_tier(cron_env, configured):
    kwargs = _run_job_and_capture_agent_kwargs(cron_env, configured)

    assert kwargs["service_tier"] == "priority"
    assert kwargs["request_overrides"] == {"service_tier": "priority"}


@pytest.mark.parametrize("configured", ["normal", "default", "standard", "off", "none", ""])
def test_cron_agent_leaves_normal_service_tier_unset(cron_env, configured):
    kwargs = _run_job_and_capture_agent_kwargs(cron_env, configured)

    assert kwargs["service_tier"] is None
    assert kwargs["request_overrides"] == {}
