"""Cron execution must preserve the authoritative terminal config."""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType


def test_run_job_terminal_config_wins_after_dotenv_reload(tmp_path, monkeypatch):
    """A stale .env must not clobber config.yaml before terminal tools run."""
    import cron.scheduler as sched
    import hermes_cli.env_loader as env_loader
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP

    hermes_home = tmp_path / "hermes-home"
    job_workdir = tmp_path / "job-workdir"
    configured_cwd = tmp_path / "configured-cwd"
    hermes_home.mkdir()
    job_workdir.mkdir()
    configured_cwd.mkdir()

    volumes = ["/srv/data:/output:ro", "/srv/cache:/cache"]
    (hermes_home / "config.yaml").write_text(
        json.dumps(
            {
                "terminal": {
                    "backend": "docker",
                    "docker_image": "example/hermes-cron:test",
                    "container_cpu": 3.5,
                    "container_memory": 7168,
                    "docker_volumes": volumes,
                    "cwd": str(configured_cwd),
                }
            }
        ),
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "\n".join(
            (
                "TERMINAL_ENV=local",
                "TERMINAL_DOCKER_IMAGE=stale/image:old",
                "TERMINAL_CONTAINER_CPU=1",
                "TERMINAL_CONTAINER_MEMORY=5120",
                "TERMINAL_DOCKER_VOLUMES=[]",
                "TERMINAL_CWD=/from-stale-dotenv",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    # Register every env mutation with monkeypatch: the canonical helper
    # writes directly to os.environ and otherwise its merged defaults could
    # leak into later tests in this worker.
    for env_var in TERMINAL_CONFIG_ENV_MAP.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")
    monkeypatch.setenv("TERMINAL_CWD", "/original-cwd")
    monkeypatch.setattr(sched, "_hermes_home", hermes_home)

    # Exercise the real profile .env loader, but isolate unrelated external
    # secret/managed-scope integrations from this terminal-config contract.
    monkeypatch.setattr(env_loader, "reset_secret_source_cache", lambda: None)
    monkeypatch.setattr(
        env_loader, "_apply_external_secret_sources", lambda *_a: None
    )
    monkeypatch.setattr(env_loader, "_apply_managed_env", lambda: None)

    observed = {}

    class FakeAgent:
        def __init__(self, **_kwargs):
            from tools.terminal_tool import _get_env_config

            observed.update(_get_env_config())
            observed["raw_terminal_cwd"] = os.environ.get("TERMINAL_CWD")

        def run_conversation(self, *_args, **_kwargs):
            return {"final_response": "done", "messages": []}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

        def close(self):
            return None

    fake_run_agent = ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_hermes_state = ModuleType("hermes_state")
    fake_hermes_state.SessionDB = lambda: None
    monkeypatch.setitem(sys.modules, "hermes_state", fake_hermes_state)

    fake_mcp_tool = ModuleType("tools.mcp_tool")
    fake_mcp_tool.discover_mcp_tools = lambda: []
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mcp_tool)

    from hermes_cli import runtime_provider

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "provider": "test",
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(sched, "_build_job_prompt", lambda *_a, **_kw: "prompt")
    monkeypatch.setattr(sched, "_resolve_origin", lambda _job: None)
    monkeypatch.setattr(sched, "_resolve_delivery_target", lambda _job: None)
    monkeypatch.setattr(sched, "_resolve_cron_enabled_toolsets", lambda *_a: None)
    monkeypatch.setattr(sched, "_resolve_cron_disabled_toolsets", lambda *_a: None)
    monkeypatch.setattr(
        sched,
        "_teardown_cron_agent",
        lambda agent, _job_id: agent.close() if agent is not None else None,
    )

    success, _output, response, error = sched.run_job(
        {
            "id": "standalone-run",
            "name": "standalone run",
            "prompt": "check cleanup",
            "model": "test-model",
            "workdir": str(job_workdir),
            "schedule_display": "manual",
        }
    )

    assert success is True, error
    assert response == "done"
    assert observed["env_type"] == "docker"
    assert observed["docker_image"] == "example/hermes-cron:test"
    assert observed["container_cpu"] == 3.5
    assert observed["container_memory"] == 7168
    assert observed["docker_volumes"] == volumes
    assert observed["raw_terminal_cwd"] == str(job_workdir)
    assert os.environ["TERMINAL_CWD"] == "/original-cwd"
