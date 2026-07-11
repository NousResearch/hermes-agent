import sys
import types

import pytest

from cron.jobs import create_job, get_job, list_jobs, save_jobs, update_job


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_create_stores_normalized_reasoning_effort(tmp_cron_dir):
    job = create_job(prompt="Think deeply", schedule="30m", reasoning_effort=" MAX ")
    assert job["reasoning_effort"] == "max"
    assert get_job(job["id"])["reasoning_effort"] == "max"


def test_create_stores_none_as_explicit_override(tmp_cron_dir):
    job = create_job(prompt="No reasoning", schedule="30m", reasoning_effort="NONE")
    assert job["reasoning_effort"] == "none"


def test_create_invalid_reasoning_effort_raises(tmp_cron_dir):
    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        create_job(prompt="Bad", schedule="30m", reasoning_effort="turbo")


def test_update_changes_preserves_and_clears_reasoning_effort(tmp_cron_dir):
    job = create_job(prompt="Update me", schedule="30m", reasoning_effort="low")

    updated = update_job(job["id"], {"reasoning_effort": "MAX"})
    assert updated["reasoning_effort"] == "max"

    preserved = update_job(job["id"], {"name": "renamed"})
    assert preserved["reasoning_effort"] == "max"

    cleared = update_job(job["id"], {"reasoning_effort": ""})
    assert cleared is not None
    assert cleared["reasoning_effort"] is None

    updated = update_job(job["id"], {"reasoning_effort": "none"})
    assert updated["reasoning_effort"] == "none"

    cleared = update_job(job["id"], {"reasoning_effort": None})
    assert cleared["reasoning_effort"] is None


def test_legacy_invalid_reasoning_effort_is_read_safe(tmp_cron_dir):
    save_jobs([
        {
            "id": "abc123deadbe",
            "name": "legacy",
            "prompt": "legacy",
            "schedule_display": "every 60m",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "enabled": True,
            "reasoning_effort": "turbo",
        }
    ])

    jobs = list_jobs()
    assert jobs[0]["reasoning_effort"] is None
    assert get_job("abc123deadbe")["reasoning_effort"] is None


def test_legacy_false_reasoning_effort_is_normalized_to_none(tmp_cron_dir):
    save_jobs([
        {
            "id": "abc123deadbe",
            "name": "legacy",
            "prompt": "legacy",
            "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
            "enabled": True,
            "reasoning_effort": False,
        }
    ])

    assert list_jobs()[0]["reasoning_effort"] == "none"
    assert get_job("abc123deadbe")["reasoning_effort"] == "none"


@pytest.fixture()
def scheduler_harness(tmp_path, monkeypatch):
    """Patch heavy scheduler dependencies and capture AIAgent kwargs."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            return {"completed": True, "final_response": "ok"}

        def get_activity_summary(self):
            return {"seconds_since_activity": 0.0}

    fake_run_agent = types.ModuleType("run_agent")
    setattr(fake_run_agent, "AIAgent", FakeAIAgent)
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    fake_state = types.ModuleType("hermes_state")

    class FakeSessionDB:
        pass

    setattr(fake_state, "SessionDB", FakeSessionDB)
    monkeypatch.setitem(sys.modules, "hermes_state", fake_state)

    fake_runtime_provider = types.ModuleType("hermes_cli.runtime_provider")
    setattr(
        fake_runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "provider": kwargs.get("requested") or "openai-codex",
            "api_key": "[REDACTED]",
            "base_url": None,
            "api_mode": None,
        },
    )
    setattr(fake_runtime_provider, "format_runtime_provider_error", lambda exc: str(exc))
    monkeypatch.setitem(sys.modules, "hermes_cli.runtime_provider", fake_runtime_provider)

    fake_auth = types.ModuleType("hermes_cli.auth")

    class FakeAuthError(Exception):
        pass

    setattr(fake_auth, "AuthError", FakeAuthError)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)

    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: True)
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_mcp = types.ModuleType("tools.mcp_tool")
    setattr(fake_mcp, "discover_mcp_tools", lambda: [])
    monkeypatch.setitem(sys.modules, "tools.mcp_tool", fake_mcp)

    return home, captured


def _minimal_job(reasoning_effort=None):
    job = {
        "id": "abc123deadbe",
        "name": "reasoning test",
        "prompt": "Say ok",
        "schedule": {"kind": "interval", "minutes": 60, "display": "every 60m"},
        "schedule_display": "every 60m",
        "enabled": True,
        "deliver": "local",
    }
    if reasoning_effort is not None:
        job["reasoning_effort"] = reasoning_effort
    return job


def test_scheduler_uses_max_job_reasoning_effort_over_global(scheduler_harness):
    home, captured = scheduler_harness
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\nagent:\n  reasoning_effort: high\n",
        encoding="utf-8",
    )

    from cron.scheduler import run_job

    success, _doc, final_response, error = run_job(_minimal_job("max"))

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert captured["reasoning_config"] == {"enabled": True, "effort": "max"}


def test_scheduler_falls_back_to_global_reasoning_effort(scheduler_harness):
    home, captured = scheduler_harness
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\nagent:\n  reasoning_effort: high\n",
        encoding="utf-8",
    )

    from cron.scheduler import run_job

    success, _doc, final_response, error = run_job(_minimal_job())

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert captured["reasoning_config"] == {"enabled": True, "effort": "high"}


@pytest.mark.parametrize("job_effort", ["none", False])
def test_scheduler_disabled_override_does_not_fall_back(scheduler_harness, job_effort):
    home, captured = scheduler_harness
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\nagent:\n  reasoning_effort: high\n",
        encoding="utf-8",
    )

    from cron.scheduler import run_job

    success, _doc, final_response, error = run_job(_minimal_job(job_effort))

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert captured["reasoning_config"] == {"enabled": False}


def test_scheduler_invalid_hand_edited_value_falls_back_to_global(scheduler_harness):
    home, captured = scheduler_harness
    (home / "config.yaml").write_text(
        "model:\n  default: gpt-5.5\n  provider: openai-codex\nagent:\n  reasoning_effort: medium\n",
        encoding="utf-8",
    )

    from cron.scheduler import run_job

    success, _doc, final_response, error = run_job(_minimal_job("turbo"))

    assert success is True
    assert final_response == "ok"
    assert error is None
    assert captured["reasoning_config"] == {"enabled": True, "effort": "medium"}


def test_scheduler_no_agent_ignores_reasoning_and_never_constructs_agent(
    scheduler_harness
):
    _home, captured = scheduler_harness
    scripts_dir = _home / "scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "silent.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o755)

    job = _minimal_job("high")
    job["no_agent"] = True
    job["script"] = "silent.sh"

    from cron.scheduler import run_job, SILENT_MARKER

    success, _doc, final_response, error = run_job(job)

    assert success is True
    assert final_response == SILENT_MARKER
    assert error is None
    assert "reasoning_config" not in captured