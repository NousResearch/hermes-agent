"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    # Reload modules that cache get_hermes_home() at import time.
    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


# ---------------------------------------------------------------------------
# create_job / update_job: data-layer semantics
# ---------------------------------------------------------------------------


def test_create_job_no_agent_requires_script(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="no_agent=True requires a script"):
        create_job(prompt=None, schedule="every 5m", no_agent=True)


def test_create_job_no_agent_stores_field(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "watchdog.sh"
    script_path.write_text("#!/bin/bash\necho hi\n")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="watchdog.sh",
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["script"] == "watchdog.sh"
    # Prompt can be empty/None for no_agent jobs.
    assert job["prompt"] in {None, ""}


def test_create_job_default_is_not_no_agent(hermes_env):
    from cron.jobs import create_job

    job = create_job(prompt="say hi", schedule="every 5m", deliver="local")
    assert job.get("no_agent") is False


def test_update_job_roundtrips_no_agent_flag(hermes_env):
    from cron.jobs import create_job, update_job, get_job

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")
    job = create_job(prompt=None, schedule="every 5m", script="w.sh", no_agent=True, deliver="local")

    update_job(job["id"], {"no_agent": False})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False

    update_job(job["id"], {"no_agent": True})
    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is True


# ---------------------------------------------------------------------------
# cronjob tool: API-layer validation
# ---------------------------------------------------------------------------


def test_cronjob_tool_create_no_agent_without_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(action="create", schedule="every 5m", no_agent=True, deliver="local")
    )
    assert result.get("success") is False
    assert "no_agent=True requires a script" in result.get("error", "")


def test_cronjob_tool_create_no_agent_with_script_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["no_agent"] is True
    assert result["job"]["script"] == "alert.sh"


def test_cronjob_tool_update_toggles_no_agent(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    off = json.loads(cronjob(action="update", job_id=job_id, no_agent=False, prompt="run"))
    assert off["success"] is True
    assert off["job"].get("no_agent") in {False, None}

    on = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert on["success"] is True
    assert on["job"]["no_agent"] is True


def test_cronjob_tool_update_no_agent_without_script_errors(hermes_env):
    """Flipping no_agent=True on a job that has no script must fail."""
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(action="create", schedule="every 5m", prompt="do a thing", deliver="local")
    )
    job_id = created["job_id"]

    result = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert result.get("success") is False
    assert "without a script" in result.get("error", "")


def test_cronjob_tool_create_does_not_require_prompt_when_no_agent(hermes_env):
    """The 'prompt or skill required' rule is relaxed for no_agent jobs."""
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "w.sh"
    script_path.write_text("echo hi\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="w.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True


# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert "RAM 92% on host" in final_response
    assert "RAM 92% on host" in doc


def test_run_job_no_agent_empty_output_is_silent(hermes_env):
    """Empty stdout → SILENT_MARKER, which suppresses delivery downstream."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\n# nothing to say\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_wake_gate_is_silent(hermes_env):
    """wakeAgent=false gate in stdout triggers a silent run."""
    from cron.jobs import create_job
    from cron.scheduler import run_job, SILENT_MARKER

    script_path = hermes_env / "scripts" / "gated.sh"
    script_path.write_text('#!/bin/bash\necho \'{"wakeAgent": false}\'\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="gated.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert final_response == SILENT_MARKER


def test_run_job_no_agent_script_failure_delivers_error(hermes_env):
    """Non-zero exit → success=False, error alert is the delivered message."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "broken.sh"
    script_path.write_text("#!/bin/bash\necho oops >&2\nexit 3\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="broken.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is False
    assert error is not None
    assert "oops" in final_response or "exited with code 3" in final_response
    assert "Cron watchdog" in final_response  # alert header


def test_run_job_no_agent_never_invokes_aiagent(hermes_env):
    """no_agent jobs must NOT import/construct the AIAgent."""
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )

    with patch("run_agent.AIAgent") as ai_mock:
        from cron.scheduler import run_job

        run_job(job)

    ai_mock.assert_not_called()


def test_run_job_no_agent_command_center_tracking_is_opt_in(hermes_env):
    """Untracked no_agent jobs must not create dashboard registry state."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    run_job(job)

    assert not (hermes_env / "state" / "agent_registry.json").exists()


def test_run_job_no_agent_command_center_tracking_records_status(hermes_env):
    """Opt-in no_agent scripts show up in Command Center registry."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None,
        name="Health Watchdog",
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deliver="local",
    )
    job["command_center"] = {"track": True, "team": "ops", "risk_level": "read_only"}

    success, _doc, _final_response, error = run_job(job)

    registry = json.loads((hermes_env / "state" / "agent_registry.json").read_text())
    agent = registry["agents"][0]
    assert success is True
    assert error is None
    assert agent["agent_id"] == f"cron-{job['id']}"
    assert agent["team"] == "ops"
    assert agent["runner"] == "cron"
    assert agent["model"] == "no_agent-script"
    assert agent["mission"] == "Health Watchdog"
    assert agent["status"] == "completed"
    assert agent["current_step"] == "completed with output"
    assert agent["risk_level"] == "read_only"


def test_run_one_job_command_center_tracking_links_output_artifact(hermes_env):
    """End-to-end cron firing links the saved cron output to the registry record."""
    from cron.jobs import create_job
    from cron.scheduler import run_one_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    job = create_job(
        prompt=None,
        name="Tracked Cron",
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deliver="local",
    )
    job["command_center_track"] = True

    assert run_one_job(job) is True

    registry = json.loads((hermes_env / "state" / "agent_registry.json").read_text())
    agent = registry["agents"][0]
    assert agent["status"] == "completed"
    assert agent["artifact"]
    assert agent["artifact"].endswith(".md")
    assert "cron/output" in agent["artifact"]
    assert (hermes_env / "cron" / "output" / job["id"]).exists()


def test_run_job_agent_command_center_tracking_records_success(hermes_env):
    """Opt-in LLM cron jobs show their agent lifecycle in Command Center."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    job = create_job(
        prompt="say hi",
        name="LLM Digest",
        schedule="every 5m",
        deliver="local",
        model="test-model",
    )
    job["command_center"] = {"track": True, "team": "ops", "risk_level": "read_only"}

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "hi", "completed": True}
    agent.get_activity_summary.return_value = {
        "seconds_since_activity": 0,
        "last_activity_desc": "api response",
        "current_tool": None,
    }

    with (
        patch("run_agent.AIAgent", return_value=agent),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": "test"}),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
    ):
        success, _doc, final_response, error = run_job(job)

    registry = json.loads((hermes_env / "state" / "agent_registry.json").read_text())
    record = registry["agents"][0]
    assert success is True
    assert final_response == "hi"
    assert error is None
    assert record["agent_id"] == f"cron-{job['id']}"
    assert record["model"] == "cron-agent"
    assert record["status"] == "completed"
    assert record["current_step"] == "agent completed"
    assert record["risk_level"] == "read_only"


def test_run_job_agent_command_center_tracking_records_failure(hermes_env):
    """Tracked LLM cron failures should leave a visible blocker."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    job = create_job(
        prompt="say hi",
        name="Broken LLM Digest",
        schedule="every 5m",
        deliver="local",
        model="test-model",
    )
    job["command_center_track"] = True

    agent = MagicMock()
    agent.run_conversation.side_effect = RuntimeError("model down")
    agent.get_activity_summary.return_value = {"seconds_since_activity": 0}

    with (
        patch("run_agent.AIAgent", return_value=agent),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={"provider": "test"}),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
    ):
        success, _doc, final_response, error = run_job(job)

    registry = json.loads((hermes_env / "state" / "agent_registry.json").read_text())
    record = registry["agents"][0]
    assert success is False
    assert final_response == ""
    assert error is not None
    assert "model down" in error
    assert record["status"] == "failed"
    assert record["current_step"] == "agent failed"
    assert "model down" in record["blocker"]


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_shell_script_runs_via_bash(hermes_env):
    """.sh files should execute under /bin/bash even without a shebang line."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "shelly.sh"
    # No shebang — relies on the interpreter-by-extension rule.
    script_path.write_text('echo "shell: $BASH_VERSION" | head -c 7\n')

    ok, output = _run_job_script("shelly.sh")
    assert ok is True
    assert output.startswith("shell:")


def test_run_job_script_bash_extension_also_runs_via_bash(hermes_env):
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "thing.bash"
    script_path.write_text('printf "via bash\\n"\n')

    ok, output = _run_job_script("thing.bash")
    assert ok is True
    assert output == "via bash"


def test_run_job_script_python_still_runs_via_python(hermes_env):
    """Regression: .py files must keep running via sys.executable."""
    from cron.scheduler import _run_job_script

    script_path = hermes_env / "scripts" / "py.py"
    script_path.write_text("import sys\nprint(f'python {sys.version_info.major}')\n")

    ok, output = _run_job_script("py.py")
    assert ok is True
    assert output.startswith("python ")


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output
