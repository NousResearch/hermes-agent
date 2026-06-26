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
import sys
from unittest.mock import patch

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


def test_create_job_no_agent_missing_script_rejected_at_data_layer(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="Script file not found"):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="missing.sh",
            no_agent=True,
            deliver="local",
        )


def test_create_job_no_agent_inline_script_rejected_at_data_layer(hermes_env):
    from cron.jobs import create_job

    with pytest.raises(ValueError, match="inline script content"):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="#!/bin/bash\necho hello",
            no_agent=True,
            deliver="local",
        )


def test_create_job_no_agent_traversal_rejected_at_data_layer(hermes_env):
    from cron.jobs import create_job, list_jobs

    with pytest.raises(ValueError, match="escapes the scripts directory"):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="../outside.sh",
            no_agent=True,
            deliver="local",
        )

    assert list_jobs(include_disabled=True) == []


def test_create_job_no_agent_unknown_home_user_rejected_at_data_layer(hermes_env):
    from cron.jobs import create_job, list_jobs

    with pytest.raises(ValueError):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="~definitely_missing_hermes_user_53037/watchdog.sh",
            no_agent=True,
            deliver="local",
        )

    assert list_jobs(include_disabled=True) == []


def test_create_job_no_agent_symlink_loop_rejected_as_value_error(hermes_env):
    from cron.jobs import create_job, list_jobs

    a = hermes_env / "scripts" / "a.sh"
    b = hermes_env / "scripts" / "b.sh"
    try:
        a.symlink_to(b)
        b.symlink_to(a)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")

    with pytest.raises(ValueError):
        create_job(
            prompt=None,
            schedule="every 5m",
            script="a.sh",
            no_agent=True,
            deliver="local",
        )

    assert list_jobs(include_disabled=True) == []


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


def test_create_job_no_agent_absolute_script_inside_scripts_succeeds(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "absolute.sh"
    script_path.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script=str(script_path),
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["script"] == str(script_path)


def test_create_job_no_agent_uses_scoped_profile_scripts(hermes_env, tmp_path):
    from cron.jobs import create_job, use_cron_store

    profile_home = tmp_path / "profile-home"
    scripts_dir = profile_home / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "profile-task.py").write_text("print('ok')\n", encoding="utf-8")

    with use_cron_store(profile_home):
        job = create_job(
            prompt=None,
            schedule="every 5m",
            script="profile-task.py",
            no_agent=True,
            deliver="local",
        )

    assert job["script"] == "profile-task.py"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows drive-letter regression")
def test_create_job_no_agent_windows_absolute_script_inside_scripts_succeeds(hermes_env):
    from cron.jobs import create_job

    script_path = hermes_env / "scripts" / "absolute.ps1"
    script_path.write_text("Write-Output ok\n", encoding="utf-8")

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script=str(script_path),
        no_agent=True,
        deliver="local",
    )
    assert job["no_agent"] is True
    assert job["script"] == str(script_path)


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


def test_update_job_no_agent_missing_script_rejected_at_data_layer(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    job = create_job(
        prompt="summarize later",
        schedule="every 5m",
        script="later.sh",
        deliver="local",
    )

    with pytest.raises(ValueError, match="Script file not found"):
        update_job(job["id"], {"no_agent": True})

    reloaded = get_job(job["id"])
    assert reloaded["no_agent"] is False
    assert reloaded["script"] == "later.sh"


def test_update_job_active_no_agent_missing_script_rejected_on_any_update(hermes_env):
    from cron.jobs import create_job, get_job, update_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n", encoding="utf-8")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="alert.sh",
        no_agent=True,
        deliver="local",
    )

    script_path.unlink()
    with pytest.raises(ValueError, match="Script file not found"):
        update_job(job["id"], {"name": "renamed"})

    reloaded = get_job(job["id"])
    assert reloaded["name"] != "renamed"

    paused = update_job(job["id"], {"state": "paused", "enabled": False})
    assert paused["state"] == "paused"


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


def test_cronjob_tool_create_no_agent_subdirectory_script_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    nested = hermes_env / "scripts" / "watchers"
    nested.mkdir()
    script_path = nested / "alert.sh"
    script_path.write_text("#!/bin/bash\necho alert\n")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="watchers/alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["script"] == "watchers/alert.sh"


def test_cronjob_tool_create_no_agent_existing_script_with_space_succeeds(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "daily report.py"
    script_path.write_text("print('ok')\n", encoding="utf-8")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="daily report.py",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["script"] == "daily report.py"


def test_cronjob_tool_create_no_agent_absolute_script_inside_scripts_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n", encoding="utf-8")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script=str(script_path),
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "relative" in result.get("error", "").lower()


def test_cronjob_tool_create_no_agent_symlink_escape_errors(hermes_env, tmp_path):
    from tools.cronjob_tools import cronjob

    outside = tmp_path / "outside.sh"
    outside.write_text("echo escaped\n", encoding="utf-8")
    link = hermes_env / "scripts" / "escape.sh"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink unavailable")

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="escape.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "escapes" in result.get("error", "").lower()


def test_cronjob_tool_create_no_agent_missing_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="missing.sh",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()

    listing = json.loads(cronjob(action="list", include_disabled=True))
    assert listing["jobs"] == []


def test_cronjob_tool_create_no_agent_inline_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="#!/bin/bash\necho alert",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "not inline script content" in result.get("error", "")


def test_cronjob_tool_create_no_agent_shell_snippet_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="echo alert && uptime",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()


def test_cronjob_tool_create_no_agent_overlong_script_path_errors_concisely(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="x" * 5000,
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "too long" in result.get("error", "").lower()
    assert len(result.get("error", "")) < 500


def test_cronjob_tool_create_no_agent_directory_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    script_dir = hermes_env / "scripts" / "watchers"
    script_dir.mkdir()

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="watchers",
            no_agent=True,
            deliver="local",
        )
    )
    assert result.get("success") is False
    assert "not a regular file" in result.get("error", "").lower()


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


def test_cronjob_tool_update_no_agent_with_missing_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(action="create", schedule="every 5m", prompt="do a thing", deliver="local")
    )
    job_id = created["job_id"]

    result = json.loads(
        cronjob(action="update", job_id=job_id, script="missing.sh", no_agent=True)
    )
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()

    listing = json.loads(cronjob(action="list", include_disabled=True))
    job = next(j for j in listing["jobs"] if j["job_id"] == job_id)
    assert job.get("no_agent") in {False, None}
    assert "script" not in job


def test_cronjob_tool_update_existing_missing_script_cannot_enable_no_agent(hermes_env):
    from tools.cronjob_tools import cronjob

    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt="do a thing",
            script="later.sh",
            deliver="local",
        )
    )
    job_id = created["job_id"]

    result = json.loads(cronjob(action="update", job_id=job_id, no_agent=True))
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()


def test_cronjob_tool_update_existing_absolute_script_can_enable_no_agent(hermes_env):
    from cron.jobs import create_job
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "absolute.sh"
    script_path.write_text("echo ok\n", encoding="utf-8")
    job = create_job(
        prompt="analyze",
        schedule="every 5m",
        script=str(script_path),
        deliver="local",
    )

    result = json.loads(cronjob(action="update", job_id=job["id"], no_agent=True))
    assert result.get("success") is True
    assert result["job"]["no_agent"] is True
    assert result["job"]["script"] == str(script_path)


def test_cronjob_tool_update_no_agent_cannot_clear_script(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n")
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    result = json.loads(cronjob(action="update", job_id=job_id, script=""))
    assert result.get("success") is False
    assert "cannot clear script" in result.get("error", "").lower()


def test_cronjob_tool_update_can_disable_no_agent_and_clear_script(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n")
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    result = json.loads(
        cronjob(
            action="update",
            job_id=job_id,
            no_agent=False,
            script="",
            prompt="run normally",
        )
    )
    assert result["success"] is True
    assert result["job"].get("no_agent") in {False, None}
    assert "script" not in result["job"]


def test_cronjob_tool_resume_no_agent_missing_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n")
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]
    assert json.loads(cronjob(action="pause", job_id=job_id))["success"] is True

    script_path.unlink()
    result = json.loads(cronjob(action="resume", job_id=job_id))
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()


def test_cronjob_tool_resume_persisted_absolute_script_inside_scripts_succeeds(hermes_env):
    from cron.jobs import create_job, pause_job
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "absolute.sh"
    script_path.write_text("echo alert\n", encoding="utf-8")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script=str(script_path),
        no_agent=True,
        deliver="local",
    )
    pause_job(job["id"])

    result = json.loads(cronjob(action="resume", job_id=job["id"]))
    assert result.get("success") is True
    assert result["job"]["script"] == str(script_path)


def test_cronjob_tool_run_no_agent_missing_script_errors(hermes_env):
    from tools.cronjob_tools import cronjob

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("echo alert\n", encoding="utf-8")
    created = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            script="alert.sh",
            no_agent=True,
            deliver="local",
        )
    )
    job_id = created["job_id"]

    script_path.unlink()
    result = json.loads(cronjob(action="run", job_id=job_id))
    assert result.get("success") is False
    assert "script file not found" in result.get("error", "").lower()


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


def test_cronjob_tool_create_agent_backed_script_with_space_still_allowed(hermes_env):
    from tools.cronjob_tools import cronjob

    result = json.loads(
        cronjob(
            action="create",
            schedule="every 5m",
            prompt="Analyze the output.",
            script="daily report.py",
            deliver="local",
        )
    )
    assert result.get("success") is True
    assert result["job"]["script"] == "daily report.py"


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
