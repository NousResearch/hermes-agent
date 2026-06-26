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


# ---------------------------------------------------------------------------
# Profile scoping (#53077)
# ---------------------------------------------------------------------------


def _make_profile_dir(hermes_env, name):
    """Create a named profile directory under hermes_env/profiles/<name>."""
    prof_dir = hermes_env / "profiles" / name
    prof_dir.mkdir(parents=True, exist_ok=True)
    (prof_dir / "scripts").mkdir(exist_ok=True)
    (prof_dir / "cron").mkdir(exist_ok=True)
    return prof_dir


def test_create_job_captures_default_profile(hermes_env):
    """create_job() auto-tags with 'default' when no profile is explicit."""
    from cron.jobs import create_job

    job = create_job(prompt="hi", schedule="every 5m", deliver="local")
    assert job.get("profile") == "default"


def test_create_job_explicit_profile_stored(hermes_env):
    """create_job() stores an explicit profile name."""
    from cron.jobs import create_job

    _make_profile_dir(hermes_env, "donna")
    job = create_job(
        prompt="hi", schedule="every 5m", deliver="local", profile="donna",
    )
    assert job["profile"] == "donna"


def test_create_job_auto_captures_named_profile(hermes_env, monkeypatch):
    """When HERMES_HOME points at a named profile, create_job() captures it."""
    from cron.jobs import create_job

    donna_dir = _make_profile_dir(hermes_env, "donna")
    monkeypatch.setenv("HERMES_HOME", str(donna_dir))
    import importlib
    import hermes_constants
    import cron.jobs
    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)

    job = cron.jobs.create_job(prompt="hi", schedule="every 5m", deliver="local")
    assert job["profile"] == "donna"


def test_normalize_backfills_legacy_profile_to_default(hermes_env):
    """Legacy jobs without a profile field get 'default' on normalization."""
    from cron.jobs import _normalize_job_record

    legacy = {
        "id": "legacy-1",
        "prompt": "hi",
        "schedule": "every 5m",
        "enabled": True,
    }
    normalized = _normalize_job_record(legacy)
    assert normalized["profile"] == "default"


def test_resolve_profile_home_default_returns_root(hermes_env):
    """'default' / empty / None resolves to the root HERMES_HOME."""
    from cron.jobs import resolve_profile_home

    for name in ("default", "", None):
        result = resolve_profile_home(name)
        assert result == hermes_env.resolve()


def test_resolve_profile_home_named_returns_profiles_dir(hermes_env):
    """Named profile resolves to <root>/profiles/<name>."""
    from cron.jobs import resolve_profile_home

    _make_profile_dir(hermes_env, "donna")
    result = resolve_profile_home("donna")
    assert result == (hermes_env / "profiles" / "donna").resolve()


def test_resolve_profile_home_missing_returns_none(hermes_env):
    """Non-existent profile returns None (triggers fallback warning)."""
    from cron.jobs import resolve_profile_home

    assert resolve_profile_home("ghost") is None


def test_current_profile_name_default(hermes_env):
    """current_profile_name() returns 'default' for root HERMES_HOME."""
    from cron.jobs import current_profile_name

    assert current_profile_name() == "default"


def test_current_profile_name_named(hermes_env, monkeypatch):
    """current_profile_name() derives profile from HERMES_HOME path."""
    donna_dir = _make_profile_dir(hermes_env, "donna")
    monkeypatch.setenv("HERMES_HOME", str(donna_dir))
    import importlib
    import hermes_constants
    import cron.jobs
    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)

    assert cron.jobs.current_profile_name() == "donna"


def test_run_job_no_agent_with_profile_sets_hermes_home(hermes_env, monkeypatch):
    """no_agent job with a named profile runs under that profile's HERMES_HOME.

    Regression for #53077: profile scoping must happen BEFORE the no_agent
    short-circuit so script resolution uses the correct profile's config.
    """
    from cron.scheduler import run_job

    donna_dir = _make_profile_dir(hermes_env, "donna")
    # Write a script under donna's scripts dir
    script_path = donna_dir / "scripts" / "whoami.sh"
    script_path.write_text("#!/bin/bash\necho $HERMES_HOME\n")
    script_path.chmod(0o755)

    job = {
        "id": "prof-test-1",
        "name": "profile no_agent test",
        "prompt": "",
        "schedule": "every 5m",
        "script": "whoami.sh",
        "no_agent": True,
        "enabled": True,
        "profile": "donna",
        "deliver": "local",
    }

    captured_env = {}

    original_run_script = None

    def _capture_script(script_path_arg):
        """Capture HERMES_HOME at the moment the script would run."""
        import os
        captured_env["HERMES_HOME"] = os.environ.get("HERMES_HOME")
        # Return success with empty output
        return (True, "")

    with patch("cron.scheduler._run_job_script", side_effect=_capture_script):
        run_job(job)

    # HERMES_HOME must have been set to donna's profile dir during execution
    assert captured_env.get("HERMES_HOME") == str(donna_dir)


def test_run_job_restores_hermes_home_after_profile_scoped(hermes_env):
    """HERMES_HOME is restored to the ticker's value after a profile-scoped job."""
    import os
    from cron.scheduler import run_job

    donna_dir = _make_profile_dir(hermes_env, "donna")
    original_home = os.environ.get("HERMES_HOME")

    job = {
        "id": "prof-restore-1",
        "name": "restore test",
        "prompt": "",
        "schedule": "every 5m",
        "script": "noop.sh",
        "no_agent": True,
        "enabled": True,
        "profile": "donna",
        "deliver": "local",
    }

    with patch("cron.scheduler._run_job_script", return_value=(True, "")):
        run_job(job)

    # HERMES_HOME must be restored to the ticker's original value
    assert os.environ.get("HERMES_HOME") == original_home


def test_run_job_missing_profile_falls_back_with_warning(hermes_env, caplog):
    """Job referencing a deleted profile runs under ticker's home + warns."""
    import logging
    from cron.scheduler import run_job

    job = {
        "id": "prof-missing-1",
        "name": "missing profile test",
        "prompt": "",
        "schedule": "every 5m",
        "script": "noop.sh",
        "no_agent": True,
        "enabled": True,
        "profile": "ghost",  # doesn't exist
        "deliver": "local",
    }

    with patch("cron.scheduler._run_job_script", return_value=(True, "")) as mock_run:
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            run_job(job)

    # Script should still run (under the ticker's own HERMES_HOME)
    mock_run.assert_called_once()
    # Warning should have been logged
    assert any("no longer exists" in r.message for r in caplog.records)


def test_cronjob_tool_create_stores_profile(hermes_env):
    """cronjob tool passes profile through to create_job."""
    from tools.cronjob_tools import cronjob

    _make_profile_dir(hermes_env, "donna")
    result = json.loads(cronjob(
        action="create",
        schedule="every 5m",
        prompt="hello",
        profile="donna",
    ))
    assert result["success"] is True
    assert result["job"]["profile"] == "donna"


def test_cronjob_tool_update_changes_profile(hermes_env):
    """cronjob tool update changes the profile field."""
    from tools.cronjob_tools import cronjob

    _make_profile_dir(hermes_env, "donna")
    create_result = json.loads(cronjob(
        action="create",
        schedule="every 5m",
        prompt="hello",
    ))
    job_id = create_result["job_id"]

    update_result = json.loads(cronjob(
        action="update",
        job_id=job_id,
        profile="donna",
    ))
    assert update_result["success"] is True
    assert update_result["job"]["profile"] == "donna"


def test_needs_sequential_routes_profile_mismatch_to_sequential(hermes_env):
    """Profile-mismatched jobs are identified as needing sequential execution.

    The actual _needs_sequential() helper is defined inside tick(); we verify
    the same logic here to confirm that a profile-mismatched job would route
    to the sequential pool (preventing concurrent HERMES_HOME mutation).
    """
    from cron.jobs import resolve_profile_home
    from hermes_constants import get_hermes_home

    _make_profile_dir(hermes_env, "donna")

    # Job whose profile doesn't match the ticker's home
    job_mismatch = {"profile": "donna", "workdir": ""}
    prof = (job_mismatch.get("profile") or "default").strip() or "default"
    phome = resolve_profile_home(prof)
    needs_seq = phome is not None and phome != get_hermes_home().resolve()
    assert needs_seq is True  # → sequential pool

    # Job whose profile matches the ticker's home
    job_match = {"profile": "default", "workdir": ""}
    prof2 = (job_match.get("profile") or "default").strip() or "default"
    phome2 = resolve_profile_home(prof2)
    needs_seq2 = phome2 is not None and phome2 != get_hermes_home().resolve()
    assert needs_seq2 is False  # → parallel pool

    # Job with workdir always sequential
    job_workdir = {"profile": "default", "workdir": "/some/dir"}
    needs_seq3 = bool((job_workdir.get("workdir") or "").strip())
    assert needs_seq3 is True  # → sequential pool
