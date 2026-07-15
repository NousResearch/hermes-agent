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
    """Empty stdout produces a mechanical structured suppression receipt."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\n# nothing to say\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    run_result = run_job(job)
    success, doc, final_response, error = run_result
    assert success is True
    assert error is None
    assert final_response == ""
    assert run_result.delivery_outcome["action"] == "suppress"
    assert run_result.delivery_outcome["turn_id"] == run_result.turn_id


def test_run_job_no_agent_wake_gate_is_silent(hermes_env):
    """wakeAgent=false gate in stdout triggers a silent run."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "gated.sh"
    script_path.write_text('#!/bin/bash\necho \'{"wakeAgent": false}\'\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="gated.sh", no_agent=True, deliver="local"
    )
    run_result = run_job(job)
    success, doc, final_response, error = run_result
    assert success is True
    assert final_response == ""
    assert run_result.delivery_outcome["action"] == "suppress"


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


def test_failure_summary_does_not_classify_numeric_substrings(hermes_env):
    """Transport must not mistake ``429`` inside a Git SHA for a rate limit."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    sha = "afa582956fd0429530642dd04798ec3c0ac63b39"
    delivered = _summarize_cron_failure_for_delivery(
        {"name": "Fork upstream auto sync PR routine"},
        f"blocked_merge_conflicts at fork main {sha}",
    )

    assert delivered == (
        "⚠️ Cron 'Fork upstream auto sync PR routine' failed. "
        "Full details saved in cron output."
    )
    assert "rate limit" not in delivered.lower()
    assert "fallback" not in delivered.lower()


def test_no_agent_failure_delivery_preserves_authoritative_script_receipt(hermes_env):
    from cron.scheduler import _cron_failure_delivery_content

    receipt = (
        "⚠ Cron watchdog 'Fork upstream auto sync PR routine' script failed\n\n"
        "Status: blocked_merge_conflicts\nConflicts:\n- gateway/run.py"
    )
    delivered = _cron_failure_delivery_content(
        {"name": "Fork upstream auto sync PR routine", "no_agent": True},
        receipt,
        "fork SHA contains 429 but this is not a provider error",
        no_agent_script_failed=True,
    )

    assert delivered == receipt


def test_no_agent_failure_delivery_is_bounded_but_saved_evidence_is_complete(
    hermes_env,
):
    from cron.jobs import create_job
    from cron.scheduler import (
        _MAX_NO_AGENT_FAILURE_RECEIPT_CHARS,
        _cron_failure_delivery_content,
        run_job,
    )

    script_path = hermes_env / "scripts" / "oversized.py"
    script_path.write_text(
        "import sys\n"
        "print('S' * 12000, file=sys.stderr)\n"
        "print('O' * 12000)\n"
        "raise SystemExit(9)\n"
    )
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="oversized.py",
        no_agent=True,
        deliver="local",
    )

    success, doc, final_response, error = run_job(job)
    assert success is False
    assert len(doc) > _MAX_NO_AGENT_FAILURE_RECEIPT_CHARS
    delivered = _cron_failure_delivery_content(
        job,
        final_response,
        error,
        no_agent_script_failed=True,
    )
    assert len(delivered) == _MAX_NO_AGENT_FAILURE_RECEIPT_CHARS
    assert "failure receipt truncated for delivery" in delivered
    assert "S" * 100 in delivered
    assert "O" * 100 in delivered


def test_run_one_job_delivers_no_agent_failure_receipt_without_reclassification(
    hermes_env,
):
    import cron.scheduler as scheduler

    job = {
        "id": "808bddb875ee",
        "name": "Fork upstream auto sync PR routine",
        "no_agent": True,
    }
    receipt = (
        "⚠ Cron watchdog 'Fork upstream auto sync PR routine' script failed\n\n"
        "Status: blocked_merge_conflicts\n"
        "fork_main: afa582956fd0429530642dd04798ec3c0ac63b39"
    )

    def confirmed_delivery(*_args, delivery_observation, **_kwargs):
        delivery_observation.update(
            {
                "target_count": 1,
                "attempted": True,
                "confirmed_target_count": 1,
                "confirmed": True,
                "unconfirmed": False,
            }
        )
        return None

    with (
        patch("cron.scheduler.claim_dispatch", return_value=True),
        patch("agent.secret_scope.set_secret_scope", return_value=None),
        patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
        patch("agent.secret_scope.reset_secret_scope"),
        patch(
            "cron.scheduler.run_job",
            return_value=(False, "full output", receipt, "blocked_merge_conflicts"),
        ),
        patch("cron.scheduler.save_job_output", return_value="/tmp/output.md"),
        patch(
            "cron.scheduler._deliver_result", side_effect=confirmed_delivery
        ) as deliver,
        patch("cron.scheduler.mark_job_run") as mark,
    ):
        assert scheduler.run_one_job(job) is True

    assert deliver.call_args.args[1] == receipt
    assert "provider" not in deliver.call_args.args[1].lower()
    assert mark.call_args.kwargs["delivery_status"] == "confirmed"


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
