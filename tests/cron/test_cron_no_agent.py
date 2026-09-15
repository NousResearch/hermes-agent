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
import pathlib
import subprocess
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


def test_run_job_no_agent_reloads_dotenv_before_script(hermes_env, monkeypatch):
    """Regression: a standalone cron tick process starts without home-channel
    vars in its environment, and the agent path's per-run dotenv reload never
    executes for no_agent jobs — delivery home channels stayed unresolved.
    run_job must load .env at the top of the no_agent branch."""
    import hermes_cli.env_loader as env_loader
    from cron.jobs import create_job
    from cron.scheduler import run_job

    loaded_homes: list = []

    def fake_load(*, hermes_home=None, project_env=None):
        loaded_homes.append(hermes_home)
        return []

    monkeypatch.setattr(env_loader, "load_hermes_dotenv", fake_load)

    script_path = hermes_env / "scripts" / "probe.sh"
    script_path.write_text('#!/bin/bash\necho "ok"\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="probe.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert loaded_homes, "load_hermes_dotenv was not called on the no_agent path"
    assert str(loaded_homes[0]) == str(hermes_env)


def test_timed_out_no_agent_script_delivery_is_not_mislabeled_as_provider_failure(
    hermes_env, monkeypatch,
):
    """A watchdog timeout happens before any LLM/provider call.

    The delivery summary must preserve that process-level failure taxonomy and
    must not claim a provider fallback was attempted or exhausted.
    """
    from cron.jobs import create_job
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

    (hermes_env / "scripts" / "slow.py").write_text("import time; time.sleep(999)\n")
    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="slow.py",
        no_agent=True,
        deliver="telegram",
        name="slow watchdog",
    )
    delivered = []

    # The script runner uses Popen + a polling loop (cancel/timeout aware),
    # so simulate a process that never finishes: communicate() always times
    # out and the script deadline is shrunk to keep the test fast.
    class _NeverFinishes:
        returncode = None
        pid = 0
        stdout = None
        stderr = None

        def __init__(self, *_args, **_kwargs):
            pass

        def poll(self):
            return None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="slow.py", timeout=timeout)

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="slow.py", timeout=timeout)

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(scheduler.subprocess, "Popen", _NeverFinishes)
    monkeypatch.setattr(sched_script, "_get_script_timeout", lambda: 1)
    monkeypatch.setattr(sched_script, "_terminate_cron_script_process",
        lambda proc: setattr(proc, "returncode", -15),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content),
    )

    assert scheduler.run_one_job(job) is True
    assert len(delivered) == 1
    assert "script timed out" in delivered[0].lower()
    assert "provider" not in delivered[0].lower()
    assert "fallback" not in delivered[0].lower()


def test_agent_provider_timeout_delivery_keeps_fallback_guidance(hermes_env, monkeypatch):
    """Provider timeout classification remains available to agent-backed jobs."""
    from cron.jobs import create_job
    import cron.scheduler as scheduler
    from cron import scheduler_script as sched_script

    job = create_job(
        prompt="Summarize the overnight logs.",
        schedule="every 5m",
        deliver="telegram",
        name="provider-backed report",
    )
    delivered = []

    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_args, **_kwargs: (
            False,
            "# Cron Job: provider-backed report\n\nprovider request timed out\n",
            "",
            "ReadTimeout: provider request timed out after fallback attempts",
        ),
    )
    monkeypatch.setattr(
        scheduler,
        "_deliver_result",
        lambda _job, content, **_kwargs: delivered.append(content),
    )

    assert scheduler.run_one_job(job) is True
    assert len(delivered) == 1
    assert "provider timeout" in delivered[0].lower()
    # Chain wording is now honest (#85508): exhausted when configured,
    # "no fallback chain configured" guidance otherwise.
    assert "fallback chain" in delivered[0].lower()


# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------


def test_run_job_script_path_traversal_still_blocked(hermes_env):
    """Security regression: shell-script support must NOT loosen containment."""
    from cron.scheduler_script import _run_job_script

    # Absolute path outside the scripts dir should be rejected.
    ok, output = _run_job_script("/etc/passwd")
    assert ok is False
    assert "Blocked" in output or "outside" in output


def test_run_job_script_nul_path_fails_cleanly(hermes_env):
    """Sibling of the lifecycle-guard ingestion fix: a NUL-bearing script
    value can survive to fire time (the creation-time guard treats it as
    "nothing to scan"), and ``Path.expanduser()`` raises ValueError — not
    OSError — on it. The scheduler must fail the run with a report, not
    crash with an unhandled exception.

    Regression (#86829): the assertion pins the *eager rejection* contract
    — the specific "NUL byte" report is only produced by the pre-check
    added in the fix. On Linux the legacy guard would swallow the
    expanduser() ValueError and report a generic invalid-path message, so
    a bare "Blocked" assertion could not tell the fixed code from the
    unfixed code; on Windows the unfixed code crashes outright."""
    from cron.scheduler_script import _run_job_script

    ok, output = _run_job_script("~user\x00bad.sh")
    assert ok is False
    assert "NUL byte" in output


def test_run_job_script_nul_rejected_before_any_path_call(hermes_env, monkeypatch):
    """The eager NUL check must run before ``Path(...)`` is ever constructed.

    On Windows ``expanduser()`` never expands ``~user`` and never raises,
    so without the pre-check the NUL surfaces later from ``resolve()`` /
    ``exists()`` — outside the guard's try — and the uncaught ValueError
    crashes the scheduler (#86829). Stubbing ``Path`` with a hard failure
    proves the rejection happens before any pathlib call on every
    platform, not just the ones where expanduser happens to raise."""
    import cron.scheduler as scheduler_module
    from cron import scheduler_script as sched_script

    def boom(*_args, **_kwargs):
        raise AssertionError("Path must not be touched for a NUL-bearing script path")

    monkeypatch.setattr(scheduler_module, "Path", boom)
    ok, output = sched_script._run_job_script("nul\x00byte.sh")
    assert ok is False
    assert "NUL byte" in output


def test_run_job_script_accepts_pathlike_script_path(hermes_env):
    """The eager NUL guard must not crash on a non-str script_path.

    ``"\x00" in script_path`` raises TypeError for a pathlib.Path (not
    iterable), so a Path passed by a future caller would crash the
    scheduler at the guard itself. The guard coerces with str() first;
    a valid Path must still run the script end-to-end (regression for
    the #86832 review point)."""
    from cron.scheduler_script import _run_job_script

    script = hermes_env / "scripts" / "probe.py"
    script.write_text('print("pathlike ok")\n', encoding="utf-8")

    ok, output = _run_job_script(pathlib.Path(script))
    assert ok is True
    assert "pathlike ok" in output


# ---------------------------------------------------------------------------
# _summarize_cron_failure_for_delivery: mode-aware failure attribution
# ---------------------------------------------------------------------------
#
# The summarizer classified failures by substring-matching the error prose and
# mapped any hit onto a provider-shaped explanation. For a no_agent job that is
# structurally impossible — run_job short-circuits before any model is reached —
# so a script whose own text happened to contain "timed out", "429" or
# "authentication" had its failure attributed to a provider it never called.
#
# Observed in practice: _run_job_script reports a timeout as "Script timed out
# after {n}s: {path}", which was delivered to chat as "provider timeout. Fallback
# chain was exhausted or unavailable." for a job that never opened a socket.
#
# The summarizer had no direct test coverage — the only test referencing it
# mocks it out and asserts on its arguments — which is why this shipped.


@pytest.mark.parametrize(
    "error",
    [
        "Script timed out after 900s: /home/u/.hermes/scripts/nightly.sh",
        "Script failed: curl returned 429 from api.example.com",
        "Script failed: gpg authentication failed for key",
        "Script failed: ReadTimeout contacting localhost",
    ],
)
def test_no_agent_failure_never_blamed_on_a_provider(error):
    """A script job's failure must never be reported as a provider/fallback failure."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "nightly-job", "no_agent": True, "script": "nightly.sh"}
    msg = _summarize_cron_failure_for_delivery(job, error)

    assert "provider" not in msg.lower()
    assert "fallback chain" not in msg.lower()
    # The operator must be pointed at what actually failed.
    assert "script" in msg.lower()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("ReadTimeout: provider did not respond", "provider timeout"),
        ("HTTP 429 rate limit exceeded", "provider rate limit"),
        ("HTTP 401 authentication failed", "provider authentication error"),
    ],
)
def test_agent_job_provider_classification_unchanged(error, expected):
    """Regression guard: agent-mode jobs keep the provider-shaped summaries."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "daily-digest", "no_agent": False}
    assert expected in _summarize_cron_failure_for_delivery(job, error)


def test_a_routed_profile_script_never_receives_a_launch_only_name(hermes_env, monkeypatch):
    """A no_agent script fired for a SIBLING profile runs with that profile's scope overlaid and
    NONE of the launch profile's residue (#107695 review): a name the launch ``.env`` defines, and a
    name a launch external source SUPPLIED — applied, or skipped because a process value already won
    (``skipped_existing``, so it never entered the provenance map) — reach the child unset. The
    routed profile's own values come through, and the parent ``os.environ`` is never mutated."""
    import os

    from agent import secret_scope
    from agent.secret_sources import registry as reg_module
    from agent.secret_sources.base import FetchResult
    from agent.secret_sources.registry import ApplyReport, SourceReport
    from cron.scheduler_script import _run_job_script
    from hermes_cli import env_loader
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

    launch = get_process_hermes_home()
    (launch / ".env").write_text("LAUNCH_ONLY_VALUE=launch-only\nCUSTOM_CRON_VALUE=launch\n", encoding="utf-8")
    (launch / "config.yaml").write_text("secrets:\n  test-source:\n    enabled: true\n", encoding="utf-8")
    for name, value in (("LAUNCH_ONLY_VALUE", "launch-only"), ("CUSTOM_CRON_VALUE", "launch"),
                        ("LAUNCH_VAULT_ONLY", "launch-vault-value"), ("LAUNCH_SKIPPED_SECRET", "launch-value")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(env_loader, "_SOURCE_SUPPLIED_NAMES", set())
    monkeypatch.setattr(env_loader, "_SECRET_SOURCES", {"LAUNCH_VAULT_ONLY": "vault", "ROUTED_VAULT_ONLY": "vault"})
    monkeypatch.setattr(env_loader, "_APPLIED_HOMES", set())
    monkeypatch.setattr(env_loader, "_SECRET_SOURCE_VALUES_BY_HOME", {})
    monkeypatch.setattr(reg_module, "apply_all", lambda _cfg, home_path, **_kw: ApplyReport(
        sources=[SourceReport(name="test-source", label="Test Source", result=FetchResult(),
                              applied=[], skipped_existing=["LAUNCH_SKIPPED_SECRET"])],
        provenance={}))
    env_loader._apply_external_secret_sources(launch)  # the real registry path; the source loses to the env
    environ_before = dict(os.environ)

    routed = launch / "profiles" / "ops"
    (routed / "scripts").mkdir(parents=True, exist_ok=True)
    # Under the routed home override the runner resolves scripts against THAT profile's scripts dir.
    script = routed / "scripts" / "probe_launch_only.sh"
    script.write_text(
        '#!/bin/bash\necho "${CUSTOM_CRON_VALUE}|${ROUTED_VAULT_ONLY}|${LAUNCH_ONLY_VALUE:-<unset>}'
        '|${LAUNCH_VAULT_ONLY:-<unset>}|${LAUNCH_SKIPPED_SECRET:-<unset>}"\n'
    )

    home_token = set_hermes_home_override(str(routed))
    context_token = secret_scope.set_multiplex_context(True)
    scope_token = secret_scope.set_secret_scope(
        {"CUSTOM_CRON_VALUE": "routed", "ROUTED_VAULT_ONLY": "routed-vault-value"})
    try:
        ok, output = _run_job_script("probe_launch_only.sh")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(context_token)
        reset_hermes_home_override(home_token)

    assert ok, output
    assert output.strip() == "routed|routed-vault-value|<unset>|<unset>|<unset>"
    assert dict(os.environ) == environ_before


def test_a_routed_profile_script_keeps_administrator_managed_values_over_its_own(hermes_env, monkeypatch):
    """Managed-scope precedence (#107695 review on f5f88d5058): the administrator's managed ``.env`` is
    applied LAST with override in the launch process, so it beats the user's own ``.env``. Managed keys
    are not launch residue, and they are re-applied over the routed scope so the child sees the same
    precedence the launch process does."""
    import os

    from agent import secret_scope
    from cron.scheduler_script import _run_job_script
    from hermes_cli import env_loader, managed_scope
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

    launch = get_process_hermes_home()
    managed = launch / "managed"
    managed.mkdir()
    (managed / ".env").write_text("ORG_POLICY_FLAG=managed-value\n", encoding="utf-8")
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set(env_loader._LOADED_DOTENV_KEYS))
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", set())
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed)
    monkeypatch.setenv("ORG_POLICY_FLAG", "placeholder")
    env_loader._apply_managed_env()  # the boot-time managed load
    assert os.environ["ORG_POLICY_FLAG"] == "managed-value"

    routed = launch / "profiles" / "ops"
    (routed / "scripts").mkdir(parents=True, exist_ok=True)
    script = routed / "scripts" / "probe_policy.sh"
    script.write_text('#!/bin/bash\necho "${ORG_POLICY_FLAG:-<unset>}"\n')

    home_token = set_hermes_home_override(str(routed))
    context_token = secret_scope.set_multiplex_context(True)
    # The routed user's own .env carries a competing value for the managed key.
    scope_token = secret_scope.set_secret_scope({"ORG_POLICY_FLAG": "user-value"})
    try:
        ok, output = _run_job_script("probe_policy.sh")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(context_token)
        reset_hermes_home_override(home_token)

    assert ok, output
    assert output.strip() == "managed-value"
    assert os.environ["ORG_POLICY_FLAG"] == "managed-value"  # parent untouched
