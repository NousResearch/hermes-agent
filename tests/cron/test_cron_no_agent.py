"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations

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
    import hermes_state
    importlib.reload(hermes_state)  # DEFAULT_DB_PATH binds at import time
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




# ---------------------------------------------------------------------------
# scheduler.run_job: short-circuit behavior
# ---------------------------------------------------------------------------


def test_run_job_no_agent_success_returns_script_stdout(hermes_env):
    """Happy path: script exits 0 with output, delivered verbatim."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/usr/bin/env bash\necho 'RAM 92% on host'\n")

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
    script_path.write_text('#!/usr/bin/env bash\necho "ok"\n')

    job = create_job(
        prompt=None, schedule="every 5m", script="probe.sh", no_agent=True, deliver="local"
    )
    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert loaded_homes, "load_hermes_dotenv was not called on the no_agent path"
    assert str(loaded_homes[0]) == str(hermes_env)


_PRESENCE_PROBE = (
    "#!/usr/bin/env bash\n"
    'for n in JOB_SVC_TOKEN LAUNCH_ONLY_TOKEN; do [ -n "${!n}" ] && echo "$n=set" || echo "$n=MISSING"; done\n'
)


def test_no_agent_script_gets_owning_profiles_declared_secret_never_launch_residue(
    hermes_env, monkeypatch, tmp_path,
):
    """Routed profile B declares JOB_SVC_TOKEN in terminal.env_passthrough and defines it only in
    its own .env (never in the process env); the launch profile A's .env credential is in the
    process env. B's script sees its own secret and not A's (#114209)."""
    from agent.secret_scope import (
        build_profile_secret_scope, reset_secret_scope, set_multiplex_active, set_secret_scope)
    from cron.scheduler_script import _run_job_script
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    (hermes_env / ".env").write_text("LAUNCH_ONLY_TOKEN=launch-secret\n", encoding="utf-8")
    monkeypatch.setenv("LAUNCH_ONLY_TOKEN", "launch-secret")  # what load_hermes_dotenv() did at startup
    monkeypatch.delenv("JOB_SVC_TOKEN", raising=False)
    routed = tmp_path / "routed"
    (routed / "scripts").mkdir(parents=True)
    (routed / ".env").write_text("JOB_SVC_TOKEN=routed-secret\n", encoding="utf-8")
    (routed / "config.yaml").write_text("terminal:\n  env_passthrough: [JOB_SVC_TOKEN]\n", encoding="utf-8")
    (routed / "scripts" / "probe.sh").write_text(_PRESENCE_PROBE, encoding="utf-8")

    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(routed))
    scope_token = set_secret_scope(build_profile_secret_scope(routed))
    try:
        ok, output = _run_job_script("probe.sh")
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)

    assert ok is True
    assert output.splitlines() == ["JOB_SVC_TOKEN=set", "LAUNCH_ONLY_TOKEN=MISSING"]


def test_no_agent_script_of_launch_profile_keeps_its_own_env_credential(hermes_env, monkeypatch):
    """Single-profile documented flow: the launch profile's own script still inherits the
    credential its .env put in the process env — nothing is stripped for a non-routed job."""
    from cron.scheduler_script import _run_job_script

    (hermes_env / ".env").write_text("LAUNCH_ONLY_TOKEN=launch-secret\n", encoding="utf-8")
    monkeypatch.setenv("LAUNCH_ONLY_TOKEN", "launch-secret")
    monkeypatch.delenv("JOB_SVC_TOKEN", raising=False)
    (hermes_env / "scripts" / "probe.sh").write_text(_PRESENCE_PROBE, encoding="utf-8")

    ok, output = _run_job_script("probe.sh")

    assert ok is True
    assert output.splitlines() == ["JOB_SVC_TOKEN=MISSING", "LAUNCH_ONLY_TOKEN=set"]






# ---------------------------------------------------------------------------
# _run_job_script: shell-script support
# ---------------------------------------------------------------------------




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

    assert "ai model service" not in msg.lower()
    assert "backup provider" not in msg.lower()
    # The operator must be pointed at what actually failed.
    assert "script" in msg.lower()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("ReadTimeout: provider did not respond", "did not respond in time"),
        ("HTTP 429 rate limit exceeded", "rate-limited"),
        ("HTTP 401 authentication failed", "rejected the sign-in"),
    ],
)
def test_agent_job_provider_classification_unchanged(error, expected):
    """Regression guard: agent-mode jobs keep the provider-shaped summaries."""
    from cron.scheduler import _summarize_cron_failure_for_delivery

    job = {"name": "daily-digest", "no_agent": False}
    assert expected in _summarize_cron_failure_for_delivery(job, error)


# ---------------------------------------------------------------------------
# no_agent jobs honoring a configured Python interpreter
# ---------------------------------------------------------------------------


def test_run_job_no_agent_uses_configured_interpreter(hermes_env):
    """A no-agent job's script must run through the configured interpreter.

    Proves the override survives the no_agent branch of ``run_job`` →
    ``_run_job_script_with_claim_heartbeat`` → ``_run_job_script``.
    """
    import stat as _stat
    import sys

    from cron.jobs import create_job
    from cron.scheduler import run_job

    # A wrapper that re-execs the real interpreter with an env marker.
    wrapper = hermes_env / "venv" / "bin" / "python3"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "env = os.environ.copy()\n"
        'env["CRON_WRAPPER_USED"] = "1"\n'
        "os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | _stat.S_IXUSR)

    script_path = hermes_env / "scripts" / "marker.py"
    script_path.write_text(
        'import os\nprint(os.environ.get("CRON_WRAPPER_USED", "0"))\n'
    )

    job = create_job(
        prompt=None,
        schedule="every 5m",
        script="marker.py",
        no_agent=True,
        deliver="local",
        interpreter=str(wrapper),
    )

    success, doc, final_response, error = run_job(job)
    assert success is True
    assert error is None
    assert final_response == "1"
    assert "1" in doc


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
        '#!/usr/bin/env bash\necho "${CUSTOM_CRON_VALUE}|${ROUTED_VAULT_ONLY}|${LAUNCH_ONLY_VALUE:-<unset>}'
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
    script.write_text('#!/usr/bin/env bash\necho "${ORG_POLICY_FLAG:-<unset>}"\n')

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


# ---------------------------------------------------------------------------
# run_job: no_agent runs are recorded as run-history sessions (#44080)
# ---------------------------------------------------------------------------


def _job_runs(job_id):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        return db.list_cron_job_runs(job_id)
    finally:
        db.close()


def _session_messages(session_id):
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        return db.get_messages(session_id)
    finally:
        db.close()


def test_run_job_no_agent_success_records_run_session(hermes_env):
    """A successful no_agent run must appear in the job's run history."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "alert.sh"
    script_path.write_text("#!/bin/bash\necho 'RAM 92% on host'\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="alert.sh", no_agent=True, deliver="local"
    )
    success, _, _, _ = run_job(job)
    assert success is True

    runs = _job_runs(job["id"])
    assert len(runs) == 1
    run = runs[0]
    assert run["id"].startswith(f"cron_{job['id']}_")
    assert run["source"] == "cron"
    # The run finished, so the GUI must not show it as still active.
    assert run["ended_at"] is not None
    assert run["end_reason"] == "cron_complete"

    # Script output is persisted so the GUI can show what the run produced.
    messages = _session_messages(run["id"])
    roles = [m["role"] for m in messages]
    assert "user" in roles and "assistant" in roles
    assistant_text = " ".join(
        str(m.get("content") or "") for m in messages if m["role"] == "assistant"
    )
    assert "RAM 92% on host" in assistant_text


def test_run_job_no_agent_failure_records_run_session(hermes_env):
    """A failed script run must be visible in run history, not silent."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "broken.sh"
    script_path.write_text("#!/bin/bash\necho oops >&2\nexit 3\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="broken.sh", no_agent=True, deliver="local"
    )
    success, _, _, _ = run_job(job)
    assert success is False

    runs = _job_runs(job["id"])
    assert len(runs) == 1
    run = runs[0]
    assert run["ended_at"] is not None
    assert run["end_reason"] == "cron_failed"

    messages = _session_messages(run["id"])
    assistant_text = " ".join(
        str(m.get("content") or "") for m in messages if m["role"] == "assistant"
    )
    assert "script failed" in assistant_text


def test_run_job_no_agent_silent_records_run_session(hermes_env):
    """A silent run (empty stdout) still leaves a run record."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "quiet.sh"
    script_path.write_text("#!/bin/bash\ntrue\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="quiet.sh", no_agent=True, deliver="local"
    )
    success, _, _, _ = run_job(job)
    assert success is True

    runs = _job_runs(job["id"])
    assert len(runs) == 1
    assert runs[0]["ended_at"] is not None


def test_run_job_no_agent_broken_session_store_does_not_break_run(hermes_env):
    """A broken SessionDB must not prevent the script from running."""
    from cron.jobs import create_job
    from cron.scheduler import run_job

    script_path = hermes_env / "scripts" / "ok.sh"
    script_path.write_text("#!/bin/bash\necho fine\n")

    job = create_job(
        prompt=None, schedule="every 5m", script="ok.sh", no_agent=True, deliver="local"
    )

    with patch("hermes_state.SessionDB", side_effect=RuntimeError("db locked")):
        success, output, final_response, error = run_job(job)

    assert success is True
    assert "fine" in output
