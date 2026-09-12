"""Tests for cronjob no_agent mode — script-driven jobs that skip the LLM.

Covers:

* ``create_job(no_agent=True)`` shape, validation, and serialization.
* ``cronjob(action='create', no_agent=True)`` tool-level validation.
* ``cronjob(action='update')`` flipping no_agent on/off.
* ``scheduler.run_job`` short-circuit path: success/silent/failure.
* Shell script support in ``_run_job_script`` (.sh runs via bash).
"""

from __future__ import annotations


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


def test_a_routed_profile_script_receives_its_own_profile_env(hermes_env, monkeypatch):
    """A no_agent script fired for a SIBLING profile sees that profile's .env values — via the
    installed scope, never by copying them into the parent's os.environ (#107692 review)."""
    import os

    from agent import secret_scope
    from cron.scheduler_script import _run_job_script

    monkeypatch.setenv("CUSTOM_CRON_VALUE", "launch")
    monkeypatch.delenv("ROUTED_ONLY_VALUE", raising=False)
    script = hermes_env / "scripts" / "probe_env.sh"
    script.write_text('#!/bin/bash\necho "${CUSTOM_CRON_VALUE}|${ROUTED_ONLY_VALUE}"\n')

    context_token = secret_scope.set_multiplex_context(True)
    scope_token = secret_scope.set_secret_scope(
        {"CUSTOM_CRON_VALUE": "routed", "ROUTED_ONLY_VALUE": "routed-only"})
    try:
        ok, output = _run_job_script("probe_env.sh")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(context_token)

    assert ok, output
    assert output.strip() == "routed|routed-only"
    assert os.environ["CUSTOM_CRON_VALUE"] == "launch"  # the parent process was not mutated


def test_a_routed_profile_script_never_receives_a_launch_profile_only_value(hermes_env, monkeypatch):
    """Negative control for the overlay above (#107695 review): a name the LAUNCH profile's .env
    defines and the routed scope does not must reach the routed child UNSET — not with the launch
    value. The secret scrub only knows classified names, so a custom or unclassified secret would
    otherwise cross the profile boundary; the launch profile's dotenv residue is dropped first."""
    import os

    from agent import secret_scope
    from cron.scheduler_script import _run_job_script
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

    launch = get_process_hermes_home()
    (launch / ".env").write_text("LAUNCH_ONLY_VALUE=launch-only\nCUSTOM_CRON_VALUE=launch\n", encoding="utf-8")
    monkeypatch.setenv("LAUNCH_ONLY_VALUE", "launch-only")
    monkeypatch.setenv("CUSTOM_CRON_VALUE", "launch")
    routed = launch / "profiles" / "ops"
    (routed / "scripts").mkdir(parents=True, exist_ok=True)
    # Under the routed home override the runner resolves scripts against THAT profile's scripts dir.
    script = routed / "scripts" / "probe_launch_only.sh"
    script.write_text('#!/bin/bash\necho "${CUSTOM_CRON_VALUE}|${LAUNCH_ONLY_VALUE:-<unset>}"\n')

    home_token = set_hermes_home_override(str(routed))
    context_token = secret_scope.set_multiplex_context(True)
    scope_token = secret_scope.set_secret_scope({"CUSTOM_CRON_VALUE": "routed"})
    try:
        ok, output = _run_job_script("probe_launch_only.sh")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(context_token)
        reset_hermes_home_override(home_token)

    assert ok, output
    assert output.strip() == "routed|<unset>"
    assert os.environ["LAUNCH_ONLY_VALUE"] == "launch-only"  # the parent process was not mutated


def test_a_routed_profile_script_never_receives_a_launch_external_source_value(hermes_env, monkeypatch):
    """External secret sources (vault, 1Password, ...) write their names into the shared
    ``os.environ`` too, and ``strip_launch_profile_env`` only knows dotenv- and terminal-owned
    names. A name the LAUNCH profile's source supplied must still reach the routed child unset
    (#107695 review); a name the ROUTED profile's own source supplies must come through."""
    import os

    from agent import secret_scope
    from cron.scheduler_script import _run_job_script
    from hermes_cli import env_loader
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

    launch = get_process_hermes_home()
    routed = launch / "profiles" / "ops"
    (routed / "scripts").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LAUNCH_VAULT_ONLY", "launch-vault-value")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "LAUNCH_VAULT_ONLY", "vault")
    monkeypatch.setitem(env_loader._SECRET_SOURCES, "ROUTED_VAULT_ONLY", "vault")
    script = routed / "scripts" / "probe_vault.sh"
    script.write_text('#!/bin/bash\necho "${LAUNCH_VAULT_ONLY:-<unset>}|${ROUTED_VAULT_ONLY:-<unset>}"\n')

    home_token = set_hermes_home_override(str(routed))
    context_token = secret_scope.set_multiplex_context(True)
    scope_token = secret_scope.set_secret_scope({"ROUTED_VAULT_ONLY": "routed-vault-value"})
    try:
        ok, output = _run_job_script("probe_vault.sh")
    finally:
        secret_scope.reset_secret_scope(scope_token)
        secret_scope.reset_multiplex_context(context_token)
        reset_hermes_home_override(home_token)

    assert ok, output
    assert output.strip() == "<unset>|routed-vault-value"
    assert os.environ["LAUNCH_VAULT_ONLY"] == "launch-vault-value"  # parent untouched
