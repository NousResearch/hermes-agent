"""End-to-end: the real external-worker payload runner serves a scope credential
to a no_agent script child.

This drives ``cron.scheduler._run_external_worker_payload`` exactly as the
gateway does — durable execution handoff/adoption, payload file, ack file,
multiplex flag, the job read from the profile's own cron store — down to a real
no_agent script child. It is the behavior proof for the multiplex
gateway-dispatch path that used to fail with "SOLORECALL_API_KEY is not set"
while the standalone ticker path kept working.
"""

from __future__ import annotations

import json

import pytest

_SCOPE_NAME = "SCRIPT_BRIDGE_E2E_VAR"


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "ops-e2e"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir(parents=True)
    (home / ".env").write_text(f"{_SCOPE_NAME}=gateway-dispatch-value\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv(_SCOPE_NAME, raising=False)
    return home


@pytest.fixture(autouse=True)
def _reset_contextvars():
    from hermes_constants import get_hermes_home_override, set_hermes_home_override

    before = get_hermes_home_override()
    yield
    if get_hermes_home_override() is not before:
        set_hermes_home_override(before)


def _run_gateway_fire(profile_home, tmp_path, monkeypatch, script_body: str, job_id: str):
    """Drive one gateway-dispatched fire and return the child's observations.

    Mirrors the gateway: job stored in the profile's cron store, durable attempt
    created and fenced for handoff, payload written, worker adopts + acks, then
    the no_agent script runs as a real child process.
    """
    import cron.executions as executions
    from cron.jobs import create_job, get_job, use_cron_store
    from cron.scheduler import _run_external_worker_payload

    observed = tmp_path / f"{job_id}-child-env.json"
    (profile_home / "scripts" / "watchdog.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        f"Path({str(observed)!r}).write_text(json.dumps({{"
        f"{script_body}"
        "}))\n"
        "print('watchdog ok')\n",
        encoding="utf-8",
    )

    with use_cron_store(profile_home):
        job = create_job(
            prompt=None, schedule="every 1h", name=job_id, script="watchdog.py",
            no_agent=True, deliver="local",
        )
        job = get_job(job["id"])
        assert job is not None

    # Durable attempt created by the gateway before spawn, then fenced for handoff.
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", profile_home / "cron" / "executions.db")
    execution = executions.create_execution(job["id"], source="builtin")
    assert executions.mark_execution_handoff_pending(execution["id"]) is not None
    job["execution_id"] = execution["id"]

    payload = tmp_path / "payload.json"
    ack = tmp_path / "payload.ready"
    payload.write_text(
        json.dumps({
            "job": job,
            "profile_home": str(profile_home),
            "multiplex_active": True,
        }),
        encoding="utf-8",
    )

    assert _run_external_worker_payload(payload, ack) is True

    assert ack.exists(), "worker never published its ownership acknowledgement"
    assert not payload.exists(), "handoff payload must be consumed"
    assert observed.exists(), "no_agent script child did not run"
    return json.loads(observed.read_text(encoding="utf-8"))


def test_external_worker_payload_delivers_scope_credential_to_script_child(
    profile_home, tmp_path, monkeypatch,
):
    """The reported failure, end to end through the gateway's own runner."""
    child = _run_gateway_fire(
        profile_home, tmp_path, monkeypatch,
        script_body=(
            f"'scope': os.environ.get({_SCOPE_NAME!r}), "
            "'hermes_home': os.environ.get('HERMES_HOME')"
        ),
        job_id="e2e-scope-job",
    )

    assert child["scope"] == "gateway-dispatch-value"
    assert child["hermes_home"] == str(profile_home)


def test_external_worker_payload_scope_beats_ambient_sibling_value(
    profile_home, tmp_path, monkeypatch,
):
    """Full path, leak direction: the server process env already carries another
    profile's value for the same credential name (a shared launch env). The
    child must resolve the served profile's value, never the ambient one."""
    monkeypatch.setenv(_SCOPE_NAME, "ambient-other-profile-value")

    child = _run_gateway_fire(
        profile_home, tmp_path, monkeypatch,
        script_body=f"'scope': os.environ.get({_SCOPE_NAME!r})",
        job_id="e2e-sibling-job",
    )

    assert child["scope"] == "gateway-dispatch-value"
