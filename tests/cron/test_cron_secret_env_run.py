"""Job-scoped ``secret_env`` end to end: a no_agent script receives the value and nothing else does.

Real ``run_job`` / ``_execute_job_now`` against a temp ``HERMES_HOME``; the only fake is the
``wincred`` backend, swapped for an in-memory one (the credential store is never touched).
A run that resolves anything locks the script first, which only Windows can do: elsewhere
secret_env fails closed (``test_cron_secret_env_boundaries.py``), so those tests run on Windows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil

import pytest

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
TARGET = "hermes/notice-report-slack"
NAME = "NOTICE_REPORT_SLACK_TOKEN"

_ECHO_SCRIPT = f"""\
import hashlib, os, pathlib, sys
pathlib.Path(sys.argv[0]).with_suffix(".started").write_text("1")
value = os.environ.get("{NAME}", "")
print("sha=" + hashlib.sha256(value.encode()).hexdigest())
print("raw=" + value)
print("err=" + value, file=sys.stderr)
sys.exit(int(os.environ.get("CRON_SECRET_TEST_EXIT", "0")))
"""


class FakeBackend:
    def __init__(self, values):
        self.values = values
        self.calls: list[str] = []

    def resolve(self, target):
        self.calls.append(target)
        return self.values[target]


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "scripts").mkdir(parents=True)
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    import cron.jobs
    import cron.scheduler
    import hermes_constants
    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    importlib.reload(cron.scheduler)
    return home


@pytest.fixture
def backend(monkeypatch):
    from cron import secret_env

    fake = FakeBackend({TARGET: FAKE_VALUE})
    monkeypatch.setattr(secret_env, "_default_backends", lambda: {"wincred": fake})
    return fake


def _create(home, script_body=_ECHO_SCRIPT, name="echo.py", **kw):
    from cron.jobs import create_job

    script = home / "scripts" / name
    script.write_text(script_body, encoding="utf-8")
    kw.setdefault("secret_env", {NAME: f"wincred://{TARGET}"})
    kw.setdefault("secret_env_script_sha256", hashlib.sha256(script.read_bytes()).hexdigest())
    return create_job(prompt=None, schedule="every 5m", script=name, no_agent=True,
                      deliver="local", **kw), script


def _files_holding_value(root) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and FAKE_VALUE.encode() in path.read_bytes():
            hits.append(str(path))
    return hits


def _expected_sha() -> str:
    return "sha=" + hashlib.sha256(FAKE_VALUE.encode()).hexdigest()


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("redact_enabled", [True, False])
def test_no_agent_script_receives_the_value_and_output_is_scrubbed(
        hermes_env, backend, monkeypatch, caplog, redact_enabled):
    import agent.redact
    from cron.scheduler import run_job

    monkeypatch.setattr(agent.redact, "_REDACT_ENABLED", redact_enabled)
    monkeypatch.setenv(NAME, "parent-process-value")  # scrub-then-overlay wins over inheritance
    caplog.set_level(logging.DEBUG)
    job, _ = _create(hermes_env)

    success, doc, final_response, error = run_job(job)

    assert success is True and error is None
    assert _expected_sha() in final_response
    assert "raw=[REDACTED-SECRET-ENV]" in final_response
    for text in (doc, final_response, caplog.text):
        assert FAKE_VALUE not in text and TARGET not in text
    assert os.environ[NAME] == "parent-process-value"
    assert backend.calls == [TARGET]


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("redact_enabled", [True, False])
def test_failed_script_output_is_scrubbed_everywhere_it_persists(
        hermes_env, backend, monkeypatch, caplog, redact_enabled):
    import agent.redact
    from tools.cronjob_tools import _execute_job_now

    monkeypatch.setattr(agent.redact, "_REDACT_ENABLED", redact_enabled)
    monkeypatch.setenv("CRON_SECRET_TEST_EXIT", "3")
    caplog.set_level(logging.DEBUG)
    job, _ = _create(hermes_env)

    result = _execute_job_now(job)

    assert result["claimed"] is True and result["success"] is False
    assert FAKE_VALUE not in json.dumps(result) and FAKE_VALUE not in caplog.text
    stored = json.loads((hermes_env / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"][0]
    assert "[REDACTED-SECRET-ENV]" in stored["last_error"]
    assert _files_holding_value(hermes_env) == []
    assert NAME not in os.environ


@pytest.mark.platforms("windows")
def test_manual_run_succeeds_and_persists_nothing_sensitive(hermes_env, backend, caplog):
    from tools.cronjob_tools import _execute_job_now

    caplog.set_level(logging.DEBUG)
    job, _ = _create(hermes_env)
    result = _execute_job_now(job)
    assert result["success"] is True
    outputs = list((hermes_env / "cron" / "output").rglob("*.md"))
    assert outputs and _expected_sha() in outputs[0].read_text(encoding="utf-8")
    assert _files_holding_value(hermes_env) == []
    assert FAKE_VALUE not in caplog.text and TARGET not in caplog.text


@pytest.mark.platforms("windows")
def test_resolution_failure_never_starts_the_script(hermes_env, monkeypatch, caplog):
    from cron import secret_env
    from cron.scheduler import run_job

    monkeypatch.setattr(secret_env, "_default_backends", lambda: {"wincred": FakeBackend({})})
    caplog.set_level(logging.DEBUG)
    job, script = _create(hermes_env)

    success, doc, alert, error = run_job(job)

    assert success is False
    assert not script.with_suffix(".started").exists()
    assert error == f"Secret env resolution failed: {NAME} (internal)"
    assert error in alert
    for text in (doc, alert, error, caplog.text):
        assert TARGET not in text and FAKE_VALUE not in text


@pytest.mark.platforms("windows")
def test_changed_script_is_neither_resolved_nor_started(hermes_env, backend):
    from cron.jobs import get_job, update_job
    from cron.scheduler import run_job
    from cron.secret_env import script_sha256

    job, script = _create(hermes_env)
    script.write_text(_ECHO_SCRIPT + "# tampered\n", encoding="utf-8")

    success, _doc, _alert, error = run_job(job)
    assert success is False and "repin-script" in error
    assert backend.calls == []
    assert not script.with_suffix(".started").exists()

    update_job(job["id"], {"secret_env_script_sha256": script_sha256(script)})
    success, _doc, final_response, _error = run_job(get_job(job["id"]))
    assert success is True and _expected_sha() in final_response


def test_wake_gate_and_monitor_scripts_never_receive_the_value(hermes_env, backend):
    """Agent-job pre-run scripts and monitor scripts share ``_run_job_script`` but never get a
    resolver, even on a hand-edited record that carries ``secret_env``."""
    from cron.monitor import _run_monitor_source
    from cron.scheduler_script import _run_job_script_with_claim_heartbeat

    _create(hermes_env)
    hand_edited = {"id": "x", "no_agent": False, "script": "echo.py", "monitor_script": "echo.py",
                   "secret_env": {NAME: f"wincred://{TARGET}"}}
    empty_sha = "sha=" + hashlib.sha256(b"").hexdigest()

    ok, output = _run_job_script_with_claim_heartbeat(hand_edited, "echo.py")
    assert ok and empty_sha in output
    ok, output = _run_monitor_source(hand_edited)
    assert ok and empty_sha in output
    assert backend.calls == []


@pytest.mark.platforms("windows")
def test_a_sibling_job_without_secret_env_gets_nothing(hermes_env, backend):
    from cron.jobs import create_job
    from cron.scheduler import run_job

    job, _ = _create(hermes_env)
    assert run_job(job)[0] is True
    plain = create_job(prompt=None, schedule="every 5m", script="echo.py", no_agent=True,
                       deliver="local")
    success, _doc, final_response, _error = run_job(plain)
    assert success is True
    assert "sha=" + hashlib.sha256(b"").hexdigest() in final_response
    assert backend.calls == [TARGET]


@pytest.mark.platforms("windows")
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash (Git Bash on Windows)")
def test_shell_script_receives_the_value(hermes_env, backend):
    from cron.scheduler import run_job

    body = f'#!/usr/bin/env bash\nprintf "len=%s\\n" "${{#{NAME}}}"\n'
    job, _ = _create(hermes_env, script_body=body, name="len.sh")
    success, _doc, final_response, _error = run_job(job)
    assert success is True and final_response == f"len={len(FAKE_VALUE)}"


@pytest.mark.platforms("windows")
def test_hand_edited_raw_value_fails_closed(hermes_env, backend):
    from cron.scheduler import run_job

    job, script = _create(hermes_env)
    job = {**job, "secret_env": {NAME: FAKE_VALUE}}
    success, doc, alert, error = run_job(job)
    assert success is False
    assert error == f"Secret env resolution failed: {NAME} (ref_invalid)"
    assert FAKE_VALUE not in doc + alert
    assert not script.with_suffix(".started").exists()
