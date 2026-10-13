"""``hermes cron`` secret_env lane (CORP-764): parsing, edit merge, repin, and what gets printed.

The dashboard update route is covered here too: it must refuse secret_env outright.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import pytest

from cron.jobs import create_job, get_job
from hermes_cli.cron import cron_command
from hermes_cli.subcommands.cron import build_cron_parser

FAKE_VALUE = "FAKE-SECRET-VALUE-0123456789"
TARGET = "hermes/notice-report-slack"
NAME = "NOTICE_REPORT_SLACK_TOKEN"


@pytest.fixture()
def cron_env(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
    scripts = get_hermes_home() / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "notice.py").write_text("print('notice')\n", encoding="utf-8")
    return tmp_path, scripts


def _run(*argv):
    parser = argparse.ArgumentParser(prog="hermes")
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=cron_command)
    return cron_command(parser.parse_args(["cron", *argv]))


def _create(capsys, *extra):
    rc = _run("create", "every 1h", "--no-agent", "--script", "notice.py",
              "--secret-env", f"{NAME}=wincred://{TARGET}", *extra)
    out = capsys.readouterr().out
    assert rc == 0, out
    job_id = out.split("Created job: ", 1)[1].split()[0]
    return job_id, out


def _all_output(capsys, out=""):
    captured = capsys.readouterr()
    return out + captured.out + captured.err


def test_create_stores_reference_pins_script_and_prints_name_and_scheme(cron_env, capsys):
    _, scripts = cron_env
    job_id, out = _create(capsys)
    stored = get_job(job_id)
    assert stored["secret_env"] == {NAME: f"wincred://{TARGET}"}
    assert stored["secret_env_script_sha256"] == hashlib.sha256(
        (scripts / "notice.py").read_bytes()).hexdigest()
    assert f"{NAME} (wincred)" in out
    assert TARGET not in out


def test_list_shows_names_and_schemes_only(cron_env, capsys):
    _create(capsys)
    cron_command(argparse.Namespace(cron_command="list", all=True))
    out = capsys.readouterr().out
    assert f"{NAME} (wincred)" in out
    assert TARGET not in out


@pytest.mark.parametrize("item", [FAKE_VALUE, f"{NAME}"])
def test_entry_without_equals_is_rejected_without_echo(cron_env, capsys, item):
    rc = _run("create", "every 1h", "--no-agent", "--script", "notice.py", "--secret-env", item)
    out = _all_output(capsys)
    assert rc == 1
    assert "NAME=REF" in out
    assert FAKE_VALUE not in out


def test_duplicate_and_raw_values_are_rejected_without_echo(cron_env, capsys):
    rc = _run("create", "every 1h", "--no-agent", "--script", "notice.py",
              "--secret-env", f"{NAME}=wincred://{TARGET}", "--secret-env", f"{NAME}=wincred://x")
    assert rc == 1 and "more than once" in _all_output(capsys)
    rc = _run("create", "every 1h", "--no-agent", "--script", "notice.py",
              "--secret-env", f"{NAME}={FAKE_VALUE}")
    out = _all_output(capsys)
    assert rc == 1 and FAKE_VALUE not in out


def test_edit_adds_replaces_removes_and_clears(cron_env, capsys):
    tmp_path, _ = cron_env
    job_id, _ = _create(capsys)

    assert _run("edit", job_id, "--secret-env", "JOB_OTHER_TOKEN=wincred://hermes/other",
                "--secret-env", f"{NAME}=wincred://hermes/new") == 0
    assert get_job(job_id)["secret_env"] == {
        NAME: "wincred://hermes/new", "JOB_OTHER_TOKEN": "wincred://hermes/other"}

    assert _run("edit", job_id, "--remove-secret-env", "JOB_OTHER_TOKEN") == 0
    assert get_job(job_id)["secret_env"] == {NAME: "wincred://hermes/new"}

    assert _run("edit", job_id, "--remove-secret-env", "NOT_SET_TOKEN") == 1

    assert _run("edit", job_id, "--clear-secret-env") == 0
    stored = get_job(job_id)
    assert "secret_env" not in stored and "secret_env_script_sha256" not in stored
    out = _all_output(capsys)
    assert TARGET not in out and "hermes/new" not in out and "hermes/other" not in out
    raw = (tmp_path / "cron" / "jobs.json").read_text(encoding="utf-8")
    assert "secret_env" not in raw


def test_repin_script_is_an_explicit_operation(cron_env, capsys):
    _, scripts = cron_env
    job_id, _ = _create(capsys)
    pinned = get_job(job_id)["secret_env_script_sha256"]

    (scripts / "notice.py").write_text("print('reviewed edit')\n", encoding="utf-8")
    assert _run("edit", job_id, "--name", "renamed") == 0
    assert get_job(job_id)["secret_env_script_sha256"] == pinned

    assert _run("edit", job_id, "--repin-script") == 0
    assert get_job(job_id)["secret_env_script_sha256"] == hashlib.sha256(
        (scripts / "notice.py").read_bytes()).hexdigest() != pinned


def test_secret_env_on_an_agent_job_is_refused(cron_env, capsys):
    rc = _run("create", "every 1h", "check things", "--secret-env", f"{NAME}=wincred://{TARGET}")
    assert rc == 1 and "no_agent" in _all_output(capsys)


# ---------------------------------------------------------------------------
# Dashboard: secret_env is CLI-only
# ---------------------------------------------------------------------------


@pytest.fixture()
def dashboard(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_cli import profiles
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "scripts").mkdir(parents=True, exist_ok=True)
    (home / "scripts" / "notice.py").write_text("print('notice')\n", encoding="utf-8")
    (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: home / "profiles")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, home


@pytest.mark.parametrize("updates", [
    {"secret_env": {NAME: f"wincred://{TARGET}"}},
    {"secret_env": None},
    {"secret_env_script_sha256": "b" * 64},
])
def test_dashboard_update_refuses_secret_env(dashboard, updates):
    client, _ = dashboard
    job = create_job(prompt=None, schedule="every 1h", script="notice.py", no_agent=True,
                     secret_env={NAME: f"wincred://{TARGET}"}, secret_env_script_sha256="a" * 64)

    response = client.put(f"/api/cron/jobs/{job['id']}", params={"profile": "default"},
                          json={"updates": updates})

    assert response.status_code == 400
    assert "hermes cron edit" in response.json()["detail"]
    stored = get_job(job["id"])
    assert stored["secret_env"] == {NAME: f"wincred://{TARGET}"}
    assert stored["secret_env_script_sha256"] == "a" * 64


def test_dashboard_create_ignores_secret_env(dashboard):
    client, home = dashboard
    response = client.post("/api/cron/jobs", params={"profile": "default"}, json={
        "schedule": "every 1h", "script": "notice.py", "no_agent": True,
        "secret_env": {NAME: f"wincred://{TARGET}"}})
    assert response.status_code == 200
    raw = json.dumps(json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8")))
    assert "secret_env" not in raw


def test_edit_interpreter_on_secret_env_job_needs_repin(cron_env, capsys):
    job_id, _ = _create(capsys)
    assert _run("edit", job_id, "--interpreter", "C:/elsewhere/python.exe") == 1
    assert "repin-script" in _all_output(capsys)
    assert not get_job(job_id).get("interpreter")
    assert _run("edit", job_id, "--interpreter", "C:/elsewhere/python.exe", "--repin-script") == 0
    assert get_job(job_id)["interpreter"] == "C:/elsewhere/python.exe"


def test_dashboard_cannot_retarget_a_secret_env_job(dashboard):
    client, _ = dashboard
    job = create_job(prompt=None, schedule="every 1h", script="notice.py", no_agent=True,
                     secret_env={NAME: f"wincred://{TARGET}"}, secret_env_script_sha256="a" * 64)
    response = client.put(f"/api/cron/jobs/{job['id']}", params={"profile": "default"},
                          json={"updates": {"interpreter": "C:/elsewhere/python.exe"}})
    assert response.status_code == 400
    assert not get_job(job["id"]).get("interpreter")


def test_cli_errors_do_not_echo_a_pasted_value_as_a_name(cron_env, capsys):
    job_id, _ = _create(capsys)
    assert _run("edit", job_id, "--remove-secret-env", FAKE_VALUE) == 1
    out = _all_output(capsys)
    assert "<invalid name>" in out and FAKE_VALUE not in out
