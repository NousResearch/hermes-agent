"""litco-supervisor: the slot API, driven over HTTP with a recorded command runner."""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import urllib.error
import urllib.request

import pytest

from tests.deploy._supervisor_fake import HOST, FakeRunner, sup
from tests.host._load import load_script

SUPERVISOR_SECRET = "sv-SUPERVISOR-SECRET-0123456789abcdef"
HOST_SECRET = "hs-SLOT-HOST-SECRET-0123456789abcdef01"
AGENT_TOKEN = "lkm_SLOT_AGENT_TOKEN_SECRET"
MODEL_KEY = "sk-ant-SLOT-MODEL-KEY-SECRET"
SLACK_TOKEN = "xoxb-SLOT-SLACK-SECRET"
SECRETS = (SUPERVISOR_SECRET, HOST_SECRET, AGENT_TOKEN, MODEL_KEY, SLACK_TOKEN)

MATTER_A = "a1b2c3d4-0000-4000-8000-000000000001"
MATTER_B = "b9c8d7e6-0000-4000-8000-000000000002"


def body(matter_id=MATTER_A, **extra):
    doc = {
        "matterId": matter_id, "instanceUrl": "https://firm.litco.ai", "hostSecret": HOST_SECRET,
        "agentToken": AGENT_TOKEN, "modelProvider": "anthropic", "model": "anthropic/claude-opus-4.6",
        "modelKey": MODEL_KEY, "channelEnv": {"SLACK_BOT_TOKEN": SLACK_TOKEN},
    }
    doc.update(extra)
    return doc


class Harness:
    def __init__(self, tmp_path, runner=None, **config):
        self.tmp_path = tmp_path
        self.runner = runner or FakeRunner()
        self.logs: list = []
        self.responses: list = []
        self.config = sup.Config(
            secret=SUPERVISOR_SECRET, bind="127.0.0.1", port=0, interface="",
            state_dir=tmp_path / "state", run_dir=tmp_path / "run", home_root=tmp_path / "srv" / "m",
            ref_file=tmp_path / "REF", **config,
        )
        self.start()

    def start(self):
        self.supervisor = sup.Supervisor(self.config, runner=self.runner, log=self.logs.append)
        self.server = sup.make_server(self.config, self.supervisor)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def restart(self):
        self.close()
        self.start()

    def request(self, method, path, payload=None, token=SUPERVISOR_SECRET, raw=None):
        data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method)
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode()
                status = resp.status
        except urllib.error.HTTPError as err:
            text = err.read().decode()
            status = err.code
        self.responses.append(text)
        return status, json.loads(text)

    def env_path(self, slot_id):
        return self.config.run_dir / f"{slot_id}.env"

    def state_text(self):
        return self.config.state_path.read_text()


@pytest.fixture
def harness(tmp_path):
    h = Harness(tmp_path)
    yield h
    h.close()


def assert_no_secret(text):
    for secret in SECRETS:
        assert secret not in text


# ── PUT /slots/{id} ─────────────────────────────────────────────────────────

def test_put_creates_user_home_linger_env_and_starts_the_unit(harness):
    status, doc = harness.request("PUT", "/slots/a1b2c3d4", body())
    assert status == 200
    assert doc["created"] is True and doc["userCreated"] is True and doc["started"] is True
    assert doc["slot"] == {"id": "a1b2c3d4", "matterId": MATTER_A, "port": 8800,
                           "unixUser": "m_a1b2c3d4", "state": "running"}

    home = str(harness.config.home_root / "a1b2c3d4")
    calls = harness.runner.calls
    names = harness.runner.names()
    assert names.index("getent passwd") < names.index("useradd --home-dir") < names.index("id -u")
    useradd = calls[names.index("useradd --home-dir")]
    assert useradd[-1] == "m_a1b2c3d4" and useradd[2] == home and "--no-create-home" in useradd
    assert "UID_MIN=20000" in useradd
    assert ["install", "-d", "-m", "0700", "-o", "m_a1b2c3d4", "-g", "m_a1b2c3d4", home] in calls
    assert ["install", "-d", "-m", "0711", "-o", "root", "-g", "root", str(harness.config.home_root)] in calls
    assert ["loginctl", "enable-linger", "m_a1b2c3d4"] in calls
    assert ["systemctl", "start", "litco-agent@a1b2c3d4.service"] in calls
    # the user, home and linger come before the start
    assert names.index("loginctl enable-linger") < names.index("systemctl start")


def test_env_file_is_0600_and_carries_the_slot_contract(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    env = harness.env_path("a1b2c3d4")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert stat.S_IMODE(harness.config.run_dir.stat().st_mode) == 0o700
    values = sup.parse_env_file(env.read_text())
    home = harness.config.home_root / "a1b2c3d4"
    assert values["LITCO_SLOT_ID"] == "a1b2c3d4"
    assert values["LITCO_MATTER_ID"] == MATTER_A
    assert values["LITCO_SLOT_PORT"] == values["LITCO_TURN_PORT"] == "8800"
    assert values["LITCO_TURN_HOST"] == "0.0.0.0"
    assert values["LITCO_MATTER_HOME"] == str(home / "matter")
    assert values["HERMES_HOME"] == str(home / ".hermes")
    assert values["XDG_RUNTIME_DIR"] == "/run/user/20000"
    assert values["LITCO_HOST_SECRET"] == HOST_SECRET
    assert values["LITCO_AGENT_TOKEN"] == AGENT_TOKEN
    assert values["ANTHROPIC_API_KEY"] == MODEL_KEY
    assert values["SLACK_BOT_TOKEN"] == SLACK_TOKEN
    assert values["LITCO_MODEL_KEY_ENV"] == ""
    # the supervisor's own secret never reaches a slot
    assert SUPERVISOR_SECRET not in env.read_text()


def test_custom_provider_key_goes_to_litco_model_api_key(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body(modelProvider="custom", modelBaseUrl="https://proxy.example/v1"))
    values = sup.parse_env_file(harness.env_path("a1b2c3d4").read_text())
    assert values["LITCO_MODEL_API_KEY"] == MODEL_KEY
    assert values["LITCO_MODEL_KEY_ENV"] == "LITCO_MODEL_API_KEY"
    assert "ANTHROPIC_API_KEY" not in values


def test_existing_user_is_reused(tmp_path):
    h = Harness(tmp_path, runner=FakeRunner(users=["m_a1b2c3d4"]))
    try:
        status, doc = h.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 200 and doc["userCreated"] is False
        assert "useradd --home-dir" not in h.runner.names()
        # home mode and linger are still asserted every time
        assert ["loginctl", "enable-linger", "m_a1b2c3d4"] in h.runner.calls
    finally:
        h.close()


def test_put_on_a_running_slot_does_not_restart_it(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    before = len([c for c in harness.runner.calls if c[:2] == ["systemctl", "start"]])
    status, doc = harness.request("PUT", "/slots/a1b2c3d4", body())
    after = len([c for c in harness.runner.calls if c[:2] == ["systemctl", "start"]])
    assert status == 200 and doc["started"] is False and doc["created"] is False
    assert before == after == 1
    assert doc["slot"]["state"] == "running" and doc["slot"]["port"] == 8800


def test_put_on_a_failed_unit_resets_it_then_starts(tmp_path):
    runner = FakeRunner(users=["m_a1b2c3d4"])
    h = Harness(tmp_path, runner=runner)
    try:
        h.request("PUT", "/slots/a1b2c3d4", body())
        runner.units["litco-agent@a1b2c3d4.service"] = "failed"
        status, doc = h.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 200 and doc["started"] is True
        names = runner.names()
        assert names.index("systemctl reset-failed") < len(names) - 1 - names[::-1].index("systemctl start")
    finally:
        h.close()


# ── ports ───────────────────────────────────────────────────────────────────

def test_ports_are_8800_plus_index_and_persist_across_restarts(harness):
    assert harness.request("PUT", "/slots/aaaa1111", body(MATTER_A))[1]["slot"]["port"] == 8800
    assert harness.request("PUT", "/slots/bbbb2222", body(MATTER_B))[1]["slot"]["port"] == 8801
    state = json.loads(harness.state_text())
    assert {sid: rec["port"] for sid, rec in state["slots"].items()} == {"aaaa1111": 8800, "bbbb2222": 8801}
    assert stat.S_IMODE(harness.config.state_path.stat().st_mode) == 0o600

    harness.restart()
    status, doc = harness.request("GET", "/slots")
    assert status == 200
    assert [(s["id"], s["port"]) for s in doc["slots"]] == [("aaaa1111", 8800), ("bbbb2222", 8801)]
    third = harness.request("PUT", "/slots/cccc3333", body("c0000000-0000-4000-8000-000000000003"))
    assert third[1]["slot"]["port"] == 8802


def test_a_stopped_slot_keeps_its_port(harness):
    harness.request("PUT", "/slots/aaaa1111", body(MATTER_A))
    harness.request("POST", "/slots/aaaa1111/stop")
    harness.supervisor.join_background(5)
    assert harness.request("PUT", "/slots/bbbb2222", body(MATTER_B))[1]["slot"]["port"] == 8801
    assert harness.request("PUT", "/slots/aaaa1111", body(MATTER_A))[1]["slot"]["port"] == 8800


def test_no_free_port_is_503(tmp_path):
    h = Harness(tmp_path, max_slots=1)
    try:
        assert h.request("PUT", "/slots/aaaa1111", body(MATTER_A))[0] == 200
        status, doc = h.request("PUT", "/slots/bbbb2222", body(MATTER_B))
        assert status == 503 and doc["error"] == "no_free_port"
        assert "bbbb2222" not in json.loads(h.state_text())["slots"]
    finally:
        h.close()


# ── identity conflicts ──────────────────────────────────────────────────────

def test_a_slot_id_stays_with_its_matter(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body(MATTER_A))
    status, doc = harness.request("PUT", "/slots/a1b2c3d4", body(MATTER_B))
    assert status == 409 and doc["error"] == "slot_matter_mismatch"
    status, doc = harness.request("PUT", "/slots/zzzz9999", body(MATTER_A))
    assert status == 409 and doc["error"] == "matter_has_slot"
    assert sup.parse_env_file(harness.env_path("a1b2c3d4").read_text())["LITCO_MATTER_ID"] == MATTER_A


# ── POST /slots/{id}/stop ───────────────────────────────────────────────────

def test_stop_drains_then_stops_then_deletes_the_env_file(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    env = harness.env_path("a1b2c3d4")
    seen = {}
    harness.runner.on_stop = lambda unit: seen.setdefault("env_during_stop", env.exists())

    status, doc = harness.request("POST", "/slots/a1b2c3d4/stop")
    assert status == 202 and doc["slot"]["state"] == "stopping"
    harness.supervisor.join_background(5)

    assert ["systemctl", "stop", "litco-agent@a1b2c3d4.service"] in harness.runner.calls
    assert seen == {"env_during_stop": True}, "the env file must outlive the drain and stop"
    assert not env.exists()
    status, doc = harness.request("GET", "/slots")
    assert doc["slots"][0]["state"] == "stopped"
    assert json.loads(harness.state_text())["slots"]["a1b2c3d4"]["desired"] == "stopped"
    # the home is kept
    assert not any(c[0] in ("userdel", "rm") for c in harness.runner.calls)


def test_put_while_stopping_is_409_and_stop_is_idempotent(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    gate = threading.Event()
    harness.runner.stop_gate = gate
    assert harness.request("POST", "/slots/a1b2c3d4/stop")[0] == 202
    try:
        status, doc = harness.request("GET", "/slots")
        assert doc["slots"][0]["state"] == "stopping"
        status, doc = harness.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 409 and doc["error"] == "slot_stopping"
        status, doc = harness.request("POST", "/slots/a1b2c3d4/stop")
        assert status == 202 and doc["slot"]["state"] == "stopping"
    finally:
        gate.set()
        harness.supervisor.join_background(5)
    assert len([c for c in harness.runner.calls if c[:2] == ["systemctl", "stop"]]) == 1
    assert harness.request("PUT", "/slots/a1b2c3d4", body())[1]["started"] is True


def test_stop_of_a_stopped_slot_is_200_and_clears_any_env_file(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    harness.runner.units["litco-agent@a1b2c3d4.service"] = "inactive"
    status, doc = harness.request("POST", "/slots/a1b2c3d4/stop")
    assert status == 200 and doc["slot"]["state"] == "stopped"
    assert not harness.env_path("a1b2c3d4").exists()
    assert ["systemctl", "stop", "litco-agent@a1b2c3d4.service"] not in harness.runner.calls


def test_stop_unknown_slot_is_404(harness):
    status, doc = harness.request("POST", "/slots/nope1234/stop")
    assert status == 404 and doc["error"] == "no_such_slot"


# ── failures ────────────────────────────────────────────────────────────────

def test_start_failure_is_500_and_leaves_no_env_file(tmp_path):
    h = Harness(tmp_path, runner=FakeRunner(fail={"systemctl start": 1}))
    try:
        status, doc = h.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 500 and doc["error"] == "start_failed"
        assert not h.env_path("a1b2c3d4").exists()
        assert json.loads(h.state_text())["slots"]["a1b2c3d4"]["desired"] == "stopped"
        assert_no_secret(json.dumps(doc) + "\n".join(h.logs))
    finally:
        h.close()


def test_useradd_failure_is_500_and_writes_no_env_file(tmp_path):
    h = Harness(tmp_path, runner=FakeRunner(fail={"useradd": 9}))
    try:
        status, doc = h.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 500 and doc["error"] == "useradd_failed"
        assert not h.env_path("a1b2c3d4").exists()
        assert "systemctl start" not in h.runner.names()
    finally:
        h.close()


# ── validation ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("slot_id", ["A1B2C3D4", "a" * 31, "ab-cd", "ab_cd", "..", "%2e%2e"])
def test_bad_slot_ids_are_refused(harness, slot_id):
    status, doc = harness.request("PUT", f"/slots/{slot_id}", body())
    assert status in (400, 404)
    assert harness.runner.calls == []


@pytest.mark.parametrize("change, field", [
    ({"hostSecret": "short"}, "hostSecret"),
    ({"hostSecret": None}, "hostSecret"),
    ({"agentToken": "not-a-matter-token-SECRET"}, "agentToken"),
    ({"matterId": "bad id!"}, "matterId"),
    ({"instanceUrl": "ftp://x"}, "instanceUrl"),
    ({"modelProvider": "Not A Slug"}, "modelProvider"),
    ({"modelProvider": "mystery"}, "provider"),
    ({"model": "a b"}, "model"),
    ({"approvalsMode": "yolo"}, "approvalsMode"),
    ({"channelEnv": {"AWS_SECRET_ACCESS_KEY": "x"}}, "channelEnv"),
    ({"hostSecret": HOST_SECRET + "\nLITCO_AGENT_TOKEN=lkm_evil"}, "hostSecret"),
    ({"extraEnv": {"LD_PRELOAD": "/tmp/x.so"}}, "extraEnv"),
])
def test_bad_bodies_are_400_and_never_echo_values(harness, change, field):
    status, doc = harness.request("PUT", "/slots/a1b2c3d4", body(**change))
    assert status == 400 and doc["error"] == "invalid_body"
    assert field in doc["message"]
    assert_no_secret(doc["message"])
    assert "lkm_evil" not in doc["message"]
    assert harness.runner.calls == [] or all(c[0] == "systemctl" for c in harness.runner.calls)
    assert not harness.env_path("a1b2c3d4").exists()


def test_non_json_and_oversized_bodies(harness):
    assert harness.request("PUT", "/slots/a1b2c3d4", raw=b"{not json")[1]["error"] == "invalid_json"
    status, doc = harness.request("PUT", "/slots/a1b2c3d4", raw=b" " * (sup.MAX_BODY + 1))
    assert status == 413


# ── auth and secrecy ────────────────────────────────────────────────────────

@pytest.mark.parametrize("method, path", [
    ("GET", "/slots"), ("PUT", "/slots/a1b2c3d4"), ("POST", "/slots/a1b2c3d4/stop"),
])
@pytest.mark.parametrize("token", [None, "", "wrong-secret", HOST_SECRET])
def test_every_slot_route_requires_the_supervisor_secret(harness, method, path, token):
    payload = body() if method == "PUT" else None
    status, doc = harness.request(method, path, payload, token=token)
    assert status == 401 and doc["error"] == "unauthorized"
    assert harness.runner.calls == []


def test_health_needs_no_secret_and_reports_none(harness):
    (harness.tmp_path / "REF").write_text("abc123\n")
    status, doc = harness.request("GET", "/health", token=None)
    assert status == 200 and doc["ok"] is True and doc["ref"] == "abc123"
    assert set(doc) == {"ok", "service", "ref", "uptimeSeconds"}


def test_unknown_route_and_method(harness):
    assert harness.request("GET", "/nope")[0] == 404
    assert harness.request("DELETE", "/slots/a1b2c3d4")[0] == 405
    assert harness.request("POST", "/slots")[0] == 405


def test_no_response_log_or_state_file_carries_a_secret(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    harness.request("PUT", "/slots/a1b2c3d4", body())
    harness.request("PUT", "/slots/b9c8d7e6", body(MATTER_B))
    harness.request("GET", "/slots")
    harness.request("GET", "/health", token=None)
    harness.request("POST", "/slots/a1b2c3d4/stop")
    harness.supervisor.join_background(5)
    harness.request("GET", "/slots")
    assert len(harness.responses) == 7
    for text in harness.responses:
        assert_no_secret(text)
    assert_no_secret(harness.state_text())
    assert_no_secret("\n".join(harness.logs))
    for argv in harness.runner.calls:
        assert_no_secret(" ".join(argv))


# ── restart and reboot ──────────────────────────────────────────────────────

def test_reboot_reconciles_running_slots_without_env_files(tmp_path):
    runner = FakeRunner()
    h = Harness(tmp_path, runner=runner)
    try:
        h.request("PUT", "/slots/a1b2c3d4", body())
        # A reboot: /run is empty and nothing is running.
        h.env_path("a1b2c3d4").unlink()
        runner.units.clear()
        h.restart()
        assert json.loads(h.state_text())["slots"]["a1b2c3d4"]["desired"] == "stopped"
        assert h.request("GET", "/slots")[1]["slots"][0]["state"] == "stopped"
        # The control plane's next PUT brings it back with the same port.
        status, doc = h.request("PUT", "/slots/a1b2c3d4", body())
        assert status == 200 and doc["started"] is True and doc["slot"]["port"] == 8800
    finally:
        h.close()


def test_restart_mid_stop_deletes_the_leftover_env_file(tmp_path):
    runner = FakeRunner()
    h = Harness(tmp_path, runner=runner)
    try:
        h.request("PUT", "/slots/a1b2c3d4", body())
        # The supervisor died after `systemctl stop` finished but before it deleted the file.
        runner.units["litco-agent@a1b2c3d4.service"] = "inactive"
        assert h.env_path("a1b2c3d4").exists()
        h.restart()
        assert not h.env_path("a1b2c3d4").exists()
    finally:
        h.close()


def test_get_slots_sweeps_env_files_of_slots_that_stopped_later(harness):
    harness.request("PUT", "/slots/a1b2c3d4", body())
    harness.supervisor._update("a1b2c3d4", desired="stopped")
    harness.runner.units["litco-agent@a1b2c3d4.service"] = "inactive"
    harness.request("GET", "/slots")
    assert not harness.env_path("a1b2c3d4").exists()


# ── configuration ───────────────────────────────────────────────────────────

def write_env(path, text, mode=0o600):
    path.write_text(text)
    os.chmod(path, mode)
    return path


def test_load_config_reads_the_secret_from_the_file_not_the_environment(tmp_path):
    env = write_env(tmp_path / "sup.env", f'# machine secret\nLITCO_SUPERVISOR_SECRET="{SUPERVISOR_SECRET}"\n')
    config = sup.load_config(env, environ={"LITCO_SUPERVISOR_SECRET": "x" * 40, "LITCO_SLOT_MAX": "8"})
    assert config.secret == SUPERVISOR_SECRET
    assert (config.port, config.interface, config.base_port, config.max_slots) == (8700, "tailscale0", 8800, 8)
    assert config.state_path == sup.Path("/var/lib/litco-supervisor/slots.json")
    assert config.run_dir == sup.Path("/run/litco-agent") and config.home_root == sup.Path("/srv/litco/m")


def test_load_config_allows_an_empty_interface_for_the_container_smoke(tmp_path):
    env = write_env(tmp_path / "sup.env",
                    f"LITCO_SUPERVISOR_SECRET={SUPERVISOR_SECRET}\nLITCO_SUPERVISOR_INTERFACE=\n")
    assert sup.load_config(env, environ={}).interface == ""


@pytest.mark.parametrize("text, mode, message", [
    (f"LITCO_SUPERVISOR_SECRET={SUPERVISOR_SECRET}\n", 0o640, "chmod 600"),
    ("LITCO_SUPERVISOR_SECRET=short\n", 0o600, "32 characters"),
    ("", 0o600, "LITCO_SUPERVISOR_SECRET"),
    (f"LITCO_SUPERVISOR_SECRET={SUPERVISOR_SECRET}\nLITCO_SUPERVISOR_PORT=8800\n", 0o600, "slot port range"),
    (f"LITCO_SUPERVISOR_SECRET={SUPERVISOR_SECRET}\nLITCO_SUPERVISOR_BIND=tailnet\n", 0o600, "IP address"),
    (f"LITCO_SUPERVISOR_SECRET={SUPERVISOR_SECRET}\nnot a line\n", 0o600, "KEY=VALUE"),
])
def test_load_config_refuses_bad_files(tmp_path, text, mode, message):
    env = write_env(tmp_path / "sup.env", text, mode)
    with pytest.raises(sup.ConfigError, match=message) as info:
        sup.load_config(env, environ={})
    assert SUPERVISOR_SECRET not in str(info.value)


def test_env_file_round_trips_quotes_and_backslashes():
    value = 'a"b\\c$d'
    assert sup.parse_env_file(sup.env_line("K", value)) == {"K": value}


def test_provider_key_map_matches_the_cloud_init_renderer():
    render = load_script("render_user_data", "render-user-data.py")
    assert sup.PROVIDER_KEY_ENV == render.PROVIDER_KEY_ENV
    assert sup.CHANNEL_ENV_KEYS == render.CHANNEL_ENV_KEYS


def test_real_runner_passes_no_environment_to_children(monkeypatch):
    monkeypatch.setenv("LITCO_SUPERVISOR_SECRET", SUPERVISOR_SECRET)
    result = sup.run_command(["env"])
    assert result.returncode == 0
    assert SUPERVISOR_SECRET not in result.stdout
    assert set(line.split("=", 1)[0] for line in result.stdout.splitlines()) <= {"PATH", "LANG", "LC_ALL"}
    assert sup.run_command(["/nonexistent/litco-command"]).returncode == 127


def test_bind_to_device_sets_so_bindtodevice():
    calls = []

    class FakeSocket:
        def setsockopt(self, level, option, value):
            calls.append((level, option, value))

    sup.bind_to_device(FakeSocket(), "tailscale0")
    assert calls == [(socket.SOL_SOCKET, getattr(socket, "SO_BINDTODEVICE", 25), b"tailscale0\0")]


def test_main_refuses_to_run_as_non_root(monkeypatch, capsys):
    monkeypatch.setattr(sup.os, "geteuid", lambda: 1000)
    assert sup.main(["--env-file", "/nonexistent"]) == 1
    assert "root" in capsys.readouterr().err


# ── the unit ────────────────────────────────────────────────────────────────

def parse_unit(text):
    sections, current = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        key, sep, value = line.partition("=")
        assert sep and current is not None, line
        current.setdefault(key.strip(), []).append(value.strip())
    return sections


def test_supervisor_unit_runs_the_daemon_as_root_with_its_env_file():
    unit = parse_unit((HOST / "litco-supervisor.service").read_text())
    svc = unit["Service"]
    assert svc["ExecStart"] == [
        "/usr/bin/python3 -I /usr/local/sbin/litco-supervisor --env-file /etc/litco-supervisor.env"]
    assert "User" not in svc, "the supervisor must run as root to create users"
    assert "EnvironmentFile" not in svc, "the daemon reads its secret file itself"
    assert svc["Restart"] == ["always"]
    assert svc["IPAddressDeny"] == ["169.254.169.254"]
    assert svc["UMask"] == ["0077"]
    assert unit["Unit"]["ConditionPathExists"] == ["/etc/litco-supervisor.env"]
    assert "tailscaled.service" in unit["Unit"]["After"][0]
    assert unit["Install"]["WantedBy"] == ["multi-user.target"]
