"""litco-agent-init and litco-agent-drain as a machine-host slot runs them (FIRM_AGENT_HOST 3.2, 3.6)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_yaml import safe_load
from tests.deploy._supervisor_fake import sup
from tests.host._load import HOST, REPO, load_script

init = load_script("litco_agent_init_slot", "litco-agent-init")

SECRETS = {"LITCO_HOST_SECRET": "hs-SECRET-VALUE-0123456789abcdef0123", "LITCO_AGENT_TOKEN": "lkm_SECRET"}


def slot_env(slot_root: Path, **extra) -> dict:
    """What the template unit and the supervisor's env file give a slot, secrets included."""
    home = slot_root / "ab12cd34"
    env = dict(SECRETS)
    env.update({
        "LITCO_SLOT_ID": "ab12cd34",
        "LITCO_SLOT_PORT": "8803",
        "LITCO_MATTER_ID": "matter-42",
        "LITCO_INSTANCE_URL": "https://acme.litco.ai",
        "HERMES_HOME": str(home / ".hermes"),
        "LITCO_MATTER_HOME": str(home / "matter"),
        "LITCO_APP_DIR": str(REPO),
    })
    env.update(extra)
    return env


# ── litco-agent-init ────────────────────────────────────────────────────────

def test_slot_port_becomes_the_turn_port_and_the_profile_lands_in_the_slot_home(tmp_path):
    assert init.main(slot_env(tmp_path), slot_root=tmp_path) == 0
    home = tmp_path / "ab12cd34"
    config = safe_load((home / ".hermes" / "config.yaml").read_text())
    assert config["gateway"]["platforms"]["litco_turn"] == {
        "enabled": True, "host": "0.0.0.0", "port": 8803, "matter_id": "matter-42"}
    assert (home / "matter").is_dir()
    for path in (home / ".hermes").iterdir():
        for value in SECRETS.values():
            assert value not in path.read_text()


def test_slot_defaults_its_homes_inside_the_slot(tmp_path):
    env = slot_env(tmp_path)
    del env["HERMES_HOME"], env["LITCO_MATTER_HOME"]
    values = init.read_settings(env, slot_root=tmp_path)
    assert values["HERMES_HOME"] == str(tmp_path / "ab12cd34" / ".hermes")
    assert values["LITCO_MATTER_HOME"] == str(tmp_path / "ab12cd34" / "matter")


def test_profile_names_the_resolved_release_not_the_current_symlink(tmp_path):
    releases = tmp_path / "opt" / "releases"
    release = releases / "host-1"
    (release / "hermes_cli").mkdir(parents=True)
    shutil.copy(REPO / "hermes_cli" / "config_defaults.py", release / "hermes_cli" / "config_defaults.py")
    current = tmp_path / "opt" / "current"
    current.symlink_to(release)
    env = slot_env(tmp_path, LITCO_APP_DIR=str(current), LITCO_PROFILE_DIR=str(HOST / "profile"))
    assert init.main(env, slot_root=tmp_path) == 0
    config = safe_load((tmp_path / "ab12cd34" / ".hermes" / "config.yaml").read_text())
    assert config["skills"]["external_dirs"] == [str(release.resolve() / "litco" / "skills")]


@pytest.mark.parametrize("change", [
    {"LITCO_SLOT_ID": "AB/../x"},
    {"LITCO_SLOT_ID": "a" * 31},
    {"LITCO_SLOT_PORT": ""},
    {"LITCO_TURN_PORT": "8765"},                       # the platform would bind this one, not the slot's
    {"HERMES_HOME": "/srv/litco/m/other/.hermes"},     # outside this slot's home
    {"LITCO_MATTER_HOME": "{root}/ab12cd34/../zz99/matter"},
    {"LITCO_MATTER_HOME": "{root}/ab12cd34"},          # the home itself, not a directory in it
])
def test_slot_settings_that_would_cross_slots_are_refused(tmp_path, change, capsys):
    change = {k: v.replace("{root}", str(tmp_path)) for k, v in change.items()}
    assert init.main(slot_env(tmp_path, **change), slot_root=tmp_path) == 78
    assert "litco-agent-init:" in capsys.readouterr().err
    assert not list(tmp_path.rglob("config.yaml"))


def test_slot_id_length_matches_the_supervisor_and_the_unit(tmp_path):
    """One rule everywhere: 1-30 lowercase letters and digits (m_<id> fits a 32-character user name)."""
    longest = "a" * 30
    env = slot_env(tmp_path, LITCO_SLOT_ID=longest, HERMES_HOME=str(tmp_path / longest / ".hermes"),
                   LITCO_MATTER_HOME=str(tmp_path / longest / "matter"))
    assert init.read_settings(env, slot_root=tmp_path)["LITCO_TURN_PORT"] == "8803"
    assert init._SLOT_ID.pattern == sup.SLOT_ID.pattern
    with pytest.raises(init.InitError, match="LITCO_SLOT_ID must be 1-30 lowercase letters and digits"):
        init.read_settings(dict(env, LITCO_SLOT_ID="a" * 31), slot_root=tmp_path)


def test_slot_mode_reads_no_secret(tmp_path):
    class Recording(dict):
        read: set = set()

        def get(self, key, default=None):
            self.read.add(key)
            return super().get(key, default)

    env = Recording(slot_env(tmp_path))
    assert init.main(env, slot_root=tmp_path) == 0
    assert env.read <= set(init.NON_SECRET_KEYS)
    assert not env.read & set(SECRETS)


# ── litco-agent-drain ───────────────────────────────────────────────────────

class _TurnServer(BaseHTTPRequestHandler):
    """/drain and /health as the turn server answers them; records what the drain sent."""

    drain_status = 202
    counts: list = []
    drain_secrets: list = []
    log: list = []

    def do_POST(self):
        if self.path != "/drain":
            self.send_error(404)
            return
        self.log.append("POST /drain")
        type(self).drain_secrets.append(self.headers.get("X-Host-Secret"))
        self.send_response(self.drain_status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        self.log.append("GET /health")
        active = self.counts.pop(0) if self.counts else 0
        body = json.dumps({"ok": True, "draining": True, "activeTurns": active}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def turn_server():
    _TurnServer.drain_status, _TurnServer.counts = 202, []
    _TurnServer.drain_secrets, _TurnServer.log = [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TurnServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def _drain(**env):
    full = {k: v for k, v in os.environ.items() if not k.startswith("LITCO_")}
    full.update(LITCO_TURN_HOST="0.0.0.0", LITCO_DRAIN_INTERVAL_SECONDS="0.05", **env)
    return subprocess.run(["bash", str(HOST / "litco-agent-drain")], capture_output=True, text=True, env=full,
                          timeout=30)


needs_curl = pytest.mark.skipif(shutil.which("curl") is None, reason="curl not available")


@needs_curl
def test_drain_flips_the_slot_to_draining_with_its_host_secret_then_waits(turn_server):
    _TurnServer.counts = [2, 1, 0]
    # The slot's port wins over a per-matter LITCO_TURN_PORT that points elsewhere.
    out = _drain(LITCO_SLOT_PORT=str(turn_server), LITCO_TURN_PORT="1", **SECRETS)
    assert out.returncode == 0, out.stderr
    assert _TurnServer.log[0] == "POST /drain"
    assert _TurnServer.drain_secrets == [SECRETS["LITCO_HOST_SECRET"]]
    assert "new turns get 503" in out.stdout and "no running turns; stopping" in out.stdout
    assert _TurnServer.counts == []


@needs_curl
@pytest.mark.parametrize("status", [404, 405])
def test_drain_against_a_turn_server_without_drain_still_waits_for_running_turns(turn_server, status):
    _TurnServer.drain_status, _TurnServer.counts = status, [1, 0]
    out = _drain(LITCO_SLOT_PORT=str(turn_server), **SECRETS)
    assert out.returncode == 0
    assert "has no /drain" in out.stdout and "no running turns; stopping" in out.stdout
    assert _TurnServer.counts == []


@needs_curl
def test_drain_with_nothing_listening_exits_at_once():
    out = _drain(LITCO_SLOT_PORT="1", **SECRETS)
    assert out.returncode == 0 and "nothing to drain" in out.stdout
