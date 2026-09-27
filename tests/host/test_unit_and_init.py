"""litco-agent.service, litco-agent-init and litco-agent-drain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_yaml import safe_load
from tests.host._load import HOST, REPO, load_script

init = load_script("litco_agent_init", "litco-agent-init")
UNIT = HOST / "litco-agent.service"

SECRETS = {
    "LITCO_HOST_SECRET": "hs-SECRET-VALUE-0123456789abcdef0123",
    "LITCO_AGENT_TOKEN": "lkm_SECRET_TOKEN",
    "ANTHROPIC_API_KEY": "sk-ant-SECRET",
    "LITCO_MODEL_API_KEY": "sk-custom-SECRET",
    "SLACK_BOT_TOKEN": "xoxb-SECRET",
}


def parse_unit(text: str) -> dict:
    sections: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        assert current is not None, f"directive outside a section: {line}"
        key, sep, value = line.partition("=")
        assert sep, f"not a directive: {line}"
        current.setdefault(key.strip(), []).append(value.strip())
    return sections


# ── the unit ────────────────────────────────────────────────────────────────

def test_unit_parses_and_carries_the_required_directives():
    unit = parse_unit(UNIT.read_text())
    assert set(unit) == {"Unit", "Service", "Install"}
    svc = unit["Service"]
    assert svc["User"] == ["hermes"] and svc["Group"] == ["hermes"]
    assert svc["EnvironmentFile"] == ["/etc/litco-agent/env"]
    assert svc["Restart"] == ["always"]
    assert svc["KillMode"] == ["mixed"]
    assert svc["TimeoutStopSec"] == ["infinity"]
    assert svc["ExecStartPre"] == ["/usr/local/bin/litco-agent-init"]
    assert svc["ExecStart"] == ["/opt/litco-agent/app/.venv/bin/python -m hermes_cli.main gateway run"]
    assert svc["ExecStop"][0] == "/usr/local/bin/litco-agent-drain", "drain must run before anything else on stop"
    assert "HERMES_HOME=/home/hermes/.hermes" in svc["Environment"]
    assert unit["Unit"]["ConditionPathExists"] == ["/etc/litco-agent/env"]
    assert unit["Install"]["WantedBy"] == ["multi-user.target"]
    # no secret is baked into the unit
    assert not any(k in UNIT.read_text() for k in SECRETS)


def test_unit_paths_match_what_install_host_installs():
    install = (HOST / "install-host.sh").read_text()
    assert "/usr/local/bin/litco-agent-init" in install
    assert "/usr/local/bin/litco-agent-drain" in install
    assert "/etc/systemd/system/litco-agent.service" in install
    assert 'HERMES_UID=10000' in install and "user@10000.service" in UNIT.read_text()


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze not available on this host")
def test_systemd_analyze_verify(tmp_path):
    copy = tmp_path / "litco-agent.service"
    copy.write_text(UNIT.read_text())
    out = subprocess.run(["systemd-analyze", "verify", str(copy)], capture_output=True, text=True)
    # Missing binaries/users on a dev box are expected; syntax errors are not.
    assert "Unknown key" not in out.stderr and "Invalid" not in out.stderr, out.stderr


@pytest.mark.parametrize("script", ["build-image.sh", "install-host.sh", "smoke.sh", "litco-agent-drain"])
def test_shell_scripts_parse(script):
    assert subprocess.run(["bash", "-n", str(HOST / script)]).returncode == 0


# ── litco-agent-init ────────────────────────────────────────────────────────

class RecordingEnv(dict):
    """A mapping that records every key read from it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def env_for(tmp_path, **extra) -> RecordingEnv:
    env = RecordingEnv(SECRETS)
    env.update({
        "LITCO_MATTER_ID": "matter-42",
        "LITCO_INSTANCE_URL": "https://acme.litco.ai",
        "LITCO_MATTER_HOME": str(tmp_path / "matter"),
        "LITCO_MODEL": "anthropic/claude-opus-4.6",
        "LITCO_MODEL_PROVIDER": "anthropic",
        "LITCO_TURN_HOST": "0.0.0.0",
        "LITCO_TURN_PORT": "8765",
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "LITCO_APP_DIR": str(REPO),
    })
    env.update(extra)
    return env


def rendered(tmp_path) -> tuple[dict, str]:
    home = tmp_path / "hermes-home"
    return safe_load((home / "config.yaml").read_text()), (home / "SOUL.md").read_text()


def test_init_renders_config_and_soul(tmp_path):
    assert init.main(env_for(tmp_path)) == 0
    config, soul = rendered(tmp_path)
    assert config["model"] == {"default": "anthropic/claude-opus-4.6", "provider": "anthropic"}
    tools = config["platform_toolsets"]["litco_turn"]
    assert tools == ["terminal", "browser", "file", "code_execution", "delegation", "cronjob", "memory",
                     "skills", "vision", "web"]
    assert config["platform_toolsets"]["slack"] == tools
    assert config["gateway"]["platforms"]["litco_turn"] == {
        "enabled": True, "host": "0.0.0.0", "port": 8765, "matter_id": "matter-42"}
    assert config["terminal"]["cwd"] == f"{tmp_path / 'matter'}/shared"
    assert "approvals" not in config, "no approvals override unless the owner sets LITCO_APPROVALS_MODE"
    assert "key_env" not in config["model"]
    assert "matter-42" in soul and "https://acme.litco.ai" in soul and "{{" not in soul
    assert (tmp_path / "matter").is_dir()


def test_init_disables_tool_search_and_points_skills_at_the_checkout(tmp_path):
    app_dir = tmp_path / "opt" / "litco-agent" / "app"
    (app_dir / "hermes_cli").mkdir(parents=True)
    shutil.copy(REPO / "hermes_cli" / "config_defaults.py", app_dir / "hermes_cli" / "config_defaults.py")
    env = env_for(tmp_path, LITCO_APP_DIR=str(app_dir), LITCO_PROFILE_DIR=str(HOST / "profile"))
    assert init.main(env) == 0
    config = rendered(tmp_path)[0]
    assert config["tools"]["tool_search"]["enabled"] == "off"  # the string, not YAML 1.1 false
    assert config["skills"]["external_dirs"] == [str(app_dir / "litco" / "skills")]
    # resolved from the non-secret env alone
    assert env.read <= set(init.NON_SECRET_KEYS) and not env.read & set(SECRETS)
    for path in (tmp_path / "hermes-home").iterdir():
        for value in SECRETS.values():
            assert value not in path.read_text()


def test_skills_dir_defaults_to_the_image_checkout():
    values = init.read_settings({"LITCO_MATTER_ID": "m1"})
    assert values["LITCO_APP_DIR"] == "/opt/litco-agent/app"
    assert init.substitutions(dict(values, LITCO_APP_DIR=str(REPO)))["LITCO_SKILLS_DIR"] == \
        str(REPO / "litco" / "skills")


def test_init_config_version_matches_this_checkout(tmp_path):
    import re
    expected = re.search(r'"_config_version":\s*(\d+)', (REPO / "hermes_cli/config_defaults.py").read_text())
    assert init.main(env_for(tmp_path)) == 0
    assert rendered(tmp_path)[0]["_config_version"] == int(expected.group(1))


def test_init_never_reads_or_writes_secrets(tmp_path):
    env = env_for(tmp_path, LITCO_MODEL_PROVIDER="custom", LITCO_MODEL_BASE_URL="https://proxy.litco.ai/v1",
                  LITCO_MODEL_KEY_ENV="LITCO_MODEL_API_KEY")
    assert init.main(env) == 0
    assert env.read <= set(init.NON_SECRET_KEYS), f"read non-allowlisted keys: {env.read - set(init.NON_SECRET_KEYS)}"
    assert not env.read & set(SECRETS)
    home = tmp_path / "hermes-home"
    assert sorted(p.name for p in home.iterdir()) == ["SOUL.md", "config.yaml"]
    assert not (home / ".env").exists()
    for path in home.iterdir():
        text = path.read_text()
        for value in SECRETS.values():
            assert value not in text, f"secret leaked into {path.name}"
    config = rendered(tmp_path)[0]
    # custom provider: the NAME of the key variable, never the value
    assert config["model"]["key_env"] == "LITCO_MODEL_API_KEY"
    assert config["model"]["base_url"] == "https://proxy.litco.ai/v1"


def test_init_source_does_not_touch_secret_names_or_env_files():
    source = (HOST / "litco-agent-init").read_text()
    for name in ("LITCO_HOST_SECRET", "LITCO_AGENT_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
                 ".env\"", "os.environ["):
        assert name not in source, name


@pytest.mark.parametrize("mode", ["manual", "smart", "off"])
def test_approvals_knob_passes_through_the_owner_choice(tmp_path, mode):
    assert init.main(env_for(tmp_path, LITCO_APPROVALS_MODE=mode)) == 0
    assert rendered(tmp_path)[0]["approvals"] == {"mode": mode}


@pytest.mark.parametrize("extra", [
    {"LITCO_APPROVALS_MODE": "yolo"},
    {"LITCO_MATTER_ID": ""},
    {"LITCO_MATTER_ID": "a/b"},
    {"LITCO_TURN_PORT": "http"},
    {"LITCO_MODEL": 'x"\nprovider: evil'},
    {"LITCO_MODEL_KEY_ENV": "lower; rm"},
    {"LITCO_APP_DIR": '/opt/x"\nevil: 1'},
])
def test_init_refuses_bad_settings_with_ex_config(tmp_path, extra, capsys):
    assert init.main(env_for(tmp_path, **extra)) == 78
    assert not (tmp_path / "hermes-home" / "config.yaml").exists()
    assert "litco-agent-init:" in capsys.readouterr().err


def test_template_with_unknown_placeholder_is_refused():
    with pytest.raises(init.InitError, match="LITCO_HOST_SECRET"):
        init.render("key: {{LITCO_HOST_SECRET}}", {"LITCO_MATTER_ID": "m"})


def test_rendered_config_is_accepted_by_hermes_toolset_resolution(tmp_path):
    assert init.main(env_for(tmp_path)) == 0
    config = rendered(tmp_path)[0]
    from hermes_cli.tools_config import _get_platform_tools
    enabled = _get_platform_tools(config, "litco_turn")
    for toolset in ("terminal", "browser", "file", "code_execution", "delegation", "cronjob", "memory",
                    "skills", "vision", "web"):
        assert toolset in enabled, toolset


# ── litco-agent-drain ───────────────────────────────────────────────────────

class _Health(BaseHTTPRequestHandler):
    counts: list = []

    def do_GET(self):
        active = self.counts.pop(0) if self.counts else 0
        body = json.dumps({"ok": True, "activeTurns": active}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _drain(port: int, **extra):
    env = dict(os.environ, LITCO_TURN_HOST="127.0.0.1", LITCO_TURN_PORT=str(port),
               LITCO_DRAIN_INTERVAL_SECONDS="0.05", **extra)
    return subprocess.run(["bash", str(HOST / "litco-agent-drain")], capture_output=True, text=True, env=env,
                          timeout=30)


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not available")
def test_drain_waits_until_no_turn_is_running():
    _Health.counts = [2, 1, 1, 0]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = _drain(server.server_address[1])
    finally:
        server.shutdown()
    assert out.returncode == 0
    assert "waiting for 2 running turn(s)" in out.stdout
    assert "no running turns; stopping" in out.stdout
    assert _Health.counts == []


def test_drain_exits_at_once_when_server_is_down():
    start = time.monotonic()
    out = _drain(1)  # nothing listens on port 1
    assert out.returncode == 0 and "nothing to drain" in out.stdout
    assert time.monotonic() - start < 10


@pytest.mark.skipif(shutil.which("curl") is None, reason="curl not available")
def test_drain_max_seconds_cap():
    _Health.counts = [1] * 10_000
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = _drain(server.server_address[1], LITCO_DRAIN_MAX_SECONDS="1")
    finally:
        server.shutdown()
    assert out.returncode == 0 and "stopping anyway" in out.stdout
