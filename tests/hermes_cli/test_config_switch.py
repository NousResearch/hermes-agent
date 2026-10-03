"""``hermes_cli.config.config_switch`` and the distribution switches it reads.

Each switch test carries a POSITIVE CONTROL: the same call under the default config (no
``config.yaml``) takes today's path, so "refused" is never the subject simply not reaching the
switch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(config: dict) -> None:
    """Write ``config`` as this HERMES_HOME's config.yaml (YAML is a JSON superset)."""
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


def _switch_off(section: str, key: str) -> None:
    _write_config({section: {key: False}})


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [(None, True, True), (None, False, False), (False, True, False), ("off", True, False),
     ("yes", False, True), (True, False, True), ("garbage", True, False)],
)
def test_config_switch_reads_a_boolean_with_default(value, default, expected):
    from hermes_cli.config import config_switch

    _write_config({"feature": {"flag": value}})
    assert config_switch("feature", "flag", default=default) is expected


def test_config_switch_absent_key_and_unreadable_config_are_the_default(monkeypatch):
    from hermes_cli import config

    assert config.config_switch("feature", "missing") is True
    assert config.config_switch("feature", "missing", default=False) is False

    def _boom():
        raise OSError("unreadable")

    monkeypatch.setattr(config, "load_config_readonly", _boom)
    assert config.config_switch("feature", "flag") is True
    assert config.config_switch("feature", "flag", default=False) is False


def test_stdio_mcp_servers_off_keeps_http_servers():
    from tools.mcp_tool_common import mcp_server_enabled

    stdio, http = {"command": "npx", "args": ["-y", "server"]}, {"url": "https://mcp.example/mcp"}
    assert mcp_server_enabled(stdio) and mcp_server_enabled(http)  # positive control

    _switch_off("mcp", "stdio_servers")
    assert not mcp_server_enabled(stdio)
    assert mcp_server_enabled(http)


def test_mcp_client_off_disables_every_server():
    from tools.mcp_tool_common import mcp_client_enabled, mcp_server_enabled

    http = {"url": "https://mcp.example/mcp"}
    assert mcp_client_enabled() and mcp_server_enabled(http)  # positive control

    _switch_off("mcp", "client")
    assert not mcp_client_enabled()
    assert not mcp_server_enabled(http)


def test_external_execution_backends_are_refused_before_their_builder(monkeypatch):
    from tools import terminal_tool_backends as backends

    built: list[str] = []
    for name in ("docker", "local"):
        monkeypatch.setitem(backends._ENV_BUILDERS, name,
                            lambda _n=name, **_kw: built.append(_n) or types.SimpleNamespace())

    backends._create_environment("docker", "image", "/work", 10)  # positive control
    assert built == ["docker"]

    _switch_off("terminal", "external_backends")
    built.clear()
    with pytest.raises(ValueError, match="terminal.external_backends"):
        backends._create_environment("docker", "image", "/work", 10)
    assert built == []
    backends._create_environment("local", "", "/work", 10)  # local stays
    assert built == ["local"]


def test_gateway_starts_no_platform_adapter_when_switched_off():
    from gateway.config import Platform
    from gateway.run_adapters import GatewayAdapterLifecycleMixin

    class _Runner(GatewayAdapterLifecycleMixin):
        def __init__(self):
            self.made = []

        def _instantiate_adapter(self, platform, config):
            self.made.append(platform)
            return types.SimpleNamespace()

    runner = _Runner()
    assert runner._create_adapter(Platform.TELEGRAM, object()) is not None  # positive control
    assert runner.made == [Platform.TELEGRAM]

    _switch_off("gateway", "platform_adapters")
    runner.made.clear()
    assert runner._create_adapter(Platform.TELEGRAM, object()) is None
    assert runner.made == []


def test_git_probe_off_spawns_no_git(monkeypatch, tmp_path):
    from tui_gateway import git_probe

    spawned: list = []
    monkeypatch.setattr(git_probe, "bounded_git_probe", lambda argv, timeout: spawned.append(argv) or "main")

    assert git_probe.run_git(str(tmp_path), "branch", "--show-current") == "main"  # positive control
    assert spawned

    _switch_off("sessions", "git_probe")
    spawned.clear()
    assert git_probe.run_git(str(tmp_path), "branch", "--show-current") == ""
    assert spawned == []


def _dispatch(request: dict) -> dict:
    from tui_gateway import server
    from tui_gateway.transport import bind_transport, reset_transport

    token = bind_transport(None)
    try:
        return server.handle_request(request)
    finally:
        reset_transport(token)


def test_voice_mode_and_wake_word_are_not_offered_when_switched_off(monkeypatch):
    from tui_gateway import server

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setitem(sys.modules, "tools.voice_mode", types.SimpleNamespace(
        check_voice_requirements=lambda: {"available": True, "details": ""}))
    monkeypatch.setenv("HERMES_VOICE", "0")
    toggle_on = {"id": "v", "method": "voice.toggle", "params": {"action": "on"}}
    wake = {"id": "w", "method": "wake.start", "params": {"surface": "gui"}}

    assert "result" in _dispatch(toggle_on)  # positive control
    assert os.environ["HERMES_VOICE"] == "1"
    assert (_dispatch(wake).get("result") or {}).get("reason") != "voice_mode_unavailable"

    _switch_off("voice", "mode_enabled")
    monkeypatch.setenv("HERMES_VOICE", "0")
    refused = _dispatch(toggle_on)
    assert refused["error"]["code"] == 4015 and "voice.mode_enabled" in refused["error"]["message"]
    assert os.environ["HERMES_VOICE"] == "0"
    assert _dispatch(wake)["result"] == {"started": False, "reason": "voice_mode_unavailable"}


def test_config_schema_loads_without_wake_word():
    """A distribution that does not package ``tools.wake_word`` still loads the config schema."""

    def providers(block_wake_word: bool) -> list:
        code = ("import json, sys\n"
                + ("sys.modules['tools.wake_word'] = None\n" if block_wake_word else "")
                + "import hermes_cli.web_server_config as m\n"
                  "print(json.dumps(m.CONFIG_SCHEMA['wake_word.provider']['options']))\n")
        done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=120, check=True)
        return json.loads(done.stdout.strip().splitlines()[-1])

    assert len(providers(block_wake_word=False)) > 1  # positive control: the engines are offered
    assert providers(block_wake_word=True) == ["auto"]
