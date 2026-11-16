"""Bridge ownership markers never carry a bridged value, and never reach an agent's child process.

``TELEGRAM_PROXY`` (a proxy URL can embed credentials) and the allowlists are bridged from YAML and
stripped from every agent subprocess. Their ``HERMES_BRIDGED_*`` markers used to hold the same
plaintext, and the ``HERMES_`` prefix passed the scrub, so a terminal command could read them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from gateway.platforms._shared import bridged_env_digest, bridged_env_marker

_PROXY = "http://user:s3cretpw@proxy.example:3128"
_NAMES = ("TELEGRAM_PROXY", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_REQUIRE_MENTION", "DISCORD_ALLOWED_ROLES",
          "DISCORD_ALLOW_BOTS")


@pytest.fixture(autouse=True)
def clean_bridge(monkeypatch):
    from hermes_cli import env_loader

    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    monkeypatch.setattr(env_loader, "_MANAGED_DOTENV_KEYS", set())
    for name in _NAMES:
        for key in (name, bridged_env_marker(name)):
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key)


def _write(home, text):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_no_bridged_value_reaches_an_agent_subprocess(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config
    from tools.environments.local import _sanitize_subprocess_env, hermes_subprocess_env

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write(home, f"telegram:\n  allow_from: ['111']\n  proxy_url: '{_PROXY}'\n")

    load_gateway_config()

    assert os.environ["TELEGRAM_PROXY"] == _PROXY  # the gateway itself still gets the value
    for child in (hermes_subprocess_env(), _sanitize_subprocess_env(dict(os.environ))):
        assert not [k for k, v in child.items() if "s3cretpw" in v or k.startswith("HERMES_BRIDGED_")]
        assert "TELEGRAM_ALLOWED_USERS" not in child


def test_bridge_still_refreshes_a_changed_yaml_value(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write(home, f"telegram:\n  allow_from: ['111', '222']\n  proxy_url: '{_PROXY}'\n")
    load_gateway_config()
    assert os.environ[bridged_env_marker("TELEGRAM_PROXY")] == bridged_env_digest(_PROXY)

    _write(home, "telegram:\n  allow_from: ['111']\n  proxy_url: 'http://proxy.example:8080'\n")
    load_gateway_config()

    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111"
    assert os.environ["TELEGRAM_PROXY"] == "http://proxy.example:8080"


def test_only_a_same_profile_restart_child_keeps_the_markers(tmp_path, monkeypatch):
    from gateway.config import load_gateway_config
    from tools.environments.local import served_profile_child_env

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write(home, "telegram:\n  reactions: false\n")
    load_gateway_config()

    restart = served_profile_child_env(base=dict(os.environ), target_home=home, inherit_credentials=True)
    other = served_profile_child_env(base=dict(os.environ), target_home=tmp_path / "other",
                                     inherit_credentials=True)

    assert restart[bridged_env_marker("TELEGRAM_REACTIONS")] == bridged_env_digest("false")
    assert not [k for k in other if k.startswith("HERMES_BRIDGED_")]


def test_a_load_for_another_home_never_rewrites_the_launch_profiles_values(tmp_path, monkeypatch):
    """No secret scope bound: the HERMES_HOME override alone must keep B's YAML off A's env."""
    from gateway.config import load_gateway_config
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    launch, other = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(launch))
    _write(launch, "telegram:\n  allow_from: ['111']\n  require_mention: true\n")
    _write(other, "telegram:\n  allow_from: ['999']\n")
    load_gateway_config()

    token = set_hermes_home_override(other)
    try:
        load_gateway_config()
    finally:
        reset_hermes_home_override(token)

    assert os.environ["TELEGRAM_ALLOWED_USERS"] == "111"
    assert os.environ["TELEGRAM_REQUIRE_MENTION"] == "true"


_RESTARTED_GATEWAY = """
import json, os, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[2])
from gateway.config_loader import load_yaml_layer
load_yaml_layer(Path(sys.argv[1]), {})
print(json.dumps({k: os.environ.get(k) for k in json.loads(sys.argv[3])}))
"""


@pytest.mark.parametrize("shell", ["hermes_subprocess_env", "terminal"])
def test_a_gateway_restarted_from_an_agent_shell_applies_the_tightened_yaml(tmp_path, monkeypatch, shell):
    """The agent's shell runs ``hermes gateway restart``: the bridged allowlists it inherited are the
    bridge's, so the new gateway must re-derive them from config.yaml, not keep them as operator-set."""
    from gateway.config import load_gateway_config
    from tools.environments.local import (
        _sanitize_subprocess_env, hermes_subprocess_env, served_profile_child_env)

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _write(home, "discord:\n  allowed_roles: ['55', '66']\n  allow_bots: all\ntelegram:\n  allow_from: ['1', '2']\n")
    load_gateway_config()
    assert os.environ["DISCORD_ALLOWED_ROLES"] == "55,66"

    agent_shell = (hermes_subprocess_env(inherit_credentials=True) if shell == "hermes_subprocess_env"
                   else _sanitize_subprocess_env(dict(os.environ)))
    _write(home, "discord:\n  allowed_roles: ['66']\n  allow_bots: none\ntelegram:\n  allow_from: ['2']\n")
    restart = served_profile_child_env(base=agent_shell, target_home=home, inherit_credentials=True)
    names = ["DISCORD_ALLOWED_ROLES", "DISCORD_ALLOW_BOTS", "TELEGRAM_ALLOWED_USERS"]
    child = subprocess.run([sys.executable, "-c", _RESTARTED_GATEWAY, str(home), json.dumps(sys.path),
                            json.dumps(names)], env=restart, capture_output=True, text=True, timeout=120,
                           check=False)

    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == {
        "DISCORD_ALLOWED_ROLES": "66", "DISCORD_ALLOW_BOTS": "none", "TELEGRAM_ALLOWED_USERS": "2"}


def test_an_operator_set_value_survives_the_marker_scrub(monkeypatch):
    from tools.environments.local import hermes_subprocess_env

    monkeypatch.setenv("DISCORD_ALLOWED_ROLES", "77")
    monkeypatch.setenv(bridged_env_marker("DISCORD_ALLOWED_ROLES"), bridged_env_digest("55"))

    child = hermes_subprocess_env(inherit_credentials=True)

    assert child["DISCORD_ALLOWED_ROLES"] == "77"
    assert bridged_env_marker("DISCORD_ALLOWED_ROLES") not in child
