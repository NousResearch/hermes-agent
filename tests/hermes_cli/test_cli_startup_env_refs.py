"""Startup discovery must not resolve unrelated config references before secrets load."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from hermes_cli import __version__


@pytest.mark.parametrize("argv", (["gateway", "restart", "--help"], ["--version"]))
@pytest.mark.parametrize("named_profile", (False, True), ids=("default", "named"))
@pytest.mark.parametrize("source", ("dotenv", pytest.param("command", marks=pytest.mark.linux_only)))
def test_startup_leaves_config_refs_for_runtime(tmp_path, argv, named_profile, source):
    home = tmp_path / ".hermes"
    if named_profile:
        home = home / "profiles" / "work"
    home.mkdir(parents=True)
    config = {"mcp_servers": {"foobar": {
        "command": "example", "env": {"FOOBAR_API_KEY": "${env:FOOBAR_API_KEY}"},
    }}}
    if source == "dotenv":
        (home / ".env").write_text("FOOBAR_API_KEY=test-key\n", encoding="utf-8")
    else:
        config["secrets"] = {"command": {
            "enabled": True, "command": "printf 'FOOBAR_API_KEY=test-key\\n'",
        }}
    if argv == ["--version"]:
        config.setdefault("secrets", {})["bitwarden"] = {"enabled": True}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
    (home / ".update_check").write_text(
        json.dumps({"ts": time.time(), "behind": 0, "rev": None, "ver": __version__}),
        encoding="utf-8",
    )
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(repo))
    env.pop("FOOBAR_API_KEY", None)
    code = """
import runpy
import sys
sys.argv[0] = 'hermes'
try:
    runpy.run_module('hermes_cli.main', run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
if '--version' in sys.argv:
    assert 'agent.secret_sources.bitwarden' not in sys.modules
    assert 'hermes_cli.env_loader' not in sys.modules
    assert 'dotenv' not in sys.modules
else:
    from hermes_cli.config import load_config
    assert load_config()['mcp_servers']['foobar']['env']['FOOBAR_API_KEY'] == 'test-key'
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *argv], cwd=repo, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Config ref" not in result.stderr


@pytest.mark.parametrize("managed_plugins, expected_enabled, expected_disabled", (
    ({}, {"allowed"}, {"blocked"}),
    ({"enabled": ["blocked"]}, {"blocked"}, {"blocked"}),
    ({"disabled": ["${env:PLUGIN_NAME}"]}, {"allowed"}, {"allowed"}),
    ({"enabled": []}, set(), {"blocked"}),
    ({"enabled": None, "disabled": None}, None, set()),
))
@pytest.mark.parametrize("broken_config", (
    pytest.param("plugins: [\n", id="syntax-error"),
    pytest.param("- accidental-list-root\n", id="list-root"),
    pytest.param("accidental-scalar-root\n", id="scalar-root"),
))
def test_plugin_gates_resolve_only_their_own_refs(
    tmp_path, monkeypatch, caplog, managed_plugins, expected_enabled, expected_disabled,
    broken_config,
):
    from hermes_cli.config import load_config
    from hermes_cli.plugins_discovery import _get_disabled_plugins, _get_enabled_plugins

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PLUGIN_NAME", "allowed")
    monkeypatch.delenv("MISSING_PLUGIN_SECRET", raising=False)
    (home / "config.yaml").write_text(json.dumps({"plugins": {
        "enabled": ["${env:PLUGIN_NAME}"], "disabled": ["blocked"],
        "example": {"api_key": "${env:MISSING_PLUGIN_SECRET}"},
    }}), encoding="utf-8")
    (managed / "config.yaml").write_text(
        json.dumps({"plugins": managed_plugins, "model": {"api_key": "${env:MISSING_PLUGIN_SECRET}"}}),
        encoding="utf-8",
    )

    assert _get_enabled_plugins() == expected_enabled
    assert _get_disabled_plugins() == expected_disabled
    assert "Config ref" not in caplog.text
    assert load_config()["plugins"]["example"]["api_key"] == "${env:MISSING_PLUGIN_SECRET}"
    assert "MISSING_PLUGIN_SECRET is not set" in caplog.text

    # A broken edit must retain the policy cached by the runtime load above,
    # including user denials of plugins on the managed allow-list.
    (home / "config.yaml").write_text(broken_config, encoding="utf-8")
    assert (_get_enabled_plugins(), _get_disabled_plugins()) == (
        expected_enabled, expected_disabled,
    )
