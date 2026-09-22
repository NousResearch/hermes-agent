"""Config-set must not turn model routing identifiers into scalars (#117345)."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import yaml

from hermes_cli.config import config_command, read_raw_config, set_config_value


@pytest.mark.parametrize("key", [
    "model.provider", "model.default", "model.name", "model.model",
    "model.base_url", "model.api_base", "model.api_mode", "model",
])
@pytest.mark.parametrize("value", ["2", "007", "0", "1.5", "off", "null", "", "[local]"])
def test_model_route_strings_survive_config_set(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "original", "context_length": 4096}}),
        encoding="utf-8",
    )
    config_command(argparse.Namespace(config_command="set", key=key, value=value, force=False))
    stored = read_raw_config()["model"]
    leaf = "default" if key == "model" else key.split(".")[1]
    if leaf == "api_base":
        leaf = "base_url"
        if not value:
            # The existing alias normalizer drops empty aliases; do not change it.
            assert leaf not in stored
            return
    assert stored[leaf] == value
    assert isinstance(stored[leaf], str)
    assert stored["context_length"] == 4096


@pytest.mark.parametrize("key", ["model.api_key", "model.api"])
@pytest.mark.parametrize("value, expected_key", [
    ("0007", "0007"), ("123456", "123456"), ("false", "false"),
    ("007", "no-key-required"), ("off", "no-key-required"),
])
def test_model_credential_strings_reach_runtime(tmp_path, key, value, expected_key):
    # Isolate before imports: auth/config modules can cache paths and credentials.
    env = {name: os.environ[name] for name in ("PATH", "SYSTEMROOT", "WINDIR")
           if name in os.environ}
    for name in ("HOME", "USERPROFILE", "HERMES_HOME", "APPDATA", "LOCALAPPDATA",
                 "TEMP", "TMP"):
        env[name] = str(tmp_path)
    env["PYTHONUTF8"] = "1"
    env["HERMES_TEST_ISOLATION"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import argparse
            import os
            from pathlib import Path
            import sys

            network_attempts = []
            def deny_network(event, args):
                if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                    network_attempts.append(event)
                    raise AssertionError("No network allowed in config regression")
            sys.addaudithook(deny_network)

            import yaml
            from hermes_cli.config import config_command
            from hermes_cli.runtime_provider import resolve_runtime_provider

            key, value, expected_key = sys.argv[1:]
            config_path = Path(os.environ["HERMES_HOME"]) / "config.yaml"
            config_path.write_text(yaml.safe_dump({"model": {
                "provider": "custom", "default": "synthetic-model",
                "base_url": "http://127.0.0.1:1/v1",
            }}), encoding="utf-8")
            config_command(argparse.Namespace(
                config_command="set", key=key, value=value, force=False))
            stored = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            leaf = key.split(".")[1]
            assert stored["model"][leaf] == value, stored["model"][leaf]
            assert isinstance(stored["model"][leaf], str)
            runtime = resolve_runtime_provider(requested="custom")
            assert runtime["provider"] == "custom", runtime
            assert runtime["base_url"] == "http://127.0.0.1:1/v1", runtime
            # Short secrets still fail the existing minimum-length requirement.
            assert runtime["api_key"] == expected_key, runtime
            assert not network_attempts, network_attempts
        """), key, value, expected_key],
        cwd=Path(__file__).resolve().parents[2], env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_non_string_settings_keep_coercion_and_container_refusal(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key, value in [
        ("model.context_length", "8192"), ("agent.max_turns", "7"),
        ("compression.threshold", "0.5"), ("checkpoints.enabled", "false"),
    ]:
        set_config_value(key, value)
    stored = read_raw_config()
    assert stored["model"]["context_length"] == 8192
    assert stored["agent"]["max_turns"] == 7
    assert stored["compression"]["threshold"] == 0.5
    assert stored["checkpoints"]["enabled"] is False
    before = (tmp_path / "config.yaml").read_bytes()
    with pytest.raises(SystemExit):
        set_config_value("model.aliases", "not-a-mapping")
    assert (tmp_path / "config.yaml").read_bytes() == before
