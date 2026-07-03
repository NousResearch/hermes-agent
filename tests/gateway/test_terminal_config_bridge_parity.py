"""Pure terminal config/env bridge parity tests.

These tests cover the config-to-TERMINAL_* bridge without launching a gateway,
agent, or terminal backend.  They use a temp HERMES_HOME/config.yaml and execute
only the gateway bootstrap bridge block, stopping before adapter imports/startup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "cli.py"
GATEWAY_RUN_PATH = REPO_ROOT / "gateway" / "run.py"

TERMINAL_ENV_KEYS = (
    "TERMINAL_ENV",
    "TERMINAL_CWD",
    "TERMINAL_TIMEOUT",
    "TERMINAL_HOME_MODE",
    "MESSAGING_CWD",
    "HERMES_QUIET",
    "HERMES_EXEC_ASK",
)


def _clear_terminal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in TERMINAL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _write_config(home: Path, terminal: dict[str, object] | None = None) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    config = {} if terminal is None else {"terminal": terminal}
    path = home / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run_cli_bridge(home: Path, monkeypatch: pytest.MonkeyPatch, cwd: Path):
    """Call cli.load_cli_config() against a temp Hermes home."""
    _clear_terminal_env(monkeypatch)
    monkeypatch.chdir(cwd)
    import cli

    monkeypatch.setattr(cli, "_hermes_home", home)
    return cli.load_cli_config()


def _gateway_bridge_source() -> str:
    """Extract only gateway/run.py's bootstrap config bridge block.

    Importing gateway.run would start real gateway initialization.  Executing the
    bounded source slice exercises the actual bridge code while stopping before
    gateway config/platform imports.
    """
    source = GATEWAY_RUN_PATH.read_text(encoding="utf-8")
    start_marker = "# Bridge config.yaml values into the environment so os.getenv() picks them up."
    end_marker = "from gateway.config import ("
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _run_gateway_bridge(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ns = {
        "__file__": str(GATEWAY_RUN_PATH),
        "_hermes_home": home,
        "os": os,
        "Path": Path,
        "sys": sys,
        "json": __import__("json"),
    }
    exec(compile(_gateway_bridge_source(), str(GATEWAY_RUN_PATH), "exec"), ns)


def test_cli_terminal_config_overrides_stale_terminal_env(tmp_path, monkeypatch):
    """Explicit terminal config is authoritative over inherited TERMINAL_* env."""
    home = tmp_path / "home"
    cwd = tmp_path / "launch-cwd"
    cwd.mkdir()
    terminal_cwd = tmp_path / "configured-cwd"
    terminal_cwd.mkdir()
    _write_config(
        home,
        {
            "backend": "docker",
            "cwd": str(terminal_cwd),
            "timeout": 77,
            "home_mode": "copy",
        },
    )

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/stale/cwd")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "13")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "auto")

    import cli

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(cli, "_hermes_home", home)
    config = cli.load_cli_config()

    assert config["terminal"]["env_type"] == "docker"
    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_CWD"] == str(terminal_cwd)
    assert os.environ["TERMINAL_TIMEOUT"] == "77"
    assert os.environ["TERMINAL_HOME_MODE"] == "copy"


def test_cli_terminal_env_backfills_when_terminal_section_missing(tmp_path, monkeypatch):
    """Without a terminal section, existing env survives except CLI cwd resolution."""
    home = tmp_path / "home"
    cwd = tmp_path / "launch-cwd"
    cwd.mkdir()
    _write_config(home, terminal=None)

    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_CWD", "/preexisting/cwd")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "55")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "persistent")

    import cli

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(cli, "_hermes_home", home)
    config = cli.load_cli_config()

    assert config["terminal"]["env_type"] == "local"
    assert os.environ["TERMINAL_ENV"] == "ssh"
    assert os.environ["TERMINAL_TIMEOUT"] == "55"
    assert os.environ["TERMINAL_HOME_MODE"] == "persistent"
    # CLI launch cwd is intentionally authoritative for local CLI sessions.
    assert os.environ["TERMINAL_CWD"] == str(cwd)


def test_gateway_terminal_config_overrides_stale_terminal_env(tmp_path, monkeypatch):
    """Gateway bridge maps documented terminal keys and overrides stale env."""
    home = tmp_path / "home"
    configured_cwd = tmp_path / "gateway-cwd"
    configured_cwd.mkdir()
    _write_config(
        home,
        {
            "backend": "docker",
            "cwd": str(configured_cwd),
            "timeout": 88,
            "home_mode": "copy",
        },
    )

    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", "/stale/gateway-cwd")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "14")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "auto")

    _run_gateway_bridge(home, monkeypatch)

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_CWD"] == str(configured_cwd)
    assert os.environ["TERMINAL_TIMEOUT"] == "88"
    assert os.environ["TERMINAL_HOME_MODE"] == "copy"


def test_gateway_terminal_cwd_placeholder_falls_back_without_clobbering_backend(
    tmp_path, monkeypatch
):
    """Gateway skips cwd placeholders, then falls back to MESSAGING_CWD/home."""
    home = tmp_path / "home"
    messaging_cwd = tmp_path / "messaging-cwd"
    messaging_cwd.mkdir()
    _write_config(
        home,
        {
            "backend": "docker",
            "cwd": ".",
            "timeout": 99,
            "home_mode": "copy",
        },
    )

    monkeypatch.setenv("MESSAGING_CWD", str(messaging_cwd))

    _run_gateway_bridge(home, monkeypatch)

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_CWD"] == str(messaging_cwd)
    assert os.environ["TERMINAL_TIMEOUT"] == "99"
    assert os.environ["TERMINAL_HOME_MODE"] == "copy"


def test_gateway_terminal_env_type_alias_is_not_documented_backend(tmp_path, monkeypatch):
    """Current gateway contract bridges documented terminal.backend only.

    CLI accepts legacy terminal.env_type, but the gateway bridge intentionally
    reads terminal.backend.  Locking this distinction keeps parity tests honest
    without asserting a source change in this tests-only packet.
    """
    home = tmp_path / "home"
    _write_config(home, {"env_type": "docker", "timeout": 12, "home_mode": "copy"})

    monkeypatch.setenv("TERMINAL_ENV", "local")

    _run_gateway_bridge(home, monkeypatch)

    assert os.environ["TERMINAL_ENV"] == "local"
    assert os.environ["TERMINAL_TIMEOUT"] == "12"
    assert os.environ["TERMINAL_HOME_MODE"] == "copy"
