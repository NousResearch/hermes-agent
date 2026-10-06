"""config.yaml ``sessions.*`` write-patience keys reach the env carriers hermes_state reads.

``hermes_state.SessionDB`` resolves its write budgets from ``HERMES_WRITE_PATIENCE_S`` /
``HERMES_TRANSCRIPT_WRITE_PATIENCE_S`` (see ``tests/hermes_state/test_write_patience_config.py``).
config.yaml is the canonical surface; the gateway and the CLI mirror it into those carriers. These
tests pin both mirror halves, so a key that stops being bridged fails here rather than silently
reverting to the compiled default on a host that configured a longer wait.
"""

from __future__ import annotations

import os
from pathlib import Path

import hermes_yaml as yaml

import gateway.run as gateway_run
from hermes_cli.cli_config_load import _mirror_config_to_env


def _write_home(tmp_path: Path, sessions_cfg: dict) -> Path:
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"sessions": sessions_cfg}), encoding="utf-8"
    )
    (hermes_home / ".env").write_text("", encoding="utf-8")
    return hermes_home


def test_gateway_bridge_carries_write_patience(tmp_path, monkeypatch):
    home = _write_home(tmp_path, {"write_patience_s": 45, "transcript_write_patience_s": 300})
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setenv("HERMES_WRITE_PATIENCE_S", "1")
    monkeypatch.setenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", "1")
    gateway_run._reload_runtime_env_preserving_config_authority()
    assert os.environ["HERMES_WRITE_PATIENCE_S"] == "45"
    assert os.environ["HERMES_TRANSCRIPT_WRITE_PATIENCE_S"] == "300"


def test_cli_mirror_carries_write_patience(monkeypatch):
    monkeypatch.delenv("HERMES_WRITE_PATIENCE_S", raising=False)
    monkeypatch.delenv("HERMES_TRANSCRIPT_WRITE_PATIENCE_S", raising=False)
    _mirror_config_to_env(
        {"terminal": {}, "sessions": {"write_patience_s": 45, "transcript_write_patience_s": 300}},
        False,
    )
    assert os.environ["HERMES_WRITE_PATIENCE_S"] == "45"
    assert os.environ["HERMES_TRANSCRIPT_WRITE_PATIENCE_S"] == "300"
