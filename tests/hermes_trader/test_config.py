"""Tests for hermes_trader.config YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_trader.config import TraderConfig, load_trader_config


def test_defaults_when_no_config_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_TRADER_CONFIG", raising=False)
    cfg = load_trader_config()
    assert cfg.mode == "paper"
    assert cfg.primary_chain == "base"
    assert cfg.allowed_chains == ["base", "ethereum", "arbitrum"]
    assert cfg.mcp_server_name == "defi-trading"


def test_load_from_yaml_file(tmp_path):
    path = tmp_path / "hermes_trader.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mode": "paper",
                "primary_chain": "base",
                "allowed_chains": ["base", "arbitrum"],
                "max_position_pct": 2.5,
                "mcp_server_name": "defi-trading",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_trader_config(path)
    assert cfg.mode == "paper"
    assert cfg.primary_chain == "base"
    assert cfg.allowed_chains == ["base", "arbitrum"]
    assert cfg.max_position_pct == 2.5


def test_invalid_mode_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("mode: sandbox\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid trader mode"):
        load_trader_config(path)


def test_env_override_path(tmp_path, monkeypatch):
    path = tmp_path / "from_env.yaml"
    path.write_text("mode: paper\nprimary_chain: ethereum\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_TRADER_CONFIG", str(path))
    cfg = load_trader_config()
    assert cfg.primary_chain == "ethereum"


def test_example_config_parses():
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "config" / "hermes_trader.example.yaml"
    if not example.is_file():
        pytest.skip("config/hermes_trader.example.yaml not present")
    cfg = load_trader_config(example)
    assert cfg.mode == "paper"
    assert cfg.primary_chain == "base"
    assert TraderConfig.from_mapping(yaml.safe_load(example.read_text(encoding="utf-8")))