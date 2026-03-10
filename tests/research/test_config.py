"""Tests for research.config."""

import yaml

from research.config import (
    DEFAULT_RESEARCH_CONFIG,
    get_research_config,
    get_research_mode,
    get_workspace_dir,
    research_enabled,
)


def test_default_config_has_required_keys():
    for key in ("enabled", "mode", "workspace_dir", "max_total_cost_usd", "tinker"):
        assert key in DEFAULT_RESEARCH_CONFIG


def test_get_research_config_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    cfg = get_research_config()
    assert cfg["enabled"] is True
    assert cfg["mode"] == "approval"
    assert cfg["max_total_cost_usd"] == 50


def test_get_research_config_file_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"research": {"mode": "auto", "max_total_cost_usd": 100}}),
        encoding="utf-8",
    )
    cfg = get_research_config()
    assert cfg["mode"] == "auto"
    assert cfg["max_total_cost_usd"] == 100
    # Defaults preserved for unset keys
    assert cfg["workspace_dir"] == ".hermes-research"


def test_env_var_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"research": {"mode": "auto"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_RESEARCH_MODE", "approval")
    cfg = get_research_config()
    assert cfg["mode"] == "approval"


def test_research_enabled_env_false(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESEARCH_ENABLED", "false")
    assert research_enabled() is False


def test_research_enabled_env_true(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RESEARCH_ENABLED", "true")
    assert research_enabled() is True


def test_get_research_mode_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert get_research_mode() == "approval"


def test_get_research_mode_override():
    assert get_research_mode("auto") == "auto"
    assert get_research_mode("approval") == "approval"
    assert get_research_mode("invalid") == "approval"


def test_get_workspace_dir_relative(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ws = get_workspace_dir(tmp_path)
    assert ws == tmp_path / ".hermes-research"


def test_get_workspace_dir_absolute(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    abs_path = tmp_path / "custom-workspace"
    config_path.write_text(
        yaml.safe_dump({"research": {"workspace_dir": str(abs_path)}}),
        encoding="utf-8",
    )
    ws = get_workspace_dir(tmp_path)
    assert ws == abs_path
