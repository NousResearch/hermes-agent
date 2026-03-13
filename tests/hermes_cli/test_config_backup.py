"""Tests for hermes_cli.config_backup module."""

import time
import argparse
from pathlib import Path

import yaml

from hermes_cli.config import load_config, save_config
from hermes_cli.config_backup import (
    create_backup,
    prune_backups,
    list_backups,
    restore_routing_sections,
    restore_full,
    format_backup_summary,
    get_backup_dir,
)


def _write_config(tmp_path, data):
    """Helper to write a config.yaml in the fake HERMES_HOME."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data, f)
    return config_path


def test_create_backup_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = _write_config(tmp_path, {"model": "test"})

    result = create_backup(config_path, reason="routing")

    assert result is not None
    assert result.exists()
    assert "routing" in result.name
    assert result.parent == get_backup_dir()


def test_create_backup_no_config_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    missing = tmp_path / "config.yaml"

    result = create_backup(missing, reason="test")

    assert result is None


def test_create_backup_duplicate_second(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = _write_config(tmp_path, {"model": "test"})

    b1 = create_backup(config_path, reason="dup")
    b2 = create_backup(config_path, reason="dup")

    assert b1 != b2
    assert b1.exists()
    assert b2.exists()


def test_prune_backups_keeps_max(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    backup_dir = get_backup_dir()
    backup_dir.mkdir(parents=True)

    # Create 15 backups with different mtimes
    for i in range(15):
        p = backup_dir / f"config_20260313_{100000 + i}.yaml"
        p.write_text(f"backup {i}")
        # Ensure distinct mtimes
        p.touch()

    deleted = prune_backups(max_count=10)

    assert len(deleted) == 5
    remaining = list(backup_dir.glob("config_*.yaml"))
    assert len(remaining) == 10


def test_list_backups_order(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = _write_config(tmp_path, {"model": "test"})

    b1 = create_backup(config_path, reason="first")
    time.sleep(1.1)  # ensure distinct mtime (filesystem granularity)
    b2 = create_backup(config_path, reason="second")

    backups = list_backups()
    assert len(backups) == 2
    # Newest first
    assert "second" in backups[0][0].name
    assert "first" in backups[1][0].name


def test_list_backups_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert list_backups() == []


def test_restore_routing_sections_selective(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Write initial config with profiles and other settings
    initial = {
        "model": {"default": "deepseek-chat", "provider": "auto"},
        "model_profiles": {
            "chat": {"model": "gpt-5.4", "provider": "openai-codex", "base_url": "", "api_key_env": "", "api_key": ""},
            "coding": {"model": "deepseek-chat", "provider": "deepseek", "base_url": "", "api_key_env": "", "api_key": ""},
        },
        "model_routing": {"rules": [{"if_toolsets_any": ["terminal"], "profile": "coding"}]},
        "timezone": "Europe/Helsinki",
        "display": {"compact": True},
    }
    config_path = _write_config(tmp_path, initial)

    # Create a backup of this good state
    backup = create_backup(config_path, reason="good_state")

    # Now mess up the config
    messed = dict(initial)
    messed["model_profiles"] = {}
    messed["model_routing"] = {"rules": []}
    messed["timezone"] = "US/Pacific"  # unrelated change
    _write_config(tmp_path, messed)

    # Restore routing sections only
    result = restore_routing_sections(backup)

    # Routing sections restored
    assert result["model_profiles"]["chat"]["model"] == "gpt-5.4"
    assert len(result["model_routing"]["rules"]) == 1
    # Unrelated changes preserved (not reverted)
    assert result["timezone"] == "US/Pacific"


def test_restore_full(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    original = {
        "model": {"default": "deepseek-chat"},
        "timezone": "Europe/Helsinki",
        "model_profiles": {"chat": {"model": "gpt-5.4", "provider": "", "base_url": "", "api_key_env": "", "api_key": ""}},
    }
    config_path = _write_config(tmp_path, original)
    backup = create_backup(config_path, reason="original")

    # Overwrite
    _write_config(tmp_path, {"model": {"default": "something-else"}, "timezone": "US/Pacific"})

    # Full restore
    result = restore_full(backup)

    assert result["timezone"] == "Europe/Helsinki"
    assert result["model"]["default"] == "deepseek-chat"


def test_restore_creates_pre_restore_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    config_path = _write_config(tmp_path, {"model": "test"})
    backup = create_backup(config_path, reason="original")
    _write_config(tmp_path, {"model": "changed"})

    restore_routing_sections(backup)

    backups = list_backups()
    pre_restore = [b for b in backups if "pre_restore" in b[0].name]
    assert len(pre_restore) >= 1


def test_format_backup_summary(tmp_path):
    path = tmp_path / "config_test.yaml"
    data = {
        "model": {"default": "deepseek-chat"},
        "model_profiles": {
            "chat": {"model": "gpt-5.4", "provider": "openai-codex"},
            "coding": {"model": "", "provider": ""},
            "research": {"model": "gemini-3-flash", "provider": "google"},
        },
        "model_routing": {"rules": [{"profile": "coding"}]},
    }
    with open(path, "w") as f:
        yaml.dump(data, f)

    summary = format_backup_summary(path)
    assert "chat" in summary
    assert "research" in summary
    assert "1 rule" in summary
    assert "deepseek-chat" in summary


def test_format_backup_summary_empty(tmp_path):
    path = tmp_path / "config_empty.yaml"
    path.write_text("{}")
    summary = format_backup_summary(path)
    assert summary  # should not crash


def test_save_config_with_backup_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, {"model": "original"})

    config = load_config()
    config["model"] = "changed"
    save_config(config, backup_reason="test_save")

    backups = list_backups()
    assert len(backups) == 1
    assert "test_save" in backups[0][0].name


def test_save_config_without_backup_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(tmp_path, {"model": "original"})

    config = load_config()
    save_config(config)

    backups = list_backups()
    assert len(backups) == 0


def test_cmd_routing_reset_creates_backup(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.main import cmd_configure_model_routing

    # Set up initial config with profiles
    initial = {
        "model_profiles": {"chat": {"model": "gpt-5.4", "provider": "openai-codex", "base_url": "", "api_key_env": "", "api_key": ""}},
    }
    _write_config(tmp_path, initial)

    cmd_configure_model_routing(argparse.Namespace(
        reset=True, restore=False, restore_full=False, list_backups=False,
    ))

    backups = list_backups()
    assert len(backups) >= 1
    assert "reset" in backups[0][0].name
    out = capsys.readouterr().out.lower()
    assert "reset" in out
    assert "backup" in out


def test_cmd_routing_list_backups_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.main import cmd_configure_model_routing

    cmd_configure_model_routing(argparse.Namespace(
        reset=False, restore=False, restore_full=False, list_backups=True,
    ))

    out = capsys.readouterr().out
    assert "No config backups found" in out
