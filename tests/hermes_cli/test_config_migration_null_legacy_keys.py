"""A legacy key the migration ladder retires is removed even when it is written as ``null``.

Saves keep an explicit ``null`` (it can mean something a missing key does not), so a step that
decides by the value instead of the key's presence leaves a null legacy key on disk for good,
and ``hermes doctor`` / startup validation warn about it on every run.
"""

import os
from unittest.mock import patch

import pytest
import hermes_yaml as yaml


@pytest.mark.parametrize(
    ("current_ver", "config"),
    [
        (15, {"display": {"tool_progress_overrides": None}}),
        (16, {"compression": {"summary_model": None, "summary_provider": None, "summary_base_url": None}}),
        (44, {"base_url": None}),  # root-level model key, retired by the load/save normalizer
    ],
    ids=["v16-tool-progress-overrides", "v17-compression-summary", "root-base-url"],
)
def test_migration_retires_a_legacy_key_written_as_null(tmp_path, current_ver, config):
    from hermes_cli.config import DEFAULT_CONFIG, migrate_config, validate_config_structure
    from hermes_cli.doctor_config import collect_deprecated_config_keys

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"_config_version": current_ver, **config}), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        issues = [issue.message for issue in validate_config_structure(raw)]

    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert collect_deprecated_config_keys(raw) == []
    assert "base_url" not in raw
    assert "model" not in raw  # retiring a null-only root key must not leave an empty model section
    assert not [message for message in issues if "base_url" in message]


@pytest.mark.parametrize("section, legacy, migrated_path, migrated_value", [
    ("display", {"tool_progress_overrides": None}, (), None),
    ("compression", {"summary_model": None, "summary_provider": None, "summary_base_url": None}, (), None),
    ("display", {"tool_progress_overrides": {"telegram": "all"}},
     ("display", "platforms", "telegram", "tool_progress"), "all"),
    ("compression", {"summary_model": "fast-model"},
     ("auxiliary", "compression", "model"), "fast-model"),
])
def test_current_config_retires_legacy_keys(tmp_path, section, legacy, migrated_path, migrated_value):
    from hermes_cli.config import DEFAULT_CONFIG, migrate_config
    from hermes_cli.doctor_config import collect_deprecated_config_keys

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"_config_version": DEFAULT_CONFIG["_config_version"],
                                           section: legacy}), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert collect_deprecated_config_keys(raw) == []
    for part in migrated_path:
        raw = raw[part]
    if migrated_path:
        assert raw == migrated_value


@pytest.mark.parametrize("overrides, expected", [
    ({}, {}),
    ({"telegram": "all", "discord": "compact"},
     {"telegram": {"tool_progress": "all"}, "discord": {"tool_progress": "compact"}}),
])
def test_display_migration_moves_overrides_and_retires_legacy_key(tmp_path, overrides, expected):
    from hermes_cli.config import migrate_config
    from hermes_cli.doctor_config import collect_deprecated_config_keys

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"_config_version": 15,
                                           "display": {"tool_progress_overrides": overrides}}),
                           encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        migrate_config(interactive=False, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert collect_deprecated_config_keys(raw) == []
    assert raw.get("display", {}).get("platforms", {}) == expected


@pytest.mark.parametrize("config, resolved", [
    ({"display": {"platforms": {"telegram": {"tool_progress": None}},
                  "tool_progress_overrides": {"telegram": "all"}}}, "all"),
    ({"display": {"platforms": {"telegram": {"tool_progress": "new"}},
                  "tool_progress_overrides": {"telegram": "all"}}}, "new"),
    ({"compression": {"summary_provider": "openrouter"},
      "auxiliary": {"compression": {"provider": "auto"}}}, "auto"),
], ids=["null-platform", "explicit-platform", "explicit-auto-provider"])
def test_replay_preserves_resolved_conflicts(tmp_path, config, resolved):
    from gateway.display_config import resolve_tool_progress
    from hermes_cli.config import DEFAULT_CONFIG, migrate_config
    from hermes_cli.config_effective import load_user_config_effective

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"_config_version": DEFAULT_CONFIG["_config_version"],
                                           **config}), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        before = load_user_config_effective()
        migrate_config(interactive=False, quiet=True)
        after = load_user_config_effective()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if "display" in config:
        read = lambda cfg: resolve_tool_progress(cfg, "telegram")[0]
        assert "tool_progress_overrides" not in raw["display"]
    else:
        read = lambda cfg: cfg["auxiliary"]["compression"]["provider"]
        assert "summary_provider" not in raw["compression"]
    assert read(before) == read(after) == resolved


@pytest.mark.parametrize("legacy_provider", [None, "openrouter"])
def test_regular_load_and_save_retires_current_legacy_keys(tmp_path, legacy_provider):
    from hermes_cli.config import DEFAULT_CONFIG, load_config, read_raw_config, save_config
    from hermes_cli.config_effective import load_user_config_effective

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "_config_version": DEFAULT_CONFIG["_config_version"],
        "display": {"tool_progress_overrides": {"telegram": "all"}},
        "compression": {"summary_provider": legacy_provider},
    }), encoding="utf-8")
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        before = load_user_config_effective()
        loaded = load_config()
        assert loaded["display"]["platforms"]["telegram"]["tool_progress"] == "all"
        if legacy_provider is not None:
            assert loaded["auxiliary"]["compression"]["provider"] == legacy_provider
        assert "tool_progress_overrides" in read_raw_config()["display"]
        save_config(loaded)
        after = load_user_config_effective()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert before["display"]["platforms"] == after["display"]["platforms"]
    if legacy_provider is not None:
        assert before["auxiliary"]["compression"]["provider"] == after["auxiliary"]["compression"]["provider"]
    assert "tool_progress_overrides" not in raw["display"]
    assert "summary_provider" not in raw.get("compression", {})
