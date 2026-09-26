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
