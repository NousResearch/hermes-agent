"""Tests for default-drift detection (#91501).

User-set config values whose upstream default changed should surface an
informational notice — without ever rewriting the user's value.
"""

import pytest

from hermes_cli.config_migrations import DEFAULT_HISTORY, collect_default_drift


@pytest.mark.parametrize("quiet", [True, False])
def test_real_migration_persists_version_and_retains_user_value(tmp_path, monkeypatch, capsys, quiet):
    from hermes_cli import config
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "_config_version: 32\ncompression:\n  hygiene_hard_message_limit: 400\n"
    )

    result = config.migrate_config(interactive=False, quiet=quiet)

    saved = config.read_raw_config()
    assert saved["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert saved["compression"]["hygiene_hard_message_limit"] == 400
    assert any("compression.hygiene_hard_message_limit" in warning for warning in result["warnings"])
    assert ("Config default changed since your last version" in capsys.readouterr().out) is (not quiet)



class TestCollectDefaultDrift:
    def test_user_set_changed_key_is_flagged(self):
        cfg = {
            "delegation": {"child_timeout_seconds": 600},
            "_config_version": 30,
        }
        drifts = collect_default_drift(cfg, previous_version=30)
        assert len(drifts) == 1
        assert drifts[0]["key"] == "delegation.child_timeout_seconds"
        assert "600" in drifts[0]["value"]
        assert "removed" in drifts[0]["change"]

    def test_unchanged_intent_not_flagged(self):
        # User value equals the CURRENT default → nothing to tell them.
        cfg = {"compression": {"hygiene_hard_message_limit": 5000}}
        drifts = collect_default_drift(cfg, previous_version=32)
        assert drifts == []

    def test_key_absent_means_tracking_default(self):
        cfg = {"delegation": {}, "_config_version": 30}
        assert collect_default_drift(cfg, previous_version=30) == []

    def test_no_changes_in_window(self):
        # Change landed at v31; upgrading from 31 → nothing new to report.
        cfg = {"delegation": {"child_timeout_seconds": 600}}
        assert collect_default_drift(cfg, previous_version=31) == []

    def test_empty_or_invalid_config(self):
        assert collect_default_drift(None, previous_version=1) == []
        assert collect_default_drift("not-a-dict", previous_version=1) == []

    def test_registry_entries_resolve_to_real_defaults(self):
        """Every registry key must exist in DEFAULT_CONFIG with a non-missing default."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        for key in DEFAULT_HISTORY:
            node = DEFAULT_CONFIG
            for part in key.split("."):
                assert isinstance(node, dict) and part in node, f"{key} missing from DEFAULT_CONFIG"
                node = node[part]

    def test_hygiene_limit_change_reported_with_old_and_new(self):
        cfg = {"compression": {"hygiene_hard_message_limit": 400}}
        drifts = collect_default_drift(cfg, previous_version=32)
        assert len(drifts) == 1
        assert "400" in drifts[0]["change"]  # old default
        assert "5000" in drifts[0]["change"]  # current default
