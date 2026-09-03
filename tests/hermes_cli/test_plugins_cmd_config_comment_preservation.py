"""Regression test for #92554: `hermes plugins enable/disable` destroyed every
user comment in config.yaml (and reinjected the default boilerplate) because
_save_enabled_set/_save_disabled_set re-serialized the whole document through
save_config() instead of writing just the changed key.
"""

import os
from unittest.mock import patch

import pytest

CONFIG_WITH_COMMENTS = """\
# TOP COMMENT — must survive
model:
  provider: test
plugins:
  # rationale for the enabled list
  enabled: []
"""


class TestPluginEnableDisablePreserveComments:
    def test_save_enabled_set_preserves_user_comments(self, tmp_path):
        from hermes_cli.plugins_cmd import _save_enabled_set

        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_WITH_COMMENTS, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _save_enabled_set({"demo"})

        written = config_path.read_text(encoding="utf-8")
        assert "# TOP COMMENT — must survive" in written
        assert "# rationale for the enabled list" in written
        assert "demo" in written
        # No boilerplate reinjection from the full-document writer.
        assert "Fallback Model" not in written

    def test_save_disabled_set_preserves_user_comments(self, tmp_path):
        from hermes_cli.plugins_cmd import _save_disabled_set

        config_path = tmp_path / "config.yaml"
        config_path.write_text(CONFIG_WITH_COMMENTS, encoding="utf-8")

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _save_disabled_set({"demo"})

        written = config_path.read_text(encoding="utf-8")
        assert "# TOP COMMENT — must survive" in written
        assert "# rationale for the enabled list" in written
        assert "demo" in written

    def test_save_enabled_set_creates_config_on_fresh_install(self, tmp_path):
        """No pre-existing config.yaml: the round-trip writer must create one
        instead of erroring, matching save_config()'s old create-on-write
        behavior for a brand-new HERMES_HOME.
        """
        from hermes_cli.plugins_cmd import _save_enabled_set

        config_path = tmp_path / "config.yaml"
        assert not config_path.exists()

        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            _save_enabled_set({"demo"})

        written = config_path.read_text(encoding="utf-8")
        assert "demo" in written


class TestPluginEnableDisableRespectsManagedScope:
    """The round-trip writer must still honor per-key managed-scope pins.

    save_config() used to strip any leaf pinned by managed_scope before
    writing (config.py:_strip_dotted_keys) so a bulk write could never
    persist a value the managed layer would override on next load. Routing
    through atomic_roundtrip_yaml_update dropped that check entirely,
    silently writing admin-pinned plugin lists to the user's own
    config.yaml instead of rejecting the write.
    """

    @pytest.fixture
    def managed_homes(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        managed = tmp_path / "managed"
        managed.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
        import hermes_cli.config as cfg
        from hermes_cli import managed_scope

        cfg._LOAD_CONFIG_CACHE.clear()
        cfg._RAW_CONFIG_CACHE.clear()
        managed_scope.invalidate_managed_cache()
        (managed / "config.yaml").write_text(
            "plugins:\n  enabled:\n    - admin_required_plugin\n"
            "  disabled:\n    - admin_blocked_plugin\n",
            encoding="utf-8",
        )
        managed_scope.invalidate_managed_cache()
        (home / "config.yaml").write_text(CONFIG_WITH_COMMENTS, encoding="utf-8")
        return home, managed

    def test_save_enabled_set_rejects_managed_key(self, managed_homes, capsys):
        from hermes_cli.plugins_cmd import _save_enabled_set

        home, _managed = managed_homes
        _save_enabled_set({"user_chosen_plugin"})

        assert "managed" in capsys.readouterr().err.lower()
        written = (home / "config.yaml").read_text(encoding="utf-8")
        assert "user_chosen_plugin" not in written

    def test_save_disabled_set_rejects_managed_key(self, managed_homes, capsys):
        from hermes_cli.plugins_cmd import _save_disabled_set

        home, _managed = managed_homes
        _save_disabled_set({"user_chosen_plugin"})

        assert "managed" in capsys.readouterr().err.lower()
        written = (home / "config.yaml").read_text(encoding="utf-8")
        assert "user_chosen_plugin" not in written
