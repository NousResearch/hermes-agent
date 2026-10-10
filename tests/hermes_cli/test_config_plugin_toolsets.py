"""Config migration recognizes enabled plugin toolsets discovered on demand."""

import hermes_yaml as yaml

from hermes_cli.config import DEFAULT_CONFIG, migrate_config


class TestPlatformToolsetMigrationValidation:
    def test_validation_discovers_enabled_plugin_toolsets(
        self, tmp_path, monkeypatch
    ):
        plugin_dir = tmp_path / "plugins" / "late-bound-toolset"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(
            "name: late-bound-toolset\n",
            encoding="utf-8",
        )
        (plugin_dir / "__init__.py").write_text(
            "def register(ctx):\n"
            "    ctx.register_tool(\n"
            "        name='late_bound_echo', toolset='late-bound-tools',\n"
            "        schema={'name': 'late_bound_echo', 'description': 'Echo',\n"
            "                'parameters': {'type': 'object', 'properties': {}}},\n"
            "        handler=lambda args, **kw: '{}',\n"
            "    )\n",
            encoding="utf-8",
        )
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "_config_version": DEFAULT_CONFIG["_config_version"],
                    "platform_toolsets": {"cli": ["late-bound-tools", "unknown-tools"]},
                    "plugins": {"enabled": ["late-bound-toolset"]},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        results = migrate_config(interactive=False, quiet=True)
        assert not any("late-bound-tools" in w for w in results["warnings"])
        assert any("unknown-tools" in w for w in results["warnings"])
