"""Config v50: the bundled tirith scanner left Hermes."""

import hermes_yaml as yaml


class TestRetiredTirithKeys:
    def test_v50_drops_tirith_keys_and_enables_nothing(self, tmp_path, monkeypatch):
        """The scanner left core: its keys are dropped, sibling security settings survive, and no
        plugin is enabled in its place."""
        from hermes_cli.config import DEFAULT_CONFIG
        from hermes_cli.config_migrations import run_migrations

        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump({
            "_config_version": 49,
            "security": {"redact_secrets": True, "tirith_enabled": True, "tirith_fail_open": False},
        }), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        run_migrations(49, {"env_added": [], "config_added": [], "warnings": []}, quiet=True)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw["security"] == {"redact_secrets": True}
        assert not (raw.get("plugins") or {}).get("enabled")
        assert not any(key.startswith("tirith") for key in DEFAULT_CONFIG["security"])


    def test_v50_points_fail_closed_users_at_the_plugin(self, tmp_path, monkeypatch):
        """Only a user who had tirith blocking commands (fail_open false) is told where it went."""
        from hermes_cli.config_migrations import run_migrations

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        for fail_open, expect in ((False, True), (True, False)):
            (tmp_path / "config.yaml").write_text(yaml.safe_dump({
                "_config_version": 49, "security": {"tirith_enabled": True, "tirith_fail_open": fail_open},
            }), encoding="utf-8")
            results = {"env_added": [], "config_added": [], "warnings": []}
            run_migrations(49, results, quiet=True)
            assert any("sheeki03/hermes-plugin-tirith" in w for w in results["warnings"]) is expect
