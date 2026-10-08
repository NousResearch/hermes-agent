"""Config v50: the bundled tirith scanner left Hermes; the user's tirith settings stay."""

import hermes_yaml as yaml


class TestRetiredTirithKeys:
    def _migrate(self, tmp_path, monkeypatch, security):
        from hermes_cli.config_migrations import run_migrations

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"_config_version": 49, "security": security}), encoding="utf-8")
        results = {"env_added": [], "config_added": [], "warnings": []}
        run_migrations(49, results, quiet=True)
        return yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8")), results

    def test_v50_keeps_tirith_keys_and_points_at_the_plugin(self, tmp_path, monkeypatch):
        """The upgrade never deletes the user's tirith settings; it tells them where tirith went."""
        security = {"redact_secrets": True, "tirith_enabled": True, "tirith_fail_open": False}
        raw, results = self._migrate(tmp_path, monkeypatch, dict(security))
        assert raw["security"] == security
        assert not (raw.get("plugins") or {}).get("enabled")
        assert any("sheeki03/hermes-plugin-tirith" in w for w in results["warnings"])

    def test_v50_is_silent_without_tirith_keys(self, tmp_path, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG

        raw, results = self._migrate(tmp_path, monkeypatch, {"redact_secrets": True})
        assert raw["security"] == {"redact_secrets": True}
        assert not any("tirith" in w for w in results["warnings"])
        assert not any(key.startswith("tirith") for key in DEFAULT_CONFIG["security"])
