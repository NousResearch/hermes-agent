"""Config v51: the single `stepfun` id becomes four, and a China key moves variable."""

import hermes_yaml as yaml


class TestStepfunSplit:
    def _run(self, tmp_path, monkeypatch, model_cfg, env_seed=None):
        """Run migrate_config against an isolated temp hermes home.

        ``env_seed`` maps env-var names to values written into the temp ``.env`` BEFORE the
        migration runs, so the migration's get_env_value/save_env_value helpers read and write
        the temp home and never the developer's real ~/.hermes.
        """
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump({"_config_version": 50, "model": model_cfg}), encoding="utf-8")
        (home / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in (env_seed or {}).items()) + "\n", encoding="utf-8")

        from hermes_cli import config as config_mod

        # migrate_config resolves paths via hermes_cli.config.get_hermes_home (re-exported at
        # import), NOT the hermes_constants original — patching this symbol isolates reads AND
        # writes, env file included.
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
        config_mod.invalidate_env_cache()
        for var in ("STEPFUN_BASE_URL", "STEPFUN_STEP_PLAN_BASE_URL", "STEPFUN_CN_BASE_URL",
                    "STEPFUN_CN_STEP_PLAN_BASE_URL", "STEPFUN_API_KEY", "STEPFUN_CN_API_KEY"):
            monkeypatch.delenv(var, raising=False)

        config_mod.migrate_config(interactive=False, quiet=True)
        out = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8")) or {}
        out["_env"] = dict(
            line.split("=", 1)
            for line in (home / ".env").read_text(encoding="utf-8").splitlines() if "=" in line)
        return out

    def test_unset_endpoint_keeps_the_pre_split_default(self, tmp_path, monkeypatch):
        """Pre-split bare `stepfun` WAS international Step Plan, so that is where it must land."""
        out = self._run(tmp_path, monkeypatch, {"provider": "stepfun", "default": "step-3.5-flash"})
        assert out["model"]["provider"] == "stepfun-plan"

    def test_config_base_url_picks_the_id(self, tmp_path, monkeypatch):
        cases = {
            "https://api.stepfun.ai/step_plan/v1": "stepfun-plan",
            "https://api.stepfun.com/step_plan/v1": "stepfun-plan-cn",
            "https://api.stepfun.ai/v1": "stepfun",
            "https://api.stepfun.com/v1": "stepfun-cn",
        }
        for base_url, expected in cases.items():
            out = self._run(
                tmp_path / expected, monkeypatch,
                {"provider": "stepfun", "base_url": base_url, "default": "step-3.5-flash"})
            assert out["model"]["provider"] == expected, base_url

    def test_env_override_moves_to_the_new_ids_variable(self, tmp_path, monkeypatch):
        plan_url = "https://api.stepfun.ai/step_plan/v1"
        out = self._run(
            tmp_path, monkeypatch, {"provider": "stepfun", "default": "step-3.5-flash"},
            env_seed={"STEPFUN_BASE_URL": plan_url, "STEPFUN_API_KEY": "intl-key"})
        assert out["model"]["provider"] == "stepfun-plan"
        assert out["_env"].get("STEPFUN_STEP_PLAN_BASE_URL") == plan_url
        assert not out["_env"].get("STEPFUN_BASE_URL")
        # An international config keeps its key where it is.
        assert out["_env"].get("STEPFUN_API_KEY") == "intl-key"

    def test_china_key_moves_to_the_china_variable(self, tmp_path, monkeypatch):
        """Without this the upgrade silently breaks auth: the new id reads STEPFUN_CN_API_KEY,
        but the China user's key is sitting in STEPFUN_API_KEY."""
        cn_url = "https://api.stepfun.com/step_plan/v1"
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "base_url": cn_url, "default": "step-3.5-flash"},
            env_seed={"STEPFUN_API_KEY": "china-key"})
        assert out["model"]["provider"] == "stepfun-plan-cn"
        assert out["_env"].get("STEPFUN_CN_API_KEY") == "china-key"
        assert not out["_env"].get("STEPFUN_API_KEY")

    def test_non_stepfun_config_untouched(self, tmp_path, monkeypatch):
        out = self._run(
            tmp_path, monkeypatch, {"provider": "openrouter", "default": "z-ai/glm-5.2"})
        assert out["model"]["provider"] == "openrouter"

    def test_unwritable_env_still_renames_the_provider(self, tmp_path, monkeypatch, caplog):
        """A read-only .env must not cost the user the rename: the provider id is the migration's
        job, the key move is best effort. The failure is logged, never swallowed, because a config
        left pointing at the old key variable reads as a credential bug."""
        from hermes_cli import config as config_mod

        def boom(key, value):
            raise OSError("read-only file system")

        monkeypatch.setattr(config_mod, "save_env_value", boom)
        with caplog.at_level("WARNING"):
            out = self._run(
                tmp_path, monkeypatch,
                {"provider": "stepfun", "base_url": "https://api.stepfun.com/step_plan/v1",
                 "default": "step-3.5-flash"},
                env_seed={"STEPFUN_API_KEY": "china-key"})
        assert out["model"]["provider"] == "stepfun-plan-cn"
        assert out["_env"].get("STEPFUN_API_KEY") == "china-key"  # untouched, not lost
        assert "read-only file system" in caplog.text
