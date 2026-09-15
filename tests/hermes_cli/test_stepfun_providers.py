"""StepFun provider wiring: standard-chat `stepfun` + step-plan `stepfun-plan`."""

from __future__ import annotations

import sys
import types

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv


class TestStepfunProfiles:
    def test_standard_profile_registered(self):
        from providers import get_provider_profile

        prof = get_provider_profile("stepfun")
        assert prof is not None
        assert prof.base_url == "https://api.stepfun.ai/v1"
        assert prof.env_vars == ("STEPFUN_API_KEY",)

    def test_plan_profile_registered(self):
        from providers import get_provider_profile

        prof = get_provider_profile("stepfun-plan")
        assert prof is not None
        assert prof.base_url == "https://api.stepfun.ai/step_plan/v1"
        assert "stepfun-coding-plan" in prof.aliases

    def test_models_dev_mapping(self):
        from agent.models_dev import PROVIDER_TO_MODELS_DEV

        assert PROVIDER_TO_MODELS_DEV["stepfun"] == "stepfun-ai"
        assert PROVIDER_TO_MODELS_DEV["stepfun-plan"] == "stepfun-ai-step-plan"


class TestStepfunIdentity:
    def test_overlays(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        std = HERMES_OVERLAYS["stepfun"]
        assert std.transport == "openai_chat"
        assert std.base_url_override == "https://api.stepfun.ai/v1"
        assert std.base_url_env_var == "STEPFUN_BASE_URL"

        plan = HERMES_OVERLAYS["stepfun-plan"]
        assert plan.transport == "openai_chat"
        assert plan.base_url_override == "https://api.stepfun.ai/step_plan/v1"
        assert plan.base_url_env_var == "STEPFUN_STEP_PLAN_BASE_URL"

    def test_aliases(self):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("step") == "stepfun-plan"
        assert normalize_provider("stepfun-coding-plan") == "stepfun-plan"

    def test_resolve_provider_aliases(self, monkeypatch):
        # resolve_provider's alias map must match providers.py ALIASES so the
        # auth-resolution path and the registry agree on these spellings.
        from hermes_cli.auth import resolve_provider

        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-test-key")
        assert resolve_provider("stepfun-ai") == "stepfun"
        assert resolve_provider("stepfun-step-plan") == "stepfun-plan"

    def test_labels(self):
        from hermes_cli.providers import _LABEL_OVERRIDES

        assert _LABEL_OVERRIDES["stepfun"] == "StepFun"
        assert _LABEL_OVERRIDES["stepfun-plan"] == "StepFun Step Plan"

    def test_registry(self):
        from hermes_cli.auth import (
            PROVIDER_REGISTRY,
            STEPFUN_STEP_PLAN_INTL_BASE_URL,
        )

        assert PROVIDER_REGISTRY["stepfun"].inference_base_url == "https://api.stepfun.ai/v1"
        assert (
            PROVIDER_REGISTRY["stepfun-plan"].inference_base_url
            == STEPFUN_STEP_PLAN_INTL_BASE_URL
        )


class TestStepfunRegionToggle:
    def test_base_url_for_region_standard(self):
        from hermes_cli.main import _stepfun_base_url_for_region

        assert _stepfun_base_url_for_region("international", "standard") == "https://api.stepfun.ai/v1"
        assert _stepfun_base_url_for_region("china", "standard") == "https://api.stepfun.com/v1"

    def test_base_url_for_region_plan(self):
        from hermes_cli.main import _stepfun_base_url_for_region

        assert _stepfun_base_url_for_region("international", "plan") == "https://api.stepfun.ai/step_plan/v1"
        assert _stepfun_base_url_for_region("china", "plan") == "https://api.stepfun.com/step_plan/v1"

    def test_provider_models_lists(self):
        from hermes_cli.models import _PROVIDER_MODELS

        for pid in ("stepfun", "stepfun-plan"):
            assert "step-3.7-flash" in _PROVIDER_MODELS[pid]
            assert "step-3.5-flash" in _PROVIDER_MODELS[pid]


class TestStepfunMetadata:
    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "stepfun" in _PROVIDER_PREFIXES
        assert "stepfun-plan" in _PROVIDER_PREFIXES

    def test_vendor_prefix_unchanged(self):
        # `step-*` model slugs still resolve to the models.dev vendor `stepfun`,
        # independent of the hermes provider-id split.
        from hermes_cli.model_normalize import _VENDOR_PREFIXES

        assert _VENDOR_PREFIXES["step"] == "stepfun"


class TestStepfunEnvAndDoctor:
    def test_step_plan_base_url_env_registered(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "STEPFUN_STEP_PLAN_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["STEPFUN_STEP_PLAN_BASE_URL"]["category"] == "provider"

    def test_doctor_probes_both_stepfun_endpoints(self, monkeypatch, tmp_path):
        import contextlib
        import io
        from argparse import Namespace

        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        (home / ".env").write_text("STEPFUN_API_KEY=***\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-test-key")

        for env_name in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "GLM_API_KEY",
            "ZAI_API_KEY",
            "Z_AI_API_KEY",
            "KIMI_API_KEY",
            "KIMI_CN_API_KEY",
            "ARCEEAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "HF_TOKEN",
            "DASHSCOPE_API_KEY",
            "MINIMAX_API_KEY",
            "MINIMAX_CN_API_KEY",
            "KILOCODE_API_KEY",
            "OPENCODE_ZEN_API_KEY",
            "OPENCODE_GO_API_KEY",
            "XIAOMI_API_KEY",
            "GMI_API_KEY",
            "STEPFUN_BASE_URL",
            "STEPFUN_STEP_PLAN_BASE_URL",
        ):
            monkeypatch.delenv(env_name, raising=False)

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        try:
            from hermes_cli import auth as _auth_mod

            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        except Exception:
            pass

        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            return types.SimpleNamespace(status_code=200)

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        # Rebuild the cached apikey-provider list so the new StepFun rows are
        # picked up even if a prior test populated the module-level cache.
        monkeypatch.setattr(doctor_mod, "_APIKEY_PROVIDERS_CACHE", None)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "StepFun" in out
        assert any(url == "https://api.stepfun.ai/v1/models" for url, _, _ in calls)
        assert any(
            url == "https://api.stepfun.ai/step_plan/v1/models" for url, _, _ in calls
        )


class TestStepfunMigration:
    def _run(self, tmp_path, monkeypatch, model_cfg, env_seed=None):
        """Run the v33→v34 migration against an isolated temp hermes home.

        ``env_seed`` maps env-var names to values written into the temp
        ``.env`` BEFORE the migration runs, so the migration's
        ``get_env_value``/``save_env_value``/``remove_env_value`` helpers
        read and write the temp home (never the developer's real ~/.hermes).
        """
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        cfg = {"_config_version": 33, "model": model_cfg}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (home / ".env").write_text(
            "\n".join(f"{k}={v}" for k, v in (env_seed or {}).items()) + "\n",
            encoding="utf-8",
        )

        from hermes_cli import config as config_mod

        # `migrate_config` resolves config path via `hermes_cli.config.get_hermes_home`
        # (re-exported at module import), NOT the hermes_constants original — patch the
        # symbol the migration actually calls so reads AND writes land in tmp_path.
        # This same symbol backs get_env_path()/get_config_path(), so env reads/writes
        # (get_env_value/save_env_value/remove_env_value) are isolated too.
        monkeypatch.setattr(config_mod, "get_hermes_home", lambda: home)
        # Invalidate the raw-config cache so the temp file is read fresh.
        monkeypatch.setattr(config_mod, "_RAW_CONFIG_CACHE", {})
        # Invalidate the .env cache and clear real-process env so get_env_value
        # reads from the temp .env rather than a stale os.environ value.
        monkeypatch.setattr(config_mod, "_env_cache", None)
        for _k in ("STEPFUN_BASE_URL", "STEPFUN_STEP_PLAN_BASE_URL", "STEPFUN_API_KEY"):
            monkeypatch.delenv(_k, raising=False)

        config_mod.migrate_config(interactive=False, quiet=True)
        out = yaml.safe_load((home / "config.yaml").read_text()) or {}
        out["_env"] = {
            k: v
            for k, v in (
                line.split("=", 1)
                for line in (home / ".env").read_text(encoding="utf-8").splitlines()
                if "=" in line
            )
        }
        return out

    def test_step_plan_config_migrated(self, tmp_path, monkeypatch):
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "base_url": "https://api.stepfun.ai/step_plan/v1", "default": "step-3.5-flash"},
        )
        assert out["model"]["provider"] == "stepfun-plan"

    def test_standard_config_untouched(self, tmp_path, monkeypatch):
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "base_url": "https://api.stepfun.ai/v1", "default": "step-3.5-flash"},
        )
        assert out["model"]["provider"] == "stepfun"

    def test_env_var_step_plan_migrated(self, tmp_path, monkeypatch):
        # User configured Step Plan via STEPFUN_BASE_URL env var (no base_url in
        # config.yaml). Migration must rewrite provider → stepfun-plan AND move
        # the env value to the plan-specific var so it attaches to the right id.
        plan_url = "https://api.stepfun.ai/step_plan/v1"
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "default": "step-3.5-flash"},
            env_seed={"STEPFUN_BASE_URL": plan_url},
        )
        assert out["model"]["provider"] == "stepfun-plan"
        env = out["_env"]
        assert env.get("STEPFUN_STEP_PLAN_BASE_URL") == plan_url
        assert "STEPFUN_BASE_URL" not in env

    def test_env_var_standard_untouched(self, tmp_path, monkeypatch):
        # Plain /v1 URL under STEPFUN_BASE_URL → standard chat, env vars untouched.
        std_url = "https://api.stepfun.ai/v1"
        out = self._run(
            tmp_path, monkeypatch,
            {"provider": "stepfun", "default": "step-3.5-flash"},
            env_seed={"STEPFUN_BASE_URL": std_url},
        )
        assert out["model"]["provider"] == "stepfun"
        env = out["_env"]
        assert env.get("STEPFUN_BASE_URL") == std_url
        assert "STEPFUN_STEP_PLAN_BASE_URL" not in env


class TestStepfunRegionModelsDev:
    """models.dev metadata must resolve per the selected StepFun region.

    The China (`.com`) and International (`.ai`) endpoints are distinct
    models.dev catalogs; only the China Step Plan catalog carries
    `step-router-v1`. Regression coverage for both regions and both ids.
    """

    REGISTRY = {
        "stepfun-ai": {  # International standard
            "api": "https://api.stepfun.ai/v1",
            "models": {"step-3.7-flash": {"tool_call": True, "limit": {"context": 65536}}},
        },
        "stepfun": {  # China standard
            "api": "https://api.stepfun.com/v1",
            "models": {"step-3.7-flash": {"tool_call": True, "limit": {"context": 65536}}},
        },
        "stepfun-ai-step-plan": {  # International Step Plan
            "api": "https://api.stepfun.ai/step_plan/v1",
            "models": {"step-3.7-flash": {"tool_call": True, "limit": {"context": 65536}}},
        },
        "stepfun-step-plan": {  # China Step Plan — carries step-router-v1
            "api": "https://api.stepfun.com/step_plan/v1",
            "models": {
                "step-3.7-flash": {"tool_call": True, "limit": {"context": 65536}},
                "step-router-v1": {"tool_call": True, "limit": {"context": 32768}},
            },
        },
    }

    def _patch(self, monkeypatch):
        import agent.models_dev as md
        monkeypatch.setattr(md, "fetch_models_dev", lambda *a, **k: self.REGISTRY)
        # Isolate from any real ~/.hermes/.env override lookups.
        monkeypatch.setattr(md, "_read_base_url_override", lambda env_var: None)
        return md

    def test_resolve_id_by_base_url(self, monkeypatch):
        md = self._patch(monkeypatch)
        assert md._resolve_models_dev_id("stepfun") == "stepfun-ai"
        assert md._resolve_models_dev_id("stepfun-plan") == "stepfun-ai-step-plan"
        assert (
            md._resolve_models_dev_id("stepfun", "https://api.stepfun.com/v1")
            == "stepfun"
        )
        assert (
            md._resolve_models_dev_id("stepfun-plan", "https://api.stepfun.com/step_plan/v1")
            == "stepfun-step-plan"
        )

    def test_resolve_id_from_env_override(self, monkeypatch):
        import agent.models_dev as md
        monkeypatch.setattr(md, "fetch_models_dev", lambda *a, **k: self.REGISTRY)
        monkeypatch.setenv("STEPFUN_STEP_PLAN_BASE_URL", "https://api.stepfun.com/step_plan/v1")
        monkeypatch.delenv("STEPFUN_BASE_URL", raising=False)
        # China env override → China Step Plan id even with no base_url passed.
        assert md._resolve_models_dev_id("stepfun-plan") == "stepfun-step-plan"
        # Standard id still International (its own env var unset).
        assert md._resolve_models_dev_id("stepfun") == "stepfun-ai"

    def test_china_step_plan_catalog_has_router(self, monkeypatch):
        md = self._patch(monkeypatch)
        cn = md.list_provider_models("stepfun-plan", base_url="https://api.stepfun.com/step_plan/v1")
        intl = md.list_provider_models("stepfun-plan", base_url="https://api.stepfun.ai/step_plan/v1")
        assert "step-router-v1" in cn
        assert "step-router-v1" not in intl

    def test_context_lookup_routes_by_region(self, monkeypatch):
        md = self._patch(monkeypatch)
        # step-router-v1 only exists in the China Step Plan catalog.
        assert md.lookup_models_dev_context(
            "stepfun-plan", "step-router-v1", base_url="https://api.stepfun.com/step_plan/v1"
        ) == 32768
        assert md.lookup_models_dev_context(
            "stepfun-plan", "step-router-v1", base_url="https://api.stepfun.ai/step_plan/v1"
        ) is None
