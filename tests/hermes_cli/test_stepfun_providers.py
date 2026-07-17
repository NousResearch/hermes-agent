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
