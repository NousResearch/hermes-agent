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
