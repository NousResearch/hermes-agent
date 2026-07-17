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
