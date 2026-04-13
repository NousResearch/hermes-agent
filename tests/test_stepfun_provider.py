"""Tests for StepFun / StepFun Plan provider wiring.

Verifies that the stepfun and stepfun-plan providers are correctly
registered across all layers: providers, auth, models, auxiliary client,
model metadata, and alias resolution.
"""

import pytest


# ---------------------------------------------------------------------------
# providers.py
# ---------------------------------------------------------------------------

class TestProvidersOverlay:
    """HERMES_OVERLAYS and ALIASES in hermes_cli/providers.py."""

    def test_stepfun_overlay_registered(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "stepfun" in HERMES_OVERLAYS
        assert HERMES_OVERLAYS["stepfun"].transport == "openai_chat"
        assert HERMES_OVERLAYS["stepfun"].base_url_env_var == "STEPFUN_BASE_URL"

    def test_stepfun_plan_overlay_registered(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "stepfun-plan" in HERMES_OVERLAYS
        assert HERMES_OVERLAYS["stepfun-plan"].transport == "openai_chat"
        assert HERMES_OVERLAYS["stepfun-plan"].base_url_env_var == "STEPFUN_PLAN_BASE_URL"

    def test_aliases_resolve_to_stepfun(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("step-fun") == "stepfun"
        assert normalize_provider("step_fun") == "stepfun"
        assert normalize_provider("stepfun") == "stepfun"

    def test_aliases_resolve_to_stepfun_plan(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("stepplan") == "stepfun-plan"
        assert normalize_provider("step-plan") == "stepfun-plan"
        assert normalize_provider("step_plan") == "stepfun-plan"

    def test_get_provider_stepfun(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("stepfun")
        assert pdef is not None
        assert pdef.id == "stepfun"
        assert pdef.transport == "openai_chat"
        assert "STEPFUN_API_KEY" in pdef.api_key_env_vars

    def test_get_provider_stepfun_plan(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("stepfun-plan")
        assert pdef is not None
        assert pdef.id == "stepfun-plan"
        assert pdef.transport == "openai_chat"

    def test_determine_api_mode(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("stepfun") == "chat_completions"
        assert determine_api_mode("stepfun-plan") == "chat_completions"


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

class TestModels:
    """_PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES in models.py."""

    def test_stepfun_models_exist(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "stepfun" in _PROVIDER_MODELS
        assert "step-3.5-flash" in _PROVIDER_MODELS["stepfun"]

    def test_stepfun_plan_models_exist(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "stepfun-plan" in _PROVIDER_MODELS
        assert "step-3.5-flash" in _PROVIDER_MODELS["stepfun-plan"]
        assert "step-3.5-flash-2603" in _PROVIDER_MODELS["stepfun-plan"]

    def test_provider_labels(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS.get("stepfun") == "StepFun"
        assert _PROVIDER_LABELS.get("stepfun-plan") == "StepFun Plan"

    def test_provider_aliases(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("step-fun") == "stepfun"
        assert normalize_provider("stepplan") == "stepfun-plan"
        assert normalize_provider("step-plan") == "stepfun-plan"

    def test_provider_label_function(self):
        from hermes_cli.models import provider_label
        assert provider_label("stepfun") == "StepFun"
        assert provider_label("stepfun-plan") == "StepFun Plan"


# ---------------------------------------------------------------------------
# model_metadata.py
# ---------------------------------------------------------------------------

class TestModelMetadata:
    """Context lengths and URL inference in model_metadata.py."""

    def test_context_length_step_3_5(self):
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        # step-3.5 prefix should match 262144
        assert any(
            k.startswith("step-3.5") and v == 262144
            for k, v in DEFAULT_CONTEXT_LENGTHS.items()
        )

    def test_url_to_provider_mapping(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.stepfun.com") == "stepfun"
        assert _URL_TO_PROVIDER.get("api.stepfun.ai") == "stepfun"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "stepfun" in _PROVIDER_PREFIXES
        assert "stepfun-plan" in _PROVIDER_PREFIXES
        assert "step-fun" in _PROVIDER_PREFIXES
        assert "step-plan" in _PROVIDER_PREFIXES


# ---------------------------------------------------------------------------
# models_dev.py
# ---------------------------------------------------------------------------

class TestModelsDev:
    """PROVIDER_TO_MODELS_DEV mapping in models_dev.py."""

    def test_stepfun_mapped(self):
        from agent.models_dev import PROVIDER_TO_MODELS_DEV
        assert PROVIDER_TO_MODELS_DEV.get("stepfun") == "stepfun"

    def test_stepfun_plan_mapped(self):
        from agent.models_dev import PROVIDER_TO_MODELS_DEV
        # Both map to the same models.dev provider
        assert PROVIDER_TO_MODELS_DEV.get("stepfun-plan") == "stepfun"


# ---------------------------------------------------------------------------
# auxiliary_client.py (requires openai SDK)
# ---------------------------------------------------------------------------

_has_openai = True
try:
    import openai  # noqa: F401
except ImportError:
    _has_openai = False


@pytest.mark.skipif(not _has_openai, reason="openai SDK not installed")
class TestAuxiliaryClient:
    """Aux model defaults in auxiliary_client.py."""

    def test_stepfun_aux_model(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS.get("stepfun") == "step-3.5-flash"

    def test_stepfun_plan_aux_model(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS.get("stepfun-plan") == "step-3.5-flash"

    def test_aux_aliases(self):
        from agent.auxiliary_client import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("step-fun") == "stepfun"
        assert _PROVIDER_ALIASES.get("stepplan") == "stepfun-plan"
        assert _PROVIDER_ALIASES.get("step-plan") == "stepfun-plan"
