"""StepFun /model picker merges the Step Plan live catalog with the curated list.

The StepFun provider's inference endpoint is the Step Plan API, whose /models
returns a subset of the full catalog (it omits ``step-3.7-flash``). Every real
picker surface resolves through ``provider_model_ids`` -> the stepfun fetcher
(``_api_key_provider_live``), so the merge must happen there: a working live
response may no longer shadow curated-only models. (#41147)
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest

from hermes_cli.models import provider_model_ids

STEPFUN_CREDS = {
    "provider": "stepfun",
    "api_key": "sk-stepfun-test",
    "base_url": "https://api.stepfun.ai/step_plan/v1",
}

STEP_PLAN_LIVE_IDS = ["step-3.5-flash", "step-3.5-flash-2603"]


@pytest.fixture()
def hermetic_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so the picker cannot read a real disk cache row."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "stepfun-home"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    yield home
    reset_hermes_home_override(token)


@contextmanager
def step_plan_live_listing(ids):
    """Patch the stepfun credential resolver and /v1/models probe."""
    with patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        return_value=dict(STEPFUN_CREDS),
    ), patch("hermes_cli.models.fetch_api_models", return_value=list(ids)):
        yield


class TestStepfunPickerMergesLiveWithCurated:
    def test_live_step_plan_response_still_lists_curated_only_models(self, hermetic_home):
        """The reported repro: live Step Plan up, step-3.7-flash still missing."""
        with step_plan_live_listing(STEP_PLAN_LIVE_IDS):
            ids = provider_model_ids("stepfun")
        assert "step-3.7-flash" in ids, (
            "stepfun picker dropped a curated model when the Step Plan live fetch succeeded"
        )
        # Live rows keep their position ahead of curated-only additions.
        assert ids[:2] == STEP_PLAN_LIVE_IDS

    def test_merge_dedupes_and_keeps_the_curated_floor(self, hermetic_home):
        with step_plan_live_listing(STEP_PLAN_LIVE_IDS):
            ids = provider_model_ids("stepfun")
        assert len(ids) == len(set(i.lower() for i in ids)), "duplicate rows in merged picker"
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        for curated_id in _PROVIDER_MODELS["stepfun"]:
            assert curated_id in ids

    def test_failed_live_fetch_keeps_curated_placeholder(self, hermetic_home):
        """Outage fallback unchanged: the curated list, flagged as placeholder."""
        with step_plan_live_listing([]):
            ids = provider_model_ids("stepfun")
        from hermes_cli.models_catalog_static import CuratedFallbackModels, _PROVIDER_MODELS

        assert list(ids) == list(_PROVIDER_MODELS["stepfun"])
        # The placeholder flag rides on the list subclass, so the disk cache can
        # tell an outage row from the account's real catalog.
        assert isinstance(ids, CuratedFallbackModels)

    def test_wizard_offer_merges_live_and_curated(self, hermetic_home):
        """The setup wizard's StepFun flow must offer the same merged list."""
        from hermes_cli.model_setup_flows import _model_flow_stepfun

        captured = {}

        def fake_pick(model_list, *args, **kwargs):
            captured["models"] = list(model_list)

        with patch(
            "hermes_cli.model_setup_flows._ensure_flow_api_key", return_value=(None, "sk-stepfun-test", False)
        ), patch(
            "hermes_cli.main_provider_setup._prompt_provider_choice", return_value=0
        ), patch("hermes_cli.config.save_env_value"), patch(
            "hermes_cli.model_setup_flows._pick_model_or_prompt", side_effect=fake_pick
        ), patch(
            "hermes_cli.model_setup_flows._finish_model", return_value=None
        ):
            _model_flow_stepfun({})
        assert "step-3.7-flash" in captured["models"], (
            "StepFun setup wizard offered the unmerged Step Plan list"
        )
        # Contract: the wizard offers the same MERGED list as the picker — every
        # Step Plan live id plus the curated floor, no duplicates.
        assert set(STEP_PLAN_LIVE_IDS) <= {m.lower() for m in captured["models"]}
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        for curated_id in _PROVIDER_MODELS["stepfun"]:
            assert curated_id in captured["models"]
        assert len(captured["models"]) == len({m.lower() for m in captured["models"]})
