"""Tests for the managed llama.cpp runtime branch in /model validation (#115237)."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.models_validate import validate_requested_model


def _validate(model, staged=("Qwen3-4B-Q4_K_M",), live=("other-model",), **kw):
    """validate against a managed-local world: staged files on disk + a live listing that
    (spawn-only as it is) hasn't learned the new file yet."""
    with (
        patch("hermes_cli.models.fetch_api_models", return_value=list(live)),
        patch(
            "hermes_cli.models.probe_api_models",
            return_value={
                "models": list(live),
                "probed_url": "http://127.0.0.1:18434/v1/models",
                "resolved_base_url": "http://127.0.0.1:18434/v1",
                "suggested_base_url": None,
                "used_fallback": False,
            },
        ),
        patch(
            "hermes_cli.local_runtime.bootstrap.staged_model_ids",
            return_value=list(staged),
        ),
    ):
        return validate_requested_model(
            model, "llamacpp", base_url="http://127.0.0.1:18434/v1", **kw
        )


class TestManagedLocalValidation:
    def test_staged_but_not_live_accepted(self):
        """The Use button's exact case: downloaded, not yet in the spawn-only live listing."""
        result = _validate("Qwen3-4B-Q4_K_M")
        assert (result["accepted"], result["persist"], result["recognized"]) == (
            True,
            True,
            True,
        )

    def test_not_staged_and_not_live_still_rejected(self):
        result = _validate("Llama-4-90B-Q4_K_M", staged=())
        assert result["accepted"] is False


class TestRefusedStagedModel:
    """A staged GGUF the launch planner refused never reaches the router's preset list, so every
    request for it 400s "model not found". Selecting it must fail up front, with the reason."""

    def _validate_with_refusal(self, model, refused_id="Big-33B-Q4_K_M"):
        from hermes_cli.local_runtime.presets import PresetEntry

        decisions = {
            refused_id: PresetEntry(
                model_id=refused_id, window=0, spilled=False,
                refusal="needs ~29.0 GiB at the 64K floor but only ~25.6 GiB are available",
            )
        }
        with patch(
            "hermes_cli.local_runtime.presets.read_preset_decisions", return_value=decisions
        ):
            return _validate(model, staged=("Qwen3-4B-Q4_K_M", refused_id))

    def test_refused_model_is_rejected_with_the_planners_reason(self):
        result = self._validate_with_refusal("Big-33B-Q4_K_M")
        assert (result["accepted"], result["persist"]) == (False, False)
        assert "can't run on this machine" in result["message"]
        assert "25.6 GiB" in result["message"]

    def test_refusal_match_is_case_insensitive(self):
        result = self._validate_with_refusal("big-33b-q4_k_m")
        assert result["accepted"] is False

    def test_servable_staged_model_is_still_accepted_beside_a_refused_one(self):
        result = self._validate_with_refusal("Qwen3-4B-Q4_K_M")
        assert result["accepted"] is True
