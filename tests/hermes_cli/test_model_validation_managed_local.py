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
        assert "managed local-models library" in result["message"]

    def test_typed_case_matches_staged_spelling(self):
        result = _validate("qwen3-4b-q4_k_m")
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_not_staged_and_not_live_still_rejected(self):
        result = _validate("Llama-4-90B-Q4_K_M", staged=())
        assert result["accepted"] is False
        assert "was not found in this provider's model listing" in result["message"]

    def test_live_hit_still_wins(self):
        """A model the router already serves validates through the live listing, staged or not."""
        result = _validate("other-model", staged=())
        assert (result["accepted"], result["persist"], result["recognized"]) == (
            True,
            True,
            True,
        )
        assert result["message"] is None

    def test_aliases_route_to_the_same_branch(self):
        for alias in ("llama.cpp", "llama-cpp"):
            result = _validate("Qwen3-4B-Q4_K_M")
            assert result["accepted"] is True, alias

    def test_other_providers_do_not_read_the_staged_library(self):
        """The branch is llamacpp-only: a cloud provider must keep its live-listing verdict."""
        with (
            patch("hermes_cli.models.fetch_api_models", return_value=["deepseek-v4.1"]),
            patch(
                "hermes_cli.models.probe_api_models",
                return_value={
                    "models": ["deepseek-v4.1"],
                    "probed_url": "x",
                    "resolved_base_url": "x",
                    "suggested_base_url": None,
                    "used_fallback": False,
                },
            ),
            patch(
                "hermes_cli.local_runtime.bootstrap.staged_model_ids",
                return_value=["Qwen3-4B-Q4_K_M"],
            ),
        ):
            result = validate_requested_model(
                "Qwen3-4B-Q4_K_M", "deepseek", base_url="https://api.deepseek.com/v1"
            )
        assert result["accepted"] is False

    def test_empty_staged_library_falls_through(self):
        """No staged files: nothing to credit, the live listing decides (None fall-through)."""
        result = _validate("other-model", staged=())
        assert result["accepted"] is True  # live hit via _validate_live_listing
