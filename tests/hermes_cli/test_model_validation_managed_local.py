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
