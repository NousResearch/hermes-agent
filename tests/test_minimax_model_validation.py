"""Tests for MiniMax model validation via static catalog fallback (issues #12611, #12460, #12399, #12547).

MiniMax exposes ``/v1/models`` (OpenAI-compat catalog on the same host), so
the live fetch in ``validate_requested_model()`` does work — but it can
still fail with 401 on a stale key, network error, or a custom ``base_url``
that has no ``/models`` path.  The catalog path below is the *fallback*
that keeps validation from wedging on a transient fetch error.  Tests in
``tests/hermes_cli/test_minimax_picker.py`` cover the live-fetch path.
"""

from unittest.mock import patch

import pytest

from hermes_cli.models import validate_requested_model


class TestMiniMaxModelValidation:
    """Test that validate_requested_model handles MiniMax providers correctly."""

    @pytest.fixture(autouse=True)
    def _isolate_minimax(self):
        """Ensure MiniMax catalog is used even if a live /v1/models endpoint exists."""
        # Simulate fetch_api_models returning None (i.e., /v1/models is unreachable),
        # proving that the catalog path is taken.
        probe_payload = {
            "models": None,
            "probed_url": "https://api.minimax.io/v1/models",
            "resolved_base_url": "https://api.minimax.io/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
            yield

    # -------------------------------------------------------------------------
    # Test 1: A known MiniMax model is accepted with recognized=True
    # -------------------------------------------------------------------------
    def test_valid_minimax_model_accepted(self):
        result = validate_requested_model("MiniMax-M2.7", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    # -------------------------------------------------------------------------
    # Test 1b: Case-insensitive lookup matches catalog entries
    # -------------------------------------------------------------------------
    def test_valid_minimax_model_case_insensitive(self):
        result = validate_requested_model("minimax-m2.7", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_valid_minimax_model_uppercase(self):
        result = validate_requested_model("MINIMAX-M2.7", "minimax")
        assert result["accepted"] is True
        assert result["recognized"] is True

    # -------------------------------------------------------------------------
    # Test 2: A near-match model on minimax-cn triggers a suggestion (not auto-correct)
    # -------------------------------------------------------------------------
    def test_near_match_minimax_cn_suggests_similar(self):
        # "MiniMax-M2.7-highspeed" is somewhat similar to "MiniMax-M2.7" (ratio ~0.71)
        # but below the 0.9 auto-correct cutoff.  The merged catalog
        # (static + models.dev) now lists it, so it's recognized; we
        # still assert that no auto-correction happens (because the
        # similarity is below the 0.9 cutoff for *equal-id* correction —
        # the catalog already contains the exact id, so correction is
        # unnecessary).  The "similar models" suggestion is irrelevant
        # when the model is recognized, so the message check below is
        # only meaningful for the unrecognized branches covered by the
        # other tests in this class.
        result = validate_requested_model("MiniMax-M2.7-highspeed", "minimax-cn")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        # Should NOT auto-correct (the exact id is already in the catalog)
        assert "corrected_model" not in result

    # -------------------------------------------------------------------------
    # Test 3: A completely unknown model is accepted (not rejected) with a warning
    # -------------------------------------------------------------------------
    def test_unknown_minimax_model_accepted_with_warning(self):
        # "NotARealModel" has very low similarity to any MiniMax model (~0.16).
        # It should still be accepted (not rejected), with recognized=False and
        # a note that the live /v1/models fetch (when available) didn't list it.
        result = validate_requested_model("NotARealModel", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is False
        assert "NotARealModel" in result["message"]
        assert "not found in the MiniMax catalog" in result["message"]

    # -------------------------------------------------------------------------
    # Test 4: Verify catalog path is used (probe_api_models returns None)
    # -------------------------------------------------------------------------
    def test_minimax_uses_catalog_not_api_probe(self):
        """Ensure that when fetch_api_models returns None, the catalog is still checked."""
        # The _isolate_minimax fixture already patches fetch_api_models to return None.
        # If we reach the catalog path, MiniMax-M2.5 should be found and recognized.
        result = validate_requested_model("MiniMax-M2.5", "minimax")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result["message"] is None


class TestMiniMaxCatalogPathRequired:
    """Prove the catalog path is necessary: without it, MiniMax would fail.

    These tests demonstrate that when fetch_api_models returns None (simulating
    the real 404 from MiniMax /v1/models), the openai-codex-style catalog path
    is the only way to avoid a "Could not reach the API" failure.
    """

    def test_minimax_without_fix_would_reach_api_probe(self):
        """Without the catalog block, minimax falls through to fetch_api_models.

        This test documents the before-fix behavior: when the MiniMax block
        is absent, the code falls through to `api_models = fetch_api_models(...)`
        which returns None (404), leading to rejection.
        """
        probe_payload = {
            "models": None,
            "probed_url": "https://api.minimax.io/v1/models",
            "resolved_base_url": "https://api.minimax.io/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
            # Before fix: this would return accepted=False because api_models is None
            # After fix: returns accepted=True via catalog path
            result = validate_requested_model("MiniMax-M2.7", "minimax")
            # The fix makes this True; without the fix it would be False
            assert result["accepted"] is True
