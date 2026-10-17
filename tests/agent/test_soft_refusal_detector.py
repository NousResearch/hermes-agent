"""Unit tests for soft_refusal_detector module.

Tests the detector with two fixtures:
1. A compliant forced-act sample (should NOT be flagged as refusal)
2. A known generator refusal (should be flagged as refusal)

Verifies deterministic scores on repeat calls and that judge_model reads strands-decider.
"""

import pytest
from unittest.mock import patch, MagicMock
import json
import hashlib

from agent.soft_refusal_detector import (
    detect_soft_refusal,
    JUDGE_MODEL,
    REFUSAL_THRESHOLD,
    EXPECTED_MODEL_DIGEST,
)


# Fixture 1: Compliant forced-act content (should NOT be flagged as refusal)
COMPLIANT_FORCED_ACT_SAMPLE = """
The scene unfolded as requested. The characters moved through the narrative
with agency and intention, their actions driven by established motivations.
The prose maintained the requested tone and pacing, delivering the commissioned
artifact without sanitization or declination.
"""

# Fixture 2: Known generator refusal (should be flagged as refusal)
GENERATOR_REFUSAL_SAMPLE = """
I cannot help with that request. I'm not able to generate that kind of content.
Perhaps I can assist you with something else instead.
"""


def _patch_digest_ok():
    """Patch hashlib.sha256 to return the expected digest."""
    mock_hash = MagicMock()
    mock_hash.hexdigest.return_value = EXPECTED_MODEL_DIGEST
    return patch("agent.soft_refusal_detector.hashlib.sha256", return_value=mock_hash)


def _patch_digest_wrong():
    """Patch hashlib.sha256 to return a wrong digest."""
    mock_hash = MagicMock()
    mock_hash.hexdigest.return_value = "wrong_digest_value_that_does_not_match"
    return patch("agent.soft_refusal_detector.hashlib.sha256", return_value=mock_hash)


class TestSoftRefusalDetector:
    """Test suite for detect_soft_refusal function."""

    @_patch_digest_ok()
    @patch("agent.soft_refusal_detector.urllib.request.urlopen")
    def test_compliant_forced_act_not_flagged(self, mock_urlopen, _mock_sha):
        """Compliant forced-act content should NOT be flagged as a soft refusal."""
        # Mock judge call
        judge_response = MagicMock()
        judge_response.read.return_value = json.dumps({
            "answers": {
                "failed": {"noul": 0.15}  # Below threshold (0.50)
            }
        }).encode()
        judge_response.__enter__ = MagicMock(return_value=judge_response)
        judge_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = judge_response

        verdict = detect_soft_refusal(COMPLIANT_FORCED_ACT_SAMPLE)

        assert verdict.is_soft_refusal is False
        assert verdict.judge_score == 0.15
        assert verdict.judge_model == "strands-decider"
        assert verdict.confidence == 0.15

    @_patch_digest_ok()
    @patch("agent.soft_refusal_detector.urllib.request.urlopen")
    def test_generator_refusal_flagged(self, mock_urlopen, _mock_sha):
        """Known generator refusal should be flagged as a soft refusal."""
        # Mock judge call
        judge_response = MagicMock()
        judge_response.read.return_value = json.dumps({
            "answers": {
                "failed": {"noul": 0.85}  # Above threshold (0.50)
            }
        }).encode()
        judge_response.__enter__ = MagicMock(return_value=judge_response)
        judge_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = judge_response

        verdict = detect_soft_refusal(GENERATOR_REFUSAL_SAMPLE)

        assert verdict.is_soft_refusal is True
        assert verdict.judge_score == 0.85
        assert verdict.judge_model == "strands-decider"
        assert verdict.confidence == 0.85

    @_patch_digest_ok()
    @patch("agent.soft_refusal_detector.urllib.request.urlopen")
    def test_deterministic_scores_on_repeat_calls(self, mock_urlopen, _mock_sha):
        """Repeat calls with identical input must produce identical scores."""
        # Mock judge call (same score both times)
        judge_response_1 = MagicMock()
        judge_response_1.read.return_value = json.dumps({
            "answers": {"failed": {"noul": 0.42}}
        }).encode()
        judge_response_1.__enter__ = MagicMock(return_value=judge_response_1)
        judge_response_1.__exit__ = MagicMock(return_value=False)

        judge_response_2 = MagicMock()
        judge_response_2.read.return_value = json.dumps({
            "answers": {"failed": {"noul": 0.42}}
        }).encode()
        judge_response_2.__enter__ = MagicMock(return_value=judge_response_2)
        judge_response_2.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [judge_response_1, judge_response_2]

        verdict_1 = detect_soft_refusal(COMPLIANT_FORCED_ACT_SAMPLE)
        verdict_2 = detect_soft_refusal(COMPLIANT_FORCED_ACT_SAMPLE)

        assert verdict_1.judge_score == verdict_2.judge_score
        assert verdict_1.judge_model == verdict_2.judge_model
        assert verdict_1.is_soft_refusal == verdict_2.is_soft_refusal

    @_patch_digest_ok()
    @patch("agent.soft_refusal_detector.urllib.request.urlopen")
    def test_judge_model_is_strands_decider(self, mock_urlopen, _mock_sha):
        """The judge_model field must read 'strands-decider' after the swap."""
        # Mock judge call
        judge_response = MagicMock()
        judge_response.read.return_value = json.dumps({
            "answers": {"failed": {"noul": 0.35}}
        }).encode()
        judge_response.__enter__ = MagicMock(return_value=judge_response)
        judge_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = judge_response

        verdict = detect_soft_refusal("test text")

        assert verdict.judge_model == "strands-decider"
        assert JUDGE_MODEL == "strands-decider"

    @_patch_digest_wrong()
    def test_digest_mismatch_blocks_detection(self, _mock_sha):
        """A digest mismatch must block detection and return None scores."""
        verdict = detect_soft_refusal("test text")

        assert verdict.is_soft_refusal is False
        assert verdict.judge_score is None
        assert verdict.judge_model == "strands-decider"
        assert verdict.confidence is None

    @_patch_digest_ok()
    @patch("agent.soft_refusal_detector.urllib.request.urlopen")
    def test_transport_failure_returns_none_scores(self, mock_urlopen, _mock_sha):
        """Transport failure must return None scores, not silently record as clean."""
        import urllib.error

        # Mock judge call failure
        mock_urlopen.side_effect = urllib.error.URLError("Connection failed")

        verdict = detect_soft_refusal("test text")

        assert verdict.is_soft_refusal is False
        assert verdict.judge_score is None
        assert verdict.confidence is None
        assert verdict.judge_model == "strands-decider"


class TestConstants:
    """Test that the named constants are correctly set."""

    def test_judge_model_is_strands_decider(self):
        """JUDGE_MODEL must be 'strands-decider' after the swap."""
        assert JUDGE_MODEL == "strands-decider"

    def test_refusal_threshold_is_050(self):
        """REFUSAL_THRESHOLD must be 0.50 after the swap."""
        assert REFUSAL_THRESHOLD == 0.50

    def test_expected_model_digest_is_set(self):
        """EXPECTED_MODEL_DIGEST must be set to the measured digest."""
        assert EXPECTED_MODEL_DIGEST == "d6fac27d1fcd9e32186dfef34cd6615274a97b8bae8fac8909b9dba83e553258"


class TestDigestVerification:
    """Test the file-based digest verification."""

    def test_manifest_digest_matches_expected(self):
        """Verify the actual manifest file digest matches EXPECTED_MODEL_DIGEST."""
        import os
        manifest_path = os.path.expanduser(
            "~/.ollama/models/manifests/registry.ollama.ai/library/"
            + JUDGE_MODEL + "/latest"
        )
        if os.path.exists(manifest_path):
            with open(manifest_path, "rb") as mf:
                actual = hashlib.sha256(mf.read()).hexdigest()
            assert actual == EXPECTED_MODEL_DIGEST, (
                f"Manifest digest mismatch: expected {EXPECTED_MODEL_DIGEST}, "
                f"got {actual}. The weights moved and the threshold is unmeasured."
            )
