"""End-to-end: Gemini Web Bridge 503 with extension_disconnected is non-retryable + fallback.

Verifies that a 503 carrying the bridge's ``extension_disconnected`` error code is
classified as a deterministic (non-retryable) failure that engages the fallback
chain, and that the bridge resilience config treats 503 as a non-retryable
status so the fallback ladder engages.
"""

from pathlib import Path

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class MockAPIError(Exception):
    def __init__(self, message="Service Unavailable", status_code=503, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.body = body if body is not None else {
            "error": {"code": "extension_disconnected", "type": "service_unavailable"}
        }


def test_bridge_503_extension_disconnected_is_non_retryable_with_fallback():
    err = MockAPIError(
        body={"error": {"code": "extension_disconnected", "type": "service_unavailable"}},
        status_code=503,
    )
    classified = classify_api_error(err)

    assert classified.status_code == 503
    assert classified.retryable is False
    assert classified.should_fallback is True


def test_bridge_503_extension_disconnected_reason_is_terminal():
    err = MockAPIError(
        body={"error": {"code": "extension_disconnected", "type": "service_unavailable"}},
        status_code=503,
    )
    classified = classify_api_error(err)
    # Non-retryable bridge failure routes to the fallback chain (server_error family).
    assert classified.reason == FailoverReason.server_error
    assert classified.should_fallback is True
    assert classified.retryable is False


def test_fallback_on_status_config_includes_503(tmp_path):
    """Bridge resilience config must list 503 so the fallback ladder engages."""
    import sys

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from bridge_resilience import BridgeResilienceManager

    test_yaml = tmp_path / "hermes-routing.config.yaml"
    test_yaml.write_text(
        """
bridge_resilience:
  defaults:
    max_retries: 3
    skip_retry_on_status: [422, 404, 409]
  gemini-web-bridge:
    skip_retry_on_status: [422, 503, -32000]
  aipass-web-bridge:
    skip_retry_on_status: [404, 409, 503]
"""
    )

    mgr = BridgeResilienceManager(config_path=str(test_yaml))
    gemini_cfg = mgr.get_bridge_config("gemini-web-bridge")
    assert 503 in gemini_cfg.retry.skip_retry_on_status
    assert mgr.should_retry(gemini_cfg.retry, attempt=0, status_code=503, exc=None) is False

    aipass_cfg = mgr.get_bridge_config("aipass-web-bridge")
    assert 503 in aipass_cfg.retry.skip_retry_on_status
    assert mgr.should_retry(aipass_cfg.retry, attempt=0, status_code=503, exc=None) is False
