"""Regression tests for the stale-Copilot-credential 400 self-heal.

A long-lived process (e.g. the Hermes Council daemon, or any multi-hour agent)
holds a cached Copilot+Claude client whose bearer token's entitlement later
rotates. Copilot then returns ``400 model_not_supported`` instead of a clean
401. Before the fix, ``_is_auth_error`` only matched 401, so the stale-credential
400 bypassed the refresh+evict+retry path entirely and the cached client looped
forever. These tests pin the corrected classification.
"""

from agent.auxiliary_client import (
    _is_auth_error,
    _is_stale_copilot_credential_error,
    _normalize_aux_provider,
)


class _FakeAPIError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


_COPILOT_400 = (
    "Error code: 400 - {'error': {'message': 'The requested model is not "
    "supported.', 'code': 'model_not_supported', 'param': 'model', 'type': "
    "'invalid_request_error'}}"
)
_CODEX_401 = (
    "Error code: 401 - {'error': {'message': 'Your authentication token has "
    "been invalidated.', 'code': 'token_invalidated'}}"
)


def test_stale_copilot_400_is_detected():
    exc = _FakeAPIError(_COPILOT_400, status_code=400)
    assert _is_stale_copilot_credential_error(exc) is True


def test_clean_401_is_not_a_stale_copilot_400():
    exc = _FakeAPIError(_CODEX_401, status_code=401)
    assert _is_stale_copilot_credential_error(exc) is False


def test_unrelated_400_is_not_stale_credential():
    exc = _FakeAPIError(
        "Error code: 400 - {'error': {'message': 'temperature must be <= 2'}}",
        status_code=400,
    )
    assert _is_stale_copilot_credential_error(exc) is False


def _refresh_gate(exc, provider):
    """Mirror the exact gate logic in call_llm's except chain."""
    norm = _normalize_aux_provider(provider)
    refreshable = _is_auth_error(exc) or (
        norm == "copilot" and _is_stale_copilot_credential_error(exc)
    )
    return refreshable and provider not in {"auto", "", None}


def test_refresh_gate_fires_for_copilot_400():
    # The bug fix: copilot stale-credential 400 is now refreshable.
    assert _refresh_gate(_FakeAPIError(_COPILOT_400, status_code=400), "copilot") is True


def test_refresh_gate_still_fires_for_401():
    # No regression: a real 401 still triggers refresh.
    assert _refresh_gate(_FakeAPIError(_CODEX_401, status_code=401), "copilot") is True


def test_refresh_gate_does_not_fire_for_400_on_other_provider():
    # A genuinely-wrong-model 400 on a non-copilot provider must NOT loop-refresh.
    assert _refresh_gate(_FakeAPIError(_COPILOT_400, status_code=400), "openrouter") is False


def test_refresh_gate_excludes_auto_provider():
    assert _refresh_gate(_FakeAPIError(_COPILOT_400, status_code=400), "auto") is False
