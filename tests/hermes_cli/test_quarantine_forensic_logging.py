"""Redaction-safe forensic logging at the Nous OAuth quarantine path.

A NAS-hosted Fly agent's Nous bootstrap session can take a terminal
``invalid_grant`` and get quarantined (dead tokens cleared from auth.json).
Historically this was completely silent — no WARNING+ record at the terminal
rejection, only a downstream "No access token found" warning once the pool was
already empty. The Fly log drain is WARNING-only, so nothing about the terminal
death reached centralized logging. These tests lock in that
``_quarantine_nous_oauth_state`` now emits a WARNING+ forensic record, and — the
load-bearing assertion — that the raw refresh token never appears in that output.
"""

import hashlib
import logging

from hermes_cli.auth import AuthError, _quarantine_nous_oauth_state


# A distinctive, obviously-fake refresh token so the redaction assertion is
# unambiguous if it ever leaks.
_FAKE_RT = "nous_rt_LEAK_CANARY_do_not_log_raw_0123456789abcdef"
_EXPECTED_FP = hashlib.sha256(_FAKE_RT.encode("utf-8")).hexdigest()[:12]


def _make_state(**overrides):
    state = {
        "portal_base_url": "https://portal.example.com",
        "client_id": "test-client-id",
        "access_token": "nous_at_SECRET_access_token_material",
        "refresh_token": _FAKE_RT,
        "agent_key": "nous_agent_key_SECRET_material",
        "agent_key_id": "ak-12345",
        "expires_at": "2020-01-01T00:00:00+00:00",  # in the past
        "obtained_at": "2019-12-31T00:00:00+00:00",
    }
    state.update(overrides)
    return state


def _error():
    return AuthError(
        "invalid_grant: token expired or revoked",
        provider="nous",
        code="invalid_grant",
        relogin_required=True,
    )


def test_quarantine_emits_warning(caplog):
    state = _make_state()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        _quarantine_nous_oauth_state(state, _error(), reason="unit_test_quarantine")

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected at least one WARNING+ record from quarantine"
    assert any("quarantined" in r.getMessage() for r in warnings)


def test_warning_contains_hash_prefix_and_error_code(caplog):
    state = _make_state()
    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        _quarantine_nous_oauth_state(state, _error(), reason="unit_test_quarantine")

    text = caplog.text
    assert _EXPECTED_FP in text, (
        f"expected refresh-token hash prefix {_EXPECTED_FP} in log output"
    )
    assert "invalid_grant" in text, "expected error.code in log output"
    assert "unit_test_quarantine" in text, "expected reason in log output"


def test_raw_refresh_token_never_logged(caplog):
    """Load-bearing redaction-safety test: the raw secret must never appear."""
    state = _make_state()
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.auth"):
        _quarantine_nous_oauth_state(state, _error(), reason="unit_test_quarantine")

    text = caplog.text
    assert _FAKE_RT not in text, "RAW refresh token leaked into log output!"
    # Belt-and-suspenders: the access token and agent key must not leak either.
    assert "nous_at_SECRET_access_token_material" not in text
    assert "nous_agent_key_SECRET_material" not in text


def test_quarantine_no_refresh_token_does_not_throw(caplog):
    state = _make_state()
    state.pop("refresh_token", None)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
        # Must not raise even when there is no refresh token to fingerprint.
        _quarantine_nous_oauth_state(state, _error(), reason="unit_test_no_rt")

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING even when refresh_token is absent"
    # Fingerprint should be null/None, and definitely not the canary prefix.
    assert _EXPECTED_FP not in caplog.text


def test_quarantine_clears_token_material():
    """Regression guard: the quarantine still clears dead token keys."""
    state = _make_state()
    _quarantine_nous_oauth_state(state, _error(), reason="unit_test_quarantine")
    for key in ("access_token", "refresh_token", "agent_key", "agent_key_id", "expires_at"):
        assert key not in state, f"{key} should have been cleared by quarantine"
    assert state["last_auth_error"]["code"] == "invalid_grant"
