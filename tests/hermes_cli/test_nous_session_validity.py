"""Tests for get_nous_session_validity — the /api/status classifier NAS reads
to decide whether to re-mint a hosted-agent bootstrap session.

The anti-flap contract is the load-bearing property: only a *terminal* auth
failure may report "terminal" (a spurious "terminal" triggers an unnecessary
NAS re-mint + machine restart on a healthy box). A mid-rotation blip, a
transient error, or a merely-expiring token must NOT report "terminal".
"""

import hermes_cli.auth as auth
from hermes_cli.auth import (
    NOUS_SESSION_TERMINAL,
    NOUS_SESSION_UNKNOWN,
    NOUS_SESSION_VALID,
    get_nous_session_validity,
)


def _clear_cache():
    auth.invalidate_nous_auth_status_cache()


def test_valid_when_logged_in(monkeypatch):
    """A healthy login → 'valid'."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {
        "access_token": "at", "refresh_token": "rt",
    })
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {"logged_in": True})
    assert get_nous_session_validity() == NOUS_SESSION_VALID


def test_terminal_on_persisted_quarantine_marker(monkeypatch):
    """A persisted last_auth_error.relogin_required with tokens cleared →
    'terminal'. This is the exact on-disk state the incident produced."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {
        # tokens cleared by the quarantine path
        "last_auth_error": {"relogin_required": True, "code": "invalid_grant"},
    })
    # status would also say not-logged-in, but the marker short-circuits first
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {"logged_in": False})
    assert get_nous_session_validity() == NOUS_SESSION_TERMINAL


def test_terminal_on_relogin_required_status(monkeypatch):
    """Not logged in + relogin_required from the live status → 'terminal'."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {
        "refresh_token": "rt",  # present, but status resolution fails terminally
    })
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {
        "logged_in": False, "relogin_required": True, "error_code": "invalid_grant",
    })
    assert get_nous_session_validity() == NOUS_SESSION_TERMINAL


def test_unknown_when_no_provider_state(monkeypatch):
    """No Nous provider state at all → 'unknown' (never terminal)."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: None)
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {"logged_in": False})
    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_anti_flap_transient_not_logged_in_is_unknown(monkeypatch):
    """ANTI-FLAP: not-logged-in WITHOUT relogin_required (a transient/network
    blip) must be 'unknown', NOT 'terminal' — otherwise a healthy box mid-blip
    triggers a spurious re-mint."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {
        "access_token": "at", "refresh_token": "rt",
    })
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {
        "logged_in": False, "error": "connection reset",  # no relogin_required
    })
    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_stale_quarantine_marker_ignored_after_relogin(monkeypatch):
    """ANTI-FLAP: a leftover last_auth_error marker must NOT report 'terminal'
    once a subsequent login has repopulated tokens."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {
        "access_token": "new-at", "refresh_token": "new-rt",
        "last_auth_error": {"relogin_required": True, "code": "invalid_grant"},
    })
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {"logged_in": True})
    assert get_nous_session_validity() == NOUS_SESSION_VALID


def test_status_exception_is_unknown_not_terminal(monkeypatch):
    """If status computation itself throws, that's indeterminate → 'unknown'."""
    monkeypatch.setattr(auth, "get_provider_auth_state", lambda p: {"refresh_token": "rt"})

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(auth, "get_nous_auth_status", _boom)
    assert get_nous_session_validity() == NOUS_SESSION_UNKNOWN


def test_provider_state_exception_falls_through_to_status(monkeypatch):
    """If reading provider state throws, fall through to status (don't crash)."""
    def _boom(p):
        raise RuntimeError("disk error")

    monkeypatch.setattr(auth, "get_provider_auth_state", _boom)
    monkeypatch.setattr(auth, "get_nous_auth_status", lambda: {"logged_in": True})
    assert get_nous_session_validity() == NOUS_SESSION_VALID
