"""Tests for _require_token() gated-mode bypass (issue #42139).

In gated dashboard mode (non-loopback bind with auth), cookie auth is
authoritative and the legacy _SESSION_TOKEN is never issued.  The
_require_token() function must trust the gate and not reject
cookie-authenticated clients with 401.
"""

import pytest
from unittest.mock import patch

from fastapi import HTTPException


def _make_request(auth_required: bool = False):
    """Build a minimal stand-in that _require_token can inspect."""
    class _State:
        pass

    class _Req:
        pass

    req = _Req()
    req.app = type("App", (), {"state": _State()})()
    req.app.state.auth_required = auth_required
    return req


# ── Non-gated mode (legacy _SESSION_TOKEN path) ──────────────────────


def test_require_token_rejects_without_token_in_normal_mode():
    """Non-gated mode with no token → 401."""
    from hermes_cli.web_server import _require_token

    req = _make_request(auth_required=False)
    with patch("hermes_cli.web_server._has_valid_session_token", return_value=False):
        with pytest.raises(HTTPException) as exc_info:
            _require_token(req)
    assert exc_info.value.status_code == 401


def test_require_token_passes_with_valid_token_in_normal_mode():
    """Non-gated mode with valid session token → passes."""
    from hermes_cli.web_server import _require_token

    req = _make_request(auth_required=False)
    with patch("hermes_cli.web_server._has_valid_session_token", return_value=True):
        _require_token(req)  # should not raise


# ── Gated mode (cookie auth is authoritative) ────────────────────────


def test_require_token_passes_without_token_in_gated_mode():
    """Gated mode (auth_required=True) passes even without session token.

    In gated mode the legacy _SESSION_TOKEN is never issued — cookie auth
    is authoritative.  _require_token must trust the gate.  (issue #42139)
    """
    from hermes_cli.web_server import _require_token

    req = _make_request(auth_required=True)
    # No mock needed — the function should return before checking the token
    _require_token(req)


def test_require_token_passes_with_token_in_gated_mode():
    """Gated mode with a token also passes (belt-and-suspenders)."""
    from hermes_cli.web_server import _require_token

    req = _make_request(auth_required=True)
    with patch("hermes_cli.web_server._has_valid_session_token", return_value=True):
        _require_token(req)
