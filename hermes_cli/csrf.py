"""CSRF protection middleware for the Hermes dashboard.

Protects state-changing endpoints (POST/PUT/DELETE/PATCH) from cross-site
request forgery attacks by requiring a CSRF token that is bound to the
session token.

How it works
------------
1. When the dashboard HTML is served, a CSRF token is generated as
   ``HMAC-SHA256(session_token + "csrf")``.
2. The token is injected into the SPA via ``window.__HERMES_CSRF_TOKEN__``.
3. All state-changing API requests must include the token in the
   ``X-CSRF-Token`` header.
4. The middleware verifies the token before allowing the request through.

Why not double-submit cookie?
-----------------------------
The double-submit cookie pattern (sending CSRF token in both a cookie and
a header) is vulnerable when the cookie's ``Domain`` attribute is not
strictly set (subdomain takeover).  Using a header-only approach with a
token derived from the session token is simpler and equally secure for
same-site dashboard deployments.

Config
------
```yaml
security:
  csrf_enabled: true    # default: true
```
"""

import hashlib
import hmac
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_CSRF_HEADER_NAME = "X-CSRF-Token"
_CSRF_SALT = "hermes-csrf-v1"


def generate_csrf_token(session_token: str) -> str:
    """Generate a CSRF token bound to the session token.

    Parameters
    ----------
    session_token:
        The current session token (from web_server._SESSION_TOKEN).

    Returns
    -------
    str
        Hex-encoded CSRF token.
    """
    return hmac.new(
        session_token.encode("utf-8"),
        _CSRF_SALT.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_csrf_token(
    session_token: str,
    csrf_token: Optional[str],
    *,
    enabled: bool = True,
) -> bool:
    """Verify a CSRF token against the session token.

    Parameters
    ----------
    session_token:
        The current session token.
    csrf_token:
        The CSRF token from the request header.
    enabled:
        If False, skip verification (e.g. for development).

    Returns
    -------
    bool
        True if the token is valid, or if CSRF is disabled.
    """
    if not enabled:
        return True

    if not csrf_token:
        return False

    expected = generate_csrf_token(session_token)
    return hmac.compare_digest(csrf_token.encode("utf-8"), expected.encode("utf-8"))


def is_state_changing_method(method: str) -> bool:
    """Check if an HTTP method changes server state.

    Safe methods (GET, HEAD, OPTIONS) do not require CSRF protection.
    """
    return method.upper() not in {"GET", "HEAD", "OPTIONS"}


def extract_csrf_token(headers: dict[str, str]) -> Optional[str]:
    """Extract the CSRF token from request headers.

    Checks both the standard ``X-CSRF-Token`` header and the
    ``X-Hermes-CSRF-Token`` alias (for backward compatibility).
    """
    return (
        headers.get(_CSRF_HEADER_NAME)
        or headers.get("x-hermes-csrf-token")
    )


class CSRFError(Exception):
    """Raised when CSRF verification fails."""

    def __init__(self, detail: str = "Invalid or missing CSRF token"):
        self.detail = detail
        super().__init__(detail)
