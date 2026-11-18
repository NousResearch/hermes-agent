"""One report on the wire: a POST over a fresh connection, and the reply reduced to an outcome."""

from __future__ import annotations

import http.client
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Optional

from gateway.status_report.target import ReportTarget

logger = logging.getLogger(__name__)

OutcomeKind = Literal["stored", "throttled", "rejected", "failed", "refused"]
# urllib's default names the interpreter and its version, which some edge filters refuse.
_USER_AGENT = "hermes-gateway-status-report/1"
_THROTTLED = frozenset({429, 503})
_REJECTED = frozenset({401, 403})


@dataclass(frozen=True)
class Outcome:
    """How one report went. ``status`` is None when no HTTP reply came back (network error, timeout);
    ``retry_after_s`` is the receiver's ``Retry-After`` and only ever set on ``throttled``."""

    kind: OutcomeKind
    status: Optional[int] = None
    retry_after_s: Optional[float] = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect: urllib would repeat the ``Authorization`` header to wherever it points."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_report(target: ReportTarget, body: bytes, *, timeout: float, opener: Any = None) -> Outcome:
    """POST *body* to *target* and classify the reply. Blocking; never raises for a network or HTTP
    failure, and never logs the URL, the credential or an exception's text (urllib quotes both).

    Every call opens its own connection (urllib keeps none), so no socket is reused across a
    machine suspend. The reply body is not read: nothing in it changes what the sender does."""
    request = urllib.request.Request(target.url, data=body, method="POST", headers={
        "Authorization": f"Bearer {target.credential}", "Content-Type": "application/json",
        "User-Agent": _USER_AGENT})
    open_url = opener or urllib.request.build_opener(_NoRedirect).open
    try:
        with open_url(request, timeout=timeout) as response:
            status, headers = int(response.status), response.headers
    except urllib.error.HTTPError as exc:
        status, headers = exc.code, exc.headers
        exc.close()
    except (http.client.HTTPException, OSError, ValueError) as exc:
        logger.debug("status report: no reply (%s)", type(exc).__name__)
        return Outcome("failed")
    if 200 <= status < 300:
        return Outcome("stored", status)
    if status in _THROTTLED:
        return Outcome("throttled", status, _retry_after_s(headers.get("Retry-After")))
    if status in _REJECTED:
        return Outcome("rejected", status)
    return Outcome("failed" if status >= 500 else "refused", status)


def _retry_after_s(value: Optional[str]) -> Optional[float]:
    """``Retry-After`` as a positive number of seconds. Its HTTP-date form, zero and anything else
    are None: the sender then keeps its normal interval. Read with ``float()``, never ``int()``: a
    digit string past the interpreter's integer limit would raise, where ``float()`` reads it as
    infinity and the sender's cap applies."""
    text = (value or "").strip()
    if not (text.isascii() and text.isdigit()):
        return None
    seconds = float(text)
    return seconds if seconds > 0 else None
