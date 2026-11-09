"""What a failed Bot API send means for a resend. The standalone text retry, the text plain-text
fallback and the album sender (gateway adapter and standalone) all read this one classifier, so
they cannot disagree on "may it already be in the chat".

PTB keys errors on class (``BadRequest`` subclasses ``NetworkError`` yet is permanent), so status
digits in a message never decide a resend. PTB words a timeout-class answer as such ("Timed out",
"Gateway Timeout (504)"), and a 5xx whose body is not Telegram's JSON (a fronting proxy's HTML
page) arrives as a ``NetworkError`` chained to the failed parse of that body.
"""

from __future__ import annotations

REFUSED = "refused"  # Telegram answered with a refusal (BadRequest, Forbidden, InvalidToken)
FLOOD = "flood"  # a flood wait: answered, nothing posted, ``retry_after`` says when
UNSENT = "unsent"  # connect/pool failure: the request never left
SERVER_ERROR = "server_error"  # Telegram's own 5xx JSON answer (Bad Gateway, Internal Server Error)
LOST = "lost"  # the request left and no readable answer came back (timeout, 504, reset, HTML 5xx)
LOCAL = "local"  # not a Telegram/transport error (raised before or outside the request)

# A request in one of these may already be in the chat, so it is never resent in another shape
# (per photo, plain text). On Telegram's own 5xx JSON answer (``SERVER_ERROR``, its policy since
# c928ebb1b1a) the standalone text send and both album lanes repeat the SAME request, with the same
# budget: ``SAME_REQUEST_ATTEMPTS`` tries, ``same_request_backoff(attempt)`` seconds apart.
MAY_HAVE_DELIVERED = frozenset({LOST, SERVER_ERROR})
SAME_REQUEST_ATTEMPTS = 3


def same_request_backoff(attempt: int) -> float:
    """Seconds before resending the same request after failed try ``attempt`` (0-based)."""
    return float(2 ** attempt)


def send_error_kind(error: Exception) -> str:
    """One of the module constants for a failed send; ``LOST`` when the adapter is unavailable."""
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter as tg, _iter_exception_graph
    except ImportError:
        return LOST
    if getattr(error, "retry_after", None) is not None:
        return FLOOD
    if tg._is_bad_request_error(error) or tg._looks_like_auth_error(error):
        return REFUSED
    if (tg._looks_like_connect_timeout(error) or tg._looks_like_pool_timeout(error)
            or tg._exception_graph_matches(error, "connecterror")):
        return UNSENT
    # A timeout (PTB TimedOut, the send deadline, a 504 Gateway Timeout) or a raw socket error that
    # may have failed mid-write.
    if tg._is_timed_out(error) or "timeout" in str(error).lower() or isinstance(error, OSError):
        return LOST
    if not tg._looks_like_network_error(error):
        return LOCAL
    # Chained: a transport error after the write, or a 5xx body that was not Telegram's JSON.
    return LOST if any(cause is not error for cause in _iter_exception_graph(error)) else SERVER_ERROR


def may_have_delivered(error: Exception) -> bool:
    """True when the request may already be in the chat: report it ambiguous, never resend it
    in another shape."""
    return send_error_kind(error) in MAY_HAVE_DELIVERED


def is_entity_parse_error(error: Exception, markers: tuple = ("parse", "markdown", "html", "entit")) -> bool:
    """A failure a plain-text resend fixes: Telegram refused the formatting (a ``BadRequest`` such
    as ``can't parse entities``), or a local error naming it that never reached Telegram. Never a
    transport/5xx error: PTB's "Parsing the server response ... failed" for an HTML 502/504 page
    may follow a posted message."""
    return any(marker in str(error).lower() for marker in markers) and send_error_kind(error) in (REFUSED, LOCAL)
