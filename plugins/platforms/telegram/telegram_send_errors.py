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


def maybe_delivered_result(error: Exception, safe_error: str, **fields):
    """The gateway ``SendResult`` for a failed send: one that may be in the chat (``LOST``) is
    ``delivery_uncertain`` and not retryable, so ``_send_with_retry`` neither resends it nor falls
    back to plain text; everything else stays retryable as before."""
    from gateway.platforms.base import SendResult

    if send_error_kind(error) == LOST:
        return SendResult(success=False, error=safe_error, retryable=False,
                          raw_response={"delivery_uncertain": True}, **fields)
    return SendResult(success=False, error=safe_error, retryable=True, **fields)


def rich_failure_result(error: Exception, safe_error: str, **fields):
    """``maybe_delivered_result`` for a failed ``sendRichMessage``, plus ``raw_response["never_sent"]``
    when the request never left or was a flood refusal: the standalone sender reads that proof as
    "nothing sent", so cron may queue a replay."""
    result = maybe_delivered_result(error, safe_error, **fields)
    kind = send_error_kind(error)
    if kind in (UNSENT, FLOOD):
        result.raw_response = {**(result.raw_response or {}), "never_sent": True,
                               **({"never_left": True} if kind == UNSENT else {})}
    return result


def idless_send_result(adapter_name: str, chat_id, what: str):
    """A send Telegram accepted whose result carries no ``message_id``: delivery is unknown, so it is
    ``delivery_uncertain`` like a lost reply (#147616). Never resent here, and never ``success``: a
    success without an id would finalize the delivery ledger row as delivered and prune it, so a reply
    that never landed is never redelivered. The ledger redelivers it once, with its duplicate marker."""
    import logging

    from gateway.platforms.base import SendResult

    logging.getLogger(__name__).warning("[%s] %s for chat %s returned no message_id", adapter_name, what, chat_id)
    return SendResult(success=False, error=f"{what} returned no message_id", raw_response={"delivery_uncertain": True})


def is_entity_parse_error(error: Exception, markers: tuple = ("parse", "markdown", "html", "entit")) -> bool:
    """A failure a plain-text resend fixes: Telegram refused the formatting (a ``BadRequest`` such
    as ``can't parse entities``), or a local error naming it that never reached Telegram. Never a
    transport/5xx error: PTB's "Parsing the server response ... failed" for an HTML 502/504 page
    may follow a posted message."""
    return any(marker in str(error).lower() for marker in markers) and send_error_kind(error) in (REFUSED, LOCAL)


def classify_send_exception(adapter, e: Exception):
    """The failed ``SendResult`` for an exception escaping ``adapter``'s chunk loop. ``retryable`` doubles
    as "non-delivery is certain": a plain ``TimedOut`` may have reached Telegram, so it is neither re-sent
    by ``_send_with_retry`` nor resumed from; a wrapped ConnectTimeout / httpx pool timeout never left."""
    from gateway.platforms.base import SendResult, classify_send_error
    from plugins.platforms.telegram.adapter import _redact_telegram_error_text, logger

    safe_error = _redact_telegram_error_text(e)
    logger.error("[%s] Failed to send Telegram message: %s", adapter.name, safe_error)
    err_str = str(e).lower()
    # Content exceeded 4096 chars: fail so the stream consumer enters fallback mode.
    if "message_too_long" in err_str or "too long" in err_str:
        logger.debug("[%s] send() content too long, falling back to new-message continuation", adapter.name)
        return SendResult(success=False, error="message_too_long", error_kind="too_long")
    return maybe_delivered_result(e, safe_error, error_kind=classify_send_error(e))


def explain_username_chat_not_found(result, chat_id):
    """The Bot API resolves ``@username`` only for public channels and supergroups, so a home channel
    set to a person's handle answers every send with a bare "Chat not found" (#95121). ``result`` is a
    standalone result dict (a new dict comes back) or a gateway ``SendResult`` (its error is amended)."""
    from plugins.platforms.telegram.telegram_ids import looks_like_telegram_username

    is_dict = isinstance(result, dict)
    error = result.get("error") if is_dict else getattr(result, "error", None)
    if not error or "chat not found" not in str(error).lower() or not looks_like_telegram_username(chat_id):
        return result
    hint = (f" Telegram resolves '{str(chat_id).strip()}' only when it is a public channel or supergroup, never "
            "a private chat. Use the numeric chat id instead (for the home channel: send /sethome in that "
            "chat, or set TELEGRAM_HOME_CHANNEL to the id).")
    error = str(error).rstrip(". ") + "." + hint
    if is_dict:
        return {**result, "error": error}
    result.error = error
    return result
