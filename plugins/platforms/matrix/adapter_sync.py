"""Matrix sync-loop helpers split out of adapter.py: which sync errors are permanent."""


# Auth errcodes that genuinely require re-authentication (never retried).
_MATRIX_PERMANENT_ERRCODES = frozenset({
    "m_unknown_token",
    "m_missing_token",
    "m_forbidden",
})


def _is_permanent_matrix_auth_error(exc: BaseException) -> bool:
    """Return True only for genuine auth failures that must stop the sync loop.

    A transient homeserver outage surfaces as a 5xx whose body may be an HTML
    error page (Umbrel's app-proxy returns one). Naive substring checks like
    ``"403" in str(exc)`` false-positive on digits embedded in that HTML (an SVG
    path coordinate such as ``1403.2`` contains ``403``) or in the ``since`` token
    echoed by a timeout message, which stopped the sync loop permanently on a
    passing blip. mautrix raises ``MatrixRequestError`` with ``errcode`` and
    ``http_status`` for every non-2xx, so classify on those alone; anything
    without a structured auth signal (timeouts, dropped connections, 5xx) is
    retried. Deliberately not ``.status``/``.status_code``/``.code``: those
    belong to unrelated exception shapes (aiohttp responses, OS errno) and can
    misclassify on a coincidental integer.
    """
    errcode = getattr(exc, "errcode", None)
    if isinstance(errcode, str) and errcode.strip().lower() in _MATRIX_PERMANENT_ERRCODES:
        return True
    status = getattr(exc, "http_status", None)
    return isinstance(status, int) and status in (401, 403)
