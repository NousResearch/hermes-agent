"""Fixed, separately authorized read-only cron execution receipt projection."""

from __future__ import annotations

import hmac
import re


_JOB_ID = re.compile(r"[a-f0-9]{12,64}\Z")
_EXECUTION_ID = re.compile(r"[a-f0-9]{32}\Z")
_NO_STORE_HEADERS = {"Cache-Control": "no-store"}


def _not_available(web):
    """Do not disclose whether receipt observation is configured or a record exists."""
    return web.json_response(
        {"error": {"message": "Not found", "code": "cron_receipt_not_found"}},
        status=404,
        headers=_NO_STORE_HEADERS,
    )


def _authorize(adapter, request, *, web):
    """Accept only the dedicated observer secret, never the general API key."""
    from hermes_cli.auth import has_usable_secret

    expected = adapter._cron_receipt_observer_key
    primary = adapter._expected_api_key()
    if not has_usable_secret(expected, min_length=16):
        return _not_available(web)
    expected = expected.strip()
    try:
        # Reused general API keys retain broad authority, so fail closed rather than silently
        # treating one as a read-only observer credential.
        primary = str(primary or "").strip()
        if primary and hmac.compare_digest(
            expected.encode("utf-8"), primary.encode("utf-8")
        ):
            return _not_available(web)
        headers = request.headers.getall("Authorization", [])
        if len(headers) != 1:
            return web.json_response(
                {
                    "error": {
                        "message": "Unauthorized",
                        "code": "cron_receipt_observer_auth_failed",
                    }
                },
                status=401,
                headers=_NO_STORE_HEADERS,
            )
        header = headers[0]
        supplied = header[7:].strip() if header.startswith("Bearer ") else ""
        if supplied and hmac.compare_digest(
            supplied.encode("utf-8"), expected.encode("utf-8")
        ):
            return None
    except UnicodeError:
        pass
    return web.json_response(
        {
            "error": {
                "message": "Unauthorized",
                "code": "cron_receipt_observer_auth_failed",
            }
        },
        status=401,
        headers=_NO_STORE_HEADERS,
    )


async def handle(adapter, request, *, web):
    """Return one immutable, content-free receipt after dedicated observer authentication."""
    denied = _authorize(adapter, request, web=web)
    if denied is not None:
        return denied
    job_id = request.match_info.get("job_id", "")
    execution_id = request.match_info.get("execution_id", "")
    if not (_JOB_ID.fullmatch(job_id) and _EXECUTION_ID.fullmatch(execution_id)):
        return _not_available(web)
    try:
        from cron.executions import get_public_execution_receipt

        receipt = get_public_execution_receipt(job_id, execution_id)
    except Exception:
        return _not_available(web)
    if receipt is None:
        return _not_available(web)
    return web.json_response(receipt, headers=_NO_STORE_HEADERS)
