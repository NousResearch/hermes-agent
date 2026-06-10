"""Guarded JSON HTTP client shared by the real call providers.

Port of OpenClaw's guarded-json-api: every request is pinned to the
provider's fixed API hostname (no config-driven base URLs that could
exfiltrate credentials), bounded in time and size, and errors never echo
credentials back into logs.
"""

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 15.0
_MAX_RESPONSE_BYTES = 1_000_000


class ProviderApiError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


async def guarded_json_request(
    method: str,
    url: str,
    *,
    allowed_host: str,
    headers: Optional[Dict[str, str]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    auth: Optional[tuple] = None,
    form_body: Optional[Dict[str, Any]] = None,
    allow_not_found: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    error_prefix: str = "provider API error",
) -> Optional[Dict[str, Any]]:
    """POST/GET JSON to a pinned provider host.

    Returns the parsed JSON object ({} for empty bodies), or ``None`` for a
    404 when ``allow_not_found`` is set. Raises :class:`ProviderApiError`
    otherwise.
    """
    host = urlparse(url).hostname or ""
    if host.lower() != allowed_host.lower():
        raise ProviderApiError(
            f"{error_prefix}: refusing request to host {host!r} "
            f"(allowed: {allowed_host})"
        )

    import httpx

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                data=form_body,
                auth=auth,
            )
    except httpx.HTTPError as e:
        raise ProviderApiError(f"{error_prefix}: {type(e).__name__}: {e}") from e

    if response.status_code == 404 and allow_not_found:
        return None
    if response.status_code >= 400:
        # Bounded body excerpt; carrier error payloads are not secret but
        # keep them short to avoid log spam.
        excerpt = response.text[:300]
        raise ProviderApiError(
            f"{error_prefix}: HTTP {response.status_code}: {excerpt}",
            status=response.status_code,
        )

    content = response.content[:_MAX_RESPONSE_BYTES]
    if not content.strip():
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ProviderApiError(f"{error_prefix}: invalid JSON response") from e
    return data if isinstance(data, dict) else {"data": data}
