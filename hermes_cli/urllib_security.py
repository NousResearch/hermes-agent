"""Security policy for credential-bearing stdlib urllib requests."""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any

# Headers safe to forward to a different origin. Everything else is dropped:
# custom provider headers routinely carry credentials under arbitrary names.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset({"accept", "user-agent"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def url_origin(url: str) -> tuple[str, str, int | None]:
    """Return a normalized (scheme, hostname, effective port) origin."""
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    return (
        scheme,
        (parsed.hostname or "").lower().rstrip("."),
        port or _DEFAULT_PORTS.get(scheme),
    )


class SafeCredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve request headers only while redirects stay on one origin."""

    def __init__(
        self,
        original_url: str,
        *,
        cross_origin_safe_headers: Iterable[str] = _CROSS_ORIGIN_SAFE_HEADERS,
    ) -> None:
        self._original_origin = url_origin(original_url)
        self._cross_origin_safe_headers = frozenset(
            str(name).lower() for name in cross_origin_safe_headers
        )

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Let urllib enforce status/method semantics first (notably 307/308).
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        resolved_url = urllib.parse.urljoin(req.full_url, newurl)
        if url_origin(resolved_url) != self._original_origin:
            # Use an allowlist rather than guessing credential header names.
            # normalize_extra_headers permits arbitrary secret-bearing names.
            for name, _value in list(redirected.header_items()):
                if name.lower() not in self._cross_origin_safe_headers:
                    redirected.remove_header(name)
        return redirected


def open_credentialed_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] = urllib.request.build_opener,
):
    """Open a request without forwarding credentials across origins.

    ``opener_factory`` is an explicit instrumentation/test seam. Security is
    never disabled based on global ``urlopen`` identity.
    """
    opener = opener_factory(SafeCredentialRedirectHandler(request.full_url))
    return opener.open(request, timeout=timeout)


__all__ = [
    "SafeCredentialRedirectHandler",
    "open_credentialed_url",
    "url_origin",
]
