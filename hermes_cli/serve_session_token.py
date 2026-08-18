"""Loopback session-token client for ``hermes serve --print-session-token``.

Desktop SSH attach uses this instead of spawning an isolated ``--port 0``
serve: probe the machine-level loopback backend (default 127.0.0.1:9119),
print the process-local session token, and exit. The token is published only
on loopback / ungated ``/api/status`` — gated binds never include it.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Callable
from urllib.request import Request


class ServeTokenError(ValueError):
    """Typed failure so Desktop can distinguish missing / gated / old serves."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def probe_host(host: str | None) -> str:
    """Map wildcard binds to a loopback address suitable for local probing."""
    normalized = (host or "127.0.0.1").strip().strip("[]")
    if normalized in {"", "0.0.0.0", "::"}:
        return "127.0.0.1"
    return normalized


def fetch_loopback_session_token(
    host: str | None,
    port: int,
    *,
    timeout: float = 3.0,
    urlopen: Callable[..., object] | None = None,
) -> str:
    """Return the loopback session token published by a running serve.

    Raises :class:`ServeTokenError` with ``kind``:

    * ``missing`` — nothing accepted TCP on host:port
    * ``gated`` — the bind requires OAuth/password
    * ``old`` — serve is up but did not publish ``session_token``
    * ``error`` — transport / parse failure
    """
    target = probe_host(host)
    url = f"http://{target}:{int(port)}/api/status"
    opener = urlopen or urllib.request.urlopen
    request = Request(url, method="GET")
    try:
        with opener(request, timeout=timeout) as resp:  # type: ignore[arg-type]
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ServeTokenError("missing", f"No hermes serve/dashboard is listening on {target}:{int(port)}.") from exc

    try:
        body = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ServeTokenError("error", f"Could not parse {url}: {exc}") from exc

    if not isinstance(body, dict):
        raise ServeTokenError("error", f"{url} did not return a JSON object.")

    if body.get("auth_required"):
        raise ServeTokenError(
            "gated",
            "This serve requires sign-in; loopback token attach is not available.",
        )

    token = body.get("session_token")
    if not isinstance(token, str) or not token.strip():
        raise ServeTokenError(
            "old",
            "Serve is running but did not publish a loopback session token. "
            "Update Hermes on the remote host.",
        )

    return token.strip()


def print_session_token(host: str | None, port: int) -> int:
    """CLI entry: print the token to stdout or an error to stderr."""
    try:
        print(fetch_loopback_session_token(host, port))
        return 0
    except ServeTokenError as exc:
        print(str(exc), file=sys.stderr)
        return 1
