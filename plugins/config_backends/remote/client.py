"""HTTP client for the agent routes of the config plane (contract.md §7.1, §7.2).

Plain HTTP (D9), stdlib only: it runs inside ``load_hermes_dotenv`` before most of Hermes is
imported. Every request re-resolves the bearer (contract §10.2).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .credentials import plane_token

DEFAULT_URL = "https://config-config.nousresearch.com"
URL_ENV = "HERMES_CONFIG_REMOTE_URL"
INSTANCE_ENV = "HERMES_CONFIG_INSTANCE_ID"
_HTTP_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class Response:
    status: int
    body: Dict[str, Any]
    etag: Optional[str]
    retry_after: Optional[float]

    @property
    def error(self) -> str:
        return str(self.body.get("error") or f"http_{self.status}")

    @property
    def message(self) -> str:
        return str(self.body.get("message") or f"HTTP {self.status}")


class TransportError(RuntimeError):
    """The request never produced an HTTP response (DNS, connect, TLS, timeout)."""


def base_url() -> str:
    return (os.environ.get(URL_ENV, "").strip() or DEFAULT_URL).rstrip("/")


def instance_id() -> str:
    return os.environ.get(INSTANCE_ENV, "").strip()


def _parse_body(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _retry_after(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def request(method: str, home: Path, profile: str, *, etag: Optional[str] = None,
            body: Optional[Dict[str, Any]] = None) -> Response:
    """One call to ``/v1/config/self?profile=<profile>`` as ``home``'s agent."""
    url = f"{base_url()}/v1/config/self?{urllib.parse.urlencode({'profile': profile})}"
    headers = {
        "Authorization": f"Bearer {plane_token(home)}",
        "X-Hermes-Instance-Id": instance_id(),
        "Accept": "application/json",
    }
    data = None
    if body is not None:
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if etag:
        headers["If-None-Match"] = etag
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
            return Response(resp.status, _parse_body(resp.read()), resp.headers.get("ETag"),
                            _retry_after(resp.headers.get("Retry-After")))
    except urllib.error.HTTPError as exc:  # every non-2xx, including 304
        with exc:
            raw = exc.read() if exc.fp is not None else b""
        return Response(exc.code, _parse_body(raw), exc.headers.get("ETag") if exc.headers else None,
                        _retry_after(exc.headers.get("Retry-After") if exc.headers else None))
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise TransportError(f"{method} {url}: {getattr(exc, 'reason', exc)}") from exc
