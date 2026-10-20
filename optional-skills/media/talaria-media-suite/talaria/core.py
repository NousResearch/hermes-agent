"""Shared plumbing for talaria: service registry, env-driven config
resolution, the HTTP layer (explicit timeouts, typed errors, secret
redaction), tick math, and text output formatting.

Service modules import NOTHING else from core beyond this module's public
names. Security invariants: stdlib-only urllib, per-request
timeout on every call, default TLS verification (never disabled), secrets
only ever read from environment variables, and no error message, warning,
or --json output may contain a key/token value or a key-bearing URL.
"""

import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from urllib import parse as urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__all__ = [
    "Service", "SERVICES", "TalariaError", "ConfigError", "AuthError",
    "TimeoutError", "ApiError", "NetworkError", "host", "port_for",
    "base_url", "headers", "api", "api_get", "ticks_mmss", "ticks_hours",
    "table", "approx_banner", "APPROX_BANNER", "output",
]


# --------------------------------------------------------------------------
# Service registry — the single table every module reads.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Service:
    name: str
    key_env: str
    port_env: str
    default_port: int
    api_base: str
    auth: str            # "x-api-key" | "mediabrowser"


SERVICES = {
    "sonarr":    Service("sonarr",    "SONARR_KEY",    "SONARR_PORT",    8989, "/api/v3", "x-api-key"),
    "radarr":    Service("radarr",    "RADARR_KEY",    "RADARR_PORT",    7878, "/api/v3", "x-api-key"),
    "lidarr":    Service("lidarr",    "LIDARR_KEY",    "LIDARR_PORT",    8686, "/api/v1", "x-api-key"),
    "prowlarr":  Service("prowlarr",  "PROWLARR_KEY",  "PROWLARR_PORT",  9696, "/api/v1", "x-api-key"),
    "bazarr":    Service("bazarr",    "BAZARR_KEY",    "BAZARR_PORT",    6767, "/api",    "x-api-key"),
    "overseerr": Service("overseerr", "OVERSEERR_KEY", "OVERSEERR_PORT", 5055, "/api/v1", "x-api-key"),
    "jellyfin":  Service("jellyfin",  "JELLYFIN_KEY",  "JELLYFIN_PORT",  8096, "",        "mediabrowser"),
}


# --------------------------------------------------------------------------
# Error taxonomy — all subclass TalariaError; caught ONLY in __main__.
# The redaction rule: no str(exc) may carry a key/token value.
# --------------------------------------------------------------------------

class TalariaError(Exception):
    """Base class for every talaria failure (exit code 1)."""


class ConfigError(TalariaError):
    """An environment variable is missing (names the var + setup hint)."""


class AuthError(TalariaError):
    """HTTP 401/403 — the service rejected the API key."""


class TimeoutError(TalariaError):
    """The request exceeded its explicit timeout."""


class ApiError(TalariaError):
    """Any other HTTP error status (server message included when JSON)."""


class NetworkError(TalariaError):
    """Connection refused / unreachable / DNS failure."""


# --------------------------------------------------------------------------
# Config resolution (port overrides).
# --------------------------------------------------------------------------

def host(override=None):
    """TALARIA_HOST if set, else ConfigError. Never sends a request."""
    if override:
        return override
    value = os.environ.get("TALARIA_HOST")
    if value:
        return value
    raise ConfigError("host not set — run: python3 -m talaria setup <host>")


def port_for(svc_name, override=None):
    svc = SERVICES[svc_name]
    if override:
        return int(override)
    raw = os.environ.get(svc.port_env)
    if raw:
        return int(raw)
    return svc.default_port


def base_url(svc_name, host_override=None):
    """http://{host}:{port} — https:// URLs would work unchanged: urllib
    verifies certificates by default and nothing here can disable it."""
    return "http://%s:%d" % (host(host_override), port_for(svc_name))


def headers(svc_name):
    """Auth headers. Keys NEVER ride URLs or query params."""
    svc = SERVICES[svc_name]
    key = os.environ.get(svc.key_env)
    if not key:
        raise ConfigError(
            "%s not set — run: python3 -m talaria setup" % svc.key_env)
    if svc.auth == "mediabrowser":
        # minimal bare form, the only future-proof channel on 12.x
        return {"Authorization": 'MediaBrowser Token="%s"' % key}
    return {"X-Api-Key": key}


# --------------------------------------------------------------------------
# HTTP layer — explicit timeout on EVERY call; typed errors; redaction.
# --------------------------------------------------------------------------

def _redact(text, svc_name):
    """Strip the service's key value if a server ever echoes it back."""
    key = os.environ.get(SERVICES[svc_name].key_env) or ""
    if key and key in text:
        text = text.replace(key, "***")
    return text


def api(svc_name, method, path, params=None, body=None, timeout=10.0,
        host_override=None, auth=True, root=False):
    """One HTTP call. Returns parsed JSON ({} for an empty body).

    - params: dict; list values encode as repeated keys (doseq)
    - body: JSON-encodable object (sent with Content-Type: application/json)
    - timeout: seconds, always passed through to urlopen
    - auth: False for the unauthenticated probes (/initialize.json,
      /System/Info/Public) — no key is required or attached
    - root: True when path is root-relative (the /initialize.json probe
      lives at the server root, outside every api base)
    """
    svc = SERVICES[svc_name]
    url = base_url(svc_name, host_override) + ("" if root else svc.api_base) \
        + path
    if params:
        url += "?" + urlparse.urlencode(params, doseq=True)
    hdrs = headers(svc_name) if auth else {}
    data = None
    if body is not None:
        hdrs = dict(hdrs)
        hdrs["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except HTTPError as exc:
        try:
            payload = exc.read()
            detail = json.loads(payload.decode("utf-8", "replace"))
            message = detail.get("message") or detail.get("error") or ""
        except Exception:
            message = ""
        message = _redact(str(message), svc_name)
        if exc.code in (401, 403):
            raise AuthError("%s rejected the API key" % svc.name) from None
        raise ApiError("%s HTTP %d%s" % (
            svc.name, exc.code, ": " + message if message else "")) from None
    except socket.timeout:
        raise TimeoutError(
            "timeout after %gs connecting to %s at %s" % (
                timeout, svc.name, base_url(svc_name, host_override))
        ) from None
    except URLError as exc:
        reason = exc.reason
        if isinstance(reason, socket.timeout):
            raise TimeoutError(
                "timeout after %gs connecting to %s at %s" % (
                    timeout, svc.name, base_url(svc_name, host_override))
            ) from None
        raise NetworkError(
            "cannot reach %s at %s (%s)" % (
                svc.name, base_url(svc_name, host_override),
                _redact(str(reason), svc_name))
        ) from None
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return {"raw": raw.decode("utf-8", "replace")[:2000]}


def api_get(svc_name, path, params=None, timeout=10.0, host_override=None,
            auth=True, retries=2):
    """Idempotent-GET wrapper: 2 retries (0.5 s / 1 s backoff) on transport
    failures (URLError/timeout) ONLY — never on HTTP 4xx/5xx."""
    attempt = 0
    while True:
        try:
            return api(svc_name, "GET", path, params=params, timeout=timeout,
                       host_override=host_override, auth=auth)
        except (NetworkError, TimeoutError):
            if attempt >= retries:
                raise
            time.sleep(0.5 * (2 ** attempt))
            attempt += 1


# --------------------------------------------------------------------------
# .NET ticks: 1 tick = 100 ns; seconds = ticks / 1e7.
# --------------------------------------------------------------------------

def ticks_seconds(ticks):
    return (ticks or 0) / 1e7


def ticks_mmss(ticks):
    total = int(round(ticks_seconds(ticks)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%d:%02d:%02d" % (h, m, s)
    return "%d:%02d" % (m, s)


def ticks_hours(ticks):
    return round(ticks_seconds(ticks) / 3600, 1)


# --------------------------------------------------------------------------
# Output formatting.
# --------------------------------------------------------------------------

APPROX_BANNER = ("≈ approximate — derived from item play-state data, "
                 "not session history")


def approx_banner():
    """Mandatory first line of every approximation sub's output (P8)."""
    return APPROX_BANNER


def output(args, payload, text, approx=False):
    """Emit one subcommand's result: --json prints the payload (approx subs
    carry "approx": true), otherwise the human text (approx subs are
    preceded by the banner line). honest-output rule."""
    if getattr(args, "json", False):
        if approx and isinstance(payload, dict):
            payload = dict(payload)
            payload["approx"] = True
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    if approx:
        print(APPROX_BANNER)
    if text:
        print(text)


def table(cols, rows):
    """Aligned text table: header row, dashed rule, then data rows."""
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(c)) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    out = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)),
           "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join(cell.ljust(widths[i])
                             for i, cell in enumerate(row)))
    return "\n".join(out)
