"""HTTP client for the World Monitor REST API (koala73/worldmonitor)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    from hermes_cli.config import get_env_value
except Exception:  # pragma: no cover
    get_env_value = None  # type: ignore[assignment]

DEFAULT_CLOUD_BASE = "https://api.worldmonitor.app"
DEFAULT_LOCAL_PORT = 46123
DEFAULT_DEV_PORT = 3000
ENV_API_BASE = "WORLDMONITOR_API_BASE"
ENV_API_KEY = "WORLDMONITOR_API_KEY"
ENV_LOCAL_PORT = "WORLDMONITOR_LOCAL_PORT"
API_KEY_HEADER = "X-WorldMonitor-Key"


def _sidecar_base(port: int) -> str:
    return f"http://127.0.0.1:{port}".rstrip("/")


def _sidecar_port_from_env() -> int | None:
    raw = (
        os.environ.get(ENV_LOCAL_PORT, "").strip()
        or (get_env_value(ENV_LOCAL_PORT) if get_env_value else "") or ""
    )
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def resolve_api_base(*, prefer_sidecar: bool = True) -> str:
    """Return API base URL (explicit base, local sidecar, or cloud default)."""
    for candidate in (
        os.environ.get(ENV_API_BASE, "").strip(),
        (get_env_value(ENV_API_BASE) if get_env_value else "") or "",
    ):
        if candidate:
            return candidate.rstrip("/")

    port = _sidecar_port_from_env()
    if port:
        return _sidecar_base(port)

    if prefer_sidecar:
        try:
            from .auth_setup import probe_sidecar

            probe = probe_sidecar(DEFAULT_LOCAL_PORT)
            if probe.get("running"):
                return _sidecar_base(DEFAULT_LOCAL_PORT)
        except Exception:
            pass

        try:
            from .dev_server import probe_dev_server

            dev_probe = probe_dev_server(DEFAULT_DEV_PORT)
            if dev_probe.get("running"):
                return dev_probe["base_url"]
        except Exception:
            pass

    return DEFAULT_CLOUD_BASE


def resolve_api_key() -> str:
    for candidate in (
        os.environ.get(ENV_API_KEY, "").strip(),
        (get_env_value(ENV_API_KEY) if get_env_value else "") or "",
    ):
        if candidate:
            return candidate
    return ""


def connectivity_status() -> dict[str, Any]:
    base = resolve_api_base()
    key = resolve_api_key()
    return {
        "api_base": base,
        "api_key_configured": bool(key),
        "local_sidecar": base.startswith("http://127.0.0.1:") or base.startswith("http://localhost:"),
        "local_dev": ":3000" in base or base.endswith(f":{DEFAULT_DEV_PORT}"),
        "cloud_api": base.rstrip("/") == DEFAULT_CLOUD_BASE,
    }


def _request(path: str, params: dict[str, Any] | None = None, *, timeout: float = 45.0) -> dict[str, Any]:
    base = resolve_api_base()
    query = ""
    if params:
        filtered = {k: v for k, v in params.items() if v is not None and v != ""}
        if filtered:
            query = "?" + urllib.parse.urlencode(filtered, doseq=True)
    url = f"{base}{path}{query}"

    headers = {
        "Accept": "application/json",
        "User-Agent": "hermes-worldmonitor-osint/0.1",
    }
    api_key = resolve_api_key()
    if api_key:
        # wm_ keys must use X-WorldMonitor-Key only — Bearer wm_… fails OAuth resolution (401).
        headers[API_KEY_HEADER] = api_key

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        hint = None
        if exc.code == 401:
            hint = (
                "Set WORLDMONITOR_API_KEY in ~/.hermes/.env, or run the World Monitor "
                "local sidecar (port 46123) and set WORLDMONITOR_API_BASE=http://127.0.0.1:46123"
            )
        raise RuntimeError(
            json.dumps(
                {
                    "success": False,
                    "http_status": exc.code,
                    "url": url,
                    "error": detail or exc.reason,
                    "hint": hint,
                },
                ensure_ascii=False,
            )
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            json.dumps(
                {
                    "success": False,
                    "url": url,
                    "error": str(exc.reason or exc),
                    "hint": "Check network, API base URL, and whether the local sidecar is running.",
                },
                ensure_ascii=False,
            )
        ) from exc

    if not body.strip():
        return {}
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from World Monitor API: {exc}") from exc
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def get_risk_scores(region: str = "") -> dict[str, Any]:
    params = {"region": region} if region else None
    return _request("/api/intelligence/v1/get-risk-scores", params)


def get_country_risk(country_code: str) -> dict[str, Any]:
    return _request(
        "/api/intelligence/v1/get-country-risk",
        {"country_code": country_code.upper()},
    )


def get_country_intel_brief(country_code: str, framework: str = "") -> dict[str, Any]:
    params: dict[str, Any] = {"country_code": country_code.upper()}
    if framework:
        params["framework"] = framework[:2000]
    return _request("/api/intelligence/v1/get-country-intel-brief", params)


def get_regional_brief(region_id: str) -> dict[str, Any]:
    return _request(
        "/api/intelligence/v1/get-regional-brief",
        {"region_id": region_id},
    )


def list_feed_digest(*, variant: str = "full", lang: str = "en") -> dict[str, Any]:
    return _request(
        "/api/news/v1/list-feed-digest",
        {"variant": variant, "lang": lang},
    )


def snapshot_japan_security(*, news_lang: str = "en", news_limit: int = 12) -> dict[str, Any]:
    """Aggregate JP-focused World Monitor feeds for OSINT fusion."""
    from . import free_web

    conn = connectivity_status()
    has_paid = bool(conn.get("api_key_configured") or conn.get("local_sidecar") or conn.get("local_dev"))

    if not has_paid:
        free = free_web.free_snapshot(
            focus="japan_security",
            news_lang=news_lang,
            news_limit=news_limit,
            include_shell=False,
        )
        free["api"] = conn
        free["tier_mode"] = "free_web"
        return free

    out: dict[str, Any] = {
        "success": True,
        "focus": "japan_security",
        "tier_mode": "pro_or_sidecar",
        "api": conn,
        "sections": {},
        "errors": [],
    }

    fetches = (
        ("country_risk_jp", lambda: get_country_risk("JP")),
        ("country_intel_brief_jp", lambda: get_country_intel_brief("JP")),
        ("regional_brief_east_asia", lambda: get_regional_brief("east-asia")),
        ("risk_scores", lambda: get_risk_scores("east-asia")),
        ("news_digest", lambda: list_feed_digest(variant="full", lang=news_lang)),
    )
    for key, fn in fetches:
        try:
            out["sections"][key] = fn()
        except Exception as exc:  # pragma: no cover - network dependent
            out["errors"].append({"section": key, "error": str(exc)})

    # Backfill news / public feeds from Free web when paid sections fail.
    if out["errors"]:
        try:
            free = free_web.free_snapshot(
                focus="japan_security",
                news_lang=news_lang,
                news_limit=news_limit,
                include_shell=False,
            )
            for key, val in (free.get("sections") or {}).items():
                out["sections"].setdefault(key, val)
            if free.get("news_headlines"):
                out["news_headlines"] = free["news_headlines"]
            out["free_web_backfill"] = True
        except Exception as exc:
            out["errors"].append({"section": "free_web_backfill", "error": str(exc)})

    digest = out["sections"].get("news_digest") or {}
    items = digest.get("items") or digest.get("feeds") or digest.get("articles") or []
    if isinstance(items, list) and news_limit > 0:
        out["news_headlines"] = items[:news_limit]
    elif not out.get("news_headlines") and isinstance(digest.get("categories"), dict):
        out["news_headlines"] = free_web._news_headlines(
            digest, limit=news_limit, focus="japan_security"
        )

    out["success"] = len(out["sections"]) > 0
    return out
