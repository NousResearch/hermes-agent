"""Config + env resolver for ``hermes tunnel``.

Mirrors the env-over-config precedence of ``HERMES_DASHBOARD_PUBLIC_URL``
(see ``hermes_cli/dashboard_auth/prefix.py:resolve_public_url``): a
``HERMES_TUNNEL_*`` env var wins over the ``tunnel`` config block only when
its value is non-empty after strip.
"""

from __future__ import annotations

import os
import re
from typing import Optional

ORIGIN_SPEC_RE = re.compile(r"^(?P<sub>[A-Za-z0-9._-]+)=(?P<host>[A-Za-z0-9.\-]+):(?P<port>\d+)$")


def _env(name: str) -> Optional[str]:
    v = os.environ.get(name, "")
    return v.strip() or None


def _load_tunnel_section() -> dict:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return {}
    section = cfg.get("tunnel") if isinstance(cfg, dict) else None
    return section if isinstance(section, dict) else {}


def parse_origin(spec: str) -> dict:
    m = ORIGIN_SPEC_RE.match(spec.strip())
    if not m:
        raise ValueError(f"bad --origin spec (want sub=host:port): {spec!r}")
    return {"subdomain": m.group("sub"), "host": m.group("host"), "port": int(m.group("port"))}


def resolve_tunnel_config(cli_origins: Optional[list] = None) -> dict:
    sec = _load_tunnel_section()
    routes = [dict(r) for r in sec.get("routes", []) if isinstance(r, dict)]

    if cli_origins:
        by_sub = {r["subdomain"]: r for r in routes}
        for spec in cli_origins:
            parsed = parse_origin(spec)
            by_sub[parsed["subdomain"]] = parsed
        routes = list(by_sub.values())

    admin_env = _env("HERMES_TUNNEL_ADMIN")
    admin = [a.strip() for a in admin_env.split(",") if a.strip()] if admin_env else list(sec.get("admin", []))

    def _int_env(name, default, config_key):
        v = _env(name)
        if v is None:
            return int(sec.get(config_key, default))
        try:
            return int(v)
        except ValueError:
            return default

    return {
        "enabled": bool(sec.get("enabled", False)),
        "zone": _env("HERMES_TUNNEL_ZONE") or sec.get("zone", ""),
        "tunnel_name": _env("HERMES_TUNNEL_NAME") or sec.get("tunnel_name", ""),
        "credentials_file": _env("HERMES_TUNNEL_CREDS") or sec.get("credentials_file", ""),
        "metrics_port": _int_env("HERMES_TUNNEL_METRICS_PORT", 0, "metrics_port"),
        "idle_timeout_seconds": _int_env("HERMES_TUNNEL_IDLE_TIMEOUT", 1800, "idle_timeout_seconds"),
        "drain_seconds": _int_env("HERMES_TUNNEL_DRAIN_SECONDS", 15, "drain_seconds"),
        "poll_interval_seconds": _int_env("HERMES_TUNNEL_POLL_INTERVAL", 5, "poll_interval_seconds"),
        "admin": admin,
        "routes": routes,
    }