#!/usr/bin/env python3
"""Publish a health snapshot of the local Hermes AI stack.

This script probes a small, fixed set of *local* services that commonly sit
alongside a self-hosted Hermes deployment (the Hermes gateway and dashboard
plus auxiliary infra like n8n, Letta, Qdrant, LiteLLM, Langfuse, Promptfoo and
LangGraph), records a lightweight liveness result for each, and writes the
aggregate to ``~/.hermes/system_components_status.json``.

The dashboard's read-only ``GET /api/analytics/system-components`` endpoint
serves that file (annotating it with staleness). The endpoint never probes
services itself, so this publisher is the only thing that touches the network
— run it from cron (or by hand) on whatever cadence you like::

    python scripts/publish_system_components_status.py
    python scripts/publish_system_components_status.py --pretty

Design constraints:

* **No secrets.** Probes are plain unauthenticated ``GET`` requests; the
  snapshot records only name, endpoint, status, HTTP code, latency and a
  short error string. No headers, bodies or credentials are stored. The
  auxiliary-routes section reads only the ``provider``/``model`` *scalar*
  fields from ``config.yaml`` — never a whole config subtree.
* **Safe when services are absent.** A connection refusal or timeout is a
  normal, expected outcome and is recorded as ``absent``/``down`` — never an
  exception that aborts the run.
* **Stdlib only.** Uses ``urllib`` with short timeouts so it has no
  third-party dependencies and can run in any environment Hermes runs in.

Each component endpoint is overridable via an environment variable
(``HERMES_SYSCOMP_<NAME>_URL``) so non-default ports / remote hosts work
without editing this file.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Per-probe network timeout. Kept short: an unreachable local service should
# fail fast rather than stall the whole snapshot.
_PROBE_TIMEOUT_SECONDS = 2.0

# Components we know how to probe. ``health_path`` is appended to ``base`` for
# the probe; ``admin_url`` (if set) is surfaced in the dashboard as a link to
# the service's own UI. ``env`` names the override variable for the base URL.
#
# These are *local* infra services (self-hosted alongside Hermes), distinct
# from the cloud LLM providers in ``hermes_cli/doctor.py`` — different concern,
# so intentionally not shared with that table.
_COMPONENTS: List[Dict[str, Any]] = [
    {
        "name": "Hermes Gateway",
        "env": "HERMES_SYSCOMP_GATEWAY_URL",
        "base": "http://127.0.0.1:8642",
        "health_path": "/health",
    },
    {
        "name": "Hermes Dashboard",
        "env": "HERMES_SYSCOMP_DASHBOARD_URL",
        "base": "http://127.0.0.1:9119",
        "health_path": "/api/status",
    },
    {
        "name": "n8n",
        "env": "HERMES_SYSCOMP_N8N_URL",
        "base": "http://127.0.0.1:5678",
        "health_path": "/healthz",
        "admin_url": "http://127.0.0.1:5678",
    },
    {
        "name": "n8n-mcp",
        "env": "HERMES_SYSCOMP_N8N_MCP_URL",
        "base": "http://127.0.0.1:3000",
        "health_path": "/health",
    },
    {
        "name": "Letta",
        "env": "HERMES_SYSCOMP_LETTA_URL",
        "base": "http://127.0.0.1:8283",
        "health_path": "/v1/health/",
        "admin_url": "http://127.0.0.1:8283",
    },
    {
        "name": "Qdrant",
        "env": "HERMES_SYSCOMP_QDRANT_URL",
        "base": "http://127.0.0.1:6333",
        "health_path": "/",
        "admin_url": "http://127.0.0.1:6333/dashboard",
    },
    {
        "name": "LiteLLM",
        "env": "HERMES_SYSCOMP_LITELLM_URL",
        "base": "http://127.0.0.1:4000",
        "health_path": "/health/readiness",
        "admin_url": "http://127.0.0.1:4000/ui",
    },
    {
        "name": "Langfuse",
        "env": "HERMES_SYSCOMP_LANGFUSE_URL",
        "base": "http://127.0.0.1:3001",
        "health_path": "/api/public/health",
        "admin_url": "http://127.0.0.1:3001",
    },
    {
        "name": "Promptfoo",
        "env": "HERMES_SYSCOMP_PROMPTFOO_URL",
        "base": "http://127.0.0.1:3002",
        "health_path": "/",
        "admin_url": "http://127.0.0.1:3002",
    },
    {
        "name": "LangGraph",
        "env": "HERMES_SYSCOMP_LANGGRAPH_URL",
        "base": "http://127.0.0.1:8001",
        "health_path": "/health",
        "admin_url": "http://127.0.0.1:8001",
    },
]


def _hermes_home() -> Path:
    """Resolve the Hermes home dir without importing the heavy CLI package.

    Mirrors ``hermes_cli`` resolution (``HERMES_HOME`` override, else
    ``~/.hermes``) but stays dependency-free so the publisher runs anywhere.
    """
    raw = os.environ.get("HERMES_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _probe(component: Dict[str, Any]) -> Dict[str, Any]:
    """Probe one component with a short-timeout unauthenticated GET.

    Never raises: connection refused / timeout / DNS failure are recorded as
    ``absent`` (service not running) and any HTTP response — even an error
    code — is recorded as ``up``/``degraded`` since it proves the port is
    serving.
    """
    base = (os.environ.get(component["env"], "") or component["base"]).rstrip("/")
    url = base + component.get("health_path", "/")
    result: Dict[str, Any] = {
        "name": component["name"],
        "endpoint": url,
        "status": "down",
        "http_status": None,
        "latency_ms": None,
        "error": None,
    }
    if component.get("admin_url"):
        result["admin_url"] = component["admin_url"]

    started = time.monotonic()
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "hermes-syscomp/1"})
    try:
        with urllib.request.urlopen(req, timeout=_PROBE_TIMEOUT_SECONDS) as resp:
            code = resp.getcode()
            result["http_status"] = code
            result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
            result["status"] = "up" if 200 <= code < 400 else "degraded"
    except urllib.error.HTTPError as exc:
        # The port is serving and answered, just with a non-2xx/3xx status —
        # that's a live service, not an absent one.
        result["http_status"] = exc.code
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        result["status"] = "degraded"
        result["error"] = f"HTTP {exc.code}"
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        # Connection refused / no route => the service isn't running locally.
        text = str(reason)
        result["status"] = "absent"
        result["error"] = text[:200]
    except Exception as exc:  # pragma: no cover - defensive catch-all
        result["status"] = "down"
        result["error"] = str(exc)[:200]
    return result


def _load_auxiliary_routes() -> Optional[Dict[str, Any]]:
    """Read auxiliary-model cost-guardrail info from ``config.yaml``.

    Best-effort and secret-free: extracts only the scalar
    ``auxiliary.compression.provider`` / ``.model`` (and the same for the
    ``vision`` / ``web_extract`` tasks) — never a whole config subtree, so
    there is no path for credentials to land in the snapshot. Cost flagging
    (e.g. OpenRouter Fusion) is left to the dashboard's display layer.
    """
    config_path = _hermes_home() / "config.yaml"
    if not config_path.exists():
        return None
    try:
        import yaml  # local import: optional dependency, only needed here.

        with config_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    except Exception:
        return None
    if not isinstance(config, dict):
        return None

    auxiliary = config.get("auxiliary") or {}
    if not isinstance(auxiliary, dict):
        return None

    def _scalar(value: Any) -> Optional[str]:
        return value if isinstance(value, str) and value.strip() else None

    def _task(name: str) -> Dict[str, Optional[str]]:
        task_cfg = auxiliary.get(name) or {}
        if not isinstance(task_cfg, dict):
            task_cfg = {}
        return {
            "provider": _scalar(task_cfg.get("provider")),
            "model": _scalar(task_cfg.get("model")),
        }

    compression = _task("compression")
    return {
        "compression_provider": compression["provider"],
        "compression_model": compression["model"],
        "vision": _task("vision"),
        "web_extract": _task("web_extract"),
    }


def build_snapshot() -> Dict[str, Any]:
    """Probe every known component and assemble the snapshot payload."""
    components = [_probe(c) for c in _COMPONENTS]
    now = time.time()
    summary = {
        "total": len(components),
        "up": sum(1 for c in components if c["status"] == "up"),
        "degraded": sum(1 for c in components if c["status"] == "degraded"),
        "down": sum(1 for c in components if c["status"] == "down"),
        "absent": sum(1 for c in components if c["status"] == "absent"),
    }
    return {
        "timestamp": now,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "source": "publish_system_components_status.py",
        "summary": summary,
        "components": components,
        "auxiliary_routes": _load_auxiliary_routes(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: ~/.hermes/system_components_status.json).",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print the JSON output."
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the snapshot to stdout.",
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot()

    out_path = args.output or (_hermes_home() / "system_components_status.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    text = json.dumps(snapshot, indent=indent, sort_keys=False)
    # Write atomically-ish via a temp file so the dashboard never reads a
    # half-written snapshot.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(out_path)

    s = snapshot["summary"]
    print(
        f"Wrote {out_path} — {s['up']} up, {s['degraded']} degraded, "
        f"{s['absent']} absent, {s['down']} down (of {s['total']})."
    )
    if args.stdout:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
