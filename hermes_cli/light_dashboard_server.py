"""Lightweight stdlib dashboard backend.

This module is intentionally independent from ``hermes_cli.web_server``.  The
full dashboard imports FastAPI, Pydantic schemas, route tables, plugin APIs, and
the React bundle machinery at module import time.  ``hermes dashboard --light``
uses this server when operators only need status plus recent sessions on
memory-constrained hosts.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

from hermes_cli import __release_date__, __version__

_log = logging.getLogger(__name__)

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_HEAVY_SESSION_FIELDS = ("system_prompt", "model_config")


def _is_loopback_bind(host: str) -> bool:
    return (host or "127.0.0.1").strip().lower() in _LOOPBACK_HOSTS


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _coerce_int(value: str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _open_session_db_read_only():
    from hermes_state import DEFAULT_DB_PATH, SessionDB

    if not Path(DEFAULT_DB_PATH).exists():
        return None
    return SessionDB(read_only=True)


def _strip_session_list_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        for key in _HEAVY_SESSION_FIELDS:
            row.pop(key, None)
    return rows


def _active_session_count() -> int:
    db = _open_session_db_read_only()
    if db is None:
        return 0
    try:
        now = time.time()
        rows = db.list_sessions_rich(limit=50, compact_rows=True)
        return sum(
            1
            for row in rows
            if row.get("ended_at") is None
            and (now - row.get("last_active", row.get("started_at", 0))) < 300
        )
    except Exception as exc:
        _log.debug("light dashboard active-session count unavailable: %s", exc)
        return 0
    finally:
        db.close()


def build_status_payload() -> dict[str, Any]:
    """Return the lightweight status payload without importing the full server."""
    from gateway.status import (
        derive_gateway_busy,
        derive_gateway_drainable,
        get_running_pid_cached,
        parse_active_agents,
        read_runtime_status,
    )

    runtime = read_runtime_status() or {}
    gateway_pid = get_running_pid_cached()
    gateway_running = gateway_pid is not None
    gateway_state = runtime.get("gateway_state")
    if not gateway_running:
        gateway_state = gateway_state if gateway_state == "startup_failed" else "stopped"

    active_agents = parse_active_agents(runtime.get("active_agents", 0))
    return {
        "mode": "lightweight",
        "version": __version__,
        "release_date": __release_date__,
        "gateway_running": gateway_running,
        "gateway_state": gateway_state,
        "gateway_pid": gateway_pid,
        "gateway_platforms": runtime.get("platforms") or {},
        "gateway_exit_reason": runtime.get("exit_reason"),
        "gateway_updated_at": runtime.get("updated_at"),
        "active_agents": active_agents,
        "gateway_busy": derive_gateway_busy(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
            active_agents=active_agents,
        ),
        "gateway_drainable": derive_gateway_drainable(
            gateway_running=gateway_running,
            gateway_state=gateway_state,
        ),
        "active_sessions": _active_session_count(),
        "auth_required": False,
        "auth_providers": [],
    }


def build_sessions_payload(
    *, limit: int = 20, offset: int = 0, order: str = "recent"
) -> dict[str, Any]:
    """Return recent session metadata using compact read-only SQLite queries."""
    if order not in {"created", "recent"}:
        raise ValueError("order must be one of: created, recent")

    db = _open_session_db_read_only()
    if db is None:
        return {"sessions": [], "total": 0, "limit": limit, "offset": offset}

    try:
        rows = db.list_sessions_rich(
            limit=limit,
            offset=offset,
            order_by_last_active=order == "recent",
            compact_rows=True,
        )
        total = db.session_count(exclude_children=True)
    finally:
        db.close()

    now = time.time()
    sessions = _strip_session_list_rows([dict(row) for row in rows])
    for row in sessions:
        row["archived"] = bool(row.get("archived"))
        row["is_active"] = (
            row.get("ended_at") is None
            and (now - row.get("last_active", row.get("started_at", 0))) < 300
        )
    return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}


def _dashboard_html() -> bytes:
    return b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hermes Lightweight Dashboard</title>
  <style>
    :root{color-scheme:light dark;font-family:Inter,Segoe UI,system-ui,sans-serif}
    body{margin:0;background:#f6f7f9;color:#15171a}
    main{max-width:1040px;margin:0 auto;padding:28px 20px 44px}
    header{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:22px}
    h1{font-size:24px;margin:0 0 6px}
    p{margin:0;color:#59606a}
    .pill{border:1px solid #ccd2da;border-radius:999px;padding:6px 10px;background:#fff;font-size:13px}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:18px}
    .panel,.metric{background:#fff;border:1px solid #dfe3e8;border-radius:8px;padding:14px}
    .metric span{display:block;color:#68707b;font-size:12px;margin-bottom:6px}
    .metric strong{font-size:20px}
    .panel h2{font-size:16px;margin:0 0 12px}
    table{width:100%;border-collapse:collapse;font-size:14px}
    th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #edf0f3;vertical-align:top}
    th{color:#68707b;font-weight:600}
    code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
    .muted{color:#68707b}.error{color:#b42318}
    @media (max-width:760px){header{display:block}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media (prefers-color-scheme:dark){
      body{background:#101214;color:#eef1f4}.panel,.metric,.pill{background:#171a1f;border-color:#2a3038}
      p,.muted,th{color:#a8b0ba}td,th{border-bottom-color:#252b33}
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Hermes Lightweight Dashboard</h1>
        <p>Status and recent sessions without the full admin backend.</p>
      </div>
      <div class="pill" id="updated">Loading...</div>
    </header>
    <section class="grid">
      <div class="metric"><span>Gateway</span><strong id="gateway">-</strong></div>
      <div class="metric"><span>State</span><strong id="state">-</strong></div>
      <div class="metric"><span>Active sessions</span><strong id="activeSessions">-</strong></div>
      <div class="metric"><span>Active agents</span><strong id="activeAgents">-</strong></div>
    </section>
    <section class="panel">
      <h2>Recent Sessions</h2>
      <table>
        <thead><tr><th>Title</th><th>Source</th><th>Model</th><th>Messages</th><th>Last active</th></tr></thead>
        <tbody id="sessions"><tr><td colspan="5" class="muted">Loading...</td></tr></tbody>
      </table>
    </section>
  </main>
  <script>
    const fmtTime = (value) => {
      if (!value) return "-";
      const date = new Date(Number(value) * 1000);
      return Number.isNaN(date.getTime()) ? "-" : date.toLocaleString();
    };
    const setText = (id, value) => { document.getElementById(id).textContent = value; };
    async function refresh() {
      try {
        const [statusRes, sessionsRes] = await Promise.all([
          fetch("/api/status"),
          fetch("/api/sessions?limit=20&order=recent"),
        ]);
        if (!statusRes.ok || !sessionsRes.ok) throw new Error("request failed");
        const status = await statusRes.json();
        const sessions = await sessionsRes.json();
        setText("gateway", status.gateway_running ? "Running" : "Stopped");
        setText("state", status.gateway_state || "-");
        setText("activeSessions", String(status.active_sessions ?? 0));
        setText("activeAgents", String(status.active_agents ?? 0));
        setText("updated", `Hermes ${status.version} | ${new Date().toLocaleTimeString()}`);
        const body = document.getElementById("sessions");
        body.replaceChildren();
        if (!sessions.sessions.length) {
          const row = body.insertRow();
          const cell = row.insertCell();
          cell.colSpan = 5;
          cell.className = "muted";
          cell.textContent = "No sessions found.";
          return;
        }
        for (const session of sessions.sessions) {
          const row = body.insertRow();
          row.insertCell().textContent = session.title || session.preview || session.id || "-";
          row.insertCell().textContent = session.source || "-";
          row.insertCell().textContent = session.model || "-";
          row.insertCell().textContent = String(session.message_count ?? 0);
          row.insertCell().textContent = fmtTime(session.last_active || session.started_at);
        }
      } catch (err) {
        setText("updated", "Refresh failed");
        document.getElementById("updated").className = "pill error";
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class _LightDashboardHandler(BaseHTTPRequestHandler):
    server_version = "HermesLightDashboard/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        _log.debug("light dashboard request: " + fmt, *args)

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        allowed_hosts = getattr(self.server, "allowed_host_headers", set())
        if _normalise_host_header(self.headers.get("Host")) not in allowed_hosts:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": "Invalid Host header"})
            return

        parsed = urlsplit(self.path)
        try:
            if parsed.path in ("", "/"):
                self._send(HTTPStatus.OK, _dashboard_html(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self._send_json(HTTPStatus.OK, build_status_payload())
                return
            if parsed.path == "/api/sessions":
                params = parse_qs(parsed.query)
                order = (params.get("order") or ["recent"])[0]
                limit = _coerce_int((params.get("limit") or [None])[0], default=20, minimum=1, maximum=100)
                offset = _coerce_int((params.get("offset") or [None])[0], default=0, minimum=0, maximum=100000)
                self._send_json(
                    HTTPStatus.OK,
                    build_sessions_payload(limit=limit, offset=offset, order=order),
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(exc)})
        except Exception:
            _log.exception("light dashboard request failed")
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": "Internal server error"})


class _IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _server_class_for_host(host: str):
    return _IPv6ThreadingHTTPServer if host == "::1" else ThreadingHTTPServer


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _normalise_host_header(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("["):
        end = raw.find("]")
        return raw[: end + 1] if end != -1 else raw
    return raw.rsplit(":", 1)[0]


def _allowed_host_headers(host: str) -> set[str]:
    raw = host.strip().lower()
    return {raw, _url_host(raw)}


def _write_dashboard_ready_file(actual_port: int) -> None:
    target = os.environ.get("HERMES_DESKTOP_READY_FILE")
    if not target:
        return

    tmp_name = ""
    try:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"port": int(actual_port)}, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
            tmp_name = fh.name
        os.replace(tmp_name, path)
    except Exception as exc:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except Exception:
                pass
        _log.warning("Failed to write dashboard ready file %r: %s", target, exc)


def _maybe_open_browser(host: str, actual_port: int, open_browser: bool, initial_profile: str) -> None:
    if not open_browser:
        return
    if sys_platform_is_headless_linux():
        return

    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    display_host = _url_host(display_host)
    url = f"http://{display_host}:{actual_port}"
    if initial_profile:
        url += f"/?profile={quote(initial_profile)}"

    def _open() -> None:
        try:
            time.sleep(1.0)
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def sys_platform_is_headless_linux() -> bool:
    import sys

    return (
        sys.platform == "linux"
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")
    )


def start_light_dashboard_server(
    *,
    host: str = "127.0.0.1",
    port: int = 9119,
    open_browser: bool = True,
    initial_profile: str = "",
) -> None:
    """Start the lightweight loopback-only dashboard server."""
    if not _is_loopback_bind(host):
        raise SystemExit(
            "Refusing to start lightweight dashboard on a non-loopback bind. "
            "Use the full dashboard for authenticated public/reverse-proxy "
            "deployments, or bind --host 127.0.0.1 and reach it through SSH/"
            "Tailscale tunneling."
        )

    server_cls = _server_class_for_host(host)
    httpd = server_cls((host, port), _LightDashboardHandler)
    httpd.allowed_host_headers = _allowed_host_headers(host)
    actual_port = int(httpd.server_address[1])
    _write_dashboard_ready_file(actual_port)
    display_host = _url_host(host)
    print(f"HERMES_DASHBOARD_READY port={actual_port}", flush=True)
    print(f"  Hermes Lightweight Dashboard -> http://{display_host}:{actual_port}")
    _maybe_open_browser(host, actual_port, open_browser, initial_profile)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
