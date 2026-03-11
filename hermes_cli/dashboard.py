"""
Lightweight local dashboard for Hermes runtime telemetry.

Serves a single-page neon-themed UI + JSON API from an embedded HTTP server.
No external frontend build step required.

Usage:
    hermes dashboard
    hermes dashboard --port 8765 --open
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")).expanduser()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        return False


def _gateway_status() -> dict[str, Any]:
    try:
        from gateway.status import get_running_pid

        pid = get_running_pid()
    except Exception:
        pid = None

    return {
        "running": bool(pid),
        "pid": pid,
    }


def _sessions_summary(limit: int = 20) -> dict[str, Any]:
    db_path = _hermes_home() / "state.db"
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "exists": False,
            "open_sessions": 0,
            "sources": [],
            "recent": [],
        }

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        open_sessions = conn.execute(
            "SELECT COUNT(*) AS c FROM sessions WHERE ended_at IS NULL"
        ).fetchone()["c"]

        sources = [
            dict(r)
            for r in conn.execute(
                "SELECT source, COUNT(*) AS count FROM sessions GROUP BY source ORDER BY count DESC"
            ).fetchall()
        ]

        recent_rows = conn.execute(
            """
            SELECT id, source, title, started_at, ended_at, message_count
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        recent = []
        for r in recent_rows:
            row = dict(r)
            # Keep payload compact in UI
            recent.append(
                {
                    "id": row.get("id"),
                    "source": row.get("source"),
                    "title": row.get("title") or "",
                    "started_at": row.get("started_at"),
                    "ended_at": row.get("ended_at"),
                    "message_count": row.get("message_count") or 0,
                }
            )

        return {
            "db_path": str(db_path),
            "exists": True,
            "open_sessions": int(open_sessions or 0),
            "sources": sources,
            "recent": recent,
        }
    finally:
        conn.close()


def _cron_summary(limit: int = 25) -> dict[str, Any]:
    jobs_path = _hermes_home() / "cron" / "jobs.json"
    if not jobs_path.exists():
        return {
            "jobs_path": str(jobs_path),
            "exists": False,
            "enabled_jobs": 0,
            "jobs": [],
        }

    try:
        payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    jobs = payload.get("jobs") if isinstance(payload, dict) else []
    jobs = jobs if isinstance(jobs, list) else []

    enabled = [j for j in jobs if j.get("enabled", True)]

    compact = []
    for j in enabled[:limit]:
        compact.append(
            {
                "id": j.get("id"),
                "name": j.get("name", ""),
                "schedule": j.get("schedule_display") or j.get("schedule", {}).get("display", ""),
                "next_run_at": j.get("next_run_at"),
                "last_run_at": j.get("last_run_at"),
                "last_status": j.get("last_status"),
                "deliver": j.get("deliver", "local"),
                "repeat": j.get("repeat", {}),
            }
        )

    return {
        "jobs_path": str(jobs_path),
        "exists": True,
        "enabled_jobs": len(enabled),
        "jobs": compact,
    }


def _processes_summary(limit: int = 25) -> dict[str, Any]:
    proc_path = _hermes_home() / "processes.json"
    if not proc_path.exists():
        return {
            "path": str(proc_path),
            "exists": False,
            "alive": 0,
            "entries": [],
        }

    try:
        entries = json.loads(proc_path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            entries = []
    except Exception:
        entries = []

    now = time.time()
    compact = []
    alive = 0
    for e in entries[:limit]:
        pid = e.get("pid")
        is_alive = _is_pid_alive(pid)
        if is_alive:
            alive += 1
        started_at = float(e.get("started_at", 0) or 0)
        age_seconds = int(now - started_at) if started_at else None
        compact.append(
            {
                "session_id": e.get("session_id"),
                "command": e.get("command", ""),
                "pid": pid,
                "alive": is_alive,
                "cwd": e.get("cwd", ""),
                "age_seconds": age_seconds,
                "task_id": e.get("task_id", ""),
                "session_key": e.get("session_key", ""),
            }
        )

    return {
        "path": str(proc_path),
        "exists": True,
        "alive": alive,
        "entries": compact,
    }


def _overview() -> dict[str, Any]:
    gw = _gateway_status()
    sess = _sessions_summary()
    cron = _cron_summary()
    procs = _processes_summary()

    return {
        "now": _iso_now(),
        "gateway": gw,
        "sessions": {
            "open_sessions": sess.get("open_sessions", 0),
            "sources": sess.get("sources", []),
            "recent": sess.get("recent", []),
            "db_path": sess.get("db_path"),
        },
        "cron": {
            "enabled_jobs": cron.get("enabled_jobs", 0),
            "jobs": cron.get("jobs", []),
            "jobs_path": cron.get("jobs_path"),
        },
        "processes": {
            "alive": procs.get("alive", 0),
            "entries": procs.get("entries", []),
            "path": procs.get("path"),
        },
    }


_INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hermes Ops Dashboard</title>
  <style>
    :root {
      --bg: #090b14;
      --panel: #12172a;
      --line: #2a335a;
      --text: #d9def8;
      --muted: #8f98c6;
      --cyan: #45e8ff;
      --pink: #ff4bd8;
      --green: #43f68f;
      --yellow: #ffd24a;
      --red: #ff5d75;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 20% 10%, #1b1140 0%, var(--bg) 45%), var(--bg);
      min-height: 100vh;
    }
    .scanlines::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255,255,255,0.03),
        rgba(255,255,255,0.03) 1px,
        rgba(0,0,0,0) 2px,
        rgba(0,0,0,0) 4px
      );
      opacity: 0.15;
    }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 {
      margin: 0 0 4px;
      color: var(--cyan);
      text-shadow: 0 0 14px rgba(69, 232, 255, 0.55);
      letter-spacing: 0.08em;
      font-size: 24px;
      text-transform: uppercase;
    }
    .sub { color: var(--muted); margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }
    .card {
      background: linear-gradient(180deg, rgba(18,23,42,.95), rgba(12,16,30,.95));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 0 18px rgba(69,232,255,0.08), inset 0 0 18px rgba(255,75,216,0.03);
    }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
    .value { margin-top: 6px; font-size: 28px; font-weight: 700; }
    .value.ok { color: var(--green); text-shadow: 0 0 12px rgba(67,246,143,.45); }
    .value.bad { color: var(--red); text-shadow: 0 0 12px rgba(255,93,117,.45); }
    .value.pink { color: var(--pink); text-shadow: 0 0 12px rgba(255,75,216,.35); }
    .value.cyan { color: var(--cyan); text-shadow: 0 0 12px rgba(69,232,255,.35); }
    .mt { margin-top: 14px; }
    .panel-title { margin: 0 0 8px; color: var(--pink); letter-spacing: .06em; text-transform: uppercase; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #21284a; padding: 8px 6px; text-align: left; }
    th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: .08em; }
    .pill { padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; display: inline-block; }
    .pill.ok { background: rgba(67,246,143,.15); color: var(--green); }
    .pill.err { background: rgba(255,93,117,.15); color: var(--red); }
    .pill.warn { background: rgba(255,210,74,.15); color: var(--yellow); }
    .muted { color: var(--muted); font-size: 12px; }
    @media (max-width: 980px) { .grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
    @media (max-width: 620px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body class=\"scanlines\">
  <div class=\"wrap\">
    <h1>Hermes Ops // NERV Console</h1>
    <div class=\"sub\">Live runtime telemetry • auto-refresh every 3s • local-only</div>

    <div class=\"grid\">
      <div class=\"card\"><div class=\"label\">Gateway</div><div id=\"gateway\" class=\"value\">—</div><div class=\"muted\" id=\"gatewayPid\"></div></div>
      <div class=\"card\"><div class=\"label\">Open Sessions</div><div id=\"openSessions\" class=\"value cyan\">0</div><div class=\"muted\">state.db</div></div>
      <div class=\"card\"><div class=\"label\">Enabled Cron Jobs</div><div id=\"cronCount\" class=\"value pink\">0</div><div class=\"muted\">jobs.json</div></div>
      <div class=\"card\"><div class=\"label\">Alive Background Procs</div><div id=\"procCount\" class=\"value cyan\">0</div><div class=\"muted\">processes.json</div></div>
    </div>

    <div class=\"card mt\">
      <h3 class=\"panel-title\">Sessions by Source</h3>
      <div id=\"sources\" class=\"muted\">—</div>
    </div>

    <div class=\"card mt\">
      <h3 class=\"panel-title\">Recent Sessions</h3>
      <table>
        <thead><tr><th>ID</th><th>Source</th><th>Title</th><th>Msgs</th><th>Started</th><th>Status</th></tr></thead>
        <tbody id=\"recentSessions\"></tbody>
      </table>
    </div>

    <div class=\"card mt\">
      <h3 class=\"panel-title\">Cron Jobs</h3>
      <table>
        <thead><tr><th>ID</th><th>Name</th><th>Schedule</th><th>Next Run</th><th>Last</th><th>Status</th></tr></thead>
        <tbody id=\"cronJobs\"></tbody>
      </table>
    </div>

    <div class=\"card mt\">
      <h3 class=\"panel-title\">Background Processes</h3>
      <table>
        <thead><tr><th>Session</th><th>PID</th><th>Alive</th><th>Age(s)</th><th>Command</th></tr></thead>
        <tbody id=\"procs\"></tbody>
      </table>
    </div>

    <div class=\"muted mt\" id=\"lastUpdated\">Last update: —</div>
  </div>

  <script>
    const fmtTs = (v) => {
      if (!v) return '—';
      const n = Number(v);
      if (!Number.isNaN(n) && n > 1000000000) return new Date(n * 1000).toLocaleString();
      const d = new Date(v);
      if (!Number.isNaN(d.getTime())) return d.toLocaleString();
      return String(v);
    };

    const pill = (txt, kind) => `<span class=\"pill ${kind}\">${txt}</span>`;

    async function refresh() {
      try {
        const res = await fetch('/api/overview', { cache: 'no-store' });
        const data = await res.json();

        const gw = data.gateway || {};
        const gwEl = document.getElementById('gateway');
        gwEl.textContent = gw.running ? 'ONLINE' : 'OFFLINE';
        gwEl.className = `value ${gw.running ? 'ok' : 'bad'}`;
        document.getElementById('gatewayPid').textContent = gw.pid ? `PID ${gw.pid}` : 'No running gateway PID';

        document.getElementById('openSessions').textContent = data.sessions?.open_sessions ?? 0;
        document.getElementById('cronCount').textContent = data.cron?.enabled_jobs ?? 0;
        document.getElementById('procCount').textContent = data.processes?.alive ?? 0;

        const src = (data.sessions?.sources || []).map(s => `${s.source}: ${s.count}`).join(' • ');
        document.getElementById('sources').textContent = src || 'No session source data yet';

        const recent = document.getElementById('recentSessions');
        recent.innerHTML = (data.sessions?.recent || []).map(s => {
          const status = s.ended_at ? pill('closed','warn') : pill('open','ok');
          return `<tr>
            <td>${(s.id || '').slice(0,12)}</td>
            <td>${s.source || '—'}</td>
            <td>${(s.title || '').slice(0,40) || '—'}</td>
            <td>${s.message_count ?? 0}</td>
            <td>${fmtTs(s.started_at)}</td>
            <td>${status}</td>
          </tr>`;
        }).join('') || `<tr><td colspan=\"6\" class=\"muted\">No sessions found.</td></tr>`;

        const jobs = document.getElementById('cronJobs');
        jobs.innerHTML = (data.cron?.jobs || []).map(j => {
          const st = j.last_status === 'ok' ? pill('ok','ok') : j.last_status === 'error' ? pill('error','err') : pill('pending','warn');
          return `<tr>
            <td>${(j.id || '').slice(0,12)}</td>
            <td>${(j.name || '').slice(0,40)}</td>
            <td>${j.schedule || '—'}</td>
            <td>${fmtTs(j.next_run_at)}</td>
            <td>${fmtTs(j.last_run_at)}</td>
            <td>${st}</td>
          </tr>`;
        }).join('') || `<tr><td colspan=\"6\" class=\"muted\">No enabled cron jobs.</td></tr>`;

        const procs = document.getElementById('procs');
        procs.innerHTML = (data.processes?.entries || []).map(p => {
          const live = p.alive ? pill('alive','ok') : pill('dead','err');
          return `<tr>
            <td>${(p.session_id || '').slice(0,12)}</td>
            <td>${p.pid || '—'}</td>
            <td>${live}</td>
            <td>${p.age_seconds ?? '—'}</td>
            <td>${(p.command || '').slice(0,90)}</td>
          </tr>`;
        }).join('') || `<tr><td colspan=\"5\" class=\"muted\">No tracked process entries.</td></tr>`;

        document.getElementById('lastUpdated').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
      } catch (err) {
        document.getElementById('lastUpdated').textContent = `Update error: ${err}`;
      }
    }

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "HermesDashboard/1.0"

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _html(self, html: str) -> None:
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep CLI clean; dashboard is local diagnostics.
        return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._html(_INDEX_HTML)
            return

        if path == "/health":
            self._json({"ok": True, "now": _iso_now()})
            return

        if path == "/api/overview":
            self._json(_overview())
            return

        if path == "/api/status":
            self._json({"now": _iso_now(), "gateway": _gateway_status()})
            return

        if path == "/api/sessions":
            self._json({"now": _iso_now(), **_sessions_summary()})
            return

        if path == "/api/cron":
            self._json({"now": _iso_now(), **_cron_summary()})
            return

        if path == "/api/processes":
            self._json({"now": _iso_now(), **_processes_summary()})
            return

        self._json({"error": "Not found", "path": path}, status=404)


def run_dashboard(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    """Run local dashboard server in the foreground."""
    server = ThreadingHTTPServer((host, int(port)), _DashboardHandler)
    url = f"http://{host}:{port}"

    print(f"⚡ Hermes Dashboard running at {url}")
    print("   Press Ctrl+C to stop")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        server.server_close()
