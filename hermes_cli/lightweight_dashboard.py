"""Memory-bounded read-only dashboard built on the Python standard library."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import tempfile
import threading
import time
import webbrowser
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, quote, unquote, urlsplit

from hermes_cli import __release_date__, __version__

logger = logging.getLogger(__name__)

_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_SESSION_BLOB_FIELDS = frozenset({"system_prompt", "model_config"})
_TRANSCRIPT_FIELD_LIMIT = 64 * 1024
_FILE_PREVIEW_LIMIT = 512 * 1024
_DIRECTORY_ENTRY_LIMIT = 500
_LOG_WINDOW_LIMIT = 512 * 1024
_LOG_NAMES = {
    "agent": "agent.log",
    "desktop": "desktop.log",
    "errors": "errors.log",
    "gateway": "gateway.log",
    "gui": "gui.log",
    "mcp": "mcp-stderr.log",
}
_PRIVATE_FILES = frozenset({
    ".git-credentials",
    ".netrc",
    "auth.json",
    "auth.lock",
    "config.yaml",
    "credentials",
    "google_oauth.json",
    "google_oauth_pending.json",
    "google_token.json",
    "webhook_subscriptions.json",
})
_PRIVATE_DIRECTORIES = frozenset({
    ".aws",
    ".git",
    ".gnupg",
    ".ssh",
    "mcp-tokens",
    "pairing",
})
_CONFIG_VIEW = {
    "agent": frozenset({"max_iterations", "reasoning_effort"}),
    "dashboard": frozenset({"mode", "theme"}),
    "delegation": frozenset({
        "child_timeout_seconds",
        "max_concurrent_children",
        "max_iterations",
        "max_spawn_depth",
        "orchestrator_enabled",
    }),
    "logging": frozenset({"level"}),
    "memory": frozenset({"provider"}),
    "terminal": frozenset({"backend", "cwd", "timeout"}),
}


class DashboardProblem(Exception):
    def __init__(self, status: HTTPStatus, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _limited_int(raw: str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _compact_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if len(value) <= _TRANSCRIPT_FIELD_LIMIT:
            return value
        return value[:_TRANSCRIPT_FIELD_LIMIT] + "\n...[truncated]"
    try:
        rendered = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(value)
    if len(rendered) <= _TRANSCRIPT_FIELD_LIMIT:
        return value
    return {
        "preview": rendered[:_TRANSCRIPT_FIELD_LIMIT] + "...[truncated]",
        "truncated": True,
    }


def _private_path(path: Path) -> bool:
    name = path.name.lower()
    if name == ".env" or name == ".envrc" or name.startswith(".env."):
        return True
    if name in _PRIVATE_FILES:
        return True
    return any(part.lower() in _PRIVATE_DIRECTORIES for part in path.parts)


@dataclass(frozen=True)
class ProfileView:
    requested_name: str | None = None

    @cached_property
    def identity(self) -> tuple[str, Path]:
        from hermes_cli import profiles

        raw = (self.requested_name or "default").strip() or "default"
        name = profiles.normalize_profile_name(raw)
        try:
            profiles.validate_profile_name(name)
        except ValueError as exc:
            raise DashboardProblem(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if not profiles.profile_exists(name):
            raise DashboardProblem(
                HTTPStatus.NOT_FOUND, f"Profile {name!r} does not exist"
            )
        return name, profiles.get_profile_dir(name)

    @property
    def name(self) -> str:
        return self.identity[0]

    @property
    def home(self) -> Path:
        return self.identity[1]

    @contextmanager
    def session_db(self) -> Iterator[Any | None]:
        path = self.home / "state.db"
        if not path.exists():
            yield None
            return
        from hermes_state import SessionDB

        database = SessionDB(db_path=path, read_only=True)
        try:
            yield database
        finally:
            database.close()

    def raw_config(self) -> dict[str, Any]:
        path = self.home / "config.yaml"
        if not path.exists():
            return {}
        try:
            import yaml

            config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            raise DashboardProblem(
                HTTPStatus.BAD_REQUEST, "Profile config could not be parsed"
            ) from exc
        if not isinstance(config, dict):
            raise DashboardProblem(
                HTTPStatus.BAD_REQUEST, "Profile config must contain a mapping"
            )
        return config

    def status(self) -> dict[str, Any]:
        from gateway.status import (
            derive_gateway_busy,
            derive_gateway_drainable,
            get_runtime_status_running_pid,
            get_running_pid_cached,
            parse_active_agents,
            read_runtime_status,
        )

        runtime = read_runtime_status(self.home / "gateway_state.json") or {}
        pid = get_running_pid_cached(self.home / "gateway.pid", cleanup_stale=False)
        if pid is None:
            pid = get_runtime_status_running_pid(runtime, expected_home=self.home)
        running = pid is not None
        state = runtime.get("gateway_state")
        if not running and state != "startup_failed":
            state = "stopped"
        agents = parse_active_agents(runtime.get("active_agents", 0))
        recent = self.sessions(limit=50, offset=0, order="recent")["sessions"]
        active = sum(1 for row in recent if row["is_active"])
        return {
            "active_agents": agents,
            "active_sessions": active,
            "gateway_busy": derive_gateway_busy(
                gateway_running=running,
                gateway_state=state,
                active_agents=agents,
            ),
            "gateway_drainable": derive_gateway_drainable(
                gateway_running=running,
                gateway_state=state,
            ),
            "gateway_pid": pid,
            "gateway_platforms": runtime.get("platforms") or {},
            "gateway_running": running,
            "gateway_state": state,
            "mode": "lightweight",
            "profile": self.name,
            "release_date": __release_date__,
            "version": __version__,
        }

    def sessions(self, *, limit: int, offset: int, order: str) -> dict[str, Any]:
        if order not in {"created", "recent"}:
            raise DashboardProblem(
                HTTPStatus.BAD_REQUEST, "order must be created or recent"
            )
        with self.session_db() as database:
            if database is None:
                return {
                    "limit": limit,
                    "offset": offset,
                    "sessions": [],
                    "total": 0,
                }
            rows = database.list_sessions_rich(
                compact_rows=True,
                limit=limit,
                offset=offset,
                order_by_last_active=order == "recent",
            )
            total = database.session_count(exclude_children=True)
        now = time.time()
        sessions: list[dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            for key in _SESSION_BLOB_FIELDS:
                row.pop(key, None)
            last_active = row.get("last_active", row.get("started_at", 0))
            row["archived"] = bool(row.get("archived"))
            row["is_active"] = bool(
                row.get("ended_at") is None and now - last_active < 300
            )
            sessions.append(row)
        return {
            "limit": limit,
            "offset": offset,
            "sessions": sessions,
            "total": total,
        }

    def session_metadata(self, session_id: str) -> dict[str, Any]:
        with self.session_db() as database:
            resolved = database.resolve_session_id(session_id) if database else None
            row = database.get_session(resolved) if database and resolved else None
        if row is None:
            raise DashboardProblem(HTTPStatus.NOT_FOUND, "Session not found")
        result = dict(row)
        for key in _SESSION_BLOB_FIELDS:
            result.pop(key, None)
        result["archived"] = bool(result.get("archived"))
        result["profile"] = self.name
        return result

    def transcript(self, session_id: str, *, limit: int, offset: int) -> dict[str, Any]:
        with self.session_db() as database:
            resolved = database.resolve_session_id(session_id) if database else None
            if database is None or resolved is None:
                raise DashboardProblem(HTTPStatus.NOT_FOUND, "Session not found")
            resumed = database.resolve_resume_session_id(resolved)
            rows = database.get_messages(resumed, limit=limit, offset=offset)
        messages = []
        bounded_fields = {
            "codex_message_items",
            "codex_reasoning_items",
            "content",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "tool_calls",
        }
        for source in rows:
            message = dict(source)
            for key in bounded_fields.intersection(message):
                message[key] = _compact_value(message[key])
            messages.append(message)
        return {
            "messages": messages,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "returned": len(messages),
            },
            "session_id": resumed,
        }

    def files_root(self) -> Path:
        configured = os.environ.get("HERMES_DASHBOARD_FILES_ROOT", "").strip()
        if configured:
            candidate = Path(configured).expanduser()
        else:
            terminal = self.raw_config().get("terminal") or {}
            cwd = (
                str(terminal.get("cwd") or "").strip()
                if isinstance(terminal, dict)
                else ""
            )
            candidate = (
                Path(cwd).expanduser()
                if cwd not in {"", ".", "auto", "cwd"}
                else Path.cwd()
            )
        try:
            root = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise DashboardProblem(
                HTTPStatus.NOT_FOUND, "Managed files root is unavailable"
            ) from exc
        if not root.is_dir():
            raise DashboardProblem(
                HTTPStatus.NOT_FOUND, "Managed files root is unavailable"
            )
        return root

    def resolve_file(self, requested: str | None) -> tuple[Path, Path]:
        root = self.files_root()
        raw = str(requested or "").strip()
        if "\x00" in raw:
            raise DashboardProblem(HTTPStatus.BAD_REQUEST, "Invalid path")
        candidate = Path(raw).expanduser() if raw else root
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            target = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            raise DashboardProblem(HTTPStatus.NOT_FOUND, "Path not found") from exc
        if target != root and root not in target.parents:
            raise DashboardProblem(
                HTTPStatus.FORBIDDEN, "Path is outside the managed files root"
            )
        if _private_path(target):
            raise DashboardProblem(
                HTTPStatus.FORBIDDEN, "Access to sensitive files is not allowed"
            )
        return root, target

    def directory(self, requested: str | None) -> dict[str, Any]:
        root, target = self.resolve_file(requested)
        if not target.is_dir():
            raise DashboardProblem(HTTPStatus.BAD_REQUEST, "Path is not a directory")
        try:
            children = sorted(
                (child for child in target.iterdir() if not _private_path(child)),
                key=lambda child: (not child.is_dir(), child.name.lower()),
            )
        except PermissionError as exc:
            raise DashboardProblem(
                HTTPStatus.FORBIDDEN, "Directory is not readable"
            ) from exc
        entries = []
        for child in children[:_DIRECTORY_ENTRY_LIMIT]:
            try:
                resolved = child.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    continue
                stat_result = resolved.stat()
            except (FileNotFoundError, OSError, RuntimeError):
                continue
            entries.append({
                "is_directory": resolved.is_dir(),
                "mtime": stat_result.st_mtime,
                "name": child.name,
                "path": str(resolved.relative_to(root)),
                "size": None if resolved.is_dir() else stat_result.st_size,
            })
        return {
            "entries": entries,
            "parent": None if target == root else str(target.parent.relative_to(root)),
            "path": "." if target == root else str(target.relative_to(root)),
            "read_only": True,
            "root": str(root),
            "truncated": len(children) > _DIRECTORY_ENTRY_LIMIT,
        }

    def file_preview(self, requested: str) -> dict[str, Any]:
        root, target = self.resolve_file(requested)
        if not target.is_file():
            raise DashboardProblem(HTTPStatus.BAD_REQUEST, "Path is not a file")
        size = target.stat().st_size
        if size > _FILE_PREVIEW_LIMIT:
            raise DashboardProblem(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"File exceeds the {_FILE_PREVIEW_LIMIT // 1024} KiB preview limit",
            )
        try:
            data = target.read_bytes()
        except PermissionError as exc:
            raise DashboardProblem(
                HTTPStatus.FORBIDDEN, "File is not readable"
            ) from exc
        if b"\x00" in data:
            raise DashboardProblem(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Binary files cannot be previewed",
            )
        return {
            "content": data.decode("utf-8", errors="replace"),
            "name": target.name,
            "path": str(target.relative_to(root)),
            "read_only": True,
            "size": size,
        }

    def logs(self, *, name: str, lines: int, search: str) -> dict[str, Any]:
        filename = _LOG_NAMES.get(name)
        if filename is None:
            raise DashboardProblem(HTTPStatus.BAD_REQUEST, f"Unknown log file: {name}")
        path = self.home / "logs" / filename
        if not path.exists():
            return {"file": name, "lines": [], "truncated": False}
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - _LOG_WINDOW_LIMIT))
                content = handle.read(_LOG_WINDOW_LIMIT)
        except PermissionError as exc:
            raise DashboardProblem(
                HTTPStatus.FORBIDDEN, "Log file is not readable"
            ) from exc
        result = content.decode("utf-8", errors="replace").splitlines()
        if search:
            needle = search.lower()
            result = [line for line in result if needle in line.lower()]
        return {
            "file": name,
            "lines": result[-lines:],
            "truncated": size > _LOG_WINDOW_LIMIT,
        }

    def safe_config(self) -> dict[str, Any]:
        source = self.raw_config()
        result: dict[str, Any] = {}
        for key in ("api_mode", "model", "provider"):
            value = source.get(key)
            if isinstance(value, (str, int, float, bool, type(None))) and key in source:
                result[key] = value
        for section, fields in _CONFIG_VIEW.items():
            values = source.get(section)
            if not isinstance(values, dict):
                continue
            visible = {key: values[key] for key in fields if key in values}
            if visible:
                result[section] = visible
        result.setdefault("dashboard", {})["mode"] = "lightweight"
        return {"config": result, "profile": self.name, "read_only": True}


_PAGE = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Lightweight Dashboard</title><style>
:root{color-scheme:light dark;font-family:Inter,Segoe UI,system-ui,sans-serif}*{box-sizing:border-box}
body{margin:0;background:#f7f8fa;color:#16191d}main{max-width:1120px;margin:auto;padding:22px 18px 48px}
header{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:18px}h1{font-size:22px;margin:0}h2{font-size:16px;margin:0 0 12px}
.meta,.empty{color:#66707c;font-size:13px}.error{color:#b42318}nav{display:flex;gap:4px;border-bottom:1px solid #d9dee5;margin-bottom:18px;overflow-x:auto}
button,input,select{font:inherit}button{cursor:pointer}.tab{border:0;border-bottom:2px solid transparent;background:transparent;padding:9px 12px;color:#58616d}
.tab.active{border-bottom-color:#b7791f;color:inherit;font-weight:600}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:18px}
.metric,.surface{background:#fff;border:1px solid #dfe3e8;border-radius:7px;padding:13px}.metric{min-height:76px}.metric span{display:block;color:#68717c;font-size:12px;margin-bottom:7px}.metric strong{font-size:19px}
.view{display:none}.view.active{display:block}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,.9fr);gap:14px}.scroll{overflow:auto;max-height:64vh}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.toolbar button,.toolbar input,.toolbar select{height:34px;border:1px solid #cbd2da;border-radius:6px;background:#fff;color:inherit;padding:0 10px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid #edf0f3;vertical-align:top}th{color:#68717c;white-space:nowrap}
.link{border:0;background:transparent;color:#0969da;padding:0;text-align:left}.empty{padding:24px 8px;text-align:center}.path{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:#59636f;overflow-wrap:anywhere}
pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace}.message{border-left:3px solid #cbd2da;background:#f8f9fb;padding:8px 10px;margin:8px 0}.message.user{border-color:#b7791f}.message.assistant{border-color:#16836b}.role{font-size:11px;font-weight:700;text-transform:uppercase;color:#68717c;margin-bottom:5px}
@media(max-width:800px){header{align-items:flex-start}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.split{grid-template-columns:1fr}nav{gap:2px}.tab{padding:9px 7px;font-size:13px}}
@media(prefers-color-scheme:dark){body{background:#111419;color:#edf0f3}.metric,.surface,.toolbar button,.toolbar input,.toolbar select{background:#181c22;border-color:#303740}nav,th,td{border-bottom-color:#2b3139}.meta,.empty,.path,th,.tab{color:#aab2bc}.message{background:#14181d}}
</style></head><body><main><header><h1>Hermes Lightweight Dashboard</h1><div id="updated" class="meta">Loading...</div></header>
<nav><button class="tab active" data-view="overview">Overview</button><button class="tab" data-view="sessions">Sessions</button><button class="tab" data-view="files">Files</button><button class="tab" data-view="logs">Logs</button><button class="tab" data-view="config">Config</button></nav>
<section id="overview" class="view active"><div class="grid"><div class="metric"><span>Gateway</span><strong id="gateway">-</strong></div><div class="metric"><span>State</span><strong id="state">-</strong></div><div class="metric"><span>Active sessions</span><strong id="activeSessions">-</strong></div><div class="metric"><span>Active agents</span><strong id="activeAgents">-</strong></div></div><div class="surface"><h2>Recent Sessions</h2><div id="recent"></div></div></section>
<section id="sessions" class="view"><div class="split"><div class="surface scroll"><h2>Sessions</h2><div id="sessionList"></div></div><div class="surface scroll"><h2 id="sessionTitle">Session Detail</h2><div id="transcript" class="empty">Select a session</div></div></div></section>
<section id="files" class="view"><div class="split"><div class="surface scroll"><div class="toolbar"><button id="up">Up</button><button id="reloadFiles">Refresh</button></div><div id="filePath" class="path">.</div><div id="fileList"></div></div><div class="surface scroll"><h2 id="previewTitle">File Preview</h2><pre id="preview">Select a file</pre></div></div></section>
<section id="logs" class="view"><div class="surface"><div class="toolbar"><select id="logName"><option>agent</option><option>errors</option><option>gateway</option><option>gui</option><option>desktop</option><option>mcp</option></select><input id="logSearch" type="search" placeholder="Filter"><button id="reloadLogs">Refresh</button></div><pre id="logText" class="scroll">Loading...</pre></div></section>
<section id="config" class="view"><div class="surface"><pre id="configText">Loading...</pre></div></section>
</main><script>
const profile=new URLSearchParams(location.search).get("profile");const text=(id,value)=>document.getElementById(id).textContent=value;
const showValue=value=>typeof value==="string"?value:JSON.stringify(value,null,2);const fmt=value=>value?new Date(Number(value)*1000).toLocaleString():"-";
async function get(path,params={}){const query=new URLSearchParams(params);if(profile)query.set("profile",profile);const response=await fetch(`${path}?${query}`);const body=await response.json();if(!response.ok)throw new Error(body.detail||"Request failed");return body}
function table(rows,target){const host=document.getElementById(target);host.replaceChildren();if(!rows.length){host.className="empty";host.textContent="No sessions";return}host.className="";const grid=document.createElement("table");grid.innerHTML="<thead><tr><th>Title</th><th>Source</th><th>Model</th><th>Messages</th><th>Last active</th></tr></thead>";const body=grid.createTBody();for(const item of rows){const row=body.insertRow();const cell=row.insertCell();const button=document.createElement("button");button.className="link";button.textContent=item.title||item.preview||item.id;button.onclick=()=>{activate("sessions");loadSession(item.id)};cell.append(button);row.insertCell().textContent=item.source||"-";row.insertCell().textContent=item.model||"-";row.insertCell().textContent=String(item.message_count??0);row.insertCell().textContent=fmt(item.last_active||item.started_at)}host.append(grid)}
async function refresh(){try{const [status,items]=await Promise.all([get("/api/status"),get("/api/sessions",{limit:"30",order:"recent"})]);text("gateway",status.gateway_running?"Running":"Stopped");text("state",status.gateway_state||"-");text("activeSessions",String(status.active_sessions??0));text("activeAgents",String(status.active_agents??0));text("updated",`Hermes ${status.version} | ${status.profile} | ${new Date().toLocaleTimeString()}`);table(items.sessions,"recent");table(items.sessions,"sessionList")}catch(error){text("updated",error.message);document.getElementById("updated").className="error"}}
async function loadSession(id){const host=document.getElementById("transcript");host.textContent="Loading...";try{const [detail,page]=await Promise.all([get(`/api/sessions/${encodeURIComponent(id)}`),get(`/api/sessions/${encodeURIComponent(id)}/messages`,{limit:"50"})]);text("sessionTitle",detail.title||detail.id);host.replaceChildren();for(const item of page.messages){const box=document.createElement("div");box.className=`message ${item.role||""}`;const role=document.createElement("div");role.className="role";role.textContent=item.role||"message";const content=document.createElement("pre");content.textContent=showValue(item.content??item.tool_calls??"");box.append(role,content);host.append(box)}}catch(error){host.textContent=error.message;host.className="error"}}
let currentPath="",parentPath=null;async function loadDirectory(path=""){const host=document.getElementById("fileList");host.textContent="Loading...";try{const data=await get("/api/files",path?{path}:{});currentPath=data.path==="."?"":data.path;parentPath=data.parent;text("filePath",`${data.root} / ${data.path}`);host.replaceChildren();const grid=document.createElement("table");grid.innerHTML="<thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead>";const body=grid.createTBody();for(const item of data.entries){const row=body.insertRow();const cell=row.insertCell();const button=document.createElement("button");button.className="link";button.textContent=`${item.is_directory?"/":""}${item.name}`;button.onclick=()=>item.is_directory?loadDirectory(item.path):loadFile(item.path);cell.append(button);row.insertCell().textContent=item.is_directory?"-":String(item.size??0);row.insertCell().textContent=fmt(item.mtime)}host.append(grid)}catch(error){host.textContent=error.message;host.className="error"}}
async function loadFile(path){try{const data=await get("/api/files/read",{path});text("previewTitle",data.name);text("preview",data.content)}catch(error){text("preview",error.message)}}
async function loadLogs(){try{const data=await get("/api/logs",{file:document.getElementById("logName").value,lines:"300",search:document.getElementById("logSearch").value});text("logText",data.lines.join("\\n")||"No log entries")}catch(error){text("logText",error.message)}}
async function loadConfig(){try{text("configText",JSON.stringify(await get("/api/config"),null,2))}catch(error){text("configText",error.message)}}
function activate(id){document.querySelectorAll(".view").forEach(node=>node.classList.toggle("active",node.id===id));document.querySelectorAll(".tab").forEach(node=>node.classList.toggle("active",node.dataset.view===id));if(id==="files")loadDirectory(currentPath);if(id==="logs")loadLogs();if(id==="config")loadConfig()}
document.querySelectorAll(".tab").forEach(button=>button.onclick=()=>activate(button.dataset.view));document.getElementById("up").onclick=()=>loadDirectory(parentPath||"");document.getElementById("reloadFiles").onclick=()=>loadDirectory(currentPath);document.getElementById("reloadLogs").onclick=loadLogs;document.getElementById("logName").onchange=loadLogs;refresh();setInterval(refresh,5000);
</script></body></html>"""


def _normal_host(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw.startswith("["):
        closing = raw.find("]")
        return raw[: closing + 1] if closing >= 0 else raw
    return raw.rsplit(":", 1)[0]


def _ip_host(value: str) -> bool:
    candidate = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return True


def _loopback(host: str) -> bool:
    lowered = (host or "127.0.0.1").strip().lower()
    if lowered in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


class LightweightHandler(BaseHTTPRequestHandler):
    server_version = "HermesLightweight/1"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("lightweight dashboard: " + format, *args)

    def reply(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def json_reply(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.reply(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        host = _normal_host(self.headers.get("Host"))
        exact_hosts = getattr(self.server, "accepted_hosts", set())
        accept_ip = bool(getattr(self.server, "accept_ip_hosts", False))
        if host not in exact_hosts and not (accept_ip and _ip_host(host)):
            self.json_reply(HTTPStatus.BAD_REQUEST, {"detail": "Invalid Host header"})
            return
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        profile = (query.get("profile") or [None])[0]
        view = ProfileView(profile)
        try:
            if parsed.path in {"", "/"}:
                self.reply(HTTPStatus.OK, _PAGE, "text/html; charset=utf-8")
            elif parsed.path == "/api/status":
                self.json_reply(HTTPStatus.OK, view.status())
            elif parsed.path == "/api/sessions":
                self.json_reply(
                    HTTPStatus.OK,
                    view.sessions(
                        limit=_limited_int(
                            (query.get("limit") or [None])[0],
                            default=20,
                            minimum=1,
                            maximum=100,
                        ),
                        offset=_limited_int(
                            (query.get("offset") or [None])[0],
                            default=0,
                            minimum=0,
                            maximum=100000,
                        ),
                        order=(query.get("order") or ["recent"])[0],
                    ),
                )
            elif parsed.path.startswith("/api/sessions/"):
                suffix = parsed.path.removeprefix("/api/sessions/")
                if suffix.endswith("/messages"):
                    session_id = unquote(suffix.removesuffix("/messages"))
                    payload = view.transcript(
                        session_id,
                        limit=_limited_int(
                            (query.get("limit") or [None])[0],
                            default=30,
                            minimum=1,
                            maximum=50,
                        ),
                        offset=_limited_int(
                            (query.get("offset") or [None])[0],
                            default=0,
                            minimum=0,
                            maximum=100000,
                        ),
                    )
                else:
                    payload = view.session_metadata(unquote(suffix))
                self.json_reply(HTTPStatus.OK, payload)
            elif parsed.path == "/api/files/read":
                self.json_reply(
                    HTTPStatus.OK,
                    view.file_preview((query.get("path") or [""])[0]),
                )
            elif parsed.path == "/api/files":
                self.json_reply(
                    HTTPStatus.OK,
                    view.directory((query.get("path") or [None])[0]),
                )
            elif parsed.path == "/api/logs":
                self.json_reply(
                    HTTPStatus.OK,
                    view.logs(
                        name=(query.get("file") or ["agent"])[0],
                        lines=_limited_int(
                            (query.get("lines") or [None])[0],
                            default=100,
                            minimum=1,
                            maximum=500,
                        ),
                        search=(query.get("search") or [""])[0],
                    ),
                )
            elif parsed.path == "/api/config":
                self.json_reply(HTTPStatus.OK, view.safe_config())
            else:
                self.json_reply(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
        except DashboardProblem as exc:
            self.json_reply(exc.status, {"detail": exc.detail})
        except Exception:
            logger.exception("lightweight dashboard request failed")
            self.json_reply(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"detail": "Internal server error"},
            )


class IPv6LightweightServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def _ready_file(port: int) -> None:
    destination = os.environ.get("HERMES_DESKTOP_READY_FILE", "").strip()
    if not destination:
        return
    path = Path(destination)
    temporary = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump({"port": port}, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    except Exception as exc:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        logger.warning("Could not write dashboard ready file: %s", exc)


def _browser_url(host: str, port: int, initial_profile: str) -> str:
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    url = f"http://{display_host}:{port}/"
    return f"{url}?profile={quote(initial_profile)}" if initial_profile else url


def run_lightweight_dashboard(
    *,
    host: str,
    port: int,
    open_browser: bool,
    initial_profile: str,
    allow_remote: bool,
) -> None:
    """Run the lightweight dashboard without importing the full web backend."""
    if not _loopback(host) and not allow_remote:
        raise SystemExit(
            "Lightweight dashboard refuses non-loopback binds by default. "
            "Use a tunnel or pass --insecure on a trusted private network."
        )
    if allow_remote and not _loopback(host):
        print(
            "WARNING: lightweight dashboard remote access is unauthenticated and read-only.",
            flush=True,
        )
    server_type = IPv6LightweightServer if ":" in host else ThreadingHTTPServer
    server = server_type((host, port), LightweightHandler)
    normalized = host.strip().lower()
    bracketed = f"[{normalized}]" if ":" in normalized else normalized
    server.accepted_hosts = {normalized, bracketed}
    server.accept_ip_hosts = bool(allow_remote and not _loopback(host))
    actual_port = int(server.server_address[1])
    _ready_file(actual_port)
    url = _browser_url(host, actual_port, initial_profile)
    print(f"HERMES_DASHBOARD_READY port={actual_port}", flush=True)
    print(f"  Hermes Lightweight Dashboard -> {url}")
    if open_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
