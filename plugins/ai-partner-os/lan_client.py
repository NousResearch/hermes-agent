"""HTTP client for AI Partner OS LAN server (default port 8899)."""

from __future__ import annotations

import http.cookiejar
import json
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from hermes_constants import get_hermes_home

DEFAULT_LAN_PORT = 8899
DEFAULT_LAN_HOST = "127.0.0.1"
LAN_AUTH_PATH = "/api/auth"
LAN_ROUTES_CACHE = get_hermes_home() / "ai_partner_os_lan_routes.json"

# Known + candidate routes (mobile sync + action queue probes).
PROBE_GET_PATHS = (
    "/",
    "/api",
    "/api/",
    "/api/health",
    "/api/status",
    "/api/ping",
    "/mobile",
    "/mobile/",
)
ACTION_POST_PATHS = (
    "/api/action",
    "/api/pc-actions",
    "/api/pc_actions",
    "/api/pc/action",
    "/api/pc/action/queue",
    "/api/actions",
    "/api/actions/queue",
    "/api/queue",
    "/api/mobile/action",
    "/api/mobile/actions",
    "/api/mobile/pc-action",
    "/api/mobile/pc-actions",
)
MUSIC_POST_PATHS = ("/api/music/notify",)
PROBE_POST_PATHS = ACTION_POST_PATHS + MUSIC_POST_PATHS + (
    "/api/chat/send",
    "/api/chat/message",
    "/api/chat/proactive",
    "/api/proactive",
    "/api/message",
    "/api/messages",
)


def lan_port_open(host: str = DEFAULT_LAN_HOST, port: int = DEFAULT_LAN_PORT, *, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_lan_hosts(host: str | None = None, *, include_localhost_fallback: bool = True) -> list[str]:
    primary = (host or DEFAULT_LAN_HOST).strip() or DEFAULT_LAN_HOST
    hosts = [primary]
    if include_localhost_fallback and primary not in {"127.0.0.1", "localhost"}:
        hosts.append("127.0.0.1")
    return hosts


class LanSession:
    """Cookie-backed LAN session (POST /api/auth with PIN)."""

    def __init__(self, host: str = DEFAULT_LAN_HOST, port: int = DEFAULT_LAN_PORT) -> None:
        self.host = host.strip() or DEFAULT_LAN_HOST
        self.port = int(port)
        self._jar = http.cookiejar.CookieJar()
        self._opener = build_opener(HTTPCookieProcessor(self._jar))
        self.authenticated = False
        self.auth_result: dict[str, Any] | None = None

    def authenticate(self, pin: str, *, timeout: float = 10.0) -> dict[str, Any]:
        if not pin:
            return {"ok": False, "error": "LAN PIN is required for authenticated API access"}
        result = self.request("POST", LAN_AUTH_PATH, body={"pin": pin}, timeout=timeout)
        self.auth_result = result
        status = int(result.get("status") or 0)
        self.authenticated = bool(result.get("ok") and status and status < 400)
        return result

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        url = f"http://{self.host}:{self.port}{path}"
        data = None
        req_headers = {"Accept": "application/json"}
        if headers:
            req_headers.update(headers)
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            req_headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=req_headers, method=method.upper())
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                if not raw.strip():
                    return {"ok": True, "status": resp.status, "body": None, "host": self.host}
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = raw
                return {"ok": True, "status": resp.status, "body": parsed, "host": self.host}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "status": exc.code,
                "error": detail or exc.reason,
                "path": path,
                "host": self.host,
            }
        except (URLError, OSError, TimeoutError) as exc:
            return {"ok": False, "error": str(exc), "path": path, "host": self.host}


def _load_cached_routes() -> list[str]:
    if not LAN_ROUTES_CACHE.is_file():
        return []
    try:
        data = json.loads(LAN_ROUTES_CACHE.read_text(encoding="utf-8"))
        routes = data.get("post_action_routes") if isinstance(data, dict) else None
        if isinstance(routes, list):
            return [str(r) for r in routes if str(r).startswith("/")]
    except (OSError, json.JSONDecodeError):
        pass
    return []


def _load_cached_host() -> str | None:
    if not LAN_ROUTES_CACHE.is_file():
        return None
    try:
        data = json.loads(LAN_ROUTES_CACHE.read_text(encoding="utf-8"))
        host = data.get("host") if isinstance(data, dict) else None
        text = str(host or "").strip()
        return text or None
    except (OSError, json.JSONDecodeError):
        return None


def _save_cached_routes(
    post_action_routes: list[str],
    probe_report: list[dict[str, Any]],
    *,
    host: str,
    port: int,
    auth: dict[str, Any] | None = None,
) -> None:
    LAN_ROUTES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host,
        "port": port,
        "post_action_routes": post_action_routes,
        "probe_report": probe_report,
        "auth": auth,
    }
    LAN_ROUTES_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _request_unauthenticated(
    method: str,
    path: str,
    *,
    host: str = DEFAULT_LAN_HOST,
    port: int = DEFAULT_LAN_PORT,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    url = f"http://{host}:{port}{path}"
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=req_headers, method=method.upper())
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {"ok": True, "status": resp.status, "body": None}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            return {"ok": True, "status": resp.status, "body": parsed}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": detail or exc.reason, "path": path}
    except (URLError, OSError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "path": path}


def _probe_host(
    session: LanSession,
    *,
    pin: str,
) -> dict[str, Any] | None:
    if not lan_port_open(host=session.host, port=session.port):
        return None

    auth = session.authenticate(pin) if pin else {"ok": True, "skipped": True}
    if pin and not session.authenticated:
        return {
            "ok": False,
            "host": session.host,
            "port": session.port,
            "error": auth.get("error") or "LAN PIN authentication failed",
            "auth": auth,
        }

    report: list[dict[str, Any]] = [{"method": "POST", "path": LAN_AUTH_PATH, **auth}]
    for path in PROBE_GET_PATHS:
        result = session.request("GET", path, timeout=5.0)
        report.append({"method": "GET", "path": path, **result})

    sample_body = {"action": "openWindow", "params": {"window": "chat"}}
    working_posts: list[str] = []
    for path in PROBE_POST_PATHS:
        result = session.request("POST", path, body=sample_body, timeout=5.0)
        entry = {"method": "POST", "path": path, **result}
        report.append(entry)
        status = int(result.get("status") or 0)
        if path in MUSIC_POST_PATHS or path not in ACTION_POST_PATHS:
            continue
        if result.get("ok") and status and status < 500 and status not in (404, 405):
            working_posts.append(path)

    music = notify_music("probe", "probe.wav", "probe", host=session.host, port=session.port, session=session)
    report.append({"method": "POST", "path": "/api/music/notify", **music})

    return {
        "ok": True,
        "host": session.host,
        "port": session.port,
        "authenticated": session.authenticated,
        "auth": auth,
        "working_post_routes": working_posts,
        "probe_count": len(report),
        "probe_report": report,
    }


def discover_lan_api(
    *,
    host: str = DEFAULT_LAN_HOST,
    hosts: list[str] | None = None,
    port: int = DEFAULT_LAN_PORT,
    pin: str = "",
) -> dict[str, Any]:
    candidates = hosts or resolve_lan_hosts(host)
    attempts: list[dict[str, Any]] = []

    for candidate in candidates:
        session = LanSession(host=candidate, port=port)
        result = _probe_host(session, pin=pin)
        if result is None:
            attempts.append({"host": candidate, "port": port, "ok": False, "error": "port closed"})
            continue
        attempts.append({"host": candidate, "port": port, **result})
        if result.get("ok"):
            working_posts = result.get("working_post_routes") or []
            if working_posts:
                _save_cached_routes(
                    working_posts,
                    result.get("probe_report") or [],
                    host=candidate,
                    port=port,
                    auth=result.get("auth"),
                )
            return {
                **result,
                "hosts_tried": candidates,
                "cached_routes_file": str(LAN_ROUTES_CACHE),
            }

    return {
        "ok": False,
        "error": f"LAN port {port} is not reachable on any host",
        "hosts_tried": candidates,
        "port": port,
        "attempts": attempts,
        "hint": "Start AI Partner OS, enable LAN under Settings > Phone, and verify lan_host/lan_pin.",
    }


def queue_pc_action(
    action: str,
    params: dict[str, Any] | None = None,
    *,
    host: str = DEFAULT_LAN_HOST,
    hosts: list[str] | None = None,
    port: int = DEFAULT_LAN_PORT,
    pin: str = "",
) -> dict[str, Any]:
    """Queue an OS action for the desktop UI (same path mobile clients use)."""
    payload = {"action": action, "params": params or {}}
    cached_host = _load_cached_host()
    candidate_hosts: list[str] = []
    for item in (cached_host, host, *(hosts or resolve_lan_hosts(host))):
        if item and item not in candidate_hosts:
            candidate_hosts.append(item)

    all_errors: list[str] = []
    for candidate in candidate_hosts:
        if not lan_port_open(host=candidate, port=port):
            all_errors.append(f"{candidate}: port closed")
            continue

        session = LanSession(host=candidate, port=port)
        if pin:
            auth = session.authenticate(pin)
            if not session.authenticated:
                all_errors.append(f"{candidate}: auth failed ({auth.get('error') or auth.get('status')})")
                continue

        action_paths = [p for p in _load_cached_routes() if p in ACTION_POST_PATHS]
        action_paths += [p for p in ACTION_POST_PATHS if p not in action_paths]
        seen: set[str] = set()
        for path in action_paths:
            if path in seen:
                continue
            seen.add(path)
            result = session.request("POST", path, body=payload)
            status = int(result.get("status") or 0)
            body = result.get("body") if isinstance(result.get("body"), dict) else {}
            success = body.get("success")
            if result.get("ok") and status and status not in (401, 403, 404, 405):
                if success is False:
                    all_errors.append(f"{candidate}{path}: {body.get('message') or 'success=false'}")
                    continue
                result["endpoint"] = path
                result["host"] = candidate
                result["authenticated"] = session.authenticated
                return result
            all_errors.append(f"{candidate}{path}: {result.get('error') or result.get('status')}")

    return {
        "ok": False,
        "error": "LAN action queue failed on all known endpoints",
        "attempts": all_errors,
        "hosts_tried": candidate_hosts,
        "hint": (
            "LAN /api/action accepts POST but rejects all OS_ACTIONS in ver0.2.3. "
            "Desktop UI must poll poll_pc_actions via embedded Eel."
        ),
    }


def notify_music(
    track_id: str,
    file_name: str,
    track_name: str,
    *,
    host: str = DEFAULT_LAN_HOST,
    port: int = DEFAULT_LAN_PORT,
    session: LanSession | None = None,
    pin: str = "",
) -> dict[str, Any]:
    body = {"trackId": track_id, "fileName": file_name, "trackName": track_name}
    if session is not None:
        return session.request("POST", "/api/music/notify", body=body)
    if pin:
        session = LanSession(host=host, port=port)
        auth = session.authenticate(pin)
        if not session.authenticated:
            return {"ok": False, "error": auth.get("error") or "LAN auth failed", "auth": auth}
        return session.request("POST", "/api/music/notify", body=body)
    return _request_unauthenticated(
        "POST",
        "/api/music/notify",
        host=host,
        port=port,
        body=body,
    )


def send_proactive_message(
    message: str,
    *,
    host: str = DEFAULT_LAN_HOST,
    hosts: list[str] | None = None,
    port: int = DEFAULT_LAN_PORT,
    pin: str = "",
) -> dict[str, Any]:
    """Push chat text over LAN when Eel RPC is unavailable."""
    body = {"message": message, "type": "hermes-gui", "proactive": True}
    chat_paths = ("/api/chat", "/api/chat/proactive", "/api/proactive", "/api/chat/message")
    candidate_hosts = []
    for item in (host, *(hosts or resolve_lan_hosts(host))):
        if item and item not in candidate_hosts:
            candidate_hosts.append(item)

    for candidate in candidate_hosts:
        if not lan_port_open(host=candidate, port=port):
            continue
        session = LanSession(host=candidate, port=port)
        if pin:
            auth = session.authenticate(pin)
            if not session.authenticated:
                continue
        for path in chat_paths:
            result = session.request("POST", path, body=body, timeout=10.0)
            status = int(result.get("status") or 0)
            if result.get("ok") and status and status not in (401, 403, 404, 405):
                result["endpoint"] = path
                result["host"] = candidate
                return result
    return {
        "ok": False,
        "error": "No LAN proactive message endpoint responded",
        "host": host,
        "port": port,
        "hint": (
            "ver0.2.3 LAN server has no assistant-message inject API. "
            "Desktop chat/VRM updates require Eel (embedded WebView only)."
        ),
    }


def lan_authenticated_status(
    *,
    host: str = DEFAULT_LAN_HOST,
    hosts: list[str] | None = None,
    port: int = DEFAULT_LAN_PORT,
    pin: str = "",
) -> dict[str, Any]:
    """Quick LAN auth + settings probe for status()/diagnostics."""
    for candidate in hosts or resolve_lan_hosts(host):
        if not lan_port_open(host=candidate, port=port):
            continue
        session = LanSession(host=candidate, port=port)
        auth = session.authenticate(pin) if pin else {"ok": True, "skipped": True}
        settings = session.request("GET", "/api/settings", timeout=5.0)
        png = session.request("GET", "/api/pngtuber", timeout=5.0)
        return {
            "ok": session.authenticated or not pin,
            "host": candidate,
            "port": port,
            "authenticated": session.authenticated,
            "auth": auth,
            "settings": settings.get("body"),
            "pngtuber": png.get("body"),
        }
    return {"ok": False, "error": f"LAN port {port} closed on all hosts"}
