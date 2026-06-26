"""Minimal Eel WebSocket RPC client for AI Partner OS."""

from __future__ import annotations

import json
import random
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

try:
    import websockets.sync.client as ws_client
except ImportError:  # pragma: no cover
    ws_client = None  # type: ignore[assignment]


DEFAULT_EEL_PORTS = (8000, 8080, 8888, 8001, 8002, 8898, 9000)
DEFAULT_EEL_PAGES = ("index.html", "web/index.html", "app/index.html", "main.html")


def _http_reachable(host: str, port: int, *, timeout: float = 1.5) -> bool:
    for path in ("/", "/index.html"):
        try:
            with urlopen(f"http://{host}:{port}{path}", timeout=timeout) as resp:
                if resp.status < 500:
                    return True
        except (URLError, OSError, TimeoutError, ValueError):
            continue
    return False


def _rpc_probe(host: str, port: int, page: str, *, timeout: float = 4.0) -> dict[str, Any]:
    if ws_client is None:
        return {"ok": False, "error": "websockets package is not installed"}
    client = EelClient(host=host, port=port, page=page)
    try:
        client.connect(timeout=timeout)
        value = client.call("get_lan_status", timeout=min(timeout, 8.0))
        return {"ok": True, "port": port, "page": page, "lan_status": value}
    except Exception as exc:
        return {"ok": False, "port": port, "page": page, "error": str(exc)}
    finally:
        client.close()


def detect_eel_endpoint(
    host: str = "127.0.0.1",
    ports: tuple[int, ...] = DEFAULT_EEL_PORTS,
    pages: tuple[str, ...] = DEFAULT_EEL_PAGES,
) -> dict[str, Any] | None:
    """Return first host/port/page combo where Eel RPC responds."""
    for port in ports:
        if not _http_reachable(host, port):
            continue
        for page in pages:
            probe = _rpc_probe(host, port, page)
            if probe.get("ok"):
                return {
                    "host": host,
                    "port": port,
                    "page": page,
                    "lan_status": probe.get("lan_status"),
                }
    return None


def probe_eel_endpoints(
    host: str = "127.0.0.1",
    ports: tuple[int, ...] = DEFAULT_EEL_PORTS,
    pages: tuple[str, ...] = DEFAULT_EEL_PAGES,
) -> dict[str, Any]:
    """Full scan report for diagnostics (CLI probe-eel)."""
    report: list[dict[str, Any]] = []
    for port in ports:
        http_ok = _http_reachable(host, port)
        entry: dict[str, Any] = {"port": port, "http_reachable": http_ok}
        if not http_ok:
            report.append(entry)
            continue
        for page in pages:
            probe = _rpc_probe(host, port, page)
            entry = {"port": port, "page": page, "http_reachable": True, **probe}
            report.append(entry)
            if probe.get("ok"):
                return {
                    "ok": True,
                    "host": host,
                    "endpoint": {"host": host, "port": port, "page": page},
                    "report": report,
                }
    return {
        "ok": False,
        "host": host,
        "report": report,
        "embedded_only_hint": (
            "AI Partner OS ver0.2.3 binds Eel inside the desktop WebView only. "
            "External processes cannot reach save_proactive_message / play_tts_on_pc / VRM controls."
        ),
    }


def detect_eel_port(host: str = "127.0.0.1", ports: tuple[int, ...] = DEFAULT_EEL_PORTS) -> int | None:
    endpoint = detect_eel_endpoint(host=host, ports=ports)
    return int(endpoint["port"]) if endpoint else None


class EelClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, *, page: str = "index.html") -> None:
        self.host = host
        self.port = int(port)
        self.page = page.lstrip("/") or "index.html"
        self._ws: Any = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/eel?page={self.page}"

    def connect(self, *, timeout: float = 5.0) -> None:
        if ws_client is None:
            raise RuntimeError("websockets package is not installed.")
        self._ws = ws_client.connect(self.ws_url, open_timeout=timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = self._ws.recv(timeout=max(0.1, deadline - time.time()))
            except TimeoutError:
                continue
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("status") == "ready":
                return
        return

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def call(self, name: str, *args: Any, timeout: float = 120.0) -> Any:
        if self._ws is None:
            raise RuntimeError("EelClient is not connected.")
        call_id = f"{int(time.time() * 1000)}{random.randint(0, 9999)}"
        payload = {"call": call_id, "name": name, "args": list(args)}
        self._ws.send(json.dumps(payload, ensure_ascii=False))
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self._ws.recv(timeout=max(0.1, deadline - time.time()))
            message = json.loads(raw)
            if not isinstance(message, dict):
                continue
            if message.get("return") != call_id:
                continue
            if message.get("status") == "ok":
                return message.get("value")
            error = message.get("error") or "unknown Eel error"
            raise RuntimeError(str(error))
        raise TimeoutError(f"Eel call {name!r} timed out after {timeout}s")


def eel_call(
    name: str,
    *args: Any,
    host: str = "127.0.0.1",
    port: int | None = None,
    page: str | None = None,
    timeout: float = 120.0,
) -> Any:
    candidate_ports = (port,) + DEFAULT_EEL_PORTS if port else DEFAULT_EEL_PORTS
    if not any(_http_reachable(host, int(p), timeout=0.6) for p in candidate_ports):
        raise RuntimeError(
            "Eel RPC endpoint not found (AI Partner OS desktop WebView only — no external TCP listener)."
        )
    endpoint = None
    if port is None or page is None:
        endpoint = detect_eel_endpoint(host=host, ports=candidate_ports)
    chosen_port = port or (endpoint or {}).get("port") or DEFAULT_EEL_PORTS[0]
    chosen_page = page or (endpoint or {}).get("page") or DEFAULT_EEL_PAGES[0]
    client = EelClient(host=host, port=int(chosen_port), page=str(chosen_page))
    try:
        client.connect(timeout=min(10.0, timeout))
        return client.call(name, *args, timeout=timeout)
    finally:
        client.close()


def eel_available(host: str = "127.0.0.1", port: int | None = None) -> dict[str, Any]:
    endpoint = detect_eel_endpoint(host=host, ports=(port,) + DEFAULT_EEL_PORTS if port else DEFAULT_EEL_PORTS)
    if not endpoint:
        http_only_port = detect_eel_port(host=host) if port is None else (port if _http_reachable(host, port or 0) else None)
        return {
            "ok": False,
            "rpc_ok": False,
            "error": "Eel RPC endpoint not found",
            "host": host,
            "http_only_port": http_only_port,
            "hint": "AI Partner OS desktop UI must be open. Eel WebSocket is local-only.",
        }
    return {
        "ok": True,
        "rpc_ok": True,
        "host": endpoint["host"],
        "port": endpoint["port"],
        "page": endpoint["page"],
        "lan_status": endpoint.get("lan_status"),
    }
