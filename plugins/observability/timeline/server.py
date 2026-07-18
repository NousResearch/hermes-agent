"""Local HTTP server for the Hermes timeline dashboard."""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, cast
from urllib.parse import parse_qs, urlencode, urlparse

from .dashboard import dashboard_data, render_dashboard_html


class TimelineServer(ThreadingHTTPServer):
    """HTTP server carrying timeline query dependencies."""

    def __init__(self, server_address, RequestHandlerClass, *, deps: dict[str, Any], default_filters: dict[str, Any]):
        super().__init__(server_address, RequestHandlerClass)
        self.deps = deps
        self.default_filters = default_filters


def _one(params: dict[str, list[str]], name: str, default: str = "") -> str:
    values = params.get(name)
    if not values:
        return default
    return values[0]


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return max(1, min(500, int(_one(params, name, str(default)))))
    except Exception:
        return default


def _filters_from_query(params: dict[str, list[str]], defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        "limit": _int_param(params, "limit", int(defaults.get("limit") or 50)),
        "platform": _one(params, "platform", str(defaults.get("platform") or "")),
        "source": _one(params, "source", str(defaults.get("source") or "")),
        "chat_id": _one(params, "chat_id", str(defaults.get("chat_id") or "")),
        "thread_id": _one(params, "thread_id", str(defaults.get("thread_id") or "")),
    }


class TimelineRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003 - inherited API
        return

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _timeline_data(self, filters: dict[str, Any]) -> dict[str, Any]:
        server = cast(TimelineServer, self.server)
        return dashboard_data(
            list_runs=server.deps["list_runs"],
            list_thread_runs=server.deps["list_thread_runs"],
            get_run=server.deps["get_run"],
            iso=server.deps["iso"],
            **filters,
        )

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        server = cast(TimelineServer, self.server)
        filters = _filters_from_query(params, server.default_filters)
        if parsed.path in {"/api/timeline", "/api/timeline/"}:
            body = json.dumps(self._timeline_data(filters), ensure_ascii=False, default=str).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")
            return
        if parsed.path in {"/", "/index.html", "/timeline"}:
            query = urlencode({k: v for k, v in filters.items() if v not in ("", None)})
            api_url = "/api/timeline" + (f"?{query}" if query else "")
            html = render_dashboard_html(self._timeline_data(filters), api_url=api_url, poll_ms=2000)
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send_bytes(b"Not found", "text/plain; charset=utf-8", status=404)


def serve_dashboard(
    *,
    list_runs: Callable[..., list[dict[str, Any]]],
    list_thread_runs: Callable[..., list[dict[str, Any]]],
    get_run: Callable[[str], tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    iso: Callable[..., str],
    host: str = "127.0.0.1",
    port: int = 8765,
    limit: int = 50,
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    thread_id: str = "",
    open_browser: bool = False,
) -> None:
    """Run a local timeline dashboard server until interrupted."""
    defaults = {
        "limit": limit,
        "platform": platform,
        "source": source,
        "chat_id": chat_id,
        "thread_id": thread_id,
    }
    deps = {
        "list_runs": list_runs,
        "list_thread_runs": list_thread_runs,
        "get_run": get_run,
        "iso": iso,
    }
    server = TimelineServer((host, int(port)), TimelineRequestHandler, deps=deps, default_filters=defaults)
    url = f"http://{host}:{server.server_port}/"
    print(f"Hermes timeline dashboard serving at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
