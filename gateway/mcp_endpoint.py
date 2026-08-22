"""MCP Streamable HTTP endpoint chạy TRONG aiohttp API server của gateway.

FastMCP (mcp SDK) ``streamable_http_app()`` mount vào aiohttp qua ASGI bridge.
Mode phản hồi = SSE mặc định (quyết định user 2026-08-13 — KHÔNG json_response).

3 cạm bẫy đã xử lý (chi tiết: docs/spike-mcp-native.md trong repo stack):
  1. FastMCP cần lifespan task group (session manager) suốt vòng đời server —
     ``hold_mcp_lifespan()`` giữ context như background task (pattern
     ``_start_bg`` của spike).
  2. ASGI scope headers bắt buộc lowercase — aiohttp ``raw_headers`` giữ
     nguyên case, phải ``[(k.lower(), v) for k, v in ...]``.
  3. ``receive()`` KHÔNG trả ``http.disconnect`` sớm — sse-starlette
     ``_listen_for_disconnect`` sẽ cancel task group đang stream. Block vô
     thời hạn (``asyncio.Event().wait()``) là đúng; task group tự kết thúc
     khi response hoàn tất.

Cấu trúc tích hợp (api_server.py):
  - ``_http_route_table()``: thêm ``("*", "/mcp", self._handle_mcp)``; loop
    đăng ký route trong ``connect()`` tự mirror ``/p/{profile}/mcp``.
  - ``_handle_mcp``: auth ``self._check_auth(request)`` TRƯỚC khi vào bridge
    (API_SERVER_KEY, scope-aware theo /p/<profile>).
  - ``connect()``: build FastMCP app + start lifespan task (như ``_start_bg``);
    ``disconnect()``: cancel task, đóng session manager sạch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from aiohttp import web

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings

    MCP_AVAILABLE = True
except Exception:  # pragma: no cover — mcp SDK thiếu: endpoint tắt, không hard-fail
    FastMCP = None  # type: ignore[assignment]
    TransportSecuritySettings = None  # type: ignore[assignment]
    MCP_AVAILABLE = False

# Điểm mount cố định của FastMCP (settings.streamable_http_path mặc định).
# Starlette Route là exact-match nên request tới /p/<profile>/mcp phải được
# normalize scope path về đây (xem asgi_bridge) — query string giữ nguyên.
MCP_ENDPOINT_PATH = "/mcp"

# Scope lifespan (ASGI 3.0) — session manager anyio task group chạy trong đó.
_LIFESPAN_SCOPE = {
    "type": "lifespan",
    "asgi": {"version": "3.0", "spec_version": "2.4"},
    "state": {},
}


def build_mcp_server() -> Any:
    """FastMCP factory — Phase 1: 1 tool echo (Phase 2/3 thêm 9 exec + 8 admin)."""
    if not MCP_AVAILABLE:  # pragma: no cover
        return None
    # Pitfall 4 (docs/spike-mcp-native.md chưa ghi): FastMCP mặc định
    # host="127.0.0.1" -> transport_security chỉ cho phép Host LOOPBACK.
    # Gateway bind 172.17.0.1 (docker bridge) nên Host "172.17.0.1:8642"
    # bị 421 "Invalid Host header". Cho phép tường minh các host hợp lệ.
    _transport_security = TransportSecuritySettings(  # type: ignore[reportOptionalCall]
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "172.17.0.1:*",  # gateway bind (docker bridge — OWUI qua host.docker.internal)
            "host.docker.internal:*",
            "127.0.0.1:*",
            "localhost:*",
            "[::1]:*",
        ],
        allowed_origins=[
            "http://172.17.0.1:*",
            "http://host.docker.internal:*",
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
        ],
    )
    mcp = FastMCP(
        "hermes-agent-mcp",
        instructions="Hermes gateway MCP server (native endpoint 8642).",
        transport_security=_transport_security,
    )

    @mcp.tool()
    def echo(text: str) -> str:
        """Trả về đúng chuỗi nhập vào (tool smoke-test cho skeleton)."""
        return text

    # 9 execution tools (Phase 2 — 3 lớp safety trong mcp_tools_exec).
    from gateway.mcp_tools_exec import register_execution_tools

    register_execution_tools(mcp)

    # 8 admin tools + ACL qua Mongo stack (Phase 3 — FORK-1a).
    from gateway.mcp_tools_admin import register_admin_tools

    register_admin_tools(mcp)

    return mcp


async def hold_mcp_lifespan(mcp: Any, starlette_app: Any) -> None:
    """Giữ lifespan context (session manager task group) suốt vòng đời adapter.

    Chạy như background task trong connect() — tương đương on_startup của
    spike_server.py. Thiếu bước này, request đầu tiên dính
    ``RuntimeError: Task group is not initialized``.
    """
    ctx = getattr(starlette_app.router, "lifespan_context", None)
    if ctx is not None:
        async with ctx(_LIFESPAN_SCOPE):
            await asyncio.Event().wait()
    else:  # pragma: no cover — fallback: enter session manager trực tiếp
        mgr = getattr(mcp, "_session_manager", None)
        if mgr is None:
            logger.error("MCP: no lifespan context and no session manager")
            return
        async with mgr.run():
            await asyncio.Event().wait()


async def asgi_bridge(request: web.Request, starlette_app) -> web.StreamResponse:
    """ASGI -> aiohttp bridge cho một request (streamable HTTP, SSE).

    Auth phải được kiểm tra TRƯỚC khi gọi hàm này (api_server._handle_mcp).
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": request.method,
        "scheme": "http",
        # Normalize: /p/<profile>/mcp -> /mcp (Starlette Route exact-match).
        "path": MCP_ENDPOINT_PATH,
        "raw_path": MCP_ENDPOINT_PATH.encode(),
        "query_string": request.query_string.encode(),
        "root_path": "",
        # Cạm bẫy 2: headers lowercase (FastMCP validator đọc 'content-type').
        "headers": [(k.lower(), v) for k, v in request.raw_headers],
        "client": (request.remote or "127.0.0.1", 0),
        "server": (request.host.split(":")[0], request.url.port or 80),
        "state": {},
    }
    body = await request.read()
    body_done = False
    # Cạm bẫy 3: KHÔNG dùng request.wait_for_disconnection() — block vô hạn.
    _never = asyncio.Event()

    async def receive():
        nonlocal body_done
        if not body_done:
            body_done = True
            return {"type": "http.request", "body": body, "more_body": False}
        await _never.wait()
        return {"type": "http.disconnect"}

    status: list[int] = [200]
    resp_headers: list = []
    resp: Optional[web.StreamResponse] = None

    async def send(message):
        nonlocal resp
        t = message["type"]
        if t == "http.response.start":
            status[0] = message["status"]
            resp_headers[:] = message.get("headers", [])
            # Prepare NGAY (gửi status + headers) — GET SSE giữ stream mở
            # không byte nào thì client vẫn nhận được response headers.
            # send() được starlette await nên await prepare() hợp lệ.
            if resp is None:
                resp = web.StreamResponse(status=status[0])
                seen = set()
                for k, v in resp_headers:
                    kk = k.decode("latin-1")
                    # Để aiohttp tự tính content-length / transfer-encoding.
                    if kk.lower() in ("content-length", "transfer-encoding"):
                        continue
                    if kk not in seen:
                        resp.headers[kk] = v.decode("latin-1")
                        seen.add(kk)
                    else:
                        resp.headers.add(kk, v.decode("latin-1"))
            if not resp.prepared:
                await resp.prepare(request)
        elif t == "http.response.body":
            if resp is None:
                resp = web.StreamResponse(status=status[0])
                seen = set()
                for k, v in resp_headers:
                    kk = k.decode("latin-1")
                    if kk.lower() in ("content-length", "transfer-encoding"):
                        continue
                    if kk not in seen:
                        resp.headers[kk] = v.decode("latin-1")
                        seen.add(kk)
                    else:
                        resp.headers.add(kk, v.decode("latin-1"))
            if not resp.prepared:
                await resp.prepare(request)
            chunk = message.get("body", b"")
            if chunk:
                await resp.write(chunk)
            if not message.get("more_body", False):
                await resp.write_eof()

    await starlette_app(scope, receive, send)
    if resp is None:
        resp = web.Response(status=status[0])
    return resp
