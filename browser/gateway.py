"""browser.gateway — in-browser Hermes gateway host.

Runs inside the Pyodide worker. Owns:

- one logical WebSocket per page shim, each driven through the REAL
  upstream ``/api/ws`` ASGI route by httpx-ws (the same accept/guard/
  handle_ws path the dashboard's own websocket hits — no mirrored
  pre-accept logic here)
- the full upstream REST surface via ``httpx.ASGITransport`` invoking the
  real ``hermes_cli.web_server.app`` ASGI app in-process on the
  cooperative asyncio loop (browser.runtime) — every dashboard route,
  unchanged
- page-mediated outbound fetches (the httpx/urllib transports hand the
  request to the page, which performs fetch() — TLS and CORS are the
  browser's own); a synchronous coincident proxy call, so key material and
  req/resp matching never re-enter this interpreter as frame plumbing

JS contract (js.hermesBridge over coincident's sync proxy):
  emit(wsId, text)            — outbound text frame to a socket
  wsAccepted(wsId)            — socket is open on the wire
  wsClosed(wsId, code, reason)
  pump(ms) -> [str]           — park until the page hands over inbound frames
  fetchRequest(url, method, headersJson, bodyB64) -> dict
  restReply(id, status, headersJson, bodyB64)
  log(level, msg)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time

import httpx
from starlette.websockets import WebSocketDisconnect

# httpx-ws subclasses anyio.AsyncContextManagerMixin at import time; anyio
# 4.9 removed the deprecated mixin. Reproduce it faithfully: __aenter__
# delegates to the subclass's __asynccontextmanager__ (where httpx-ws builds
# its streams/task group) — a plain "return self" shim would skip that.
import anyio  # noqa: E402

if not hasattr(anyio, "AsyncContextManagerMixin"):
    import contextlib

    class _AsyncContextManagerMixin:
        @contextlib.asynccontextmanager
        async def __asynccontextmanager__(self):
            yield self

        async def __aenter__(self):
            self.__cm = self.__asynccontextmanager__()
            return await self.__cm.__aenter__()

        async def __aexit__(self, exc_type, exc_value, tb):
            return await self.__cm.__aexit__(exc_type, exc_value, tb)

    anyio.AsyncContextManagerMixin = _AsyncContextManagerMixin

import httpx_ws  # noqa: E402
import httpx_ws.transport  # noqa: E402


def _js():
    import js  # type: ignore

    return js.hermesBridge


_sockets: dict[int, "asyncio.Queue"] = {}
_app = None
_lifespan_cm = None
_session_token = ""
_public_host = ""


# --------------------------------------------------------------- socket glue
#
# Each page-side socket is an in-memory queue of inbound frames; a task
# drives the REAL upstream /api/ws ASGI route through httpx-ws (the same
# accept/guard/handle_ws path the dashboard's own websocket hits — no
# mirrored pre-accept logic here). Page frames queue -> aws.send_text;
# aws.receive_text -> emit() back to the page.


def ws_open(ws_id: int, path: str, headers: dict | None = None) -> None:
    q: asyncio.Queue = asyncio.Queue()
    _sockets[ws_id] = q
    from . import runtime as browser_runtime

    async def _run():
        loop = browser_runtime.get_loop()
        code, reason = 1000, ""
        try:
            transport = httpx_ws.transport.ASGIWebSocketTransport(app=_app)
            async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://browser.local") as client:
                async with httpx_ws.aconnect_ws(
                        path, client, headers=headers or {}) as aws:
                    _js().wsAccepted(ws_id)

                    async def _outbound():
                        while True:
                            item = await q.get()
                            if isinstance(item, BaseException):
                                raise item
                            await aws.send_text(item)

                    async def _inbound():
                        while True:
                            _js().emit(ws_id, await aws.receive_text())

                    tasks = [
                        asyncio.Task(_outbound(), loop=loop),
                        asyncio.Task(_inbound(), loop=loop),
                    ]
                    done, pending = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_EXCEPTION)
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if isinstance(exc, WebSocketDisconnect):
                            code, reason = exc.code, exc.reason or ""
                        elif exc is not None:
                            raise exc
        except WebSocketDisconnect as e:
            code, reason = e.code, e.reason or ""
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            _js().log("err", f"ws_open {ws_id}: {sys.exc_info()[1]!r}")
            code, reason = 1011, ""
        _sockets.pop(ws_id, None)
        _js().wsClosed(ws_id, code, reason)

    try:
        asyncio.Task(_run(), loop=browser_runtime.get_loop())
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _js().log("err", f"ws_open task creation failed: {sys.exc_info()[1]!r}")


def ws_close(ws_id: int) -> None:
    q = _sockets.pop(ws_id, None)
    if q is not None:
        q.put_nowait(WebSocketDisconnect(code=1000, reason="page closed"))


def ws_send(ws_id: int, data: str) -> None:
    q = _sockets.get(ws_id)
    if q is None:
        return
    q.put_nowait(data)


def _route_pump_frame(frame: dict) -> None:
    """Frames arriving while a handler is blocked in pump()."""
    t = frame.get("t")
    if t == "ws-send":
        ws_send(frame["id"], frame["data"])
    elif t == "ws-open":
        ws_open(frame["id"], frame.get("path", ""), frame.get("headers"))
    elif t == "ws-close":
        ws_close(frame["id"])
    elif t == "rest":
        _handle_rest(frame)


def handle(raw: str) -> None:
    """JS pump entry: one raw JSON frame handed over by pullInbound."""
    try:
        frame = json.loads(raw)
    except Exception:
        return
    _route_pump_frame(frame)


# ------------------------------------------------------------------- REST


def _handle_rest(frame: dict) -> None:
    """Route a page REST request through the real FastAPI app.

    httpx.ASGITransport drives the real app in-process — headers pass
    verbatim (the renderer already attaches X-Hermes-Session-Token).
    """
    rid = frame["id"]
    method = frame.get("method", "GET")
    raw_path = frame.get("path", "/")
    headers = frame.get("headers") or {}
    body = frame.get("body") or {}
    if "b64" in body:
        body_bytes = base64.b64decode(body["b64"])
    else:
        body_bytes = (body.get("text") or "").encode()

    async def _invoke():
        transport = httpx.ASGITransport(app=_app)
        async with httpx.AsyncClient(
                transport=transport,
                base_url="http://browser.local") as client:
            return await client.request(
                method, raw_path, headers=headers, content=body_bytes)

    def _reply(resp):
        # postMessage can't structured-clone a PyProxy — headers cross as
        # JSON. httpx.Headers is already str->str.
        _js().restReply(
            rid,
            resp.status_code,
            json.dumps(dict(resp.headers)),
            base64.b64encode(resp.content).decode(),
        )

    from . import runtime as browser_runtime

    def _done(task):
        import traceback

        exc = task.exception()
        if exc is not None:
            traceback.print_exception(exc)
            _js().restReply(
                rid, 500, json.dumps({"content-type": "application/json"}),
                base64.b64encode(json.dumps(
                    {"detail": f"internal error: {exc}"}).encode()).decode())
            return
        try:
            _reply(task.result())
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    # No run_sync here: pump is reentrant, so a synchronous wait inside
    # _route_pump_frame lets every new REST frame nest one stack level deeper
    # and the in-flight requests starve — the observed boot wedge. REST is
    # queued on the loop like a real server: the task steps during pump and
    # the reply goes out from its done-callback.
    try:
        _loop = browser_runtime.get_loop()
        t = asyncio.Task(_invoke(), loop=_loop)
        t.add_done_callback(_done)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        _js().log("err", f"rest task creation failed: {sys.exc_info()[1]!r}")


# --------------------------------------------------------------------- net


def _proxy_result(raw) -> dict:
    """coincident returns a JsProxy; normalize to a plain dict."""
    try:
        return raw.to_py() if hasattr(raw, "to_py") else dict(raw)
    except Exception:
        return {"error": "unserializable proxy result"}


def fetch_blocking(url: str, method: str, headers: dict, body_b64: str | None,
                   timeout_s: float = 120.0) -> dict:
    """Synchronous page-mediated fetch (a coincident proxy call parks the
    worker until the page resolves it — TLS and CORS belong to the
    browser). `timeout_s` is enforced page-side on the same channel."""
    resp = _proxy_result(
        _js().fetchRequest(url, method, json.dumps(headers),
                           body_b64 or ""))
    resp.setdefault("status", 599)
    resp.setdefault("headers", {})
    resp.setdefault("body", "")
    return resp


def install() -> None:
    """Bind the pump router so frames arriving mid-wait get routed."""
    from . import runtime as browser_runtime

    browser_runtime.set_frame_router(lambda f: _route_pump_frame(
        json.loads(f) if isinstance(f, str) else f))


def _boot_web_app() -> None:
    """Import the real FastAPI app, set serve-mode state, run ASGI lifespan."""
    global _app, _lifespan_cm
    from . import runtime as browser_runtime

    browser_runtime.get_loop()

    import os

    os.environ.setdefault("HERMES_SERVE_HEADLESS", "1")
    # The private-session token the page mints doubles as the dashboard
    # session token — same credential model as upstream serve.
    if _session_token:
        os.environ["HERMES_DASHBOARD_SESSION_TOKEN"] = _session_token

    sys.stderr.write("browser.gateway: importing web_server\n")
    from hermes_cli import web_server

    sys.stderr.write("browser.gateway: web_server imported\n")
    _app = web_server.app
    state = _app.state
    state.ui_surface = "serve"
    state.auth_required = False
    # Host/Origin validation accepts loopback aliases automatically; the
    # page's real hostname is the operator-declared public host — upstream's
    # `dashboard.public_url` → trusted_public_hosts path.
    state.trusted_public_hosts = (
        frozenset({_public_host.lower()}) if _public_host else frozenset())
    state.bound_host = "127.0.0.1"
    state.bound_port = 0
    state.initial_profile = "default"
    state.web_dist = None
    state.ssh_isolated_clients = set()

    sys.stderr.write("browser.gateway: entering lifespan\n")
    _lifespan_cm = _app.router.lifespan_context(_app)
    browser_runtime.run_sync(_lifespan_cm.__aenter__(), timeout=600)
    sys.stderr.write("browser.gateway boot: web_server app + lifespan up\n")


def _seed_models_dev_cache() -> None:
    """Seed ~/.hermes/models_dev_cache.json from the shipped models.dev
    snapshot when no real cache exists yet. Upstream's fetch_models_dev
    serves a disk snapshot of any age instantly and refreshes in the
    background — seeding removes the one blocking cold fetch that made
    model.save_key crawl on first use. The file is backdated past the
    registry TTL so the first read also kicks a live refresh."""
    import gzip

    home = os.environ.get("HERMES_HOME", "/hermes-home")
    cache_path = os.path.join(home, "models_dev_cache.json")
    seed = "/hermes-py/models-dev-seed.json.gz"
    if os.path.exists(cache_path) or not os.path.exists(seed):
        return
    try:
        payload = gzip.decompress(open(seed, "rb").read())
        with open(cache_path, "wb") as fh:
            fh.write(payload)
        etag_seed = "/hermes-py/models-dev-seed.etag"
        if os.path.exists(etag_seed):
            with open(etag_seed, encoding="utf-8") as fh:
                etag = fh.read().strip()
            if etag:
                with open(os.path.join(home, "models_dev_cache.etag"), "w", encoding="utf-8") as fh:
                    fh.write(etag)
        # Backdate past _MODELS_DEV_CACHE_TTL (4h): stage-3 disk reads serve
        # it instantly AND arm the background refresh on first use.
        stale = time.time() - (4 * 3600 + 60)
        os.utime(cache_path, (stale, stale))
        sys.stderr.write("browser.gateway: models.dev cache seeded from snapshot\n")
    except Exception as err:
        sys.stderr.write(f"browser.gateway: models.dev seed failed: {err}\n")


def boot(session_token: str = "", public_host: str = "") -> None:
    """Entry called by the worker once the FS is populated."""
    global _session_token, _public_host
    from . import bootstrap_py as browser_bootstrap
    from . import runtime as browser_runtime

    _session_token = session_token
    _public_host = public_host
    browser_runtime.install()
    browser_bootstrap.install()
    install()
    browser_runtime.get_loop()
    _seed_models_dev_cache()
    import tui_gateway.server as _server  # noqa: F401 — import sanity

    _boot_web_app()
    sys.stderr.write("browser.gateway boot: tui_gateway loaded\n")
