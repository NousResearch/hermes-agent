"""A small, real (``mcp`` 2.x SDK) stand-in for the hosted artifacts MCP server.

It implements the client-facing wire of the hosted artifacts plane: stateless Streamable HTTP at
``/v1/artifacts/mcp``, a required ``Authorization: Bearer`` header, the six ``artifact_*`` tools
(results are JSON text plus ``structuredContent``), compare-and-set edits that answer a stale
``base_version`` with a ``stale_base`` tool error carrying ``head_version``, and the shared
``ui://hermes/artifact-viewer`` resource that ``artifact_show`` points at through
``_meta.ui.resourceUri``. State lives in memory for the life of the process.

Run it as a subprocess (``python tests/fakes/hosted_artifacts_server.py``) with:

* ``ARTIFACTS_FAKE_PORT_FILE`` — required: the bound port is written there once listening.
* ``ARTIFACTS_FAKE_PORT`` — port to bind (default 0, ephemeral).
* ``ARTIFACTS_FAKE_TOKEN`` — when set, the only bearer accepted; otherwise any non-empty bearer.
* ``ARTIFACTS_FAKE_LOG`` — JSONL file receiving each POSTed JSON-RPC body and its bearer.
* ``ARTIFACTS_FAKE_PORTAL`` — base of ``viewer_url`` (default ``http://portal.localhost``).

or in-process with :func:`serve_in_thread`, which returns the base URL and a stop callable.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from typing import Any, Callable, Optional

MCP_PATH = "/v1/artifacts/mcp"
VIEWER_URI = "ui://hermes/artifact-viewer"
VIEWER_MIME = "text/html;profile=mcp-app"
_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"
VIEWER_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Artifact</title></head>
<body style="margin:0"><iframe id="v" sandbox="allow-scripts allow-same-origin"
style="border:0;width:100%;height:100vh"></iframe></body></html>"""


def _artifact_id() -> str:
    value = secrets.randbits(128)
    return "".join(_CROCKFORD[(value >> (5 * i)) & 31] for i in range(25, -1, -1))


class Store:
    """Owner-scoped artifacts with immutable numbered versions."""

    def __init__(self, port_ref: Callable[[], int], portal: str) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._port_ref = port_ref
        self._portal = portal.rstrip("/")

    def _urls(self, artifact_id: str, version: int) -> dict[str, str]:
        port = self._port_ref()
        return {"viewer_url": f"{self._portal}/a/{artifact_id}",
                "content_url": f"http://{artifact_id}.artifacts.localhost:{port}/__av?t=fake-{version}"}

    def create(self, title: str, files: list[dict]) -> dict:
        artifact_id = _artifact_id()
        with self._lock:
            self._items[artifact_id] = {"title": title, "versions": [list(files)],
                                        "share": {"scope": "CREATOR_ONLY", "subjects": [], "pin": "latest"}}
        return {"id": artifact_id, "version": 1, **self._urls(artifact_id, 1)}

    def get(self, artifact_id: str) -> Optional[dict]:
        with self._lock:
            return self._items.get(artifact_id)

    def update(self, artifact_id: str, base_version: int, title: Optional[str],
               edits: Optional[list[dict]], files: Optional[list[dict]]) -> dict:
        with self._lock:
            item = self._items.get(artifact_id)
            if item is None:
                return {"error": "not_found", "message": "artifact not found"}
            head = len(item["versions"])
            if base_version != head:
                return {"error": "stale_base", "head_version": head,
                        "message": f"base_version {base_version} is not the head ({head})"}
            if (edits is None) == (files is None):
                return {"error": "invalid_request", "message": "send exactly one of edits or files"}
            new_files = list(files) if files is not None else _apply_edits(item["versions"][-1], edits or [])
            if isinstance(new_files, dict):
                return new_files
            item["versions"].append(new_files)
            if title:
                item["title"] = title
            version = len(item["versions"])
        return {"id": artifact_id, "version": version, **self._urls(artifact_id, version)}

    def _version(self, item: dict, version: Optional[int]) -> Any:
        """The chosen version number (``None`` = the head), or the contract error for a bad one."""
        if version is None:
            return len(item["versions"])
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            return {"error": "invalid_request", "message": "version must be a positive integer"}
        if version > len(item["versions"]):
            return {"error": "not_found", "message": "version not found"}
        return version

    def show(self, artifact_id: str, version: Optional[int]) -> dict:
        item = self.get(artifact_id)
        if item is None:
            return {"error": "not_found", "message": "artifact not found"}
        chosen = self._version(item, version)
        if isinstance(chosen, dict):
            return chosen
        return {"id": artifact_id, "version": chosen, "title": item["title"], **self._urls(artifact_id, chosen)}

    def read(self, artifact_id: str, version: Optional[int]) -> dict:
        """``artifact_get``: one version's files, validated before any lookup."""
        item = self.get(artifact_id)
        if item is None:
            return {"error": "not_found", "message": "artifact not found"}
        chosen = self._version(item, version)
        if isinstance(chosen, dict):
            return chosen
        files = [{**f, "size": len(f.get("content", ""))} for f in item["versions"][chosen - 1]]
        return {"id": artifact_id, "title": item["title"], "kind": "html", "head_version": len(item["versions"]),
                "version": chosen, "files": files, "share": item["share"],
                "viewer_url": self._urls(artifact_id, chosen)["viewer_url"]}

    def listing(self) -> dict:
        with self._lock:
            items = [{"id": k, "title": v["title"], "kind": "html", "head_version": len(v["versions"]),
                      "updated_at": "1970-01-01T00:00:00Z",
                      "share": {"scope": v["share"]["scope"], "pin": v["share"]["pin"]}}
                     for k, v in self._items.items()]
        return {"items": items, "next_cursor": None}


def _apply_edits(files: list[dict], edits: list[dict]) -> Any:
    by_path = {f["path"]: dict(f) for f in files}
    for edit in edits:
        target = by_path.get(edit.get("path"))
        old = edit.get("old", "")
        if target is None or not old or target["content"].count(old) != 1:
            return {"error": "edit_not_applied", "message": f"edit not applied: {edit.get('path')}"}
        target["content"] = target["content"].replace(old, edit.get("new", ""), 1)
    return list(by_path.values())


def build_server(store: Store):
    from mcp.server import MCPServer
    from mcp_types import CallToolResult, TextContent

    server = MCPServer("hermes-artifacts")

    def result(payload: dict):
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))],
                              structured_content=payload, is_error="error" in payload)

    @server.tool()
    def artifact_create(title: str, files: list[dict]):
        """Create an artifact from files ({path, content, encoding})."""
        return result(store.create(title, files))

    @server.tool()
    def artifact_update(id: str, base_version: int, title: str | None = None,
                        edits: list[dict] | None = None, files: list[dict] | None = None):
        """New version from exact {path, old, new} edits or a full file set; base_version must be the head."""
        return result(store.update(id, base_version, title, edits, files))

    @server.tool()
    def artifact_get(id: str, version: int | None = None):
        """Read one version's files."""
        return result(store.read(id, version))

    @server.tool()
    def artifact_list(cursor: str | None = None):
        """List the caller's artifacts."""
        return result(store.listing())

    @server.tool()
    def artifact_share(id: str, scope: str, subjects: list[str] | None = None, pin: str | int = "latest"):
        """Replace the artifact's share."""
        item = store.get(id)
        if item is None:
            return result({"error": "not_found", "message": "artifact not found"})
        item["share"] = {"scope": scope, "subjects": list(subjects or []), "pin": pin}
        return result(dict(item["share"]))

    @server.tool(meta={"ui": {"resourceUri": VIEWER_URI}})
    def artifact_show(id: str, version: int | None = None):
        """Show an artifact version in the viewer."""
        return result(store.show(id, version))

    @server.resource(VIEWER_URI, name="artifact-viewer", mime_type=VIEWER_MIME)
    def viewer() -> str:
        return VIEWER_HTML

    return server


def _guarded(inner, expected_token: Optional[str], log_path: Optional[str]):
    """ASGI wrapper: 401 without the bearer, 404 off the MCP path, optional JSONL request log."""

    async def app(scope, receive, send):
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        if scope.get("path") != MCP_PATH:
            await _respond(send, 404, {"error": "not_found", "message": "not found"})
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not token or (expected_token and token != expected_token):
            await _respond(send, 401, {"error": "unauthorized", "message": "missing or invalid bearer"})
            return
        body = b""
        if scope.get("method") == "POST":
            chunks = []
            while True:
                event = await receive()
                chunks.append(event.get("body", b""))
                if not event.get("more_body"):
                    break
            body = b"".join(chunks)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:  # noqa: ASYNC230 -- tiny local test log
                    fh.write(json.dumps({"bearer": token, "body": body.decode("utf-8", "replace")}) + "\n")
        replayed = False

        async def replay():
            nonlocal replayed
            if scope.get("method") == "POST" and not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await inner(scope, replay, send)

    return app


async def _respond(send, status: int, payload: dict) -> None:
    await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": json.dumps(payload).encode()})


def _build_app(port_ref: Callable[[], int], *, token: Optional[str], log_path: Optional[str], portal: str):
    store = Store(port_ref, portal)
    server = build_server(store)
    inner = server.streamable_http_app(streamable_http_path=MCP_PATH, stateless_http=True)
    return _guarded(inner, token, log_path), store


def serve_in_thread(*, token: Optional[str] = None, log_path: Optional[str] = None,
                    portal: str = "http://portal.localhost") -> tuple[str, Callable[[], None], Store]:
    """Serve on an ephemeral loopback port in a daemon thread: ``(base_url, stop, store)``."""
    import socket

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app, store = _build_app(lambda: port, token=token, log_path=log_path, portal=portal)
    userver = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=userver.run, kwargs={"sockets": [sock]}, name="artifacts-fake", daemon=True)
    thread.start()
    from tests.waiting import wait_until

    wait_until(lambda: userver.started or not thread.is_alive(), 30, what="fake artifacts server bind")

    def stop() -> None:
        userver.should_exit = True
        thread.join(timeout=10)
        sock.close()

    return f"http://127.0.0.1:{port}", stop, store


def main() -> int:
    import asyncio
    import socket

    import uvicorn

    port_file = os.environ["ARTIFACTS_FAKE_PORT_FILE"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(os.environ.get("ARTIFACTS_FAKE_PORT", "0") or 0)))
    port = sock.getsockname()[1]
    app, _ = _build_app(lambda: port, token=os.environ.get("ARTIFACTS_FAKE_TOKEN") or None,
                        log_path=os.environ.get("ARTIFACTS_FAKE_LOG") or None,
                        portal=os.environ.get("ARTIFACTS_FAKE_PORTAL", "http://portal.localhost"))
    userver = uvicorn.Server(uvicorn.Config(app, log_level="warning", lifespan="on"))

    async def run() -> None:
        from tests.waiting import async_wait_until

        task = asyncio.create_task(userver.serve(sockets=[sock]))
        await async_wait_until(lambda: userver.started or task.done(), 30, what="fake artifacts server bind")
        if task.done():
            await task  # surfaces the bind/startup error
            return
        tmp = port_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:  # noqa: ASYNC230 -- one small local write
            fh.write(f"{port} {os.getpid()}")
        os.replace(tmp, port_file)
        await task

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    sys.exit(main())
