"""A small, real (``mcp`` 2.x SDK) stand-in for the hosted artifacts MCP server.

It implements the client-facing wire of the hosted artifacts plane: stateless Streamable HTTP at
``/v1/artifacts/mcp``, a required ``Authorization: Bearer`` header, the six ``artifact_*`` tools
(results are JSON text plus ``structuredContent``), compare-and-set edits that answer a stale
``base_version`` with a ``stale_base`` tool error carrying ``head_version``, and the shared
``ui://hermes/artifact-viewer`` resource that ``artifact_show`` points at through
``_meta.ui.resourceUri``. Inputs are checked like the hosted server checks them: optional fields
are non-nullable, titles and edit lists non-empty, base64 must decode, files and versions have size
caps, shares need a known scope and fitting subjects, unknown versions are not_found, and the list
is paginated. State lives in memory for the life of the process.

Run it as a subprocess (``python tests/fakes/hosted_artifacts_server.py``) with:

* ``ARTIFACTS_FAKE_PORT_FILE`` — required: the bound port is written there once listening.
* ``ARTIFACTS_FAKE_PORT`` — port to bind (default 0, ephemeral).
* ``ARTIFACTS_FAKE_TOKEN`` — when set, the only bearer accepted; otherwise any non-empty bearer.
* ``ARTIFACTS_FAKE_LOG`` — JSONL file receiving each POSTed JSON-RPC body and its bearer.
* ``ARTIFACTS_FAKE_PORTAL`` — base of ``viewer_url`` (default ``http://portal.localhost``).

or in-process with :func:`serve_in_thread`, which returns the base URL and a stop callable.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import secrets
import sys
import threading
from typing import Any, Callable, Optional

from pydantic import StrictInt

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


_PATH_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
MAX_TITLE_CHARS = 200
MAX_FILES = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_VERSION_BYTES = 10 * 1024 * 1024
MAX_SUBJECTS = 200
MAX_SUBJECT_CHARS = 128
MAX_CURSOR_CHARS = 256
LIST_PAGE_SIZE = 50
SHARE_SCOPES = ("CREATOR_ONLY", "ORG", "MEMBERS", "GROUPS")


class Refused(Exception):
    """A contract error: ``body`` is the tool's error result."""

    def __init__(self, code: str, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.body = {"error": code, "message": message, **extra}


def _invalid(message: str) -> Refused:
    return Refused("invalid_request", message)


def _title(title: Any) -> str:
    if not isinstance(title, str) or not title.strip():
        raise _invalid("title must be a non-empty string")
    if len(title) > MAX_TITLE_CHARS:
        raise _invalid(f"title is longer than {MAX_TITLE_CHARS} characters")
    return title


def _files(files: Any) -> list[dict]:
    """A full file set, checked the way the hosted server checks it (paths, encodings, size caps)."""
    if not isinstance(files, list) or not files:
        raise _invalid("files must be a non-empty array")
    if len(files) > MAX_FILES:
        raise _invalid(f"at most {MAX_FILES} files per version")
    total, checked, seen = 0, [], set()
    for entry in files:
        if not isinstance(entry, dict):
            raise _invalid("each file must be an object")
        path = entry.get("path")
        if not isinstance(path, str) or not _PATH_RE.match(path) or ".." in path.split("/") or path in seen:
            raise _invalid(f"invalid or duplicate path: {path!r}")
        seen.add(path)
        content, encoding = entry.get("content"), entry.get("encoding", "utf8")
        if not isinstance(content, str):
            raise _invalid(f"{path}: content must be a string")
        if encoding == "utf8":
            size = len(content.encode("utf-8"))
        elif encoding == "base64":
            if not _BASE64_RE.match(content):
                raise _invalid(f"{path}: content is not valid base64")
            try:
                size = len(base64.b64decode(content, validate=True))
            except (binascii.Error, ValueError):
                raise _invalid(f"{path}: content is not valid base64") from None
        else:
            raise _invalid(f"{path}: encoding must be utf8 or base64")
        if size > MAX_FILE_BYTES:
            raise Refused("artifact_too_large", f"{path} exceeds {MAX_FILE_BYTES} bytes")
        total += size
        if total > MAX_VERSION_BYTES:
            raise Refused("artifact_too_large", f"version exceeds {MAX_VERSION_BYTES} bytes")
        checked.append({"path": path, "content": content, "encoding": encoding})
    return checked


class Store:
    """Owner-scoped artifacts with immutable numbered versions. Every method returns the tool's
    result body: the success payload, or ``{"error": <contract code>, "message": ...}``."""

    def __init__(self, port_ref: Callable[[], int], portal: str) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict[str, Any]] = {}
        self._clock = 0  # list order: most recently changed first
        self._port_ref = port_ref
        self._portal = portal.rstrip("/")

    def _urls(self, artifact_id: str, version: int) -> dict[str, str]:
        port = self._port_ref()
        return {"viewer_url": f"{self._portal}/a/{artifact_id}",
                "content_url": f"http://{artifact_id}.artifacts.localhost:{port}/__av?t=fake-{version}"}

    def _touch(self, item: dict) -> None:
        self._clock += 1
        item["changed"] = self._clock

    def _found(self, artifact_id: Any) -> dict:
        if not isinstance(artifact_id, str) or not artifact_id:
            raise _invalid("id must be a non-empty string")
        item = self._items.get(artifact_id)
        if item is None:
            raise Refused("not_found", "artifact not found")
        return item

    @staticmethod
    def _answer(run: Callable[[], dict]) -> dict:
        try:
            return run()
        except Refused as refused:
            return refused.body

    def create(self, title: Any, files: Any) -> dict:
        def run() -> dict:
            checked = (_title(title), _files(files))
            artifact_id = _artifact_id()
            with self._lock:
                item = {"title": checked[0], "versions": [checked[1]],
                        "share": {"scope": "CREATOR_ONLY", "subjects": [], "pin": "latest"}}
                self._touch(item)
                self._items[artifact_id] = item
            return {"id": artifact_id, "version": 1, **self._urls(artifact_id, 1)}
        return self._answer(run)

    def get(self, artifact_id: str) -> Optional[dict]:
        with self._lock:
            return self._items.get(artifact_id)

    def update(self, artifact_id: Any, base_version: Any, title: Any = None,
               edits: Any = None, files: Any = None) -> dict:
        def run() -> dict:
            with self._lock:
                item = self._found(artifact_id)
                if isinstance(base_version, bool) or not isinstance(base_version, int) or base_version < 0:
                    raise _invalid("base_version must be a non-negative integer")
                head = len(item["versions"])
                if base_version != head:
                    raise Refused("stale_base", f"base_version {base_version} is not the head ({head})",
                                  head_version=head)
                if (edits is None) == (files is None):
                    raise _invalid("give exactly one of edits or files")
                new_title = item["title"] if title is None else _title(title)
                new_files = _files(files) if files is not None else _apply_edits(item["versions"][-1], edits)
                item["versions"].append(new_files)
                item["title"] = new_title
                self._touch(item)
                version = len(item["versions"])
            return {"id": artifact_id, "version": version, **self._urls(artifact_id, version)}
        return self._answer(run)

    def _version(self, item: dict, version: Any) -> int:
        """The chosen version number (``None`` = the head)."""
        if version is None:
            return len(item["versions"])
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise _invalid("version must be a positive integer")
        if version > len(item["versions"]):
            raise Refused("not_found", "version not found")
        return version

    def show(self, artifact_id: Any, version: Any = None) -> dict:
        def run() -> dict:
            item = self._found(artifact_id)
            chosen = self._version(item, version)
            return {"id": artifact_id, "version": chosen, "title": item["title"], **self._urls(artifact_id, chosen)}
        return self._answer(run)

    def read(self, artifact_id: Any, version: Any = None) -> dict:
        """``artifact_get``: one version's files, validated before any lookup."""
        def run() -> dict:
            item = self._found(artifact_id)
            chosen = self._version(item, version)
            files = [{**f, "size": len(f["content"])} for f in item["versions"][chosen - 1]]
            return {"id": artifact_id, "title": item["title"], "kind": "html",
                    "head_version": len(item["versions"]), "version": chosen, "files": files,
                    "share": item["share"], "viewer_url": self._urls(artifact_id, chosen)["viewer_url"]}
        return self._answer(run)

    def share(self, artifact_id: Any, scope: Any, subjects: Any = None, pin: Any = None) -> dict:
        def run() -> dict:
            with self._lock:
                item = self._found(artifact_id)
                if scope not in SHARE_SCOPES:
                    raise _invalid("scope must be CREATOR_ONLY, ORG, MEMBERS or GROUPS")
                raw = [] if subjects is None else subjects
                if (not isinstance(raw, list) or len(raw) > MAX_SUBJECTS
                        or not all(isinstance(x, str) and 0 < len(x) <= MAX_SUBJECT_CHARS for x in raw)):
                    raise _invalid(f"subjects must be at most {MAX_SUBJECTS} non-empty strings")
                if scope in ("CREATOR_ONLY", "ORG") and raw:
                    raise _invalid(f"subjects must be empty for {scope}")
                chosen = "latest" if pin is None else pin
                if chosen != "latest" and (isinstance(chosen, bool) or not isinstance(chosen, int)
                                           or not 1 <= chosen <= len(item["versions"])):
                    raise _invalid('pin must be "latest" or a retained version number')
                item["share"] = {"scope": scope, "subjects": sorted(set(raw)), "pin": chosen}
                return dict(item["share"])
        return self._answer(run)

    def listing(self, cursor: Any = None) -> dict:
        """The caller's artifacts, most recently changed first, ``LIST_PAGE_SIZE`` per page."""
        def run() -> dict:
            if cursor is not None and (not isinstance(cursor, str) or len(cursor) > MAX_CURSOR_CHARS):
                raise _invalid(f"cursor must be a string of at most {MAX_CURSOR_CHARS} chars")
            start = 0
            if cursor:
                if not cursor.isdigit():
                    raise _invalid("unknown cursor")
                start = int(cursor)
            with self._lock:
                ordered = sorted(self._items.items(), key=lambda kv: kv[1]["changed"], reverse=True)
                page = [{"id": k, "title": v["title"], "kind": "html", "head_version": len(v["versions"]),
                         "updated_at": "1970-01-01T00:00:00Z",
                         "share": {"scope": v["share"]["scope"], "pin": v["share"]["pin"]}}
                        for k, v in ordered[start:start + LIST_PAGE_SIZE]]
                more = start + LIST_PAGE_SIZE < len(ordered)
            return {"items": page, "next_cursor": str(start + LIST_PAGE_SIZE) if more else None}
        return self._answer(run)


def _apply_edits(files: list[dict], edits: Any) -> list[dict]:
    if not isinstance(edits, list) or not edits:
        raise _invalid("edits must be a non-empty array")
    by_path = {f["path"]: dict(f) for f in files}
    for edit in edits:
        if not isinstance(edit, dict) or not all(isinstance(edit.get(k), str) for k in ("path", "old", "new")):
            raise _invalid("each edit needs string path, old and new")
        target = by_path.get(edit["path"])
        if target is None or not edit["old"] or target["content"].count(edit["old"]) != 1:
            raise Refused("edit_not_applied", f"edit not applied: {edit['path']}")
        target["content"] = target["content"].replace(edit["old"], edit["new"], 1)
    return _files(list(by_path.values()))


def build_server(store: Store):
    from mcp.server import MCPServer
    from mcp_types import CallToolResult, TextContent

    server = MCPServer("hermes-artifacts")

    def result(payload: dict):
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))],
                              structured_content=payload, is_error="error" in payload)

    # Optional parameters are typed without null, like the hosted server's input schemas: an omitted
    # one arrives as None, an explicit null is refused by argument validation.
    @server.tool()
    def artifact_create(title: str, files: list[dict]):
        """Create an artifact from files ({path, content, encoding})."""
        return result(store.create(title, files))

    @server.tool()
    def artifact_update(id: str, base_version: StrictInt, title: str = None,  # noqa: RUF013 -- null is refused
                        edits: list[dict] = None, files: list[dict] = None):  # noqa: RUF013
        """New version from exact {path, old, new} edits or a full file set; base_version must be the head."""
        return result(store.update(id, base_version, title, edits, files))

    @server.tool()
    def artifact_get(id: str, version: StrictInt = None):  # null is refused
        """Read one version's files."""
        return result(store.read(id, version))

    @server.tool()
    def artifact_list(cursor: str = None):  # noqa: RUF013 -- null is refused
        """List the caller's artifacts, newest first; pass next_cursor for the next page."""
        return result(store.listing(cursor))

    @server.tool()
    def artifact_share(id: str, scope: str, subjects: list[str] = None,  # noqa: RUF013 -- null is refused
                       pin: str | StrictInt = None):
        """Replace the artifact's share."""
        return result(store.share(id, scope, subjects, pin))

    @server.tool(meta={"ui": {"resourceUri": VIEWER_URI}})
    def artifact_show(id: str, version: StrictInt = None):  # null is refused
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
