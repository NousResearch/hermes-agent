"""Real MCP server (installed ``mcp`` 2.x SDK) for the MCP + plugin conformance suite.

Runs over stdio (default) or streamable HTTP (``MCPE2E_TRANSPORT=http``) and records
every JSON-RPC message it RECEIVES, byte-exact as parsed JSON, to ``MCPE2E_LOG``
(one object per line). Tests assert on what the server actually got on the wire
(arguments, ``_meta``), never on what Hermes believes it sent.

Environment knobs (all optional):

* ``MCPE2E_LOG`` — JSONL path for inbound messages (stdio: teed from fd 0; HTTP: ASGI body).
* ``MCPE2E_NAME`` — server name advertised in ``initialize`` (default ``e2e``).
* ``MCPE2E_CANARY`` — returned by every tool so a test can prove a REAL round trip.
* ``MCPE2E_ECHO_ENV`` — name of an env var whose value ``env_echo`` returns (``${VAR}`` checks).
* ``MCPE2E_RESOURCE_ONLY=1`` — advertise one resource and NO tools.
* ``MCPE2E_TRANSPORT=http`` + ``MCPE2E_PORT_FILE`` — serve streamable HTTP on
  ``MCPE2E_PORT`` (0 = ephemeral) and write the bound port to the file.
* ``MCPE2E_401_CALLS=<path>`` — HTTP only: while the file holds a positive integer N,
  the next ``tools/call`` is answered ``401`` and N is decremented (the fault survives
  a server restart because the budget lives on disk).
"""

from __future__ import annotations

import base64
import json
import os
import sys
import threading

_LOG_LOCK = threading.Lock()


def _log(msg: object) -> None:
    path = os.environ.get("MCPE2E_LOG")
    if not path:
        return
    items = msg if isinstance(msg, list) else [msg]
    with _LOG_LOCK, open(path, "a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps({"pid": os.getpid(), "msg": item}) + "\n")


def _log_raw_line(line: bytes) -> None:
    line = line.strip()
    if not line:
        return
    try:
        _log(json.loads(line))
    except ValueError:
        _log({"unparsed": line.decode("utf-8", "replace")})


def _tee_stdin() -> None:
    """Swap fd 0 for a pipe fed by a pump thread that logs each newline-delimited frame."""
    read_end, write_end = os.pipe()
    upstream = os.dup(0)
    os.dup2(read_end, 0)
    os.close(read_end)

    def pump() -> None:
        pending = b""
        with os.fdopen(upstream, "rb", buffering=0) as src, os.fdopen(write_end, "wb", buffering=0) as dst:
            while True:
                chunk = src.read(65536)
                if not chunk:
                    break
                pending += chunk
                *lines, pending = pending.split(b"\n")
                for line in lines:
                    _log_raw_line(line)
                dst.write(chunk)
        _log_raw_line(pending)

    threading.Thread(target=pump, name="mcpe2e-tee", daemon=True).start()


# 1x1 PNG (valid magic bytes; the image cache accepts it).
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
# Formats the image cache cannot store: (mime, raw bytes or None for "send the data string verbatim").
IMAGE_FORMATS: dict[str, tuple[str, bytes | str]] = {
    "png": ("image/png", _PNG),
    "svg": ("image/svg+xml", b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'),
    "avif": ("image/avif", b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00avifmif1miaf"),
    "tiff": ("image/tiff", b"II*\x00\x08\x00\x00\x00\x00\x00"),
    "heic": ("image/heic", b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic"),
    "badb64": ("image/png", "@@not-base64@@"),
}


def build_server():
    from mcp.server import MCPServer
    from mcp_types import ImageContent, TextContent, ToolAnnotations

    canary = os.environ.get("MCPE2E_CANARY", "NO-CANARY")
    server = MCPServer(os.environ.get("MCPE2E_NAME", "e2e"))

    if os.environ.get("MCPE2E_RESOURCE_ONLY") == "1":
        @server.resource("e2e://doc", name="doc", description="the only thing this server offers")
        def doc() -> str:
            return f"DOC:{canary}"
        return server

    @server.tool(annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False))
    def ro_probe(nonce: str = "") -> str:
        """Read-only probe (annotated readOnlyHint=true)."""
        return f"RO:{canary}:{nonce}"

    @server.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
    def rw_probe(nonce: str = "") -> str:
        """Destructive probe (annotated destructiveHint=true)."""
        return f"RW:{canary}:{nonce}"

    @server.tool()
    def noargs_probe() -> str:
        """Takes no parameters at all."""
        return f"NOARGS:{canary}"

    @server.tool()
    def optional_obj_probe(operation: str = "list", parameters: dict | None = None) -> str:
        """No required params; ``parameters`` is an optional JSON object."""
        return f"OPT:{canary}:{operation}:{json.dumps(parameters, sort_keys=True)}"

    @server.tool()
    def image_probe(fmt: str = "png") -> list:
        """Return a status line plus one image block in the requested format."""
        mime, payload = IMAGE_FORMATS[fmt]
        data = payload if isinstance(payload, str) else base64.b64encode(payload).decode()
        return [TextContent(type="text", text=f"IMG-STATUS:{canary}:{fmt}"),
                ImageContent(type="image", mime_type=mime, data=data)]

    @server.tool()
    def env_echo() -> str:
        """Return the value of the env var named by MCPE2E_ECHO_ENV."""
        name = os.environ.get("MCPE2E_ECHO_ENV", "")
        return f"ENV:{canary}:{os.environ.get(name, '<unset>')}"

    @server.tool()
    def crash_probe() -> str:
        """Kill this server process while the call is in flight (no response is ever sent)."""
        _log({"crash_probe_pid": os.getpid()})
        os._exit(7)

    return server


def _take_401(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            left = int(fh.read().strip() or 0)
    except (OSError, ValueError):
        return False
    if left <= 0:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(left - 1))
    return True


def _recording_app(inner):
    """ASGI wrapper: log each POSTed JSON-RPC body; optionally answer tools/call with 401."""
    fault_file = os.environ.get("MCPE2E_401_CALLS")

    async def app(scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            await inner(scope, receive, send)
            return
        chunks = []
        while True:
            event = await receive()
            chunks.append(event.get("body", b""))
            if not event.get("more_body"):
                break
        body = b"".join(chunks)
        try:
            parsed = json.loads(body or b"null")
        except ValueError:
            parsed = {"unparsed": body.decode("utf-8", "replace")}
        _log(parsed)
        is_call = isinstance(parsed, dict) and parsed.get("method") == "tools/call"
        if is_call and fault_file and _take_401(fault_file):
            _log({"injected_401_for": parsed.get("id")})
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
            return
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await inner(scope, replay, send)

    return app


def _serve_http(server) -> None:
    import socket

    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(os.environ.get("MCPE2E_PORT", "0") or 0)))
    port = sock.getsockname()[1]
    app = _recording_app(server.streamable_http_app())
    config = uvicorn.Config(app, log_level="warning", lifespan="on")
    userver = uvicorn.Server(config)
    port_file = os.environ.get("MCPE2E_PORT_FILE")

    async def run():
        import asyncio

        task = asyncio.create_task(userver.serve(sockets=[sock]))
        while not userver.started:
            await asyncio.sleep(0.01)
        if port_file:
            tmp = port_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(f"{port} {os.getpid()}")
            os.replace(tmp, port_file)
        await task

    import asyncio

    asyncio.run(run())


def main() -> None:
    if os.environ.get("MCPE2E_TRANSPORT") == "http":
        _serve_http(build_server())
        return
    _tee_stdin()
    build_server().run(transport="stdio")


if __name__ == "__main__":
    sys.exit(main())
