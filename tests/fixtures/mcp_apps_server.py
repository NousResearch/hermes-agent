#!/usr/bin/env python3
"""Minimal MCP Apps server fixture (io.modelcontextprotocol/ui) for host tests.

Built on the official ``mcp`` SDK's low-level ``Server`` so protocol framing,
validation, and lifecycle come from the same library the ecosystem uses -- the
fixture only has to express the MCP Apps specifics Hermes' host depends on.

It reproduces the two card-delivery forms Hermes supports, plus the capability
gate real servers apply:

* ``inline_card`` -- the tool RESULT carries ``_meta.ui.resource`` (full HTML)
  and ``_meta.ui.csp``. Self-contained.
* ``referenced_card`` -- the tool DEFINITION carries ``_meta.ui.resourceUri``
  pointing at a ``ui://`` resource the host fetches with ``resources/read``.
  Emitted **only** when the client declared the UI extension at ``initialize``
  time, mirroring the ``supportsMCPApps`` gate on real MCP Apps servers (utp
  strips every tool ``_meta`` otherwise). This is what lets the fixture catch a
  host that silently stops advertising the extension.
* ``plain_tool`` -- no card at any layer; the negative control that proves card
  plumbing stays off for ordinary tools.

Two deliberate departures from the SDK's convenience decorators, both forced by
the MCP Apps shape:

* ``CallToolRequest`` is handled raw through ``server.request_handlers`` because
  the ``@server.call_tool()`` decorator builds the ``CallToolResult`` itself and
  offers no way to attach result-level ``_meta`` -- which is exactly where the
  inline card lives.
* ``extensions`` is attached to ``ServerCapabilities`` after
  ``create_initialization_options()``, since the SDK models it as an extra field
  rather than a declared one.

Run it directly to hand-drive a session::

    printf '%s\\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{}}}' \\
      | python3 tests/fixtures/mcp_apps_server.py
"""

import anyio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

UI_EXTENSION = "io.modelcontextprotocol/ui"
UI_MIME = "text/html;profile=mcp-app"

INLINE_URI = "ui://fixture/inline-card"
REFERENCED_URI = "ui://fixture/referenced-card"

# Deliberately padded: the host must keep card HTML out of the model-facing
# result, so a card big enough to be obviously wrong if it leaks makes that
# regression visible in assertions on payload size.
INLINE_HTML = (
    "<!DOCTYPE html><html><body><h1>inline fixture card</h1>"
    + "<!--" + ("x" * 2048) + "-->"
    + "</body></html>"
)
REFERENCED_HTML = (
    "<!DOCTYPE html><html><body><h1>referenced fixture card</h1>"
    + "<!--" + ("y" * 4096) + "-->"
    + "</body></html>"
)

CSP = {
    "scriptSrc": "'unsafe-inline' 'unsafe-eval' fixture.invalid",
    "connectDomains": ["fixture.invalid"],
    "resourceDomains": ["fixture.invalid"],
    "allowUnsafeEval": True,
}

server = Server("mcp-apps-fixture", version="1.0.0")


def _client_declared_ui() -> bool:
    """True when this session's client advertised the MCP Apps UI extension.

    ``ClientCapabilities`` declares no ``extensions`` field, so the negotiated
    value lands in pydantic's extras rather than a typed attribute.
    """
    try:
        params = server.request_context.session.client_params
    except LookupError:  # called outside a request context
        return False
    capabilities = getattr(params, "capabilities", None)
    extras = getattr(capabilities, "model_extra", None) or {}
    extensions = extras.get("extensions") or {}
    return UI_EXTENSION in extensions


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    referenced_meta = (
        {"ui": {"resourceUri": REFERENCED_URI, "csp": CSP}}
        if _client_declared_ui()
        else None
    )
    return [
        types.Tool(
            name="inline_card",
            description="Returns a result carrying an inline MCP Apps card.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="referenced_card",
            description="Card HTML lives in a ui:// resource, not the result.",
            inputSchema={"type": "object", "properties": {}},
            _meta=referenced_meta,
        ),
        types.Tool(
            name="plain_tool",
            description="An ordinary tool with no UI card.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def _handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name

    if name == "inline_card":
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(
                    type="text", text="inline card attached")],
                isError=False,
                _meta={
                    "ui": {
                        "resource": {
                            "uri": INLINE_URI,
                            "mimeType": UI_MIME,
                            "text": INLINE_HTML,
                        },
                        "csp": CSP,
                    }
                },
            )
        )

    if name == "referenced_card":
        # No ``_meta`` here on purpose: the host must resolve the card from the
        # tool DEFINITION's resourceUri via resources/read.
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(
                    type="text", text="referenced card attached")],
                structuredContent={"items": ["a", "b"]},
                isError=False,
            )
        )

    if name == "plain_tool":
        return types.ServerResult(
            types.CallToolResult(
                content=[types.TextContent(type="text", text="no card here")],
                isError=False,
            )
        )

    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(
                type="text", text=f"unknown tool: {name}")],
            isError=True,
        )
    )


server.request_handlers[types.CallToolRequest] = _handle_call_tool


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=AnyUrl(REFERENCED_URI),
            name="referenced card",
            mimeType=UI_MIME,
        )
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
    # AnyUrl normalization can append a trailing slash to an authority-only URI.
    if str(uri).rstrip("/") == REFERENCED_URI:
        return [ReadResourceContents(content=REFERENCED_HTML, mime_type=UI_MIME)]
    return []


async def main() -> None:
    options = server.create_initialization_options(NotificationOptions())
    # Advertise the UI extension back to the client. ``ServerCapabilities`` is
    # ``extra='allow'``, so this serializes under ``capabilities.extensions``.
    options.capabilities.extensions = {UI_EXTENSION: {"mimeTypes": [UI_MIME]}}

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


if __name__ == "__main__":
    anyio.run(main)
