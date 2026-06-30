#!/usr/bin/env python3
"""A minimal in-process LSP server used by tests.

Speaks just enough LSP to drive :class:`agent.lsp.client.LSPClient`
through a full lifecycle: ``initialize``, ``initialized``,
``textDocument/didOpen``, ``textDocument/didChange``, then a
``textDocument/publishDiagnostics`` notification followed by
``shutdown`` + ``exit``.

Behaviour (all behaviours selectable via env var ``MOCK_LSP_SCRIPT``):

- ``"clean"`` — initialize, accept didOpen/didChange, push empty
  diagnostics on every open/change, exit cleanly on shutdown.
- ``"errors"`` — same as ``clean`` but the published diagnostics
  carry one severity-1 entry pointing at line 0:0.
- ``"crash"`` — exit immediately after responding to ``initialize``
  (simulates a crashing server).
- ``"slow"`` — same as ``clean`` but sleeps 1s before responding to
  ``initialize`` (lets us test timeout behaviour).

The script writes JSON-RPC framed messages to stdout and reads from
stdin.  No third-party dependencies — uses only stdlib so it runs
under whatever Python the test process picks up.
"""
from __future__ import annotations

import json
import os
import sys
import time


def read_message():
    """Read one Content-Length framed JSON-RPC message from stdin."""
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.rstrip(b"\r\n")
        if not line:
            break
        k, _, v = line.decode("ascii").partition(":")
        headers[k.strip().lower()] = v.strip()
    n = int(headers["content-length"])
    body = sys.stdin.buffer.read(n)
    return json.loads(body.decode("utf-8"))


def write_message(obj):
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def main():
    script = os.environ.get("MOCK_LSP_SCRIPT", "clean")

    while True:
        msg = read_message()
        if msg is None:
            return 0

        if "id" in msg and msg.get("method") == "initialize":
            if script == "slow":
                time.sleep(1.0)
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 1,  # Full
                            "diagnosticProvider": {"interFileDependencies": False, "workspaceDiagnostics": False},
                        },
                        "serverInfo": {"name": "mock-lsp", "version": "0.1"},
                    },
                }
            )
            if script == "crash":
                return 0
            continue

        if msg.get("method") == "initialized":
            continue

        if msg.get("method") == "workspace/didChangeConfiguration":
            continue

        if msg.get("method") == "workspace/didChangeWatchedFiles":
            continue

        if msg.get("method") in {"textDocument/didOpen", "textDocument/didChange"}:
            params = msg.get("params") or {}
            td = params.get("textDocument") or {}
            uri = td.get("uri", "")
            version = td.get("version", 0)
            diagnostics = []
            if script == "errors":
                diagnostics = [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 5},
                        },
                        "severity": 1,
                        "code": "MOCK001",
                        "source": "mock-lsp",
                        "message": "synthetic error from mock-lsp",
                    }
                ]
            write_message(
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/publishDiagnostics",
                    "params": {
                        "uri": uri,
                        "version": version,
                        "diagnostics": diagnostics,
                    },
                }
            )
            continue

        if msg.get("method") == "textDocument/diagnostic":
            # Pull endpoint — return empty.
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {"kind": "full", "items": []},
                }
            )
            continue

        if msg.get("method") == "textDocument/didSave":
            continue

        # --- LSP query methods (for the "query" script mode) ---
        # Exclude lifecycle methods: initialize, shutdown, and exit are
        # handled above — the query branch only catches code-intelligence
        # requests that the regular clean/errors script doesn't know about.
        if script == "query" and "id" in msg:
            method = msg.get("method", "")
            # Lifecycle requests should not fall into the query handler
            if method not in ("initialize", "shutdown", "exit"):
                req_id = msg["id"]
                params = msg.get("params") or {}
                result = _handle_query(method, params)
                if result is not None:
                    write_message({"jsonrpc": "2.0", "id": req_id, "result": result})
                    continue

        if msg.get("method") == "shutdown":
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            continue

        if msg.get("method") == "exit":
            return 0

        # Unknown request: respond with method-not-found.
        if "id" in msg:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "error": {"code": -32601, "message": f"method not found: {msg.get('method')}"},
                }
            )


# ---------------------------------------------------------------------------
# Query handler — responds to LSP query methods when script == "query"
# ---------------------------------------------------------------------------


def _handle_query(method: str, params: dict):
    """Dispatch an LSP query method and return a mock result, or None if unknown."""
    uri = (params.get("textDocument") or {}).get("uri", "") if "textDocument" in method else ""

    if method == "textDocument/definition":
        position = params.get("position", {})
        return {
            "uri": uri or "file:///mock.py",
            "range": {
                "start": {"line": position.get("line", 0) - 1 if position.get("line", 0) > 0 else 0, "character": 0},
                "end": {"line": position.get("line", 0) - 1 if position.get("line", 0) > 0 else 0, "character": 10},
            },
        }

    if method == "textDocument/references":
        return [
            {
                "uri": uri or "file:///mock.py",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 5},
                },
            },
            {
                "uri": uri.replace(".py", "_test.py") if uri else "file:///mock_test.py",
                "range": {
                    "start": {"line": 10, "character": 2},
                    "end": {"line": 10, "character": 7},
                },
            },
        ]

    if method == "textDocument/hover":
        return {
            "contents": {
                "kind": "markdown",
                "value": "**int**\n\nRepresents an integer type in Python.",
            },
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 0, "character": 1},
            },
        }

    if method == "textDocument/documentSymbol":
        return [
            {
                "name": "Foo",
                "kind": 5,  # Class
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 3, "character": 0},
                },
                "selectionRange": {
                    "start": {"line": 0, "character": 6},
                    "end": {"line": 0, "character": 9},
                },
                "children": [
                    {
                        "name": "bar",
                        "kind": 6,  # Method
                        "range": {
                            "start": {"line": 1, "character": 8},
                            "end": {"line": 2, "character": 12},
                        },
                        "selectionRange": {
                            "start": {"line": 1, "character": 8},
                            "end": {"line": 1, "character": 11},
                        },
                    },
                ],
            },
        ]

    if method == "workspace/symbol":
        query = params.get("query", "")
        return [
            {
                "name": f"{query.title()}Class",
                "kind": 5,
                "location": {
                    "uri": "file:///src/main.py",
                    "range": {
                        "start": {"line": 0, "character": 6},
                        "end": {"line": 0, "character": 10 + len(query)},
                    },
                },
            },
            {
                "name": f"{query.lower()}_function",
                "kind": 12,  # Function
                "location": {
                    "uri": "file:///src/utils.py",
                    "range": {
                        "start": {"line": 5, "character": 4},
                        "end": {"line": 5, "character": 4 + len(query) + 10},
                    },
                },
            },
        ]

    return None  # Let the unknown-request handler respond with method-not-found


if __name__ == "__main__":
    sys.exit(main())
