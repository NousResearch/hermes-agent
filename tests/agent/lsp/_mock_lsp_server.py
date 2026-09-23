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
- ``"oom_abort"`` — prints a V8-style out-of-memory trace to stderr and
  aborts (SIGABRT) before answering ``initialize`` — models a Node
  language-server whose heap ceiling is too small for the workspace.
- ``"slow"`` — same as ``clean`` but sleeps 1s before responding to
  ``initialize`` (lets us test timeout behaviour).
- ``"slow_tree"`` — like ``slow``, with a child that ignores SIGTERM and
  a launcher that exits on SIGTERM (tests hard process-tree cleanup).
- ``"stale"`` — pushes one error on ``didOpen``, then goes SILENT on
  ``didChange`` (no push) and rejects the pull endpoint with
  method-not-found.  Models a slow tsserver that hasn't re-checked
  the edited content yet — the ghost-diagnostics scenario.
- ``"slow_push"`` — like ``stale`` on didOpen (one error) but on
  ``didChange`` sleeps ``MOCK_LSP_PUSH_DELAY`` seconds (default 1.0)
  and then pushes EMPTY diagnostics.  Models a server that fixes
  the ghost if you actually wait for it.  Pull endpoint rejects.
- ``"versionless"`` — errors on ``didOpen``, clean on ``didChange``, and
  no ``version`` field in any publishDiagnostics (the client credits
  each push with its current document version at receipt).  Push-only:
  the pull endpoint rejects.
- ``"incremental"`` — apply ranged edits as UTF-16 and expose the server's
  document mirror through hover, for synchronization contract tests.
- ``"clean_eof"`` — closes stdout after ``didOpen`` but keeps the
  process and stdin alive.
- ``"malformed_frame"`` — writes an invalid frame after ``didOpen``,
  then keeps the process and stdin alive.

The script writes JSON-RPC framed messages to stdout and reads from
stdin.  No third-party dependencies — uses only stdlib so it runs
under whatever Python the test process picks up.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
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
    documents = {}
    if script == "slow_tree":
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os, pathlib, signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path(os.environ['MOCK_LSP_CHILD_PID']).write_text(str(os.getpid())); "
                "time.sleep(60)",
            ],
            env=os.environ,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if script == "oom_abort":
        sys.stderr.write(
            "<oproject>:28982 ms: Mark-Compact 2041.4 (2055.6) -> 2038.4 (2058.4) MB\n"
            "  1317.07 ms (average mu = 0.307, current mu = 0.134)\n"
            "<--- Last few GCs --->\n"
            "FATAL ERROR: Reached heap limit Allocation failed - JavaScript heap out of memory\n"
        )
        sys.stderr.flush()
        os.abort()

    while True:
        msg = read_message()
        if msg is None:
            return 0

        if "id" in msg and msg.get("method") == "initialize":
            if script == "init_error":
                # A conformant JSON-RPC error to `initialize`, then exit: the client must
                # surface it as an LSPRequestError carrying the exit details.
                write_message({"jsonrpc": "2.0", "id": msg["id"],
                               "error": {"code": -32602, "message": "bad init"}})
                return 0
            if script in {"slow", "slow_tree"}:
                time.sleep(1.0)
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": msg["id"],
                    "result": {
                        "capabilities": {
                            "textDocumentSync": 2 if script == "incremental" else 1,
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

        if msg.get("method") == "workspace/didChangeWorkspaceFolders":
            # Multi-root tests observe attached folders through this log.
            log_path = os.environ.get("MOCK_LSP_FOLDERS_LOG")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(msg.get("params")) + "\n")
            continue

        if msg.get("method") in {"textDocument/didOpen", "textDocument/didChange"}:
            params = msg.get("params") or {}
            td = params.get("textDocument") or {}
            uri = td.get("uri", "")
            version = td.get("version", 0)
            is_change = msg.get("method") == "textDocument/didChange"
            if script == "incremental":
                text = documents.get(uri, "") if is_change else td["text"]
                for change in params.get("contentChanges", []):
                    if "range" not in change:
                        text = change["text"]
                        continue
                    # Apply the range using the protocol's UTF-16 offsets, as a real server does.
                    lines = text.splitlines(keepends=True)
                    def byte_offset(position):
                        prefix = "".join(lines[:position["line"]])
                        return len(prefix.encode("utf-16-le")) + position["character"] * 2
                    start = byte_offset(change["range"]["start"])
                    end = byte_offset(change["range"]["end"])
                    encoded = text.encode("utf-16-le")
                    text = (encoded[:start] + change["text"].encode("utf-16-le") + encoded[end:]).decode("utf-16-le")
                documents[uri] = text
            if not is_change and script in {"clean_eof", "malformed_frame"}:
                if script == "malformed_frame":
                    sys.stdout.buffer.write(b"Content-Length: invalid\r\n\r\n")
                    sys.stdout.buffer.flush()
                os.close(sys.stdout.fileno())
                while read_message() is not None:
                    pass
                return 0
            error_diag = [
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
            if script == "silent":
                # Never publishes diagnostics and the pull endpoint is
                # rejected (below).  Models a slow server that only
                # re-reports after a didChange it never receives — the
                # baseline snapshot has to wait out its full budget.
                continue
            if script == "stale":
                # Ghost scenario: publish an error for the ORIGINAL
                # content, then never publish again after edits.
                if not is_change:
                    write_message(
                        {
                            "jsonrpc": "2.0",
                            "method": "textDocument/publishDiagnostics",
                            "params": {"uri": uri, "version": version, "diagnostics": error_diag},
                        }
                    )
                continue
            if script == "slow_push":
                diagnostics = error_diag
                if is_change:
                    time.sleep(float(os.environ.get("MOCK_LSP_PUSH_DELAY", "1.0")))
                    diagnostics = []
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "version": version, "diagnostics": diagnostics},
                    }
                )
                continue
            diagnostics = []
            if script == "errors":
                diagnostics = error_diag
            if script == "versionless":
                # Servers that never echo a document version: the client credits the
                # push with its current version at receipt.
                diagnostics = [] if is_change else error_diag
            params = {"uri": uri, "version": version, "diagnostics": diagnostics}
            if script == "versionless":
                del params["version"]
            write_message({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": params})
            continue

        if script == "incremental" and msg.get("method") == "textDocument/hover":
            uri = msg["params"]["textDocument"]["uri"]
            write_message({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "contents": {"kind": "plaintext", "value": documents[uri]},
            }})
            continue

        if msg.get("method") == "textDocument/diagnostic":
            if script in {"stale", "slow_push", "versionless", "silent"}:
                # These scripts model push-only servers so the ghost
                # can't be papered over by the pull channel.
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": msg["id"],
                        "error": {"code": -32601, "message": "method not found"},
                    }
                )
                continue
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


if __name__ == "__main__":
    sys.exit(main())
