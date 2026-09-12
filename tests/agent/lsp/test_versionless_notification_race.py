"""Regression test: a versionless publishDiagnostics reply arriving while
``open_file``'s didChange send is still in flight must be tagged with the
version that send is advancing to, not the one it's superseding — otherwise
``wait_for_diagnostics`` rejects a genuinely fresh push as stale.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PEER = r'''

import json
import sys
import time


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.rstrip(b"\r\n")
        if not line:
            break
        key, _, value = line.decode("ascii").partition(":")
        headers[key.strip().lower()] = value.strip()
    body = sys.stdin.buffer.read(int(headers["content-length"]))
    return json.loads(body.decode("utf-8"))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(
        f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
    )
    sys.stdout.buffer.flush()


def publish(uri, version, diagnostics):
    send({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {"uri": uri, "version": version, "diagnostics": diagnostics},
    })


error = [{
    "range": {"start": {"line": 0, "character": 0},
              "end": {"line": 0, "character": 1}},
    "severity": 1,
    "code": "OLD001",
    "source": "audit-child",
    "message": "stale version must not be treated as current",
}]

while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {
            "capabilities": {"textDocumentSync": 1},
        }})
    elif method == "initialized":
        pass
    elif method in {"textDocument/didOpen", "textDocument/didChange"}:
        document = message.get("params", {}).get("textDocument", {})
        uri = document.get("uri", "")
        version = int(document.get("version", 0))
        if method.endswith("didOpen"):
            publish(uri, version, error)
        else:
            # Deliberately out-of-order: stale error first, current clean second.
            publish(uri, max(0, version - 1), error)
            time.sleep(0.03)
            publish(uri, version, [])
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": message["id"], "result": None})
    elif method == "exit":
        break

'''


@pytest.mark.asyncio
async def test_versionless_reply_during_notification_is_fresh(tmp_path, monkeypatch):
    from agent.lsp.client import LSPClient
    m = SimpleNamespace(_SERVER=PEER)
    text = m._SERVER.replace('"version": version, ', '').replace('publish(uri, max(0, version - 1), error)', 'pass').replace('time.sleep(0.03)', 'pass')
    script = tmp_path / 'versionless.py'
    script.write_text(text)
    p = tmp_path / 'x.py'
    p.write_text('bad\n')
    c = LSPClient(server_id='audit-versionless', workspace_root=str(tmp_path), command=[sys.executable, str(script)], cwd=str(tmp_path))
    await c.start()
    try:
        old = await c.open_file(str(p), language_id='python')
        assert await c.wait_for_diagnostics(str(p), old, timeout=2)
        send = c._send_notification

        async def paused_send(method, params):
            counter = c._push_counter
            await send(method, params)
            if method == 'textDocument/didChange':

                async def seen():
                    while c._push_counter == counter:
                        await asyncio.sleep(0.001)
                await asyncio.wait_for(seen(), 2)
        monkeypatch.setattr(c, '_send_notification', paused_send)
        p.write_text('clean\n')
        version = await c.open_file(str(p), language_id='python')
        assert await c.wait_for_diagnostics(str(p), version, timeout=0.3), 'Real versionless reply was tagged with old document version during legal I/O interleaving'
    finally:
        await c.shutdown()
