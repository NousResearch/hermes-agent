"""Configurable heap sizing for spawned Node language servers (#116446).

``lsp.max_heap_mb`` is applied at spawn as ``--max-old-space-size`` via
``NODE_OPTIONS``, replacing any inherited value.  Coverage:

- token-level helper: strips both ``--flag=N`` and ``--flag N`` forms and
  keeps unrelated flags;
- end-to-end: a real ``LSPService._spawn_client`` spawn of a Node fake
  server whose stderr reports ``v8.getHeapStatistics().heap_size_limit``
  and the ``NODE_OPTIONS`` it actually received;
- the config default itself.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from agent.lsp.manager import LSPService, node_options_with_heap
from agent.lsp.servers import ServerContext, ServerDef, SpawnSpec
from hermes_cli.config_defaults import DEFAULT_CONFIG

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")

# Node reports heap_size_limit slightly above --max-old-space-size (young/gen spaces);
# band = configured MB .. 1.5x that (measured: 768 MB -> ~816 MiB).
def _limit_mb(line: str) -> float:
    return float(line.split("=", 1)[1]) / (1024 * 1024)


def test_helper_replaces_inherited_equals_form() -> None:
    out = node_options_with_heap("--max-old-space-size=30000 --trace-warnings", 768)
    assert out == "--trace-warnings --max-old-space-size=768"


def test_helper_strips_space_separated_form() -> None:
    # ``--max-old-space-size 30000`` is two tokens; the size must go too.
    out = node_options_with_heap("--max-old-space-size 30000 --expose-gc", 512)
    assert out == "--expose-gc --max-old-space-size=512"


def test_helper_empty_base_appends_flag() -> None:
    assert node_options_with_heap(None, 2048) == "--max-old-space-size=2048"


def test_config_default_is_2048() -> None:
    assert DEFAULT_CONFIG["lsp"]["max_heap_mb"] == 2048


# Minimal Node LSP server: reports its real heap limit + received NODE_OPTIONS on
# stderr, answers initialize/shutdown, exits on ``exit``.
_FAKE_NODE_SERVER = """
const v8 = require("v8");
process.stderr.write("HERMES_TEST_HEAP_LIMIT=" + v8.getHeapStatistics().heap_size_limit + "\\n");
process.stderr.write("HERMES_TEST_NODE_OPTIONS=" + (process.env.NODE_OPTIONS || "") + "\\n");
let buf = "";
function send(obj) {
  const s = JSON.stringify(obj);
  process.stdout.write("Content-Length: " + Buffer.byteLength(s) + "\\r\\n\\r\\n" + s);
}
process.stdin.on("data", (d) => {
  buf += d;
  for (;;) {
    const hdrEnd = buf.indexOf("\\r\\n\\r\\n");
    if (hdrEnd < 0) return;
    const m = /content-length:\\s*(\\d+)/i.exec(buf.slice(0, hdrEnd));
    if (!m) { buf = buf.slice(hdrEnd + 4); continue; }
    const len = parseInt(m[1], 10);
    const start = hdrEnd + 4;
    if (buf.length < start + len) return;
    let msg;
    try { msg = JSON.parse(buf.slice(start, start + len)); } catch (e) { return; }
    buf = buf.slice(start + len);
    if (msg.method === "initialize") {
      send({ jsonrpc: "2.0", id: msg.id, result: { capabilities: { textDocumentSync: 1 } } });
    } else if (msg.method === "shutdown") {
      send({ jsonrpc: "2.0", id: msg.id, result: null });
    } else if (msg.method === "exit") {
      process.exit(0);
    }
  }
});
"""


def _node_server_def(script: Path) -> ServerDef:
    def build(root: str, ctx: ServerContext) -> SpawnSpec:
        return SpawnSpec(["node", str(script)], root, root)
    return ServerDef("fake-node-ls", (".fake",), lambda fp, ws: ws, build)


@pytest.mark.asyncio
async def test_spawned_node_server_gets_configured_heap(tmp_path: Path, monkeypatch) -> None:
    """Full path: manager applies lsp.max_heap_mb at spawn; the real Node process
    reports the configured limit and no trace of an inherited heap flag."""
    script = tmp_path / "fake_ls.js"
    script.write_text(_FAKE_NODE_SERVER, encoding="utf-8")
    # Simulate a gateway parent whose NODE_OPTIONS carries a (huge) heap flag + extras.
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=30000 --trace-warnings")

    svc = LSPService(
        enabled=True, wait_mode="document", wait_timeout=5.0, install_strategy="manual",
        idle_timeout=0, max_heap_mb=768,
    )
    client = None
    try:
        client = svc._loop.run(svc._spawn_client(_node_server_def(script), str(tmp_path)), timeout=30.0)
        assert client is not None and client.is_running
        # stderr is drained asynchronously — poll briefly for both report lines.
        deadline = asyncio.get_event_loop().time() + 5.0
        lines: list[str] = []
        while asyncio.get_event_loop().time() < deadline:
            lines = list(client._stderr_tail)
            if any(l.startswith("HERMES_TEST_HEAP_LIMIT=") for l in lines) and any(
                l.startswith("HERMES_TEST_NODE_OPTIONS=") for l in lines
            ):
                break
            await asyncio.sleep(0.05)
        limit_line = next(l for l in lines if l.startswith("HERMES_TEST_HEAP_LIMIT="))
        opts_line = next(l for l in lines if l.startswith("HERMES_TEST_NODE_OPTIONS="))
        # v8.getHeapStatistics().heap_size_limit reflects the configured 768 MB cap,
        # not Node's ~4 GiB default and not the inherited 30000 MB flag.
        limit = _limit_mb(limit_line)
        assert 768 <= limit <= 768 * 1.5, f"heap_size_limit {limit:.0f} MB outside configured band"
        # Inherited heap flag replaced; unrelated inherited flag preserved.
        opts = opts_line.split("=", 1)[1]
        assert "30000" not in opts
        assert "--max-old-space-size=768" in opts
        assert "--trace-warnings" in opts
        # The service reports the setting for `hermes lsp status`.
        assert svc.get_status()["max_heap_mb"] == 768
    finally:
        if client is not None:
            # The client lives on the service's background loop — shut it down THERE.
            svc._loop.run(client.shutdown(), timeout=10.0)
        svc.shutdown()


@pytest.mark.asyncio
async def test_zero_disables_override(tmp_path: Path, monkeypatch) -> None:
    """max_heap_mb: 0 = leave inherited NODE_OPTIONS alone (documented opt-out)."""
    script = tmp_path / "fake_ls.js"
    script.write_text(_FAKE_NODE_SERVER, encoding="utf-8")
    monkeypatch.setenv("NODE_OPTIONS", "--max-old-space-size=30000 --trace-warnings")

    svc = LSPService(
        enabled=True, wait_mode="document", wait_timeout=5.0, install_strategy="manual",
        idle_timeout=0, max_heap_mb=0,
    )
    client = None
    try:
        client = svc._loop.run(svc._spawn_client(_node_server_def(script), str(tmp_path)), timeout=30.0)
        assert client is not None and client.is_running
        deadline = asyncio.get_event_loop().time() + 5.0
        lines: list[str] = []
        while asyncio.get_event_loop().time() < deadline:
            lines = list(client._stderr_tail)
            if any(l.startswith("HERMES_TEST_NODE_OPTIONS=") for l in lines):
                break
            await asyncio.sleep(0.05)
        opts_line = next(l for l in lines if l.startswith("HERMES_TEST_NODE_OPTIONS="))
        assert opts_line.split("=", 1)[1] == "--max-old-space-size=30000 --trace-warnings"
    finally:
        if client is not None:
            # The client lives on the service's background loop — shut it down THERE.
            svc._loop.run(client.shutdown(), timeout=10.0)
        svc.shutdown()
