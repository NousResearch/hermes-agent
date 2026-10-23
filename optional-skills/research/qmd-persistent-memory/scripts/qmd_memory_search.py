#!/usr/bin/env python3
"""
qmd_memory_search.py — expose QMD semantic search to an MCP server.

Folds local, private, semantic search over your persistent memory stores into a
FastMCP server (reuse an existing one, or run this standalone on port 8920).

Requires: python3 -c "import mcp"  (pip install "mcp[cli]")  and the `qmd` CLI
installed with node-llama-cpp (see the qmd-persistent-memory skill).

Config (env, or edit the defaults):
  QMD_INDEX   - the named QMD index you created via `qmd --index <name> ...`
  QMD_BASES   - optional "label=collectionName,label=collectionName" so the
                `base=` arg maps to a single collection. When unset, base is
                ignored and search spans the whole index.
  QMD_PORT    - port for standalone mode (default 8920).

Run standalone:       python3 qmd_memory_search.py
Wire into your server: import the two functions, or copy their bodies.
"""
import json
import os
import shutil
import subprocess as sp

from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # allow import for testing without the mcp package
    FastMCP = None

QMD_BIN = shutil.which("qmd")
QMD_INDEX = os.environ.get("QMD_INDEX", "memory")
# "label=collectionName,label=collectionName" -> base arg to one collection.
QMD_BASES = {}
for _pair in os.environ.get("QMD_BASES", "").split(","):
    if "=" in _pair:
        _label, _col = _pair.split("=", 1)
        QMD_BASES[_label.strip()] = _col.strip()


def _run_qmd(args, timeout):
    """Run QMD and return (returncode, stdout). Raises RuntimeError on failure."""
    if not QMD_BIN:
        raise RuntimeError("qmd not installed. See the qmd-persistent-memory skill prerequisites.")
    cmd = [QMD_BIN, "--index", QMD_INDEX, *args]
    try:
        r = sp.run(cmd, capture_output=True, text=True, timeout=timeout)
    except sp.TimeoutExpired:
        raise RuntimeError("qmd query timed out (first hybrid call loads large models). Use mode='fast'.")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:800] or r.stdout[:800])
    return r.stdout


def semantic_search_memory(query: str, n: int = 6, base: str = "all", mode: str = "fast") -> str:
    """Semantic (meaning-based) search over memory using local QMD embeddings.

    Finds conceptually related memory even when the wording differs. `base`
    scopes to one collection when labels were configured (else ignored).
    `mode`: "fast" (vector-only, ~5s, default) | "hybrid" (vector+keyword+rerank,
    multi-minute cold start on CPU — use for hard lookups)."""
    _col = QMD_BASES.get(base)  # None for "all"/unknown -> whole index
    sub = "vsearch" if mode == "fast" else "query"
    args = [sub, query, "--json", "-n", str(n)]
    if _col:
        args += ["--collection", _col]
    try:
        raw = _run_qmd(args, timeout=300 if mode == "fast" else 1800)
    except RuntimeError as e:
        return f"QMD error: {e}"
    # QMD prints a banner ("Expanding query… Searching…") BEFORE the JSON.
    if "[" in raw:
        raw = raw[raw.index("["):]
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return f"QMD parse error: {raw[:800]}"
    out = []
    for row in rows:
        path = row.get("file") or row.get("path") or "?"
        path = path.split("?index=")[0].replace("qmd://", "")  # strip scheme + index kwarg
        score = row.get("score", 0)
        snippet = (row.get("snippet") or row.get("excerpt") or "").replace("\n", " ")[:300]
        out.append(f"[{score:.2f}] {path}\n  {snippet}")
    return "\n\n".join(out) if out else "No relevant results."


def qmd_status() -> str:
    """Report the local QMD semantic-search index health."""
    try:
        return _run_qmd(["status"], timeout=30)
    except RuntimeError as e:
        return f"QMD error: {e}"


if FastMCP is not None:
    mcp = FastMCP("qmd-memory", port=int(os.environ.get("QMD_PORT", "8920")),
                  host="0.0.0.0", streamable_http_path="/mcp")

    @mcp.tool()
    def semantic_search_memory_tool(query: str, n: int = 6, base: str = "all", mode: str = "fast") -> str:
        """Semantic (meaning-based) search over memory. mode: fast | hybrid."""
        return semantic_search_memory(query, n=n, base=base, mode=mode)

    @mcp.tool()
    def qmd_status_tool() -> str:
        """Report the local QMD semantic-search index health."""
        return qmd_status()