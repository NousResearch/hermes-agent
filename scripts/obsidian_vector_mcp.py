#!/usr/bin/env python3
"""
Obsidian / LLM Wiki vector search — stdio MCP server.

Exposes one tool: search_obsidian, which searches a pre-built SQLite vector
index using cosine similarity on embeddings from the local backend script.

Usage:
    python scripts/obsidian_vector_mcp.py            # run as stdio MCP server
    python scripts/obsidian_vector_mcp.py --help     # show help
    python scripts/obsidian_vector_mcp.py --smoke-test  # self-test (hash mode, no Ollama)

MCP client config (e.g., claude_desktop_config.json or ~/.hermes/mcp_config.json):
    {
        "mcpServers": {
            "obsidian-vector": {
                "command": "python",
                "args": ["/path/to/hermes-agent/scripts/obsidian_vector_mcp.py"]
            }
        }
    }

Environment variables:
    OBSIDIAN_VECTOR_BACKEND      Path to llm_wiki_vector_search.py
                                 Default: ~/.hermes/scripts/llm_wiki_vector_search.py
    LLM_WIKI_VECTOR_INDEX        SQLite index path
                                 Default: ~/.hermes/indexes/llm-wiki-vector.sqlite
    LLM_WIKI_EMBEDDING_BASE_URL  Embedding endpoint base URL
                                 Default: http://127.0.0.1:11434/v1
    LLM_WIKI_EMBEDDING_MODEL     Embedding model name
                                 Default: mxbai-embed-large
    LLM_WIKI_EMBEDDING_MODE      "ollama" or "hash" (hash = deterministic offline mode)
                                 Default: ollama
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy MCP SDK import — matches the pattern in mcp_serve.py
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    FastMCP = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def _default_backend_path() -> Path:
    return Path(
        os.environ.get(
            "OBSIDIAN_VECTOR_BACKEND",
            str(_hermes_home() / "scripts" / "llm_wiki_vector_search.py"),
        )
    )


def _default_index_path() -> Path:
    return Path(
        os.environ.get(
            "LLM_WIKI_VECTOR_INDEX",
            str(_hermes_home() / "indexes" / "llm-wiki-vector.sqlite"),
        )
    )


# ---------------------------------------------------------------------------
# Backend loader — loads the external script by file path, cached per path
# ---------------------------------------------------------------------------

_backend_cache: dict[str, object] = {}


def _load_backend(path: Path):
    key = str(path)
    if key in _backend_cache:
        return _backend_cache[key]
    if not path.exists():
        raise FileNotFoundError(f"backend script not found: {path}")
    spec = importlib.util.spec_from_file_location("_llm_wiki_vector_search", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load backend from {path}")
    mod = importlib.util.module_from_spec(spec)
    # Some backend modules use decorators (for example dataclasses) that look
    # themselves up in sys.modules while executing. importlib does not insert
    # modules created from file locations automatically, so do it explicitly.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _backend_cache[key] = mod
    return mod


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SNIPPET_CHARS = 420


def _title_from(heading: str, path: str) -> str:
    return heading.strip() if heading.strip() else Path(path).stem


def _snippet(text: str, max_chars: int = _SNIPPET_CHARS) -> str:
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"  # …


# ---------------------------------------------------------------------------
# Core search — used by both the MCP tool and the CLI smoke test
# ---------------------------------------------------------------------------


def _run_search(
    query: str,
    limit: int,
    source: Optional[str],
    include_content: bool,
    backend_path: Path,
    index_path: Path,
) -> str:
    query = query.strip()
    if not query:
        return json.dumps({"error": "query must not be empty"})

    # load backend
    try:
        bk = _load_backend(backend_path)
    except FileNotFoundError as e:
        return json.dumps(
            {
                "error": str(e),
                "hint": (
                    "set OBSIDIAN_VECTOR_BACKEND or ensure "
                    "~/.hermes/scripts/llm_wiki_vector_search.py exists"
                ),
            }
        )
    except Exception as e:
        return json.dumps({"error": f"failed to load backend: {e}"})

    # validate index
    index_p = Path(index_path).expanduser()
    if not index_p.exists():
        return json.dumps(
            {
                "error": f"index not found: {index_p}",
                "hint": (
                    "run: python ~/.hermes/scripts/llm_wiki_vector_search.py "
                    "index --wiki ~/llm-wiki"
                ),
            }
        )

    try:
        db = bk.connect(index_p)
        row = db.execute("select count(*) from chunks").fetchone()
        if not row or row[0] == 0:
            return json.dumps(
                {
                    "error": f"index is empty: {index_p}",
                    "hint": (
                        "run: python ~/.hermes/scripts/llm_wiki_vector_search.py "
                        "index --wiki ~/llm-wiki"
                    ),
                }
            )
    except sqlite3.Error as e:
        return json.dumps({"error": f"cannot open index: {e}"})

    embed_base = os.environ.get("LLM_WIKI_EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
    model = os.environ.get("LLM_WIKI_EMBEDDING_MODEL", "mxbai-embed-large")

    # embed query
    try:
        qvec = bk.embed_texts([query], embed_base, model)[0]
    except Exception as e:
        return json.dumps(
            {
                "error": f"embedding failed: {e}",
                "hint": (
                    "ensure Ollama is running, or set "
                    "LLM_WIKI_EMBEDDING_MODE=hash for offline/test mode"
                ),
            }
        )

    # fetch and score all chunks for this model
    try:
        rows = db.execute(
            "select path, heading, chunk_index, text, embedding_json "
            "from chunks where model=?",
            (model,),
        ).fetchall()
    except sqlite3.Error as e:
        return json.dumps({"error": f"search query failed: {e}"})

    scored: list[tuple[float, str, str, int, str]] = []
    for path, heading, chunk_index, text, emb_json in rows:
        try:
            score = bk.cosine(qvec, json.loads(emb_json))
        except Exception:
            continue
        scored.append((score, path, heading, chunk_index, text))

    scored.sort(reverse=True)
    top = scored[: max(1, int(limit))]

    src_label = source or "llm-wiki"
    results = []
    for score, path, heading, chunk_index, text in top:
        item: dict = {
            "title": _title_from(heading, path),
            "path": path,
            "heading": heading,
            "chunk_index": chunk_index,
            "snippet": _snippet(text),
            "score": round(float(score), 6),
            "source": src_label,
            "metadata": {"model": model, "index": str(index_p)},
        }
        if include_content:
            item["content"] = text
        results.append(item)

    return json.dumps(
        {"query": query, "count": len(results), "results": results},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def _build_mcp() -> "FastMCP":  # type: ignore[return]
    if not _MCP_AVAILABLE:
        raise RuntimeError(
            "mcp package not installed — run: pip install 'hermes-agent[mcp]'"
        )

    mcp = FastMCP("obsidian-vector")
    _backend = _default_backend_path()
    _index = _default_index_path()

    @mcp.tool()
    def search_obsidian(
        query: str,
        limit: int = 8,
        source: Optional[str] = None,
        include_content: bool = False,
    ) -> str:
        """Search the Obsidian/LLM Wiki by semantic similarity.

        Returns a JSON string with query, count, and a results list sorted by
        relevance score. Each result has title, path, heading, chunk_index,
        snippet, score, source, metadata, and optionally content.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results (default 8).
            source: Label for the result source (default "llm-wiki").
            include_content: Include full chunk text in results (default false).
        """
        return _run_search(query, limit, source, include_content, _backend, _index)

    return mcp


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _smoke_test() -> int:
    """Quick self-test using hash embeddings — no Ollama needed."""
    import tempfile
    import time

    os.environ["LLM_WIKI_EMBEDDING_MODE"] = "hash"
    backend_path = _default_backend_path()

    if not backend_path.exists():
        print(
            f"smoke-test SKIP: backend not found at {backend_path}\n"
            "set OBSIDIAN_VECTOR_BACKEND to point at llm_wiki_vector_search.py",
            file=sys.stderr,
        )
        return 1

    try:
        bk = _load_backend(backend_path)
    except Exception as e:
        print(f"smoke-test FAIL: cannot load backend: {e}", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        tmp_idx = Path(f.name)

    try:
        db = bk.connect(tmp_idx)
        model = "mxbai-embed-large"
        now = time.time()
        for i, text in enumerate(
            [
                "Transformers use self-attention to relate tokens in context.",
                "Multi-head attention applies the mechanism across subspaces.",
                "Convolutional networks use spatial filters to detect patterns.",
            ]
        ):
            vec = bk.embed_texts([text], "", model)[0]
            db.execute(
                "insert into chunks(path,heading,chunk_index,text,embedding_json,"
                "file_mtime,file_hash,model,updated_at) values(?,?,?,?,?,?,?,?,?)",
                (
                    f"note{i}.md",
                    f"Section {i}",
                    i,
                    text,
                    json.dumps(vec),
                    now,
                    f"h{i}",
                    model,
                    now,
                ),
            )
        db.commit()

        result_json = _run_search(
            "attention mechanism", 2, None, True, backend_path, tmp_idx
        )
        data = json.loads(result_json)

        assert "results" in data, f"missing results key: {data}"
        assert data["count"] > 0, "no results returned"
        assert "content" in data["results"][0], "include_content=True missing content field"
        print(
            f"smoke-test PASS — {data['count']} results for 'attention mechanism', "
            f"top score {data['results'][0]['score']:.4f}"
        )
        return 0
    except AssertionError as e:
        print(f"smoke-test FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"smoke-test FAIL: unexpected error: {e}", file=sys.stderr)
        return 1
    finally:
        tmp_idx.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Obsidian/LLM Wiki vector search — stdio MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a self-test using hash embeddings (no Ollama) then exit",
    )
    args = p.parse_args(argv)

    if args.smoke_test:
        return _smoke_test()

    if not _MCP_AVAILABLE:
        print(
            json.dumps(
                {
                    "error": "mcp package not installed",
                    "hint": "pip install 'hermes-agent[mcp]' or pip install mcp",
                }
            ),
            file=sys.stderr,
        )
        return 1

    _build_mcp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
