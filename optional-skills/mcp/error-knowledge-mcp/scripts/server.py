"""
error-knowledge MCP Server — cross-project error pattern knowledge base.

Records, searches, and archives bug patterns so agents learn from past
mistakes across projects. Each record stores: title, scope (generic vs
business-specific), category, language/framework, project, reproduction
steps, root cause, logical boundary, affected files, and fix summary.

Storage layout:
  <root>/
    generic/<lang>/<slug>.md             ← language-level patterns
    business-specific/<project>/<slug>.md  ← project-local patterns

Backed by file-based markdown + TF-IDF cached index. No external DB.
Zero external dependencies beyond the MCP SDK.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# ── Configuration ──────────────────────────────────────────────────────────

DEFAULT_ROOT = Path.home() / ".hermes" / "knowledge" / "errors"
ROOT = Path(os.environ.get("ERROR_KNOWLEDGE_ROOT", str(DEFAULT_ROOT)))
AUTO_THRESHOLD = int(os.environ.get("ERROR_KNOWLEDGE_AUTO_ARCHIVE", "5000"))
DEDUP_RATIO = float(os.environ.get("ERROR_KNOWLEDGE_DEDUP_RATIO", "0.65"))

GENERIC = ROOT / "generic"
BUSINESS = ROOT / "business-specific"

for d in [ROOT, GENERIC, BUSINESS]:
    d.mkdir(parents=True, exist_ok=True)

INDEX_FILE = ROOT / "_index.json"

# ── Frontmatter field definitions ──────────────────────────────────────────

FIELDS = [
    "title", "scope", "category", "lang", "project", "date",
    "reproduce_steps", "root_cause", "boundary", "files", "fix_summary",
]

SECTION_HEADERS: dict[str, str] = {
    "root_cause": "## Root Cause\n\n",
    "reproduce_steps": "## Reproduction Steps\n\n",
    "fix_summary": "## Fix Summary\n\n",
    "boundary": "## Logical Boundary\n\n",
}

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
TOKEN_RE = re.compile(r"\w+")

FIELD_WEIGHTS: dict[str, float] = {
    "title": 3.0,
    "root_cause": 2.0,
    "fix_summary": 1.5,
    "reproduce_steps": 1.0,
    "_body": 0.5,
}

server = Server("error-knowledge")

# ── Cached index ───────────────────────────────────────────────────────────

_index_cache: list[dict] | None = None
_index_mtime: float = 0.0
_index_dirty: bool = False


def _invalidate_cache():
    """Mark the in-memory index as stale so next search reloads it."""
    global _index_cache, _index_dirty
    # Don't clear immediately — keep serving stale data until the next
    # search triggers a reload via _load_index().
    _index_dirty = True


def _load_index() -> list[dict]:
    """Load index from cache or disk. Returns records list."""
    global _index_cache, _index_mtime, _index_dirty

    # Check if disk index changed (other processes may have modified files)
    try:
        current_mtime = INDEX_FILE.stat().st_mtime
    except OSError:
        current_mtime = 0.0

    if _index_cache is not None and not _index_dirty and current_mtime <= _index_mtime:
        return _index_cache

    # Rebuild from files
    records = []
    for fp in ROOT.rglob("*.md"):
        if fp.parent == ROOT and fp.name.startswith("_"):
            continue
        if "_index" in fp.name:
            continue
        rec = _parse(fp)
        if rec:
            records.append(rec)

    _index_cache = records
    _index_mtime = current_mtime
    _index_dirty = False

    # Persist to disk for cross-process sharing
    INDEX_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def _refresh_cache(added: list[dict] | None = None):
    """Append new records to in-memory cache without full rebuild.
    
    Call this after writing a new record to avoid scanning all files.
    Falls back to full load if the cache is cold.
    """
    global _index_cache
    if _index_cache is None:
        _load_index()
        return
    if added:
        _index_cache.extend(added)
    INDEX_FILE.write_text(
        json.dumps(_index_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Helpers ────────────────────────────────────────────────────────────────


def _slug(text: str) -> str:
    """Convert a title into a filesystem-safe slug ending in .md."""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text.lower().strip())
    return s[:80].strip("-") + ".md"


def _resolve_path(record: dict) -> Path:
    scope = record.get("scope", "generic")
    lang = (record.get("lang") or "unknown").lower().replace("/", "-").replace(" ", "-")
    project = (record.get("project") or "unknown").lower().replace("/", "-").replace(" ", "-")
    slug = _slug(record.get("title", "untitled"))
    parent = GENERIC / lang if scope == "generic" else BUSINESS / project
    parent.mkdir(parents=True, exist_ok=True)
    return parent / slug


def _parse(path: Path) -> dict | None:
    """Parse a markdown file with YAML-like frontmatter."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fields: dict[str, Any] = {}
    for line in m.group(1).strip().split("\n"):
        if ":" in line:
            k, _, v = line.partition(":")
            fields[k.strip()] = v.strip()
    if not fields.get("title"):
        return None
    fields["_body"] = text[m.end():].strip()
    fields["_file"] = str(path.relative_to(ROOT))
    return fields


def _dedup_check(record: dict) -> str | None:
    """Return the existing file path if a likely duplicate exists.
    
    Uses SequenceMatcher ratio on titles (same scope + lang). The
    DEDUP_RATIO threshold (default 0.65) controls sensitivity — higher
    means only near-identical titles are considered duplicates.
    """
    scope = record.get("scope", "generic")
    lang = (record.get("lang") or "").lower()
    title = (record.get("title") or "").lower().strip()

    for rec in _load_index():
        if rec.get("scope") != scope:
            continue
        if (rec.get("lang") or "").lower() != lang:
            continue
        existing = (rec.get("title") or "").lower().strip()
        if not title or not existing:
            continue
        # Fast path: substring containment (cheap)
        if title in existing or existing in title:
            return rec.get("_file")
        # Fuzzy path: similarity ratio
        ratio = SequenceMatcher(None, title, existing).ratio()
        if ratio >= DEDUP_RATIO:
            return rec.get("_file")
    return None


def _format_markdown(record: dict) -> str:
    """Build a markdown file with frontmatter and optional sections."""
    lines = ["---"]
    for field in FIELDS:
        val = record.get(field, "")
        if val:
            lines.append(f"{field}: {val}")
    lines.append("---\n")
    for section_field in ("root_cause", "reproduce_steps", "fix_summary", "boundary"):
        content = record.get(section_field, "")
        if content:
            header = SECTION_HEADERS.get(section_field, f"## {section_field}\n\n")
            lines.append(header + content)
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _compute_tfidf(query_tokens: list[str], doc_field_texts: dict[str, str], num_docs: int) -> float:
    """Compute a TF-IDF-like score for a document against a query.
    
    Uses field-weighted term frequency with IDF. Pure stdlib — no external
    NLP dependencies. score = sum(weight * tf * idf) for each query term
    that appears in the document.
    """
    score = 0.0
    for qt in query_tokens:
        idf = 1.0  # Base IDF; we approximate with a simple scheme
        for field, weight in FIELD_WEIGHTS.items():
            text = doc_field_texts.get(field, "")
            if not text:
                continue
            doc_tokens = _tokenize(text)
            if not doc_tokens:
                continue
            tf = doc_tokens.count(qt) / len(doc_tokens)
            if tf > 0:
                score += weight * tf * idf
    return score


def _search_local(
    keywords: str = "",
    category: str = "",
    project: str = "",
    lang: str = "",
    scope: str = "",
    limit: int = 10,
) -> list[dict]:
    """TF-IDF search over the error knowledge directory. Pure stdlib."""
    kw = keywords.lower().strip() if keywords else ""
    cats = [c.strip().lower() for c in category.split(",") if c.strip()] if category else []
    projs = [p.strip().lower() for p in project.split(",") if p.strip()] if project else []
    langs = [l.strip().lower() for l in lang.split(",") if l.strip()] if lang else []
    scopes = [s.strip().lower() for s in scope.split(",") if s.strip()] if scope else []

    records = _load_index()
    results: list[dict] = []

    query_tokens = _tokenize(kw) if kw else []

    for rec in records:
        rec_scope = (rec.get("scope") or "").lower()
        rec_cat = (rec.get("category") or "").lower()
        rec_proj = (rec.get("project") or "").lower()
        rec_lang = (rec.get("lang") or "").lower()

        if scopes and rec_scope not in scopes:
            continue
        if cats and rec_cat not in cats:
            continue
        if projs and rec_proj not in projs:
            continue
        if langs and rec_lang not in langs:
            continue

        if not query_tokens:
            results.append(rec)
            continue

        doc_texts = {
            "title": rec.get("title", ""),
            "root_cause": rec.get("root_cause", ""),
            "fix_summary": rec.get("fix_summary", ""),
            "reproduce_steps": rec.get("reproduce_steps", ""),
            "_body": rec.get("_body", ""),
        }

        score = _compute_tfidf(query_tokens, doc_texts, len(records))
        if score > 0:
            rec["_score"] = round(score, 4)
            results.append(rec)

    results.sort(key=lambda r: r.get("_score", 1), reverse=True)
    for r in results:
        r.pop("_score", None)
        r.pop("_body", None)
        r.pop("_file", None)
    return results[:limit]


def _auto_archive():
    """Move flat files into scope subdirectories when the total exceeds threshold."""
    count = 0
    for fp in ROOT.rglob("*.md"):
        try:
            rel = str(fp.relative_to(ROOT))
        except ValueError:
            continue
        if "/" in rel or fp.parent == ROOT and fp.name.startswith("_"):
            continue
        meta = _parse(fp)
        if not meta:
            continue
        dest = _resolve_path(meta)
        if dest == fp:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(fp), str(dest))
        count += 1
    if count:
        _invalidate_cache()


# ── MCP tool registration ─────────────────────────────────────────────────


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_error_patterns",
            description="Search error knowledge base (TF-IDF, real-time). "
            "Use qmd MCP for full vault search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Search keywords"},
                    "category": {
                        "type": "string",
                        "description": "Filter: null_pointer, concurrency, performance, security, config, type_error, logic",
                    },
                    "project": {"type": "string", "description": "Filter by project name"},
                    "lang": {"type": "string", "description": "Filter by language/framework"},
                    "scope": {
                        "type": "string",
                        "description": "Filter: generic (language-level) or business-specific (project-local)",
                    },
                    "limit": {"type": "integer", "default": 10, "description": "Max results"},
                },
                "required": ["keywords"],
            },
        ),
        types.Tool(
            name="record_error_pattern",
            description="Save an error record. Duplicates (same scope, same lang, similar title) "
            "are auto-skipped. Uses fuzzy title matching (SequenceMatcher, default threshold 0.65). "
            "Configurable via ERROR_KNOWLEDGE_DEDUP_RATIO env var.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Short error title"},
                    "scope": {
                        "type": "string",
                        "default": "generic",
                        "description": "generic (language-level, reusable across projects) "
                        "or business-specific (project-local)",
                    },
                    "category": {
                        "type": "string",
                        "description": "null_pointer, concurrency, performance, security, config, type_error, logic",
                    },
                    "lang": {
                        "type": "string",
                        "description": "Required for generic scope. e.g. csharp, python, go",
                    },
                    "project": {
                        "type": "string",
                        "description": "Required for business-specific scope",
                    },
                    "reproduce_steps": {"type": "string", "description": "Steps to reproduce"},
                    "root_cause": {"type": "string", "description": "Root cause analysis"},
                    "boundary": {"type": "string", "description": "Logical boundary / preconditions"},
                    "files": {"type": "string", "description": "Affected file paths"},
                    "fix_summary": {"type": "string", "description": "How it was fixed"},
                },
                "required": ["title", "category", "root_cause", "fix_summary"],
            },
        ),
        types.Tool(
            name="knowledge_stats",
            description="View knowledge base statistics: total records, by scope, by language, by project.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "search_error_patterns":
        results = _search_local(
            keywords=arguments.get("keywords", ""),
            category=arguments.get("category", ""),
            project=arguments.get("project", ""),
            lang=arguments.get("lang", ""),
            scope=arguments.get("scope", ""),
            limit=arguments.get("limit", 10),
        )
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"found": len(results), "records": results}, ensure_ascii=False, indent=2
                ),
            )
        ]

    if name == "record_error_pattern":
        record = {
            "title": arguments.get("title", ""),
            "scope": arguments.get("scope", "generic"),
            "category": arguments.get("category", ""),
            "lang": arguments.get("lang", ""),
            "project": arguments.get("project", ""),
            "date": str(date.today()),
            "reproduce_steps": arguments.get("reproduce_steps", ""),
            "root_cause": arguments.get("root_cause", ""),
            "boundary": arguments.get("boundary", ""),
            "files": arguments.get("files", ""),
            "fix_summary": arguments.get("fix_summary", ""),
        }
        dup = _dedup_check(record)
        if dup:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(
                        {"status": "skipped", "reason": "duplicate", "existing": dup},
                        ensure_ascii=False,
                    ),
                )
            ]
        fp = _resolve_path(record)
        fp.write_text(_format_markdown(record), encoding="utf-8")
        _refresh_cache(added=[record | {"_file": str(fp.relative_to(ROOT))}])
        if len(list(ROOT.rglob("*.md"))) >= AUTO_THRESHOLD:
            _auto_archive()
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {"status": "ok", "file": str(fp.relative_to(ROOT))}, ensure_ascii=False
                ),
            )
        ]

    if name == "knowledge_stats":
        records = _load_index()
        generic = [r for r in records if r.get("scope") == "generic"]
        biz = [r for r in records if r.get("scope") == "business-specific"]
        langs: dict[str, int] = {}
        projects: dict[str, int] = {}
        for r in records:
            l = r.get("lang", "unknown") or "unknown"
            langs[l] = langs.get(l, 0) + 1
            p = r.get("project", "unknown") or "unknown"
            projects[p] = projects.get(p, 0) + 1
        return [
            types.TextContent(
                type="text",
                text=json.dumps(
                    {
                        "total": len(records),
                        "generic": len(generic),
                        "business_specific": len(biz),
                        "by_language": dict(sorted(langs.items(), key=lambda x: -x[1])),
                        "by_project": dict(sorted(projects.items(), key=lambda x: -x[1])),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        ]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="error-knowledge",
                server_version="0.4.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli():
    """Command-line entry point for direct querying.
    
    Usage:
        python -m error_knowledge_server --query "null pointer"
        python -m error_knowledge_server --stats
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split(".")[0])
    parser.add_argument("--query", help="Search keywords")
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    parser.add_argument("--limit", type=int, default=10, help="Max results")
    args = parser.parse_args()

    if args.stats:
        records = _load_index()
        generic = sum(1 for r in records if r.get("scope") == "generic")
        biz = sum(1 for r in records if r.get("scope") == "business-specific")
        print(f"Total: {len(records)} (generic: {generic}, business: {biz})")
        return

    if args.query:
        results = _search_local(keywords=args.query, limit=args.limit)
        if not results:
            print("No matches found.")
            return
        for i, r in enumerate(results, 1):
            title = r.get("title", "?")
            scope = r.get("scope", "?")
            lang = r.get("lang", r.get("project", "?"))
            print(f"{i:>3}. [{scope}] [{lang}] {title}")
        return

    parser.print_help()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
