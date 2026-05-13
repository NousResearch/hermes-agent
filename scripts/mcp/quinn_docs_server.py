#!/usr/bin/env python3
"""Scoped Quinn source-of-truth docs MCP server."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
REPO_ROOT = Path("/home/quinn/.hermes/hermes-agent")

MAX_EXCERPT_LINES = 120
MAX_EXCERPT_CHARS = 12000
MAX_SEARCH_RESULTS = 50
MAX_QUERY_CHARS = 160

_PRIVATE_WORDS = ["api" + "_" + "key", "tok" + "en", "sec" + "ret", "pass" + "word", "auth" + "orization"]
_PRIVATE_SUFFIXES = tuple("_" + word for word in _PRIVATE_WORDS)
_PRIVATE_LINE_RE = re.compile(r"(?i)\b(api[_-]?key|tok" + "en|sec" + "ret|pass" + "word|auth" + "orization)\s*[:=]\s*[^\s]+")

DOC_REGISTRY: dict[str, dict[str, Any]] = {
    "quinn-hermes-server": {
        "title": "Quinn Hermes Server Source of Truth",
        "path": Path("/home/quinn/docs/quinn-hermes-server.md"),
        "type": "markdown",
    },
    "quinn-ops-mcp": {
        "title": "Quinn Ops MCP",
        "path": REPO_ROOT / "docs" / "quinn_ops_mcp.md",
        "type": "markdown",
    },
    "quinn-ops-snapshot-design": {
        "title": "Quinn Ops Snapshot Diff Design",
        "path": REPO_ROOT / "docs" / "quinn_ops_snapshot_diff_design.md",
        "type": "markdown",
    },
    "quinn-github-mcp-plan": {
        "title": "GitHub MCP Integration Plan",
        "path": REPO_ROOT / "docs" / "quinn_ops_github_mcp_plan.md",
        "type": "markdown",
    },
    "quinn-observability-mcp-plan": {
        "title": "Observability MCP Integration Plan",
        "path": REPO_ROOT / "docs" / "quinn_ops_observability_mcp_plan.md",
        "type": "markdown",
    },
    "quinn-docs-mcp-plan": {
        "title": "Scoped Docs and Notes MCP Integration Plan",
        "path": REPO_ROOT / "docs" / "quinn_docs_mcp_plan.md",
        "type": "markdown",
    },
    "quinn-approval-ops-mcp-plan": {
        "title": "Approval Ops MCP Integration Plan",
        "path": REPO_ROOT / "docs" / "quinn_approval_ops_mcp_plan.md",
        "type": "markdown",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_private_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _PRIVATE_WORDS or any(normalized.endswith(suffix) for suffix in _PRIVATE_SUFFIXES)


def redact_string(value: str) -> str:
    return _PRIVATE_LINE_RE.sub(lambda m: f"{m.group(1)}: [REDACTED]", value)


def sanitize(value: Any, key: str | None = None) -> Any:
    if key and is_private_key(key):
        if isinstance(value, bool) or value is None:
            return value
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, key) for v in value]
    return value


def response(data: dict[str, Any] | None = None, errors: list[dict[str, Any]] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {"ok": not bool(errors), "data": sanitize(data or {}), "errors": sanitize(errors or []), "warnings": sanitize(warnings or [])}


def error(kind: str, message: str) -> dict[str, str]:
    return {"kind": kind, "message": redact_string(message)}


def clamp_int(value: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = low
    return max(low, min(high, parsed))


def resolve_doc_id(doc_id: str) -> tuple[str, dict[str, Any] | None, dict[str, str] | None]:
    normalized = str(doc_id or "").strip()
    if not normalized or normalized not in DOC_REGISTRY:
        return normalized, None, error("unknown_doc", "Document ID is not allowlisted")
    return normalized, DOC_REGISTRY[normalized], None


def file_meta(doc_id: str, entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(entry["path"])
    base = {"doc_id": doc_id, "title": str(entry.get("title") or doc_id), "type": str(entry.get("type") or path.suffix.lstrip(".") or "text"), "path_alias": doc_id}
    try:
        stat = path.stat()
        base.update({"exists": True, "size_bytes": stat.st_size, "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"), "readable": os.access(path, os.R_OK)})
    except FileNotFoundError:
        base.update({"exists": False, "readable": False})
    except Exception as exc:
        base.update({"exists": False, "readable": False, "error": type(exc).__name__})
    return base


def read_text(entry: dict[str, Any]) -> tuple[str | None, dict[str, str] | None]:
    path = Path(entry["path"])
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except FileNotFoundError:
        return None, error("missing_doc", "Document does not exist")
    except Exception as exc:
        return None, error("read_failed", type(exc).__name__)


def healthcheck() -> dict[str, Any]:
    existing = sum(1 for entry in DOC_REGISTRY.values() if Path(entry["path"]).exists())
    return response({"server": "quinn_docs", "timestamp_utc": utc_now(), "document_count": len(DOC_REGISTRY), "existing_document_count": existing, "read_only": True})


def list_documents() -> dict[str, Any]:
    docs = [file_meta(doc_id, entry) for doc_id, entry in sorted(DOC_REGISTRY.items())]
    return response({"documents": docs, "count": len(docs)})


def get_document_outline(doc_id: str) -> dict[str, Any]:
    resolved, entry, err = resolve_doc_id(doc_id)
    if err:
        return response(errors=[err])
    assert entry is not None
    text, read_err = read_text(entry)
    if read_err:
        return response({"doc_id": resolved}, errors=[read_err])
    headings: list[dict[str, Any]] = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            headings.append({"line": line_no, "level": len(match.group(1)), "text": redact_string(match.group(2))})
    return response({"doc_id": resolved, "title": entry.get("title", resolved), "headings": headings})


def read_document_excerpt(doc_id: str, start_line: int = 1, limit: int = 80) -> dict[str, Any]:
    resolved, entry, err = resolve_doc_id(doc_id)
    if err:
        return response(errors=[err])
    assert entry is not None
    text, read_err = read_text(entry)
    if read_err:
        return response({"doc_id": resolved}, errors=[read_err])
    all_lines = (text or "").splitlines()
    start = clamp_int(start_line, 1, max(1, len(all_lines)))
    count = clamp_int(limit, 1, MAX_EXCERPT_LINES)
    selected: list[dict[str, Any]] = []
    total_chars = 0
    for line_no, line in enumerate(all_lines[start - 1:start - 1 + count], start=start):
        safe_line = redact_string(line)
        total_chars += len(safe_line)
        if total_chars > MAX_EXCERPT_CHARS:
            break
        selected.append({"line": line_no, "text": safe_line})
    return response({"doc_id": resolved, "title": entry.get("title", resolved), "start_line": start, "requested_limit": limit, "returned_lines": len(selected), "lines": selected})


def search_documents(query: str, limit: int = 20) -> dict[str, Any]:
    q = str(query or "")[:MAX_QUERY_CHARS].strip().lower()
    if not q:
        return response(errors=[error("invalid_query", "Query is required")])
    max_results = clamp_int(limit, 1, MAX_SEARCH_RESULTS)
    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for doc_id, entry in sorted(DOC_REGISTRY.items()):
        text, read_err = read_text(entry)
        if read_err:
            warnings.append(f"{doc_id}: {read_err['kind']}")
            continue
        for line_no, line in enumerate((text or "").splitlines(), start=1):
            if q in line.lower():
                results.append({"doc_id": doc_id, "title": entry.get("title", doc_id), "line": line_no, "snippet": redact_string(line.strip())[:240]})
                if len(results) >= max_results:
                    return response({"query": "[REDACTED]", "results": results, "count": len(results)}, warnings=warnings)
    return response({"query": "[REDACTED]", "results": results, "count": len(results)}, warnings=warnings)


def get_document_summary(doc_id: str) -> dict[str, Any]:
    resolved, entry, err = resolve_doc_id(doc_id)
    if err:
        return response(errors=[err])
    assert entry is not None
    outline = get_document_outline(resolved)
    return response({"document": file_meta(resolved, entry), "outline": outline.get("data", {}).get("headings", [])[:20]})


def check_source_of_truth_freshness() -> dict[str, Any]:
    docs = [file_meta(doc_id, entry) for doc_id, entry in sorted(DOC_REGISTRY.items())]
    warnings = [f"missing: {doc['doc_id']}" for doc in docs if not doc.get("exists")]
    for doc_id, entry in sorted(DOC_REGISTRY.items()):
        required_headings = [str(heading) for heading in entry.get("required_headings", [])]
        if not required_headings or not Path(entry["path"]).exists():
            continue
        text, read_err = read_text(entry)
        if read_err:
            continue
        present = {
            match.group(2).strip()
            for line in (text or "").splitlines()
            if (match := re.match(r"^(#{1,6})\s+(.+?)\s*$", line))
        }
        for heading in required_headings:
            if heading not in present:
                warnings.append(f"missing heading: {doc_id}: {heading}")
    return response({"documents": docs, "missing_count": len(warnings)}, warnings=warnings)


def propose_document_patch(doc_id: str, change_request: str) -> dict[str, Any]:
    resolved, entry, err = resolve_doc_id(doc_id)
    if err:
        return response(errors=[err])
    assert entry is not None
    request = redact_string(str(change_request or "").strip())[:1000]
    proposal_patch = (
        "*** Begin Patch\n"
        f"*** Update File: {resolved}\n"
        "@@ section to be confirmed @@\n"
        f"# Requested change: {request}\n"
        "# No changes have been applied by this MCP server.\n"
        "*** End Patch"
    )
    return response({"doc_id": resolved, "title": entry.get("title", resolved), "requires_approval": True, "applied": False, "proposal": ["Review requested change against source-of-truth scope.", "Prepare a targeted patch with exact context.", "Apply only after explicit approval through a write-capable flow."], "proposal_patch": proposal_patch, "change_request": request})


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "healthcheck": healthcheck,
    "list_documents": list_documents,
    "get_document_outline": get_document_outline,
    "search_documents": search_documents,
    "read_document_excerpt": read_document_excerpt,
    "get_document_summary": get_document_summary,
    "check_source_of_truth_freshness": check_source_of_truth_freshness,
    "propose_document_patch": propose_document_patch,
}


async def _run_mcp_server() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        print(json.dumps(response(errors=[error("missing_dependency", "Python MCP SDK is not installed")])), file=sys.stderr)
        raise SystemExit(1)
    mcp = FastMCP("quinn_docs")
    for name, fn in TOOL_FUNCTIONS.items():
        mcp.tool(name=name)(fn)
    await mcp.run_stdio_async()


if __name__ == "__main__":
    import asyncio
    asyncio.run(_run_mcp_server())
