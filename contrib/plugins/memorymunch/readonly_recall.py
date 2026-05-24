#!/usr/bin/env python3
"""MemoryMunch recall wrapper for Hermes.

Primary path: original MemoryMunch smart_search bridge. That preserves the real
vault + DB keyword + vector + activation parallel search behavior instead of the
old db_keyword_readonly prototype.

Important: original smart_search calls recall_memories, which updates activation
metadata/logs in the MemoryMunch DB by design. This wrapper labels that honestly
as original_smart_search_bridge, not SELECT-only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_BRIDGE = str(Path(__file__).with_name("original_bridge.py"))
SAFE_WORD_RE = re.compile(r"[A-Za-z0-9_:@.#-]{3,}")


def extract_keywords(query: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "what", "when", "where",
        "why", "how", "you", "your", "are", "was", "were", "have", "has", "had", "about",
        "into", "memory", "memories",
    }
    out = []
    for w in SAFE_WORD_RE.findall((query or "").lower()):
        if w not in stop and len(w) >= 3 and w not in out:
            out.append(w)
    return out[:8]


def classify_provenance(entity: str, scope_entity: str | None, *, found_in: str = "", hop_depth: int = 0) -> str:
    if hop_depth > 0:
        return "GRAPH_LINKED_OUTWARD"
    if scope_entity and entity == scope_entity:
        return "OWN_SCOPE"
    if entity == "system":
        return "SYSTEM_SHARED"
    if entity == "general":
        return "GENERAL_SHARED"
    # Original smart_search can return cross-scope connected context. Keep soft-wall label explicit.
    return "GRAPH_LINKED_OUTWARD"


def bridge_call(tool: str, args: dict[str, Any], *, timeout: int = 90) -> dict[str, Any]:
    bridge = os.environ.get("HERMES_MEMORYMUNCH_ORIGINAL_BRIDGE", DEFAULT_BRIDGE)
    env = os.environ.copy()
    env.setdefault("MEMORYMUNCH_ORIGINAL_REPO", "/mnt/c/Users/paulcooke1976/memorymunch-mcp")
    env.setdefault("MEMORYMUNCH_VAULT_PATH", "/mnt/c/Users/paulcooke1976/memorymunch-vault")
    proc = subprocess.run(
        [sys.executable, bridge],
        input=json.dumps({"tool": tool, "args": args}, ensure_ascii=False),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"bridge rc={proc.returncode}")
    return json.loads(proc.stdout)


def normalize_smart_result(row: dict[str, Any], scope_entity: str | None) -> dict[str, Any]:
    found_in = str(row.get("found_in") or "")
    sources = [s for s in found_in.split("+") if s] or ["original_smart_search"]
    atom_id = row.get("id") or row.get("atom_id")
    content = str(row.get("content") or row.get("content_preview") or "")
    return {
        "atom_id": atom_id,
        "memory_type": row.get("type") or row.get("memory_type"),
        "entity": row.get("entity"),
        "domain": row.get("domain", "general"),
        "activation_weight": float(row.get("activation_weight", row.get("weight", 0)) or 0),
        "decay_rate": float(row.get("decay_rate", 0) or 0),
        "content_preview": content[:900],
        "provenance_class": classify_provenance(str(row.get("entity") or ""), scope_entity, found_in=found_in),
        "sources": sources,
        "found_in": found_in,
        "search_score": row.get("search_score"),
        "seed_reason": "original_memorymunch_smart_search",
        "hop_depth": int(row.get("hop_depth", 0) or 0),
        "linked_from": row.get("linked_from"),
        "scope_warning": "" if str(row.get("entity") or "") in {scope_entity, "system", "general"} else "outward/cross-scope context; do not treat as own identity",
    }


def recall(query: str, scope_entity: str | None, scope_domain: str | None, max_results: int) -> dict[str, Any]:
    keywords = extract_keywords(query)
    concepts = keywords or ([query[:40]] if query.strip() else [])
    payload = bridge_call(
        "smart_search",
        {
            "query": query,
            "concepts": concepts,
            "entities": [scope_entity] if scope_entity else [],
            "scope_entity": scope_entity,
            "max_results": max_results,
        },
        timeout=int(os.environ.get("HERMES_MEMORYMUNCH_RECALL_TIMEOUT", "120")),
    )
    result = payload.get("result") or {}
    rows = result.get("results") or []
    normalized = [normalize_smart_result(row, scope_entity) for row in rows]
    meta = dict(result.get("_meta") or {})
    return {
        "mode": "original_smart_search_bridge",
        "query": query,
        "scope_entity": scope_entity,
        "scope_domain": scope_domain,
        "keywords": keywords,
        "returned": len(normalized),
        "results": normalized,
        "source_model": "OBSIDIAN_VAULT+DB_KEYWORD+VECTOR+ACTIVATION",
        "parallel_search": True,
        "vault_path": payload.get("vault_path"),
        "original_repo": payload.get("original_repo"),
        "original_meta": meta,
        "write_paths_disabled": False,
        "notes": [
            "Calls original MemoryMunch smart_search through Hermes bridge.",
            "Original smart_search runs vault + DB keyword + vector + activation paths in parallel.",
            "Original activation recall mutates activation metadata/logs by MemoryMunch design; this is not db_keyword_readonly.",
            "Canonical reseed remains vault_to_db via sync_vault; raw transcripts are not canonical atoms.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--scope-entity")
    ap.add_argument("--scope-domain")
    ap.add_argument("--max-results", type=int, default=5)
    args = ap.parse_args()
    print(json.dumps(recall(args.query, args.scope_entity, args.scope_domain, args.max_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
