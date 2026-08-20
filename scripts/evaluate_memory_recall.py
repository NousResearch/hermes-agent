#!/usr/bin/env python3
"""Evaluate session_search ground truth against holographic fact recall.

The harness is intentionally offline-safe: it reads configured state.db files,
creates a temporary memory_store.db for seeded expected facts, and never writes
to live profile memory databases.

Corpus format (JSON):
[
  {
    "id": "lineage_aggregate_counts",
    "state_db": "/path/to/state.db",
    "session_query": "working waiting",
    "expected_session_substrings": ["working: 2 | waiting: 1"],
    "seed_facts": [
      {
        "content": "Lineage status renders working: 2 | waiting: 1 | blocked: 0 | dormant: 3.",
        "category": "project",
        "tags": "lineage,status,working,waiting,blocked,dormant"
      }
    ],
    "fact_queries": ["Lineage status working waiting blocked dormant"],
    "probe_entities": ["Lineage status"],
    "reason_entities": [["Lineage status", "working waiting"]],
    "expected_fact_substrings": ["working: 2 | waiting: 1"]
  }
]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hermes_state import SessionDB
from plugins.memory.holographic.retrieval import FactRetriever
from plugins.memory.holographic.store import MemoryStore
from tools.session_search_tool import session_search


def _contains_any(results: list[dict[str, Any]], substrings: list[str]) -> bool:
    haystack = json.dumps(results, ensure_ascii=False)
    return all(s in haystack for s in substrings)


def evaluate_item(item: dict[str, Any], workdir: Path) -> dict[str, Any]:
    state_db = Path(item["state_db"])
    session_data = json.loads(
        session_search(
            query=item["session_query"],
            limit=int(item.get("session_limit", 5)),
            sort=item.get("sort", "newest"),
            role_filter=item.get("role_filter", "user,assistant,tool"),
            db=SessionDB(state_db),
        )
    )

    memory_db = workdir / f"{item['id']}.memory_store.db"
    store = MemoryStore(memory_db, default_trust=float(item.get("default_trust", 0.8)))
    try:
        for fact in item.get("seed_facts", []):
            store.add_fact(
                fact["content"],
                category=fact.get("category", "general"),
                tags=fact.get("tags", ""),
            )

        retriever = FactRetriever(
            store,
            hrr_weight=float(item.get("hrr_weight", 0.3)),
            min_hrr_score=float(item.get("min_hrr_score", 0.35)),
        )
        fact_searches = {
            q: retriever.search(q, category=item.get("category"), min_trust=0.0, limit=5)
            for q in item.get("fact_queries", [])
        }
        probes = {
            entity: retriever.probe(entity, category=item.get("category"), limit=5)
            for entity in item.get("probe_entities", [])
        }
        reasons = {
            " | ".join(entities): retriever.reason(entities, category=item.get("category"), limit=5)
            for entities in item.get("reason_entities", [])
        }
    finally:
        store.close()

    expected_session = item.get("expected_session_substrings", [])
    expected_fact = item.get("expected_fact_substrings", [])
    fact_result_list = [r for rows in fact_searches.values() for r in rows]

    return {
        "id": item["id"],
        "state_db": str(state_db),
        "memory_db": str(memory_db),
        "session_search_ok": _contains_any(session_data.get("results", []), expected_session) if expected_session else None,
        "fact_search_ok": _contains_any(fact_result_list, expected_fact) if expected_fact else None,
        "session_search": {
            "count": session_data.get("count"),
            "matches": [
                {
                    "session_id": r.get("session_id"),
                    "match_message_id": r.get("match_message_id"),
                    "snippet": r.get("snippet"),
                }
                for r in session_data.get("results", [])
            ],
        },
        "fact_searches": fact_searches,
        "probes": probes,
        "reasons": reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSON corpus path")
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = parser.parse_args()

    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="hermes-memory-recall-") as td:
        workdir = Path(td)
        report = {
            "corpus": str(args.corpus),
            "workdir": str(workdir),
            "items": [evaluate_item(item, workdir) for item in corpus],
        }
        output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
