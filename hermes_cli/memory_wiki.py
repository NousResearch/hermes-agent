"""CLI helpers for exporting local memory-wiki indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_memory_wiki_index(args) -> int:
    """Export or print the read-only built-in memory wiki index."""

    from agent.memory_wiki import build_memory_wiki_index, select_memory_context
    from hermes_constants import get_hermes_home

    index = build_memory_wiki_index()
    query = getattr(args, "query", None)
    if query:
        payload = {
            "index_version": index["version"],
            "selection": select_memory_context(
                query,
                index=index,
                max_chars=int(getattr(args, "max_chars", 1200) or 1200),
            ),
        }
    else:
        payload = index

    out = getattr(args, "out", None)
    if out:
        path = Path(out).expanduser()
        if not path.is_absolute():
            path = get_hermes_home() / "memory-wiki" / path
        _write_json(path, payload)
        print(f"  ✓ Wrote memory wiki index: {path}")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
