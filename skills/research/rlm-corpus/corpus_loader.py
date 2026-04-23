"""Load an ingested corpus from its JSON cache directory into memory.

The returned object is a plain dict keyed by filename. Values mirror the JSON
schema emitted by ``ingestion.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("rlm_corpus.loader")


def is_cache_dir(path: Path) -> bool:
    """A directory is treated as a cache dir if it contains at least one
    non-underscore-prefixed *.json file."""
    if not path.is_dir():
        return False
    for f in path.glob("*.json"):
        if not f.name.startswith("_"):
            return True
    return False


def load_corpus(cache_dir: Path) -> dict[str, dict[str, Any]]:
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache directory does not exist: {cache_dir}")

    corpus: dict[str, dict[str, Any]] = {}
    for f in sorted(cache_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue  # manifest / error log
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            log.warning("skipping malformed cache entry %s: %s", f, exc)
            continue
        # Use the original filename (not the flattened cache filename) as the
        # corpus key so citations like [paper.pdf] resolve cleanly.
        original_path = Path(payload.get("file_path", f.stem))
        key = original_path.name
        # Disambiguate collisions by falling back to the flattened name.
        if key in corpus:
            key = f.stem
        corpus[key] = payload

    log.info("loaded %d documents from %s", len(corpus), cache_dir)
    return corpus


def corpus_summary(corpus: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_docs": len(corpus),
        "total_chars": sum(
            (d.get("stats") or {}).get("char_count", 0) for d in corpus.values()
        ),
        "files": [
            {
                "filename": name,
                "title": (d.get("metadata") or {}).get("title"),
                "char_count": (d.get("stats") or {}).get("char_count", 0),
            }
            for name, d in corpus.items()
        ],
    }
