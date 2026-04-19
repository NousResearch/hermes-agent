"""Content-addressed LLM response cache.

Every completed ``LLMClient.complete()`` call is keyed by a blake2b hash
of its request body (model + system + user + temperature + max_tokens +
seed) and stored in the experiment's SQLite file. A second call with
the same request returns the cached response without touching the
network or the budget ledger.

Why per-experiment rather than global
-------------------------------------
Cache-as-file ships with the experiment directory, so:

* ``evolver replay --seed N`` is bit-for-bit reproducible on any machine
  that has the experiment dir — no shared ``~/.cache`` required.
* Copying / zipping an experiment preserves its cached history.
* Cross-experiment leakage is structurally impossible; the cache cannot
  serve a response from a different experiment's model / prompt.

Why blake2b-16 (128 bits)
-------------------------
Matches the existing content-addressing scheme used in ``storage.py``
for candidate IDs; 128 bits is comfortably past the birthday bound for
the populations we run (≤ 10^6 cached rows per experiment).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Optional


def request_body_fingerprint(body: dict) -> str:
    """Canonical fingerprint of a chat-completion request body.

    We deliberately ignore the ``headers`` and any fields we don't care
    about (``stream``, ``user`` tag, etc.) so logically-equivalent calls
    hit the same cache row.
    """
    # Keep the normalised shape stable — if we ever add a field that
    # legitimately changes the semantic meaning of a response, bump the
    # version string to invalidate old entries.
    payload = {
        "v":           "1",
        "model":       body.get("model", ""),
        "temperature": float(body.get("temperature", 0.0)),
        "max_tokens":  int(body.get("max_tokens", 0)),
        "seed":        body.get("seed"),
        "messages":    body.get("messages", []),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()


@dataclass
class CacheEntry:
    response:          str
    prompt_tokens:     int
    completion_tokens: int
    model:             str
    created_at:        int


@dataclass
class ResponseCache:
    """Thin SQLite-backed key-value cache.

    The cache reuses the experiment's ``lineage.db`` (passed as a live
    connection) so it shares WAL / journal / foreign-key config with the
    rest of the skill and is picked up by ``storage.lineage_hash`` for
    replay-hash computation.
    """

    conn: sqlite3.Connection
    hits:   int = field(default=0, init=False)
    misses: int = field(default=0, init=False)

    def get(self, body: dict) -> Optional[CacheEntry]:
        key = request_body_fingerprint(body)
        row = self.conn.execute(
            "SELECT response, prompt_tokens, completion_tokens, model, created_at "
            "FROM llm_cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return CacheEntry(
            response=row["response"],
            prompt_tokens=int(row["prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            model=row["model"],
            created_at=int(row["created_at"]),
        )

    def put(
        self,
        body: dict,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> str:
        key = request_body_fingerprint(body)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(key, response, prompt_tokens, completion_tokens, model, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, response, int(prompt_tokens), int(completion_tokens),
                 body.get("model", ""), int(time.time())),
            )
        return key

    def stats(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*)                                 AS rows, "
            "       COALESCE(SUM(prompt_tokens), 0)          AS in_toks, "
            "       COALESCE(SUM(completion_tokens), 0)      AS out_toks "
            "FROM llm_cache"
        ).fetchone()
        return {
            "rows":      int(row["rows"]),
            "in_toks":   int(row["in_toks"]),
            "out_toks":  int(row["out_toks"]),
            "hits":      self.hits,
            "misses":    self.misses,
        }

    def purge(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM llm_cache").fetchone()
        n = int(row["n"])
        with self.conn:
            self.conn.execute("DELETE FROM llm_cache")
        return n
