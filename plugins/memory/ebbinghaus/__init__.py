"""Local Ebbinghaus-inspired memory provider.

This plugin models durable memory as encoded cue sets with a simple
Ebbinghaus forgetting curve:

    retention = exp(-elapsed_days / stability_days)

Memories are stored locally in SQLite, searched with lexical/cue overlap,
and strengthened by explicit recall or rehearsal.
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from agent.memory_provider import MemoryProvider
from hermes_cli.config import cfg_get
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    content          TEXT NOT NULL UNIQUE,
    encoded          TEXT NOT NULL,
    cues             TEXT DEFAULT '',
    tags             TEXT DEFAULT '',
    salience         REAL DEFAULT 0.6,
    valence          REAL DEFAULT 0.0,
    strength         REAL DEFAULT 1.0,
    rehearsal_count  INTEGER DEFAULT 0,
    retrieval_count  INTEGER DEFAULT 0,
    source           TEXT DEFAULT '',
    session_id       TEXT DEFAULT '',
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    last_rehearsed_at REAL,
    last_retrieved_at REAL
);

CREATE INDEX IF NOT EXISTS idx_ebbinghaus_tags ON memories(tags);
CREATE INDEX IF NOT EXISTS idx_ebbinghaus_updated ON memories(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_ebbinghaus_salience ON memories(salience DESC);
"""

_TOKEN_RE = re.compile(r"[\w][\w.+#:/-]{1,}", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]+")
_SPACE_RE = re.compile(r"\s+")
_MAX_STRENGTH = 6.0
_STOPWORDS = {
    "about", "after", "also", "and", "are", "because", "been", "but",
    "can", "could", "for", "from", "has", "have", "into", "not", "of",
    "our", "the", "that", "this", "use", "was", "were", "with", "you",
    "your", "です", "ます", "して", "した", "こと", "これ", "それ",
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def forgetting_retention(elapsed_days: float, stability_days: float) -> float:
    """Return retention in [0, 1] for the Ebbinghaus exponential curve."""
    if elapsed_days <= 0:
        return 1.0
    stability = max(0.01, float(stability_days))
    return _clamp(math.exp(-float(elapsed_days) / stability), 0.0, 1.0)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return _SPACE_RE.sub(" ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Encode text into retrieval cues.

    The tokenizer is intentionally local and dependency-free. It captures
    latin/identifier-like terms plus short kana/kanji n-grams so Japanese
    memories can still be found without a morphological analyzer.
    """
    normalized = _normalize_text(text).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0).strip("._-/#:")
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    for chunk in _CJK_RE.findall(normalized):
        if len(chunk) < 2:
            continue
        tokens.extend(chunk[i:i + 2] for i in range(0, len(chunk) - 1))
        if len(chunk) >= 3:
            tokens.extend(chunk[i:i + 3] for i in range(0, len(chunk) - 2))
    return tokens


def _cue_counts(text: str | Iterable[str]) -> Counter:
    if isinstance(text, str):
        return Counter(_tokenize(text))
    counts: Counter = Counter()
    for item in text:
        counts.update(_tokenize(str(item)))
    return counts


def _top_cues(counts: Counter, limit: int = 64) -> list[str]:
    return [
        token for token, _count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
    ]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    overlap = set(a) & set(b)
    numerator = sum(a[token] * b[token] for token in overlap)
    if numerator <= 0:
        return 0.0
    left = math.sqrt(sum(value * value for value in a.values()))
    right = math.sqrt(sum(value * value for value in b.values()))
    if left == 0 or right == 0:
        return 0.0
    return float(numerator / (left * right))


def _split_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = re.split(r"[,;\n]", tags)
    elif isinstance(tags, Iterable):
        raw = [str(item) for item in tags]
    else:
        raw = [str(tags)]
    cleaned = []
    seen = set()
    for tag in raw:
        value = _normalize_text(tag).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _join_tags(tags: Iterable[str]) -> str:
    return ",".join(_split_tags(list(tags)))


def _encode_memory(content: str, tags: Iterable[str]) -> dict:
    counts = _cue_counts([content, *list(tags)])
    cues = _top_cues(counts)
    return {
        "version": 1,
        "kind": "cue_encoding",
        "summary": _normalize_text(content)[:280],
        "cue_vector": dict((token, counts[token]) for token in cues),
        "cues": cues,
        "length": len(content),
    }


class EbbinghausMemoryStore:
    """SQLite store for encoded memory traces."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        base_stability_days: float = 3.0,
        decay_threshold: float = 0.08,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_stability_days = max(0.05, float(base_stability_days))
        self.decay_threshold = _clamp(float(decay_threshold), 0.0, 1.0)
        self._time_fn = time_fn or time.time
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        try:
            from hermes_state import apply_wal_with_fallback
            apply_wal_with_fallback(self._conn, db_label="ebbinghaus_memory.db")
        except Exception:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> float:
        return float(self._time_fn())

    def remember(
        self,
        content: str,
        *,
        tags: Any = None,
        salience: float = 0.65,
        valence: float = 0.0,
        source: str = "",
        session_id: str = "",
    ) -> dict:
        content = _normalize_text(content)
        if not content:
            raise ValueError("content must not be empty")

        tag_list = _split_tags(tags)
        encoded = _encode_memory(content, tag_list)
        cues = " ".join(encoded["cues"])
        now = self._now()
        salience = _clamp(float(salience), 0.05, 1.0)
        valence = _clamp(float(valence), -1.0, 1.0)
        tag_text = _join_tags(tag_list)

        try:
            cur = self._conn.execute(
                """
                INSERT INTO memories (
                    content, encoded, cues, tags, salience, valence, strength,
                    source, session_id, created_at, updated_at, last_rehearsed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content, json.dumps(encoded, ensure_ascii=False), cues, tag_text,
                    salience, valence, 1.0 + salience, source, session_id,
                    now, now, now,
                ),
            )
            self._conn.commit()
            memory_id = int(cur.lastrowid)
            return {"memory_id": memory_id, "status": "remembered", **self.get(memory_id)}
        except sqlite3.IntegrityError:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE content = ?", (content,)
            ).fetchone()
            if not row:
                raise
            merged_tags = sorted(set(_split_tags(row["tags"])) | set(tag_list))
            self._conn.execute(
                """
                UPDATE memories
                SET tags = ?, salience = MAX(salience, ?),
                    rehearsal_count = rehearsal_count + 1,
                    strength = MIN(?, strength + 0.15),
                    last_rehearsed_at = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (_join_tags(merged_tags), salience, _MAX_STRENGTH, now, now, row["memory_id"]),
            )
            self._conn.commit()
            memory_id = int(row["memory_id"])
            return {"memory_id": memory_id, "status": "reinforced", **self.get(memory_id)}

    def get(self, memory_id: int) -> dict:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (int(memory_id),)
        ).fetchone()
        if not row:
            raise KeyError(f"memory_id not found: {memory_id}")
        return self._row_to_result(row, query_score=None)

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.12,
        reinforce: bool = False,
    ) -> list[dict]:
        query = _normalize_text(query)
        if not query:
            return []
        query_counts = _cue_counts(query)
        query_lower = query.lower()
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        scored: list[dict] = []

        for row in rows:
            encoded = self._decode(row["encoded"])
            memory_counts = Counter(encoded.get("cue_vector") or {})
            tags = _split_tags(row["tags"])
            tag_counts = _cue_counts(tags)
            lexical = _cosine(query_counts, memory_counts + tag_counts)
            substring = 0.35 if query_lower in str(row["content"]).lower() else 0.0
            tag_bonus = 0.12 if set(_split_tags(query)) & set(tags) else 0.0
            if lexical <= 0 and substring <= 0 and tag_bonus <= 0:
                continue

            retention = self._retention(row)
            salience = float(row["salience"] or 0.0)
            rehearsal_bonus = min(0.08, math.log1p(int(row["rehearsal_count"] or 0)) * 0.025)
            score = (
                max(lexical, substring) * 0.68
                + retention * 0.18
                + salience * 0.08
                + tag_bonus
                + rehearsal_bonus
            )
            if score < min_score:
                continue
            scored.append(self._row_to_result(row, query_score=score, retention=retention))

        scored.sort(key=lambda item: (item["score"], item["retention"], item["salience"]), reverse=True)
        results = scored[: max(1, int(limit))]
        if reinforce:
            for result in results:
                self._reinforce_retrieval(result["memory_id"])
            results = [self.get(result["memory_id"]) | {"score": result["score"]} for result in results]
        return results

    def rehearse(self, *, memory_id: int | None = None, query: str = "", limit: int = 1) -> list[dict]:
        targets: list[int] = []
        if memory_id is not None:
            targets.append(int(memory_id))
        elif query:
            targets.extend(item["memory_id"] for item in self.recall(query, limit=limit, reinforce=False))
        else:
            raise ValueError("memory_id or query is required")

        now = self._now()
        for target in targets:
            self._conn.execute(
                """
                UPDATE memories
                SET rehearsal_count = rehearsal_count + 1,
                    strength = MIN(?, strength + 0.25),
                    last_rehearsed_at = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                (_MAX_STRENGTH, now, now, target),
            )
        self._conn.commit()
        return [self.get(target) for target in targets]

    def forget(self, memory_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (int(memory_id),)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def decay(self, *, threshold: float | None = None, prune: bool = False, limit: int = 50) -> dict:
        threshold = self.decay_threshold if threshold is None else _clamp(float(threshold), 0.0, 1.0)
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        decayed = [
            self._row_to_result(row, query_score=None)
            for row in rows
            if self._retention(row) <= threshold
        ]
        decayed.sort(key=lambda item: item["retention"])
        decayed = decayed[: max(1, int(limit))]
        pruned: list[int] = []
        if prune and decayed:
            pruned = [int(item["memory_id"]) for item in decayed]
            self._conn.executemany(
                "DELETE FROM memories WHERE memory_id = ?",
                [(memory_id,) for memory_id in pruned],
            )
            self._conn.commit()
        return {"threshold": threshold, "decayed": decayed, "pruned": pruned}

    def sleep_cycle(
        self,
        *,
        rehearse_threshold: float = 0.45,
        forget_threshold: float | None = None,
        salience_keep_threshold: float = 0.7,
        prune: bool = False,
        limit: int = 200,
    ) -> dict:
        """Run a sleep-like memory maintenance pass.

        Low-retention high-salience traces are rehearsed to consolidate them;
        low-retention low-salience traces are marked forgotten and optionally
        pruned. This keeps the Ebbinghaus curve as the retention model while
        adding a nightly-style consolidation/forgetting policy.
        """
        rehearse_threshold = _clamp(float(rehearse_threshold), 0.0, 1.0)
        forget_threshold = (
            self.decay_threshold if forget_threshold is None else _clamp(float(forget_threshold), 0.0, 1.0)
        )
        salience_keep_threshold = _clamp(float(salience_keep_threshold), 0.0, 1.0)
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        snapshots = [self._row_to_result(row, query_score=None) for row in rows]
        snapshots.sort(key=lambda item: (item["retention"], -item["salience"]))
        snapshots = snapshots[: max(1, int(limit))]

        rehearsed: list[int] = []
        forgotten: list[int] = []
        for item in snapshots:
            memory_id = int(item["memory_id"])
            retention = float(item["retention"])
            salience = float(item["salience"])
            if retention <= rehearse_threshold and salience >= salience_keep_threshold:
                rehearsed.append(memory_id)
            elif retention <= forget_threshold and salience < salience_keep_threshold:
                forgotten.append(memory_id)

        if rehearsed:
            now = self._now()
            self._conn.executemany(
                """
                UPDATE memories
                SET rehearsal_count = rehearsal_count + 1,
                    strength = MIN(?, strength + 0.25),
                    last_rehearsed_at = ?, updated_at = ?
                WHERE memory_id = ?
                """,
                [(_MAX_STRENGTH, now, now, memory_id) for memory_id in rehearsed],
            )
        pruned: list[int] = []
        if prune and forgotten:
            pruned = forgotten[:]
            self._conn.executemany(
                "DELETE FROM memories WHERE memory_id = ?",
                [(memory_id,) for memory_id in pruned],
            )
        if rehearsed or pruned:
            self._conn.commit()

        return {
            "mode": "sleep_cycle",
            "reviewed": len(snapshots),
            "rehearse_threshold": rehearse_threshold,
            "forget_threshold": forget_threshold,
            "salience_keep_threshold": salience_keep_threshold,
            "rehearsed": rehearsed,
            "forgotten": forgotten,
            "pruned": pruned,
        }

    def list_memories(self, *, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (max(1, int(limit)),)
        ).fetchall()
        return [self._row_to_result(row, query_score=None) for row in rows]

    def stats(self) -> dict:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS count,
                   COALESCE(AVG(salience), 0) AS avg_salience,
                   COALESCE(SUM(rehearsal_count), 0) AS rehearsals,
                   COALESCE(SUM(retrieval_count), 0) AS retrievals
            FROM memories
            """
        ).fetchone()
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        retained = sum(1 for item in rows if self._retention(item) > self.decay_threshold)
        return {
            "count": int(row["count"]),
            "retained_count": retained,
            "avg_salience": round(float(row["avg_salience"]), 3),
            "rehearsal_count": int(row["rehearsals"]),
            "retrieval_count": int(row["retrievals"]),
            "decay_threshold": self.decay_threshold,
            "base_stability_days": self.base_stability_days,
            "db_path": str(self.db_path),
        }

    def _reinforce_retrieval(self, memory_id: int) -> None:
        now = self._now()
        self._conn.execute(
            """
            UPDATE memories
            SET retrieval_count = retrieval_count + 1,
                strength = MIN(?, strength + 0.08),
                last_retrieved_at = ?, updated_at = ?
            WHERE memory_id = ?
            """,
            (_MAX_STRENGTH, now, now, int(memory_id)),
        )
        self._conn.commit()

    def _stability_days(self, row: sqlite3.Row) -> float:
        salience = float(row["salience"] or 0.0)
        strength = float(row["strength"] or 1.0)
        rehearsals = int(row["rehearsal_count"] or 0)
        retrievals = int(row["retrieval_count"] or 0)
        multiplier = (
            0.45
            + (1.35 * salience)
            + (0.65 * math.log1p(rehearsals))
            + (0.25 * math.log1p(retrievals))
        )
        return max(0.05, self.base_stability_days * strength * multiplier)

    def _retention(self, row: sqlite3.Row) -> float:
        now = self._now()
        anchors = [
            float(row["created_at"] or now),
            float(row["last_rehearsed_at"] or 0),
            float(row["last_retrieved_at"] or 0),
        ]
        elapsed_days = max(0.0, (now - max(anchors)) / 86400.0)
        return forgetting_retention(elapsed_days, self._stability_days(row))

    def _row_to_result(
        self,
        row: sqlite3.Row,
        *,
        query_score: float | None,
        retention: float | None = None,
    ) -> dict:
        retention = self._retention(row) if retention is None else retention
        encoded = self._decode(row["encoded"])
        now = self._now()
        last_anchor = max(
            float(row["created_at"] or now),
            float(row["last_rehearsed_at"] or 0),
            float(row["last_retrieved_at"] or 0),
        )
        result = {
            "memory_id": int(row["memory_id"]),
            "content": row["content"],
            "tags": _split_tags(row["tags"]),
            "cues": encoded.get("cues", [])[:12],
            "salience": round(float(row["salience"] or 0.0), 3),
            "valence": round(float(row["valence"] or 0.0), 3),
            "retention": round(float(retention), 4),
            "stability_days": round(self._stability_days(row), 3),
            "age_days": round(max(0.0, (now - float(row["created_at"] or now)) / 86400.0), 3),
            "days_since_reinforcement": round(max(0.0, (now - last_anchor) / 86400.0), 3),
            "rehearsal_count": int(row["rehearsal_count"] or 0),
            "retrieval_count": int(row["retrieval_count"] or 0),
            "source": row["source"] or "",
            "session_id": row["session_id"] or "",
        }
        if query_score is not None:
            result["score"] = round(float(query_score), 4)
        return result

    @staticmethod
    def _decode(value: str) -> dict:
        try:
            decoded = json.loads(value or "{}")
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}


EBBINGHAUS_MEMORY_SCHEMA = {
    "name": "ebbinghaus_memory",
    "description": (
        "Local human-like memory. Encodes memories into retrieval cues, stores "
        "them in SQLite, recalls by cue overlap, and models decay with an "
        "Ebbinghaus forgetting curve. Use remember for durable facts, recall "
        "before relying on memory, rehearse to keep important memories strong, "
        "and decay/prune to inspect or remove forgotten traces."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["remember", "recall", "rehearse", "forget", "decay", "sleep", "list", "stats"],
            },
            "content": {"type": "string", "description": "Memory content for remember."},
            "query": {"type": "string", "description": "Cue/query for recall or rehearse."},
            "memory_id": {"type": "integer", "description": "Memory id for rehearse/forget."},
            "tags": {"type": "string", "description": "Comma-separated tags or cue labels."},
            "salience": {"type": "number", "description": "Importance from 0.05 to 1.0."},
            "valence": {"type": "number", "description": "Emotional valence from -1.0 to 1.0."},
            "limit": {"type": "integer", "description": "Maximum result count."},
            "min_score": {"type": "number", "description": "Minimum recall score."},
            "threshold": {"type": "number", "description": "Retention threshold for decay."},
            "rehearse_threshold": {"type": "number", "description": "Sleep-cycle retention threshold below which important memories are rehearsed."},
            "forget_threshold": {"type": "number", "description": "Sleep-cycle retention threshold below which low-salience memories are forgotten."},
            "salience_keep_threshold": {"type": "number", "description": "Sleep-cycle salience cutoff for consolidation instead of forgetting."},
            "prune": {"type": "boolean", "description": "Delete decayed or sleep-forgotten memories."},
        },
        "required": ["action"],
    },
}


def _load_plugin_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        import yaml

        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        with open(config_path, encoding="utf-8-sig") as handle:
            all_config = yaml.safe_load(handle) or {}
        return (
            cfg_get(all_config, "plugins", "ebbinghaus", default={})
            or cfg_get(all_config, "plugins", "ebbinghaus-memory", default={})
            or {}
        )
    except Exception:
        return {}


class EbbinghausMemoryProvider(MemoryProvider):
    """Local memory provider with cue encoding and forgetting-curve decay."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: EbbinghausMemoryStore | None = None
        self._session_id = ""
        self._max_prefetch = int(self._config.get("max_prefetch", 5))
        self._min_prefetch_score = float(self._config.get("min_prefetch_score", 0.16))
        self._auto_encode_turns = _as_bool(self._config.get("auto_encode_turns", False))

    @property
    def name(self) -> str:
        return "ebbinghaus"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        from hermes_constants import display_hermes_home
        return [
            {"key": "db_path", "description": "SQLite database path", "default": f"{display_hermes_home()}/ebbinghaus_memory.db"},
            {"key": "base_stability_days", "description": "Initial forgetting-curve stability in days", "default": "3.0"},
            {"key": "decay_threshold", "description": "Retention threshold considered forgotten", "default": "0.08"},
            {"key": "max_prefetch", "description": "Maximum memories injected before a turn", "default": "5"},
            {"key": "auto_encode_turns", "description": "Auto-store preference-like user turns", "default": "false", "choices": ["true", "false"]},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        try:
            import yaml
            config_path = Path(hermes_home) / "config.yaml"
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as handle:
                    existing = yaml.safe_load(handle) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["ebbinghaus"] = values
            with open(config_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(existing, handle, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            logger.debug("Ebbinghaus save_config failed: %s", exc)

    def initialize(self, session_id: str, **kwargs) -> None:
        raw_home = kwargs.get("hermes_home")
        if raw_home:
            hermes_home = Path(str(raw_home)).expanduser()
        else:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        default_db = hermes_home / "ebbinghaus_memory.db"
        db_path = str(self._config.get("db_path") or default_db)
        db_path = db_path.replace("$HERMES_HOME", str(hermes_home))
        db_path = db_path.replace("${HERMES_HOME}", str(hermes_home))
        self._store = EbbinghausMemoryStore(
            db_path,
            base_stability_days=float(self._config.get("base_stability_days", 3.0)),
            decay_threshold=float(self._config.get("decay_threshold", 0.08)),
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        stats = self._store.stats()
        return (
            "# Ebbinghaus Memory\n"
            f"Active. {stats['count']} encoded memories stored locally. "
            "Use ebbinghaus_memory to remember durable facts, recall relevant "
            "context, rehearse important traces, and prune forgotten ones."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query:
            return ""
        results = self._store.recall(
            query,
            limit=self._max_prefetch,
            min_score=self._min_prefetch_score,
            reinforce=False,
        )
        if not results:
            return ""
        lines = []
        for item in results:
            lines.append(
                "- "
                f"[retention={item['retention']:.2f}, salience={item['salience']:.2f}] "
                f"{item['content']}"
            )
        return "## Ebbinghaus Memory\n" + "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._auto_encode_turns or not self._store:
            return
        for content, salience in _extract_candidate_memories(user_content):
            try:
                self._store.remember(
                    content,
                    tags=["auto", "user"],
                    salience=salience,
                    source="sync_turn",
                    session_id=session_id or self._session_id,
                )
            except Exception as exc:
                logger.debug("Ebbinghaus sync_turn encode failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._auto_encode_turns or not self._store:
            return
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            for memory, salience in _extract_candidate_memories(content):
                try:
                    self._store.remember(
                        memory,
                        tags=["auto", "session"],
                        salience=salience,
                        source="session_end",
                        session_id=self._session_id,
                    )
                except Exception as exc:
                    logger.debug("Ebbinghaus session encode failed: %s", exc)

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if action not in {"add", "replace"} or not self._store or not content:
            return
        metadata = metadata or {}
        try:
            tags = ["built-in-memory", target]
            if metadata.get("platform"):
                tags.append(str(metadata["platform"]))
            self._store.remember(
                content,
                tags=tags,
                salience=0.8 if target == "user" else 0.7,
                source="memory_tool",
                session_id=str(metadata.get("session_id") or self._session_id),
            )
        except Exception as exc:
            logger.debug("Ebbinghaus memory_write mirror failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [EBBINGHAUS_MEMORY_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "ebbinghaus_memory":
            return tool_error(f"Unknown tool: {tool_name}")
        if not self._store:
            return tool_error("Ebbinghaus memory is not initialized")
        try:
            action = str(args.get("action", "")).lower()
            if action == "remember":
                return json.dumps(
                    self._store.remember(
                        args.get("content", ""),
                        tags=args.get("tags"),
                        salience=float(args.get("salience", 0.65)),
                        valence=float(args.get("valence", 0.0)),
                        source=str(args.get("source", "tool")),
                        session_id=self._session_id,
                    ),
                    ensure_ascii=False,
                )
            if action == "recall":
                return json.dumps(
                    {
                        "results": self._store.recall(
                            args.get("query", ""),
                            limit=int(args.get("limit", 5)),
                            min_score=float(args.get("min_score", 0.12)),
                            reinforce=True,
                        )
                    },
                    ensure_ascii=False,
                )
            if action == "rehearse":
                return json.dumps(
                    {
                        "results": self._store.rehearse(
                            memory_id=args.get("memory_id"),
                            query=args.get("query", ""),
                            limit=int(args.get("limit", 1)),
                        )
                    },
                    ensure_ascii=False,
                )
            if action == "forget":
                return json.dumps(
                    {"forgotten": self._store.forget(int(args["memory_id"]))},
                    ensure_ascii=False,
                )
            if action == "decay":
                return json.dumps(
                    self._store.decay(
                        threshold=args.get("threshold"),
                        prune=bool(args.get("prune", False)),
                        limit=int(args.get("limit", 50)),
                    ),
                    ensure_ascii=False,
                )
            if action == "sleep":
                return json.dumps(
                    self._store.sleep_cycle(
                        rehearse_threshold=float(args.get("rehearse_threshold", 0.45)),
                        forget_threshold=args.get("forget_threshold"),
                        salience_keep_threshold=float(args.get("salience_keep_threshold", 0.7)),
                        prune=bool(args.get("prune", False)),
                        limit=int(args.get("limit", 200)),
                    ),
                    ensure_ascii=False,
                )
            if action == "list":
                return json.dumps(
                    {"memories": self._store.list_memories(limit=int(args.get("limit", 20)))},
                    ensure_ascii=False,
                )
            if action == "stats":
                return json.dumps(self._store.stats(), ensure_ascii=False)
            return tool_error(f"Unknown action: {action}")
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def shutdown(self) -> None:
        if self._store:
            self._store.close()
            self._store = None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _extract_candidate_memories(text: str) -> list[tuple[str, float]]:
    normalized = _normalize_text(text)
    if len(normalized) < 12:
        return []
    lowered = normalized.lower()
    patterns = [
        ("remember", 0.9),
        ("don't forget", 0.9),
        ("do not forget", 0.9),
        ("i prefer", 0.8),
        ("i always", 0.75),
        ("i never", 0.75),
        ("my default", 0.75),
        ("覚えて", 0.9),
        ("忘れない", 0.9),
        ("好み", 0.75),
        ("いつも", 0.75),
        ("使う", 0.65),
    ]
    for marker, salience in patterns:
        if marker in lowered:
            return [(normalized[:700], salience)]
    return []


def register(ctx) -> None:
    """Register Ebbinghaus memory provider with the plugin system."""
    ctx.register_memory_provider(EbbinghausMemoryProvider())
