"""MemPalace memory plugin — MemoryProvider interface.

The highest-scoring AI memory system on GitHub (96.6% LongMemEval R@5).
Stores verbatim text in ChromaDB, temporal knowledge in SQLite.

This plugin uses the mempalace CLI (via system Python /usr/bin/python3)
to avoid dependency conflicts with hermes-agent's venv Python.

Local-only: no API key, no network calls, all storage on-machine.

7-Layer Memory Architecture:
  L-WM  Working Memory  │ In-memory LRU cache, 50 recent turns, zero latency
  L0    Identity Layer │ identity.txt
  L1    Narrative      │ Auto-generated essential story summary
  L2    Semantic       │ ChromaDB drawers (facts, preferences)
  L3    Episodic       │ Raw conversation logs with speaker/time/topic
  L4    Procedural     │ Step sequences, workflows, code patterns
  KG    Knowledge Graph│ Temporal, typed, inverse relations
  🔗    Cross-Palace   │ Hermes ↔ OpenClaw knowledge linking
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import subprocess
import threading
import os
import sys
import time
import zlib
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from plugins.memory.mempalace import backup as _backup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAL Batcher — zero-latency episodic memory durability
# ---------------------------------------------------------------------------
#
# Episodic writes are first appended to an NDJSON WAL file (O_APPEND,
# atomic, <1ms). A singleton background thread periodically flushes
# buffered episodes to ChromaDB in batches. On startup any un-flushed
# WAL entries are replayed automatically.
#
# This removes ChromaDB + ONNX embedding latency from the critical path
# and guarantees at-least-once persistence even if the process crashes.
# ---------------------------------------------------------------------------

MAX_WORKING_MEMORY_TURNS = 50

# Default palace path — Hermes has its OWN independent palace
_DEFAULT_PALACE_PATH = Path.home() / ".mempalace_hermes"
_IDENTITY_PATH = Path.home() / ".mempalace_hermes" / "identity.txt"

# Global singleton batcher state (shared across all provider instances)
_WAL_LOCK = threading.Lock()
_WAL_BATCHER: Optional[threading.Thread] = None
_WAL_BATCHER_STOP = threading.Event()


def _wal_path() -> Path:
    """Return the path to the episodic WAL file."""
    return Path.home() / ".mempalace_hermes" / "episodes.wal.ndjson"


def _health_path() -> Path:
    """Return the path to the health status JSON file."""
    return Path.home() / ".mempalace_hermes" / "health.json"


def _ensure_mempalace_dirs() -> None:
    """Create palace / logs directories if missing."""
    (Path.home() / ".mempalace_hermes" / "logs").mkdir(parents=True, exist_ok=True)


def _append_episode_wal(entry: Dict[str, Any]) -> None:
    """Append a single episode entry to the WAL file (thread-safe)."""
    _ensure_mempalace_dirs()
    path = _wal_path()
    line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
    with _WAL_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    _update_health(last_wal_append_at=datetime.now().isoformat())
    _backup.enqueue_incremental([str(path)])


def _read_wal() -> List[Dict[str, Any]]:
    """Read all un-flushed episodes from the WAL file."""
    path = _wal_path()
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with _WAL_LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Corrupt WAL line skipped: %s", line[:200])
        except Exception as e:
            logger.warning("Failed to read WAL: %s", e)
    return entries


def _truncate_wal() -> None:
    """Clear the WAL file after successful flush."""
    path = _wal_path()
    with _WAL_LOCK:
        try:
            if path.exists():
                path.write_text("")
        except Exception as e:
            logger.warning("Failed to truncate WAL: %s", e)


def _update_health(**kwargs) -> None:
    """Atomic update of health.json with the given key/value pairs."""
    path = _health_path()
    _ensure_mempalace_dirs()
    with _WAL_LOCK:
        try:
            data: Dict[str, Any] = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            data.update(kwargs)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            _backup.enqueue_incremental([str(path)])
        except Exception as e:
            logger.debug("Health update failed: %s", e)


def _bulk_add_episodes_to_chromadb(entries: List[Dict[str, Any]]) -> bool:
    """Flush a batch of episode entries to ChromaDB via subprocess."""
    if not entries:
        return True

    palace_path = _HERMES_PALACE
    script = (
        "import chromadb, json, uuid\n"
        "entries = REPLACEME_ENTRIES\n"
        "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
        "try:\n"
        "    col = client.get_collection('episodes')\n"
        "except Exception:\n"
        "    col = client.create_collection('episodes')\n"
        "ids = []\n"
        "docs = []\n"
        "metas = []\n"
        "for e in entries:\n"
        "    ids.append(str(uuid.uuid4()))\n"
        "    docs.append(e['content'][:4000])\n"
        "    metas.append({\n"
        "        'speaker': e.get('speaker', ''),\n"
        "        'role': e.get('role', ''),\n"
        "        'timestamp': e.get('timestamp', ''),\n"
        "        'topic': e.get('topic', 'general'),\n"
        "        'language': e.get('language', 'unknown'),\n"
        "        'importance': 0.5,\n"
        "        'source': 'hermes-sync',\n"
        "        'session_id': e.get('session_id', 'global')\n"
        "    })\n"
        "col.add(ids=ids, documents=docs, metadatas=metas)\n"
        "print('ok:' + str(len(entries)))\n"
    ).replace("REPLACEME_PALACE", repr(palace_path)).replace(
        "REPLACEME_ENTRIES", repr(entries)
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0 and result.stdout.strip().startswith("ok:"):
            return True
        logger.warning("ChromaDB bulk add failed: %s", result.stderr or result.stdout)
    except Exception as e:
        logger.warning("ChromaDB bulk add exception: %s", e)
    return False


class _WALBatcher(threading.Thread):
    """
    Singleton background thread that periodically flushes the episodic WAL
    to ChromaDB in batches.  Runs for the lifetime of the Python process.
    """

    _FLUSH_INTERVAL = 30.0   # seconds
    _MAX_BATCH = 100         # episodes per batch

    def __init__(self):
        super().__init__(daemon=True, name="mempalace-wal-batcher")
        # CRITICAL FIX: Do NOT name this _stop — Python 3.9 threading.Thread
        # has an internal _stop() method used by join(). Overwriting it with
        # an Event object causes TypeError on join().
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def run(self):
        logger.info("WAL batcher started (flush interval=%ss)", self._FLUSH_INTERVAL)
        while not self._stop_event.is_set():
            # Wait in small slices so we react quickly to stop()
            for _ in range(int(self._FLUSH_INTERVAL)):
                if self._stop_event.is_set():
                    break
                time.sleep(1.0)

            entries = _read_wal()
            if not entries:
                continue

            # Cap batch size and preserve ordering
            batch = entries[:self._MAX_BATCH]
            batch_len = len(batch)
            success = _bulk_add_episodes_to_chromadb(batch)
            if success:
                # CRITICAL FIX: Under lock, re-read WAL to capture any new appends
                # during flush, then truncate and rewrite only un-flushed entries.
                # This prevents race condition where new entries are lost.
                _ensure_mempalace_dirs()
                with _WAL_LOCK:
                    try:
                        current_entries = _read_wal()
                        # Keep entries that weren't in our batch (including new ones)
                        to_keep = current_entries[batch_len:]
                        # Atomic truncate and rewrite
                        _truncate_wal()
                        if to_keep:
                            with open(_wal_path(), "a", encoding="utf-8") as f:
                                for e in to_keep:
                                    f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
                    except Exception as e:
                        logger.warning("WAL batcher truncate/rewrite failed: %s", e)

                _update_health(
                    last_chromadb_flush_at=datetime.now().isoformat(),
                    pending_episodes=len(to_keep) if success else len(entries),
                    chromadb_ok=True,
                )
            else:
                _update_health(
                    last_chromadb_flush_error=datetime.now().isoformat(),
                    pending_episodes=len(entries),
                    chromadb_ok=False,
                )
        logger.info("WAL batcher stopped")


def _start_wal_batcher() -> None:
    """Start the singleton WAL batcher thread if it isn't already running."""
    global _WAL_BATCHER
    with _WAL_LOCK:
        if _WAL_BATCHER is None or not _WAL_BATCHER.is_alive():
            _WAL_BATCHER = _WALBatcher()
            _WAL_BATCHER.start()


def _stop_wal_batcher(timeout: float = 5.0) -> None:
    """Signal the WAL batcher to stop and wait for clean shutdown."""
    global _WAL_BATCHER
    batcher = None
    with _WAL_LOCK:
        if _WAL_BATCHER and _WAL_BATCHER.is_alive():
            batcher = _WAL_BATCHER
            batcher.request_stop()
    # CRITICAL FIX: join() must happen OUTSIDE the lock to avoid deadlock
    # with the batcher thread which acquires _WAL_LOCK during its run cycle.
    if batcher:
        batcher.join(timeout=timeout)


# ---------------------------------------------------------------------------
# Working Memory — in-process LRU cache of recent conversation turns
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """A single conversation turn in working memory."""
    role: str          # 'user' or 'assistant'
    speaker: str        # 'mars', 'hermes', 'feishu', etc.
    content: str
    timestamp: str      # ISO format
    topic: str = ""     # auto-detected topic tag
    importance: float = 0.5  # 0.0-1.0, default medium


class WorkingMemory:
    """
    In-memory LRU cache of the most recent conversation turns.
    Zero latency — no disk access.
    
    Thread-safe for concurrent access from the agent loop.
    """

    def __init__(self, max_turns: int = MAX_WORKING_MEMORY_TURNS):
        self._max = max_turns
        self._turns: OrderedDict[str, Turn] = OrderedDict()
        self._lock = threading.RLock()
        self._session_id: Optional[str] = None

    def set_session(self, session_id: str) -> None:
        """Clear working memory when starting a new session."""
        with self._lock:
            if session_id != self._session_id:
                self._turns.clear()
                self._session_id = session_id

    def add_turn(self, role: str, content: str, speaker: str = "hermes",
                 topic: str = "", importance: float = 0.5) -> str:
        """Add a turn to working memory. Episodic persistence is handled by sync_turn()."""
        import threading as _t

        timestamp = datetime.now().isoformat()
        detected_topic = topic or self._detect_topic(content) or "general"

        with self._lock:
            turn_id = f"{self._session_id or 'global'}:{timestamp}"
            turn = Turn(
                role=role,
                speaker=speaker,
                content=content[:5000],  # cap long content
                timestamp=timestamp,
                topic=detected_topic,
                importance=importance,
            )
            self._turns[turn_id] = turn
            # Evict oldest if over capacity
            while len(self._turns) > self._max:
                self._turns.popitem(last=False)

        return turn_id

    def _add_turn_no_persist(self, role: str, content: str, speaker: str = "hermes",
                             topic: str = "", importance: float = 0.5,
                             timestamp: str = "") -> str:
        """
        Internal add_turn without L3 ChromaDB persistence.
        Used by _restore_episodes to rebuild working memory from episodic storage
        without triggering re-persistence (which would cause infinite loop).
        """
        ts = timestamp or datetime.now().isoformat()
        detected_topic = topic or self._detect_topic(content) or "general"
        with self._lock:
            turn_id = f"{self._session_id or 'global'}:{ts}"
            turn = Turn(
                role=role,
                speaker=speaker,
                content=content[:5000],
                timestamp=ts,
                topic=detected_topic,
                importance=importance,
            )
            self._turns[turn_id] = turn
            while len(self._turns) > self._max:
                self._turns.popitem(last=False)
            return turn_id

    def get_recent(self, n: int = 10) -> List[Turn]:
        """Return the n most recent turns (newest last)."""
        with self._lock:
            items = list(self._turns.values())
        return items[-n:] if n < len(items) else items

    def search(self, query: str, n: int = 5) -> List[Turn]:
        """Full-text search within working memory turns."""
        q = query.lower()
        with self._lock:
            # Score by relevance (simple keyword match)
            scored = []
            for turn in self._turns.values():
                score = 0
                text = turn.content.lower()
                for word in q.split():
                    if word in text:
                        score += 1
                if score > 0:
                    scored.append((score, turn))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:n]]

    def get_context_for_wakeup(self) -> str:
        """Format recent turns as a string for injection at wakeup."""
        turns = self.get_recent(20)
        if not turns:
            return ""
        lines = ["## Working Memory (recent turns)\n"]
        for t in turns:
            lines.append(f"[{t.speaker}/{t.role}] {t.content[:300]}")
            if len(t.content) > 300:
                lines[-1] += "..."
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        """Return working memory stats."""
        with self._lock:
            return {
                "turn_count": len(self._turns),
                "max_turns": self._max,
                "session_id": self._session_id,
            }

    @staticmethod
    def _detect_topic(content: str) -> str:
        """Simple topic detection from content keywords."""
        content_lower = content.lower()
        topics = {
            "代码/编程": ["code", "function", "class", "import", "def ", "bug", "api", "git"],
            "记忆系统": ["memory", "mempalace", "chroma", "drawer", "kg", "knowledge graph"],
            "OpenClaw": ["openclaw", "agent", "chen", "ying", "wei", "cron", "task"],
            "项目/任务": ["project", "task", "implement", "build", "feature", "pr", "repo"],
            "系统配置": ["config", "setup", "install", "python", "path", "env", "api key"],
            "飞书": ["feishu", "lark", "飞书", "bot", "message", "dm"],
        }
        for topic, keywords in topics.items():
            if any(kw in content_lower for kw in keywords):
                return topic
        return ""
# L3: Episodic Memory — ChromaDB-backed raw conversation logs
# Persists every working-memory turn to ChromaDB for long-term episodic recall.
# Collection: "episodes" with speaker/time/topic metadata.
# -------------------------------------------------------------------


def _episode_get_recent(speaker: str = None, topic: str = None,
                        session_id: str = None, limit: int = 20) -> str:
    """
    Retrieve recent episodic turns from ChromaDB 'episodes' collection.
    Returns JSON string with episodes list.

    Uses offset+limit pagination: first count matching IDs, then fetch only
    the last LIMIT items. Avoids loading all documents into memory.
    """
    script = (
        "import chromadb, json\n"
        "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
        "col = client.get_or_create_collection('episodes')\n"
        "where = {}\n"
        "if SPEAKER_VAL is not None:\n"
        "    where['speaker'] = SPEAKER_VAL\n"
        "if TOPIC_VAL is not None:\n"
        "    where['topic'] = TOPIC_VAL\n"
        "if SESSION_ID_VAL is not None:\n"
        "    where['session_id'] = SESSION_ID_VAL\n"
        "# 1) Count matching IDs (lightweight, no docs)\n"
        "id_results = col.get(where=where if where else None, include=[])\n"
        "matched_ids = id_results.get('ids', [])\n"
        "total = len(matched_ids)\n"
        "limit = LIMIT_VAL\n"
        "offset = max(0, total - limit)\n"
        "# 2) Fetch only the last LIMIT items\n"
        "if offset < total:\n"
        "    results = col.get(\n"
        "        where=where if where else None,\n"
        "        offset=offset,\n"
        "        limit=limit,\n"
        "        include=['documents', 'metadatas']\n"
        "    )\n"
        "else:\n"
        "    results = {'documents': [], 'metadatas': []}\n"
        "episodes = []\n"
        "for doc, meta in zip(results.get('documents', []), results.get('metadatas', [])):\n"
        "    episodes.append({\n"
        "        'content': doc,\n"
        "        'speaker': meta.get('speaker', ''),\n"
        "        'role': meta.get('role', ''),\n"
        "        'topic': meta.get('topic', ''),\n"
        "        'timestamp': meta.get('timestamp', ''),\n"
        "        'importance': meta.get('importance', 0.5),\n"
        "        'session_id': meta.get('session_id', ''),\n"
        "    })\n"
        "# Sort by timestamp descending (newest first)\n"
        "episodes.sort(key=lambda e: e.get('timestamp', ''), reverse=True)\n"
        "print(json.dumps({'episodes': episodes, 'count': len(episodes), 'total': total}))\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "SPEAKER_VAL", repr(speaker)
    ).replace("TOPIC_VAL", repr(topic)).replace(
        "SESSION_ID_VAL", repr(session_id)
    ).replace(
        "LIMIT_VAL", str(limit)
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=15
    )
    code, stdout, stderr = result.returncode, result.stdout, result.stderr
    if code != 0:
        return json.dumps({"error": "Episode retrieval failed: %s" % stderr, "episodes": []})
    return stdout


# Global working memory instance (shared across all provider instances)
_working_memory = WorkingMemory()

# Lock to serialize episodic ChromaDB writes (prevents concurrent subprocess races)
_episodes_lock = threading.Lock()

# -------------------------------------------------------------------
# Knowledge Graph — typed entities, inverse relations, reification
# ----------------------------------------------------------------------------

# Predicate inverse map — automatically generates reverse relations
PREDICATE_INVERSES = {
    # Symmetric predicates (inverse is same)
    "knows": "known_by",
    "known_by": "knows",
    "friends_with": "friends_with",
    "collaborates_with": "collaborates_with",
    "works_with": "works_with",
    "related_to": "related_to",
    "similar_to": "similar_to",
    # Asymmetric predicates
    "uses": "used_by",
    "used_by": "uses",
    "depends_on": "depended_on_by",
    "depended_on_by": "depends_on",
    "created_by": "creates",
    "creates": "created_by",
    "owns": "owned_by",
    "owned_by": "owns",
    "manages": "managed_by",
    "managed_by": "manages",
    "part_of": "contains",
    "contains": "part_of",
    "before": "after",
    "after": "before",
    "causes": "caused_by",
    "caused_by": "causes",
}

# Entity type system for typed queries
ENTITY_TYPES = {
    "person": ["mars", "hermes", "chen", "ying", "wei", "城哥", "城哥专属助理"],
    "agent": ["openclaw", "chen", "ying", "wei", "main", "hermes-agent"],
    "project": ["hermes-agent", "mempalace", "openclaw", "mem-palace-consolidation"],
    "concept": ["memory", "knowledge-graph", "working-memory", "episodic-memory"],
    "tool": ["mempalace", "openclaw", "browser", "terminal"],
    "location": ["~/.mempalace_hermes/", "~/.mempalace/", "~/.hermes/"],
}


def _get_inverse_predicate(predicate: str) -> str:
    """Return the inverse of a predicate, or None if no inverse defined."""
    return PREDICATE_INVERSES.get(predicate)


def _guess_entity_type(entity: str) -> str:
    """Guess entity type from name."""
    entity_lower = entity.lower()
    for etype, keywords in ENTITY_TYPES.items():
        if any(kw.lower() in entity_lower or entity_lower in kw.lower() for kw in keywords):
            return etype
    return "concept"  # default


def _store_triple_with_inverse(
    subject: str, predicate: str, obj: str,
    valid_from: str, context: str = "", source: str = "",
    confidence: float = 1.0, subject_type: str = "", object_type: str = ""
) -> tuple[bool, str]:
    """
    Store a triple AND its inverse automatically.
    Detects contradictions: if (S, P, old_O) exists with valid_to=NULL,
    invalidate old_O and record as 'corrected' in belief_history.
    Returns (success, message).
    """
    import uuid
    today = datetime.now().isoformat()

    subject_type = subject_type or _guess_entity_type(subject)
    object_type = object_type or _guess_entity_type(obj)
    inverse_pred = _get_inverse_predicate(predicate)

    def _store():
        try:
            conn = sqlite3.connect(_HERMES_KG)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            cur = conn.cursor()

            # ── Contradiction detection: check for active (S, P, old_O) ──
            contradictions_found = []
            cur.execute(
                "SELECT id, object, valid_from FROM triples WHERE subject=? AND predicate=? AND (valid_to IS NULL OR valid_to='')",
                (subject, predicate)
            )
            existing_rows = cur.fetchall()
            for existing in existing_rows:
                old_triple_id, old_obj, old_valid_from = existing
                if old_obj != obj:
                    # Contradiction: new value differs from old value
                    contradictions_found.append({
                        "old_triple_id": old_triple_id,
                        "old_object": old_obj,
                        "old_valid_from": old_valid_from,
                    })
                    # Invalidate old triple
                    cur.execute(
                        "UPDATE triples SET valid_to=? WHERE id=?",
                        (today, old_triple_id)
                    )
                    # Record correction in belief_history
                    belief_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO belief_history VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (belief_id, subject, predicate, old_obj, obj, 'corrected',
                         today, source, confidence, context, old_valid_from, today)
                    )
                    # Also invalidate inverse of old triple (must match object=subject to avoid
                    # bulk-invalidating unrelated triples with same subject+predicate)
                    cur.execute(
                        "UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND (valid_to IS NULL OR valid_to='')",
                        (today, old_obj, inverse_pred or predicate, subject)
                    )

            sid = str(uuid.uuid4())
            # Main triple (valid_to='' for active triples)
            cur.execute(
                "INSERT INTO triples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, subject, predicate, obj, valid_from, '', confidence,
                 '', '', today, inverse_pred or '', context, source, subject_type, object_type)
            )
            # Inverse triple
            if inverse_pred:
                inv_sid = str(uuid.uuid4())
                cur.execute(
                    "INSERT INTO triples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (inv_sid, obj, inverse_pred, subject, valid_from, '', confidence,
                     '', '', today, predicate, context, source, object_type, subject_type)
                )
            # Record in belief_history
            belief_id = str(uuid.uuid4())
            cur.execute(
                "INSERT INTO belief_history VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (belief_id, subject, predicate, '', obj, 'created', today, source, confidence, context, valid_from, '')
            )
            conn.commit()
            conn.close()

            if contradictions_found:
                return True, ("Triple stored with %d contradiction(s) auto-invalidated. "
                              "belief_history recorded as 'corrected'. "
                              "Old object(s): %s" % (
                                  len(contradictions_found),
                                  ", ".join(c["old_object"] for c in contradictions_found)
                              ))
            return True, "Triple + inverse(%s) stored with belief history" % (inverse_pred or "none")
        except Exception as e:
            return False, str(e)

    # Run in thread to avoid blocking
    import threading
    result = [None]
    def _run():
        result[0] = _store()

    t = threading.Thread(target=_run)
    t.start()
    t.join(timeout=5)
    if result[0] is None:
        return False, "KG write timed out after 5 seconds"
    # Trigger incremental backup of the KG file on successful write
    if result[0] and getattr(result[0], "__getitem__", lambda i: None)(0) is True:
        _backup.enqueue_incremental([_HERMES_KG])
    return result[0]

# Path to system Python that has mempalace installed
_MEMPALACE_PYTHON = sys.executable
_MEMPALACE_CLI = [_MEMPALACE_PYTHON, "-m", "mempalace"]

# Default palace path — Hermes has its OWN independent palace
_DEFAULT_PALACE_PATH = Path.home() / ".mempalace_hermes"
_IDENTITY_PATH = Path.home() / ".mempalace_hermes" / "identity.txt"

# ------------------------------------------------------------------
# Tool schemas
# ------------------------------------------------------------------

WAKEUP_SCHEMA = {
    "name": "mempalace_wakeup",
    "description": (
        "Recall L0 (identity) + L1 (essential story) memories — the palace wake-up sequence. "
        "Fast (~600-900 tokens), zero search. Use at conversation start or when context is empty. "
        "Returns: who you are, essential facts, ongoing projects."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mempalace_search",
    "description": (
        "Deep semantic search across all palace memories. Searches verbatim text stored in ChromaDB. "
        "Use for: finding facts from past conversations, project history, people, decisions, preferences. "
        "Returns ranked results with similarity scores AND per-result explanations (why found)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "wing": {"type": "string", "description": "Filter to a specific wing (person/project)."},
            "room": {"type": "string", "description": "Filter to a specific room within a wing."},
            "n_results": {"type": "integer", "description": "Max results (default: 8, max: 30)."},
        },
        "required": ["query"],
    },
}

ADD_DRAWER_SCHEMA = {
    "name": "mempalace_add_drawer",
    "description": (
        "Store a memory in MemPalace with automatic L0/L1/L2 tiering. "
        "L0 = ~30 char one-line summary (fast relevance check). "
        "L1 = ~300 char key points (decision making). "
        "L2 = verbatim original (loaded on demand). "
        "Content is stored RAW — never summarized or paraphrased. "
        "Use wing='convos' for conversation content, wing='facts' for explicit facts, "
        "wing='<person>' for things about a specific person. "
        "Importance score (0-1) affects retrieval ranking and decay rate. "
        "Auto-detects duplicates (simhash) and contradictions (KG conflict detection)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The verbatim text to store."},
            "wing": {"type": "string", "description": "Wing name — e.g. 'convos', 'facts', or a person's name."},
            "room": {"type": "string", "description": "Room name — e.g. '2025-04', 'project-x', or a person's name."},
            "hall": {"type": "string", "description": "Hall type: facts, events, discoveries, preferences, advice (default: facts)."},
            "importance": {"type": "number", "description": "Importance score 0.0-1.0 (default: 0.5). High scores preserved longer."},
            "l0": {"type": "string", "description": "Optional: manually provide L0 summary (~30 chars). Auto-generated if omitted."},
            "l1": {"type": "string", "description": "Optional: manually provide L1 summary (~300 chars). Auto-generated if omitted."},
        },
        "required": ["content", "wing", "room"],
    },
}

KG_QUERY_SCHEMA = {
    "name": "mempalace_kg_query",
    "description": (
        "Query the temporal knowledge graph — entity relationships with time validity. "
        "Facts have valid_from/valid_to timestamps; old facts are invalidated, not deleted. "
        "Use for: timeline of events, how relationships evolved, historical facts."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to query."},
            "as_of": {"type": "string", "description": "Query graph as of this date (YYYY-MM-DD). Omit for current."},
        },
        "required": ["entity"],
    },
}

KG_ADD_SCHEMA = {
    "name": "mempalace_kg_add",
    "description": (
        "Add a temporal fact to the knowledge graph as a triple: subject → predicate → object. "
        "Example: ('Mars', 'works on', 'Hermes Agent'). "
        "Facts are time-versioned — use invalidate to mark when they stop being true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Relationship/predicate."},
            "object": {"type": "string", "description": "Object entity."},
            "valid_from": {"type": "string", "description": "Start date (YYYY-MM-DD). Default: today."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

KG_INVALIDATE_SCHEMA = {
    "name": "mempalace_kg_invalidate",
    "description": (
        "Invalidate a temporal fact — mark when a relationship stopped being true. "
        "Use after adding a corrected fact to replace an outdated one."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Relationship/predicate."},
            "object": {"type": "string", "description": "Object entity."},
            "ended": {"type": "string", "description": "End date (YYYY-MM-DD). Default: today."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

KG_STATS_SCHEMA = {
    "name": "mempalace_kg_stats",
    "description": "Get knowledge graph overview — entity count, triple count, connectivity stats.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

RECORD_CORRECTION_SCHEMA = {
    "name": "mempalace_record_correction",
    "description": (
        "Record a user correction — e.g. when the user says '这不是欧拉好猫'. "
        "Stores two KG triples: (subject, made_error, wrong_value) and "
        "(subject, corrected_to, correct_value), plus a belief_history entry "
        "with change_type='corrected'. "
        "Use this whenever the user corrects a factual error, misidentification, "
        "or wrong assumption."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Who made the error (default: 'Hermes')."},
            "wrong_value": {"type": "string", "description": "What was wrong — the incorrect value or statement."},
            "correct_value": {"type": "string", "description": "What is correct — the correction."},
            "context": {"type": "string", "description": "Optional context about where/how the error occurred."},
        },
        "required": ["wrong_value", "correct_value"],
    },
}

KG_ADD_TYPED_SCHEMA = {
    "name": "mempalace_kg_add_typed",
    "description": (
        "Add a typed triple to the knowledge graph with automatic inverse relation generation. "
        "Example: add (Mars, works_with, Hermes) AND automatically adds (Hermes, works_with, Mars). "
        "Supports entity types (person/agent/project/concept/tool/location), context window, "
        "and confidence scores. Use this instead of kg_add for richer knowledge representation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "Subject entity."},
            "predicate": {"type": "string", "description": "Predicate/relationship (e.g. knows, works_with, uses)."},
            "object": {"type": "string", "description": "Object entity."},
            "subject_type": {"type": "string", "description": "Type of subject: person, agent, project, concept, tool, location."},
            "object_type": {"type": "string", "description": "Type of object: person, agent, project, concept, tool, location."},
            "valid_from": {"type": "string", "description": "Start date YYYY-MM-DD (default: today)."},
            "context": {"type": "string", "description": "Context window — background/conditions for this fact."},
            "source": {"type": "string", "description": "Source of this knowledge (e.g. 'hermes-tool', 'user-input')."},
            "confidence": {"type": "number", "description": "Confidence 0.0-1.0 (default: 1.0)."},
        },
        "required": ["subject", "predicate", "object"],
    },
}

KG_QUERY_DECOMPOSED_SCHEMA = {
    "name": "mempalace_kg_query_decomposed",
    "description": (
        "Query planner for complex knowledge graph questions. "
        "Automatically decomposes queries like '城哥最近关注什么' or 'Hermes和OpenClaw什么关系' "
        "into sub-queries across entity types, predicates, and time ranges. "
        "Returns structured analysis with typed results, inverse relations, and context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language query about knowledge graph."},
            "entity": {"type": "string", "description": "Optional: focus on specific entity."},
            "entity_type": {"type": "string", "description": "Optional: filter by entity type (person/agent/project/concept/tool)."},
        },
        "required": ["query"],
    },
}

KG_BELIEF_HISTORY_SCHEMA = {
    "name": "mempalace_kg_belief_history",
    "description": (
        "Track how a specific belief/fact evolved over time. "
        "Shows the full history of changes to an entity's relationships: "
        "when a belief was created, updated, corrected, or invalidated. "
        "Example: 'Mars works_with Hermes' was created on 2026-04-14. "
        "Use to understand the evolution of knowledge and how facts changed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity to get belief history for."},
            "predicate": {"type": "string", "description": "Optional: filter by specific predicate."},
            "as_of": {"type": "string", "description": "Query belief history as of this date (YYYY-MM-DD)."},
        },
        "required": ["entity"],
    },
}

PALACE_STATUS_SCHEMA = {
    "name": "mempalace_status",
    "description": "Get palace overview — total drawers, wings, rooms, identity status, storage path.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

LIST_WINGS_SCHEMA = {
    "name": "mempalace_list_wings",
    "description": "List all wings in the palace with drawer counts.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GET_WORKING_MEMORY_SCHEMA = {
    "name": "mempalace_get_working_memory",
    "description": (
        "Get recent conversation turns from working memory (L-WM). "
        "Working memory stores the last 50 turns in-process with zero latency. "
        "Use to recall what was just discussed without disk access. "
        "Returns: speaker, role, content, timestamp, topic, importance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of recent turns to return (default: 10, max: 50)."},
            "search": {"type": "string", "description": "Optional: search query to filter turns by relevance."},
        },
        "required": [],
    },
}

SEARCH_WORKING_MEMORY_SCHEMA = {
    "name": "mempalace_search_working_memory",
    "description": (
        "Full-text search within working memory turns. "
        "Use for finding recent context: 'what did we decide about X', 'last time we talked about Y'. "
        "Returns ranked turns with relevance scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "n": {"type": "integer", "description": "Max results (default: 5)."},
        },
        "required": ["query"],
    },
}

EPISODES_SCHEMA = {
    "name": "mempalace_episodes",
    "description": (
        "Query L3 episodic long-term memory — raw conversation logs stored in ChromaDB. "
        "Unlike working memory (L-WM) which is session-scoped, L3 persists across sessions. "
        "Use for: 'what were we working on last time', 'did we discuss X before', "
        "'what happened in previous sessions'. "
        "Each episode has: speaker, role, content, topic, timestamp, session_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "speaker": {"type": "string", "description": "Filter by speaker (e.g., 'mars', 'hermes')."},
            "topic": {"type": "string", "description": "Filter by topic tag (e.g., '记忆系统', '代码/编程')."},
            "limit": {"type": "integer", "description": "Max episodes to return (default: 20, max: 100)."},
        },
        "required": [],
    },
}

CROSS_PALACE_SEARCH_SCHEMA = {
    "name": "mempalace_cross_palace_search",
    "description": (
        "Parallel search across BOTH Hermes and OpenClaw palaces simultaneously. "
        "Use when supervising OpenClaw agents or when a task might involve both palaces. "
        "Returns merged results from both sources, deduplicated and ranked by relevance. "
        "This is the primary tool for cross-palace reasoning tasks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "n_results": {"type": "integer", "description": "Max results per palace (default: 5)."},
        },
        "required": ["query"],
    },
}

# -----------------------------------------------------------------------
# L7: Proactive Prediction Retrieval — predict relevant memories before user asks
# -----------------------------------------------------------------------

PROACTIVE_PREDICT_SCHEMA = {
    "name": "mempalace_proactive_predict",
    "description": (
        "Predict and pre-fetch memories relevant to a given topic or conversation context. "
        "This is PROACTIVE — use it BEFORE the user asks, when starting a new session, "
        "detecting a topic shift, or as a background prefetch. "
        "Searches KG (entities + relationships) and ChromaDB drawers simultaneously. "
        "Returns structured predictions: likely entities involved, related facts, "
        "past decisions, and relevant procedural memories. "
        "Use when: session starts, topic changes, or before major tasks. "
        "The 'context' param should describe what the user is currently working on or asking about."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Topic or subject to predict relevant memories for. "
                    "Examples: 'Python coding', 'OpenClaw agent debugging', 'memory system', 'GitHub PR review'. "
                    "Be specific for best results — 'OpenClaw cron setup' not just 'OpenClaw'."
                ),
            },
            "context": {
                "type": "string",
                "description": (
                    "Optional additional context: what the user is trying to accomplish, "
                    "current task, or conversation direction. Helps refine predictions. "
                    "Example: 'debugging why OpenClaw agent times out', 'setting up daily standup bot'"
                ),
            },
            "depth": {
                "type": "string",
                "enum": ["shallow", "medium", "deep"],
                "description": "How deep to search (default: medium). shallow=KG only, medium=KG+drawers, deep=KG+drawers+episodes.",
            },
        },
        "required": ["topic"],
    },
}

# -----------------------------------------------------------------------
# L8: Memory Reflection — self-healing system audit
# -----------------------------------------------------------------------

MEMORY_REFLECTION_SCHEMA = {
    "name": "mempalace_memory_reflection",
    "description": (
        "Run self-healing audit of the entire memory system. "
        "Detects: orphaned entities (no relations), dangling relations (dead entities), "
        "duplicate memories (high simhash similarity), stale high-importance drawers, "
        "KG inconsistencies. Returns a structured audit report with issue counts "
        "and suggested fixes. Run weekly or after major data migrations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fix": {
                "type": "boolean",
                "description": "If true, automatically apply fixes for fixable issues (default: false).",
            },
        },
        "required": [],
    },
}

# -----------------------------------------------------------------------
# Memory Consolidation — importance decay + L2 eviction
# -----------------------------------------------------------------------

MEMORY_CONSOLIDATE_SCHEMA = {
    "name": "mempalace_consolidate",
    "description": (
        "Run memory consolidation: importance decay, L2 eviction, orphan cleanup. "
        "Decays importance scores based on access patterns (30+ days unused → score-0.2, "
        "90+ days → L2 evicted, only L0/L1 kept). "
        "High-importance (>0.7) memories are protected. "
        "This is normally run daily via cron automatically. "
        "Use manually after bulk imports or when memory feels bloated."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If true, show what would change without applying (default: false).",
            },
        },
        "required": [],
    },
}

# -----------------------------------------------------------------------
# KG Alias Table — cross-palace entity resolution
# -----------------------------------------------------------------------

KG_ALIAS_ADD_SCHEMA = {
    "name": "mempalace_kg_alias_add",
    "description": (
        "Register an entity alias for cross-palace entity resolution. "
        "Example: 'Mars' in Hermes palace = '城哥' in OpenClaw palace = 'ou_85a0...' (bot open_id). "
        "Once registered, cross-palace searches automatically resolve aliases to find related facts. "
        "Use after establishing new shared entities between Hermes and OpenClaw."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Canonical entity name."},
            "alias": {"type": "string", "description": "Alias or alternative name for the same entity."},
            "entity_type": {"type": "string", "description": "Type: person, agent, project, concept, tool, location."},
            "palace": {"type": "string", "description": "Which palace this alias belongs to: 'hermes', 'openclaw', or 'shared'."},
        },
        "required": ["entity", "alias"],
    },
}

KG_ALIAS_RESOLVE_SCHEMA = {
    "name": "mempalace_kg_alias_resolve",
    "description": (
        "Resolve an entity name to all known aliases across both Hermes and OpenClaw palaces. "
        "Use before cross-palace searches to find all names an entity might appear under. "
        "Returns canonical name + all known aliases grouped by palace."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to resolve."},
        },
        "required": ["entity"],
    },
}

# -----------------------------------------------------------------------
# Import / Export
# -----------------------------------------------------------------------

MEMORY_EXPORT_SCHEMA = {
    "name": "mempalace_export",
    "description": (
        "Export all memory data to a JSON file: KG triples, belief_history, "
        "ChromaDB collections metadata (not raw vectors), drawer metadata. "
        "This creates a portable backup that can be imported to any MemPalace instance. "
        "Filepath is optional — defaults to ~/.mempalace_hermes/export_<timestamp>.json."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Optional output filepath."},
            "include_chromadb": {"type": "boolean", "description": "Include ChromaDB raw content (default: true)."},
        },
        "required": [],
    },
}

MEMORY_IMPORT_SCHEMA = {
    "name": "mempalace_import",
    "description": (
        "Import memory from a JSON export file (mempalace_export format) "
        "or from MemPalace JSON format. Merges into existing memory — "
        "duplicates are detected via simhash and skipped. "
        "KG triples are merged preserving existing valid_from timestamps. "
        "Use to migrate memories between palaces or restore from backup."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filepath": {"type": "string", "description": "Path to JSON export file."},
            "dry_run": {"type": "boolean", "description": "If true, show what would be imported without importing (default: false)."},
        },
        "required": ["filepath"],
    },
}

OPENCLAW_WAKEUP_SCHEMA = {
    "name": "openclaw_wakeup",
    "description": (
        "Read OpenClaw team's L0 (identity) + L1 (essential story) memories — READ-ONLY. "
        "Use to inspect what OpenClaw agents know about themselves and their tasks. "
        "Returns: OpenClaw identity, essential facts, ongoing projects."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

OPENCLAW_SEARCH_SCHEMA = {
    "name": "openclaw_search",
    "description": (
        "Search OpenClaw team memories — READ-ONLY. "
        "Use to inspect OpenClaw agents' stored knowledge: conversations, decisions, task outcomes. "
        "Helps Hermes understand what OpenClaw has done/knows when supervising or debugging."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for in OpenClaw memories."},
            "wing": {"type": "string", "description": "Filter to a specific wing."},
            "room": {"type": "string", "description": "Filter to a specific room."},
            "n_results": {"type": "integer", "description": "Max results (default: 8, max: 30)."},
        },
        "required": ["query"],
    },
}

OPENCLAW_STATUS_SCHEMA = {
    "name": "openclaw_status",
    "description": (
        "Get OpenClaw palace overview — READ-ONLY. "
        "Shows total drawers, wings, rooms, identity, and storage path of OpenClaw's memory system. "
        "Use to audit what memories OpenClaw has accumulated."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

OPENCLAW_KG_QUERY_SCHEMA = {
    "name": "openclaw_kg_query",
    "description": (
        "Query OpenClaw's temporal knowledge graph — READ-ONLY. "
        "Shows entity relationships OpenClaw has stored with time validity. "
        "Use to see what facts OpenClaw agents have recorded and how they evolve."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name to query."},
            "as_of": {"type": "string", "description": "Query graph as of this date (YYYY-MM-DD)."},
        },
        "required": ["entity"],
    },
}

# ------------------------------------------------------------------
# CLI helpers
# ------------------------------------------------------------------

def _has_mempalace_cli() -> bool:
    """Check if system Python has mempalace CLI available."""
    return shutil.which(_SYSTEM_PYTHON) is not None


def _run_cli(args: List[str], timeout: int = 30) -> tuple[int, str, str]:
    """Run mempalace CLI command targeting Hermes's independent palace."""
    # --palace is a GLOBAL arg on the main parser; must come before subcommand
    cmd = _MEMPALACE_CLI + ["--palace", _HERMES_PALACE] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", f"System Python not found: {_MEMPALACE_PYTHON}"
    except Exception as e:
        return -1, "", str(e)


def _run_python(script: str, timeout: int = 15) -> tuple[int, str, str]:
    """Run inline Python script with the same Python running Hermes. Returns (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            [_MEMPALACE_PYTHON, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Python command timed out"
    except FileNotFoundError:
        return -1, "", f"System Python not found: {_MEMPALACE_PYTHON}"
    except Exception as e:
        return -1, "", str(e)


# Paths for Hermes's independent palace
_HERMES_PALACE = str(Path.home() / ".mempalace_hermes/palace")
_HERMES_IDENTITY = str(Path.home() / ".mempalace_hermes/identity.txt")
_HERMES_KG = str(Path.home() / ".mempalace_hermes/knowledge_graph.sqlite3")

# OpenClaw palace paths (READ-ONLY for Hermes — shared storage)
_OPENCLAW_PALACE = str(Path.home() / ".mempalace/palace")
_OPENCLAW_IDENTITY = str(Path.home() / ".mempalace/identity.txt")
_OPENCLAW_KG = str(Path.home() / ".mempalace/knowledge_graph.sqlite3")


def _run_mempalace(script: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run mempalace Python API with Hermes's independent palace paths."""
    return _run_python(script, timeout=timeout)


def _cli_search_with_explanation(query: str, wing: str = None, room: str = None,
                                  n_results: int = 8) -> str:
    """
    Run mempalace search with BM25 reranking + per-result explanations.
    1. Query ChromaDB directly for semantic matches
    2. Apply BM25 keyword reranking to re-score results
    3. Return enriched results with bm25_score, keyword explanation, recency
    """
    n_fetch = min(n_results * 3, 30)  # Over-fetch for BM25 reranking

    script = (
        "import sys\n"
        "import chromadb, json, re\n"
        "from datetime import datetime\n"
        "\n"
        "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
        "\n"
        "# Query mempalace_drawers collection for semantic matches\n"
        "try:\n"
        "    col = client.get_collection('mempalace_drawers')\n"
        "    raw_results = col.query(\n"
        "        query_texts=[REPLACEME_QUERY],\n"
        "        n_results=REPLACEME_N,\n"
        "        include=['documents', 'metadatas', 'distances']\n"
        "    )\n"
        "except Exception as e:\n"
        "    print(json.dumps({'error': str(e), 'results': []}))\n"
        "    exit(0)\n"
        "\n"
        "# Flatten ChromaDB result structure\n"
        "documents = raw_results.get('documents', [[]])[0]\n"
        "metadatas = raw_results.get('metadatas', [[]])[0]\n"
        "distances = raw_results.get('distances', [[]])[0]\n"
        "\n"
        "items = []\n"
        "for i, doc in enumerate(documents):\n"
        "    if not doc:\n"
        "        continue\n"
        "    items.append({\n"
        "        'document': doc,\n"
        "        'metadata': metadatas[i] if i < len(metadatas) else {},\n"
        "        'chroma_distance': distances[i] if i < len(distances) else 1.0,\n"
        "    })\n"
        "\n"
        "# ── BM25 Reranking ──────────────────────────────────────────────\n"
        "bm25_scores = [None] * len(items)\n"
        "if len(items) >= 2:\n"
        "    try:\n"
        "        import re\n"
        "        def tokenize(text):\n"
        "            # Simple tokenizer: split on non-alphanumeric, lowercase\n"
        "            toks = re.findall(r'[a-zA-Z0-9]+', text.lower())\n"
        "            return [t for t in toks if len(t) > 1]\n"
        "        from rank_bm25 import BM25Okapi\n"
        "        corpus_texts = [item['document'] for item in items]\n"
        "        tokenized = [tokenize(doc) for doc in corpus_texts]\n"
        "        bm25 = BM25Okapi(tokenized)\n"
        "        q_tokens = tokenize(REPLACEME_QUERY)\n"
        "        raw_scores = bm25.get_scores(q_tokens)\n"
        "        bm25_scores = [float(s) for s in raw_scores]\n"
        "    except Exception as e:\n"
        "        pass  # BM25 not available — use chroma_distance fallback\n"
        "\n"
        "# ── Enrich with keyword/explanation ─────────────────────────────\n"
        "q_lower = REPLACEME_QUERY.lower()\n"
        "english_words = re.findall(r'[a-zA-Z0-9]+', q_lower)\n"
        "chinese_chars = [c for c in q_lower if not c.isalnum() and not c.isspace()]\n"
        "\n"
        "def explain(item, bm25_score):\n"
        "    reasons = []\n"
        "    doc_lower = item['document'].lower()\n"
        "    matched = []\n"
        "    for w in english_words:\n"
        "        if w in doc_lower:\n"
        "            matched.append('kw:' + w)\n"
        "    for c in chinese_chars:\n"
        "        if c in doc_lower:\n"
        "            matched.append('char:' + c)\n"
        "    if matched:\n"
        "        reasons.append('matched(' + str(len(matched)) + '): ' + ', '.join(matched[:5]))\n"
        "    if bm25_score is not None:\n"
        "        reasons.append('bm25=%.2f' % bm25_score)\n"
        "    meta = item.get('metadata', {})\n"
        "    created = meta.get('created_at', '')\n"
        "    if created:\n"
        "        try:\n"
        "            days = (datetime.now() - datetime.fromisoformat(created)).days\n"
        "            recency = max(0.0, 1.0 - days / 90.0)\n"
        "            reasons.append('recency=%.2f(%dd)' % (recency, days))\n"
        "        except Exception: pass\n"
        "    imp = meta.get('importance', 0.5)\n"
        "    if imp and imp != 0.5:\n"
        "        reasons.append('imp=%.1f' % float(imp))\n"
        "    return '; '.join(reasons) if reasons else 'semantic_match'\n"
        "\n"
        "enriched = []\n"
        "for i, item in enumerate(items):\n"
        "    bm25 = bm25_scores[i] if i < len(bm25_scores) else None\n"
        "    enriched.append({\n"
        "        'document': item['document'][:500],\n"
        "        'metadata': item['metadata'],\n"
        "        'explanation': explain(item, bm25),\n"
        "        'relevance_score': bm25 if bm25 is not None else (1.0 - float(item.get('chroma_distance', 0))),\n"
        "        'bm25_score': bm25,\n"
        "        'chroma_distance': item.get('chroma_distance'),\n"
        "    })\n"
        "\n"
        "# Sort by bm25_score (desc), fallback to chroma_distance\n"
        "enriched.sort(key=lambda x: (x['bm25_score'] if x['bm25_score'] is not None else -999, "
        "                             -(x['chroma_distance'] if x['chroma_distance'] is not None else 999)),\n"
        "             reverse=True)\n"
        "enriched = enriched[:REPLACEME_N]\n"
        "\n"
        "print(json.dumps({\n"
        "    'results': enriched,\n"
        "    'query': REPLACEME_QUERY,\n"
        "    'count': len(enriched),\n"
        "    'reranking': 'bm25',\n"
        "}))\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "REPLACEME_QUERY", repr(query)
    ).replace("REPLACEME_N", str(n_fetch)
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30
    )
    code, stdout, stderr = result.returncode, result.stdout, result.stderr
    if code != 0:
        return json.dumps({"error": "Search error: %s" % stderr})
    return stdout


def _cli_search(query: str, wing: str = None, room: str = None, n_results: int = 8) -> str:
    """Run mempalace search via Python API (Hermes's independent palace)."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "result = stack.search(REPLACEME_QUERY, wing=REPLACEME_WING, room=REPLACEME_ROOM, n_results=REPLACEME_N)\n"
        "print(result)\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_HERMES_IDENTITY)
    ).replace("REPLACEME_QUERY", repr(query)).replace(
        "REPLACEME_WING", repr(wing)
    ).replace("REPLACEME_ROOM", repr(room)).replace(
        "REPLACEME_N", str(n_results)
    )
    code, stdout, stderr = _run_mempalace(script)
    if code != 0:
        return f"Search error: {stderr}"
    return stdout


def _cli_wakeup(wing: str = None) -> str:
    """Run mempalace wake-up via Python API (Hermes's independent palace)."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "result = stack.wake_up(wing=REPLACEME_WING)\n"
        "print(result)\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_HERMES_IDENTITY)
    ).replace("REPLACEME_WING", repr(wing))
    code, stdout, stderr = _run_mempalace(script, timeout=20)
    if code != 0:
        return f"Wake-up error: {stderr}"
    return stdout


def _cli_status() -> Dict[str, Any]:
    """Run mempalace status via Python API."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "import json\n"
        "print(json.dumps(stack.status()))\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_HERMES_IDENTITY)
    )
    code, stdout, stderr = _run_mempalace(script)
    if code != 0:
        return {"error": f"Status error: {stderr}"}
    try:
        return json.loads(stdout)
    except Exception:
        return {"raw": stdout}


def _cli_list_wings() -> List[Dict[str, Any]]:
    """Get wing list by querying ChromaDB metadata for distinct wing values."""
    script = (
        "import chromadb\n"
        "import json\n"
        "palace_path = REPLACEME_PALACE\n"
        "client = chromadb.PersistentClient(path=palace_path)\n"
        "try:\n"
        "    col = client.get_collection('mempalace_drawers')\n"
        "except Exception:\n"
        "    print('[]')\n"
        "    exit(0)\n"
        "try:\n"
        "    data = col.get(limit=1000, include=['metadatas'])\n"
        "    wings = {}\n"
        "    for meta in (data.get('metadatas') or []):\n"
        "        w = meta.get('wing', 'unknown')\n"
        "        if w not in wings:\n"
        "            wings[w] = {'wing': w, 'drawer_count': 0}\n"
        "        wings[w]['drawer_count'] += 1\n"
        "    result = list(wings.values())\n"
        "except Exception:\n"
        "    result = []\n"
        "print(json.dumps(result))\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE))
    code, stdout, stderr = _run_python(script)
    if code != 0:
        logger.warning("list_wings failed: %s", stderr)
        return []
    try:
        return json.loads(stdout)
    except Exception:
        return []


def _detect_language(text: str) -> str:
    """Detect primary language: 'zh', 'en', or 'mixed'."""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_words = len(re.findall(r'[a-zA-Z0-9]+', text))
    total = chinese_chars + english_words
    if total == 0:
        return "unknown"
    zh_ratio = chinese_chars / total
    if zh_ratio > 0.3:
        return "zh"
    elif english_words > 0 and zh_ratio < 0.1:
        return "en"
    elif zh_ratio > 0 and zh_ratio < 0.3:
        return "mixed"
    return "en"


def _store_episodic_turn(speaker: str, role: str, content: str, timestamp: str, session_id: str = "") -> None:
    """Store a single turn in the 'episodes' ChromaDB collection (L3 episodic memory)."""
    topic = WorkingMemory._detect_topic(content) or "general"
    lang = _detect_language(content)

    script = (
        "import chromadb, uuid\n"
        "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
        "try:\n"
        "    col = client.get_collection('episodes')\n"
        "except Exception:\n"
        "    col = client.create_collection('episodes')\n"
        "episode_id = str(uuid.uuid4())\n"
        "col.add(\n"
        "    ids=[episode_id],\n"
        "    documents=[REPLACEME_CONTENT],\n"
        "    metadatas=[{\n"
        "        'speaker': REPLACEME_SPEAKER,\n"
        "        'role': REPLACEME_ROLE,\n"
        "        'timestamp': REPLACEME_TS,\n"
        "        'topic': REPLACEME_TOPIC,\n"
        "        'language': REPLACEME_LANG,\n"
        "        'importance': 0.5,\n"
        "        'source': 'hermes-sync',\n"
        "        'session_id': REPLACEME_SID\n"
        "    }]\n"
        "    )\n"
        "print('ok:' + episode_id)\n"
    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
        "REPLACEME_CONTENT", repr(content[:4000])
    ).replace("REPLACEME_SPEAKER", repr(speaker)).replace(
        "REPLACEME_ROLE", repr(role)
    ).replace("REPLACEME_TS", repr(timestamp)).replace(
        "REPLACEME_TOPIC", repr(topic)
    ).replace("REPLACEME_LANG", repr(lang)).replace(
        "REPLACEME_SID", repr(session_id or "global")
    )
    with _episodes_lock:
        code, stdout, stderr = _run_python(script, timeout=10)
    if code != 0:
        logger.debug("episodic store failed: %s", stderr)


# ------------------------------------------------------------------
# OpenClaw read-only helpers (Hermes reads OpenClaw's memory)
# ------------------------------------------------------------------

def _openclaw_search(query: str, wing: str = None, room: str = None, n_results: int = 8) -> str:
    """Search OpenClaw palace memories (READ-ONLY)."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "result = stack.search(REPLACEME_QUERY, wing=REPLACEME_WING, room=REPLACEME_ROOM, n_results=REPLACEME_N)\n"
        "print(result)\n"
    ).replace("REPLACEME_PALACE", repr(_OPENCLAW_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_OPENCLAW_IDENTITY)
    ).replace("REPLACEME_QUERY", repr(query)).replace(
        "REPLACEME_WING", repr(wing)
    ).replace("REPLACEME_ROOM", repr(room)).replace(
        "REPLACEME_N", str(n_results)
    )
    code, stdout, stderr = _run_mempalace(script)
    if code != 0:
        return f"OpenClaw search error: {stderr}"
    return stdout


def _openclaw_wakeup(wing: str = None) -> str:
    """Wake-up OpenClaw palace (READ-ONLY)."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "result = stack.wake_up(wing=REPLACEME_WING)\n"
        "print(result)\n"
    ).replace("REPLACEME_PALACE", repr(_OPENCLAW_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_OPENCLAW_IDENTITY)
    ).replace("REPLACEME_WING", repr(wing))
    code, stdout, stderr = _run_mempalace(script, timeout=20)
    if code != 0:
        return f"OpenClaw wake-up error: {stderr}"
    return stdout


def _openclaw_status() -> Dict[str, Any]:
    """Get OpenClaw palace status (READ-ONLY)."""
    script = (
        "from mempalace.layers import MemoryStack\n"
        "stack = MemoryStack(palace_path=REPLACEME_PALACE, identity_path=REPLACEME_IDENTITY)\n"
        "import json\n"
        "print(json.dumps(stack.status()))\n"
    ).replace("REPLACEME_PALACE", repr(_OPENCLAW_PALACE)).replace(
        "REPLACEME_IDENTITY", repr(_OPENCLAW_IDENTITY)
    )
    code, stdout, stderr = _run_mempalace(script)
    if code != 0:
        return {"error": f"OpenClaw status error: {stderr}"}
    try:
        return json.loads(stdout)
    except Exception:
        return {"raw": stdout}


def _openclaw_kg_query(entity: str, as_of: str = None) -> str:
    """Query OpenClaw knowledge graph (READ-ONLY)."""
    script = (
        "import sqlite3\n"
        "import json\n"
        "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
        "cur = conn.cursor()\n"
        "query = \"SELECT subject, predicate, object, valid_from, valid_to FROM triples WHERE subject=? AND (valid_to IS NULL OR valid_to='')\"\n"
        "params = REPLACEME_PARAMS\n"
        "rows = cur.execute(query, params).fetchall()\n"
        "cols = [d[0] for d in cur.description]\n"
        "result = [dict(zip(cols, r)) for r in rows]\n"
        "print(json.dumps(result))\n"
        "conn.close()\n"
    ).replace("REPLACEME_KG", repr(_OPENCLAW_KG)).replace("REPLACEME_PARAMS", repr([entity]))
    code, stdout, stderr = _run_python(script)
    if code != 0:
        return f"OpenClaw KG query error: {stderr}"
    return stdout


# ------------------------------------------------------------------
# MemoryProvider implementation
# ------------------------------------------------------------------

class MemPalaceMemoryProvider(MemoryProvider):
    """MemPalace local memory — verbatim storage, temporal KG, CLI-based."""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._palace_path: str = str(_DEFAULT_PALACE_PATH)
        self._prefetch_result: str = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._wing_for_convos: str = "convos"
        self._last_synced_user_hash: Optional[str] = None

    @property
    def name(self) -> str:
        return "mempalace"

    def is_available(self) -> bool:
        """Return True if system Python has mempalace and palace data exists."""
        if not _has_mempalace_cli():
            return False
        palace_dir = Path(self._palace_path)
        return palace_dir.exists()

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "palace_path",
                "description": "Path to MemPalace data directory",
                "default": str(_DEFAULT_PALACE_PATH),
            },
            {
                "key": "convo_wing",
                "description": "Wing name for conversation memories",
                "default": "convos",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to $HERMES_HOME/mempalace.json."""
        config_path = Path(hermes_home) / "mempalace.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize MemPalace: working memory session + restore L3 episodic context."""
        # Set session on working memory (clears if new session)
        _working_memory.set_session(session_id)

        # Load config from $HERMES_HOME/mempalace.json if present
        hermes_home = kwargs.get("hermes_home")
        if hermes_home:
            config_path = Path(hermes_home) / "mempalace.json"
            if config_path.exists():
                try:
                    file_cfg = json.loads(config_path.read_text())
                    self._palace_path = file_cfg.get("palace_path", self._palace_path)
                    self._wing_for_convos = file_cfg.get("convo_wing", self._wing_for_convos)
                except Exception:
                    pass

        # Override with CLI --palace flag path
        palace_dir = Path(self._palace_path)
        if not palace_dir.exists():
            # Try common locations
            for candidate in [
                Path.home() / ".mempalace_hermes",
                Path.home() / ".mempalace_hermes" / "palace",
            ]:
                if candidate.exists():
                    self._palace_path = str(candidate)
                    break

        self._config = {
            "palace_path": self._palace_path,
            "identity_path": str(_IDENTITY_PATH),
            "convo_wing": self._wing_for_convos,
        }

        # L3: Restore recent episodic turns from ChromaDB into working memory
        # This ensures long-term context is available even in a fresh session
        self._restore_episodes(session_id)

        # WAL: Replay any un-flushed episodes to ChromaDB (crash recovery)
        self._replay_pending_wal()

        # Start the WAL batcher for this process (singleton, idempotent)
        _start_wal_batcher()

        # Start the cloud backup worker (singleton, idempotent)
        _backup.start_backup_worker()

        # Quick check that the CLI works
        code, _, stderr = _run_cli(["status"])
        if code != 0:
            logger.warning("MemPalace CLI check failed: %s", stderr)
        else:
            logger.info("MemPalace provider initialized. palace_path=%s session=%s",
                        self._palace_path, session_id)

    def _restore_episodes(self, session_id: str) -> None:
        """
        Restore the most recent episodic turns from ChromaDB into working memory.
        This is called on every initialize() so that fresh sessions still have
        recent conversational context without losing it across sessions.

        Runs SYNCHRONOUSLY so that working memory is populated before
        initialize() returns — the agent must have context before processing
        its first message.
        """
        try:
            raw = _episode_get_recent(session_id=session_id, limit=MAX_WORKING_MEMORY_TURNS)
            data = json.loads(raw)
            episodes = data.get("episodes", [])
            # Populate working memory (oldest first, newest last)
            for ep in episodes:
                # Use internal add_turn but WITHOUT re-persisting to episodes
                # (avoid infinite loop and duplicate entries)
                _working_memory._add_turn_no_persist(
                    role=ep.get("role", "user"),
                    content=ep.get("content", ""),
                    speaker=ep.get("speaker", "hermes"),
                    topic=ep.get("topic", ""),
                    importance=float(ep.get("importance", 0.5)),
                    timestamp=ep.get("timestamp", ""),
                )
        except Exception as e:
            logger.warning("L3 episode restore failed: %s", e)

    def _replay_pending_wal(self) -> None:
        """
        Replay any un-flushed WAL entries to ChromaDB in batches.
        Called during initialize() for crash recovery.
        """
        try:
            entries = _read_wal()
            if not entries:
                return
            logger.info("Replaying %d pending episodes from WAL", len(entries))

            # CRITICAL FIX: Batch replay to avoid timeout on large backlogs.
            batch_size = 100
            all_success = True
            total_flushed = 0
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                success = _bulk_add_episodes_to_chromadb(batch)
                if success:
                    total_flushed += len(batch)
                else:
                    all_success = False
                    logger.warning("WAL replay batch %d failed, stopping replay", i // batch_size)
                    break

            if all_success:
                _truncate_wal()
                _update_health(
                    last_chromadb_flush_at=datetime.now().isoformat(),
                    pending_episodes=0,
                    chromadb_ok=True,
                )
                logger.info("WAL replay complete: %d episodes flushed", total_flushed)
            else:
                logger.warning("WAL replay partial failure, %d/%d episodes remain pending",
                               len(entries) - total_flushed, len(entries))
                _update_health(
                    last_chromadb_flush_error=datetime.now().isoformat(),
                    pending_episodes=len(entries) - total_flushed,
                    chromadb_ok=False,
                )
        except Exception as e:
            logger.warning("WAL replay failed: %s", e)

    def system_prompt_block(self) -> str:
        return (
            "# MemPalace Memory (7-Layer Architecture)\n"
            f"Active. Palace: {self._palace_path}\n"
            "Verbatim storage (no summarization). Temporal knowledge graph.\n"
            "\n"
            "Memory layers:\n"
            "• L-WM: Working memory — last 50 turns, zero latency (mempalace_get_working_memory)\n"
            "• L2: ChromaDB drawers — facts, preferences (mempalace_search, add_drawer)\n"
            "• L3: Episodic — raw conversation logs, cross-session (mempalace_episodes)\n"
            "• KG: Temporal knowledge graph (mempalace_kg_query, kg_add)\n"
            "• Cross-Palace: Search both Hermes+OpenClaw simultaneously (mempalace_cross_palace_search)\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        # Return cached prefetch result from previous queue_prefetch call
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## MemPalace Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background wake-up for the next turn."""

        def _run():
            try:
                wake = _cli_wakeup()
                if wake:
                    with self._prefetch_lock:
                        self._prefetch_result = wake[:2000]  # cap at 2000 chars
            except Exception as e:
                logger.debug("MemPalace wake-up failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="mempalace-wakeup"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Auto-file conversation turn to Working Memory (L-WM) and Episodic Memory (L3).

        - Working Memory: in-process LRU cache, zero latency, last 50 turns
        - Episodic Memory: WAL (fast append) + ChromaDB (async batch flush)
        """
        import hashlib

        # Ensure WAL batcher is running for this process
        _start_wal_batcher()

        today = datetime.now().isoformat()

        # --- Working Memory (L-WM) — zero latency ---
        user_turn_id = _working_memory.add_turn(
            role="user",
            content=user_content,
            speaker="mars",
            importance=0.5,
        )

        # --- Episodic Memory (L3) — WAL append (<1ms) ---
        # CRITICAL FIX: Deduplicate user-side WAL writes. run_agent.py calls
        # sync_all(user_msg, "") immediately after user input, then calls
        # sync_all(user_msg, assistant_reply) after generation. Without
        # deduplication the user message would be written to WAL twice.
        user_hash = hashlib.sha256(user_content.encode()).hexdigest()[:16] if user_content else None
        if user_content and user_hash != self._last_synced_user_hash:
            _append_episode_wal({
                "speaker": "mars",
                "role": "user",
                "content": user_content,
                "timestamp": today,
                "session_id": session_id or _working_memory._session_id or "global",
                "topic": WorkingMemory._detect_topic(user_content),
                "language": _detect_language(user_content),
            })
            self._last_synced_user_hash = user_hash

        if assistant_content:
            asst_turn_id = _working_memory.add_turn(
                role="assistant",
                content=assistant_content,
                speaker="hermes",
                importance=0.5,
            )
            _append_episode_wal({
                "speaker": "hermes",
                "role": "assistant",
                "content": assistant_content,
                "timestamp": today,
                "session_id": session_id or _working_memory._session_id or "global",
                "topic": WorkingMemory._detect_topic(assistant_content),
                "language": _detect_language(assistant_content),
            })

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            WAKEUP_SCHEMA,
            SEARCH_SCHEMA,
            ADD_DRAWER_SCHEMA,
            KG_QUERY_SCHEMA,
            KG_ADD_SCHEMA,
            KG_ADD_TYPED_SCHEMA,
            KG_INVALIDATE_SCHEMA,
            KG_STATS_SCHEMA,
            KG_QUERY_DECOMPOSED_SCHEMA,
            KG_BELIEF_HISTORY_SCHEMA,
            # User correction feedback — records errors and corrections in KG
            RECORD_CORRECTION_SCHEMA,
            PALACE_STATUS_SCHEMA,
            LIST_WINGS_SCHEMA,
            # Working Memory (L-WM) — zero latency in-process cache
            GET_WORKING_MEMORY_SCHEMA,
            SEARCH_WORKING_MEMORY_SCHEMA,
            # L3: Episodic — long-term conversation logs in ChromaDB
            EPISODES_SCHEMA,
            # Cross-Palace reasoning
            CROSS_PALACE_SEARCH_SCHEMA,
            # L7: Proactive Prediction Retrieval
            PROACTIVE_PREDICT_SCHEMA,
            # L8: Memory Reflection
            MEMORY_REFLECTION_SCHEMA,
            MEMORY_CONSOLIDATE_SCHEMA,
            KG_ALIAS_ADD_SCHEMA,
            KG_ALIAS_RESOLVE_SCHEMA,
            MEMORY_EXPORT_SCHEMA,
            MEMORY_IMPORT_SCHEMA,
            # OpenClaw read-only tools
            OPENCLAW_WAKEUP_SCHEMA,
            OPENCLAW_SEARCH_SCHEMA,
            OPENCLAW_STATUS_SCHEMA,
            OPENCLAW_KG_QUERY_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "mempalace_wakeup":
                return self._handle_wakeup(args)
            elif tool_name == "mempalace_search":
                return self._handle_search(args)
            elif tool_name == "mempalace_add_drawer":
                return self._handle_add_drawer(args)
            elif tool_name == "mempalace_kg_query":
                return self._handle_kg_query(args)
            elif tool_name == "mempalace_kg_add":
                return self._handle_kg_add(args)
            elif tool_name == "mempalace_kg_add_typed":
                return self._handle_kg_add_typed(args)
            elif tool_name == "mempalace_kg_invalidate":
                return self._handle_kg_invalidate(args)
            elif tool_name == "mempalace_kg_stats":
                return self._handle_kg_stats(args)
            elif tool_name == "mempalace_kg_query_decomposed":
                return self._handle_kg_query_decomposed(args)
            elif tool_name == "mempalace_kg_belief_history":
                return self._handle_kg_belief_history(args)
            elif tool_name == "mempalace_record_correction":
                return self._handle_record_correction(args)
            elif tool_name == "mempalace_status":
                return self._handle_status(args)
            elif tool_name == "mempalace_list_wings":
                return self._handle_list_wings(args)
            # Working Memory (L-WM) — zero latency
            elif tool_name == "mempalace_get_working_memory":
                return self._handle_get_working_memory(args)
            elif tool_name == "mempalace_search_working_memory":
                return self._handle_search_working_memory(args)
            # L3: Episodic long-term memory
            elif tool_name == "mempalace_episodes":
                return self._handle_episodes(args)
            # Cross-Palace reasoning
            elif tool_name == "mempalace_cross_palace_search":
                return self._handle_cross_palace_search(args)
            # L7: Proactive Prediction Retrieval
            elif tool_name == "mempalace_proactive_predict":
                return self._handle_proactive_predict(args)
            # L8: Memory Reflection + Consolidation + Alias
            elif tool_name == "mempalace_memory_reflection":
                return self._handle_memory_reflection(args)
            elif tool_name == "mempalace_consolidate":
                return self._handle_consolidate(args)
            elif tool_name == "mempalace_kg_alias_add":
                return self._handle_kg_alias_add(args)
            elif tool_name == "mempalace_kg_alias_resolve":
                return self._handle_kg_alias_resolve(args)
            elif tool_name == "mempalace_export":
                return self._handle_export(args)
            elif tool_name == "mempalace_import":
                return self._handle_import(args)
            # OpenClaw read-only tools
            elif tool_name == "openclaw_wakeup":
                return self._handle_openclaw_wakeup(args)
            elif tool_name == "openclaw_search":
                return self._handle_openclaw_search(args)
            elif tool_name == "openclaw_status":
                return self._handle_openclaw_status(args)
            elif tool_name == "openclaw_kg_query":
                return self._handle_openclaw_kg_query(args)
            else:
                return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.exception("MemPalace tool %s failed", tool_name)
            return tool_error(f"MemPalace {tool_name} failed: {e}")

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_wakeup(self, args: Dict[str, Any]) -> str:
        wing = args.get("wing")
        result = _cli_wakeup(wing=wing)
        if result.startswith("Wake-up error:"):
            return tool_error(result)

        # Inject working memory context
        wm_context = _working_memory.get_context_for_wakeup()
        if wm_context:
            result = f"{result}\n\n{wm_context}"

        return json.dumps({"result": result, "source": "L0+L1 wake-up + L-WM working memory"})

    def _handle_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        wing = args.get("wing")
        room = args.get("room")
        n = min(int(args.get("n_results", 8)), 30)
        # Use search with explanations
        result = _cli_search_with_explanation(query=query, wing=wing, room=room, n_results=n)
        if result.startswith("Search error:"):
            return tool_error(result)
        return result

    def _extract_entities_for_drawer(self, drawer_id: str, wing: str,
                                     room: str, content: str) -> None:
        """
        Background entity extraction from drawer content into KG.
        Finds known entities (from ENTITY_TYPES) in content and stores
        (entity, mentioned_in, drawer_id) triples in the KG.
        Also detects new potential entities (capitalized multi-word phrases).
        Fire-and-forget — never blocks the agent.
        """
        import threading as _t, re as _re

        def _do():
            # ── Extract known entities ────────────────────────────────────────
            content_lower = content.lower()
            found_entities = []

            # Check each known entity keyword
            for etype, keywords in ENTITY_TYPES.items():
                for kw in keywords:
                    if kw.lower() in content_lower:
                        found_entities.append((kw, etype))

            # ── Extract new potential entities ──────────────────────────────
            # Capitalized multi-word phrases in English
            capitalized = _re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', content)
            # Chinese-style entities (连续中文字符)
            chinese = _re.findall(r'[\u4e00-\u9fff]{2,}(?:[\u4e00-\u9fff·]+[\u4e00-\u9fff]{1,})*', content)

            # Store triples for each entity found
            for entity, etype in found_entities:
                _store_triple_with_inverse(
                    subject=entity,
                    predicate="mentioned_in",
                    obj=drawer_id,
                    valid_from=datetime.now().isoformat(),
                    context="wing=%s room=%s" % (wing, room),
                    source="auto-extract",
                    confidence=0.8,
                    subject_type=etype,
                    object_type="drawer",
                )

            # Store new potential entities as "detected_entity" with low confidence
            for phrase in set(capitalized[:5]):  # limit to 5
                if len(phrase) > 3:
                    _store_triple_with_inverse(
                        subject=phrase,
                        predicate="detected_as_entity",
                        obj="drawer:" + drawer_id,
                        valid_from=datetime.now().isoformat(),
                        context="auto-extracted from content",
                        source="auto-extract",
                        confidence=0.4,
                        subject_type="concept",
                        object_type="drawer",
                    )

        _t.Thread(target=_do, daemon=True).start()

    def _handle_add_drawer(self, args: Dict[str, Any]) -> str:
        content = args.get("content", "")
        wing = args.get("wing", "facts")
        room = args.get("room", "general")
        hall = args.get("hall", "facts")
        importance = float(args.get("importance", 0.5))
        manual_l0 = args.get("l0", "")
        manual_l1 = args.get("l1", "")

        if not content:
            return tool_error("Missing required parameter: content")

        # ── L0/L1 auto-generation ─────────────────────────────────────────
        content_len = len(content)
        if manual_l0:
            l0 = manual_l0
        elif content_len <= 30:
            l0 = content
        else:
            # Truncate to ~30 chars, preserving word boundary
            l0 = content[:35].rsplit(" ", 1)[0] if len(content) > 35 else content
            if len(l0) > 35:
                l0 = l0[:32].rstrip() + "..."

        if manual_l1:
            l1 = manual_l1
        elif content_len <= 300:
            l1 = content
        else:
            l1 = content[:330].rsplit(" ", 1)[0] if len(content) > 330 else content
            if len(l1) > 330:
                l1 = l1[:320].rstrip() + "..."

        # ── Simhash deduplication ─────────────────────────────────────────
        # Compute lightweight 32-bit fingerprint
        content_hash = str(zlib.crc32(content.encode("utf-8")) & 0xFFFFFFFF)

        lang = _detect_language(content)

        # Check for duplicate (exact hash match + simhash near-duplicate)
        check_script = (
            "import chromadb, hashlib, json\n"
            "def simhash(s):\n"
            "    sig = [0] * 64\n"
            "    words = s.lower().split()\n"
            "    for w in words:\n"
            "        h = hashlib.md5(w.encode()).digest()\n"
            "        for i in range(64):\n"
            "            sig[i] += 1 if h[i % 16] > 127 else -1\n"
            "    return sum((1 << i) if s > 0 else 0 for i, s in enumerate(sig))\n"
            "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
            "try:\n"
            "    col = client.get_collection(REPLACEME_WING)\n"
            "    data = col.get(limit=500, include=['metadatas', 'documents'])\n"
            "except Exception:\n"
            "    print(json.dumps({'result': 'new'}))\n"
            "    exit(0)\n"
            "metas = data.get('metadatas') or []\n"
            "docs = data.get('documents') or []\n"
            "ids = data.get('ids') or []\n"
            "new_hash = REPLACEME_HASH\n"
            "new_doc = REPLACEME_DOC\n"
            "new_sim = simhash(new_doc)\n"
            "for i, meta in enumerate(metas):\n"
            "    if not meta:\n"
            "        continue\n"
            "    if meta.get('content_hash') == new_hash:\n"
            "        print(json.dumps({'result': 'duplicate', 'id': ids[i], 'created_at': str(meta.get('created_at', ''))}))\n"
            "        break\n"
            "    if i < len(docs) and simhash(docs[i]) is not None:\n"
            "        diff = bin(new_sim ^ simhash(docs[i])).count('1')\n"
            "        if diff <= 5:\n"
            "            print(json.dumps({'result': 'near_duplicate', 'id': ids[i], 'diff': diff}))\n"
            "            break\n"
            "else:\n"
            "    print(json.dumps({'result': 'new'}))\n"
        ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
            "REPLACEME_WING", repr(wing)
        ).replace("REPLACEME_HASH", repr(content_hash)).replace(
            "REPLACEME_DOC", repr(content[:8000])
        )
        check_code, check_stdout, check_stderr = _run_python(check_script)
        try:
            check_result = json.loads(check_stdout.strip()) if check_stdout.strip() else {"result": "new"}
        except Exception:
            check_result = {"result": "new"}

        if check_result.get("result") == "duplicate":
            return json.dumps({
                "result": "Duplicate skipped",
                "content_hash": content_hash,
                "existing_stored_at": check_result.get("created_at", ""),
                "message": "Content appears to be a duplicate (content_hash match). Skipped.",
            })

        if check_result.get("result") == "near_duplicate":
            # Update last_accessed / access_count of the near-duplicate instead of inserting
            near_id = check_result.get("id")
            update_script = (
                "import chromadb\n"
                "from datetime import datetime\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "col = client.get_collection(REPLACEME_WING)\n"
                "data = col.get(ids=[REPLACEME_ID], include=['metadatas'])\n"
                "meta = (data.get('metadatas') or [{}])[0] or {}\n"
                "meta['last_accessed'] = datetime.now().isoformat()\n"
                "meta['access_count'] = meta.get('access_count', 0) + 1\n"
                "col.update(ids=[REPLACEME_ID], metadatas=[meta])\n"
                "print('merged:' + REPLACEME_ID)\n"
            ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
                "REPLACEME_WING", repr(wing)
            ).replace("REPLACEME_ID", repr(near_id))
            up_code, up_stdout, up_stderr = _run_python(update_script)
            if up_code == 0 and up_stdout.strip().startswith("merged:"):
                return json.dumps({
                    "result": "Near-duplicate merged",
                    "merged_into_id": near_id,
                    "simhash_diff": check_result.get("diff", 0),
                    "message": "Content is very similar to an existing drawer. Updated last_accessed instead of inserting.",
                })
            # If update fails, fall through to normal insert

        # ── Store in ChromaDB ──────────────────────────────────────────────
        script = (
            "import chromadb, uuid, zlib\n"
            "from datetime import datetime\n"
            "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
            "try:\n"
            "    col = client.get_collection(REPLACEME_WING)\n"
            "except Exception:\n"
            "    col = client.create_collection(REPLACEME_WING)\n"
            "drawer_id = str(uuid.uuid4())\n"
            "now = datetime.now().isoformat()\n"
            "col.add(\n"
            "    ids=[drawer_id],\n"
            "    documents=[REPLACEME_DOC],\n"
            "    metadatas=[{\n"
            "        'wing': REPLACEME_WING_V,\n"
            "        'room': REPLACEME_ROOM_V,\n"
            "        'hall': REPLACEME_HALL_V,\n"
            "        'importance': REPLACEME_IMP,\n"
            "        'language': REPLACEME_LANG,\n"
            "        'source_file': 'hermes-tool',\n"
            "        'created_at': now,\n"
            "        'last_accessed': now,\n"
            "        'access_count': 0,\n"
            "        'content_hash': REPLACEME_HASH,\n"
            "        'l0': REPLACEME_L0_V,\n"
            "        'l1': REPLACEME_L1_V,\n"
            "        'original_length': REPLACEME_ORIG_LEN,\n"
            "    }]\n"
            ")\n"
            "print('ok:' + drawer_id)\n"
        ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
            "REPLACEME_WING", repr(wing)
        ).replace(
            "REPLACEME_DOC", repr(content[:8000])
        ).replace(
            "REPLACEME_WING_V", repr(wing)
        ).replace(
            "REPLACEME_ROOM_V", repr(room)
        ).replace(
            "REPLACEME_HALL_V", repr(hall)
        ).replace(
            "REPLACEME_IMP", str(importance)
        ).replace(
            "REPLACEME_LANG", repr(lang)
        ).replace(
            "REPLACEME_HASH", repr(content_hash)
        ).replace(
            "REPLACEME_L0_V", repr(l0[:50]) if l0 else repr("")
        ).replace(
            "REPLACEME_L1_V", repr(l1[:300]) if l1 else repr("")
        ).replace(
            "REPLACEME_ORIG_LEN", str(content_len)
        )
        code, stdout, stderr = _run_python(script, timeout=15)
        if code != 0 or not stdout.strip().startswith("ok:"):
            return tool_error("Add drawer failed: %s" % (stderr or stdout))
        drawer_id = stdout.strip().split(":", 1)[1]

        # ── Background entity extraction into KG ────────────────────────────
        # Fire-and-forget: extract entities from content and auto-populate KG
        self._extract_entities_for_drawer(
            drawer_id, wing, room, content
        )

        return json.dumps({
            "result": "Drawer stored",
            "drawer_id": drawer_id,
            "wing": wing,
            "room": room,
            "hall": hall,
            "importance": importance,
            "language": lang,
            "content_hash": content_hash,
            "l0": l0,
            "l1": l1[:100] + "..." if len(l1) > 100 else l1,
            "original_length": content_len,
        })

    def _handle_kg_query(self, args: Dict[str, Any]) -> str:
        entity = args.get("entity", "")
        if not entity:
            return tool_error("Missing required parameter: entity")

        # Query via system Python SQLite
        kg_path = _HERMES_KG
        if not Path(kg_path).exists():
            return tool_error("Knowledge graph not found")

        as_of = args.get("as_of")
        query = (
            "SELECT subject, predicate, object, valid_from, valid_to "
            "FROM triples WHERE subject=? AND (valid_to IS NULL OR valid_to='')"
        )
        params = [entity]
        if as_of:
            query += " AND (valid_from IS NULL OR valid_from <= ?)"
            params.append(as_of)

        script = (
            "import sqlite3,json; conn=sqlite3.connect(REPLACEME_KG); conn.execute('PRAGMA journal_mode=WAL'); conn.execute('PRAGMA busy_timeout=5000'); "
            "cur=conn.cursor(); "
            "rows=cur.execute(REPLACEME_QUERY, REPLACEME_PARAMS).fetchall(); "
            "cols=[d[0] for d in cur.description]; "
            "print(json.dumps([dict(zip(cols,r)) for r in rows]))"
        ).replace("REPLACEME_KG", repr(kg_path)).replace(
            "REPLACEME_QUERY", repr(query)
        ).replace("REPLACEME_PARAMS", repr(params))
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error(f"KG query failed: {stderr}")
        try:
            results = json.loads(stdout)
            return json.dumps({"result": results, "entity": entity, "as_of": as_of})
        except Exception:
            return json.dumps({"result": stdout, "entity": entity, "as_of": as_of})

    def _handle_kg_add(self, args: Dict[str, Any]) -> str:
        subject = args.get("subject", "")
        predicate = args.get("predicate", "")
        obj = args.get("object", "")
        valid_from = args.get("valid_from", "")

        if not all([subject, predicate, obj]):
            return tool_error("Missing required parameters: subject, predicate, object")

        if not Path(_HERMES_KG).exists():
            return tool_error("Knowledge graph not found")

        from datetime import date
        today = date.today().isoformat()
        valid_from = valid_from or today

        # Use _store_triple_with_inverse for full contradiction detection + belief history
        success, msg = _store_triple_with_inverse(
            subject=subject,
            predicate=predicate,
            obj=obj,
            valid_from=valid_from,
            source="hermes-tool",
            confidence=1.0,
        )
        if not success:
            return tool_error("KG add failed: %s" % msg)

        return json.dumps({
            "result": msg,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "valid_from": valid_from,
        })

    def _handle_kg_invalidate(self, args: Dict[str, Any]) -> str:
        subject = args.get("subject", "")
        predicate = args.get("predicate", "")
        obj = args.get("object", "")
        ended = args.get("ended")

        if not all([subject, predicate, obj]):
            return tool_error("Missing required parameters: subject, predicate, object")

        kg_path = _HERMES_KG
        if not Path(kg_path).exists():
            return tool_error("Knowledge graph not found")

        from datetime import date
        ended = ended or date.today().isoformat()
        today = date.today().isoformat()

        script = (
            "import sqlite3, uuid\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "cur.execute(\n"
            "    \"SELECT object, valid_from, confidence, source FROM triples WHERE subject=? AND predicate=? AND object=? AND (valid_to IS NULL OR valid_to='')\",\n"
            "    REPLACEME_SUBJ_PRED_OBJ)\n"
            "row = cur.fetchone()\n"
            "old_value = row[0] if row else ''\n"
            "old_valid_from = row[1] if row else ''\n"
            "old_confidence = row[2] if row else 1.0\n"
            "old_source = row[3] if row else ''\n"
            "cur.execute(\n"
            "    \"UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND (valid_to IS NULL OR valid_to='')\",\n"
            "    REPLACEME_ENDED_SUBJ_PRED_OBJ)\n"
            "belief_id = str(uuid.uuid4())\n"
            "cur.execute(\n"
            "    \"INSERT INTO belief_history VALUES(?,?,?,?,?,?,?,?,?,?,?,?)\",\n"
            "    (belief_id, REPLACEME_SUBJECT, REPLACEME_PREDICATE, old_value, '', 'invalidated', REPLACEME_TODAY, old_source, old_confidence, '', old_valid_from, REPLACEME_ENDED))\n"
            "conn.commit()\n"
            "print('ok')\n"
            "conn.close()\n"
        ).replace("REPLACEME_KG", repr(kg_path)).replace(
            "REPLACEME_SUBJ_PRED_OBJ", repr((subject, predicate, obj))
        ).replace("REPLACEME_ENDED_SUBJ_PRED_OBJ", repr((ended, subject, predicate, obj))
        ).replace("REPLACEME_SUBJECT", repr(subject)).replace(
            "REPLACEME_PREDICATE", repr(predicate)
        ).replace("REPLACEME_TODAY", repr(today)).replace(
            "REPLACEME_ENDED", repr(ended)
        )
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error(f"KG invalidate failed: {stderr}")
        return json.dumps({
            "result": "Triple invalidated",
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "ended": ended,
            "belief_recorded": True,
        })

    def _handle_kg_stats(self, args: Dict[str, Any]) -> str:
        kg_path = _HERMES_KG
        if not Path(kg_path).exists():
            return tool_error("Knowledge graph not found")

        script = (
            "import sqlite3,json; conn=sqlite3.connect(REPLACEME_KG); conn.execute('PRAGMA journal_mode=WAL'); conn.execute('PRAGMA busy_timeout=5000'); "
            "cur=conn.cursor(); "
            "cur.execute(\"SELECT DISTINCT subject FROM triples WHERE valid_to IS NULL OR valid_to='' UNION SELECT DISTINCT object FROM triples WHERE valid_to IS NULL OR valid_to=''\"); "
            "e=len(cur.fetchall()); "
            "t=cur.execute(\"SELECT count(*) FROM triples WHERE valid_to IS NULL OR valid_to=''\").fetchone()[0]; "
            "print(json.dumps({'entities':e,'triples':t}))"
        ).replace("REPLACEME_KG", repr(kg_path))
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error(f"KG stats failed: {stderr}")
        try:
            stats = json.loads(stdout)
            return json.dumps({"result": stats})
        except Exception:
            return json.dumps({"result": stdout})

    def _handle_kg_add_typed(self, args: Dict[str, Any]) -> str:
        """Add a typed triple with automatic inverse relation."""
        subject = args.get("subject", "")
        predicate = args.get("predicate", "")
        obj = args.get("object", "")
        subject_type = args.get("subject_type", "")
        object_type = args.get("object_type", "")
        valid_from = args.get("valid_from", "")
        context = args.get("context", "")
        source = args.get("source", "hermes-tool")
        confidence = float(args.get("confidence", 1.0))

        if not all([subject, predicate, obj]):
            return tool_error("Missing required: subject, predicate, object")

        if not Path(_HERMES_KG).exists():
            return tool_error("Knowledge graph not found")

        from datetime import date
        valid_from = valid_from or date.today().isoformat()

        success, msg = _store_triple_with_inverse(
            subject=subject,
            predicate=predicate,
            obj=obj,
            valid_from=valid_from,
            context=context,
            source=source,
            confidence=confidence,
            subject_type=subject_type,
            object_type=object_type,
        )
        if success:
            return json.dumps({
                "result": msg,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "subject_type": subject_type or _guess_entity_type(subject),
                "object_type": object_type or _guess_entity_type(obj),
                "valid_from": valid_from,
                "context": context,
                "source": source,
                "confidence": confidence,
            })
        else:
            return tool_error(f"kg_add_typed failed: {msg}")

    def _handle_kg_query_decomposed(self, args: Dict[str, Any]) -> str:
        """
        Query planner — decomposes complex KG queries into typed sub-queries.
        """
        query = args.get("query", "")
        entity = args.get("entity", "")
        entity_type = args.get("entity_type", "")

        if not query and not entity:
            return tool_error("Missing required: query or entity")

        kg_path = _HERMES_KG
        if not Path(kg_path).exists():
            return tool_error("Knowledge graph not found")

        # Build query based on available filters
        params = []
        where_clauses = []
        if entity:
            where_clauses.append("(subject=? OR object=?)")
            params.extend([entity, entity])
        if entity_type:
            where_clauses.append("(subject_type=? OR object_type=?)")
            params.extend([entity_type, entity_type])

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        sql = f"SELECT subject, predicate, object, valid_from, valid_to, confidence, context, source, subject_type, object_type FROM triples WHERE {where_sql} AND (valid_to IS NULL OR valid_to='') ORDER BY valid_from DESC LIMIT 50"

        script = (
            "import sqlite3, json\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "rows = cur.execute(REPLACEME_QUERY, REPLACEME_PARAMS).fetchall()\n"
            "cols = ['subject', 'predicate', 'object', 'valid_from', 'valid_to', 'confidence', 'context', 'source', 'subject_type', 'object_type']\n"
            "result = [dict(zip(cols, r)) for r in rows]\n"
            "print(json.dumps(result))\n"
            "conn.close()\n"
        ).replace("REPLACEME_KG", repr(kg_path)).replace("REPLACEME_QUERY", repr(sql)).replace("REPLACEME_PARAMS", repr(params))
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error(f"Query decomposed failed: {stderr}")

        try:
            rows = json.loads(stdout)
        except Exception:
            return tool_error(f"Failed to parse results: {stdout}")

        # Analyze results and build structured response
        if not rows:
            return json.dumps({
                "result": [],
                "query": query,
                "entity": entity,
                "entity_type": entity_type,
                "analysis": "No matching triples found",
            })

        # Group by subject and analyze
        by_subject: Dict[str, List] = {}
        by_predicate: Dict[str, int] = {}
        by_type: Dict[str, List] = {}
        contexts: List[Dict] = []

        for row in rows:
            s = row.get("subject", "")
            p = row.get("predicate", "")
            o = row.get("object", "")
            st = row.get("subject_type", "concept")
            ot = row.get("object_type", "concept")

            if s not in by_subject:
                by_subject[s] = []
            by_subject[s].append({"predicate": p, "object": o, "context": row.get("context", "")})

            by_predicate[p] = by_predicate.get(p, 0) + 1

            if st not in by_type:
                by_type[st] = []
            by_type[st].append(s)

            if row.get("context"):
                contexts.append({"subject": s, "predicate": p, "context": row["context"]})

        # Inverse relations found
        inverse_relations = []
        for s, triples in by_subject.items():
            for t in triples:
                inv_pred = _get_inverse_predicate(t["predicate"])
                if inv_pred:
                    inverse_relations.append({
                        "from": {"entity": s, "predicate": t["predicate"]},
                        "to": {"entity": t["object"], "inverse_predicate": inv_pred},
                    })

        analysis = f"Found {len(rows)} triples, {len(by_subject)} entities, {len(inverse_relations)} inverse relations"

        return json.dumps({
            "result": rows,
            "summary": {
                "total_triples": len(rows),
                "entities": list(by_subject.keys()),
                "by_predicate": by_predicate,
                "by_type": {k: list(set(v)) for k, v in by_type.items()},
                "inverse_relations": inverse_relations[:10],
                "contexts": contexts[:5],
                "analysis": analysis,
            },
            "query": query,
            "entity": entity,
            "entity_type": entity_type,
            "source": "KG query planner (decomposed)",
        })

    def _handle_kg_belief_history(self, args: Dict[str, Any]) -> str:
        """
        Query the belief history table — track how facts/relationships evolved.
        Shows creation, updates, corrections, and invalidations of beliefs.
        """
        entity = args.get("entity", "")
        predicate = args.get("predicate", "")
        as_of = args.get("as_of", "")

        if not entity:
            return tool_error("Missing required: entity")

        kg_path = _HERMES_KG
        if not Path(kg_path).exists():
            return tool_error("Knowledge graph not found")

        # Build query for belief_history
        where = "entity=?"
        params = [entity]
        if predicate:
            where += " AND predicate=?"
            params.append(predicate)
        if as_of:
            where += " AND (valid_to IS NULL OR valid_to='' OR valid_to>=?)"
            params.append(as_of)

        script = (
            "import sqlite3, json\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "rows = cur.execute(\"SELECT belief_id, entity, predicate, old_value, new_value, change_type, changed_at, source, confidence, context, valid_from, valid_to FROM belief_history WHERE REPLACEME_WHERE ORDER BY changed_at DESC LIMIT 50\", REPLACEME_PARAMS).fetchall()\n"
            "cols = ['belief_id', 'entity', 'predicate', 'old_value', 'new_value', 'change_type', 'changed_at', 'source', 'confidence', 'context', 'valid_from', 'valid_to']\n"
            "result = [dict(zip(cols, r)) for r in rows]\n"
            "print(json.dumps(result))\n"
            "conn.close()\n"
        ).replace("REPLACEME_KG", repr(kg_path)).replace("REPLACEME_WHERE", where).replace("REPLACEME_PARAMS", repr(params))
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error(f"Belief history query failed: {stderr}")
        try:
            history = json.loads(stdout)
        except Exception:
            return tool_error(f"Failed to parse belief history: {stdout}")

        if not history:
            return json.dumps({
                "result": [],
                "entity": entity,
                "predicate": predicate or "all",
                "message": f"No belief history found for '{entity}'",
                "source": "KG belief tracking",
            })

        # Build timeline
        timeline = []
        for entry in history:
            change_type = entry.get("change_type", "")
            if change_type == "created":
                desc = f"✅ Created: '{entry.get('entity')}' {entry.get('predicate')} '{entry.get('new_value')}'"
            elif change_type == "updated":
                desc = f"🔄 Updated: '{entry.get('old_value')}' → '{entry.get('new_value')}' (predicate: {entry.get('predicate')})"
            elif change_type == "invalidated":
                desc = f"❌ Invalidated: '{entry.get('entity')}' {entry.get('predicate')} '{entry.get('old_value')}'"
            elif change_type == "corrected":
                desc = f"🔧 Corrected: '{entry.get('old_value')}' → '{entry.get('new_value')}'"
            else:
                desc = f"• {change_type}: '{entry.get('old_value')}' → '{entry.get('new_value')}'"
            timeline.append({
                "date": entry.get("changed_at", ""),
                "change_type": change_type,
                "description": desc,
                "source": entry.get("source", ""),
                "confidence": entry.get("confidence", ""),
            })

        return json.dumps({
            "result": history,
            "timeline": timeline,
            "summary": f"{len(history)} belief events for '{entity}'",
            "entity": entity,
            "predicate": predicate or "all",
            "source": "KG belief tracking",
        })

    def _handle_status(self, args: Dict[str, Any]) -> str:
        status = _cli_status()
        if "error" in status:
            return tool_error(status["error"])
        return json.dumps({"result": status})

    def _handle_list_wings(self, args: Dict[str, Any]) -> str:
        wings = _cli_list_wings()
        return json.dumps({"result": wings})

    # ------------------------------------------------------------------
    # Working Memory (L-WM) handlers — zero latency
    # ------------------------------------------------------------------

    def _handle_get_working_memory(self, args: Dict[str, Any]) -> str:
        n = min(int(args.get("n", 10)), 50)
        search_query = args.get("search", "")

        if search_query:
            turns = _working_memory.search(search_query, n=n)
            results = [{
                "speaker": t.speaker,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
                "topic": t.topic,
                "importance": t.importance,
            } for t in turns]
        else:
            turns = _working_memory.get_recent(n=n)
            results = [{
                "speaker": t.speaker,
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp,
                "topic": t.topic,
                "importance": t.importance,
            } for t in turns]

        return json.dumps({
            "result": results,
            "total_in_memory": len(_working_memory._turns),
            "mode": "search" if search_query else "recent",
        })

    def _handle_search_working_memory(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        n = min(int(args.get("n", 5)), 20)
        turns = _working_memory.search(query, n=n)
        results = [{
            "speaker": t.speaker,
            "role": t.role,
            "content": t.content,
            "timestamp": t.timestamp,
            "topic": t.topic,
            "importance": t.importance,
        } for t in results]
        return json.dumps({
            "result": results,
            "query": query,
        })

    def _handle_episodes(self, args: Dict[str, Any]) -> str:
        """
        Query L3 episodic memory — long-term ChromaDB-backed conversation logs.
        These persist across sessions unlike the in-memory L-WM working memory.
        """
        speaker = args.get("speaker")
        topic = args.get("topic")
        limit = min(int(args.get("limit", 20)), 100)

        raw = _episode_get_recent(speaker=speaker, topic=topic, limit=limit)
        try:
            data = json.loads(raw)
        except Exception:
            return tool_error("Failed to parse episodes: %s" % raw[:200])

        if "error" in data:
            return tool_error(data["error"])

        episodes = data.get("episodes", [])
        return json.dumps({
            "episodes": episodes,
            "count": len(episodes),
            "speaker_filter": speaker,
            "topic_filter": topic,
        })

    # ------------------------------------------------------------------
    # Cross-Palace reasoning handlers
    # ------------------------------------------------------------------

    def _handle_cross_palace_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        n = min(int(args.get("n_results", 5)), 15)

        # Run both searches in parallel using threads
        hermes_result = ""
        openclaw_result = ""
        errors = []

        def run_hermes():
            nonlocal hermes_result
            hermes_result = _cli_search(query=query, n_results=n)

        def run_openclaw():
            nonlocal openclaw_result
            openclaw_result = _openclaw_search(query=query, n_results=n)

        t1 = threading.Thread(target=run_hermes)
        t2 = threading.Thread(target=run_openclaw)
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        hermes_data = []
        if hermes_result and not hermes_result.startswith("Search error:"):
            try:
                # Parse the JSON result
                parsed = json.loads(hermes_result)
                hermes_data = parsed.get("result", [])
            except Exception:
                hermes_data = [{"raw": hermes_result}]

        openclaw_data = []
        if openclaw_result and not openclaw_result.startswith("OpenClaw search error:"):
            try:
                parsed = json.loads(openclaw_result)
                openclaw_data = parsed.get("result", [])
            except Exception:
                openclaw_data = [{"raw": openclaw_result}]

        return json.dumps({
            "result": {
                "hermes": hermes_data,
                "openclaw": openclaw_data,
            },
            "query": query,
            "source": "cross-palace (Hermes + OpenClaw)",
        })

    # ------------------------------------------------------------------
    # L7: Proactive Prediction Retrieval
    # ------------------------------------------------------------------

    def _handle_proactive_predict(self, args: Dict[str, Any]) -> str:
        """
        Predict relevant memories for a topic/context BEFORE the user asks.
        Searches KG + ChromaDB + episodes in depth-appropriate layers.
        Returns structured predictions: entities, facts, decisions, procedures.
        """
        topic = args.get("topic", "")
        context = args.get("context", "")
        depth = args.get("depth", "medium")

        if not topic:
            return tool_error("Missing required parameter: topic")

        full_query = (topic + " " + context).strip()
        lang = _detect_language(full_query)

        # ── Step 1: KG prediction ──────────────────────────────────────────
        # Extract keywords from topic for KG entity search
        keywords = re.findall(r'[a-zA-Z0-9]+', topic.lower())
        chinese_chars = [c for c in topic if '\u4e00' <= c <= '\u9fff']

        # Search KG for entities whose name contains any keyword
        entity_matches = []
        if keywords or chinese_chars:
            kg_path = _HERMES_KG
            if Path(kg_path).exists():
                cond_parts = []
                params = []
                for kw in keywords:
                    cond_parts.append("(subject LIKE ? OR object LIKE ?)")
                    params.extend([f"%{kw}%", f"%{kw}%"])
                for ch in chinese_chars:
                    cond_parts.append("(subject LIKE ? OR object LIKE ?)")
                    params.extend([f"%{ch}%", f"%{ch}%"])

                where_clause = " OR ".join(cond_parts)
                # Escape % in where_clause to avoid outer %-format conflict
                where_clause_esc = where_clause.replace("%", "%%")
                # Get active triples (valid_to IS NULL OR valid_to='')
                params_tuple = tuple(params)
                script = (
                    "import sqlite3, json\n"
                    "conn = sqlite3.connect('REPLACEME_KG')\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                    "cur = conn.cursor()\n"
                    "cur.execute(\"SELECT subject, predicate, object, subject_type, object_type, confidence, valid_from, context FROM triples WHERE (REPLACEME_WHERE) AND (valid_to IS NULL OR valid_to='') LIMIT 30\", REPLACEME_PARAMS)\n"
                    "rows = cur.fetchall()\n"
                    "conn.close()\n"
                    "results = [dict(subject=r[0], predicate=r[1], object=r[2], subject_type=r[3] or 'unknown', object_type=r[4] or 'unknown', confidence=r[5], valid_from=r[6], context=r[7] or '') for r in rows]\n"
                    "print(json.dumps(results))"
                )
                script = script.replace("REPLACEME_KG", kg_path).replace("REPLACEME_WHERE", where_clause_esc).replace("REPLACEME_PARAMS", repr(params_tuple))
                code, stdout, stderr = _run_python(script)
                if code == 0 and stdout.strip():
                    try:
                        entity_matches = json.loads(stdout.strip())
                    except Exception:
                        pass

        # ── Step 2: Build predicted entity set ─────────────────────────────
        predicted_entities = {}
        for m in entity_matches:
            s = m.get("subject", "")
            o = m.get("object", "")
            if s:
                predicted_entities[s] = {"type": m.get("subject_type", "unknown"), "role": "subject"}
            if o:
                predicted_entities[o] = {"type": m.get("object_type", "unknown"), "role": "object"}

        # ── Step 3: ChromaDB drawer search (medium + deep) ──────────────────
        drawer_results = []
        if depth in ("medium", "deep"):
            full_query_text = (topic + " " + context).strip()[:500]
            drawer_script = (
                "import chromadb, json\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "all_results = []\n"
                "for col in client.list_collections():\n"
                "    try:\n"
                "        c = client.get_collection((col.name if hasattr(col, 'name') else col['name']))\n"
                "        r = c.query(query_texts=[REPLACEME_QUERY], n_results=3)\n"
                "        for i, doc in enumerate(r.get('documents', [[]])[0]):\n"
                "            meta = (r.get('metadatas', [[{}]])[0] or [{}])[i] or {}\n"
                "            all_results.append({'document': doc[:300], 'collection': (col.name if hasattr(col, 'name') else col['name']), 'metadata': meta})\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(all_results[:15]))"
            ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
                "REPLACEME_QUERY", repr(full_query_text)
            )
            code, stdout, stderr = _run_python(drawer_script)
            if code == 0 and stdout.strip():
                try:
                    drawer_results = json.loads(stdout.strip())
                except Exception:
                    pass

        # ── Step 4: Episode search (deep only) ─────────────────────────────
        episode_results = []
        if depth == "deep":
            episode_script = (
                "import chromadb, json\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "try:\n"
                "    col = client.get_collection('episodes')\n"
                "    r = col.query(query_texts=[REPLACEME_QUERY], n_results=5)\n"
                "    results = []\n"
                "    for i, doc in enumerate(r.get('documents', [[]])[0]):\n"
                "        meta = (r.get('metadatas', [[{}]])[0] or [{}])[i] or {}\n"
                "        results.append({'document': doc[:200], 'speaker': meta.get('speaker', ''), 'topic': meta.get('topic', ''), 'timestamp': meta.get('timestamp', ''), 'language': meta.get('language', 'unknown')})\n"
                "    print(json.dumps(results))\n"
                "except Exception:\n"
                "    print(json.dumps([]))"
            ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
                "REPLACEME_QUERY", repr(topic[:200])
            )
            code, stdout, stderr = _run_python(episode_script)
            if code == 0 and stdout.strip():
                try:
                    episode_results = json.loads(stdout.strip())
                except Exception:
                    pass

        # ── Step 5: Categorize predictions ────────────────────────────────
        # Group KG triples by predicate type
        facts = []
        decisions = []
        relationships = []
        procedures = []

        decision_predicates = {"decided", "concluded", "agreed", "concluded_that", "decided_that"}
        procedure_predicates = {"uses", "depends_on", "creates", "implements", "calls", "runs"}

        for m in entity_matches:
            pred = m.get("predicate", "")
            entry = {
                "subject": m.get("subject", ""),
                "predicate": pred,
                "object": m.get("object", ""),
                "confidence": m.get("confidence", 1.0),
                "context": m.get("context", ""),
            }
            if pred in decision_predicates:
                decisions.append(entry)
            elif pred in procedure_predicates:
                procedures.append(entry)
            else:
                relationships.append(entry)

        # Deduplicate drawer results by document content
        seen_docs = set()
        unique_drawers = []
        for d in drawer_results:
            doc_key = d.get("document", "")[:100]
            if doc_key and doc_key not in seen_docs:
                seen_docs.add(doc_key)
                unique_drawers.append(d)

        # ── Step 6: Generate prediction summary ────────────────────────────
        entity_summary = ", ".join(list(predicted_entities.keys())[:8]) if predicted_entities else "none"
        confidence_avg = sum(m.get("confidence", 1.0) for m in entity_matches) / max(len(entity_matches), 1)

        prediction_summary = (
            f"Predicted {len(predicted_entities)} related entities from {len(entity_matches)} KG triples. "
            f"Top entities: {entity_summary}. "
            f"Found {len(decisions)} decisions, {len(relationships)} relationships, {len(procedures)} procedures, "
            f"{len(unique_drawers)} relevant drawer memories."
        )

        return json.dumps({
            "topic": topic,
            "context": context,
            "depth": depth,
            "language_detected": lang,
            "prediction_summary": prediction_summary,
            "confidence": round(confidence_avg, 3),
            "predicted_entities": list(predicted_entities.keys())[:10],
            "kg_triples_count": len(entity_matches),
            "predictions": {
                "decisions": decisions[:5],
                "relationships": relationships[:8],
                "procedures": procedures[:5],
                "memories": unique_drawers[:8],
                "episodes": episode_results[:5] if depth == "deep" else [],
            },
            "sources_searched": (
                ["knowledge_graph"] +
                (["chromaDB_drawers"] if depth in ("medium", "deep") else []) +
                (["chromaDB_episodes"] if depth == "deep" else [])
            ),
        })

    # ------------------------------------------------------------------
    # L8: Memory Reflection — self-healing system audit
    # ------------------------------------------------------------------

    def _handle_memory_reflection(self, args: Dict[str, Any]) -> str:
        """
        Self-healing audit: orphaned entities, dangling relations,
        duplicate drawers, stale memories, KG inconsistencies.
        """
        fix = bool(args.get("fix", False))
        kg_path = _HERMES_KG
        palace_path = _HERMES_PALACE

        issues = []
        fixes_applied = []

        # ── Issue 1: KG dangling provenance triples
        #
        #    The entities table is always empty in this plugin (entity names are
        #    stored directly in triples, not via an ID indirect layer), so we CANNOT
        #    use "object not in entities" as the dangling criterion — that would
        #    flag EVERY triple as dangling.
        #
        #    The only truly dangling KG triples are `mentioned_in <session_uuid>`
        #    where the session UUID no longer exists in ChromaDB (session was GC'd).
        #    These are provenance records pointing to dead conversation logs.
        #
        dangling_rels = []
        if Path(kg_path).exists() and Path(palace_path).exists():
            script = (
                "import sqlite3, json, re, chromadb\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"SELECT subject, predicate, object FROM triples WHERE (valid_to IS NULL OR valid_to='')\")\n"
                "triples = cur.fetchall()\n"
                "conn.close()\n"
                "# Collect all ChromaDB document IDs\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "chromadb_ids = set()\n"
                "for col_info in client.list_collections():\n"
                "    try:\n"
                "        col = client.get_collection((col_info.name if hasattr(col_info, 'name') else col_info['name']))\n"
                "        data = col.get(limit=100000, include=[])\n"
                "        chromadb_ids.update(data.get('ids', []))\n"
                "    except Exception:\n"
                "        pass\n"
                "# Session UUID pattern: 8-4-4-4-12 hex groups\n"
                "uuid_pat = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)\n"
                "dangling = []\n"
                "for s, p, o in triples:\n"
                "    # mentioned_in <session_uuid> where session is gone = dangling provenance\n"
                "    # detected_as_entity drawer:<uuid> where drawer is gone = dangling extraction\n"
                "    if p in ('mentioned_in', 'detected_as_entity') and uuid_pat.match(str(o).replace('drawer:', '')) and o.replace('drawer:', '') not in chromadb_ids:\n"
                "        dangling.append((s, p, o))\n"
                "print(json.dumps(dangling[:50]))\n"
            ).replace("REPLACEME_KG", repr(kg_path)).replace(
                "REPLACEME_PALACE", repr(palace_path)
            )
            code, stdout, stderr = _run_python(script)
            if code == 0 and stdout.strip():
                try:
                    dangling_rels = json.loads(stdout.strip())
                except Exception:
                    pass

        if dangling_rels:
            issues.append({
                "type": "dangling_relations",
                "severity": "medium",
                "count": len(dangling_rels),
                "relations": dangling_rels,
                "description": "%d triples point to non-existent objects" % len(dangling_rels),
                "suggestion": "Invalidate these triples or add missing entities",
            })
            if fix:
                fix_script = (
                    "import sqlite3\n"
                    "from datetime import date\n"
                    "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                    "cur = conn.cursor()\n"
                    "today = date.today().isoformat()\n"
                    "for s, p, o in REPLACEME_DANGLING:\n"
                    "    cur.execute(\"UPDATE triples SET valid_to=? WHERE subject=? AND predicate=? AND object=? AND (valid_to IS NULL OR valid_to='')\", (today, s, p, o))\n"
                    "conn.commit()\n"
                    "conn.close()\n"
                    "print('fixed:' + str(len(REPLACEME_DANGLING)))\n"
                    "print('date:' + date.today().isoformat())\n"
                ).replace("REPLACEME_KG", repr(kg_path)).replace(
                    "REPLACEME_DANGLING", repr(dangling_rels)
                )
                code, stdout, stderr = _run_python(fix_script)
                if code == 0:
                    fixes_applied.append("Invalidated %d dangling relations" % len(dangling_rels))

        # ── Issue 3: Duplicate drawers (high simhash similarity) ─────────
        duplicate_groups = []
        if Path(palace_path).exists():
            script = (
                "import chromadb, json, hashlib\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "def simhash(s):\n"
                "    sig = [0] * 64\n"
                "    words = s.lower().split()\n"
                "    for w in words:\n"
                "        h = hashlib.md5(w.encode()).digest()\n"
                "        for i in range(64):\n"
                "            sig[i] += 1 if h[i % 16] > 127 else -1\n"
                "    return sum((1 << i) if s > 0 else 0 for i, s in enumerate(sig))\n"
                "dup_groups = []\n"
                "for col_info in client.list_collections():\n"
                "    col_name = (col_info.name if hasattr(col_info, 'name') else col_info['name'])\n"
                "    if col_name == 'episodes':\n"
                "        continue  # episodes often legitimately repeat (greetings, system prompts)\n"
                "    try:\n"
                "        col = client.get_collection(col_name)\n"
                "        data = col.get(limit=1000, include=['documents', 'metadatas'])\n"
                "        docs = []\n"
                "        for i, doc in enumerate(data.get('documents', []) or []):\n"
                "            meta = (data.get('metadatas') or [{}])[i] or {}\n"
                "            docs.append({'doc': doc[:500], 'col': col_name, 'id': data['ids'][i] if 'ids' in data else str(i), 'meta': meta})\n"
                "        used = set()\n"
                "        for i in range(len(docs)):\n"
                "            if docs[i]['id'] in used:\n"
                "                continue\n"
                "            group = [docs[i]]\n"
                "            h1 = simhash(docs[i]['doc'])\n"
                "            for j in range(i + 1, len(docs)):\n"
                "                if docs[j]['id'] in used:\n"
                "                    continue\n"
                "                h2 = simhash(docs[j]['doc'])\n"
                "                diff = bin(h1 ^ h2).count('1')\n"
                "                if diff <= 5:\n"
                "                    group.append(docs[j])\n"
                "                    used.add(docs[j]['id'])\n"
                "            if len(group) > 1:\n"
                "                used.add(group[0]['id'])\n"
                "                dup_groups.append([{'doc': g['doc'][:100], 'col': g['col'], 'id': g['id']} for g in group])\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(dup_groups[:50]))\n"
            ).replace("REPLACEME_PALACE", repr(palace_path))
            code, stdout, stderr = _run_python(script)
            if code == 0 and stdout.strip():
                try:
                    duplicate_groups = json.loads(stdout.strip())
                except Exception:
                    pass

        if duplicate_groups:
            issues.append({
                "type": "duplicate_drawers",
                "severity": "low",
                "count": len(duplicate_groups),
                "groups": duplicate_groups,
                "description": "%d groups of near-identical drawers found (simhash diff <= 5)" % len(duplicate_groups),
                "suggestion": "Merge or delete duplicate drawers manually",
            })
            if fix:
                fix_script = (
                    "import chromadb, sqlite3, hashlib, json\n"
                    "from datetime import datetime\n"
                    "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                    "def simhash(s):\n"
                    "    sig = [0] * 64\n"
                    "    words = s.lower().split()\n"
                    "    for w in words:\n"
                    "        h = hashlib.md5(w.encode()).digest()\n"
                    "        for i in range(64):\n"
                    "            sig[i] += 1 if h[i % 16] > 127 else -1\n"
                    "    return sum((1 << i) if s > 0 else 0 for i, s in enumerate(sig))\n"
                    "dup_groups = []\n"
                    "for col_info in client.list_collections():\n"
                    "    col_name = (col_info.name if hasattr(col_info, 'name') else col_info['name'])\n"
                    "    if col_name == 'episodes':\n"
                    "        continue\n"
                    "    try:\n"
                    "        col = client.get_collection(col_name)\n"
                    "        data = col.get(limit=1000, include=['documents', 'metadatas'])\n"
                    "        docs = []\n"
                    "        for i, doc in enumerate(data.get('documents', []) or []):\n"
                    "            meta = (data.get('metadatas') or [{}])[i] or {}\n"
                    "            docs.append({'doc': doc[:500], 'col': col_name, 'id': data['ids'][i], 'meta': meta})\n"
                    "        used = set()\n"
                    "        for i in range(len(docs)):\n"
                    "            if docs[i]['id'] in used:\n"
                    "                continue\n"
                    "            group = [docs[i]]\n"
                    "            h1 = simhash(docs[i]['doc'])\n"
                    "            for j in range(i + 1, len(docs)):\n"
                    "                if docs[j]['id'] in used:\n"
                    "                    continue\n"
                    "                h2 = simhash(docs[j]['doc'])\n"
                    "                diff = bin(h1 ^ h2).count('1')\n"
                    "                if diff <= 5:\n"
                    "                    group.append(docs[j])\n"
                    "                    used.add(docs[j]['id'])\n"
                    "            if len(group) > 1:\n"
                    "                used.add(group[0]['id'])\n"
                    "                dup_groups.append(group)\n"
                    "    except Exception:\n"
                    "        pass\n"
                    "deleted_total = 0\n"
                    "kg_updated = 0\n"
                    "conn = sqlite3.connect(REPLACEME_KG)\n"
                    "conn.execute('PRAGMA journal_mode=WAL')\n"
                    "conn.execute('PRAGMA busy_timeout=5000')\n"
                    "cur = conn.cursor()\n"
                    "for group in dup_groups:\n"
                    "    group.sort(key=lambda x: x['meta'].get('created_at', '9999'))\n"
                    "    keeper = group[0]\n"
                    "    to_delete = group[1:]\n"
                    "    col = client.get_collection(keeper['col'])\n"
                    "    for d in to_delete:\n"
                    "        try:\n"
                    "            col.delete(ids=[d['id']])\n"
                    "            deleted_total += 1\n"
                    "        except Exception:\n"
                    "            pass\n"
                    "        cur.execute(\"UPDATE triples SET object=? WHERE predicate='mentioned_in' AND object=? AND (valid_to IS NULL OR valid_to='')\", (keeper['id'], d['id']))\n"
                    "        kg_updated += cur.rowcount\n"
                    "conn.commit()\n"
                    "conn.close()\n"
                    "print(json.dumps({'deleted': deleted_total, 'kg_updated': kg_updated}))\n"
                ).replace("REPLACEME_PALACE", repr(palace_path)).replace("REPLACEME_KG", repr(kg_path))
                code, stdout, stderr = _run_python(fix_script)
                if code == 0 and stdout.strip():
                    try:
                        fix_result = json.loads(stdout.strip())
                        fixes_applied.append("Merged %d duplicate drawers, updated %d KG provenance triples" % (fix_result.get("deleted", 0), fix_result.get("kg_updated", 0)))
                    except Exception:
                        pass


        # ── Issue 4: Stale high-importance drawers ─────────────────────
        stale_high = []
        if Path(palace_path).exists():
            script = (
                "import chromadb, json\n"
                "from datetime import datetime, timedelta\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "threshold = (datetime.now() - timedelta(days=90)).isoformat()\n"
                "stale = []\n"
                "for col_info in client.list_collections():\n"
                "    try:\n"
                "        col = client.get_collection((col_info.name if hasattr(col_info, 'name') else col_info['name']))\n"
                "        data = col.get(limit=1000, include=['metadatas'])\n"
                "        for i, meta in enumerate((data.get('metadatas') or []) or []):\n"
                "            if not meta:\n"
                "                continue\n"
                "            last_access = meta.get('last_accessed', '')\n"
                "            importance = meta.get('importance', 0.5)\n"
                "            if last_access and last_access < threshold and importance >= 0.7:\n"
                "                stale.append({'id': data['ids'][i] if 'ids' in data else str(i), 'last_accessed': last_access, 'importance': importance, 'col': (col_info.name if hasattr(col_info, 'name') else col_info['name'])})\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(stale[:20]))\n"
            ).replace("REPLACEME_PALACE", repr(palace_path))
            code, stdout, stderr = _run_python(script)
            if code == 0 and stdout.strip():
                try:
                    stale_high = json.loads(stdout.strip())
                except Exception:
                    pass

        if stale_high:
            issues.append({
                "type": "stale_high_importance",
                "severity": "info",
                "count": len(stale_high),
                "drawers": stale_high,
                "description": "%d high-importance drawers (>0.7) untouched for 90+ days" % len(stale_high),
                "suggestion": "Review or reduce importance score",
            })

        # ── Issue 5: KG inverse relation mismatches ───────────────────
        inverse_mismatches = []
        if Path(kg_path).exists():
            script = (
                "import sqlite3, json\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"SELECT id, subject, predicate, object, inverse_predicate FROM triples WHERE (valid_to IS NULL OR valid_to='') AND inverse_predicate != '' LIMIT 200\")\n"
                "rows = cur.fetchall()\n"
                "mismatches = []\n"
                "for r in rows:\n"
                "    inv_pred = r[4]\n"
                "    expected_inv = {'knows': 'known_by', 'known_by': 'knows', 'uses': 'used_by', 'used_by': 'uses', 'works_with': 'works_with', 'friends_with': 'friends_with'}.get(r[2])\n"
                "    if expected_inv and inv_pred != expected_inv and inv_pred != r[2]:\n"
                "        mismatches.append({'id': r[0], 'triple': (r[1], r[2], r[3]), 'inverse_pred': inv_pred, 'expected': expected_inv})\n"
                "conn.close()\n"
                "print(json.dumps(mismatches[:20]))\n"
            ).replace("REPLACEME_KG", repr(kg_path))
            code, stdout, stderr = _run_python(script)
            if code == 0 and stdout.strip():
                try:
                    inverse_mismatches = json.loads(stdout.strip())
                except Exception:
                    pass

        if inverse_mismatches:
            issues.append({
                "type": "inverse_relation_mismatch",
                "severity": "medium",
                "count": len(inverse_mismatches),
                "mismatches": inverse_mismatches,
                "description": "%d triples have incorrect inverse_predicate values" % len(inverse_mismatches),
                "suggestion": "Run consolidation to fix inverse relation values",
            })

        total_issues = len(issues)

        return json.dumps({
            "total_issues": total_issues,
            "issues": issues,
            "fixes_applied": fixes_applied if fix else [],
            "fix_requested": fix,
            "summary": "%d issue types found. %s" % (
                total_issues,
                "Fixes applied: " + "; ".join(fixes_applied) if fix and fixes_applied else "No fixes applied (use fix=true to auto-apply)" if fix else "Dry run — no changes made",
            ),
        })

    def _handle_record_correction(self, args: Dict[str, Any]) -> str:
        """
        Record a user correction — e.g. when the user says '这不是欧拉好猫'.
        Stores: (subject, made_error, wrong_value) + (subject, corrected_to, correct_value)
        in KG, plus a belief_history entry with change_type='corrected'.
        """
        import uuid as _uuid

        subject = args.get("subject", "Hermes")
        wrong_value = args.get("wrong_value", "")
        correct_value = args.get("correct_value", "")
        context = args.get("context", "")

        if not wrong_value or not correct_value:
            return tool_error("Both wrong_value and correct_value are required")

        now = datetime.now().isoformat()

        # Store the error fact
        _store_triple_with_inverse(
            subject=subject,
            predicate="made_error",
            obj=wrong_value,
            valid_from=now,
            context=context or "user correction",
            source="user-feedback",
            confidence=1.0,
            subject_type="agent",
            object_type="fact",
        )

        # Store the correction
        _store_triple_with_inverse(
            subject=subject,
            predicate="corrected_to",
            obj=correct_value,
            valid_from=now,
            context=context or "user correction",
            source="user-feedback",
            confidence=1.0,
            subject_type="agent",
            object_type="fact",
        )

        # Store belief_history entry
        belief_id = str(_uuid.uuid4())
        script = (
            "import sqlite3, json\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "cur.execute("
            "'INSERT INTO belief_history "
            "(belief_id, entity, predicate, old_value, new_value, change_type, "
            "changed_at, source, confidence, context, valid_from, valid_to) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', "
            "(REPLACEME_BID, REPLACEME_SUBJECT, 'corrected', "
            "REPLACEME_WRONG, REPLACEME_CORRECT, 'corrected', "
            "REPLACEME_NOW, 'user-feedback', 1.0, "
            "REPLACEME_CONTEXT, REPLACEME_NOW, ''))\n"
            "conn.commit()\n"
            "conn.close()\n"
            "print('ok')\n"
        ).replace("REPLACEME_KG", repr(_HERMES_KG)).replace(
            "REPLACEME_BID", repr(belief_id)
        ).replace("REPLACEME_SUBJECT", repr(subject)
        ).replace("REPLACEME_WRONG", repr(wrong_value)
        ).replace("REPLACEME_CORRECT", repr(correct_value)
        ).replace("REPLACEME_NOW", repr(now)
        ).replace("REPLACEME_CONTEXT", repr(context or "user correction")
        )

        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error("Record correction failed: %s" % (stderr or stdout))

        return json.dumps({
            "result": "Correction recorded",
            "subject": subject,
            "wrong_value": wrong_value,
            "correct_value": correct_value,
            "belief_id": belief_id,
        })

    # ------------------------------------------------------------------
    # Memory Consolidation — importance decay + L2 eviction
    # ------------------------------------------------------------------

    def _handle_consolidate(self, args: Dict[str, Any]) -> str:
        """
        Run memory consolidation: importance decay, L2 eviction, orphan cleanup.
        High-importance (>0.7) memories are protected.
        """
        dry_run = bool(args.get("dry_run", False))
        palace_path = _HERMES_PALACE
        kg_path = _HERMES_KG

        changes = []

        if not Path(palace_path).exists():
            return json.dumps({"error": "Palace not found", "changes": []})

        # Decay importance for 30+ day unused, evict L2 for 90+ day unused
        script = (
            "import chromadb, json\n"
            "from datetime import datetime, timedelta\n"
            "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
            "now = datetime.now()\n"
            "threshold_30 = (now - timedelta(days=30)).isoformat()\n"
            "threshold_90 = (now - timedelta(days=90)).isoformat()\n"
            "total_updated = 0\n"
            "total_l2_evicted = 0\n"
            "details = []\n"
            "for col_info in client.list_collections():\n"
            "    try:\n"
            "        col = client.get_collection((col_info.name if hasattr(col_info, 'name') else col_info['name']))\n"
            "        data = col.get(limit=1000, include=['documents', 'metadatas', 'ids'])\n"
            "        ids_to_upd = []\n"
            "        metas_to_upd = []\n"
            "        docs_to_upd = []\n"
            "        for i, meta in enumerate((data.get('metadatas') or []) or []):\n"
            "            if not meta:\n"
            "                continue\n"
            "            last_access = meta.get('last_accessed', '')\n"
            "            importance = meta.get('importance', 0.5)\n"
            "            doc = (data.get('documents') or [''])[i] or ''\n"
            "            has_l2 = doc and len(doc) > 0\n"
            "            changed = False\n"
            "            if last_access and last_access < threshold_30 and importance > 0.1:\n"
            "                if importance > 0.7:\n"
            "                    pass  # protected\n"
            "                else:\n"
            "                    new_imp = max(0.05, importance - 0.2)\n"
            "                    meta['importance'] = new_imp\n"
            "                    changed = True\n"
            "            if last_access and last_access < threshold_90 and has_l2:\n"
            "                if importance > 0.7:\n"
            "                    pass  # protected\n"
            "                else:\n"
            "                    # evict L2 — keep L0/L1, clear raw content\n"
            "                    doc = doc[:50] if doc else doc  # keep very short stub\n"
            "                    meta['l2_evicted'] = True\n"
            "                    meta['l2_evicted_at'] = now.isoformat()\n"
            "                    total_l2_evicted += 1\n"
            "                    changed = True\n"
            "            if changed:\n"
            "                ids_to_upd.append(data['ids'][i] if 'ids' in data else str(i))\n"
            "                metas_to_upd.append(meta)\n"
            "                docs_to_upd.append(doc)\n"
            "                details.append({'id': data['ids'][i] if 'ids' in data else str(i), 'new_imp': meta.get('importance'), 'l2_evicted': meta.get('l2_evicted', False)})\n"
            "                total_updated += 1\n"
            "        if ids_to_upd and not REPLACEME_DRYRUN:\n"
            "            for idx in range(len(ids_to_upd)):\n"
            "                try:\n"
            "                    col.update(ids=[ids_to_upd[idx]], metadatas=[metas_to_upd[idx]], documents=[docs_to_upd[idx]])\n"
            "                except Exception:\n"
            "                    pass\n"
            "    except Exception:\n"
            "        pass\n"
            "print(json.dumps({'total_updated': total_updated, 'total_l2_evicted': total_l2_evicted, 'details': details[:30]}))\n"
        ).replace("REPLACEME_PALACE", repr(palace_path)).replace(
            "REPLACEME_DRYRUN", "True" if dry_run else "False"
        )
        code, stdout, stderr = _run_python(script, timeout=60)
        if code == 0 and stdout.strip():
            try:
                result = json.loads(stdout.strip())
                changes.append("Drawers updated: %d" % result.get("total_updated", 0))
                changes.append("L2 evicted: %d" % result.get("total_l2_evicted", 0))
                for d in result.get("details", [])[:5]:
                    changes.append("  - id=%(id)s: imp=%(new_imp).2f, l2_evicted=%(l2_evicted)s" % d)
            except Exception:
                pass
        else:
            changes.append("Error: %s" % stderr)

        # KG: vacuum orphaned triples
        if Path(kg_path).exists() and not dry_run:
            vacuum_script = (
                "import sqlite3\n"
                "from datetime import date, timedelta\n"
                "threshold = (date.today() - timedelta(days=365)).isoformat()\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"DELETE FROM triples WHERE valid_to IS NOT NULL AND valid_to != '' AND valid_to < ?\", (threshold,))\n"
                "deleted = cur.rowcount\n"
                "conn.commit()\n"
                "conn.close()\n"
                "print('vacuum_deleted:' + str(deleted))"
            ).replace("REPLACEME_KG", repr(kg_path))
            code, stdout, stderr = _run_python(vacuum_script)
            if code == 0:
                changes.append("Vacuumed old invalid triples")

        return json.dumps({
            "dry_run": dry_run,
            "changes": changes,
            "summary": "Consolidation %s. %s" % (
                "dry-run complete" if dry_run else "complete",
                " | ".join(changes) if changes else "No changes needed",
            ),
        })

    # ------------------------------------------------------------------
    # KG Alias Table — cross-palace entity resolution
    # ------------------------------------------------------------------

    def _ensure_alias_table(self) -> None:
        """Create alias table if it doesn't exist."""
        if not Path(_HERMES_KG).exists():
            return
        script = (
            "import sqlite3\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "cur.execute(\"CREATE TABLE IF NOT EXISTS entity_aliases (id TEXT PRIMARY KEY, entity TEXT NOT NULL, alias TEXT NOT NULL, entity_type TEXT DEFAULT '', palace TEXT DEFAULT 'shared', created_at TEXT, UNIQUE(entity, alias))\")\n"
            "conn.commit()\n"
            "conn.close()\n"
        ).replace("REPLACEME_KG", repr(_HERMES_KG))
        _run_python(script)

    def _handle_kg_alias_add(self, args: Dict[str, Any]) -> str:
        """Register an entity alias for cross-palace resolution."""
        entity = args.get("entity", "")
        alias = args.get("alias", "")
        entity_type = args.get("entity_type", "")
        palace = args.get("palace", "shared")

        if not entity or not alias:
            return tool_error("Missing required: entity and alias")

        self._ensure_alias_table()

        import uuid
        today = datetime.now().isoformat()
        script = (
            "import sqlite3\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "cur.execute(\"INSERT OR IGNORE INTO entity_aliases VALUES(?,?,?,?,?,?)\", (REPLACEME_ID, REPLACEME_ENTITY, REPLACEME_ALIAS, REPLACEME_TYPE, REPLACEME_PALACE, REPLACEME_TODAY))\n"
            "rows = cur.execute(\"SELECT changes()\").fetchone()[0]\n"
            "conn.commit()\n"
            "conn.close()\n"
            "print('ok' if rows > 0 else 'duplicate')\n"
        ).replace("REPLACEME_KG", repr(_HERMES_KG)).replace(
            "REPLACEME_ID", repr(str(uuid.uuid4()))
        ).replace("REPLACEME_ENTITY", repr(entity)).replace(
            "REPLACEME_ALIAS", repr(alias)
        ).replace("REPLACEME_TYPE", repr(entity_type)).replace(
            "REPLACEME_PALACE", repr(palace)
        ).replace("REPLACEME_TODAY", repr(today))
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error("Alias add failed: %s" % stderr)
        return json.dumps({
            "result": "Alias added" if stdout.strip() == "ok" else "Alias already exists",
            "entity": entity,
            "alias": alias,
            "entity_type": entity_type,
            "palace": palace,
        })

    def _handle_kg_alias_resolve(self, args: Dict[str, Any]) -> str:
        """Resolve an entity to all known aliases across both palaces."""
        entity = args.get("entity", "")
        if not entity:
            return tool_error("Missing required: entity")

        self._ensure_alias_table()

        script = (
            "import sqlite3, json\n"
            "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
            "cur = conn.cursor()\n"
            "cur.execute(\"SELECT entity, alias, entity_type, palace FROM entity_aliases WHERE entity=? OR alias=? ORDER BY palace\", (REPLACEME_ENTITY, REPLACEME_ENTITY))\n"
            "rows = cur.fetchall()\n"
            "by_palace = {'hermes': [], 'openclaw': [], 'shared': []}\n"
            "canonical = REPLACEME_ENTITY\n"
            "for r in rows:\n"
            "    if r[0] == REPLACEME_ENTITY:\n"
            "        canonical = r[0]\n"
            "        by_palace.get(r[3], by_palace['shared']).append(r[1])\n"
            "    elif r[1] == REPLACEME_ENTITY:\n"
            "        canonical = r[0]\n"
            "conn.close()\n"
            "print(json.dumps({'canonical': canonical, 'by_palace': by_palace, 'total_aliases': sum(len(v) for v in by_palace.values())}))\n"
        ).replace("REPLACEME_KG", repr(_HERMES_KG)).replace(
            "REPLACEME_ENTITY", repr(entity)
        )
        code, stdout, stderr = _run_python(script)
        if code != 0:
            return tool_error("Alias resolve failed: %s" % stderr)
        try:
            result = json.loads(stdout.strip())
            return json.dumps({
                "entity": entity,
                "canonical": result.get("canonical", entity),
                "aliases": result.get("by_palace", {}),
                "total_aliases": result.get("total_aliases", 0),
            })
        except Exception:
            return json.dumps({"entity": entity, "canonical": entity, "aliases": {}, "total_aliases": 0})

    # ------------------------------------------------------------------
    # Import / Export
    # ------------------------------------------------------------------

    def _handle_export(self, args: Dict[str, Any]) -> str:
        """Export all memory to JSON."""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(Path.home() / ".mempalace_hermes" / ("export_" + timestamp + ".json"))
        filepath = args.get("filepath", default_path)
        include_chromadb = bool(args.get("include_chromadb", True))

        export_data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "palace": "hermes",
            "kg_triples": [],
            "belief_history": [],
            "entity_aliases": [],
            "chromadb_collections": {},
        }

        # Export KG
        if Path(_HERMES_KG).exists():
            script = (
                "import sqlite3, json\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"SELECT * FROM triples\")\n"
                "cols = [d[0] for d in cur.description]\n"
                "triples = [dict(zip(cols, r)) for r in cur.fetchall()]\n"
                "cur.execute(\"SELECT * FROM belief_history\")\n"
                "bh_cols = [d[0] for d in cur.description]\n"
                "bh = [dict(zip(bh_cols, r)) for r in cur.fetchall()]\n"
                "try:\n"
                "    cur.execute(\"SELECT * FROM entity_aliases\")\n"
                "    ea_cols = [d[0] for d in cur.description]\n"
                "    ea = [dict(zip(ea_cols, r)) for r in cur.fetchall()]\n"
                "except Exception:\n"
                "    ea = []\n"
                "conn.close()\n"
                "print(json.dumps({'triples': triples, 'belief_history': bh, 'entity_aliases': ea}))\n"
            ).replace("REPLACEME_KG", repr(_HERMES_KG))
            code, stdout, stderr = _run_python(script)
            if code == 0 and stdout.strip():
                try:
                    kg_data = json.loads(stdout.strip())
                    export_data["kg_triples"] = kg_data.get("triples", [])
                    export_data["belief_history"] = kg_data.get("belief_history", [])
                    export_data["entity_aliases"] = kg_data.get("entity_aliases", [])
                except Exception:
                    pass

        # Export ChromaDB
        if include_chromadb and Path(_HERMES_PALACE).exists():
            script = (
                "import chromadb, json\n"
                "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                "result = {}\n"
                "for col_info in client.list_collections():\n"
                "    try:\n"
                "        col = client.get_collection((col_info.name if hasattr(col_info, 'name') else col_info['name']))\n"
                "        data = col.get(limit=1000, include=['documents', 'metadatas', 'ids'])\n"
                "        result[(col_info.name if hasattr(col_info, 'name') else col_info['name'])] = {\n"
                "            'ids': data.get('ids', []),\n"
                "            'documents': [d[:1000] if d else '' for d in (data.get('documents') or [])],\n"
                "            'metadatas': data.get('metadatas') or [],\n"
                "        }\n"
                "    except Exception:\n"
                "        pass\n"
                "print(json.dumps(result))\n"
            ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE))
            code, stdout, stderr = _run_python(script, timeout=30)
            if code == 0 and stdout.strip():
                try:
                    export_data["chromadb_collections"] = json.loads(stdout.strip())
                except Exception:
                    pass

        # Write file
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
        except Exception as e:
            return tool_error("Export write failed: %s" % e)

        return json.dumps({
            "result": "Export complete",
            "filepath": filepath,
            "kg_triples": len(export_data["kg_triples"]),
            "belief_history": len(export_data["belief_history"]),
            "chromadb_collections": len(export_data["chromadb_collections"]),
            "entity_aliases": len(export_data["entity_aliases"]),
        })

    def _handle_import(self, args: Dict[str, Any]) -> str:
        """Import memory from JSON export file."""
        filepath = args.get("filepath", "")
        dry_run = bool(args.get("dry_run", False))

        if not filepath:
            return tool_error("Missing required: filepath")

        if not Path(filepath).exists():
            return tool_error("File not found: %s" % filepath)

        try:
            data = json.loads(Path(filepath).read_text())
        except Exception as e:
            return tool_error("Invalid JSON: %s" % e)

        imported = {"kg_triples": 0, "belief_history": 0, "chromadb": 0, "entity_aliases": 0, "skipped_duplicates": 0}
        skipped = 0

        # ── Import KG triples ───────────────────────────────────────────
        for triple in data.get("kg_triples", []):
            s, p, o = triple.get("subject", ""), triple.get("predicate", ""), triple.get("object", "")
            if not all([s, p, o]):
                continue
            valid_from = triple.get("valid_from", datetime.now().isoformat()[:10])
            inverse_pred = triple.get("inverse_predicate", "")
            context = triple.get("context", "")
            source = triple.get("source", "import")
            confidence = float(triple.get("confidence", 1.0))
            subject_type = triple.get("subject_type", "")
            object_type = triple.get("object_type", "")

            if dry_run:
                imported["kg_triples"] += 1
                continue

            # Check duplicate
            check_script = (
                "import sqlite3\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"SELECT count(*) FROM triples WHERE subject=? AND predicate=? AND object=? AND (valid_to IS NULL OR valid_to='')\", (REPLACEME_S, REPLACEME_P, REPLACEME_O))\n"
                "count = cur.fetchone()[0]\n"
                "conn.close()\n"
                "print(count)\n"
            ).replace("REPLACEME_KG", repr(_HERMES_KG)).replace(
                "REPLACEME_S", repr(s)
            ).replace("REPLACEME_P", repr(p)).replace(
                "REPLACEME_O", repr(o)
            )
            code, stdout, stderr = _run_python(check_script)
            if code == 0 and stdout.strip() == "0":
                success, _ = _store_triple_with_inverse(
                    subject=s, predicate=p, obj=o,
                    valid_from=valid_from, context=context, source=source,
                    confidence=confidence, subject_type=subject_type, object_type=object_type,
                )
                if success:
                    imported["kg_triples"] += 1
            else:
                skipped += 1
                imported["skipped_duplicates"] += 1

        # ── Import entity aliases ─────────────────────────────────────────
        self._ensure_alias_table()
        for alias_rec in data.get("entity_aliases", []):
            entity = alias_rec.get("entity", "")
            alias = alias_rec.get("alias", "")
            if not entity or not alias:
                continue
            if dry_run:
                imported["entity_aliases"] += 1
                continue

            alias_script = (
                "import sqlite3\n"
                "conn = sqlite3.connect(REPLACEME_KG)\nconn.execute('PRAGMA journal_mode=WAL')\nconn.execute('PRAGMA busy_timeout=5000')\n"
                "cur = conn.cursor()\n"
                "cur.execute(\"INSERT OR IGNORE INTO entity_aliases VALUES(?,?,?,?,?,?)\", (REPLACEME_ID, REPLACEME_ENTITY, REPLACEME_ALIAS, REPLACEME_TYPE, REPLACEME_PALACE, REPLACEME_TODAY))\n"
                "rows = cur.execute(\"SELECT changes()\").fetchone()[0]\n"
                "conn.commit()\n"
                "conn.close()\n"
                "print(rows)\n"
            ).replace("REPLACEME_KG", repr(_HERMES_KG)).replace(
                "REPLACEME_ID", repr(str(alias_rec.get("id", "")))
            ).replace("REPLACEME_ENTITY", repr(entity)).replace(
                "REPLACEME_ALIAS", repr(alias)
            ).replace("REPLACEME_TYPE", repr(alias_rec.get("entity_type", ""))).replace(
                "REPLACEME_PALACE", repr(alias_rec.get("palace", "shared"))
            ).replace("REPLACEME_TODAY", repr(alias_rec.get("created_at", datetime.now().isoformat())))
            code, stdout, stderr = _run_python(alias_script)
            if code == 0 and stdout.strip() == "1":
                imported["entity_aliases"] += 1

        # ── Import ChromaDB drawers ───────────────────────────────────────
        if data.get("chromadb_collections"):
            for col_name, col_data in data["chromadb_collections"].items():
                docs = col_data.get("documents", [])
                metas = col_data.get("metadatas", [])
                ids = col_data.get("ids", [])
                for i, doc in enumerate(docs):
                    if not doc:
                        continue
                    if dry_run:
                        imported["chromadb"] += 1
                        continue

                    meta = (metas[i] or {}) if i < len(metas) else {}
                    drawer_id = ids[i] if i < len(ids) else str(uuid.uuid4())

                    imp = meta.get("importance", 0.5)
                    lang = meta.get("language", "unknown")
                    wing = meta.get("wing", col_name)
                    room = meta.get("room", "imported")
                    hall = meta.get("hall", "facts")
                    l0 = meta.get("l0", "")
                    l1 = meta.get("l1", "")
                    created_at = meta.get("created_at", datetime.now().isoformat())

                    col_script = (
                        "import chromadb, uuid\n"
                        "client = chromadb.PersistentClient(path=REPLACEME_PALACE)\n"
                        "try:\n"
                        "    col = client.get_collection(REPLACEME_COL)\n"
                        "except Exception:\n"
                        "    col = client.create_collection(REPLACEME_COL)\n"
                        "col.add(ids=[REPLACEME_ID], documents=[REPLACEME_DOC], metadatas=[{'wing': REPLACEME_WING, 'room': REPLACEME_ROOM, 'hall': REPLACEME_HALL, 'importance': REPLACEME_IMP, 'language': REPLACEME_LANG, 'l0': REPLACEME_L0, 'l1': REPLACEME_L1, 'created_at': REPLACEME_CREATED, 'last_accessed': REPLACEME_CREATED, 'access_count': 0}])\n"
                        "print('ok')\n"
                    ).replace("REPLACEME_PALACE", repr(_HERMES_PALACE)).replace(
                        "REPLACEME_COL", repr(col_name)
                    ).replace(
                        "REPLACEME_ID", repr(drawer_id)
                    ).replace(
                        "REPLACEME_DOC", repr(doc[:4000])
                    ).replace(
                        "REPLACEME_WING", repr(wing)
                    ).replace(
                        "REPLACEME_ROOM", repr(room)
                    ).replace(
                        "REPLACEME_HALL", repr(hall)
                    ).replace(
                        "REPLACEME_IMP", repr(imp)
                    ).replace(
                        "REPLACEME_LANG", repr(lang)
                    ).replace(
                        "REPLACEME_L0", repr(l0[:50]) if l0 else repr("")
                    ).replace(
                        "REPLACEME_L1", repr(l1[:300]) if l1 else repr("")
                    ).replace(
                        "REPLACEME_CREATED", repr(created_at)
                    )
                    code, stdout, stderr = _run_python(col_script)
                    if code == 0:
                        imported["chromadb"] += 1

        return json.dumps({
            "dry_run": dry_run,
            "imported": imported,
            "skipped_duplicates": skipped,
            "summary": "Import %s. KG: %d, Belief: %d, Aliases: %d, ChromaDB: %d. Skipped: %d" % (
                "dry-run" if dry_run else "complete",
                imported["kg_triples"], imported["belief_history"],
                imported["entity_aliases"], imported["chromadb"],
                imported["skipped_duplicates"],
            ),
        })

    # ------------------------------------------------------------------
    # OpenClaw read-only handlers
    # ------------------------------------------------------------------

    def _handle_openclaw_wakeup(self, args: Dict[str, Any]) -> str:
        result = _openclaw_wakeup(wing=args.get("wing"))
        if result.startswith("OpenClaw wake-up error:"):
            return tool_error(result)
        return json.dumps({"result": result, "source": "OpenClaw palace (READ-ONLY)"})

    def _handle_openclaw_search(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        wing = args.get("wing")
        room = args.get("room")
        n = min(int(args.get("n_results", 8)), 30)
        result = _openclaw_search(query=query, wing=wing, room=room, n_results=n)
        if result.startswith("OpenClaw search error:"):
            return tool_error(result)
        return json.dumps({"result": result, "query": query, "source": "OpenClaw palace (READ-ONLY)"})

    def _handle_openclaw_status(self, args: Dict[str, Any]) -> str:
        status = _openclaw_status()
        if "error" in status:
            return tool_error(status["error"])
        return json.dumps({"result": status, "source": "OpenClaw palace (READ-ONLY)"})

    def _handle_openclaw_kg_query(self, args: Dict[str, Any]) -> str:
        entity = args.get("entity", "")
        if not entity:
            return tool_error("Missing required parameter: entity")
        result = _openclaw_kg_query(entity=entity, as_of=args.get("as_of"))
        if result.startswith("OpenClaw KG query error:"):
            return tool_error(result)
        try:
            data = json.loads(result)
            return json.dumps({"result": data, "entity": entity, "source": "OpenClaw KG (READ-ONLY)"})
        except Exception:
            return json.dumps({"result": result, "entity": entity, "source": "OpenClaw KG (READ-ONLY)"})

    def shutdown(self) -> None:
        # CRITICAL FIX: Gracefully stop the WAL batcher so any in-flight flush
        # can complete before the process exits. Prevents zombie subprocesses
        # and un-flushed WAL data.
        _stop_wal_batcher(timeout=10.0)
        # Gracefully stop the cloud backup worker
        _backup.stop_backup_worker(timeout=10.0)
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        logger.info("MemPalace provider shutdown complete")


def register(ctx) -> None:
    """Register MemPalace as a memory provider plugin."""
    ctx.register_memory_provider(MemPalaceMemoryProvider())
