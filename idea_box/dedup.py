"""Deduplication checker for the Idea Box flow.

Checks a new idea against three sources:
1. Kanban open tasks — title/body similarity
2. session_search — last 30 days of conversation history
3. Mnemosyne recall — semantic memory search

Each source is queried independently.  A match from any source marks
the idea as a duplicate and carries the match details so the idea card
can surface them to Sahil.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from idea_box.models import DedupResult

logger = logging.getLogger("idea_box.dedup")


class DedupSource(str, Enum):
    """Which dedup source produced a match."""

    KANBAN = "kanban"
    SESSION_SEARCH = "session_search"
    MNEMOSYNE = "mnemosyne"


@dataclass
class DedupMatch:
    """A single dedup match from one source."""

    source: DedupSource
    title: str = ""
    summary: str = ""
    ref_id: str = ""           # task id, session id, or memory id
    score: float = 0.0         # 0-1 similarity confidence
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "title": self.title,
            "summary": self.summary,
            "ref_id": self.ref_id,
            "score": self.score,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase tokenisation for simple overlap similarity."""
    return {w for w in text.lower().split() if len(w) > 2}


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity over tokenised strings.

    Simple but effective for short text (idea titles, task summaries).
    Returns 0.0–1.0.
    """
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _within_30_days(date_str: str) -> bool:
    """Check if an ISO date string is within the last 30 days.

    session_search has no native date filter, so we post-filter on the
    ``when`` field.  Returns True for unparseable dates (best-effort —
    a session with a malformed date is included rather than excluded).
    """
    from datetime import datetime, timezone, timedelta

    try:
        # session_search returns ISO timestamps like "2026-07-20" or
        # "2026-07-20T14:32:00+00:00".  Handle both.
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return True  # Best-effort: include if we can't parse the date

    now = datetime.now(timezone.utc)
    # If the parsed datetime is naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt) <= timedelta(days=30)


# Thresholds per source — how similar is "duplicate"?
SIMILARITY_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# DedupChecker
# ---------------------------------------------------------------------------

class DedupChecker:
    """Run dedup against kanban, session_search, and Mnemosyne.

    Each source is optional and lazily imported so the module works
    in test rigs without all backends present.  Pass mock backends
    via the constructor for testing.
    """

    def __init__(
        self,
        *,
        kanban_conn: Any = None,
        session_search_fn: Any = None,
        mnemosyne_recall_fn: Any = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self._kanban_conn = kanban_conn
        self._session_search_fn = session_search_fn
        self._mnemosyne_recall_fn = mnemosyne_recall_fn
        self._threshold = similarity_threshold

    # -- Kanban ----------------------------------------------------------

    def _check_kanban(self, idea_text: str) -> list[DedupMatch]:
        """Query open Kanban tasks for title/body similarity.

        Only open tasks (todo, ready, running, triage, blocked) are
        considered — done and archived tasks are skipped.  The SQL
        ``include_archived=False`` already excludes archived at the
        query level; the Python-side ``done`` skip is intentional
        because we only dedup against active work.
        """
        matches: list[DedupMatch] = []
        if self._kanban_conn is None:
            try:
                from hermes_cli import kanban_db as kb
                conn = kb.connect()
            except Exception:
                logger.debug("Kanban unavailable for dedup", exc_info=True)
                return matches
        else:
            conn = self._kanban_conn
            from hermes_cli import kanban_db as kb

        try:
            tasks = kb.list_tasks(conn, include_archived=False)
            for task in tasks:
                # Skip done and archived — only open work matters for dedup.
                # include_archived=False handles archived at the SQL level,
                # but we also check here for defense-in-depth (and so
                # tests with mocked list_tasks that ignore the kwarg work).
                if task.status in ("done", "archived"):
                    continue
                title_sim = _jaccard_similarity(idea_text, task.title or "")
                body_sim = _jaccard_similarity(idea_text, task.body or "")
                best = max(title_sim, body_sim)
                if best >= self._threshold:
                    matches.append(DedupMatch(
                        source=DedupSource.KANBAN,
                        title=task.title or "",
                        summary=(task.body or "")[:200],
                        ref_id=task.id,
                        score=round(best, 3),
                        extra={"status": task.status},
                    ))
        except Exception:
            logger.debug("Kanban dedup query failed", exc_info=True)
        finally:
            if self._kanban_conn is None:
                try:
                    conn.close()
                except Exception:
                    pass

        return matches

    # -- session_search --------------------------------------------------

    def _check_session_search(self, idea_text: str) -> list[DedupMatch]:
        """Query last 30 days of session history for similar ideas."""
        matches: list[DedupMatch] = []
        search_fn = self._session_search_fn

        if search_fn is None:
            try:
                from tools.session_search_tool import session_search
                search_fn = session_search
            except Exception:
                logger.debug("session_search unavailable for dedup", exc_info=True)
                return matches

        try:
            raw = search_fn(query=idea_text, limit=5)
            data = json.loads(raw) if isinstance(raw, str) else raw
            # Real session_search discovery returns {"results": [...], "count": N, ...}
            if isinstance(data, dict):
                sessions = data.get("results", [])
            elif isinstance(data, list):
                sessions = data
            else:
                sessions = []

            for sess in sessions:
                snippet = ""
                if isinstance(sess, dict):
                    # Enforce 30-day window (PRD acceptance criterion #2).
                    # session_search has no native date filter; post-filter
                    # on the ``when`` field.  Sessions without a date are
                    # included (best-effort — date may be missing).
                    when_str = sess.get("when", sess.get("created_at", ""))
                    if when_str:
                        if not _within_30_days(when_str):
                            continue
                    snippet = sess.get("snippet", sess.get("title", ""))
                    sid = sess.get("session_id", sess.get("id", ""))
                    sim = _jaccard_similarity(idea_text, snippet)
                    if sim >= self._threshold:
                        matches.append(DedupMatch(
                            source=DedupSource.SESSION_SEARCH,
                            title=sess.get("title", ""),
                            summary=snippet[:200],
                            ref_id=str(sid),
                            score=round(sim, 3),
                            extra={"when": when_str},
                        ))
        except Exception:
            logger.debug("session_search dedup query failed", exc_info=True)

        return matches

    # -- Mnemosyne --------------------------------------------------------

    def _check_mnemosyne(self, idea_text: str) -> list[DedupMatch]:
        """Query Mnemosyne for semantically similar memories.

        Mnemosyne is accessed via the tool injection layer at runtime.
        When the ``mnemosyne_recall_fn`` is not injected, the dedup
        gracefully reports no matches from this source.  The caller
        (agent loop / skill) is responsible for wiring the real recall
        function when available.
        """
        matches: list[DedupMatch] = []
        recall_fn = self._mnemosyne_recall_fn

        if recall_fn is None:
            # No injected recall function — Mnemosyne dedup is a no-op.
            # The agent/skill layer wires the real recall function at
            # runtime via the constructor.  We don't guess an import
            # path here because Mnemosyne's entry point depends on
            # the configured memory provider.
            return matches

        try:
            results = recall_fn(query=idea_text, limit=5)
            if isinstance(results, str):
                results = json.loads(results)
            if isinstance(results, dict):
                memories = results.get("memories", results.get("results", []))
            elif isinstance(results, list):
                memories = results
            else:
                memories = []

            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                content = mem.get("content", "")
                sim = _jaccard_similarity(idea_text, content)
                if sim >= self._threshold:
                    matches.append(DedupMatch(
                        source=DedupSource.MNEMOSYNE,
                        title=content[:120],
                        summary=content[:200],
                        ref_id=str(mem.get("id", "")),
                        score=round(sim, 3),
                        extra={"importance": mem.get("importance", 0)},
                    ))
        except Exception:
            logger.debug("Mnemosyne dedup query failed", exc_info=True)

        return matches

    # -- Public API -------------------------------------------------------

    def check(self, idea_text: str) -> DedupResult:
        """Run all three dedup checks and return the combined result."""
        all_matches: list[DedupMatch] = []
        checked: list[str] = []

        kanban_matches = self._check_kanban(idea_text)
        if kanban_matches:
            all_matches.extend(kanban_matches)
        checked.append(DedupSource.KANBAN.value)

        session_matches = self._check_session_search(idea_text)
        if session_matches:
            all_matches.extend(session_matches)
        checked.append(DedupSource.SESSION_SEARCH.value)

        mnemosyne_matches = self._check_mnemosyne(idea_text)
        if mnemosyne_matches:
            all_matches.extend(mnemosyne_matches)
        checked.append(DedupSource.MNEMOSYNE.value)

        is_dup = len(all_matches) > 0
        return DedupResult(
            is_duplicate=is_dup,
            matches=[m.to_dict() for m in all_matches],
            checked_sources=checked,
        )