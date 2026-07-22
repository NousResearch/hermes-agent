"""Proactive opportunity detection over recent Hermes sessions.

The first version is intentionally deterministic and read-only. It mines recent
user messages for repeated task shapes and reports consent-first proposals that
can later be turned into skills, quick commands, cron jobs, Kanban templates, or
workflow runbooks.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence


_STOPWORDS = {
    "about", "again", "also", "and", "any", "are", "can", "could", "did",
    "does", "doing", "for", "from", "get", "give", "had", "has", "have",
    "help", "here", "hey", "how", "into", "just", "let", "like", "make",
    "more", "need", "now", "one", "our", "out", "please", "run", "same",
    "should", "show", "some", "that", "the", "their", "them", "then",
    "there", "this", "through", "use", "using", "want", "what", "when",
    "with", "would", "you", "your",
    "as", "at", "be", "he", "in", "it", "me", "my", "of", "on", "or",
    "she", "to", "we",
}

_TOKEN_ALIASES = {
    "analyse": "analyze",
    "analyzing": "analyze",
    "analysis": "analyze",
    "auditing": "review",
    "briefing": "brief",
    "compose": "draft",
    "composing": "draft",
    "debugging": "debug",
    "docs": "doc",
    "email": "mail",
    "emails": "mail",
    "fixing": "fix",
    "github": "git",
    "issue": "ticket",
    "issues": "ticket",
    "message": "update",
    "notes": "summary",
    "prs": "pr",
    "pull": "pr",
    "request": "pr",
    "requests": "pr",
    "recap": "summary",
    "reporting": "report",
    "reviewing": "review",
    "summaries": "summary",
    "summarise": "summarize",
    "summarize": "summary",
    "summarized": "summary",
    "summarizing": "summary",
    "tests": "test",
    "write": "draft",
    "writing": "draft",
}

_ACTION_TOKENS = {
    "analyze", "audit", "build", "check", "clean", "compare", "create",
    "debug", "deploy", "draft", "extract", "fix", "generate", "inspect",
    "monitor", "prepare", "publish", "review", "search", "summary", "test",
    "triage", "update",
}

_SCHEDULE_TOKENS = {
    "daily", "digest", "every", "friday", "hourly", "morning", "nightly",
    "recurring", "remind", "scheduled", "weekly",
}

_KANBAN_TOKENS = {
    "assign", "assignee", "board", "handoff", "kanban", "profile", "queue",
    "reviewer", "task", "worker", "worktree",
}

_PREFERENCE_TOKENS = {
    "always", "format", "pattern", "prefer", "preference", "style",
    "template", "tone",
}

_MEMORY_TOKENS = {"remember", "memory", "profile"}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_+-]{1,}")
_URL_RE = re.compile(r"https?://\S+")
_PATH_RE = re.compile(r"(?:[~./][\w./-]+)")
_PR_NUMBER_RE = re.compile(r"(?:#|pr\s*)\d+", re.IGNORECASE)


@dataclass(frozen=True)
class ProactiveEvidence:
    """One prior user message supporting an opportunity."""

    session_id: str
    source: str
    timestamp: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "source": self.source,
            "timestamp": self.timestamp,
            "text": self.text,
        }


@dataclass(frozen=True)
class ProactiveOpportunity:
    """A consent-first proposal inferred from repeated behavior."""

    key: str
    title: str
    description: str
    artifact_type: str
    confidence: float
    message_count: int
    session_count: int
    evidence: List[ProactiveEvidence]
    next_action: str
    signals: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "description": self.description,
            "artifact_type": self.artifact_type,
            "confidence": self.confidence,
            "message_count": self.message_count,
            "session_count": self.session_count,
            "evidence": [e.to_dict() for e in self.evidence],
            "next_action": self.next_action,
            "signals": list(self.signals),
        }


@dataclass
class _CandidateMessage:
    evidence: ProactiveEvidence
    tokens: set[str]
    artifact_hint: str


@dataclass
class _Cluster:
    tokens: Counter
    candidates: List[_CandidateMessage]

    def representative_tokens(self) -> set[str]:
        if not self.candidates:
            return set()
        if len(self.candidates) == 1:
            return set(self.tokens)
        # Tokens seen more than once carry the cluster. Fall back to the most
        # common singletons for sparse but repeated requests.
        repeated = {token for token, count in self.tokens.items() if count >= 2}
        if repeated:
            return repeated
        return {token for token, _count in self.tokens.most_common(6)}


class ProactiveEngine:
    """Detect repeated task opportunities from persisted session history."""

    def __init__(self, db):
        self.db = db
        self._conn = db._conn

    def generate(
        self,
        *,
        days: int = 30,
        source: Optional[str] = None,
        limit: int = 5,
        min_messages: int = 2,
    ) -> Dict[str, Any]:
        """Generate a proactive-mode report.

        ``min_messages`` is the minimum number of similar user asks needed to
        surface a proposal. At least two distinct sessions are preferred, but
        three repeated asks in one long-running session can still qualify.
        """
        days = max(1, int(days or 30))
        limit = max(1, min(20, int(limit or 5)))
        min_messages = max(2, int(min_messages or 2))
        cutoff = time.time() - days * 86400

        candidates = [
            c
            for c in (
                self._candidate_from_row(row)
                for row in self._fetch_user_messages(cutoff=cutoff, source=source)
            )
            if c is not None
        ]
        clusters = self._cluster_candidates(candidates)
        opportunities = [
            opp
            for opp in (
                self._opportunity_from_cluster(cluster, min_messages=min_messages)
                for cluster in clusters
            )
            if opp is not None
        ]
        opportunities.sort(
            key=lambda opp: (
                opp.confidence,
                opp.session_count,
                opp.message_count,
                max((e.timestamp for e in opp.evidence), default=0),
            ),
            reverse=True,
        )

        return {
            "days": days,
            "source_filter": source,
            "generated_at": time.time(),
            "opportunities": [opp.to_dict() for opp in opportunities[:limit]],
            "candidate_messages": len(candidates),
        }

    def format_terminal(self, report: Dict[str, Any]) -> str:
        """Format a proactive report for CLI and gateway output."""
        days = report.get("days", 30)
        source = report.get("source_filter")
        opportunities = report.get("opportunities") or []
        scope = f"last {days} day{'s' if days != 1 else ''}"
        if source:
            scope += f", source={source}"
        if not opportunities:
            return (
                f"No proactive opportunities yet ({scope}).\n"
                "Hermes did not find repeated task shapes strong enough to "
                "propose a reusable skill, command, workflow, Kanban template, "
                "or scheduled job. Try again after a few more similar sessions."
            )

        lines = [f"Proactive opportunities ({scope}):"]
        for idx, opp in enumerate(opportunities, 1):
            confidence = int(round(float(opp.get("confidence", 0)) * 100))
            artifact = opp.get("artifact_type", "workflow")
            lines.append("")
            lines.append(f"{idx}. {opp.get('title', 'Untitled opportunity')}")
            lines.append(
                "   "
                f"Type: {artifact} | confidence: {confidence}% | "
                f"seen {opp.get('message_count', 0)} times across "
                f"{opp.get('session_count', 0)} session(s)"
            )
            desc = opp.get("description")
            if desc:
                lines.append(f"   Why: {desc}")
            next_action = opp.get("next_action")
            if next_action:
                lines.append(f"   Next: {next_action}")
            evidence = opp.get("evidence") or []
            if evidence:
                lines.append("   Evidence:")
                for item in evidence[:3]:
                    ts = _format_timestamp(item.get("timestamp"))
                    sid = str(item.get("session_id") or "")[:10]
                    src = item.get("source") or "unknown"
                    text = _preview(item.get("text") or "", 120)
                    lines.append(f"     - {ts} {src}/{sid}: \"{text}\"")
        lines.append("")
        lines.append(
            "Preview only: nothing was created. Ask Hermes to create the "
            "suggested artifact when one of these looks right."
        )
        return "\n".join(lines)

    def _fetch_user_messages(
        self,
        *,
        cutoff: float,
        source: Optional[str],
        max_rows: int = 500,
    ) -> List[Dict[str, Any]]:
        params: list[Any] = [cutoff]
        source_clause = ""
        if source:
            source_clause = " AND s.source = ?"
            params.append(source)
        params.append(max_rows)
        with self.db._lock:
            rows = self._conn.execute(
                """
                SELECT
                    m.session_id,
                    s.source,
                    s.started_at,
                    m.timestamp,
                    m.content
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE s.started_at >= ?
                  AND m.role = 'user'
                  AND m.active = 1
                  AND m.content IS NOT NULL
                """
                + source_clause
                + """
                ORDER BY m.timestamp DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def _candidate_from_row(self, row: Dict[str, Any]) -> Optional[_CandidateMessage]:
        text = _text_from_content(self.db._decode_content(row.get("content")))
        if not _eligible_text(text):
            return None
        tokens = _token_set(text)
        if len(tokens) < 3:
            return None
        if not (tokens & _ACTION_TOKENS or tokens & _PREFERENCE_TOKENS or tokens & _MEMORY_TOKENS):
            return None
        evidence = ProactiveEvidence(
            session_id=str(row.get("session_id") or ""),
            source=str(row.get("source") or "unknown"),
            timestamp=float(row.get("timestamp") or row.get("started_at") or 0),
            text=_preview(text, 240),
        )
        return _CandidateMessage(
            evidence=evidence,
            tokens=tokens,
            artifact_hint=_artifact_type(tokens),
        )

    def _cluster_candidates(self, candidates: Sequence[_CandidateMessage]) -> List[_Cluster]:
        clusters: List[_Cluster] = []
        for candidate in candidates:
            best_cluster = None
            best_score = 0.0
            for cluster in clusters:
                rep = cluster.representative_tokens()
                score = _similarity(candidate.tokens, rep)
                if candidate.artifact_hint == _artifact_type(rep):
                    score += 0.05
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
            if best_cluster is not None and best_score >= 0.42:
                best_cluster.candidates.append(candidate)
                best_cluster.tokens.update(candidate.tokens)
            else:
                clusters.append(_Cluster(tokens=Counter(candidate.tokens), candidates=[candidate]))
        return clusters

    def _opportunity_from_cluster(
        self,
        cluster: _Cluster,
        *,
        min_messages: int,
    ) -> Optional[ProactiveOpportunity]:
        candidates = cluster.candidates
        message_count = len(candidates)
        session_count = len({c.evidence.session_id for c in candidates})
        if message_count < min_messages:
            return None
        if session_count < 2 and message_count < max(3, min_messages + 1):
            return None

        tokens = cluster.representative_tokens()
        artifact = _artifact_type(tokens)
        label = _label_from_tokens(tokens)
        confidence = _confidence(message_count=message_count, session_count=session_count)
        evidence = sorted(
            (c.evidence for c in candidates),
            key=lambda e: e.timestamp,
            reverse=True,
        )[:3]
        signals = _signals(tokens)
        key = "proactive:" + "-".join(sorted(tokens)[:8])

        return ProactiveOpportunity(
            key=key,
            title=f"Package repeated {label} as a {artifact.replace('_', ' ')}",
            description=_description(artifact, label, message_count, session_count),
            artifact_type=artifact,
            confidence=confidence,
            message_count=message_count,
            session_count=session_count,
            evidence=evidence,
            next_action=_next_action(artifact, label),
            signals=signals,
        )


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()
    return ""


def _eligible_text(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 24:
        return False
    if stripped.startswith("/"):
        return False
    if stripped.lower() in {"thanks", "thank you", "ok", "okay"}:
        return False
    return True


def _token_set(text: str) -> set[str]:
    normalized = _URL_RE.sub(" url ", text.lower())
    normalized = _PATH_RE.sub(" path ", normalized)
    normalized = _PR_NUMBER_RE.sub(" pr ", normalized)
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(normalized):
        token = _TOKEN_ALIASES.get(raw, raw)
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.add(token)
    if "pr" in tokens and "review" in tokens:
        tokens.add("code")
    return tokens


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    union = left | right
    return len(overlap) / len(union)


def _artifact_type(tokens: Iterable[str]) -> str:
    token_set = set(tokens)
    if token_set & _KANBAN_TOKENS:
        return "kanban"
    if token_set & _SCHEDULE_TOKENS and token_set & {"brief", "digest", "monitor", "report", "summary"}:
        return "cron"
    if token_set & _MEMORY_TOKENS and token_set & {"remember", "preference"}:
        return "memory"
    if token_set & _PREFERENCE_TOKENS:
        return "skill"
    workflow_score = len(token_set & {"approve", "deploy", "draft", "fix", "post", "release", "review", "test", "triage"})
    if workflow_score >= 3:
        return "workflow"
    if token_set & {"review", "summary", "draft", "search", "debug", "fix"}:
        return "quick_command"
    return "skill"


def _label_from_tokens(tokens: Iterable[str]) -> str:
    token_set = set(tokens)
    preferred = [
        "pr", "code", "review", "release", "slack", "mail", "digest",
        "brief", "debug", "test", "deploy", "ticket", "research",
        "summary", "draft", "format", "style",
    ]
    picked = [token for token in preferred if token in token_set]
    if not picked:
        picked = sorted(token_set)[:4]
    label = " ".join(picked[:5]).strip()
    return label or "workflow"


def _description(artifact: str, label: str, message_count: int, session_count: int) -> str:
    base = f"Hermes saw {message_count} similar request(s) across {session_count} session(s)"
    if artifact == "cron":
        return f"{base}; the repeated ask looks time-based or report-like enough to offer as an opt-in scheduled job."
    if artifact == "kanban":
        return f"{base}; the work appears to involve handoffs, workers, or task state that fits a Kanban template."
    if artifact == "workflow":
        return f"{base}; the requests combine multiple steps that should be captured as an inspectable runbook."
    if artifact == "quick_command":
        return f"{base}; the task shape is narrow enough to package as a reusable slash command."
    if artifact == "memory":
        return f"{base}; the repeated preference may belong in persistent memory after user confirmation."
    return f"{base}; the repeated pattern can be baked into a user-tailored skill for {label}."


def _next_action(artifact: str, label: str) -> str:
    if artifact == "cron":
        return f"Offer a `/suggestions` entry that schedules the {label} job only after explicit acceptance."
    if artifact == "kanban":
        return f"Create a Kanban task template for {label} with the right assignees, skills, and acceptance criteria."
    if artifact == "workflow":
        return f"Draft a runbook for {label}, then let the user approve before creating commands or automations."
    if artifact == "quick_command":
        return f"Create a quick command for {label} that preloads the right prompt, skills, and tool expectations."
    if artifact == "memory":
        return f"Ask the user to confirm the preference, then save it through the memory approval path."
    return f"Create or patch a skill that captures the preferred {label} workflow."


def _signals(tokens: Iterable[str]) -> List[str]:
    token_set = set(tokens)
    signals = []
    if token_set & _SCHEDULE_TOKENS:
        signals.append("recurring schedule/report language")
    if token_set & _KANBAN_TOKENS:
        signals.append("multi-agent or handoff language")
    if token_set & _PREFERENCE_TOKENS:
        signals.append("format/style preference")
    if token_set & {"pr", "review", "test", "release", "slack"}:
        signals.append("repeatable developer workflow")
    if not signals:
        signals.append("repeated user intent")
    return signals


def _confidence(*, message_count: int, session_count: int) -> float:
    raw = 0.45 + 0.13 * math.log1p(message_count) + 0.12 * math.log1p(session_count)
    return round(min(0.95, raw), 2)


def _preview(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 1)].rstrip() + "..."


def _format_timestamp(value: Any) -> str:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return "unknown-time"
    if ts <= 0:
        return "unknown-time"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def dumps_report(report: Dict[str, Any]) -> str:
    """Stable JSON representation for command surfaces."""
    return json.dumps(report, indent=2, sort_keys=True)
