"""
gateway/recall.py — /recall command implementation.

Restores context from prior sessions into a fresh session. Three modes:
  - thread  (default): prior session(s) in this same session_key
  - window:  sessions in this thread within a time window
  - topic:   FTS5 search across all sessions

Public API:
  parse_recall_args(raw: str) -> RecallSpec
  run_recall(raw_args, session_key, session_store, session_db, *, runtime_kwargs, model) -> RecallResult
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^(\d+)([hdwm])$", re.IGNORECASE)
_LAST_N_RE   = re.compile(r"^last\s+(\d+)$", re.IGNORECASE)
_BARE_N_RE   = re.compile(r"^\d+$")


def _duration_to_hours(token: str) -> Optional[int]:
    m = _DURATION_RE.match(token)
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2).lower()
    return {"h": n, "d": n * 24, "w": n * 24 * 7, "m": n * 24 * 30}[unit]


@dataclass(frozen=True)
class RecallSpec:
    mode: str                        # "thread" | "window" | "topic"
    count: int = 1                   # thread mode: how many prior sessions
    window_hours: Optional[int] = None
    query: Optional[str] = None      # topic search phrase


def parse_recall_args(raw: str) -> RecallSpec:
    """Parse /recall argument string into a structured spec."""
    s = (raw or "").strip()

    if not s:
        return RecallSpec(mode="thread", count=1)

    # bare integer → thread count
    if _BARE_N_RE.match(s):
        return RecallSpec(mode="thread", count=min(int(s), 10))

    # "last N" → thread count
    m = _LAST_N_RE.match(s)
    if m:
        return RecallSpec(mode="thread", count=min(int(m.group(1)), 10))

    # bare duration → window mode
    hrs = _duration_to_hours(s)
    if hrs is not None:
        return RecallSpec(mode="window", window_hours=hrs)

    # topic mode, possibly with trailing duration
    parts = s.rsplit(None, 1)
    if len(parts) == 2:
        trailing_hrs = _duration_to_hours(parts[1])
        if trailing_hrs is not None:
            return RecallSpec(mode="topic", query=parts[0].strip(), window_hours=trailing_hrs)

    return RecallSpec(mode="topic", query=s)


# ---------------------------------------------------------------------------
# Source references
# ---------------------------------------------------------------------------

@dataclass
class SourceRef:
    session_id: str
    title: str
    message_count: int
    ended_at: float          # unix timestamp
    excerpt: str = ""        # FTS snippet for topic mode


def _ago(ts: float) -> str:
    """Human-readable 'X ago' for a unix timestamp."""
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return f"{int(diff / 86400)}d ago"


def _format_transcript(messages: list[dict], char_cap: int) -> str:
    """Render a message list as plain text, truncated to char_cap."""
    lines = []
    total = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if not content or role not in ("user", "assistant"):
            continue
        # Truncate very long individual messages
        if len(content) > 2000:
            content = content[:2000] + " [...]"
        line = f"{role.upper()}: {content}"
        if total + len(line) > char_cap:
            lines.append("[transcript truncated for length]")
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Source resolver
# ---------------------------------------------------------------------------

def resolve_sources(
    spec: RecallSpec,
    session_key: str,
    session_store,    # gateway.session.SessionStore
    session_db,       # hermes_state.SessionDB  (may be None)
    *,
    now: Optional[float] = None,
) -> list[SourceRef]:
    """Return the list of SourceRefs to summarize for this spec."""
    now = now or time.time()

    if spec.mode == "topic":
        return _resolve_topic(spec, session_db, now)

    # thread / window — filter by session_key from session_store._entries
    refs = _prior_sessions_for_key(session_key, session_store, session_db)

    if spec.mode == "window":
        cutoff = now - ((spec.window_hours or 0) * 3600)
        refs = [r for r in refs if r.ended_at >= cutoff]
        return refs

    # thread mode: just last N
    return refs[:spec.count]


def _prior_sessions_for_key(
    session_key: str,
    session_store,
    session_db,
) -> list[SourceRef]:
    """All historical sessions for this session_key, most-recent first, excluding current.

    Strategy: walk the parent_session_id chain starting from the current session.
    Sessions are stored with source='telegram' (platform name only), NOT the
    full session_key like 'telegram:469214623:94316', so searching by source=session_key
    never matches. The parent chain is the reliable way to find prior sessions
    in the same thread.
    """
    refs: list[SourceRef] = []

    if session_db is None:
        logger.debug("recall: no session_db, cannot resolve prior sessions")
        return []

    try:
        current_entry = session_store._entries.get(session_key) if session_store else None
        current_sid = current_entry.session_id if current_entry else None

        if not current_sid:
            logger.debug("recall: no current session_id for key %s", session_key)
            return []

        # Walk the parent_session_id chain: current → parent → grandparent → ...
        # Skip the current (live) session — only return prior sessions.
        visited: set[str] = set()
        next_sid = _get_parent_session_id(session_db, current_sid)
        while next_sid and next_sid not in visited and len(refs) < 20:
            visited.add(next_sid)
            row = session_db.get_session(next_sid)
            if row is None:
                break
            title = row.get("title") or f"Session {next_sid[:8]}"
            ended_at = row.get("ended_at") or row.get("started_at") or 0
            msg_count = session_db.message_count(session_id=next_sid)
            refs.append(SourceRef(
                session_id=next_sid,
                title=title,
                message_count=msg_count,
                ended_at=float(ended_at),
            ))
            next_sid = _get_parent_session_id(session_db, next_sid)

    except Exception:
        logger.debug("recall: error querying prior sessions", exc_info=True)

    return refs


def _get_parent_session_id(session_db, session_id: str) -> Optional[str]:
    """Return the parent_session_id for the given session, or None."""
    try:
        row = session_db.get_session(session_id)
        if row is None:
            return None
        return row.get("parent_session_id")
    except Exception:
        return None


def _resolve_topic(
    spec: RecallSpec,
    session_db,
    now: float,
) -> list[SourceRef]:
    """FTS5 search across all sessions for the given topic."""
    if session_db is None or not spec.query:
        return []

    try:
        cutoff_ts = (now - spec.window_hours * 3600) if spec.window_hours else None
        # Use search_messages (FTS5) to find matching messages
        results = session_db.search_messages(
            query=spec.query,
            limit=50,
        )
        # Dedupe by session_id, keep top 5 by score (results already ranked)
        seen: dict[str, SourceRef] = {}
        for row in results:
            sid = row.get("session_id")
            if not sid or sid in seen:
                continue
            session_ts = float(row.get("session_started_at") or row.get("timestamp") or 0)
            if cutoff_ts and session_ts < cutoff_ts:
                continue
            snippet = row.get("content") or row.get("snippet") or ""
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            title = row.get("session_title") or row.get("title") or f"Session {sid[:8]}"
            msg_count = session_db.message_count(session_id=sid)
            seen[sid] = SourceRef(
                session_id=sid,
                title=title,
                message_count=msg_count,
                ended_at=session_ts,
                excerpt=snippet,
            )
            if len(seen) >= 5:
                break
        return list(seen.values())
    except Exception:
        logger.debug("recall: error in topic search", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

_SUMMARIZER_SYSTEM = """You are summarizing prior conversation(s) so a fresh AI agent \
picking up the work has the context it needs. Be specific and concrete:

- What was being worked on (project, file paths, goal, task)
- What was tried (approaches, commands run, key decisions made)
- What worked, what broke, what was left unresolved
- Any names of people, services, agents, or fleet members referenced
- Any decisions or preferences the user expressed

Keep it tight — bullet points preferred, no fluff. Skip pleasantries and meta-chatter. \
If multiple source sessions are provided, group by source with a one-line citation.

Format: plain text, no markdown headers. Aim for 200-500 words."""

_SUMMARIZER_USER_TMPL = """\
Summarize the following {n} prior session(s) for context restoration{topic_clause}.

{transcripts}

Output a single context block the agent can use to orient itself."""

# In-memory cache: (cache_key) -> summary string
_SUMMARY_CACHE: dict[str, str] = {}


def _cache_key(refs: list[SourceRef], query: Optional[str]) -> str:
    payload = json.dumps(
        [(r.session_id, r.message_count) for r in refs] + [query or ""],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def summarize_sources(
    refs: list[SourceRef],
    session_store,
    session_db,
    *,
    query: Optional[str] = None,
    runtime_kwargs: Optional[dict] = None,
    model: Optional[str] = None,
) -> str:
    """Spawn a cheap ephemeral AIAgent and summarize the source transcripts."""
    if not refs:
        return ""

    key = _cache_key(refs, query)
    if key in _SUMMARY_CACHE:
        logger.debug("recall: cache hit for %s", key[:12])
        return _SUMMARY_CACHE[key]

    # Build per-source transcript text
    PER_SOURCE_CHAR_CAP = 12_000   # ~3k tokens per source, max 5 sources = 15k
    transcripts = []
    for ref in refs:
        try:
            msgs = session_store.load_transcript(ref.session_id) if session_store else []
        except Exception:
            msgs = []
        text = _format_transcript(msgs, char_cap=PER_SOURCE_CHAR_CAP)
        if not text:
            text = "[transcript unavailable]"
        age = _ago(ref.ended_at)
        transcripts.append(
            f"--- {ref.title} ({age}, {ref.message_count} messages) ---\n{text}"
        )

    topic_clause = f" focused on: {query!r}" if query else ""
    user_msg = _SUMMARIZER_USER_TMPL.format(
        n=len(refs),
        topic_clause=topic_clause,
        transcripts="\n\n".join(transcripts),
    )

    try:
        from run_agent import AIAgent

        kw = dict(runtime_kwargs or {})
        summary_agent = AIAgent(
            **kw,
            model=model or kw.get("model") or "",
            max_iterations=4,
            quiet_mode=True,
            skip_memory=True,
            skip_context_files=True,
            enabled_toolsets=[],
        )
        try:
            summary_agent._print_fn = None  # type: ignore[assignment]
        except Exception:
            pass

        result = summary_agent.run_conversation(
            user_message=user_msg,
            system_message=_SUMMARIZER_SYSTEM,
        )
        summary = (result.get("final_response") or "").strip()
    except Exception:
        logger.warning("recall: summarizer agent failed", exc_info=True)
        summary = ""

    if summary:
        _SUMMARY_CACHE[key] = summary

    return summary


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

@dataclass
class RecallResult:
    injected_user_message: str   # prepended to new session as user-role message
    reply: str                   # sent back to the human
    source_count: int


def _format_header(spec: RecallSpec, refs: list[SourceRef]) -> str:
    if spec.mode == "thread":
        if len(refs) == 1:
            return (
                f"prior session in this thread "
                f"({refs[0].message_count} msgs, {_ago(refs[0].ended_at)})"
            )
        return f"last {len(refs)} sessions in this thread"
    if spec.mode == "window":
        return f"sessions in the last {spec.window_hours}h ({len(refs)} found)"
    if spec.mode == "topic":
        suffix = f", last {spec.window_hours}h" if spec.window_hours else ""
        return f"topic search: {spec.query!r}{suffix} ({len(refs)} sources)"
    return ""


def run_recall(
    raw_args: str,
    session_key: str,
    session_store,
    session_db,
    *,
    runtime_kwargs: Optional[dict] = None,
    model: Optional[str] = None,
) -> RecallResult:
    """
    Top-level entry point called by the gateway handler.

    Returns a RecallResult with:
      - injected_user_message: inject as a user-role message into the new session
      - reply: send back to the human
      - source_count: 0 means nothing found / injected
    """
    spec = parse_recall_args(raw_args)
    refs = resolve_sources(spec, session_key, session_store, session_db)

    if not refs:
        if spec.mode == "topic":
            msg = f"No sessions matched topic: {spec.query!r}"
        elif spec.mode == "window":
            msg = f"No sessions found in this thread within the last {spec.window_hours}h."
        else:
            msg = (
                "Nothing to recall — no prior sessions found for this thread.\n"
                "Tip: use `/recall <topic>` to search across all sessions."
            )
        return RecallResult(injected_user_message="", reply=msg, source_count=0)

    summary = summarize_sources(
        refs,
        session_store,
        session_db,
        query=spec.query,
        runtime_kwargs=runtime_kwargs,
        model=model,
    )

    if not summary:
        return RecallResult(
            injected_user_message="",
            reply="⚠️ Recall couldn't generate a summary (summarizer failed). Try again.",
            source_count=0,
        )

    header = _format_header(spec, refs)
    injected = f"[Context restored via /recall — {header}]\n\n{summary}"
    reply = f"📋 **Recall:** {header}\n\n{summary}"

    return RecallResult(
        injected_user_message=injected,
        reply=reply,
        source_count=len(refs),
    )
