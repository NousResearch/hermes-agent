"""
dreaming — automatic background memory consolidation for Hermes Agent.

Runs a 3-phase consolidation cycle during quiet hours via cron:

  Light  – Scan recent session transcripts, deduplicate, stage candidates
  REM    – Extract recurring themes and produce a Dream Diary narrative
  Deep   – Score candidates (relevance, frequency, recency, diversity,
           consolidation, richness) and promote high-scoring entries to MEMORY.md

The cycle is managed as a cron job (default: 0 3 * * *). It checks for recent
user activity before running and skips if the user was active within the
configured quiet window.

Config (plugins.entries.dreaming.config in config.yaml):

    dreaming:
      enabled: true
      frequency: "0 3 * * *"       # cron expression
      quiet_minutes: 60             # skip if user active within N minutes
      max_candidates: 50            # cap on staged candidates per cycle
      promotion_threshold: 0.6      # min score (0-1) to promote
      min_recall_count: 2           # min frequency for promotion
      lookback_days: 7              # days of session history to scan
      dream_diary_path: null        # null = ~/.hermes/DREAMS.md

Related: OpenClaw's Dreaming process (openclaw/openclaw#19685).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import textwrap
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLUGIN_NAME = "dreaming"
DEFAULT_FREQUENCY = "0 3 * * *"
DEFAULT_QUIET_MINUTES = 60
DEFAULT_MAX_CANDIDATES = 50
DEFAULT_PROMOTION_THRESHOLD = 0.6
DEFAULT_MIN_RECALL_COUNT = 2
DEFAULT_LOOKBACK_DAYS = 7

# Scoring weights (sum to 1.0)
_W_RELEVANCE = 0.30
_W_FREQUENCY = 0.24
_W_DIVERSITY = 0.15
_W_RECENCY = 0.15
_W_CONSOLIDATION = 0.10
_W_RICHNESS = 0.06

# Noise filter patterns for the Light phase
_NOISE_RE = [
    re.compile(r"^(ok|yes|no|thanks|thank you|sure|got it|nice|cool|great|perfect|alright)\b", re.I),
    re.compile(r"^[\s\W]+$"),
    re.compile(r"^(http|https|www\.)", re.I),
    re.compile(r"^```"),
    re.compile(r"^(>|#|\*|-|\d+\.)\s"),
    re.compile(r"^(error|warning|traceback|exception)\b", re.I),
    re.compile(r"^\[\[.*\]\]$"),
    re.compile(r"^MEDIA:"),
    re.compile(r"^(TICK_OK|HEARTBEAT_OK)$", re.I),
]

_ENGLISH_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "about", "up", "i", "me", "my", "we",
    "our", "you", "your", "he", "she", "it", "they", "this", "that",
    "these", "those", "what", "which", "who", "whom",
}

_MIN_STATEMENT_LEN = 20


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_cfg_cache: Dict[str, Any] = {}


def _cfg(key: str, default: Any = None) -> Any:
    """Read a single value from plugins.entries.dreaming.config (cached)."""
    if key in _cfg_cache:
        return _cfg_cache[key]
    try:
        from hermes_cli.config import cfg_get, load_config
        config = load_config()
        val = cfg_get(config, "plugins", "entries", "dreaming", "config", key)
        val = val if val is not None else default
    except Exception:
        val = default
    _cfg_cache[key] = val
    return val


def _reset_config_cache() -> None:
    """Reset the config cache. Used by tests."""
    _cfg_cache.clear()


def is_enabled() -> bool:
    return bool(_cfg("enabled", False))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class Candidate:
    """A staged memory candidate from the Light phase."""

    __slots__ = ("text", "source", "timestamp", "frequency", "score", "breakdown")

    def __init__(self, text: str, source: str = "", ts: Optional[datetime] = None):
        self.text: str = text.strip()
        self.source: str = source
        self.timestamp: datetime = ts or datetime.now(tz=timezone.utc)
        self.frequency: int = 1
        self.score: float = 0.0
        self.breakdown: Dict[str, float] = {}


class DreamReport:
    """Output of one full dreaming cycle."""

    def __init__(self):
        self.timestamp: datetime = datetime.now(tz=timezone.utc)
        self.light_count: int = 0
        self.rem_themes: List[str] = []
        self.promoted: List[str] = []
        self.skipped: List[str] = []
        self.narrative: str = ""

    def to_markdown(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        lines = [f"## Dream Cycle — {ts}", "",
                 f"**Light:** {self.light_count} candidates staged", ""]
        if self.rem_themes:
            lines.append("**REM Themes:**")
            lines.extend(f"- {t}" for t in self.rem_themes)
            lines.append("")
        if self.promoted:
            lines.append(f"**Deep:** {len(self.promoted)} promoted")
            lines.extend(f"- {p}" for p in self.promoted)
            lines.append("")
        if self.skipped:
            lines.append(f"**Skipped:** {len(self.skipped)} (below threshold)")
            lines.extend(f"- {s}" for s in self.skipped)
            lines.append("")
        if self.narrative:
            lines.append("**Dream Diary:**")
            lines.append(self.narrative)
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session store access
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    return get_hermes_home() / "sessions.db"


def _last_user_activity() -> Optional[datetime]:
    """Return the most recent user activity timestamp from the session store.

    Persisted on disk — survives gateway restarts.
    """
    p = _db_path()
    if not p.exists():
        return None
    try:
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT MAX(updated_at) AS t FROM sessions").fetchone()
        conn.close()
        if row and row["t"]:
            ts = row["t"]
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("dreaming: session store read failed: %s", exc)
    return None


def _is_quiet(minutes: int = DEFAULT_QUIET_MINUTES) -> bool:
    last = _last_user_activity()
    if last is None:
        return True
    return last < datetime.now(tz=timezone.utc) - timedelta(minutes=minutes)


def _recent_sessions(days: int = DEFAULT_LOOKBACK_DAYS) -> List[Dict[str, Any]]:
    """Return recent session transcripts from the SQLite store."""
    p = _db_path()
    if not p.exists():
        return []
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()
    sessions: List[Dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(p))
        conn.row_factory = sqlite3.Row
        for srow in conn.execute(
            "SELECT session_id, title, updated_at FROM sessions "
            "WHERE updated_at > ? ORDER BY updated_at DESC",
            (cutoff,),
        ):
            msgs = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY created_at ASC",
                (srow["session_id"],),
            ).fetchall()
            messages = [
                {"role": m["role"], "content": (m["content"][:500] + "\u2026") if isinstance(m["content"], str) and len(m["content"]) > 500 else m["content"]}
                for m in msgs
            ]
            if messages:
                sessions.append({
                    "session_id": srow["session_id"],
                    "title": srow["title"] or srow["session_id"],
                    "messages": messages,
                    "updated_at": srow["updated_at"],
                })
        conn.close()
    except Exception as exc:
        logger.warning("dreaming: session scan failed: %s", exc)
    return sessions


# ---------------------------------------------------------------------------
# Memory file I/O
# ---------------------------------------------------------------------------


def _dreams_path() -> Path:
    custom = _cfg("dream_diary_path")
    return Path(custom).expanduser() if custom else get_hermes_home() / "DREAMS.md"


def _memory_path() -> Path:
    return get_hermes_home() / "MEMORY.md"


def _read_memory() -> str:
    p = _memory_path()
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def _append_memory(entries: List[str]) -> None:
    if not entries:
        return
    existing = _read_memory()
    # Build set of meaningful content lines (strip markdown list prefixes)
    existing_content = set()
    for ln in existing.splitlines():
        cleaned = re.sub(r'^[-*]\s+', '', ln).strip().lower()
        if len(cleaned) > 10:
            existing_content.add(cleaned)
    new = []
    for e in entries:
        cleaned_e = e.lower().strip()
        is_dup = any(cleaned_e in ec or ec in cleaned_e for ec in existing_content if len(ec) > 5)
        if not is_dup:
            new.append(e)
    if not new:
        return
    ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    section = f"\n\n## Dreaming — {ts}\n" + "\n".join(f"- {e}" for e in new) + "\n"
    content = f"# MEMORY.md — Long-Term Memory\n{section}" if not existing.strip() else existing.rstrip() + "\n" + section
    _memory_path().write_text(content, encoding="utf-8")
    logger.info("dreaming: wrote %d entries to MEMORY.md", len(new))


def _append_dreams(report: DreamReport) -> None:
    p = _dreams_path()
    existing = p.read_text(encoding="utf-8") if p.exists() else ""
    md = report.to_markdown()
    content = f"# DREAMS.md — Dream Diary\n\n{md}\n" if not existing.strip() else existing.rstrip() + "\n" + md + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    logger.info("dreaming: wrote dream diary to %s", p)


# ---------------------------------------------------------------------------
# Light phase — sort, deduplicate, stage candidates
# ---------------------------------------------------------------------------


def _is_noise(text: str) -> bool:
    if len(text) < _MIN_STATEMENT_LEN:
        return True
    for pat in _NOISE_RE:
        if pat.match(text.strip()):
            return True
    return False


def _light_phase(
    sessions: List[Dict[str, Any]],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[Candidate]:
    """Extract and deduplicate memory candidates from session transcripts."""
    seen: Dict[str, Candidate] = {}
    for session in sessions:
        for msg in session["messages"]:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            for stmt in re.split(r'(?<=[.!?])\s+|\n+', content):
                stmt = stmt.strip()
                if _is_noise(stmt):
                    continue
                key = re.sub(r'\s+', ' ', stmt.lower().strip(" .!?,:;"))
                if key in seen:
                    seen[key].frequency += 1
                elif len(seen) < max_candidates:
                    seen[key] = Candidate(stmt, source=session["session_id"])
    return sorted(seen.values(), key=lambda c: c.frequency, reverse=True)


# ---------------------------------------------------------------------------
# REM phase — reflect on patterns
# ---------------------------------------------------------------------------


def _rem_phase(
    candidates: List[Candidate],
    sessions: List[Dict[str, Any]],
) -> Tuple[List[str], str]:
    """Extract recurring themes and produce a narrative."""
    if not candidates:
        return [], "No significant patterns detected this cycle."

    kw_groups: Dict[str, int] = {}
    for c in candidates:
        for w in re.findall(r'\b[a-z]{3,}\b', c.text.lower()):
            if w not in _ENGLISH_STOP:
                kw_groups[w] = kw_groups.get(w, 0) + c.frequency

    themes = [
        f"'{kw}' appeared {count} times"
        for kw, count in sorted(kw_groups.items(), key=lambda x: x[1], reverse=True)
        if count >= 2
    ][:10]

    top = candidates[:5]
    narrative = textwrap.dedent(
        f"Reviewed {len(candidates)} candidates from {len(sessions)} sessions.\n\n"
        f"Top themes: {', '.join(themes[:5]) if themes else 'none strong enough'}\n\n"
        "Most mentioned:\n" + "\n".join(f"  - {c.text[:120]}" for c in top[:3])
    )
    return themes, narrative


# ---------------------------------------------------------------------------
# Deep phase — score and promote
# ---------------------------------------------------------------------------


def _score(c: Candidate, total: int, existing: str) -> float:
    bd: Dict[str, float] = {}
    meaningful = len(re.findall(r'\b[a-z]{4,}\b', c.text.lower()))
    bd["relevance"] = min(meaningful / 10.0, 1.0) * _W_RELEVANCE
    bd["frequency"] = min(c.frequency / 5.0, 1.0) * _W_FREQUENCY
    bd["diversity"] = min(c.frequency / 3.0, 1.0) * _W_DIVERSITY
    age = (datetime.now(tz=timezone.utc) - c.timestamp).days
    bd["recency"] = max(0, 1.0 - age / DEFAULT_LOOKBACK_DAYS) * _W_RECENCY
    overlap = sum(1 for w in re.findall(r'\b[a-z]{4,}\b', c.text.lower()) if w in existing.lower())
    bd["consolidation"] = max(0, 1.0 - overlap / max(meaningful, 1)) * _W_CONSOLIDATION
    bd["richness"] = min(len(c.text) / 200.0, 1.0) * _W_RICHNESS
    c.score = sum(bd.values())
    c.breakdown = bd
    return c.score


def _deep_phase(
    candidates: List[Candidate],
    existing: str,
    threshold: float = DEFAULT_PROMOTION_THRESHOLD,
    min_recall: int = DEFAULT_MIN_RECALL_COUNT,
) -> Tuple[List[str], List[str]]:
    promoted, skipped = [], []
    for c in candidates:
        _score(c, len(candidates), existing)
        if c.score >= threshold and c.frequency >= min_recall:
            promoted.append(c.text)
        else:
            skipped.append(f"{c.text[:80]} (score={c.score:.2f}, freq={c.frequency})")
    return promoted, skipped


# ---------------------------------------------------------------------------
# Main cycle
# ---------------------------------------------------------------------------


def run_cycle(force: bool = False, verbose: bool = False) -> Optional[DreamReport]:
    """Run a full dreaming cycle (Light -> REM -> Deep).

    Returns DreamReport if a cycle ran, None if skipped.
    """
    if not is_enabled() and not force:
        logger.debug("dreaming: disabled, skipping")
        return None

    qmin = _cfg("quiet_minutes", DEFAULT_QUIET_MINUTES)
    if not force and not _is_quiet(qmin):
        logger.info("dreaming: user active within %dm, skipping", qmin)
        return None

    days = _cfg("lookback_days", DEFAULT_LOOKBACK_DAYS)
    max_c = _cfg("max_candidates", DEFAULT_MAX_CANDIDATES)
    threshold = _cfg("promotion_threshold", DEFAULT_PROMOTION_THRESHOLD)
    min_rec = _cfg("min_recall_count", DEFAULT_MIN_RECALL_COUNT)

    logger.info("dreaming: cycle starting (lookback=%dd)", days)
    report = DreamReport()

    # Light
    sessions = _recent_sessions(days)
    candidates = _light_phase(sessions, max_c)
    report.light_count = len(candidates)
    logger.info("dreaming: light — %d candidates from %d sessions", len(candidates), len(sessions))

    if not candidates:
        report.narrative = "No significant memories to consolidate."
        _append_dreams(report)
        return report

    # REM
    themes, narrative = _rem_phase(candidates, sessions)
    report.rem_themes = themes
    report.narrative = narrative
    logger.info("dreaming: rem — %d themes", len(themes))

    # Deep
    promoted, skipped = _deep_phase(candidates, _read_memory(), threshold, min_rec)
    report.promoted = promoted
    report.skipped = skipped
    if promoted:
        _append_memory(promoted)
        logger.info("dreaming: deep — %d promoted", len(promoted))
    else:
        logger.info("dreaming: deep — none promoted (threshold=%.2f)", threshold)

    _append_dreams(report)
    return report


# ---------------------------------------------------------------------------
# Cron job management
# ---------------------------------------------------------------------------


_CRON_JOB_NAME = "Memory Dreaming Cycle"


def _build_job() -> Dict[str, Any]:
    return {
        "id": f"dreaming-{uuid.uuid4().hex[:8]}",
        "name": _CRON_JOB_NAME,
        "schedule": _cfg("frequency", DEFAULT_FREQUENCY),
        "prompt": (
            "Run the Dreaming memory consolidation cycle. "
            "Call dream_run with force=true. "
            "After completion, reply DREAMING_DONE with a one-line summary."
        ),
        "enabled": True,
        "skills": [PLUGIN_NAME],
        "deliver": "local",
    }


def ensure_cron() -> Optional[str]:
    """Register the dreaming cron job if not already present. Returns job ID."""
    try:
        from cron.jobs import create_job, list_jobs
        for j in list_jobs():
            if j.get("name") == _CRON_JOB_NAME:
                logger.debug("dreaming: cron job already registered (%s)", j["id"])
                return j["id"]
        job = _build_job()
        created = create_job(
            prompt=job["prompt"],
            schedule=job["schedule"],
            name=job["name"],
            skills=job["skills"],
            deliver=job["deliver"],
        )
        logger.info("dreaming: cron job registered (%s)", created.get("id"))
        return created.get("id")
    except Exception as exc:
        logger.warning("dreaming: cron registration failed: %s", exc)
        return None


def remove_cron() -> None:
    try:
        from cron.jobs import list_jobs, remove_job
        for j in list_jobs():
            if j.get("name") == _CRON_JOB_NAME:
                remove_job(j["id"])
                logger.info("dreaming: cron job removed")
    except Exception as exc:
        logger.debug("dreaming: cron removal failed: %s", exc)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the dreaming plugin: hooks, CLI, cron, tools."""
    if not is_enabled():
        logger.info("dreaming: disabled in config — registering tools only")

    # Hooks
    ctx.register_hook("on_session_end", _on_session_end)

    # CLI
    try:
        from plugins.dreaming.cli import setup_cli, handle_cli
        ctx.register_cli_command(
            name="dream",
            help="Manage the Dreaming memory consolidation system",
            setup_fn=setup_cli,
            handler_fn=handle_cli,
            description="Run, inspect, and configure automatic memory consolidation.",
        )
    except Exception as exc:
        logger.warning("dreaming: CLI registration failed: %s", exc)

    # Tools
    try:
        ctx.register_tool(
            name="dream_run",
            toolset="dreaming",
            schema=DREAM_RUN_SCHEMA,
            handler=lambda args, **kw: _tool_run(args),
        )
        ctx.register_tool(
            name="dream_status",
            toolset="dreaming",
            schema=DREAM_STATUS_SCHEMA,
            handler=lambda args, **kw: _tool_status(),
        )
    except Exception as exc:
        logger.warning("dreaming: tool registration failed: %s", exc)

    # Cron
    if is_enabled():
        ensure_cron()

    logger.info("dreaming: plugin registered")


def _on_session_end(**kwargs) -> None:
    """Hook: called when a session ends. No-op; cron handles scheduling."""
    pass


# ---------------------------------------------------------------------------
# Tool schemas & handlers
# ---------------------------------------------------------------------------

DREAM_RUN_SCHEMA = {
    "name": "dream_run",
    "description": (
        "Run a Dreaming memory consolidation cycle. Normally called automatically "
        "by a cron job during quiet hours. Use force=True to run immediately."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "force": {
                "type": "boolean",
                "description": "Skip the quiet-hours check. Default: false.",
                "default": False,
            },
            "verbose": {
                "type": "boolean",
                "description": "Log detailed progress. Default: false.",
                "default": False,
            },
        },
        "required": [],
    },
}

DREAM_STATUS_SCHEMA = {
    "name": "dream_status",
    "description": "Get the current status of the Dreaming system.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _tool_run(args: Dict[str, Any]) -> str:
    force = args.get("force", False)
    verbose = args.get("verbose", False)
    report = run_cycle(force=force, verbose=verbose)
    if report is None:
        return json.dumps({"status": "skipped", "reason": "user_active_or_disabled"})
    return json.dumps({
        "status": "complete",
        "candidates": report.light_count,
        "themes": len(report.rem_themes),
        "promoted": len(report.promoted),
        "skipped": len(report.skipped),
    })


def _tool_status() -> str:
    last = _last_user_activity()
    return json.dumps({
        "enabled": is_enabled(),
        "frequency": _cfg("frequency", DEFAULT_FREQUENCY),
        "quiet_minutes": _cfg("quiet_minutes", DEFAULT_QUIET_MINUTES),
        "lookback_days": _cfg("lookback_days", DEFAULT_LOOKBACK_DAYS),
        "threshold": _cfg("promotion_threshold", DEFAULT_PROMOTION_THRESHOLD),
        "last_user_activity": str(last) if last else None,
    })
