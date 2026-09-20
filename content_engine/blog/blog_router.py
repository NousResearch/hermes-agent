from __future__ import annotations
from pathlib import Path
from typing import Optional
from datetime import UTC, datetime, timedelta
import json
import uuid

import activity_collector as ac
import database as db
from config import BLOG_TOPIC_RECENCY_DAYS

TOPIC_RESERVATIONS_PATH = Path(__file__).resolve().parent.parent / "blog_topics" / "topic_reservations.jsonl"
RESERVATION_TTL_MINUTES = 180
from blog.blog_streams import STREAMS, tags_for

# Generator-gate failures (skipped_generator) were previously untracked: no
# reason logged, no attempt count, and no quarantine — a topic sitting at the
# top of choose()'s priority order got re-selected and re-burned every single
# cron cycle forever with zero visibility (unlike failed_images, which has
# full tracking + a retry cron). This mirrors that same pattern for the
# generator/quality-gate path.
FAILED_GENERATOR_PATH = Path(__file__).resolve().parent.parent / "blog_topics" / "failed_generator.jsonl"
GENERATOR_FAILURE_THRESHOLD = 3


def _read_failed_generator_entries() -> list[dict]:
    if not FAILED_GENERATOR_PATH.exists():
        return []
    out: list[dict] = []
    for line in FAILED_GENERATOR_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def track_failed_generator(topic_id: str, stream: str, reason: str) -> int:
    """Record a generator/quality-gate rejection for a topic; return the new attempt count.

    Keyed by topic_id (a draft/slug may not exist yet when write() itself fails).
    Appends-or-increments, same shape as blog_pipeline's failed_images tracking.
    """
    if not topic_id:
        return 0
    today = datetime.now(UTC).date().isoformat()
    existing = _read_failed_generator_entries()
    attempts = 0
    found = False
    for e in existing:
        if e.get("topic_id") == topic_id:
            e["attempts"] = e.get("attempts", 0) + 1
            e["last_error"] = reason
            e["last_stream"] = stream
            e["date"] = today
            attempts = e["attempts"]
            found = True
            break
    if not found:
        existing.append({
            "topic_id": topic_id, "stream": stream, "attempts": 1,
            "last_error": reason, "first_failure": today, "date": today,
        })
        attempts = 1
    FAILED_GENERATOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAILED_GENERATOR_PATH, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")
    return attempts


def clear_failed_generator(topic_id: str) -> None:
    """Drop a topic's generator-failure tracking (called on eventual success)."""
    if not FAILED_GENERATOR_PATH.exists():
        return
    existing = [e for e in _read_failed_generator_entries() if e.get("topic_id") != topic_id]
    with open(FAILED_GENERATOR_PATH, "w", encoding="utf-8") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")


def get_quarantined_generator_topics(threshold: int = GENERATOR_FAILURE_THRESHOLD) -> list[dict]:
    """Entries that have hit the failure threshold — for audit/escalation, not silent retry."""
    return [e for e in _read_failed_generator_entries() if e.get("attempts", 0) >= threshold]


def _quarantined_generator_topic_ids(threshold: int = GENERATOR_FAILURE_THRESHOLD) -> set[str]:
    return {e.get("topic_id", "") for e in _read_failed_generator_entries()
            if e.get("attempts", 0) >= threshold and e.get("topic_id")}


def _recent_used(stream: str) -> list[str]:
    """Topic ids used by any blog stream within the recency window.

    Framework seeds are shared across AI/PM/Builder. A topic used by one stream
    should not be immediately reused by another stream unless explicitly queued
    as a series later.
    """
    used: set[str] = set()
    for s in STREAMS:
        try:
            used.update(db.get_recently_used_topics(
                f"blog_{s}", days=BLOG_TOPIC_RECENCY_DAYS,
            ))
        except Exception:
            continue
    return list(used)


def _reservation_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(minutes=RESERVATION_TTL_MINUTES)


def _read_reservations() -> list[dict]:
    if not TOPIC_RESERVATIONS_PATH.exists():
        return []
    out: list[dict] = []
    cutoff = _reservation_cutoff()
    for line in TOPIC_RESERVATIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            created = datetime.fromisoformat(item.get("created_at", ""))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
        except Exception:
            continue
        if created >= cutoff:
            out.append(item)
    # Opportunistically prune stale reservations.
    if out:
        TOPIC_RESERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TOPIC_RESERVATIONS_PATH.write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    elif TOPIC_RESERVATIONS_PATH.exists():
        TOPIC_RESERVATIONS_PATH.write_text("", encoding="utf-8")
    return out


def _reserved_topic_ids() -> set[str]:
    return {r.get("topic_id", "") for r in _read_reservations() if r.get("topic_id")}


def reserve(stream: str, topic_id: str, title: str = "") -> str:
    """Temporarily reserve a topic during generation.

    Reservations prevent concurrent/next stream selection without permanently
    burning the topic if generation fails. Call release(token) on failure and
    after successful record().
    """
    token = uuid.uuid4().hex
    entry = {
        "token": token,
        "stream": stream,
        "topic_id": topic_id,
        "title": title or "",
        "created_at": datetime.now(UTC).isoformat(),
    }
    existing = _read_reservations()
    existing.append(entry)
    TOPIC_RESERVATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_RESERVATIONS_PATH.write_text("".join(json.dumps(x) + "\n" for x in existing), encoding="utf-8")
    return token


def release(token: str) -> None:
    """Release a temporary topic reservation."""
    if not token or not TOPIC_RESERVATIONS_PATH.exists():
        return
    existing = [r for r in _read_reservations() if r.get("token") != token]
    TOPIC_RESERVATIONS_PATH.write_text("".join(json.dumps(x) + "\n" for x in existing), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping blank/comment lines. Returns list of parsed dicts."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(obj)
    return out


def _gather_framework_candidates() -> list[dict]:
    """Read framework seeds from blog_topics/frameworks.jsonl.

    Framework seeds get priority 8-9 so they're picked ahead of regular topics.
    """
    p = Path(__file__).resolve().parent.parent / "blog_topics" / "frameworks.jsonl"
    objs = _read_jsonl(p)
    cands = []
    for obj in objs:
        priority = obj.get("priority", 8)
        cands.append({
            "topic_id": obj.get("topic_id", ""),
            "title_hint": obj.get("title_hint", ""),
            "tags": obj.get("tags", []),
            "source_override": "manual_queue",
            "signals": [{
                "signal_id": obj.get("topic_id", ""),
                "summary": obj.get("title_hint", ""),
                "priority": priority,
            }],
            "priority": priority,
            "domain": obj.get("domain", ""),
        })
    return cands


def _gather_candidates(stream: str) -> list[dict]:
    """Gather candidate topics for a stream.

    All streams: framework seeds injected at highest priority.
    Builder: also uses activity_collector signals.
    AI/PM/Research: read from the manual topic queue if it exists.
    Research stream reuses the same JSONL queue pattern as AI/PM; populate
    ``blog_topics/research.jsonl`` to feed the lane.
    """
    if stream not in STREAMS:
        return []

    # Framework seeds are injected for ALL streams at highest priority.
    framework_cands = _gather_framework_candidates()

    if stream == "builder":
        try:
            result = ac.collect_all()
            signals = result.get("signals", [])
        except Exception:
            signals = []
        cands = []
        for sig in signals:
            cands.append({
                "topic_id": sig.get("signal_id", ""),
                "title_hint": sig.get("summary", ""),
                "tags": [],
                "source_override": None,
                "signals": [sig],
                "priority": sig.get("priority", 0),
            })
        # Builder also pulls from its own backlog queue (builder.jsonl), so the
        # stream has a durable topic source, not just live activity signals.
        manual = _read_manual_queue("builder")
        return framework_cands + manual + cands

    # AI / PM / Research: read from the manual topic queue if it exists.
    cands = _read_manual_queue(stream)
    return framework_cands + cands


def _manual_queue_path(stream: str):
    """Path to the manual topic queue for a stream."""
    return Path(__file__).resolve().parent.parent / "blog_topics" / f"{stream}.jsonl"


# Placeholder titles an external writer (kanban/dashboard UI) drops into the
# queue files. They carry no real topic, so they must never be generated.
_PLACEHOLDER_TITLES = {"", "new concept", "untitled", "tbd"}


def _read_manual_queue(stream: str) -> list[dict]:
    """Read queued topics from blog_topics/<stream>.jsonl (one JSON object per line).

    Defensively skips placeholder/empty stubs (e.g. "New Concept") so junk
    entries written by other tools can never be selected for generation.
    """
    # Per-stream default source when the queue entry doesn't carry an explicit
    # source_override. AI/PM/builder queue entries historically use
    # ``manual_queue`` so blog_publisher can mark them as user-curated; the
    # research-roundup queue should fall through to the stream's configured
    # source (``curated-roundup``) so we don't lose the lane's provenance
    # signal. Passing None here means choose() will pick up
    # ``STREAMS[stream]["source"]`` instead of the historical override.
    default_source_override = (
        None if stream == "research" else "manual_queue"
    )
    p = _manual_queue_path(stream)
    objs = _read_jsonl(p)
    out = []
    for obj in objs:
        if (obj.get("title_hint") or "").strip().lower() in _PLACEHOLDER_TITLES:
            continue
        editorial_brief = {
            key: obj.get(key, "")
            for key in (
                "post_thesis", "concrete_takeaway", "evidence_anchor",
                "gap_claim", "stream_format_rationale",
            )
        }
        out.append({
            "topic_id": obj.get("topic_id", ""),
            "title_hint": obj.get("title_hint", ""),
            "tags": obj.get("tags", []),
            "source_override": (
                obj.get("source_override") or default_source_override
            ),
            "signals": [{
                "signal_id": obj.get("topic_id", ""),
                "summary": obj.get("title_hint", ""),
                "priority": obj.get("priority", 5),
            }],
            "priority": obj.get("priority", 5),
            "editorial_brief": editorial_brief,
        })
    return out


def _get_quality_scores(stream: str) -> dict[str, int]:
    """Fetch historical quality scores for framework topics from DB.

    Returns dict of {topic_id: quality_score}.
    """
    try:
        brand = f"blog_{stream}"
        scores_raw = db.get_quality_scores(brand)
        return {r["topic_id"]: r["quality_score"] for r in scores_raw}
    except Exception:
        return {}


def choose(stream: str) -> Optional[dict]:
    """Pick the highest-priority unused topic for a stream.

    Framework topics (priority 8-9) are chosen first. Within equal priority,
    historical quality_score is used as tiebreaker (higher = preferred).

    Returns a topic dict or None when no candidates remain.
    """
    if stream not in STREAMS:
        return None
    blocked = set(_recent_used(stream)) | _reserved_topic_ids() | _quarantined_generator_topic_ids()
    cands = [c for c in _gather_candidates(stream)
             if c.get("topic_id") and c["topic_id"] not in blocked]
    if not cands:
        return None

    # Fetch quality scores for tiebreaking.
    qs = _get_quality_scores(stream)

    # Sort: descending priority, then descending quality_score, then topic_id.
    cands.sort(key=lambda c: (
        -c.get("priority", 0),
        -(qs.get(c.get("topic_id", "")) or 0),
        c.get("topic_id", ""),
    ))
    top = cands[0]
    source = top.get("source_override") or STREAMS[stream]["source"]
    # Mark framework domain so the writer can route to blueprint format.
    domain = top.get("domain", "")
    return {
        "topic_id": top["topic_id"],
        "title_hint": top.get("title_hint", ""),
        "tags": tags_for(stream, top.get("tags", [])),
        "source": source,
        "signals": top.get("signals", []),
        "domain": domain,
        "format": "blueprint" if domain else "essay",
        "editorial_brief": top.get("editorial_brief", {}),
    }


def record(stream: str, topic_id: str, title: str,
           quality_score: Optional[int] = None) -> None:
    """Persist the chosen topic id so future runs cannot re-pick it.

    Called only after a successful publish (record-on-success).
    Stores quality_score when provided for routing feedback.
    """
    try:
        db.log_topic_usage(
            topic_id=topic_id, brand=f"blog_{stream}",
            topic_text=(title or "")[:200], platform="blog",
            quality_score=quality_score,
        )
    except Exception:
        pass
    clear_failed_generator(topic_id)
