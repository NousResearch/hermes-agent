"""
Nudge storage and management.

Nudges are lightweight wake-up reminders that agents can schedule for themselves.
Unlike cron jobs, nudges don't execute any work - they simply trigger the agent
to wake up and continue processing with full context.

Nudges persist across gateway restarts and are stored in ~/.hermes/nudges.json
"""

import json
import logging
import re
import threading
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_json_write

# =============================================================================
# Configuration
# =============================================================================

HERMES_DIR = get_hermes_home().resolve()
NUDGES_FILE = HERMES_DIR / "nudges.json"

# In-process lock protecting load→modify→save cycles
_nudges_file_lock = threading.Lock()


# =============================================================================
# Schedule Parsing
# =============================================================================

class ParsedSchedule:
    """Result of parsing a schedule string."""
    def __init__(self, fire_at: datetime, is_recurring: bool = False,
                 interval_seconds: Optional[int] = None, raw_schedule: str = ""):
        self.fire_at = fire_at
        self.is_recurring = is_recurring
        self.interval_seconds = interval_seconds
        self.raw_schedule = raw_schedule


def _parse_schedule(schedule: str) -> Optional[ParsedSchedule]:
    """
    Parse a schedule string into a ParsedSchedule object.

    Supports:
    - One-time: "5m", "30s", "2h", "1d", or ISO datetime
    - Recurring: "every 5m", "every 30s", "every 2h", "every 1d"

    Returns None if parsing fails.
    """
    schedule = schedule.strip().lower()

    # Recurring: "every 5m", "every 30s", "every 2h", "every 1d"
    recurring_match = re.match(r'^every\s+(\d+)([smhd])$', schedule)
    if recurring_match:
        amount = int(recurring_match.group(1))
        unit = recurring_match.group(2)
        delta = {
            's': timedelta(seconds=amount),
            'm': timedelta(minutes=amount),
            'h': timedelta(hours=amount),
            'd': timedelta(days=amount),
        }[unit]
        interval_seconds = int(delta.total_seconds())
        fire_at = _hermes_now() + delta
        return ParsedSchedule(
            fire_at=fire_at,
            is_recurring=True,
            interval_seconds=interval_seconds,
            raw_schedule=schedule
        )

    # One-time relative: 30s, 5m, 2h, 1d
    rel_match = re.match(r'^(\d+)([smhd])$', schedule)
    if rel_match:
        amount = int(rel_match.group(1))
        unit = rel_match.group(2)
        delta = {
            's': timedelta(seconds=amount),
            'm': timedelta(minutes=amount),
            'h': timedelta(hours=amount),
            'd': timedelta(days=amount),
        }[unit]
        fire_at = _hermes_now() + delta
        return ParsedSchedule(
            fire_at=fire_at,
            is_recurring=False,
            raw_schedule=schedule
        )

    # One-time absolute: ISO datetime
    try:
        fire_at = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
        return ParsedSchedule(
            fire_at=fire_at,
            is_recurring=False,
            raw_schedule=schedule
        )
    except ValueError:
        pass

    return None


# =============================================================================
# Nudge CRUD Operations
# =============================================================================

def _ensure_dirs():
    """Ensure nudges directory exists."""
    HERMES_DIR.mkdir(parents=True, exist_ok=True)


def load_nudges() -> List[Dict[str, Any]]:
    """Load all nudges from disk."""
    if not NUDGES_FILE.exists():
        return []
    try:
        with open(NUDGES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Failed to load nudges: %s", e)
        return []


def save_nudges(nudges: List[Dict[str, Any]]):
    """Save nudges to disk atomically."""
    _ensure_dirs()
    with _nudges_file_lock:
        atomic_json_write(NUDGES_FILE, nudges, indent=2)


def create_nudge(
    session_id: str,
    session_key: str,
    schedule: str,
    context: Optional[str] = None,
    name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Create a new nudge for an agent session.

    Args:
        session_id: The agent's session ID (for context persistence)
        session_key: The session key (for routing)
        schedule: When to fire - "5m" (one-time), "every 5m" (recurring),
                  "30s", "2h", "1d", or ISO datetime (one-time)
        context: Optional context message to include when nudge fires
        name: Optional name for this nudge

    Returns:
        The created nudge dict, or None if schedule parsing failed
    """
    parsed = _parse_schedule(schedule)
    if not parsed:
        return None

    nudge_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()

    nudge = {
        "id": nudge_id,
        "name": name or f"nudge_{nudge_id}",
        "session_id": session_id,
        "session_key": session_key,
        "fire_at": parsed.fire_at.isoformat(),
        "context": context or "",
        "created_at": now,
        "fired": False,
        "fired_at": None,
        # Recurring nudge fields
        "is_recurring": parsed.is_recurring,
        "interval_seconds": parsed.interval_seconds,
        "schedule": parsed.raw_schedule,
    }

    nudges = load_nudges()
    nudges.append(nudge)
    save_nudges(nudges)

    logger.info("Created %snudge %s for session %s at %s",
                "recurring " if parsed.is_recurring else "",
                nudge_id, session_id, parsed.fire_at.isoformat())
    return nudge


def get_nudge(nudge_id: str) -> Optional[Dict[str, Any]]:
    """Get a nudge by ID."""
    nudges = load_nudges()
    for nudge in nudges:
        if nudge["id"] == nudge_id:
            return nudge
    return None


def list_nudges(
    session_id: Optional[str] = None,
    include_fired: bool = False,
) -> List[Dict[str, Any]]:
    """
    List nudges, optionally filtered by session.

    Args:
        session_id: Filter to this session only
        include_fired: Include already-fired nudges
    """
    nudges = load_nudges()
    result = []
    for nudge in nudges:
        if not include_fired and nudge.get("fired", False):
            continue
        if session_id and nudge.get("session_id") != session_id:
            continue
        result.append(nudge)
    return result


def delete_nudge(nudge_id: str) -> bool:
    """Delete a nudge by ID. Returns True if found and deleted."""
    nudges = load_nudges()
    for i, nudge in enumerate(nudges):
        if nudge["id"] == nudge_id:
            nudges.pop(i)
            save_nudges(nudges)
            logger.info("Deleted nudge %s", nudge_id)
            return True
    return False


def delete_nudges_for_session(session_id: str) -> int:
    """Delete all nudges for a session. Returns count deleted."""
    nudges = load_nudges()
    original_count = len(nudges)
    nudges = [n for n in nudges if n.get("session_id") != session_id]
    deleted = original_count - len(nudges)
    if deleted > 0:
        save_nudges(nudges)
        logger.info("Deleted %d nudges for session %s", deleted, session_id)
    return deleted


# =============================================================================
# Nudge Firing
# =============================================================================

def get_due_nudges(now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Get all nudges that are due to fire."""
    if now is None:
        now = _hermes_now()

    nudges = load_nudges()
    due = []
    for nudge in nudges:
        if nudge.get("fired", False):
            continue
        try:
            fire_at = datetime.fromisoformat(nudge["fire_at"])
            if fire_at <= now:
                due.append(nudge)
        except (ValueError, TypeError):
            continue
    return due


def mark_nudge_fired(nudge_id: str) -> bool:
    """Mark a nudge as fired. Returns True if found.

    For recurring nudges, automatically reschedules for the next interval.
    """
    nudges = load_nudges()
    for nudge in nudges:
        if nudge["id"] == nudge_id:
            is_recurring = nudge.get("is_recurring", False)
            interval_seconds = nudge.get("interval_seconds")

            if is_recurring and interval_seconds:
                # Reschedule recurring nudge
                next_fire_at = _hermes_now() + timedelta(seconds=interval_seconds)
                nudge["fire_at"] = next_fire_at.isoformat()
                # Keep fired_at for tracking, but don't mark as permanently fired
                nudge["fired_at"] = _hermes_now().isoformat()
                nudge["fire_count"] = nudge.get("fire_count", 0) + 1
                logger.info(
                    "Rescheduled recurring nudge %s for %s",
                    nudge_id, next_fire_at.isoformat()
                )
            else:
                # One-time nudge - mark as permanently fired
                nudge["fired"] = True
                nudge["fired_at"] = _hermes_now().isoformat()

            save_nudges(nudges)
            return True
    return False


def fire_nudge(nudge: Dict[str, Any]) -> bool:
    """
    Fire a nudge - mark it fired and reschedule if recurring.

    For recurring nudges, the nudge is rescheduled for the next interval.
    For one-time nudges, the nudge is marked as permanently fired.

    Returns True if the nudge was successfully fired.
    """
    return mark_nudge_fired(nudge["id"])


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_old_nudges(max_age_hours: int = 24) -> int:
    """Remove fired nudges older than max_age_hours. Returns count removed."""
    if max_age_hours <= 0:
        return 0

    cutoff = _hermes_now() - timedelta(hours=max_age_hours)
    nudges = load_nudges()
    original_count = len(nudges)

    def should_keep(nudge):
        if not nudge.get("fired", False):
            return True
        fired_at = nudge.get("fired_at")
        if not fired_at:
            return True
        try:
            return datetime.fromisoformat(fired_at) > cutoff
        except (ValueError, TypeError):
            return True

    nudges = [n for n in nudges if should_keep(n)]
    removed = original_count - len(nudges)
    if removed > 0:
        save_nudges(nudges)
        logger.info("Cleaned up %d old fired nudges", removed)
    return removed
