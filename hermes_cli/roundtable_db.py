"""SQLite-backed Roundtable Discussion engine for multi-agent topic debates.

Independent database at ``<hermes_home>/roundtable.db`` — deliberately separate
from kanban.db to avoid schema pollution and transaction conflicts.

Schema: discussions, participants, speeches, findings, convergence_history.
WAL mode for concurrent-read friendliness.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_DISCUSSION_STATUSES = {"active", "concluded", "cancelled"}
VALID_SPEECH_ORDERS = {"fixed", "random", "priority", "free"}
VALID_FINDING_TYPES = {"consensus", "disagreement", "new_point"}

INITIATION_ROUND = 0  # Round 0 = opening statement by coordinator

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Discussion:
    id: str
    topic: str
    context: Optional[str]
    status: str
    max_rounds: int
    current_round: int
    speech_order: str
    created_by: str
    created_at: int
    concluded_at: Optional[int]
    conclusion: Optional[str]
    convergence_score: Optional[float]
    output_path: Optional[str]


@dataclass
class Participant:
    discussion_id: str
    participant: str
    role: Optional[str]
    perspective: Optional[str]
    display_name: Optional[str]
    joined_at: int
    is_active: bool


@dataclass
class Speech:
    id: int
    discussion_id: str
    round: int
    participant: str
    content: str
    reply_to: Optional[int]
    created_at: int


@dataclass
class Finding:
    id: int
    discussion_id: str
    type: str
    content: str
    round: int
    related_speeches: Optional[List[int]]


@dataclass
class ConvergenceRecord:
    discussion_id: str
    round: int
    score: float
    consensus_count: int
    disagreement_count: int
    new_point_count: int


# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS discussions (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    context TEXT,
    status TEXT DEFAULT 'active'
        CHECK(status IN ('active', 'concluded', 'cancelled')),
    max_rounds INTEGER DEFAULT 5,
    current_round INTEGER DEFAULT 0,
    speech_order TEXT DEFAULT 'fixed'
        CHECK(speech_order IN ('fixed', 'random', 'priority', 'free')),
    created_by TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    concluded_at INTEGER,
    conclusion TEXT,
    convergence_score REAL,
    output_path TEXT
);

CREATE TABLE IF NOT EXISTS participants (
    discussion_id TEXT NOT NULL,
    participant TEXT NOT NULL,
    role TEXT,
    perspective TEXT,
    display_name TEXT,
    joined_at INTEGER NOT NULL,
    is_active INTEGER DEFAULT 1,
    PRIMARY KEY (discussion_id, participant),
    FOREIGN KEY (discussion_id) REFERENCES discussions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS speeches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    participant TEXT NOT NULL,
    content TEXT NOT NULL,
    reply_to INTEGER,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (discussion_id) REFERENCES discussions(id) ON DELETE CASCADE,
    FOREIGN KEY (reply_to) REFERENCES speeches(id)
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    discussion_id TEXT NOT NULL,
    type TEXT NOT NULL
        CHECK(type IN ('consensus', 'disagreement', 'new_point')),
    content TEXT NOT NULL,
    round INTEGER NOT NULL,
    related_speeches TEXT,
    FOREIGN KEY (discussion_id) REFERENCES discussions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS convergence_history (
    discussion_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    score REAL NOT NULL,
    consensus_count INTEGER,
    disagreement_count INTEGER,
    new_point_count INTEGER,
    PRIMARY KEY (discussion_id, round),
    FOREIGN KEY (discussion_id) REFERENCES discussions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_speeches_discussion
    ON speeches(discussion_id, round);
CREATE INDEX IF NOT EXISTS idx_speeches_participant
    ON speeches(discussion_id, participant);
CREATE INDEX IF NOT EXISTS idx_findings_discussion
    ON findings(discussion_id, type);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()


def _roundtable_db_path() -> Path:
    """Resolve the roundtable DB path.

    Resolution order:
    1. ``HERMES_ROUNDTABLE_DB`` env var (explicit override)
    2. ``<hermes_home>/roundtable.db`` (default)
    """
    env_path = os.environ.get("HERMES_ROUNDTABLE_DB")
    if env_path:
        return Path(env_path)
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "roundtable.db"


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the roundtable DB.

    Auto-runs schema creation on first connection to a given path.
    WAL mode + foreign keys enabled on every connection.
    """
    path = db_path or _roundtable_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    needs_init = resolved not in _INITIALIZED_PATHS
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30)
    conn.row_factory = sqlite3.Row
    from hermes_state import apply_wal_with_fallback
    apply_wal_with_fallback(conn, db_label=f"roundtable.db ({path.name})")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    if needs_init:
        conn.executescript(SCHEMA_SQL)
        _INITIALIZED_PATHS.add(resolved)
    return conn


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def _generate_discussion_id() -> str:
    """Generate a discussion ID: ``rt_`` + 8 hex chars."""
    return f"rt_{secrets.token_hex(4)}"


# ---------------------------------------------------------------------------
# CRUD: Discussions
# ---------------------------------------------------------------------------


def create_discussion(
    conn: sqlite3.Connection,
    topic: str,
    participants: List[Dict[str, Any]],
    *,
    context: Optional[str] = None,
    max_rounds: int = 5,
    speech_order: str = "fixed",
    created_by: str = "unknown",
    output_path: Optional[str] = None,
) -> Discussion:
    """Create a new discussion and register participants.

    Returns the newly created Discussion object.
    """
    if speech_order not in VALID_SPEECH_ORDERS:
        raise ValueError(f"Invalid speech_order: {speech_order}")
    if max_rounds < 1:
        raise ValueError("max_rounds must be >= 1")
    if not participants:
        raise ValueError("At least one participant is required")

    disc_id = _generate_discussion_id()
    now = int(time.time())

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT INTO discussions
               (id, topic, context, status, max_rounds, current_round,
                speech_order, created_by, created_at, output_path)
               VALUES (?, ?, ?, 'active', ?, 0, ?, ?, ?, ?)""",
            (disc_id, topic, context, max_rounds, speech_order,
             created_by, now, output_path),
        )
        for p in participants:
            profile = p.get("profile", "").strip()
            if not profile:
                raise ValueError("Each participant must have a 'profile' field")
            conn.execute(
                """INSERT INTO participants
                   (discussion_id, participant, role, perspective,
                    display_name, joined_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, 1)""",
                (
                    disc_id,
                    profile,
                    p.get("role"),
                    p.get("perspective"),
                    p.get("display_name"),
                    now,
                ),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return Discussion(
        id=disc_id,
        topic=topic,
        context=context,
        status="active",
        max_rounds=max_rounds,
        current_round=0,
        speech_order=speech_order,
        created_by=created_by,
        created_at=now,
        concluded_at=None,
        conclusion=None,
        convergence_score=None,
        output_path=output_path,
    )


def get_discussion(
    conn: sqlite3.Connection, discussion_id: str
) -> Optional[Discussion]:
    """Fetch a discussion by ID."""
    row = conn.execute(
        "SELECT * FROM discussions WHERE id = ?", (discussion_id,)
    ).fetchone()
    if not row:
        return None
    return Discussion(
        id=row["id"],
        topic=row["topic"],
        context=row["context"],
        status=row["status"],
        max_rounds=row["max_rounds"],
        current_round=row["current_round"],
        speech_order=row["speech_order"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        concluded_at=row["concluded_at"],
        conclusion=row["conclusion"],
        convergence_score=row["convergence_score"],
        output_path=row["output_path"],
    )


def list_discussions(
    conn: sqlite3.Connection,
    *,
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Discussion]:
    """List discussions, optionally filtered by status."""
    if status:
        rows = conn.execute(
            "SELECT * FROM discussions WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM discussions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        Discussion(
            id=r["id"], topic=r["topic"], context=r["context"],
            status=r["status"], max_rounds=r["max_rounds"],
            current_round=r["current_round"], speech_order=r["speech_order"],
            created_by=r["created_by"], created_at=r["created_at"],
            concluded_at=r["concluded_at"], conclusion=r["conclusion"],
            convergence_score=r["convergence_score"],
            output_path=r["output_path"],
        )
        for r in rows
    ]


def conclude_discussion(
    conn: sqlite3.Connection,
    discussion_id: str,
    *,
    conclusion: Optional[str] = None,
    convergence_score: Optional[float] = None,
) -> bool:
    """Mark a discussion as concluded.

    Returns True if the status was actually changed.
    """
    now = int(time.time())
    cur = conn.execute(
        """UPDATE discussions
           SET status = 'concluded', concluded_at = ?,
               conclusion = COALESCE(?, conclusion),
               convergence_score = COALESCE(?, convergence_score)
           WHERE id = ? AND status = 'active'""",
        (now, conclusion, convergence_score, discussion_id),
    )
    return cur.rowcount > 0


def cancel_discussion(conn: sqlite3.Connection, discussion_id: str) -> bool:
    """Cancel an active discussion.

    Returns True if the status was actually changed.
    """
    now = int(time.time())
    cur = conn.execute(
        """UPDATE discussions
           SET status = 'cancelled', concluded_at = ?
           WHERE id = ? AND status = 'active'""",
        (now, discussion_id),
    )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# CRUD: Participants
# ---------------------------------------------------------------------------


def get_participants(
    conn: sqlite3.Connection, discussion_id: str
) -> List[Participant]:
    """Get all participants for a discussion."""
    rows = conn.execute(
        """SELECT * FROM participants
           WHERE discussion_id = ? ORDER BY joined_at""",
        (discussion_id,),
    ).fetchall()
    return [
        Participant(
            discussion_id=r["discussion_id"],
            participant=r["participant"],
            role=r["role"],
            perspective=r["perspective"],
            display_name=r["display_name"],
            joined_at=r["joined_at"],
            is_active=bool(r["is_active"]),
        )
        for r in rows
    ]


def get_active_participant_names(
    conn: sqlite3.Connection, discussion_id: str
) -> List[str]:
    """Get ordered list of active participant profile names."""
    rows = conn.execute(
        """SELECT participant FROM participants
           WHERE discussion_id = ? AND is_active = 1
           ORDER BY joined_at""",
        (discussion_id,),
    ).fetchall()
    return [r["participant"] for r in rows]


# ---------------------------------------------------------------------------
# CRUD: Speeches
# ---------------------------------------------------------------------------


def add_speech(
    conn: sqlite3.Connection,
    discussion_id: str,
    participant: str,
    content: str,
    *,
    reply_to: Optional[int] = None,
    round_override: Optional[int] = None,
) -> Speech:
    """Record a speech and potentially advance the round.

    Returns the created Speech with auto-assigned round.

    Round logic:
    - Round 0 = opening (anyone can speak in round 0)
    - When all active participants have spoken in the current round,
      current_round advances by 1.
    - If current_round exceeds max_rounds after advancing, the discussion
      is auto-concluded.
    - round_override forces the speech into a specific round (e.g. 0 for
      coordinator) and skips round-advancement checks for that speech.
    """
    disc = get_discussion(conn, discussion_id)
    if not disc:
        raise ValueError(f"Discussion {discussion_id} not found")
    if disc.status != "active":
        raise ValueError(f"Discussion {discussion_id} is {disc.status}")

    now = int(time.time())
    current_round = disc.current_round
    speech_round = round_override if round_override is not None else current_round

    # Validate reply_to if provided
    if reply_to is not None:
        ref = conn.execute(
            "SELECT id FROM speeches WHERE id = ? AND discussion_id = ?",
            (reply_to, discussion_id),
        ).fetchone()
        if not ref:
            raise ValueError(
                f"reply_to speech {reply_to} not found in discussion {discussion_id}"
            )

    conn.execute("BEGIN IMMEDIATE")
    try:
        cur = conn.execute(
            """INSERT INTO speeches
               (discussion_id, round, participant, content, reply_to, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (discussion_id, speech_round, participant, content, reply_to, now),
        )
        speech_id = cur.lastrowid

        # Check if round should advance:
        # Only check for normal (non-overridden) speeches — overridden speeches
        # (e.g. coordinator) don't participate in round progression.
        active_names = get_active_participant_names(conn, discussion_id)
        round_complete = False
        discussion_complete = False

        if round_override is None:
            # All active participants have spoken in the current round
            speakers_this_round = conn.execute(
                """SELECT DISTINCT participant FROM speeches
                   WHERE discussion_id = ? AND round = ?""",
                (discussion_id, current_round),
            ).fetchall()
            spoke_names = {r["participant"] for r in speakers_this_round}

            round_complete = all(name in spoke_names for name in active_names)

        if round_complete and current_round >= 0:
            # Advance round (but not beyond max_rounds if auto-conclude)
            new_round = current_round + 1
            conn.execute(
                "UPDATE discussions SET current_round = ? WHERE id = ?",
                (new_round, discussion_id),
            )
            # Auto-conclude if exceeded max_rounds (round 0 doesn't count)
            if new_round > disc.max_rounds:
                conn.execute(
                    """UPDATE discussions
                       SET status = 'concluded', concluded_at = ?
                       WHERE id = ? AND status = 'active'""",
                    (now, discussion_id),
                )
                discussion_complete = True

        # Determine next speaker
        next_speaker = None
        if not discussion_complete and active_names:
            if disc.speech_order == "fixed":
                # Next in the participant list who hasn't spoken in new round
                target_round = disc.current_round
                if round_complete:
                    target_round = disc.current_round  # already advanced
                speakers_next = conn.execute(
                    """SELECT DISTINCT participant FROM speeches
                       WHERE discussion_id = ? AND round = ?""",
                    (discussion_id, target_round),
                ).fetchall()
                spoke_next = {r["participant"] for r in speakers_next}
                for name in active_names:
                    if name not in spoke_next:
                        next_speaker = name
                        break

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return Speech(
        id=speech_id or 0,  # lastrowid is int for AUTOINCREMENT
        discussion_id=discussion_id,
        round=speech_round,
        participant=participant,
        content=content,
        reply_to=reply_to,
        created_at=now,
    )


def get_speeches(
    conn: sqlite3.Connection,
    discussion_id: str,
    *,
    since_round: Optional[int] = None,
    participant: Optional[str] = None,
) -> List[Speech]:
    """Read speeches for a discussion with optional filters."""
    query = "SELECT * FROM speeches WHERE discussion_id = ?"
    params: list = [discussion_id]

    if since_round is not None:
        query += " AND round >= ?"
        params.append(since_round)
    if participant:
        query += " AND participant = ?"
        params.append(participant)

    query += " ORDER BY id ASC"
    rows = conn.execute(query, params).fetchall()

    return [
        Speech(
            id=r["id"],
            discussion_id=r["discussion_id"],
            round=r["round"],
            participant=r["participant"],
            content=r["content"],
            reply_to=r["reply_to"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def get_speech_count(
    conn: sqlite3.Connection, discussion_id: str
) -> int:
    """Count total speeches in a discussion."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM speeches WHERE discussion_id = ?",
        (discussion_id,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# CRUD: Findings
# ---------------------------------------------------------------------------


def add_finding(
    conn: sqlite3.Connection,
    discussion_id: str,
    finding_type: str,
    content: str,
    round_num: int,
    related_speeches: Optional[List[int]] = None,
) -> int:
    """Record a finding (consensus, disagreement, new_point).

    Returns the finding ID.
    """
    if finding_type not in VALID_FINDING_TYPES:
        raise ValueError(f"Invalid finding type: {finding_type}")

    rs_json = json.dumps(related_speeches) if related_speeches else None
    cur = conn.execute(
        """INSERT INTO findings
           (discussion_id, type, content, round, related_speeches)
           VALUES (?, ?, ?, ?, ?)""",
        (discussion_id, finding_type, content, round_num, rs_json),
    )
    return cur.lastrowid or 0


def get_findings(
    conn: sqlite3.Connection,
    discussion_id: str,
    *,
    finding_type: Optional[str] = None,
) -> List[Finding]:
    """Get findings for a discussion, optionally filtered by type."""
    if finding_type:
        rows = conn.execute(
            """SELECT * FROM findings
               WHERE discussion_id = ? AND type = ?
               ORDER BY id ASC""",
            (discussion_id, finding_type),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM findings WHERE discussion_id = ? ORDER BY id ASC",
            (discussion_id,),
        ).fetchall()

    return [
        Finding(
            id=r["id"],
            discussion_id=r["discussion_id"],
            type=r["type"],
            content=r["content"],
            round=r["round"],
            related_speeches=json.loads(r["related_speeches"])
            if r["related_speeches"]
            else None,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# CRUD: Convergence
# ---------------------------------------------------------------------------


def record_convergence(
    conn: sqlite3.Connection,
    discussion_id: str,
    round_num: int,
    score: float,
    consensus_count: int,
    disagreement_count: int,
    new_point_count: int,
) -> None:
    """Record convergence metrics for a round."""
    conn.execute(
        """INSERT OR REPLACE INTO convergence_history
           (discussion_id, round, score, consensus_count,
            disagreement_count, new_point_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (discussion_id, round_num, score, consensus_count,
         disagreement_count, new_point_count),
    )


def get_convergence_history(
    conn: sqlite3.Connection, discussion_id: str
) -> List[ConvergenceRecord]:
    """Get convergence history for a discussion."""
    rows = conn.execute(
        """SELECT * FROM convergence_history
           WHERE discussion_id = ? ORDER BY round ASC""",
        (discussion_id,),
    ).fetchall()
    return [
        ConvergenceRecord(
            discussion_id=r["discussion_id"],
            round=r["round"],
            score=r["score"],
            consensus_count=r["consensus_count"],
            disagreement_count=r["disagreement_count"],
            new_point_count=r["new_point_count"],
        )
        for r in rows
    ]
