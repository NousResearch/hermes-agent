#!/usr/bin/env python3
# EDIT HISTORY (newest first)
# 2026-08-31 | claude-fable-5 | anthropic | desktop session | v2.4.0 bot enrollment audit: unenrolled_bot_profiles() scans <profiles>/*/SOUL.md for the BOT_WIRE_MARKER ("session-coord (bot-wire v1)", shipped inside the SOUL.md blurb install.py now appends); cmd_status lists persona-bearing profiles missing it as "— UNENROLLED bot profiles" + unenrolled_bot_profiles in --json, naming the fix. Read-only, fail-open; SOUL-less profiles never flagged (nothing proves they are bots). Companion unwired_profiles(): SOUL-less profiles whose OWN memories/MEMORY.md exists without the standing rule ("session-coord (wire v1)") -> UNWIRED warning + unwired_profiles in --json (store-less fresh profiles never flagged — no evidence; bots are the blurb audit's job). Closes the invisible-unenrolled-actor gap: enrollment is per-persona/per-profile-store and manual post-install, and a missed paste previously surfaced only as a collision. selftest_cron.sh 45->51.  # noqa: E501
# 2026-08-26 | claude-opus-4-8 | anthropic | desktop session | MASTER SWITCH: whole coordination layer can be turned OFF without uninstalling — `enable`/`disable`/`switch [on|off|toggle]` verbs (persistent sentinel ~/.hermes/state/coordination_disabled; env HERMES_COORD_DISABLED overrides both directions). While OFF every verb is a fail-open no-op (register still prints a synthetic 12-hex id, claim/check succeed FREE, cron-guard stdout stays empty) so a disabled board == "not installed"; switch verbs themselves always run. coordination_enabled()/switch_source() helpers; main() gate short-circuits to disabled_noop(). Mirrors coord_guard.sh _coord_disabled(). 90/90 existing checks still green (enabled path byte-identical) + selftest_toggle.sh 28 new checks.  # noqa: E501
# 2026-08-18 | claude-fable-5 | anthropic | desktop session | v2.2 Bot Mode integration: cron_store_jobs() now merges EVERY profile store (~/.hermes/profiles/*/cron/jobs.json — a Bot Mode bot IS a profile; its Routines live in its own store) with the default store (id collision: default wins); profile jobs get a "[bot:<profile>]" name tag surfaced through radar/advisories/guard/wait-for-cron; upcoming_cron_conflicts + status rows carry "profile"; resolve_cron_job error names both store locations; HERMES_COORD_PROFILES_DIR override for tests  # noqa: E501
# 2026-08-16 | claude-fable-5 | anthropic | desktop session | v2.1 cron leg: cron-guard verb (deterministic wrapper for cron scripts: register ephemeral cron session + atomic claim; --policy skip|wait; stdout carries ONLY the coordination id so no_agent stdout-as-message semantics survive; holders get a cron_defer note on skip), cron manifest advisory (~/.hermes/state/cron_resources.json + jobs.json next_run_at -> "cron fires in ~Nm on this resource" warnings on claim/check/status; advisory only, never blocks), cron-aware verbs (HELD labels cron holders, preempt refuses vs crons — they cannot checkpoint/pause; wait or user-approved steal)  # noqa: E501
# 2026-08-16 | claude-fable-5 | anthropic | desktop session | v2: user-set priority ranks (1,2,3 + reorder), subagent lineage ranks (1a/1b via --parent/--slot, computed live so re-ranking a parent re-ranks its children), preempt protocol (user-priority request -> holder checkpoints+pauses -> auto-queued resume), pause/resume verbs, queue fencing (rank order + FIFO enforced on contended resources, liveness-gated), preempt alerts on every board touch, idempotent schema migration from v1. All v1 commands/outputs byte-compatible (19-test regression suite).  # noqa: E501
# 2026-08-16 | claude-fable-5 | anthropic | desktop session | Initial build: cooperative multi-session registry (SQLite WAL; sessions/claims/waiters/notifications; atomic all-or-nothing batch claims; shared|exclusive modes; path-boundary dir conflicts; TTL reap w/ expired-holder warnings; wait->notify co-worker protocol; exit 75 = held, matching singleflight convention)  # noqa: E501
"""
session_coord.py — cooperative coordination registry for concurrent Hermes sessions.

Sessions treat each other as CO-WORKERS: check before mutating shared resources,
wait politely when a resource is held, notify waiters when done, release at TASK
end (not per-write). Advisory protocol — enforced by standing memory rule +
skill 'multi-session-coordination', not by the kernel.

v2 adds USER PRIORITY: the user can rank sessions ("do 1, then 2, then 3"),
re-rank anytime, and a ranked session can send a preempt request that asks the
holder to checkpoint, pause, and auto-resume later. Subagents register with
--parent/--slot and inherit family rank (1a, 1b) — compared ONLY on collision.

DB: ~/.hermes/state/session_coordination.db  (override: HERMES_COORD_DB)

Resource namespaces (freeform keys; these conventions are documented):
  file:/abs/path   files or directories (claims on a dir cover everything under it)
  skill:<name>     a skill being EDITED (writers claim exclusive; readers `wait`)
  memory           the Hermes memory store (hygiene work claims this singleton)
  ui:desktop       control of the Mac desktop / foreground UI
  box:<host>       a remote machine for mutating work (e.g. box:gpu-box-1)
  cron-store       the cron jobs table
  res:<anything>   custom

Exit codes: 0 = ok/free/acquired · 75 = held/queued behind a co-worker
(EX_TEMPFAIL, same convention as singleflight.sh) · 1 = error.

Typical flow:
  session_coord.py status
  ID=$(session_coord.py register --task "memory hygiene sweep" --surface desktop)
  session_coord.py claim --id $ID --res memory --res "file:~/.hermes/skills" \
      --task "memory hygiene" [--wait --timeout 300]
  ... do the whole task ...
  session_coord.py done --id $ID          # releases everything + notifies waiters

Priority flow (user says "this session first"):
  session_coord.py prioritize --session <me> --rank 1        # record user's call
  session_coord.py preempt --id <me> --res memory            # ask holder to pause
  # holder (in its own session): checkpoint work, then:
  session_coord.py pause --id <holder> --note "ledger at ~/x, step 3 done"
  # requester claims, works, releases; holder gets notified, then:
  session_coord.py resume --id <holder> [--wait]
"""

import argparse
import contextlib
import glob
import json
import os
import re
import sqlite3
import sys
import time
import uuid


def _console_safe_stdio():
    """Make stdout/stderr never raise UnicodeEncodeError on narrow consoles
    (Windows cp1252, C-locale cron): keep the stream's own encoding but
    degrade unencodable chars (e.g. \u26a0) to '?' instead of crashing.
    JSON mode is already pure ASCII (json.dumps ensure_ascii=True)."""
    for stream in (sys.stdout, sys.stderr):
        # exotic stream wrapper — leave as-is
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")


_console_safe_stdio()

DB_PATH = os.path.expanduser(
    os.environ.get("HERMES_COORD_DB", "~/.hermes/state/session_coordination.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY,
    task TEXT,
    surface TEXT,
    started_at REAL,
    last_seen REAL,
    status TEXT DEFAULT 'active'
);
CREATE TABLE IF NOT EXISTS claims(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    resource TEXT,
    mode TEXT DEFAULT 'exclusive',
    task TEXT,
    claimed_at REAL,
    ttl_min REAL,
    released_at REAL,
    status TEXT DEFAULT 'held'
);
CREATE TABLE IF NOT EXISTS waiters(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    resource TEXT,
    since REAL,
    note TEXT,
    active INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS notifications(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    to_session TEXT,
    from_session TEXT,
    resource TEXT,
    kind TEXT,
    body TEXT,
    created_at REAL,
    read_at REAL
);
CREATE TABLE IF NOT EXISTS cron_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    job_name TEXT,
    event TEXT,           -- guarded-run | skipped | wait-timeout |
                          -- user-paused | user-resumed | user-triggered
    session_id TEXT,      -- guard session, or the interactive session acting on the job
    reason TEXT,
    created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_claims_res ON claims(resource, status);
CREATE INDEX IF NOT EXISTS idx_notif_to ON notifications(to_session, read_at);
CREATE INDEX IF NOT EXISTS idx_cronev_job ON cron_events(job_id, created_at);
"""

# v2 columns added to the v1 schema (idempotent migration; ADD COLUMN only —
# never mutates or drops v1 data, so a v1 DB upgrades in place losslessly).
MIGRATIONS = {
    "sessions": [
        ("priority", "TEXT"),            # user-set root rank: '1', '2', '1a'
        ("parent_id", "TEXT"),           # coordination id of parent session
        ("slot", "TEXT"),                # sub-priority under parent: 'a','b',...
        ("paused", "INTEGER DEFAULT 0"),
        ("checkpoint_note", "TEXT"),
        ("rank_set_at", "REAL"),
    ],
    "claims": [
        ("paused_at", "REAL"),
    ],
    "waiters": [
        ("mode", "TEXT"),                # claim mode the waiter intends
        ("last_poll", "REAL"),           # freshness: only actively-polling
                                         # waiters fence (paused rows exempt)
    ],
}

DEFAULT_TTL_MIN = 90.0
STALE_SESSION_S = 24 * 3600
EXPIRED_KEEP_S = 7 * 24 * 3600
POLL_FRESH_S = 30               # a waiter fences only while actively polling
                                # (claim --wait touches last_poll every ~4s);
                                # paused resume spots are exempt. Keeps queue
                                # order across release without letting an
                                # abandoned waiter row fence a free resource.
INF = float("inf")
PAUSE_NOTE = "PAUSED-RESUME"

# --- master switch --------------------------------------------------------
# The whole coordination layer can be turned OFF without uninstalling it.
# Precedence (this CLI and the shell cron guard honor it identically):
#   1. env HERMES_COORD_DISABLED -- explicit override in BOTH directions
#      (truthy -> OFF; 0/false/no/off -> ON), for one-shot/session/test use.
#   2. sentinel file (HERMES_COORD_DISABLED_FILE, default below) -- the
#      persistent switch set by the `disable`/`enable`/`switch` verbs.
#   3. default: ENABLED.
# OFF = every verb is a friendly no-op that never blocks a caller (register
# still prints a synthetic id; claim/check/cron-guard return success/free), so
# a disabled board behaves exactly like "no coordination installed".
COORD_DISABLED_FILE = os.path.expanduser(
    os.environ.get("HERMES_COORD_DISABLED_FILE",
                   "~/.hermes/state/coordination_disabled")
)
_SWITCH_FALSY = {"0", "false", "no", "off"}

# --- cron awareness -------------------------------------------------------
# Manifest of scheduled jobs' declared resources (advisory heads-up for
# sessions + resolution source for cron-guard --job). Optional; absence
# changes nothing.
CRON_MANIFEST = os.path.expanduser(
    os.environ.get("HERMES_COORD_CRON_MANIFEST", "~/.hermes/state/cron_resources.json")
)
CRON_JOBS_JSON = os.path.expanduser(
    os.environ.get("HERMES_COORD_CRON_JOBS", "~/.hermes/cron/jobs.json")
)
# Bot Mode (desktop hermes-bots plugin): a bot IS a Hermes profile; its
# Routines are cron jobs in the profile's OWN store. The radar scans them all.
CRON_PROFILES_DIR = os.path.expanduser(
    os.environ.get("HERMES_COORD_PROFILES_DIR", "~/.hermes/profiles")
)
CRON_ADVISORY_S = 90 * 60       # claim/check warn horizon: fires within 90 min
CRON_STATUS_S = 12 * 3600       # status shows manifested fires within 12 h

# Bot ENROLLMENT audit marker: the SOUL.md coordination blurb (shipped in
# templates/bot-soul-coordination.md, appended by install.py's bot-wiring
# step) carries this line. A profile that HAS a persona (SOUL.md) but lacks
# the marker never enrolled in the protocol — its runs are invisible to the
# board. `status` surfaces those so an unenrolled bot can't stay unnoticed
# until a collision. Keep byte-stable and in lock-step with install.py.
BOT_WIRE_MARKER = "session-coord (bot-wire v1)"
# Session-enrollment marker (the standing memory rule install.py writes to
# the main store AND to each non-bot profile's own memories/MEMORY.md — a
# profile is a full agent instance whose sessions never see the main store).
WIRE_MARKER = "session-coord (wire v1)"


def unenrolled_bot_profiles():
    """Profile names with a SOUL.md that lacks BOT_WIRE_MARKER (persona-bearing
    = bot-like; profiles without a persona are skipped — nothing proves they
    are bots). Read-only, fail-open: unreadable files contribute nothing."""
    out = []
    for soul in sorted(glob.glob(os.path.join(CRON_PROFILES_DIR, "*", "SOUL.md"))):
        try:
            with open(soul, encoding="utf-8", errors="replace") as f:
                if BOT_WIRE_MARKER not in f.read():
                    out.append(os.path.basename(os.path.dirname(soul)))
        except OSError:
            continue
    return out


def unwired_profiles():
    """Non-bot profile names whose OWN memory store exists but lacks the
    standing rule (WIRE_MARKER) — their sessions never consult the board.
    Only flags profiles with an actual memories/MEMORY.md: an existing store
    proves agent sessions run there, while a fresh profile with no store yet
    is no evidence (escalate on confirmation only). Bot profiles (SOUL.md
    present) are the blurb audit's job, not this one's. Read-only, fail-open."""
    out = []
    for mem in sorted(glob.glob(os.path.join(CRON_PROFILES_DIR, "*",
                                             "memories", "MEMORY.md"))):
        prof_dir = os.path.dirname(os.path.dirname(mem))
        if os.path.exists(os.path.join(prof_dir, "SOUL.md")):
            continue
        try:
            with open(mem, encoding="utf-8", errors="replace") as f:
                if WIRE_MARKER not in f.read():
                    out.append(os.path.basename(prof_dir))
        except OSError:
            continue
    return out


def db():
    """Open the coordination DB (WAL), create schema if missing, apply
    additive column migrations. Fail-open callers catch OperationalError."""
    d = os.path.dirname(DB_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # manual transaction control
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():
        # nosec B608 — `table` comes from hardcoded MIGRATIONS keys, never user input
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, decl in cols:
            if name not in have:
                # nosec B608 — identifiers from the hardcoded MIGRATIONS dict
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    return conn


def _canon_path(p):
    """realpath + expanduser + normcase: one canonical spelling per file.
    normcase lowercases on Windows only (case-insensitive FS), identity on
    POSIX — existing boards are unaffected on macOS/Linux."""
    return os.path.normcase(
        os.path.realpath(os.path.expanduser(p))).rstrip(os.sep)


def norm_resource(r):
    r"""Normalize a resource key. File-ish keys become file:<canonical path>.
    Accepts file:/dir: prefixes, ~ and POSIX absolutes, and Windows
    drive-letter/UNC absolutes (C:\..., \\server\share)."""
    r = r.strip()
    if r.startswith(("file:", "dir:")):
        return "file:" + _canon_path(r.split(":", 1)[1])
    if r.startswith(("/", "~")) or os.path.isabs(r):
        return "file:" + _canon_path(r)
    return r


def resources_overlap(a, b):
    """True when two resource keys denote the same/overlapping thing.
    file: keys use path-boundary prefix logic so a dir claim covers children."""
    if a == b:
        return True
    if a.startswith("file:") and b.startswith("file:"):
        pa, pb = a[5:], b[5:]
        return pa.startswith(pb + os.sep) or pb.startswith(pa + os.sep)
    return False


def modes_conflict(m1, m2):
    """True if an existing claim in mode `a` blocks a new claim in mode `b`
    (exclusive conflicts with everything; shared coexists with shared)."""
    return not ((m1 or "exclusive") == "shared" and (m2 or "exclusive") == "shared")


def now():
    """Current epoch seconds (single clock source for the whole tool)."""
    return time.time()


def short(sid):
    """First 8 chars of a session id — human-readable board labels."""
    return (sid or "?")[:8]


def age_str(ts):
    """Seconds -> compact human age like '4m' or '1h07m'."""
    s = int(now() - ts)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def coordination_enabled():
    """Master switch state. env HERMES_COORD_DISABLED (explicit, both
    directions) overrides the sentinel file; absent both -> enabled. Kept in
    lock-step with _coord_disabled() in coord_guard.sh -- change both together."""
    env = os.environ.get("HERMES_COORD_DISABLED")
    if env is not None and env.strip():
        return env.strip().lower() in _SWITCH_FALSY
    return not os.path.exists(COORD_DISABLED_FILE)


def switch_source():
    """Human explanation of what is currently deciding the master switch."""
    env = os.environ.get("HERMES_COORD_DISABLED")
    if env is not None and env.strip():
        return f"env HERMES_COORD_DISABLED={env.strip()!r}"
    if os.path.exists(COORD_DISABLED_FILE):
        return f"sentinel file {COORD_DISABLED_FILE}"
    return "default (enabled)"


# ---------------------------------------------------------------- ranks

def parse_root_rank(text):
    """'1' -> (1,); '1a' -> (1,'a'); invalid/empty -> (INF,)."""
    if not text:
        return (INF,)
    m = re.match(r"^(\d{1,4})([a-z0-9]{0,3})$", str(text).strip().lower())
    if not m:
        return (INF,)
    t = (int(m.group(1)),)
    if m.group(2):
        t += (m.group(2),)
    return t


def eff_rank(conn, sid, depth=0):
    """Effective rank tuple, computed LIVE so re-ranking a parent instantly
    re-ranks its children. Root: (int, [suffix]). Child: parent_rank + (slot,).
    Unranked -> (INF,). Lower tuple = higher priority. Parent beats own child
    (prefix tuples sort first); ranked families beat unranked sessions."""
    if depth > 5 or not sid:
        return (INF,)
    row = conn.execute(
        "SELECT priority,parent_id,slot FROM sessions WHERE id=?", (sid,)
    ).fetchone()
    if not row:
        return (INF,)
    if row["parent_id"]:
        return (*eff_rank(conn, row["parent_id"], depth + 1), (row["slot"] or "~"))
    return parse_root_rank(row["priority"])


def rank_lt(a, b):
    """True if rank string a outranks (sorts before) b. Lexicographic on
    (numeric part, letter suffix): '1' < '1a' < '1b' < '2'; None ranks last."""
    try:
        return a < b
    except TypeError:
        return False


def rank_str(t):
    """Rank for display: the string itself, or an em-dash when unranked."""
    if not t or t[0] == INF:
        base = "—"
        rest = [x for x in t[1:] if x != "~"] if t else []
        return base + "".join(str(x) for x in rest) if rest else base
    return str(int(t[0])) + "".join(str(x) for x in t[1:] if x != "~")


def resolve_session(conn, token):
    """Resolve a full id or unique prefix (>=4 chars) to a session id."""
    token = (token or "").strip()
    if not token:
        return None, "empty session id"
    rows = conn.execute(
        "SELECT id FROM sessions WHERE id LIKE ? ORDER BY started_at DESC",
        (token + "%",),
    ).fetchall()
    if not rows:
        return None, f"no session matches '{token}'"
    ids = [r["id"] for r in rows]
    if token in ids:
        return token, None
    if len(ids) > 1:
        return None, f"ambiguous prefix '{token}': {', '.join(short(i) for i in ids)}"
    return ids[0], None


# ---------------------------------------------------------------- db helpers

def touch_session(conn, sid):
    """Update last_seen for liveness tracking (called on every verb)."""
    if sid:
        conn.execute("UPDATE sessions SET last_seen=? WHERE id=?", (now(), sid))


def notify(conn, to_session, from_session, resource, kind, body, dedupe=False):
    """Queue an inbox message for another session (delivered on its next
    inbox/board call). Best-effort: never raises."""
    if dedupe:
        dup = conn.execute(
            "SELECT id FROM notifications WHERE to_session=? AND from_session=? "
            "AND resource=? AND kind=? AND read_at IS NULL",
            (to_session, from_session, resource, kind),
        ).fetchone()
        if dup:
            return
    conn.execute(
        "INSERT INTO notifications(to_session,from_session,resource,kind,body,created_at)"
        " VALUES(?,?,?,?,?,?)",
        (to_session, from_session, resource, kind, body, now()),
    )


def alert_pending(conn, sid):
    """Surface urgent unread notifications on ANY board touch (stderr, so JSON
    stdout stays clean). This is how a busy holder learns of a preempt request
    without polling inbox."""
    if not sid:
        return
    n = conn.execute(
        "SELECT COUNT(*) c FROM notifications WHERE to_session=? AND read_at IS NULL "
        "AND kind IN ('preempt_request','priority')",
        (sid,),
    ).fetchone()["c"]
    if n:
        print(
            f"⚠ {n} urgent notification(s) pending (preempt/priority) — run: "
            f"session_coord.py inbox --id {sid}",
            file=sys.stderr,
        )


def reap(conn):
    """Expire overdue claims (warn their waiters), stale sessions, dead-session
    waiters, old rows. Caller must hold a write transaction."""
    t = now()
    expired = conn.execute(
        "SELECT * FROM claims WHERE status='held' AND claimed_at + ttl_min*60 < ?", (t,)
    ).fetchall()
    for c in expired:
        conn.execute(
            "UPDATE claims SET status='expired', released_at=? WHERE id=?", (t, c["id"])
        )
        for w in conn.execute(
            "SELECT * FROM waiters WHERE active=1 AND session_id != ?", (c["session_id"],)
        ).fetchall():
            if resources_overlap(norm_resource(w["resource"]), c["resource"]):
                notify(
                    conn, w["session_id"], c["session_id"], c["resource"], "expired",
                    f"Claim on {c['resource']} EXPIRED (holder {short(c['session_id'])}, "
                    f"task '{c['task']}', held {age_str(c['claimed_at'])}). Holder may have "
                    f"died mid-task — VERIFY the resource state before proceeding.",
                )
                conn.execute("UPDATE waiters SET active=0 WHERE id=?", (w["id"],))
    # paused claims expire too (holder loses its queue spot; owner is warned)
    pexp = conn.execute(
        "SELECT * FROM claims WHERE status='paused' AND "
        "COALESCE(paused_at, claimed_at) + ttl_min*60 < ?", (t,)
    ).fetchall()
    for c in pexp:
        conn.execute(
            "UPDATE claims SET status='expired', released_at=? WHERE id=?", (t, c["id"])
        )
        notify(
            conn, c["session_id"], None, c["resource"], "expired",
            f"Your PAUSED claim on {c['resource']} expired after "
            f"{int(c['ttl_min'])}m — your resume queue spot lapsed. Re-claim manually "
            f"and VERIFY resource state (others may have modified it).",
        )
        conn.execute(
            "UPDATE waiters SET active=0 WHERE session_id=? AND resource=? AND note=?",
            (c["session_id"], c["resource"], PAUSE_NOTE),
        )
    conn.execute(
        "UPDATE sessions SET status='stale' WHERE status='active' AND last_seen < ?",
        (t - STALE_SESSION_S,),
    )
    conn.execute(
        "UPDATE waiters SET active=0 WHERE active=1 AND session_id IN "
        "(SELECT id FROM sessions WHERE status IN ('done','stale'))"
    )
    conn.execute(
        "DELETE FROM claims WHERE status IN ('released','expired','stolen') AND released_at < ?",
        (t - EXPIRED_KEEP_S,),
    )
    conn.execute(
        "DELETE FROM notifications WHERE read_at IS NOT NULL AND created_at < ?",
        (t - EXPIRED_KEEP_S,),
    )


def holders_of(conn, resource, mode, exclude_session):
    """Held claims that conflict with (resource, mode), excluding our own.
    Paused claims never block (that's the point of pausing)."""
    out = []
    for c in conn.execute("SELECT * FROM claims WHERE status='held'").fetchall():
        if c["session_id"] == exclude_session:
            continue
        if resources_overlap(c["resource"], resource) and modes_conflict(c["mode"], mode):
            out.append(c)
    return out


def fencers_for(conn, resource, mode, sid, my_rank, my_since):
    """Live waiters AHEAD of session sid in the queue for resource: strictly
    better rank, or equal rank with earlier since (FIFO). This is queue
    fencing — a free resource is not grabbable past a better-ranked/earlier
    co-worker who is actively waiting for it. Waiter rows survive release (they
    ARE the queue order across the release boundary), so fencing requires the
    waiter to be actively polling (fresh last_poll) — or to hold a paused
    resume spot — lest an abandoned row fence a free resource forever."""
    out = []
    t = now()
    rows = conn.execute(
        "SELECT w.*, s.last_seen AS s_seen, s.status AS s_status, s.paused AS s_paused "
        "FROM waiters w LEFT JOIN sessions s ON s.id = w.session_id "
        "WHERE w.active=1 AND w.session_id != ?",
        (sid,),
    ).fetchall()
    for w in rows:
        if w["s_status"] not in ("active",):
            continue
        polling = (w["last_poll"] or 0) >= t - POLL_FRESH_S
        paused_hold = (w["note"] == PAUSE_NOTE) and (w["s_paused"] or 0) == 1
        if not (polling or paused_hold):
            continue
        if not resources_overlap(norm_resource(w["resource"]), resource):
            continue
        if not modes_conflict(w["mode"], mode):
            continue
        wr = eff_rank(conn, w["session_id"])
        if rank_lt(wr, my_rank) or (wr == my_rank and w["since"] < my_since):
            out.append({"row": w, "rank": wr})
    return out


def holder_dicts(conn, rows):
    """Claim rows -> display dicts (short ids, ages, cron-holder flag)."""
    out = []
    for c in rows:
        s = conn.execute(
            "SELECT surface FROM sessions WHERE id=?", (c["session_id"],)
        ).fetchone()
        cronish = is_cron_session(s["surface"] if s else None)
        out.append({
            "session": short(c["session_id"]),
            "session_full": c["session_id"],
            "resource": c["resource"],
            "mode": c["mode"],
            "task": c["task"],
            "held_for": age_str(c["claimed_at"]),
            "ttl_min": c["ttl_min"],
            "rank": rank_str(eff_rank(conn, c["session_id"])),
            "is_cron": cronish,
        })
    return out


def fencer_dicts(conn, fencers):
    """Waiter rows -> display dicts for HELD/queue output."""
    out = []
    seen = set()
    for f in fencers:
        w = f["row"]
        key = (w["session_id"], w["resource"])
        if key in seen:
            continue
        seen.add(key)
        s = conn.execute(
            "SELECT task FROM sessions WHERE id=?", (w["session_id"],)
        ).fetchone()
        out.append(
            {
                "session": short(w["session_id"]),
                "resource": w["resource"],
                "rank": rank_str(f["rank"]),
                "waiting_for": age_str(w["since"]),
                "task": (s["task"] if s else None) or (w["note"] or "?"),
                "paused_resume": w["note"] == PAUSE_NOTE,
            }
        )
    return out


def ensure_waiter(conn, sid, resource, note, mode="exclusive"):
    """Idempotently register (or refresh) an active waiter row for
    (session, resource) and stamp last_poll for fencing liveness."""
    row = conn.execute(
        "SELECT id FROM waiters WHERE session_id=? AND resource=? AND active=1",
        (sid, resource),
    ).fetchone()
    if not row:
        conn.execute(
            "INSERT INTO waiters(session_id,resource,since,note,mode,last_poll)"
            " VALUES(?,?,?,?,?,?)",
            (sid, resource, now(), note, mode, now()),
        )
    else:
        # refresh poll freshness; `since` (queue position) is preserved
        conn.execute("UPDATE waiters SET last_poll=? WHERE id=?", (now(), row["id"]))


def clear_my_waiters(conn, sid, resources):
    """Deactivate my waiter rows for these resources (called once the
    claim succeeds or the wait is abandoned)."""
    for r in resources:
        conn.execute(
            "UPDATE waiters SET active=0 WHERE session_id=? AND resource=? AND active=1",
            (sid, r),
        )


def my_earliest_wait(conn, sid, resources):
    """If sid already queues for any of these resources, its queue position is
    that (earliest) since; otherwise it's a newcomer positioned at now()."""
    best = None
    for w in conn.execute(
        "SELECT * FROM waiters WHERE session_id=? AND active=1", (sid,)
    ).fetchall():
        for r in resources:
            if (resources_overlap(norm_resource(w["resource"]), r)
                    and (best is None or w["since"] < best)):
                best = w["since"]
    return best if best is not None else now()


def waiters_on(conn, sid):
    """Active waiters (other sessions) blocked on resources this session holds."""
    mine = conn.execute(
        "SELECT * FROM claims WHERE session_id=? AND status='held'", (sid,)
    ).fetchall()
    out = []
    for w in conn.execute(
        "SELECT * FROM waiters WHERE active=1 AND session_id != ?", (sid,)
    ).fetchall():
        for c in mine:
            if resources_overlap(norm_resource(w["resource"]), c["resource"]):
                s = conn.execute(
                    "SELECT task FROM sessions WHERE id=?", (w["session_id"],)
                ).fetchone()
                out.append(
                    {
                        "session": short(w["session_id"]),
                        "resource": c["resource"],
                        "their_task": (s["task"] if s else None) or (w["note"] or "?"),
                        "waiting_for": age_str(w["since"]),
                        "rank": rank_str(eff_rank(conn, w["session_id"])),
                    }
                )
                break
    return out


def emit(payload, as_json, human_lines):
    """Print either machine JSON (--json) or the human lines."""
    if as_json:
        print(json.dumps(payload, indent=None))
    else:
        for ln in human_lines:
            print(ln)


# ---------------------------------------------------------------- cron awareness

def is_cron_session(row_or_surface):
    """True if this session id was registered by cron-guard (surface
    'cron'). Cron holders are exempt from preempt."""
    s = row_or_surface
    if hasattr(s, "keys"):
        s = s["surface"]
    return (s or "").startswith("cron")


def load_cron_manifest():
    """{job_id: {resources:[...], name, policy, critical}} — advisory declarations
    of what each scheduled job touches. Missing/invalid file -> {} (inert)."""
    try:
        with open(CRON_MANIFEST, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    jobs = raw.get("jobs", raw) if isinstance(raw, dict) else {}
    out = {}
    if isinstance(jobs, dict):
        for jid, spec in jobs.items():
            if isinstance(spec, dict) and spec.get("resources"):
                out[str(jid)] = {
                    "resources": [norm_resource(r) for r in spec["resources"]],
                    "name": spec.get("name") or str(jid)[:12],
                    "policy": spec.get("policy") or "skip",
                    "critical": bool(spec.get("critical", False)),
                }
    return out


def _read_cron_store(path):
    """[job dicts] from one Hermes cron jobs.json. Any problem -> [].

    Fail-inert by design: cron visibility is advisory, so an unreadable,
    missing, or corrupt store must degrade to "no jobs seen" — never break
    claims/status/guards for every session on the box. Accepts the three
    store shapes Hermes has shipped: {"jobs":[...]}, bare list, {id: job}.
    """
    try:
        with open(path, encoding="utf-8") as f:
            store = json.load(f)
    except (OSError, ValueError):
        return []
    jobs = store.get("jobs", store if isinstance(store, list) else [])
    if isinstance(jobs, dict):
        jobs = list(jobs.values())
    return [j for j in jobs if isinstance(j, dict)]


def cron_store_jobs():
    """{job_id: {next_ts, last_ts, name, enabled, profile}} best-effort from
    the DEFAULT Hermes cron store PLUS every profile store
    (<profiles>/<name>/cron/jobs.json — Bot Mode: a bot is a profile and its
    Routines live in the bot's own store). Profile jobs get a load-time
    "[bot:<profile>] " name tag (skipped when the name already carries one,
    as Bot-Mode-created routines do) so every downstream radar/advisory/guard
    message names the owning bot for free. Read-only; a broken store just
    contributes nothing. Id collision across stores: default store wins."""
    from datetime import datetime

    def ts_of(v):
        """ISO-8601 string -> epoch seconds, or None. Accepts a trailing 'Z'
        (normalized to +00:00 for pre-3.11 fromisoformat compatibility)."""
        if not v:
            return None
        try:
            s = str(v).strip()
            if s.endswith(("Z", "z")):
                s = s[:-1] + "+00:00"
            return datetime.fromisoformat(s).timestamp()
        except (ValueError, TypeError):
            return None

    # Store scan order IS the collision policy: the default store is listed
    # first and first-seen wins below, so a profile job can never shadow a
    # default-store job sharing its id. sorted() keeps profile order (and thus
    # profile-vs-profile collisions) deterministic — raw glob order is
    # filesystem-dependent.
    stores = [(CRON_JOBS_JSON, None)]
    for p in sorted(glob.glob(os.path.join(CRON_PROFILES_DIR, "*", "cron", "jobs.json"))):
        # <profiles>/<name>/cron/jobs.json -> profile name is two dirs up.
        prof = os.path.basename(os.path.dirname(os.path.dirname(p)))
        stores.append((p, prof))

    out = {}
    for path, prof in stores:
        for j in _read_cron_store(path):
            jid = str(j.get("id"))
            if jid in out:
                continue  # collision: first-seen (default store) wins
            nm = j.get("name") or jid[:12]
            # Attribute profile (bot) jobs in the display name itself so every
            # downstream surface (radar, advisories, guard defer notes) names
            # the owning bot with no extra lookups. Substring check, not
            # startswith: Bot-Mode-created routines arrive pre-tagged and some
            # carry a leading marker before the tag.
            if prof and "[bot:" not in nm:
                nm = f"[bot:{prof}] {nm}"
            out[jid] = {
                "next_ts": ts_of(j.get("next_run_at")),
                "last_ts": ts_of(j.get("last_run_at")),
                "name": nm,
                "enabled": bool(j.get("enabled", True)),
                "profile": prof,
            }
    return out


def upcoming_cron_conflicts(resources, horizon_s):
    """Manifested cron jobs firing within horizon_s whose declared resources
    overlap any of `resources`. Returns [{job, name, resource, fires_in_s,
    policy, critical}]. Purely advisory — never blocks a claim."""
    man = load_cron_manifest()
    if not man:
        return []
    store = cron_store_jobs()
    t = now()
    out = []
    for jid, spec in man.items():
        j = store.get(jid)
        if not j or not j["enabled"] or j["next_ts"] is None:
            continue
        eta = j["next_ts"] - t
        if eta < -300 or eta > horizon_s:  # small grace for a just-fired tick
            continue
        for jr in spec["resources"]:
            for r in resources:
                if resources_overlap(jr, r):
                    # Profile (bot) jobs: prefer the store's tagged name so
                    # the advisory names the owning bot even when the manifest
                    # entry's name lacks the [bot:] tag.
                    nm = (j["name"] if j.get("profile")
                          else (spec["name"] or j["name"]))
                    out.append({"job": jid, "name": nm,
                                "resource": jr, "fires_in_s": max(0, int(eta)),
                                "policy": spec["policy"],
                                "critical": spec["critical"],
                                "profile": j.get("profile")})
                    break
            else:
                continue
            break
    return sorted(out, key=lambda x: x["fires_in_s"])


def cron_advisory_lines(resources, horizon_s=CRON_ADVISORY_S, ttl_min=None):
    """Human guidance when a claim/check overlaps an upcoming manifested cron.
    Presents the decision explicitly: finish first, step aside for the tick,
    or escalate to the USER to pause/trigger the job (never silently)."""
    hits = upcoming_cron_conflicts(resources, horizon_s)
    lines = []
    for h in hits:
        mins = h["fires_in_s"] // 60
        overlap = ttl_min is not None and (ttl_min * 60) >= h["fires_in_s"]
        tag = "CRITICAL cron" if h["critical"] else "cron"
        lines.append(
            f"CRON ADVISORY: {tag} '{h['name']}' ({h['job'][:12]}) fires in ~{mins}m "
            f"and touches {h['resource']}"
            + (f" — inside your claim's {int(ttl_min)}m TTL window." if overlap else ".")
        )
        if h["critical"]:
            lines.append(
                f"  Its guard policy is '{h['policy']}', but do NOT let a critical job "
                f"defer silently. Choose: (a) finish & release before it fires; "
                f"(b) at fire time, pause/release your claims and run "
                f"`wait-for-cron --job {h['job'][:12]}` then re-claim; "
                f"(c) if you must hold through it, ASK THE USER to pause or "
                f"trigger it early (cronjob tool), then record it: "
                f"`cron-note --job {h['job'][:12]} --action paused|triggered`."
            )
        elif h["policy"] == "unguarded":
            lines.append(
                f"  This job does NOT check the board (unguarded) — it WILL touch that "
                f"resource at fire time regardless of your claim. Finish first, or ask "
                f"the USER to pause/trigger it (then `cron-note --job {h['job'][:12]}`)."
            )
        else:
            lines.append(
                f"  If you still hold it then, the job's guard will {h['policy']} that "
                f"tick. Finish before, or expect the deferral (holders get notified)."
            )
    return hits, lines


# ---------------------------------------------------------------- commands

def cmd_register(a):
    """Verb: announce this session on the board; print id + co-workers.
    --id (or env HERMES_COORD_ID) registers under an EXPLICIT id — callers that
    pre-mint a memorable id get a real session row, so later release/done
    resolve it (pre-v2.3.2 the flag did not exist; claims made under a
    caller-minted id were orphaned until TTL). Explicit-id re-register is
    idempotent: refreshes task/surface/liveness and reactivates a done row."""
    conn = db()
    explicit = (getattr(a, "id", None) or "").strip()
    if explicit and not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{3,63}$", explicit):
        print(f"error: --id '{explicit}' invalid (4-64 chars: alnum . _ -, "
              "must start alphanumeric)", file=sys.stderr)
        return 2
    sid = explicit or uuid.uuid4().hex[:12]
    parent_id = None
    if a.parent:
        parent_id, err = resolve_session(conn, a.parent)
        if err:
            print(f"error: --parent: {err}", file=sys.stderr)
            return 1
    if a.rank and not re.match(r"^\d{1,4}[a-z0-9]{0,3}$", a.rank.strip().lower()):
        print(f"error: --rank '{a.rank}' invalid (use 1, 2, 3 or 1a)", file=sys.stderr)
        return 1
    if a.slot and not re.match(r"^[a-z0-9]{1,3}$", a.slot.strip().lower()):
        print(f"error: --slot '{a.slot}' invalid (a-z, 0-9, max 3 chars)", file=sys.stderr)
        return 1
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    existing = conn.execute(
        "SELECT id FROM sessions WHERE id=?", (sid,)).fetchone() if explicit else None
    if existing:
        # idempotent re-register: refresh task/surface/liveness, reactivate
        conn.execute(
            "UPDATE sessions SET task=?, surface=COALESCE(?,surface), "
            "status='active', paused=0, last_seen=? WHERE id=?",
            (a.task, a.surface, now(), sid))
    else:
        conn.execute(
            "INSERT INTO sessions(id,task,surface,started_at,last_seen,priority,parent_id,"
            "slot,rank_set_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (sid, a.task, a.surface, now(), now(),
             (a.rank.strip().lower() if a.rank else None), parent_id,
             (a.slot.strip().lower() if a.slot else None),
             now() if a.rank else None),
        )
    conn.execute("COMMIT")
    my_r = rank_str(eff_rank(conn, sid))
    others = conn.execute(
        "SELECT * FROM sessions WHERE status='active' AND id != ? ORDER BY started_at",
        (sid,),
    ).fetchall()
    lines = [sid]
    if (a.rank or parent_id) and not a.json:
        lineage = f", child of {short(parent_id)}" if parent_id else ""
        lines.append(f"# effective rank: {my_r}{lineage}")
    if others and not a.json:
        lines.append("# other active sessions (co-workers):")
        for o in others:
            r = rank_str(eff_rank(conn, o["id"]))
            paused = " [PAUSED]" if (o["paused"] or 0) else ""
            lines.append(
                f"#   {short(o['id'])}  [{o['surface'] or '?'}] rank {r}{paused} "
                f"since {age_str(o['started_at'])}: {o['task']}"
            )
    emit(
        {"id": sid, "rank": my_r, "parent": short(parent_id) if parent_id else None,
         "others": [
             {"session": short(o["id"]), "task": o["task"], "surface": o["surface"],
              "rank": rank_str(eff_rank(conn, o["id"])),
              "paused": bool(o["paused"] or 0),
              "active_for": age_str(o["started_at"])} for o in others]},
        a.json, lines,
    )
    return 0


def try_claim(conn, sid, res_mode_pairs, task, ttl):
    """One atomic all-or-nothing attempt over (resource, mode) pairs.
    Returns (ok, holders, fencers)."""
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, sid)
    blockers = []
    for r, m in res_mode_pairs:
        blockers.extend(holders_of(conn, r, m, sid))
    if blockers:
        conn.execute("COMMIT")  # reap/touch still persist
        return False, blockers, []
    my_rank = eff_rank(conn, sid)
    my_since = my_earliest_wait(conn, sid, [r for r, _ in res_mode_pairs])
    fencers = []
    for r, m in res_mode_pairs:
        fencers.extend(fencers_for(conn, r, m, sid, my_rank, my_since))
    if fencers:
        conn.execute("COMMIT")
        return False, [], fencers
    t = now()
    for r, m in res_mode_pairs:
        # idempotent refresh of our own existing claim on the same key
        conn.execute(
            "UPDATE claims SET status='released', released_at=? "
            "WHERE session_id=? AND resource=? AND status IN ('held','paused')",
            (t, sid, r),
        )
        conn.execute(
            "INSERT INTO claims(session_id,resource,mode,task,claimed_at,ttl_min)"
            " VALUES(?,?,?,?,?,?)",
            (sid, r, m, task, t, ttl),
        )
    clear_my_waiters(conn, sid, [r for r, _ in res_mode_pairs])
    conn.execute("COMMIT")
    return True, [], []


def _claim_loop(conn, a, res_mode_pairs, task, verb="CLAIMED"):
    """Shared acquire loop for claim and resume: handles held/fenced branches,
    --wait polling, waiter registration, and co-worker messaging."""
    resources = [r for r, _ in res_mode_pairs]
    deadline = now() + a.timeout
    announced = False
    while True:
        ok, blockers, fencers = try_claim(conn, a.id, res_mode_pairs, task, a.ttl)
        if ok:
            w = waiters_on(conn, a.id)
            lines = [f"{verb} ({res_mode_pairs[0][1]}): " + ", ".join(resources)]
            for x in w:
                lines.append(
                    f"NOTE: co-worker {x['session']} (rank {x['rank']}) is WAITING on "
                    f"{x['resource']} ({x['waiting_for']}) — task: {x['their_task']}. "
                    f"Finish and release promptly."
                )
            cron_hits, cron_lines = cron_advisory_lines(
                resources, ttl_min=getattr(a, "ttl", None))
            lines.extend(cron_lines)
            alert_pending(conn, a.id)
            payload = {"ok": True, "claimed": resources, "mode": res_mode_pairs[0][1],
                       "waiters_on_you": w}
            if cron_hits:
                payload["cron_advisories"] = cron_hits
            return 0, payload, lines
        wait_mode = getattr(a, "wait", False)
        if wait_mode:
            conn.execute("BEGIN IMMEDIATE")
            for r, m in res_mode_pairs:
                ensure_waiter(conn, a.id, r, task or "waiting", m)
            conn.execute("COMMIT")
            if now() + 4 <= deadline:
                if not announced and not a.json:
                    if blockers:
                        h = blockers[0]
                        print(
                            f"waiting: {h['resource']} held by co-worker "
                            f"{short(h['session_id'])} (task '{h['task']}', "
                            f"{age_str(h['claimed_at'])}) — polling...",
                            flush=True,
                        )
                    else:
                        f0 = fencer_dicts(conn, fencers)[0]
                        print(
                            f"queued: co-worker {f0['session']} (rank {f0['rank']}) is "
                            f"ahead of you for {f0['resource']} — polling in order...",
                            flush=True,
                        )
                    announced = True
                time.sleep(4)
                continue
        if blockers:
            hd = holder_dicts(conn, blockers)
            lines = ["HELD — resource(s) in use by a co-worker session:"]
            for h in hd:
                who = "CRON JOB" if h.get("is_cron") else "session"
                lines.append(
                    f"  {h['resource']} <- {who} {h['session']} (rank {h['rank']}, "
                    f"{h['mode']}, held {h['held_for']}): {h['task']}"
                )
            if any(h.get("is_cron") for h in hd):
                lines.append(
                    "note: cron holders finish on their own schedule — wait for release "
                    "(usually minutes). Crons cannot checkpoint/pause, so preempt won't "
                    "work; if one looks hung past its TTL, surface to the user before "
                    "any steal."
                )
            payload = {"ok": False, "held": True, "holders": hd,
                       "registered_waiter": bool(wait_mode)}
            if wait_mode:
                lines.append(
                    "Registered as waiter: the holder will see you're waiting and you'll "
                    "get an inbox notification when it releases. Re-run to keep waiting, "
                    "do other work, or ask the user."
                )
            my_r = eff_rank(conn, a.id)
            preemptable = [h for h in blockers
                           if rank_lt(my_r, eff_rank(conn, h["session_id"]))]
            if preemptable and my_r[0] != INF:
                lines.append(
                    f"You OUTRANK {len(preemptable)} holder(s) (your rank "
                    f"{rank_str(my_r)}). If the user confirmed priority, send a pause "
                    f"request: session_coord.py preempt --id {a.id} --res "
                    + " --res ".join(f"'{h['resource']}'" for h in preemptable)
                )
                payload["preemptable"] = True
        else:
            fd = fencer_dicts(conn, fencers)
            lines = ["QUEUED — co-worker(s) ahead of you for these resource(s):"]
            for f in fd:
                tag = " (paused, resume pending)" if f["paused_resume"] else ""
                lines.append(
                    f"  {f['resource']} <- waiter {f['session']} (rank {f['rank']}, "
                    f"waiting {f['waiting_for']}{tag}): {f['task']}"
                )
            lines.append(
                "Resource is free but queue order applies (user priority / FIFO). "
                "Use --wait to take your turn, or ask the user to re-prioritize."
            )
            payload = {"ok": False, "held": False, "fenced": True, "queue": fd,
                       "registered_waiter": bool(wait_mode)}
        alert_pending(conn, a.id)
        return 75, payload, lines


def cmd_claim(a):
    """Verb: atomically claim resources (all-or-nothing). --wait polls
    politely; prints CRON ADVISORY for upcoming manifested cron fires."""
    if not a.id:
        print("error: --id required (or set HERMES_COORD_ID)", file=sys.stderr)
        return 1
    conn = db()
    # Orphan-proofing (v2.3.2): a claim under an id that was never registered
    # auto-creates a minimal session row, so release/done can always resolve
    # the id later. Claims must never be easier to make than to release.
    if not conn.execute("SELECT 1 FROM sessions WHERE id=?", (a.id,)).fetchone():
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR IGNORE INTO sessions(id,task,surface,started_at,last_seen) "
            "VALUES(?,?,?,?,?)",
            (a.id, a.task or "(auto-registered at claim)", None, now(), now()))
        conn.execute("COMMIT")
    resources = [norm_resource(r) for r in a.res]
    task = a.task or (conn.execute(
        "SELECT task FROM sessions WHERE id=?", (a.id,)).fetchone() or {"task": None})["task"]
    pairs = [(r, a.mode) for r in resources]
    rc, payload, lines = _claim_loop(conn, a, pairs, task)
    emit(payload, a.json, lines)
    return rc


def cmd_check(a):
    """Verb: read-only availability probe (no claim, no waiter row)."""
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    conn.execute("COMMIT")
    resources = [norm_resource(r) for r in a.res]
    all_blockers = []
    for r in resources:
        all_blockers.extend(holders_of(conn, r, a.mode, a.id))
    if not all_blockers:
        lines = ["FREE: " + ", ".join(resources)]
        payload = {"free": True, "resources": resources}
        cron_hits, cron_lines = cron_advisory_lines(resources)
        lines.extend(cron_lines)
        if cron_hits:
            payload["cron_advisories"] = cron_hits
        if a.id:
            my_rank = eff_rank(conn, a.id)
            my_since = my_earliest_wait(conn, a.id, resources)
            fencers = []
            for r in resources:
                fencers.extend(fencers_for(conn, r, a.mode, a.id, my_rank, my_since))
            if fencers:
                fd = fencer_dicts(conn, fencers)
                payload["queue_ahead"] = fd
                for f in fd:
                    lines.append(
                        f"note: queue ahead — {f['session']} (rank {f['rank']}) is "
                        f"waiting on {f['resource']}; claim honors their turn first."
                    )
        alert_pending(conn, a.id)
        emit(payload, a.json, lines)
        return 0
    hd = holder_dicts(conn, all_blockers)
    lines = ["IN USE:"]
    for h in hd:
        lines.append(
            f"  {h['resource']} <- session {h['session']} (rank {h['rank']}, "
            f"{h['mode']}, held {h['held_for']}): {h['task']}"
        )
    alert_pending(conn, a.id)
    emit({"free": False, "holders": hd}, a.json, lines)
    return 75


def cmd_wait(a):
    """Poll until resource(s) are free (read-side wait; claims nothing)."""
    conn = db()
    resources = [norm_resource(r) for r in a.res]
    deadline = now() + a.timeout
    announced = False
    while True:
        conn.execute("BEGIN IMMEDIATE")
        reap(conn)
        touch_session(conn, a.id)
        blockers = []
        for r in resources:
            blockers.extend(holders_of(conn, r, a.mode, a.id))
        if blockers and a.id:
            for r in resources:
                ensure_waiter(conn, a.id, r, a.note or "waiting (read)", a.mode)
        elif not blockers and a.id:
            clear_my_waiters(conn, a.id, resources)
        conn.execute("COMMIT")
        if not blockers:
            alert_pending(conn, a.id)
            emit({"free": True, "resources": resources}, a.json,
                 ["FREE: " + ", ".join(resources)])
            return 0
        if now() + 4 > deadline:
            hd = holder_dicts(conn, blockers)
            alert_pending(conn, a.id)
            emit({"free": False, "holders": hd, "registered_waiter": bool(a.id)}, a.json,
                 [f"STILL HELD after {int(a.timeout)}s: " +
                  "; ".join(f"{h['resource']} <- {h['session']} ({h['task']})" for h in hd)])
            return 75
        if not announced and not a.json:
            h = blockers[0]
            print(f"waiting on {h['resource']} (held by {short(h['session_id'])}, "
                  f"task '{h['task']}')...", flush=True)
            announced = True
        time.sleep(4)


def _release(conn, sid, resources_or_none, final_status="released",
             statuses=("held",)):
    """Release claims; notify + deactivate waiters. Returns released keys."""
    t = now()
    ph = ",".join("?" * len(statuses))  # placeholders only ('?,?'); values bind parameterized
    q = f"SELECT * FROM claims WHERE session_id=? AND status IN ({ph})"  # nosec B608
    rows_all = conn.execute(q, (sid, *statuses)).fetchall()
    if resources_or_none is None:
        rows = rows_all
    else:
        rows = [c for c in rows_all
                if any(resources_overlap(c["resource"], r) for r in resources_or_none)]
    released = []
    for c in rows:
        conn.execute(
            "UPDATE claims SET status=?, released_at=? WHERE id=?",
            (final_status, t, c["id"]),
        )
        released.append(c["resource"])
        for w in conn.execute(
            "SELECT * FROM waiters WHERE active=1 AND session_id != ?", (sid,)
        ).fetchall():
            if resources_overlap(norm_resource(w["resource"]), c["resource"]):
                notify(
                    conn, w["session_id"], sid, c["resource"], "released",
                    f"{c['resource']} released by co-worker {short(sid)} "
                    f"(task '{c['task']}' finished cleanly). You're clear to proceed.",
                    dedupe=True,
                )
                # NOTE: waiter rows stay ACTIVE on purpose — they carry queue
                # order (rank + FIFO) across the release boundary. Deactivating
                # here made the post-release window first-poll-wins (rank
                # inversion). Rows clear when the waiter claims (try_claim),
                # sees free (cmd_wait), finishes (done), or stops polling
                # (POLL_FRESH_S staleness in fencers_for).
    return released


def cmd_release(a):
    """Verb: release specific resources early, keeping the rest."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    res = None if a.all else [norm_resource(r) for r in a.res]
    released = _release(conn, a.id, res)
    conn.execute("COMMIT")
    alert_pending(conn, a.id)
    emit({"released": released}, a.json,
         ["RELEASED: " + (", ".join(released) if released else "(nothing held)")])
    return 0


def cmd_done(a):
    """Verb: end of TASK — release all claims, notify waiters, deregister,
    and warn about crons this session paused but never resumed."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    released = _release(conn, a.id, None, statuses=("held", "paused"))
    conn.execute("UPDATE waiters SET active=0 WHERE session_id=?", (a.id,))
    conn.execute(
        "UPDATE sessions SET status='done', paused=0, last_seen=? WHERE id=?",
        (now(), a.id))
    # crons THIS session had the user pause (cron-note --action paused) and
    # never marked resumed — a paused job silently never fires again, so this
    # must never fall through the cracks at task end.
    dangling = conn.execute(
        "SELECT job_id, job_name, MAX(created_at) t FROM cron_events "
        "WHERE session_id=? AND event='user-paused' AND job_id NOT IN ("
        "  SELECT job_id FROM cron_events WHERE session_id=? "
        "  AND event IN ('user-resumed','user-triggered')"
        ") GROUP BY job_id", (a.id, a.id),
    ).fetchall()
    conn.execute("COMMIT")
    lines = [f"DONE: session {short(a.id)} deregistered; released: "
             + (", ".join(released) if released else "(nothing held)")]
    for d in dangling:
        lines.append(
            f"⚠ UNRESOLVED PAUSED CRON: '{d['job_name']}' ({str(d['job_id'])[:12]}) "
            f"was paused for this task and never resumed — it will NOT fire again "
            f"until resumed. Resume it now (cronjob action='resume', then "
            f"cron-note --action resumed), and action='run' if a tick was missed.")
    emit({"done": True, "released": released,
          "unresolved_paused_crons": [dict(d) for d in dangling]}, a.json, lines)
    return 0


def cmd_inbox(a):
    """Verb: drain my notifications (release/expiry/steal/defer/rerank)."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    rows = conn.execute(
        "SELECT * FROM notifications WHERE to_session=? AND read_at IS NULL "
        "ORDER BY created_at", (a.id,),
    ).fetchall()
    t = now()
    for r in rows:
        conn.execute("UPDATE notifications SET read_at=? WHERE id=?", (t, r["id"]))
    conn.execute("COMMIT")
    msgs = [{"kind": r["kind"], "resource": r["resource"], "from": short(r["from_session"]),
             "age": age_str(r["created_at"]), "body": r["body"]} for r in rows]
    lines = ([f"[{m['kind']} · {m['age']} ago] {m['body']}" for m in msgs]
             or ["(no new notifications)"])
    emit({"notifications": msgs}, a.json, lines)
    return 0


def cmd_pause(a):
    """Holder-side: checkpoint first, then pause. Claims stop blocking but keep
    a resume queue spot at this session's rank; waiters are told the resource
    is free for them."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    t = now()
    rows_all = conn.execute(
        "SELECT * FROM claims WHERE session_id=? AND status='held'", (a.id,)
    ).fetchall()
    targets = rows_all if not a.res else [
        c for c in rows_all
        if any(resources_overlap(c["resource"], norm_resource(r)) for r in a.res)
    ]
    paused = []
    for c in targets:
        ttl = a.ttl if a.ttl is not None else c["ttl_min"]
        conn.execute(
            "UPDATE claims SET status='paused', paused_at=?, ttl_min=? WHERE id=?",
            (t, ttl, c["id"]),
        )
        paused.append(c["resource"])
        # our resume queue spot: a waiter row at our rank, exempt from liveness
        ensure_waiter(conn, a.id, c["resource"], PAUSE_NOTE, c["mode"])
        for w in conn.execute(
            "SELECT * FROM waiters WHERE active=1 AND session_id != ?", (a.id,)
        ).fetchall():
            if resources_overlap(norm_resource(w["resource"]), c["resource"]):
                notify(
                    conn, w["session_id"], a.id, c["resource"], "paused",
                    f"{c['resource']} freed: co-worker {short(a.id)} PAUSED its task "
                    f"'{c['task']}' (checkpoint: {a.note or 'not recorded'}). "
                    f"You're clear to claim — it will resume after you finish.",
                )
                # leave their waiter ACTIVE: fencing decides who goes first
    if targets:
        conn.execute(
            "UPDATE sessions SET paused=1, checkpoint_note=? WHERE id=?",
            (a.note, a.id),
        )
    conn.execute("COMMIT")
    my_r = rank_str(eff_rank(conn, a.id))
    lines = (["PAUSED — claims parked (no longer blocking): " + ", ".join(paused),
              (f"Your resume spot is queued at rank {my_r}. You'll get a 'released' "
               f"notification when each resource frees; then run: "
               f"session_coord.py resume --id {a.id} [--wait]"),
              f"Checkpoint note: {a.note or '(none — record one next time!)'}"]
             if paused else ["(nothing held to pause)"])
    emit({"paused": paused, "checkpoint": a.note, "rank": my_r}, a.json, lines)
    return 0


def cmd_resume(a):
    """Verb: resume after preempt-pause; reprints the saved checkpoint."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    prows = conn.execute(
        "SELECT * FROM claims WHERE session_id=? AND status='paused'", (a.id,)
    ).fetchall()
    if not prows:
        conn.execute("BEGIN IMMEDIATE")
        reap(conn)
        touch_session(conn, a.id)
        conn.execute("UPDATE sessions SET paused=0 WHERE id=?", (a.id,))
        conn.execute("COMMIT")
        emit({"resumed": [], "note": "nothing paused"}, a.json,
             ["(nothing paused — already released or expired; check status)"])
        return 0
    pairs = [(c["resource"], c["mode"] or "exclusive") for c in prows]
    task = prows[0]["task"]
    a.ttl = a.ttl if a.ttl is not None else DEFAULT_TTL_MIN
    rc, payload, lines = _claim_loop(conn, a, pairs, task, verb="RESUMED")
    if rc == 0:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("UPDATE sessions SET paused=0 WHERE id=?", (a.id,))
        note = conn.execute(
            "SELECT checkpoint_note FROM sessions WHERE id=?", (a.id,)
        ).fetchone()["checkpoint_note"]
        conn.execute("COMMIT")
        if note:
            lines.append(f"Your checkpoint from pause: {note}")
            payload["checkpoint"] = note
    emit(payload, a.json, lines)
    return rc


def cmd_preempt(a):
    """Requester-side: relay the USER's priority decision to holder(s) as a
    polite pause request. Requires a user-set rank strictly better than the
    holder's — otherwise refuse and point back to the user."""
    if not a.id:
        print("error: --id required", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    my_r = eff_rank(conn, a.id)
    me = conn.execute("SELECT * FROM sessions WHERE id=?", (a.id,)).fetchone()
    if my_r[0] == INF:
        conn.execute("COMMIT")
        emit({"sent": [], "refused": [], "error": "no user priority set"}, a.json,
             [("REFUSED: this session has no user-set priority. Preemption relays a "
               "USER decision — ask the user, then record it: "
               f"session_coord.py prioritize --session {a.id} --rank 1")])
        return 75
    resources = [norm_resource(r) for r in a.res]
    sent, refused = [], []
    for r in resources:
        for h in holders_of(conn, r, "exclusive", a.id):
            hr = eff_rank(conn, h["session_id"])
            hs = conn.execute("SELECT surface FROM sessions WHERE id=?",
                              (h["session_id"],)).fetchone()
            entry = {"session": short(h["session_id"]), "resource": h["resource"],
                     "holder_rank": rank_str(hr), "task": h["task"]}
            if is_cron_session(hs["surface"] if hs else None):
                entry["is_cron"] = True
                entry["why"] = ("cron jobs cannot checkpoint/pause — wait for the tick "
                                "to finish (TTL bounds a hung one), or ask the user "
                                "before steal")
                refused.append(entry)
            elif rank_lt(my_r, hr):
                notify(
                    conn, h["session_id"], a.id, h["resource"], "preempt_request",
                    f"USER-PRIORITY REQUEST from co-worker {short(a.id)} (rank "
                    f"{rank_str(my_r)}, task '{me['task'] if me else '?'}'): the user has "
                    f"prioritized their task over yours and they need {h['resource']}, "
                    f"which you hold (task '{h['task']}'). Please finish your current "
                    f"atomic step, CHECKPOINT your state durably (ledger/notes), then run: "
                    f"python3 ~/.hermes/scripts/session_coord.py pause --id "
                    f"{h['session_id']} --note '<where you saved state>'. Your claims "
                    f"keep a resume spot at your rank — you'll be notified to resume "
                    f"when the resource frees. Reply not needed; pausing IS the ack.",
                    dedupe=True,
                )
                ensure_waiter(conn, a.id, r, a.reason or "preempting (user priority)")
                sent.append(entry)
            else:
                refused.append(entry)
    conn.execute("COMMIT")
    lines = []
    for s in sent:
        lines.append(
            f"PREEMPT REQUESTED: {s['resource']} <- holder {s['session']} (rank "
            f"{s['holder_rank']}) asked to checkpoint+pause. You are queued at rank "
            f"{rank_str(my_r)}; acquire with: claim --id {a.id} --res '{s['resource']}' --wait"
        )
    for rj in refused:
        if rj.get("is_cron"):
            lines.append(
                f"NOT SENT: {rj['resource']} is held by CRON JOB {rj['session']} "
                f"(task '{rj['task']}'). {rj['why']}."
            )
        else:
            lines.append(
                f"NOT SENT: holder {rj['session']} (rank {rj['holder_rank']}) does not rank "
                f"below you ({rank_str(my_r)}) — escalate to the user to re-prioritize."
            )
    if not sent and not refused:
        lines.append("Nothing to preempt — no conflicting held claims (resource free "
                     "or only paused claims). Just claim it.")
    emit({"sent": sent, "refused": refused, "my_rank": rank_str(my_r)}, a.json, lines)
    return 0 if not refused else 75


def cmd_prioritize(a):
    """Record the USER's priority order. Takes effect immediately: fencing and
    preemption read ranks live, and children inherit via lineage."""
    conn = db()
    changes = []
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)

    # snapshot EFFECTIVE ranks of every active session before any change, so
    # the post-change diff catches inherited moves too (a child's 1a -> 2a when
    # its PARENT is re-ranked), not just directly targeted sessions.
    pre_ranks = {
        s["id"]: rank_str(eff_rank(conn, s["id"]))
        for s in conn.execute(
            "SELECT id FROM sessions WHERE status='active'").fetchall()
    }

    def set_rank(token, rank=None, slot=None, clear=False):
        """Set/clear a session's user-assigned priority rank (validated)."""
        sid, err = resolve_session(conn, token)
        if err:
            return {"error": err, "token": token}
        old = rank_str(eff_rank(conn, sid))
        if clear:
            conn.execute(
                "UPDATE sessions SET priority=NULL, rank_set_at=? WHERE id=?",
                (now(), sid))
        if rank is not None:
            rk = rank.strip().lower()
            if not re.match(r"^\d{1,4}[a-z0-9]{0,3}$", rk):
                return {"error": f"invalid rank '{rank}'", "token": token}
            conn.execute(
                "UPDATE sessions SET priority=?, rank_set_at=? WHERE id=?",
                (rk, now(), sid))
        if slot is not None:
            sl = slot.strip().lower()
            if not re.match(r"^[a-z0-9]{1,3}$", sl):
                return {"error": f"invalid slot '{slot}'", "token": token}
            conn.execute("UPDATE sessions SET slot=? WHERE id=?", (sl, sid))
        new = rank_str(eff_rank(conn, sid))
        return {"session": short(sid), "was": old, "now": new}

    if a.order:
        for part in a.order.split(","):
            if "=" not in part:
                changes.append({"error": f"bad segment '{part}' (want id=rank)"})
                continue
            tok, rk = part.split("=", 1)
            changes.append(set_rank(tok.strip(), rank=rk.strip()))
    if a.session:
        changes.append(set_rank(a.session, rank=a.rank, slot=a.slot, clear=a.clear))
    if a.clear_all:
        conn.execute(
            "UPDATE sessions SET priority=NULL WHERE status='active'")
        changes.append({"cleared": "all active sessions"})

    # diff EFFECTIVE ranks: notify every session whose rank moved, whether it
    # was targeted directly or inherited the change from its parent (1a -> 2a).
    for sid_full, old in pre_ranks.items():
        new = rank_str(eff_rank(conn, sid_full))
        if new != old and sid_full != a.id:
            via = ""
            row = conn.execute("SELECT parent_id, priority FROM sessions WHERE id=?",
                               (sid_full,)).fetchone()
            if row and row["parent_id"] and not row["priority"]:
                via = f" (inherited via parent {short(row['parent_id'])})"
            notify(conn, sid_full, a.id, None, "priority",
                   f"USER PRIORITY UPDATE: your effective rank is now {new} "
                   f"(was {old}){via}. Queue order re-fences immediately; if you "
                   f"were told to pause but now outrank the requester, you may "
                   f"keep working.")
    conn.execute("COMMIT")

    sess = conn.execute(
        "SELECT * FROM sessions WHERE status='active' ORDER BY started_at").fetchall()
    order = sorted(sess, key=lambda s: (eff_rank(conn, s["id"]), s["started_at"]))
    lines = ["PRIORITY ORDER (user-set; lower = first):"]
    for s in order:
        r = rank_str(eff_rank(conn, s["id"]))
        lin = f" (child of {short(s['parent_id'])})" if s["parent_id"] else ""
        paused = " [PAUSED]" if (s["paused"] or 0) else ""
        lines.append(f"  {r:>4}  {short(s['id'])}{lin}{paused}: {s['task']}")
    errs = [c for c in changes if "error" in c]
    for e in errs:
        lines.append(f"error: {e.get('token','')}: {e['error']}")
    emit({"changes": changes,
          "order": [{"session": short(s["id"]),
                     "rank": rank_str(eff_rank(conn, s["id"])),
                     "task": s["task"]} for s in order]},
         a.json, lines)
    return 1 if errs else 0


def cmd_status(a):
    """Verb: whole-board report — sessions by rank, held resources,
    waiters, warnings, recent cron activity, 12h cron radar."""
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    conn.execute("COMMIT")
    t = now()
    sess = conn.execute(
        "SELECT * FROM sessions WHERE status='active' ORDER BY started_at").fetchall()
    sess = sorted(sess, key=lambda s: (eff_rank(conn, s["id"]), s["started_at"]))
    claims = conn.execute(
        "SELECT * FROM claims WHERE status='held' ORDER BY claimed_at").fetchall()
    pclaims = conn.execute(
        "SELECT * FROM claims WHERE status='paused' ORDER BY paused_at").fetchall()
    waiters = conn.execute(
        "SELECT * FROM waiters WHERE active=1 ORDER BY since").fetchall()
    recent_expired = conn.execute(
        "SELECT * FROM claims WHERE status IN ('expired','stolen') AND released_at > ? "
        "ORDER BY released_at DESC LIMIT 10", (t - 86400,),
    ).fetchall()
    payload = {
        "db": DB_PATH,
        "active_sessions": [
            {"session": short(s["id"]), "surface": s["surface"], "task": s["task"],
             "rank": rank_str(eff_rank(conn, s["id"])),
             "parent": short(s["parent_id"]) if s["parent_id"] else None,
             "paused": bool(s["paused"] or 0),
             "checkpoint": s["checkpoint_note"] if (s["paused"] or 0) else None,
             "active_for": age_str(s["started_at"]), "last_seen": age_str(s["last_seen"])}
            for s in sess],
        "held_claims": holder_dicts(conn, claims),
        "paused_claims": [
            {"session": short(c["session_id"]), "resource": c["resource"],
             "mode": c["mode"], "task": c["task"],
             "paused_for": age_str(c["paused_at"] or c["claimed_at"])}
            for c in pclaims],
        "waiters": [
            {"session": short(w["session_id"]), "resource": w["resource"],
             "rank": rank_str(eff_rank(conn, w["session_id"])),
             "waiting_for": age_str(w["since"]),
             "note": w["note"]} for w in waiters],
        "recent_expired_or_stolen": [
            {"session": short(c["session_id"]), "resource": c["resource"],
             "status": c["status"], "task": c["task"], "when": age_str(c["released_at"])}
            for c in recent_expired],
    }
    lines = [f"COORDINATION BOARD  ({DB_PATH})"]
    lines.append(f"— active sessions ({len(sess)}), priority order:" if sess
                 else "— active sessions: none")
    for s in sess:
        r = rank_str(eff_rank(conn, s["id"]))
        lin = f", child of {short(s['parent_id'])}" if s["parent_id"] else ""
        paused = ""
        if s["paused"] or 0:
            paused = f" [PAUSED — checkpoint: {s['checkpoint_note'] or 'none recorded'}]"
        lines.append(f"    rank {r:>3}  {short(s['id'])} [{s['surface'] or '?'}{lin}] "
                     f"since {age_str(s['started_at'])} (seen {age_str(s['last_seen'])} "
                     f"ago){paused}: {s['task']}")
    lines.append(f"— held resources ({len(claims)}):" if claims
                 else "— held resources: none")
    for c in claims:
        r = rank_str(eff_rank(conn, c["session_id"]))
        lines.append(f"    {c['resource']}  <- {short(c['session_id'])} (rank {r}, "
                     f"{c['mode']}, held {age_str(c['claimed_at'])}, "
                     f"ttl {int(c['ttl_min'])}m): {c['task']}")
    if pclaims:
        lines.append(f"— paused claims, resume pending ({len(pclaims)}):")
        for c in pclaims:
            lines.append(f"    {c['resource']}  <- {short(c['session_id'])} paused "
                         f"{age_str(c['paused_at'] or c['claimed_at'])} ago: {c['task']}")
    if waiters:
        lines.append(f"— waiting ({len(waiters)}), queue order = rank then FIFO:")
        for w in waiters:
            r = rank_str(eff_rank(conn, w["session_id"]))
            lines.append(f"    {short(w['session_id'])} (rank {r}) waiting on "
                         f"{w['resource']} for {age_str(w['since'])} ({w['note']})")
    if recent_expired:
        lines.append("— WARNINGS (last 24h — holder may have died mid-task; verify state):")
        for c in recent_expired:
            lines.append(f"    {c['status'].upper()}: {c['resource']} (session "
                         f"{short(c['session_id'])}, task '{c['task']}', "
                         f"{age_str(c['released_at'])} ago)")
    man = load_cron_manifest()
    if man:
        all_res = list({r for spec in man.values() for r in spec["resources"]})
        cron_soon = upcoming_cron_conflicts(all_res, CRON_STATUS_S)
        seen_jobs = set()
        cron_rows = []
        for h in cron_soon:
            if h["job"] in seen_jobs:
                continue
            seen_jobs.add(h["job"])
            spec = man[h["job"]]
            clash = sorted({c["resource"] for c in claims
                            for jr in spec["resources"]
                            if resources_overlap(jr, norm_resource(c["resource"]))})
            cron_rows.append({"job": h["job"], "name": h["name"],
                              "profile": h.get("profile"),
                              "resources": spec["resources"],
                              "policy": spec["policy"],
                              "critical": spec["critical"],
                              "fires_in_s": h["fires_in_s"],
                              "conflicts_with_held": clash})
        if cron_rows:
            payload["upcoming_crons"] = cron_rows
            lines.append(
                f"— scheduled cron jobs w/ declared resources "
                f"(next {CRON_STATUS_S // 3600}h):")
            for cr in cron_rows:
                eta = cr["fires_in_s"]
                eta_s = (f"{eta // 3600}h{(eta % 3600) // 60:02d}m"
                         if eta >= 3600 else f"{eta // 60}m")
                crit = " CRITICAL," if cr["critical"] else ""
                lines.append(f"    in ~{eta_s}: '{cr['name']}' ({cr['job'][:12]},{crit} "
                             f"on-conflict={cr['policy']}) -> {', '.join(cr['resources'])}")
                if cr["conflicts_with_held"]:
                    lines.append(f"      ⚠ CONFLICTS with currently HELD: "
                                 f"{', '.join(cr['conflicts_with_held'])} — holder(s) "
                                 f"should finish first or see `wait-for-cron`.")
    unenrolled = unenrolled_bot_profiles()
    if unenrolled:
        payload["unenrolled_bot_profiles"] = unenrolled
        lines.append(f"— UNENROLLED bot profiles ({len(unenrolled)}) — SOUL.md lacks "
                     "the coordination blurb; their runs never consult this board:")
        for name in unenrolled:
            lines.append(f"    {name}  (fix: re-run install.py, or paste "
                         "templates/bot-soul-coordination.md into its SOUL.md)")
    unwired = unwired_profiles()
    if unwired:
        payload["unwired_profiles"] = unwired
        lines.append(f"— UNWIRED profiles ({len(unwired)}) — the profile's own memory "
                     "store lacks the standing rule; its sessions never consult this "
                     "board:")
        for name in unwired:
            lines.append(f"    {name}  (fix: re-run install.py, or add the entry from "
                         "examples/memory-entry.example.md to its memories/MEMORY.md)")
    alert_pending(conn, a.id)
    emit(payload, a.json, lines)
    return 0


def cmd_steal(a):
    """Force-release a co-worker's claim. LAST RESORT: user approval or clearly-dead
    holder only. Notifies the previous holder and logs the reason."""
    if not (a.id and a.reason):
        print("error: steal requires --id and --reason", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    resources = [norm_resource(r) for r in a.res]
    stolen = []
    for c in conn.execute(
        "SELECT * FROM claims WHERE status IN ('held','paused')"
    ).fetchall():
        if c["session_id"] == a.id:
            continue
        if any(resources_overlap(c["resource"], r) for r in resources):
            conn.execute(
                "UPDATE claims SET status='stolen', released_at=? WHERE id=?",
                (now(), c["id"]),
            )
            notify(conn, c["session_id"], a.id, c["resource"], "stolen",
                   f"Your claim on {c['resource']} was force-released by "
                   f"{short(a.id)}. Reason: {a.reason}")
            stolen.append(c["resource"])
    conn.execute("COMMIT")
    emit({"stolen": stolen, "reason": a.reason}, a.json,
         ["FORCE-RELEASED: " + (", ".join(stolen) if stolen else "(nothing matched)")])
    return 0


# ---------------------------------------------------------------- cron commands

def resolve_cron_job(token):
    """Resolve a job id prefix or (unique, case-insensitive) name fragment
    against the cron store + manifest. Returns (job_id, name, store_row|None,
    err)."""
    token = (token or "").strip()
    store = cron_store_jobs()
    man = load_cron_manifest()
    ids = set(store) | set(man)
    exact = [i for i in ids if i == token]
    pref = [i for i in ids if i.startswith(token)] if len(token) >= 4 else []
    byname = [i for i in ids
              if token.lower() in ((store.get(i) or man.get(i) or {}).get("name") or "").lower()]
    for cands in (exact, pref, byname):
        cands = sorted(set(cands))
        if len(cands) == 1:
            i = cands[0]
            nm = (store.get(i) or {}).get("name") or (man.get(i) or {}).get("name") or i[:12]
            return i, nm, store.get(i), None
        if len(cands) > 1:
            return None, None, None, f"ambiguous job '{token}': {', '.join(c[:12] for c in cands)}"
    return None, None, None, (f"no cron job matches '{token}' (stores: "
                              f"{CRON_JOBS_JSON} + {CRON_PROFILES_DIR}/*/cron/jobs.json)")


def log_cron_event(conn, job_id, job_name, event, session_id, reason):
    """Append to the cron_events audit trail (ran/deferred/skipped...);
    wait-for-cron polls this table to detect an actual fire."""
    conn.execute(
        "INSERT INTO cron_events(job_id,job_name,event,session_id,reason,created_at)"
        " VALUES(?,?,?,?,?,?)",
        (job_id, job_name, event, session_id, reason, now()),
    )


def cmd_cron_guard(a):
    """Deterministic step-0 for CRON WRAPPER SCRIPTS (zero LLM tokens).
    Registers an ephemeral cron session + atomically claims the job's resources.
    stdout = coordination session id ONLY (safe under no_agent stdout-as-message:
    guard output goes to stderr). Exit: 0 acquired (caller MUST later run
    `done --id <id>`); 75 deferred per --policy (skip this tick); 1 error.
    Resources come from --res and/or the manifest entry for --job."""
    job_id, job_name = None, a.name or "cron job"
    resources = [norm_resource(r) for r in (a.res or [])]
    man = load_cron_manifest()
    if a.job:
        job_id, job_name, _store_row, err = resolve_cron_job(a.job)
        if err and not resources:
            print(f"cron-guard: {err}", file=sys.stderr)
            return 1
        if job_id and job_id in man:
            resources.extend(r for r in man[job_id]["resources"] if r not in resources)
    if not resources:
        print("cron-guard: no resources (give --res and/or a manifested --job)",
              file=sys.stderr)
        return 1
    conn = db()
    sid = uuid.uuid4().hex[:12]
    t = now()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    conn.execute(
        "INSERT INTO sessions(id,task,surface,started_at,last_seen,status)"
        " VALUES(?,?,?,?,?,'active')",
        (sid, f"cron: {job_name}" + (f" [{job_id[:12]}]" if job_id else ""),
         "cron", t, t),
    )
    conn.execute("COMMIT")

    class _A:  # minimal arg shim for the shared claim loop
        pass
    aa = _A()
    aa.id = sid
    aa.json = False
    aa.wait = (a.policy == "wait")
    aa.timeout = a.timeout
    aa.ttl = a.ttl
    pairs = [(r, "exclusive") for r in resources]

    # capture claim-loop chatter away from stdout (no_agent: stdout == message)
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc, payload, lines = _claim_loop(conn, aa, pairs,
                                         f"cron tick: {job_name}", verb="CLAIMED")
    for ln in lines:
        print(ln, file=sys.stderr)
    if rc == 0:
        log_cron_event(conn, job_id, job_name, "guarded-run", sid, None)
        print(sid)  # the ONE stdout line: callers capture GUARD_ID=$(...)
        return 0
    # deferred: tell the holder(s) a cron stepped aside for them
    conn.execute("BEGIN IMMEDIATE")
    for h in payload.get("holders", []):
        notify(conn, h["session_full"], sid, h["resource"], "cron_defer",
               f"FYI: cron job '{job_name}'" + (f" ({job_id[:12]})" if job_id else "")
               + f" fired, found you holding {h['resource']} (your task "
               f"'{h['task']}'), and politely "
               + ("waited until timeout, then skipped" if a.policy == "wait"
                  else "skipped")
               + " this tick. It runs again on its own schedule; if this was a "
               "CRITICAL job, consider asking the user to re-run it "
               "(cronjob action='run') once you release.",
               dedupe=True)
    ev = "wait-timeout" if a.policy == "wait" else "skipped"
    hold_desc = "; ".join(f"{h['resource']}<-{h['session']}({h['task']})"
                          for h in payload.get("holders", [])) or "queued/fenced"
    log_cron_event(conn, job_id, job_name, ev, sid, hold_desc)
    conn.execute("UPDATE sessions SET status='done' WHERE id=?", (sid,))
    conn.execute("COMMIT")
    print(f"cron-guard: DEFER ({ev}) — {hold_desc}", file=sys.stderr)
    return 75


def cmd_wait_for_cron(a):
    """Session-side: block (poll) until a cron job's tick completes — the
    'critical cron is about to fire; step aside and let it run' path.
    Completion = job's last_run_at advances past the fire we waited for, or its
    guarded-run/skip event lands in cron_events. Exit 0 = it ran; 75 = timeout;
    2 = it fired but SKIPPED (you still held its resources — it did NOT run)."""
    job_id, job_name, store_row, err = resolve_cron_job(a.job)
    if err:
        print(f"wait-for-cron: {err}", file=sys.stderr)
        return 1
    conn = db()
    touch_session(conn, a.id)
    start = now()
    base_last = (store_row or {}).get("last_ts") or 0
    next_ts = (store_row or {}).get("next_ts")
    deadline = start + a.timeout
    if next_ts and next_ts - start > a.timeout:
        print(f"wait-for-cron: '{job_name}' next fires in "
              f"{int((next_ts - start) / 60)}m — beyond --timeout "
              f"{int(a.timeout / 60)}m. Either raise --timeout, finish your work "
              f"first, or ask the USER to trigger it now (cronjob action='run', "
              f"then record cron-note --action triggered).", file=sys.stderr)
        return 75
    if not a.json:
        print(f"waiting for cron '{job_name}'"
              + (f" (fires in ~{max(0, int((next_ts - start) / 60))}m)" if next_ts else "")
              + " — polling...", flush=True)
    while now() < deadline:
        ev = conn.execute(
            "SELECT * FROM cron_events WHERE job_id=? AND created_at > ? "
            "ORDER BY created_at DESC LIMIT 1", (job_id, start - 1),
        ).fetchone()
        if ev and ev["event"] in ("guarded-run", "skipped", "wait-timeout"):
            if ev["event"] == "guarded-run":
                gid = ev["session_id"]
                # wait for the guard session to finish (done) or vanish
                while now() < deadline:
                    s = conn.execute("SELECT status FROM sessions WHERE id=?",
                                     (gid,)).fetchone()
                    if not s or s["status"] != "active":
                        emit({"ran": True, "job": job_id},
                             a.json, [(f"CRON RAN: '{job_name}' completed its tick. "
                                       f"Safe to (re)claim; VERIFY state if it touches "
                                       f"your files.")])
                        return 0
                    time.sleep(4)
                break
            emit({"ran": False, "deferred": ev["event"], "job": job_id}, a.json,
                 [(f"CRON DEFERRED: '{job_name}' fired but {ev['event']} (reason: "
                   f"{ev['reason']}). It did NOT run — likely YOU hold its resources. "
                   f"Release/pause first, then ask the user to re-run it "
                   f"(cronjob action='run'), or wait for its next schedule.")])
            return 2
        j = cron_store_jobs().get(job_id) or {}
        if (j.get("last_ts") or 0) > max(base_last, start - 60):
            emit({"ran": True, "job": job_id, "via": "store"}, a.json,
                 [(f"CRON RAN: '{job_name}' last_run_at advanced. Safe to (re)claim; "
                   f"VERIFY state if it touches your files.")])
            return 0
        time.sleep(4)
    emit({"ran": False, "timeout": True, "job": job_id}, a.json,
         [(f"TIMEOUT waiting for cron '{job_name}'. Check `cronjob list` / ask the "
           f"user; do not assume it ran.")])
    return 75


def cmd_cron_note(a):
    """Record a USER-approved intervention on a cron job (paused / resumed /
    triggered early) so the board carries who did it and why — a paused
    critical job must never be silently forgotten. This does NOT pause/trigger
    anything itself: use the Hermes cronjob tool for the action; this is the
    board-side receipt."""
    job_id, job_name, _, err = resolve_cron_job(a.job)
    if err:
        print(f"cron-note: {err}", file=sys.stderr)
        return 1
    conn = db()
    conn.execute("BEGIN IMMEDIATE")
    reap(conn)
    touch_session(conn, a.id)
    log_cron_event(conn, job_id, job_name, f"user-{a.action}", a.id, a.reason)
    conn.execute("COMMIT")
    lines = [f"NOTED: cron '{job_name}' {a.action} by session {short(a.id)}"
             + (f" — {a.reason}" if a.reason else "")]
    if a.action == "paused":
        lines.append(
            f"REMINDER: a paused job does not fire AT ALL. When your task is done: "
            f"resume it (cronjob action='resume'), run it if a tick was missed "
            f"(action='run'), and record: cron-note --job {job_id[:12]} "
            f"--action resumed")
    emit({"job": job_id, "action": a.action}, a.json, lines)
    return 0


# ---------------------------------------------------------------- master switch

def _synthetic_id():
    """A throwaway 12-hex id shaped like a real one, so `ID=$(... register)`
    still yields something callers can carry around while coordination is OFF."""
    return uuid.uuid4().hex[:12]


# Verbs that manage or read the switch itself always run, even while OFF.
_SWITCH_VERBS = {"switch", "enable", "disable"}

# Verbs that ACT ON AN EXISTING session addressed by --id. Their SQL is
# `WHERE id=?`/`session_id=?`, so a mistyped or truncated id matches no row and
# the UPDATE/SELECT silently affects nothing while the verb still reports
# success (the classic silent-no-op). Resolving --id up front — full id or
# unique prefix (>=4 chars), erroring on no-match/ambiguous — turns that quiet
# lie into an honest failure. Excluded on purpose: `register` (mints a NEW id),
# and `claim`/`check`/`wait` (read-side / self-attributed; a not-yet-registered
# id is legitimate there and must not hard-error).
_ID_RESOLVING_VERBS = {"release", "done", "inbox", "pause", "resume", "preempt"}


def disabled_noop(a):
    """Friendly no-op for every coordination verb while the master switch is
    OFF. It never blocks a caller and preserves each verb's stdout contract, so
    a disabled board behaves exactly like 'coordination not installed'. Return
    code is always 0 (proceed) -- OFF must never turn into a blocked caller."""
    cmd = a.cmd
    j = getattr(a, "json", False)
    note = (f"coordination DISABLED ({switch_source()}) -- no-op; "
            f"run 'session_coord.py enable' to turn it back on")
    if cmd == "register":
        sid = (getattr(a, "id", None) or "").strip() or _synthetic_id()
        emit({"id": sid, "disabled": True}, j, [sid])
        if not j:
            print(f"# {note}", file=sys.stderr)
        return 0
    if cmd == "cron-guard":
        # stdout MUST stay empty: the shell guard reads stdout as the guard id
        # and treats empty/non-hex output as fail-open (proceed unguarded).
        print(f"cron-guard: {note}", file=sys.stderr)
        return 0
    if cmd == "claim":
        emit({"claimed": True, "disabled": True}, j,
             ["CLAIMED (coordination disabled -- no board enforcement)"])
        return 0
    if cmd == "check":
        emit({"free": True, "disabled": True}, j, ["FREE (coordination disabled)"])
        return 0
    if cmd == "status":
        emit({"disabled": True, "source": switch_source()}, j,
             ["coordination board: DISABLED",
              f"  reason: {switch_source()}",
              "  re-enable with: session_coord.py enable"])
        return 0
    # wait / release / done / inbox / prioritize / preempt / pause / resume /
    # steal / wait-for-cron / cron-note: nothing to do -- succeed quietly.
    if j:
        emit({"disabled": True, "noop": cmd}, j, [])
    else:
        print(f"# {note}", file=sys.stderr)
    return 0


def cmd_switch(a):
    """Read or set the master switch. `enable`/`disable` set it persistently via
    a sentinel file (COORD_DISABLED_FILE); `switch` alone reports; `switch
    on|off|toggle` also sets it. The env var HERMES_COORD_DISABLED, when set,
    overrides the file for its process tree and is reported honestly."""
    arg = getattr(a, "state", None)
    if a.cmd == "enable" or arg in ("on", "enable"):
        want = True
    elif a.cmd == "disable" or arg in ("off", "disable"):
        want = False
    elif arg == "toggle":
        want = not coordination_enabled()
    else:  # 'status' or no argument -> report only
        want = None

    env = os.environ.get("HERMES_COORD_DISABLED")
    env_str = (env or "").strip()
    env_forcing = env_str != ""

    if want is None:
        present = "present" if os.path.exists(COORD_DISABLED_FILE) else "absent"
        state = "ENABLED" if coordination_enabled() else "DISABLED"
        lines = [f"coordination: {state}",
                 f"  decided by: {switch_source()}",
                 f"  sentinel:   {COORD_DISABLED_FILE} ({present})"]
        if env_forcing:
            lines.append(f"  NOTE: env HERMES_COORD_DISABLED={env_str!r} "
                         "overrides the sentinel for this process tree.")
        emit({"enabled": coordination_enabled(), "source": switch_source(),
              "sentinel": COORD_DISABLED_FILE,
              "sentinel_present": os.path.exists(COORD_DISABLED_FILE)}, a.json, lines)
        return 0

    if want:  # enable -> remove the sentinel
        try:
            os.remove(COORD_DISABLED_FILE)
            changed = True
        except FileNotFoundError:
            changed = False
    else:     # disable -> write the sentinel
        d = os.path.dirname(COORD_DISABLED_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        changed = not os.path.exists(COORD_DISABLED_FILE)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(COORD_DISABLED_FILE, "w", encoding="utf-8") as f:
            f.write(f"coordination disabled at {int(now())} ({stamp})\n")

    now_enabled = coordination_enabled()
    tail = "" if changed else " (already in that state)"
    lines = [f"coordination {'ENABLED' if want else 'DISABLED'}{tail}",
             f"  effective now: {'ENABLED' if now_enabled else 'DISABLED'} ({switch_source()})"]
    if env_forcing and now_enabled != want:
        lines.append(f"  WARNING: env HERMES_COORD_DISABLED={env_str!r} is "
                     "OVERRIDING the sentinel -- unset it for the file to take effect.")
    emit({"enabled": now_enabled, "requested": want, "changed": changed,
          "source": switch_source()}, a.json, lines)
    return 0


def main():
    """CLI entrypoint: build the argparse tree and dispatch to cmd_*."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, needs_res=False):
        """Attach the standard flags (--id/--json/--res...) to a subparser."""
        sp.add_argument("--id", default=os.environ.get("HERMES_COORD_ID"),
                        help="session coordination id (or env HERMES_COORD_ID)")
        sp.add_argument("--json", action="store_true")
        if needs_res:
            sp.add_argument("--res", action="append", required=True,
                            help="resource key (repeatable)")
            sp.add_argument("--mode", choices=["exclusive", "shared"], default="exclusive")

    sp = sub.add_parser("register", help="announce this session + its task")
    sp.add_argument("--id", default=os.environ.get("HERMES_COORD_ID"),
                    help="explicit session id to register under (or env "
                         "HERMES_COORD_ID); omit for an auto-generated one")
    sp.add_argument("--task", required=True)
    sp.add_argument("--surface", default=None, help="desktop/cli/telegram/cron/subagent/...")
    sp.add_argument("--parent", default=None,
                    help="parent session id — this session is its subagent; inherits "
                         "family rank")
    sp.add_argument("--slot", default=None,
                    help="sub-priority under the parent (a, b, c...) — parent assigns; "
                         "only matters on collision")
    sp.add_argument("--rank", default=None,
                    help="user-granted root rank (1, 2, 3...) if the user already "
                         "stated priority")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_register)

    sp = sub.add_parser("claim", help="atomically claim resource(s) for the TASK duration")
    common(sp, needs_res=True)
    sp.add_argument("--task", default=None)
    sp.add_argument("--ttl", type=float, default=DEFAULT_TTL_MIN,
                    help=f"minutes before claim auto-expires (default {DEFAULT_TTL_MIN:g}); "
                         "re-claim to refresh on long tasks")
    sp.add_argument("--wait", action="store_true",
                    help="politely wait (poll) if held or queued; registers as waiter")
    sp.add_argument("--timeout", type=float, default=300)
    sp.set_defaults(fn=cmd_claim)

    sp = sub.add_parser("check", help="read-only: is the resource free?")
    common(sp, needs_res=True)
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("wait", help="wait until free WITHOUT claiming (read-side)")
    common(sp, needs_res=True)
    sp.add_argument("--timeout", type=float, default=300)
    sp.add_argument("--note", default=None)
    sp.set_defaults(fn=cmd_wait)

    sp = sub.add_parser("release", help="release specific resources (or --all)")
    common(sp)
    sp.add_argument("--res", action="append", default=[])
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(fn=cmd_release)

    sp = sub.add_parser(
        "done", help="TASK finished: release everything, notify waiters, deregister")
    common(sp)
    sp.set_defaults(fn=cmd_done)

    sp = sub.add_parser("inbox", help="read my notifications (marks them read)")
    common(sp)
    sp.set_defaults(fn=cmd_inbox)

    sp = sub.add_parser("status", help="the whole board: sessions, claims, waiters, warnings")
    common(sp)
    sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("prioritize", help="record the USER's priority order (live re-fence)")
    sp.add_argument("--id", default=os.environ.get("HERMES_COORD_ID"),
                    help="requesting session (attribution only)")
    sp.add_argument("--session", default=None, help="target session id/prefix")
    sp.add_argument("--rank", default=None, help="root rank: 1, 2, 3 (or 1a)")
    sp.add_argument("--slot", default=None, help="sub-slot under parent: a, b, c")
    sp.add_argument("--clear", action="store_true", help="clear target's rank")
    sp.add_argument("--order", default=None,
                    help="bulk: 'idA=1,idB=2,idC=3' (prefixes ok) — the do-1-then-2-then-3 form")
    sp.add_argument("--clear-all", action="store_true")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_prioritize)

    sp = sub.add_parser("preempt",
                        help="relay USER priority to holder(s): ask them to checkpoint+pause")
    common(sp, needs_res=True)
    sp.add_argument("--reason", default=None)
    sp.set_defaults(fn=cmd_preempt)

    sp = sub.add_parser("pause",
                        help="holder: park my claims (checkpoint FIRST); keeps resume spot")
    common(sp)
    sp.add_argument("--res", action="append", default=[],
                    help="subset to pause (default: all held)")
    sp.add_argument("--note", default=None,
                    help="checkpoint note: where you saved state (strongly recommended)")
    sp.add_argument("--ttl", type=float, default=None,
                    help="minutes the paused spot survives (default: claim's ttl)")
    sp.set_defaults(fn=cmd_pause)

    sp = sub.add_parser("resume", help="re-acquire my paused claims (fence-respecting)")
    common(sp)
    sp.add_argument("--wait", action="store_true")
    sp.add_argument("--timeout", type=float, default=300)
    sp.add_argument("--ttl", type=float, default=None)
    sp.set_defaults(fn=cmd_resume)

    sp = sub.add_parser("steal", help="force-release a co-worker's claim (LAST RESORT)")
    common(sp, needs_res=True)
    sp.add_argument("--reason", required=True)
    sp.set_defaults(fn=cmd_steal)

    sp = sub.add_parser("cron-guard",
                        help="STEP-0 FOR CRON WRAPPERS: claim the job's resources or defer "
                             "(exit 75). Zero-LLM; stdout = guard session id only")
    sp.add_argument("--job", default=None,
                    help="Hermes cron job id or exact name (resolves resources from the "
                         "manifest + last/next run from the cron store)")
    sp.add_argument("--name", default=None, help="label if --job is not resolvable")
    sp.add_argument("--res", action="append", default=[],
                    help="extra/explicit resource keys (repeatable)")
    sp.add_argument("--policy", choices=["skip", "wait"], default="skip",
                    help="on collision: skip this tick (default) or wait up to --timeout")
    sp.add_argument("--timeout", type=float, default=600)
    sp.add_argument("--ttl", type=float, default=DEFAULT_TTL_MIN)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_cron_guard)

    sp = sub.add_parser("wait-for-cron",
                        help="session: block until a cron job's tick completes (the "
                             "'critical cron fires soon — let it run first' path)")
    common(sp)
    sp.add_argument("--job", required=True, help="cron job id or exact name")
    sp.add_argument("--timeout", type=float, default=3600,
                    help="seconds to wait (default 1h); refuses fast if next fire is "
                         "beyond this")
    sp.set_defaults(fn=cmd_wait_for_cron)

    sp = sub.add_parser("cron-note",
                        help="record a USER-approved cron intervention (paused/resumed/"
                             "triggered) — the board-side receipt, not the action itself")
    common(sp)
    sp.add_argument("--job", required=True)
    sp.add_argument("--action", required=True,
                    choices=["paused", "resumed", "triggered"])
    sp.add_argument("--reason", default=None)
    sp.set_defaults(fn=cmd_cron_note)

    sp = sub.add_parser("switch",
                        help="MASTER SWITCH: report state, or on|off|toggle the whole board")
    sp.add_argument("state", nargs="?",
                    choices=["on", "off", "toggle", "status"], default=None,
                    help="omit to report; on/off/toggle to set persistently")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_switch)

    sp = sub.add_parser("enable",
                        help="MASTER SWITCH: turn the coordination board ON (persistent)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_switch)

    sp = sub.add_parser("disable",
                        help="MASTER SWITCH: turn the board OFF — every verb becomes a "
                             "fail-open no-op (persistent)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(fn=cmd_switch)

    a = p.parse_args()
    # Master switch: while OFF, every coordination verb (but not the switch-
    # management verbs themselves) short-circuits to a fail-open no-op.
    if a.cmd not in _SWITCH_VERBS and not coordination_enabled():
        sys.exit(disabled_noop(a))
    # Resolve --id (full id OR unique prefix) to a real session for the verbs
    # that act on one, so a truncated/typo'd id fails loudly instead of matching
    # nothing and reporting success. The board DISPLAYS 8-char ids but stores
    # 12, which is exactly how an operator ends up pasting a short id.
    if a.cmd in _ID_RESOLVING_VERBS and getattr(a, "id", None):
        try:
            rconn = db()
            sid, err = resolve_session(rconn, a.id)
            rconn.close()
        except sqlite3.OperationalError:
            sid, err = None, None  # DB down: fail open, let the verb's own guard handle it
        if err:
            print(f"error: {err}", file=sys.stderr)
            sys.exit(2)
        if sid:
            a.id = sid
    try:
        sys.exit(a.fn(a))
    except sqlite3.OperationalError as e:
        # Fail open (like Hermes' own lease file): coordination must never strand a session.
        print(f"WARNING: coordination DB unavailable ({e}). Proceed WITH CAUTION — "
              "you cannot see co-worker sessions right now.", file=sys.stderr)
        sys.exit(0 if a.cmd in ("check", "wait", "status", "inbox") else 1)


if __name__ == "__main__":
    main()
