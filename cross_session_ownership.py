"""Cross-session task ownership — isolated implementation.

Reuses the existing Kanban SQLite DB and the Kanban claim/lease pattern
(``BEGIN IMMEDIATE`` + compare-and-swap on status/claim_lock). Adds ONE
ownership ledger table plus an audit-events table. It does NOT create a second
task manager, dispatcher or queue: the existing Kanban DB remains the single
task-state truth; this layer only answers "which live session owns this work
item right now".

Why this layer exists
---------------------
The Kanban claim/lease protects a *worker run*. It does not tell a freshly
started BTA chat session "another live BTA session is already handling GitHub
issue #33". Ownership facts are scattered across:

* ``task_events.owner_admitted.session_id``  (Kanban worker admission)
* ``task_runs.claim_lock`` / ``claim_expires`` (per-run worker lease)
* ``async_delegations.origin_session``        (delegate provenance)
* ``session_turn_leases``                     (turn-scoped chat liveness)

Nothing surfaces them as one query a model can run before dispatching.

Liveness reality (measured 2026-11-10)
--------------------------------------
* ``_HERMES_HOME/runtime/active_sessions.json`` = ``{"entries": []}`` — desktop
  chat sessions do not hold a registry slot between turns.
* ``gateway_heartbeats`` rows are ~15-18 days stale — not a liveness signal.
* ``session_turn_leases`` is turn-scoped (TTL ~10 min) — a session idle between
  turns holds nothing.

Therefore the ownership layer is **TTL + heartbeat** driven, with liveness as a
secondary hint. Takeover requires the lease to be expired (``STALE``) or the
owner provably dead past a grace window (``ORPHANED``).

Scope key formats
-----------------
``gh:<owner>/<repo>#<number>``   GitHub issue/PR
``kanban:<board>:<task_id>``     Kanban card
``adhoc:<opaque-id>``            any other work item
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sqlite3
import time
import uuid
from typing import Any, Callable, Iterable, Optional

__all__ = [
    "ALLOWED_SCOPE_PREFIXES",
    "DEFAULT_TTL_SECONDS",
    "MAX_SCOPE_KEY_LEN",
    "MAX_SESSION_ID_LEN",
    "MAX_TTL_SECONDS",
    "ORPHAN_GRACE_SECONDS",
    "STATE_ACTIVE",
    "STATE_ORPHANED",
    "STATE_RELEASED",
    "STATE_TAKEN_OVER",
    "VERDICT_FREE",
    "VERDICT_ORPHANED",
    "VERDICT_OWNED",
    "VERDICT_STALE",
    "LivenessProvider",
    "OwnershipError",
    "claim_ownership",
    "github_scope",
    "heartbeat_ownership",
    "init_schema",
    "inspect_ownership",
    "list_ownership",
    "normalize_repo",
    "prune_stale",
    "release_ownership",
    "scope_key",
    "takeover_ownership",
    "validate_scope_key",
    "validate_session_id",
]


class OwnershipError(RuntimeError):
    """The ownership ledger could not prove a safe decision."""


DEFAULT_TTL_SECONDS = 1800          # 30 minutes
MAX_TTL_SECONDS = 86400             # hard cap: 24h (NEO condition)
ORPHAN_GRACE_SECONDS = 300         # dead owner: takeover only after this grace
HEARTBEAT_EXTENSION = True         # heartbeat extends claim_expires

ALLOWED_SCOPE_PREFIXES = ("gh:", "kanban:", "adhoc:")
MAX_SCOPE_KEY_LEN = 1024
MAX_SESSION_ID_LEN = 256
# Session ids are Hermes-generated: ``local-<hex>``, ``<timestamp>_<hex>``,
# ``sa-<n>-<hex>``, ``agent:<profile>:<transport>:<kind>:<id>``. Restrict to
# the character class those use so arbitrary strings cannot be claimed.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")

VERDICT_FREE = "FREE"              # nobody owns it
VERDICT_OWNED = "OWNED"            # a live, unexpired owner holds it
VERDICT_STALE = "STALE"            # lease expired -> takeover allowed
VERDICT_ORPHANED = "ORPHANED"      # owner provably dead past grace -> takeover allowed

STATE_ACTIVE = "ACTIVE"
STATE_RELEASED = "RELEASED"
STATE_TAKEN_OVER = "TAKEN_OVER"
STATE_ORPHANED = "ORPHANED"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS task_ownership (
    scope_key      TEXT PRIMARY KEY,
    owner_session  TEXT NOT NULL,
    owner_profile  TEXT,
    owner_pid      INTEGER,
    owner_epoch    TEXT,
    claim_lock     TEXT NOT NULL,
    claim_expires  INTEGER NOT NULL,
    last_heartbeat INTEGER NOT NULL,
    state          TEXT NOT NULL DEFAULT 'ACTIVE',
    checkpoint     TEXT,
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_ownership_session
    ON task_ownership(owner_session);
CREATE INDEX IF NOT EXISTS idx_task_ownership_expiry
    ON task_ownership(claim_expires);
CREATE TABLE IF NOT EXISTS task_ownership_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_key  TEXT NOT NULL,
    event      TEXT NOT NULL,
    actor      TEXT,
    detail     TEXT,
    created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_ownership_events_scope
    ON task_ownership_events(scope_key, id);
"""


def scope_key(prefix: str, identifier: str) -> str:
    """Build a canonical scope key, e.g. ``gh:Andrew288369/ANDREW-GRP#33``."""
    ident = str(identifier or "").strip()
    if not ident:
        raise OwnershipError("scope identifier must be non-empty")
    return validate_scope_key(f"{prefix}:{ident}")


def github_scope(owner: str, repo: str, number: int | str) -> str:
    """Canonical ``gh:<owner>/<repo>#<n>`` — owner/repo lowercased, ``.git`` and
    ``github.com`` stripped. GitHub owner/repo is case-insensitive, so both the
    Direct and the Kanban path MUST build the key through this one function or
    the same work item yields two different keys (NEO F8 / SAGE S3)."""
    slug = normalize_repo(owner, repo)
    if not slug:
        raise OwnershipError(f"invalid owner/repo: {owner!r}/{repo!r}")
    return scope_key("gh", f"{slug}#{number}")


# --- canonical work-key derivation (ONE shared normalizer, Direct + Kanban) ---

_REPO_URL_RE = re.compile(r"(?i)^https?://(?:www\.)?github\.com/")
_REPO_HOST_RE = re.compile(r"(?i)^(?:www\.)?github\.com/")
_GH_CARD_RE = re.compile(r"(?i)^GH[-_]?(\d+)$")
_LEGACY_IDEM_RE = re.compile(r"(?i)^gh-issue:(.+):(\d+)$")
_BODY_KV_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*:\s*(.*?)\s*$")
_TITLE_GH_RE = re.compile(r"(?i)\bGH[-_]?(\d+)\b")


def normalize_repo(owner: str, repo: Optional[str] = None) -> str:
    """Return the canonical ``owner/repo`` slug, or ``""`` when unparseable.

    Lowercase (GitHub owner/repo is case-insensitive), strip a scheme, a
    ``github.com`` host, surrounding slashes and a trailing ``.git``. Shared by
    every key builder so Direct and Kanban can never disagree on casing.
    """
    slug = f"{owner}/{repo}" if repo is not None else str(owner or "")
    slug = _REPO_URL_RE.sub("", slug.strip())
    slug = _REPO_HOST_RE.sub("", slug.strip()).strip().strip("/")
    if slug.lower().endswith(".git"):
        slug = slug[: -len(".git")]
    # GitHub names cannot contain '#': drop any fragment a caller left behind
    # (e.g. ``owner/repo#33`` passed as a slug).
    parts = [p.split("#", 1)[0] for p in slug.split("/") if p.split("#", 1)[0]]
    if len(parts) < 2:
        return ""
    return f"{parts[0].lower()}/{parts[1].lower()}"


def _issue_number(text: Any) -> Optional[int]:
    """A bare ``#?<digits>`` issue number; ``None`` for anything else."""
    m = re.fullmatch(r"#?(\d+)", str(text or "").strip())
    return int(m.group(1)) if m else None


def _body_fields(task: dict) -> dict[str, str]:
    """Lowercased ``key: value`` lines from the card body; first occurrence wins."""
    fields: dict[str, str] = {}
    for line in str(task.get("body") or "").splitlines():
        m = _BODY_KV_RE.match(line)
        if m:
            fields.setdefault(m.group(1).lower(), m.group(2).strip())
    return fields


def canonical_key_from_kanban(
    task: dict,
    *,
    default_repo: Optional[str] = None,
    diagnostics: Optional[list] = None,
) -> Optional[str]:
    """Derive the CANONICAL work key from a Kanban task row.

    The Kanban ``task_id`` is not the identity of the logical work (#47 L0
    correction). The canonical key must be the same whether the work was
    dispatched Direct or through a card, so Direct and Kanban collide.

    Priority (RFC v2 §1) — derived from the card's OWN identity fields, never
    from free text that may reference some other issue (NEO F4):

      (a) body ``source_repo`` + ``source_issue``
      (b) body ``task_id: GH-n`` or title ``[GH-n ...]``, WITH a repo on the card
      (c) ``idempotency_key`` ``GH-n``, WITH a repo on the card
      (d) legacy ``idempotency_key`` ``gh-issue:<owner>/<repo>:<n>``
      (e) fallback ``kanban:<task_id>`` (unique, but never collides with Direct)

    C7 (#47 Phase 2) — no assumed repo. A bare ``GH-n`` with NO ``source_repo``
    on the card used to be mapped to ``DEFAULT_REPO``
    (``Andrew288369/ANDREW-GRP``), so ``GH-33`` in *any* repo collapsed onto
    ``gh:andrew288369/andrew-grp#33`` — a cross-repo false collision: two
    different work items sharing one ledger row, one of them silently locked
    out. Branches (b)/(c) now require a repo from the card itself. Without one
    they fall through to (e), the card's own unique id, and record a
    ``fail_closed:*`` diagnostic.

    ``default_repo`` is EXPLICIT-ONLY (``None`` by default): no caller can
    silently assume a repository. Pass it only when the caller genuinely knows
    the board is pinned to one repo.

    ``diagnostics`` (optional list) receives one reason code per call:
    ``source_repo+source_issue``, ``gh_number_with_repo``,
    ``gh_number_with_explicit_repo``, ``legacy_gh_issue``, ``card_id``,
    ``fail_closed:gh_number_without_repo`` or ``no_identity``.
    """
    if not isinstance(task, dict):
        return None
    fields = _body_fields(task)
    title = str(task.get("title") or "")
    idem = str(task.get("idempotency_key") or "").strip()
    default_slug = normalize_repo(default_repo) if default_repo else ""
    fail_closed = False

    def _diag(code: str) -> None:
        if diagnostics is not None:
            diagnostics.append(code)

    # (a) structured source_repo + source_issue
    repo = normalize_repo(fields.get("source_repo", ""))
    issue = _issue_number(fields.get("source_issue"))
    if repo and issue is not None:
        _diag("source_repo+source_issue")
        return scope_key("gh", f"{repo}#{issue}")

    # (b) task_id: GH-n (body) or [GH-n ...] (title); repo must be on the card
    card = fields.get("task_id", "").strip()
    number: Optional[int] = None
    if _GH_CARD_RE.match(card):
        number = int(_GH_CARD_RE.match(card).group(1))  # type: ignore[union-attr]
    else:
        m = _TITLE_GH_RE.search(title)
        if m:
            number = int(m.group(1))
    if number is not None:
        if repo:
            _diag("gh_number_with_repo")
            return scope_key("gh", f"{repo}#{number}")
        if default_slug:
            _diag("gh_number_with_explicit_repo")
            return scope_key("gh", f"{default_slug}#{number}")
        fail_closed = True

    # (c) idempotency_key GH-n; repo must be on the card or explicitly supplied
    m = _GH_CARD_RE.match(idem)
    if m:
        if repo:
            _diag("gh_number_with_repo")
            return scope_key("gh", f"{repo}#{m.group(1)}")
        if default_slug:
            _diag("gh_number_with_explicit_repo")
            return scope_key("gh", f"{default_slug}#{m.group(1)}")
        fail_closed = True

    if fail_closed:
        _diag("fail_closed:gh_number_without_repo")

    # (d) legacy gh-issue:<owner>/<repo>:<n>
    m = _LEGACY_IDEM_RE.match(idem)
    if m:
        slug = normalize_repo(m.group(1))
        if slug:
            _diag("legacy_gh_issue")
            return scope_key("gh", f"{slug}#{m.group(2)}")

    # (e) fallback: the card's own task_id
    task_id = str(task.get("id") or "").strip()
    if task_id:
        _diag("card_id")
        return scope_key("kanban", task_id)
    _diag("no_identity")
    return None


def claim_for_dispatch(
    conn: sqlite3.Connection,
    *,
    mode: str,
    session_id: str,
    scope: Optional[str] = None,
    task: Optional[dict] = None,
    **kwargs,
) -> dict:
    """Mode-agnostic claim for a dispatch about to happen.

    ``mode='direct'`` — ``scope`` comes from the delegate goal/context.
    ``mode='kanban'`` — ``scope`` is derived from the card via
    :func:`canonical_key_from_kanban`, so a card and a direct dispatch for the
    same logical work cannot both pass.
    """
    mode = str(mode or "").strip().lower()
    if mode not in ("direct", "kanban"):
        raise OwnershipError(f"unknown dispatch mode: {mode!r}")
    if scope is None:
        if mode == "kanban" and task is not None:
            scope = canonical_key_from_kanban(task)
        if not scope:
            raise OwnershipError(
                "cannot derive canonical scope for dispatch (fail-closed)"
            )
    return claim_ownership(conn, scope, session_id, **kwargs)


def validate_scope_key(scope: str) -> str:
    """Reject keys that are empty, oversized, off-prefix, or whitespace-bearing.

    NEO condition: the ledger must not accept arbitrary keys, otherwise a
    caller can pollute it with scopes that collide with nothing but look real.
    """
    s = str(scope or "").strip()
    if not s:
        raise OwnershipError("scope_key must be non-empty")
    if len(s) > MAX_SCOPE_KEY_LEN:
        raise OwnershipError(f"scope_key too long (>{MAX_SCOPE_KEY_LEN} chars)")
    if not s.startswith(ALLOWED_SCOPE_PREFIXES):
        raise OwnershipError(
            f"scope_key must start with one of {ALLOWED_SCOPE_PREFIXES}"
        )
    if any(ch.isspace() for ch in s):
        raise OwnershipError("scope_key must not contain whitespace")
    return s


def validate_session_id(session_id: str) -> str:
    """Reject session ids outside the Hermes character class (NEO condition).

    This is format validation, NOT authentication: production wiring must
    additionally bind the claim to a verifiable session proof (turn lease or
    signed token). The format gate stops arbitrary strings and obvious
    impersonation typos; it is not a substitute for identity assurance.
    """
    s = str(session_id or "")
    if not s:
        raise OwnershipError("session_id must be non-empty")
    if len(s) > MAX_SESSION_ID_LEN or not SESSION_ID_RE.match(s):
        raise OwnershipError(
            "session_id format invalid (allowed: letters, digits, '_', '.', ':', '-')"
        )
    return s


_SCHEMA_STATEMENTS = [s.strip() for s in _SCHEMA.split(";") if s.strip()]


def schema_statements() -> list[str]:
    """The individual DDL statements, for callers managing their own transaction."""
    return list(_SCHEMA_STATEMENTS)


def init_schema(conn: sqlite3.Connection) -> None:
    """Create the ownership tables if missing. Idempotent.

    Executes statement-by-statement (NOT ``executescript``): ``executescript``
    issues an implicit COMMIT first, which would silently destroy a caller's
    surrounding ``BEGIN IMMEDIATE`` transaction.
    """
    for stmt in _SCHEMA_STATEMENTS:
        conn.execute(stmt)


def _now() -> int:
    return int(time.time())


def _host_prefix() -> str:
    return socket.gethostname()


def _new_claim_lock() -> str:
    return f"{_host_prefix()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def _append_event(
    conn: sqlite3.Connection, scope: str, event: str,
    actor: Optional[str] = None, detail: Optional[dict] = None,
    now: Optional[int] = None,
) -> None:
    """Append an audit event. Uses the caller's logical clock when supplied so
    deterministic tests and the lease math stay on the same timeline."""
    conn.execute(
        "INSERT INTO task_ownership_events(scope_key, event, actor, detail, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (scope, event, actor, json.dumps(detail, ensure_ascii=False) if detail else None,
         int(now if now is not None else _now())),
    )


def _row_to_dict(row, cursor: Optional[sqlite3.Cursor] = None) -> dict:
    """Convert a SQLite row to a dict whether or not ``row_factory`` is set."""
    if isinstance(row, sqlite3.Row):
        return {k: row[k] for k in row.keys()}
    if cursor is not None:
        cols = [d[0] for d in cursor.description]
        return dict(zip(cols, tuple(row)))
    raise OwnershipError("row shape unsupported (no cursor description)")


def _get(conn: sqlite3.Connection, scope: str) -> Optional[dict]:
    cur = conn.execute("SELECT * FROM task_ownership WHERE scope_key = ?", (scope,))
    row = cur.fetchone()
    return _row_to_dict(row, cur) if row is not None else None


LivenessProvider = Callable[[str], str]
"""Returns 'ALIVE' | 'DEAD' | 'UNKNOWN' for a session id."""


def _liveness_unknown(_session_id: str) -> str:
    return "UNKNOWN"


def _liveness_of(owner: Optional[dict], provider: LivenessProvider,
                 now: int, orphan_grace: int) -> tuple[str, str]:
    """Return ``(verdict, reason)`` for an existing row."""
    if owner is None:
        return VERDICT_FREE, "no_active_owner"
    if str(owner.get("state") or "") != STATE_ACTIVE:
        return VERDICT_FREE, f"previous_state:{owner.get('state')}"
    expires = int(owner.get("claim_expires") or 0)
    last_hb = int(owner.get("last_heartbeat") or 0)
    if expires < now:
        return VERDICT_STALE, f"lease_expired({now - expires}s ago)"
    state = provider(str(owner.get("owner_session") or ""))
    if state == "DEAD":
        # Strictly greater: takeover only after the grace has fully elapsed.
        if now - last_hb > orphan_grace:
            return VERDICT_ORPHANED, f"owner_dead_and_grace_elapsed({now - last_hb}s)"
        return VERDICT_OWNED, f"owner_reported_dead_but_within_grace({now - last_hb}s)"
    return VERDICT_OWNED, "lease_valid"


def _validate_ttl(ttl_seconds: int) -> int:
    ttl = int(ttl_seconds)
    if ttl <= 0:
        raise OwnershipError("ttl_seconds must be positive")
    if ttl > MAX_TTL_SECONDS:
        raise OwnershipError(
            f"ttl_seconds {ttl} exceeds MAX_TTL_SECONDS ({MAX_TTL_SECONDS})"
        )
    return ttl


def claim_ownership(
    conn: sqlite3.Connection,
    scope: str,
    session_id: str,
    *,
    profile: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    checkpoint: Optional[dict] = None,
    liveness: LivenessProvider = _liveness_unknown,
    verify_session: Optional[Callable[[str], bool]] = None,
    now: Optional[int] = None,
    orphan_grace: int = ORPHAN_GRACE_SECONDS,
    join_txn: bool = False,
) -> dict:
    """Atomically claim ``scope`` for ``session_id``.

    Returns ``{"ok": bool, "verdict": str, "reason": str, "owner": str|None,
    "state": str}``. ``ok=True`` only when this caller became (or already is)
    the owner. Never raises for an ordinary conflict — a lost race is a normal
    result, not an error.

    ``verify_session`` (optional) is the production identity hook: when given,
    it must return True for the session to be allowed to claim. Format
    validation alone is not identity assurance.

    ``join_txn=True`` runs inside a write transaction the caller already opened
    (no BEGIN/COMMIT of its own), so the claim commits or rolls back with the
    caller's own writes. Required by the Kanban admission gate, which must not
    create a second, independent lock.
    """
    scope = validate_scope_key(scope)
    session_id = validate_session_id(session_id)
    now = int(now if now is not None else _now())
    ttl = _validate_ttl(ttl_seconds)
    if verify_session is not None and not verify_session(session_id):
        raise OwnershipError("session identity verification failed")

    lock = _new_claim_lock()
    result = {"ok": False, "verdict": None, "reason": None,
              "owner": None, "state": None, "scope": scope}

    with _write_txn(conn, join=join_txn):
        row = _get(conn, scope)

        if row is None:
            conn.execute(
                "INSERT INTO task_ownership"
                "(scope_key, owner_session, owner_profile, owner_pid, owner_epoch,"
                " claim_lock, claim_expires, last_heartbeat, state, checkpoint,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (scope, session_id, profile, os.getpid(), None, lock,
                 now + ttl, now, STATE_ACTIVE,
                 json.dumps(checkpoint, ensure_ascii=False) if checkpoint else None,
                 now, now),
            )
            _append_event(conn, scope, "claimed", session_id,
                          {"ttl": ttl, "lock": lock}, now=now)
            result.update(ok=True, verdict=VERDICT_OWNED, reason="claimed",
                          owner=session_id, state=STATE_ACTIVE)
            return result

        verdict, reason = _liveness_of(row, liveness, now, orphan_grace)
        result["owner"] = row["owner_session"]
        result["state"] = row["state"]

        # Same session re-claiming its own ACTIVE scope is idempotent
        # re-assertion, not a second owner.
        if (row["owner_session"] == session_id and row["state"] == STATE_ACTIVE
                and verdict == VERDICT_OWNED):
            conn.execute(
                "UPDATE task_ownership SET last_heartbeat = ?, claim_expires = ?,"
                " updated_at = ? WHERE scope_key = ? AND owner_session = ?",
                (now, now + ttl, now, scope, session_id),
            )
            _append_event(conn, scope, "reasserted", session_id, {"ttl": ttl}, now=now)
            result.update(ok=True, verdict=VERDICT_OWNED, reason="already_owner",
                          owner=session_id, state=STATE_ACTIVE)
            return result

        if verdict == VERDICT_OWNED:
            _append_event(conn, scope, "claim_conflict", session_id,
                          {"owner": row["owner_session"],
                           "reason": "owned_by_other_session"}, now=now)
            result.update(ok=False, verdict=VERDICT_OWNED,
                          reason=f"owned_by_other_session:{row['owner_session']}")
            return result

        # FREE / STALE / ORPHANED / non-ACTIVE -> claim via compare-and-swap.
        prev = row["owner_session"]
        cur = conn.execute(
            "UPDATE task_ownership SET owner_session = ?, owner_profile = ?,"
            " owner_pid = ?, owner_epoch = NULL, claim_lock = ?, claim_expires = ?,"
            " last_heartbeat = ?, state = ?, checkpoint = ?, updated_at = ?"
            " WHERE scope_key = ? AND claim_lock = ?",
            (session_id, profile, os.getpid(), lock, now + ttl, now, STATE_ACTIVE,
             json.dumps(checkpoint, ensure_ascii=False) if checkpoint else None,
             now, scope, row["claim_lock"]),
        )
        if cur.rowcount == 0:  # CAS lost to a concurrent writer
            result.update(ok=False, verdict=verdict, reason="takeover_cas_lost",
                          owner=prev)
            return result
        _append_event(conn, scope, "taken_over", session_id,
                      {"from": prev, "verdict": verdict, "reason": reason}, now=now)
        result.update(ok=True, verdict=VERDICT_OWNED, owner=session_id,
                      reason=f"taken_over_from:{prev}", state=STATE_ACTIVE)
        return result


def heartbeat_ownership(
    conn: sqlite3.Connection,
    scope: str,
    session_id: str,
    *,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    checkpoint: Optional[dict] = None,
    expected_lock: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Renew the lease. Fails when the caller is not the current owner.

    ``expected_lock`` adds a compare-and-swap on ``claim_lock`` (NEO condition):
    a caller holding a stale lock reference cannot renew after a takeover.
    """
    scope = validate_scope_key(scope)
    session_id = validate_session_id(session_id)
    now = int(now if now is not None else _now())
    ttl = _validate_ttl(ttl_seconds)
    with _write_txn(conn):
        row = _get(conn, scope)
        if row is None:
            return {"ok": False, "reason": "no_owner", "scope": scope}
        if row["owner_session"] != session_id:
            return {"ok": False, "reason": f"not_owner:{row['owner_session']}",
                    "scope": scope, "owner": row["owner_session"]}
        if int(row["claim_expires"]) < now:
            return {"ok": False, "reason": "lease_expired", "scope": scope,
                    "owner": row["owner_session"]}
        new_exp = now + ttl if HEARTBEAT_EXTENSION else int(row["claim_expires"])
        sql = ("UPDATE task_ownership SET last_heartbeat = ?, claim_expires = ?,"
               " checkpoint = COALESCE(?, checkpoint), updated_at = ?"
               " WHERE scope_key = ? AND owner_session = ?")
        params: list[Any] = [
            now, new_exp,
            json.dumps(checkpoint, ensure_ascii=False) if checkpoint else None,
            now, scope, session_id,
        ]
        if expected_lock is not None:
            sql += " AND claim_lock = ?"
            params.append(expected_lock)
        cur = conn.execute(sql, params)
        if cur.rowcount == 0:
            return {"ok": False, "reason": "cas_lock_mismatch", "scope": scope,
                    "owner": row["owner_session"]}
        _append_event(conn, scope, "heartbeat", session_id, {"ttl": ttl}, now=now)
        return {"ok": True, "reason": "heartbeat", "scope": scope,
                "expires_at": new_exp}


def release_ownership(
    conn: sqlite3.Connection,
    scope: str,
    session_id: str,
    *,
    reason: Optional[str] = None,
    expected_lock: Optional[str] = None,
    now: Optional[int] = None,
) -> dict:
    """Voluntarily release the scope (owner only).

    ``expected_lock`` adds the same defensive compare-and-swap as heartbeat.
    """
    scope = validate_scope_key(scope)
    session_id = validate_session_id(session_id)
    now = int(now if now is not None else _now())
    with _write_txn(conn):
        row = _get(conn, scope)
        if row is None:
            return {"ok": False, "reason": "no_owner", "scope": scope}
        if row["owner_session"] != session_id:
            return {"ok": False, "reason": f"not_owner:{row['owner_session']}",
                    "scope": scope}
        sql = ("UPDATE task_ownership SET state = ?, updated_at = ?"
               " WHERE scope_key = ? AND owner_session = ?")
        params: list[Any] = [STATE_RELEASED, now, scope, session_id]
        if expected_lock is not None:
            sql += " AND claim_lock = ?"
            params.append(expected_lock)
        cur = conn.execute(sql, params)
        if cur.rowcount == 0:
            return {"ok": False, "reason": "cas_lock_mismatch", "scope": scope}
        _append_event(conn, scope, "released", session_id, {"reason": reason}, now=now)
        return {"ok": True, "reason": "released", "scope": scope}


def takeover_ownership(
    conn: sqlite3.Connection,
    scope: str,
    session_id: str,
    *,
    reason: Optional[str] = None,
    profile: Optional[str] = None,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    liveness: LivenessProvider = _liveness_unknown,
    verify_session: Optional[Callable[[str], bool]] = None,
    now: Optional[int] = None,
    orphan_grace: int = ORPHAN_GRACE_SECONDS,
) -> dict:
    """Explicit takeover — only succeeds when the current owner is STALE/ORPHANED."""
    res = claim_ownership(
        conn, scope, session_id, profile=profile, ttl_seconds=ttl_seconds,
        liveness=liveness, verify_session=verify_session, now=now,
        orphan_grace=orphan_grace,
    )
    if res["ok"] and res["reason"] == "claimed":
        res["reason"] = "claimed_free"
    if reason and res["ok"]:
        res["takeover_reason"] = reason
    return res


def inspect_ownership(
    conn: sqlite3.Connection,
    scope: str,
    *,
    liveness: LivenessProvider = _liveness_unknown,
    now: Optional[int] = None,
    orphan_grace: int = ORPHAN_GRACE_SECONDS,
) -> dict:
    """Read-only verdict for a scope. Never mutates."""
    now = int(now if now is not None else _now())
    row = _get(conn, scope)
    if row is None:
        return {"scope": scope, "verdict": VERDICT_FREE, "reason": "no_owner",
                "owner": None, "state": None, "expires_at": None,
                "last_heartbeat": None, "checkpoint": None}
    verdict, reason = _liveness_of(row, liveness, now, orphan_grace)
    return {
        "scope": scope,
        "verdict": verdict,
        "reason": reason,
        "owner": row["owner_session"],
        "owner_profile": row["owner_profile"],
        "state": row["state"],
        "expires_at": row["claim_expires"],
        "last_heartbeat": row["last_heartbeat"],
        "checkpoint": json.loads(row["checkpoint"]) if row["checkpoint"] else None,
        "seconds_to_expiry": int(row["claim_expires"]) - now,
    }


def list_ownership(
    conn: sqlite3.Connection,
    *,
    session_id: Optional[str] = None,
    active_only: bool = True,
) -> list[dict]:
    """List ownership rows — the startup briefing feed."""
    sql = "SELECT * FROM task_ownership"
    clauses: list[str] = []
    params: list[Any] = []
    if active_only:
        clauses.append("state = ?")
        params.append(STATE_ACTIVE)
    if session_id:
        clauses.append("owner_session = ?")
        params.append(session_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY updated_at DESC"
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return [_row_to_dict(r, cur) for r in rows]


def detect_contention(
    conn: sqlite3.Connection,
    *,
    window_seconds: int = 300,
    now: Optional[int] = None,
) -> list[dict]:
    """Report scopes where >=2 DISTINCT sessions attempted a claim recently.

    SAGE condition: alert when two sessions report the same scope within a short
    window. Reads the audit trail only (``claimed`` / ``reasserted`` /
    ``taken_over`` / ``claim_conflict``); never mutates.
    """
    now = int(now if now is not None else _now())
    since = now - int(window_seconds)
    cur = conn.execute(
        "SELECT scope_key, actor FROM task_ownership_events"
        " WHERE event IN ('claimed','reasserted','taken_over','claim_conflict')"
        "   AND created_at >= ?"
        " ORDER BY created_at DESC", (since,),
    )
    by_scope: dict[str, set] = {}
    for row in cur.fetchall():
        by_scope.setdefault(row["scope_key"], set()).add(row["actor"])
    out = []
    for scope, actors in by_scope.items():
        if len(actors) >= 2:
            out.append({"scope_key": scope, "actors": sorted(actors),
                        "distinct_sessions": len(actors)})
    out.sort(key=lambda x: (-x["distinct_sessions"], x["scope_key"]))
    return out


def prune_stale(conn: sqlite3.Connection, *, now: Optional[int] = None) -> list[str]:
    """Mark expired ACTIVE rows ORPHANED (audit only; does not delete)."""
    now = int(now if now is not None else _now())
    pruned: list[str] = []
    with _write_txn(conn):
        rows = conn.execute(
            "SELECT scope_key, claim_lock FROM task_ownership"
            " WHERE state = ? AND claim_expires < ?",
            (STATE_ACTIVE, now),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE task_ownership SET state = ?, updated_at = ?"
                " WHERE scope_key = ? AND claim_lock = ?",
                (STATE_ORPHANED, now, row["scope_key"], row["claim_lock"]),
            )
            _append_event(conn, row["scope_key"], "lease_expired", None,
                          {"at": now}, now=now)
            pruned.append(row["scope_key"])
    return pruned


class _write_txn:
    """``BEGIN IMMEDIATE`` transaction with rollback on error.

    ``join=True`` participates in a transaction the caller already opened (e.g.
    ``gateway/session_kanban.py``'s ``kb.write_txn``): no BEGIN/COMMIT is issued,
    so the ledger write commits or rolls back together with the caller's
    admission marker. That shared atomicity is the point of the Kanban
    enforcement patch — a second, independent lock would let the ledger claim
    and the card admission diverge.
    """

    def __init__(self, conn: sqlite3.Connection, *, join: bool = False):
        self.conn = conn
        self.join = join

    def __enter__(self):
        if self.join:
            if not getattr(self.conn, "in_transaction", False):
                raise OwnershipError(
                    "join_txn requested but the caller holds no open transaction"
                )
            return self.conn
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        if self.join:
            # The caller owns the commit/rollback decision.
            return False
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            self.conn.execute("ROLLBACK")
        return False
