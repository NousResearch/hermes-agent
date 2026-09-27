"""Cross-profile relay for Slack slash commands.

Slack slash-command names are workspace-global: when several Hermes
profiles each install their own Slack app in ONE workspace and register
the same command names, Slack delivers every ``/command`` to exactly one
app (the most recently reinstalled), regardless of which channel it was
typed in. The receiving gateway would then answer from the wrong profile.

This module gives the receiving adapter what it needs to fix that:

1. ``resolve_channel_owner(channel_id)`` — map a Slack channel (or bot-DM)
   id to the profile that owns it, using only artifacts every profile
   already writes: ``config.yaml``'s ``slack.free_response_channels`` /
   ``allowed_channels`` (explicit intent, score 2) and
   ``channel_directory.json``'s observed Slack channels (score 1). The
   unique top scorer owns the channel; ties or no claimant resolve to
   ``None`` (caller handles the command locally — the pre-relay behavior).

2. A small SQLite queue (``slack-slash-relay.db`` in the hermes root,
   next to ``kanban.db``) that the receiving adapter enqueues foreign
   slash payloads into and every profile's adapter polls for its own
   inbox. The receiving app acks silently; the owning profile replies with
   its own bot token (``chat.postEphemeral``) so Slack shows the right app
   name. The payload travels verbatim, including ``response_url`` (Slack
   scopes it to the invocation, not the receiving app), which serves only
   as a fallback that posts a NEW ephemeral message.

Rows hold a ``response_url`` (usable to post into the channel for ~30
minutes) and the user's command text, so the database is created with mode
0600 and every consumer poll deletes rows older than 24 hours. SQLite WAL
mode is unreliable on network filesystems; keep the hermes root local.

A single-app install never resolves a foreign channel, so nothing is ever
relayed there, but the adapter still creates the database and polls it every
1.5 s. ``slack.slash_relay: false`` (or ``SLACK_SLASH_RELAY=false``) turns
the relay off entirely: no database, no consumer task.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DB_FILENAME = "slack-slash-relay.db"

# Rows older than this are never claimed: the payload's response_url dies
# at ~30 minutes anyway, and a many-minutes-late slash answer helps nobody.
DEFAULT_CLAIM_MAX_AGE_S = 600.0

# Completed / expired rows are deleted past this age.
DEFAULT_PURGE_KEEP_S = 86_400.0

# How long a computed channel→profile map is trusted before rescanning the
# profile homes (a rescan is a handful of small file reads).
OWNER_CACHE_TTL_S = 60.0

# root path str → (monotonic timestamp, {channel_id: profile})
_owner_cache: Dict[str, tuple] = {}


def _default_root() -> Path:
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root()


def _resolve_root(root: Optional[Any]) -> Path:
    return Path(root) if root is not None else _default_root()


# ---------------------------------------------------------------------------
# Channel → profile resolution
# ---------------------------------------------------------------------------

def _profile_homes(root: Path) -> Dict[str, Path]:
    """Return {profile_name: home_dir} for the default + named profiles."""
    homes: Dict[str, Path] = {}
    if (root / "config.yaml").exists():
        homes["default"] = root
    profiles_root = root / "profiles"
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if entry.name != "default" and (entry / "config.yaml").exists():
                homes[entry.name] = entry
    return homes


def _split_channel_csv(raw: Any) -> Set[str]:
    """Parse a channels value that may be a list, CSV string, or scalar."""
    if isinstance(raw, list):
        return {str(part).strip() for part in raw if str(part).strip()}
    s = str(raw).strip() if raw is not None else ""
    if not s:
        return set()
    return {part.strip() for part in s.split(",") if part.strip()}


def _explicit_channel_claims(home: Path) -> Set[str]:
    """Channels a profile claims in config.yaml (slack.free_response_channels
    / slack.allowed_channels, under the top-level ``slack:`` block or a
    ``platforms.slack`` block)."""
    path = home / "config.yaml"
    try:
        from utils import fast_safe_load

        data = fast_safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except Exception as exc:
        # Fail open for THIS profile only, but loudly: a silent empty set
        # makes every explicit claim vanish without trace.
        logger.warning("slash-relay: cannot read %s: %s", path, exc)
        return set()
    if not isinstance(data, dict):
        return set()
    platforms = data.get("platforms")
    blocks = [
        data.get("slack"),
        platforms.get("slack") if isinstance(platforms, dict) else None,
    ]
    claims: Set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for key in ("free_response_channels", "allowed_channels"):
            claims |= _split_channel_csv(block.get(key))
    return claims


def _observed_channel_claims(home: Path) -> Set[str]:
    """Slack channel ids a profile's gateway has actually seen traffic in
    (channel_directory.json), including its own bot-DM channels. Thread
    entries like ``C123:171...`` collapse to the base channel id."""
    path = home / "channel_directory.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except Exception as exc:
        logger.warning("slash-relay: cannot read %s: %s", path, exc)
        return set()
    if not isinstance(data, dict):
        return set()
    platforms = data.get("platforms")
    entries = platforms.get("slack") if isinstance(platforms, dict) else None
    claims: Set[str] = set()
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            base = str(entry.get("id") or "").split(":", 1)[0].strip()
            if base:
                claims.add(base)
    return claims


def build_channel_owner_map(root: Optional[Any] = None) -> Dict[str, str]:
    """Compute {channel_id: owning_profile} across all profile homes.

    Explicit config claims score 2, observed directory claims score 1; a
    channel maps to a profile only when that profile is the UNIQUE top
    scorer. This settles template-config overlap: a profile that both
    configures a channel and has actually seen it (score 3) beats profiles
    that merely inherited the channel in a copied config (score 2).
    """
    root_path = _resolve_root(root)
    scores: Dict[str, Dict[str, int]] = {}
    for name, home in _profile_homes(root_path).items():
        for cid in _explicit_channel_claims(home):
            scores.setdefault(cid, {})[name] = scores.get(cid, {}).get(name, 0) + 2
        for cid in _observed_channel_claims(home):
            scores.setdefault(cid, {})[name] = scores.get(cid, {}).get(name, 0) + 1
    owners: Dict[str, str] = {}
    for cid, per_profile in scores.items():
        best = max(per_profile.values())
        winners = [p for p, s in per_profile.items() if s == best]
        if len(winners) == 1:
            owners[cid] = winners[0]
        else:
            logger.debug(
                "slash-relay: channel %s claimed equally by %s — leaving unrouted",
                cid,
                winners,
            )
    return owners


def resolve_channel_owner(
    channel_id: str,
    root: Optional[Any] = None,
    *,
    ttl_s: float = OWNER_CACHE_TTL_S,
) -> Optional[str]:
    """Return the profile that owns ``channel_id``, or None if unknown or
    contested. Results are cached per hermes root for ``ttl_s`` seconds."""
    base = str(channel_id or "").split(":", 1)[0].strip()
    if not base:
        return None
    root_path = _resolve_root(root)
    key = str(root_path)
    now = time.monotonic()
    cached = _owner_cache.get(key)
    if cached is None or now - cached[0] > ttl_s:
        try:
            cached = (now, build_channel_owner_map(root_path))
        except Exception:
            logger.warning(
                "slash-relay: channel owner scan failed", exc_info=True
            )
            cached = (now, {})
        _owner_cache[key] = cached
    return cached[1].get(base)


def clear_owner_cache() -> None:
    """Testing hook: drop all cached channel→profile maps."""
    _owner_cache.clear()


# ---------------------------------------------------------------------------
# Relay queue
# ---------------------------------------------------------------------------

_DB_MODE = 0o600


def _restrict_db_files(db_path: Path) -> None:
    """chmod the database and its WAL sidecars to 0600 (best effort)."""
    for suffix in ("", "-wal", "-shm"):
        try:
            os.chmod(f"{db_path}{suffix}", _DB_MODE)
        except OSError:
            pass


def _connect(root: Path) -> sqlite3.Connection:
    db_path = root / DB_FILENAME
    created = False
    if not db_path.exists():
        # Owner-only from the first byte: rows carry response_url and command text.
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.close(os.open(str(db_path), os.O_CREAT | os.O_EXCL | os.O_RDWR, _DB_MODE))
            created = True
        except FileExistsError:
            pass
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS slash_relay (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            target_profile TEXT NOT NULL,
            source_profile TEXT NOT NULL,
            payload        TEXT NOT NULL,
            created_at     REAL NOT NULL,
            claimed_at     REAL,
            claimed_by     TEXT,
            done_at        REAL
        )
        """
    )
    if created:
        # SQLite gives the -wal/-shm sidecars the database's mode; enforce it
        # anyway in case a umask or an older file got in the way.
        _restrict_db_files(db_path)
    return conn


def enqueue(
    target_profile: str,
    source_profile: str,
    payload: Dict[str, Any],
    *,
    root: Optional[Any] = None,
) -> int:
    """Queue a slash payload for ``target_profile``. Returns the row id."""
    root_path = _resolve_root(root)
    with closing(_connect(root_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO slash_relay"
            " (target_profile, source_profile, payload, created_at)"
            " VALUES (?, ?, ?, ?)",
            (target_profile, source_profile, json.dumps(payload), time.time()),
        )
        return int(cur.lastrowid)


def claim_pending(
    profile: str,
    *,
    root: Optional[Any] = None,
    max_age_s: float = DEFAULT_CLAIM_MAX_AGE_S,
    keep_s: float = DEFAULT_PURGE_KEEP_S,
) -> List[Dict[str, Any]]:
    """Atomically claim every unclaimed, unexpired row addressed to
    ``profile``. Returns [{"id": int, "payload": dict}, ...]; a row is
    handed out exactly once across all processes (single-UPDATE claim with
    a unique token). Undecodable payloads are marked done and skipped.

    Each poll also deletes rows older than ``keep_s`` (24 h by default), so
    stale ``response_url`` values and command text never accumulate."""
    root_path = _resolve_root(root)
    now = time.time()
    token = f"{profile}:{uuid.uuid4().hex}"
    with closing(_connect(root_path)) as conn, conn:
        conn.execute(
            "DELETE FROM slash_relay WHERE created_at < ?", (now - keep_s,)
        )
        conn.execute(
            "UPDATE slash_relay SET claimed_at = ?, claimed_by = ?"
            " WHERE target_profile = ? AND claimed_at IS NULL"
            "   AND created_at >= ?",
            (now, token, profile, now - max_age_s),
        )
        rows = conn.execute(
            "SELECT id, payload FROM slash_relay WHERE claimed_by = ?",
            (token,),
        ).fetchall()
    claimed: List[Dict[str, Any]] = []
    for row_id, payload_json in rows:
        try:
            claimed.append({"id": int(row_id), "payload": json.loads(payload_json)})
        except Exception:
            logger.warning("slash-relay: dropping undecodable row %s", row_id)
            mark_done(int(row_id), root=root_path)
    return claimed


def mark_done(row_id: int, *, root: Optional[Any] = None) -> None:
    root_path = _resolve_root(root)
    with closing(_connect(root_path)) as conn, conn:
        conn.execute(
            "UPDATE slash_relay SET done_at = ? WHERE id = ?",
            (time.time(), row_id),
        )


def is_claimed(row_id: int, *, root: Optional[Any] = None) -> bool:
    """True when the row was picked up (or finished) by its target profile."""
    root_path = _resolve_root(root)
    with closing(_connect(root_path)) as conn:
        row = conn.execute(
            "SELECT claimed_at, done_at FROM slash_relay WHERE id = ?",
            (row_id,),
        ).fetchone()
    if row is None:
        return False
    return row[0] is not None or row[1] is not None


def purge(
    *,
    root: Optional[Any] = None,
    keep_s: float = DEFAULT_PURGE_KEEP_S,
) -> int:
    """Delete rows older than ``keep_s``. Returns the number removed."""
    root_path = _resolve_root(root)
    with closing(_connect(root_path)) as conn, conn:
        cur = conn.execute(
            "DELETE FROM slash_relay WHERE created_at < ?",
            (time.time() - keep_s,),
        )
        return int(cur.rowcount or 0)
