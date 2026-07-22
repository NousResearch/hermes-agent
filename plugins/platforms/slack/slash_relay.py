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
   inbox. The payload travels verbatim, including ``response_url`` —
   which Slack scopes to the invocation, not the receiving app, so the
   owning profile's reply can still replace the ephemeral ack.

Single-profile installs never observe a foreign channel, so the whole
mechanism is a no-op there.
"""

from __future__ import annotations

import json
import logging
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
    try:
        import yaml

        data = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    except Exception:
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
    try:
        data = json.loads(
            (home / "channel_directory.json").read_text(encoding="utf-8")
        )
    except Exception:
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

def _connect(root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(root / DB_FILENAME), timeout=5.0)
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
) -> List[Dict[str, Any]]:
    """Atomically claim every unclaimed, unexpired row addressed to
    ``profile``. Returns [{"id": int, "payload": dict}, ...]; a row is
    handed out exactly once across all processes (single-UPDATE claim with
    a unique token). Undecodable payloads are marked done and skipped."""
    root_path = _resolve_root(root)
    now = time.time()
    token = f"{profile}:{uuid.uuid4().hex}"
    with closing(_connect(root_path)) as conn, conn:
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
