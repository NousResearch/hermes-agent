"""
session_orchestration/ingest.py — z-harness webhook ingest handler.

Responsibilities
----------------
1. Persistent event_id dedup (SQLite table ``session_orchestration_event_dedup``
   in state.db) — survives restart, preventing double-posts on webhook retries.
2. Per-source rate limiting (in-memory fixed-window, keyed by payload ``source``).
3. Correlation: ``(run_id, canonical_repo_id(remote_url=repo))`` → existing
   registry row → enqueue ``update`` intent.
4. Adopt: unknown ``run_id``/``repo`` → enqueue ``adopt`` intent; row routed to
   the default "external runs" feed channel via ``discord_thread_id``.
5. Feed push: notify the configured Discord feed channel via the gateway runner.

Single-writer discipline
------------------------
This module NEVER writes to ``session_orchestration`` directly.  It only calls
``registry.enqueue_intent(...)`` which the cron watcher drains.

HMAC validation
---------------
The z-harness sender (T015, ``notify-watchdog.sh``) signs with:
    openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $NF}'
and sends the hex digest in ``X-Z-Harness-Signature: sha256=<hex>``.

The webhook adapter (``gateway/platforms/webhook.py``) validates the
``X-Z-Harness-Signature`` header via its generic HMAC path (``X-Webhook-Signature``
regex falls through — z-harness uses a dedicated header).  We validate it here
too for defence-in-depth when ``process_z_harness_alert`` is called with a raw
payload dict that already passed adapter-level validation.  When invoked via the
webhook adapter (``deliver_only=False, deliver="session_orchestration_ingest"``),
the adapter has already rejected bad signatures before calling us.

Configuration (from Hermes config.yaml under ``session_orchestration``)
------------------------------------------------------------------------
  session_orchestration:
    enabled: true
    feed_channel_id: "1234567890"          # Discord channel id for the unified feed
    external_runs_thread_id: "9876543210"  # Discord thread for adopted external runs
    rate_limit_per_source: 20              # max events/minute per source (default 20)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from session_orchestration.registry import (
    SessionOrchestrationRegistry,
    canonical_repo_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL for persistent event dedup table
# ---------------------------------------------------------------------------

_DEDUP_DDL = """
CREATE TABLE IF NOT EXISTS session_orchestration_event_dedup (
    event_id   TEXT PRIMARY KEY,
    source     TEXT,
    seen_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_so_dedup_seen_at
    ON session_orchestration_event_dedup(seen_at);
"""

# How long to retain event_id records in the dedup table (1 hour).
_DEDUP_TTL_SECONDS = 3600

# In-memory per-source rate limiter (fixed 60-second window).
# Keyed by source string → list of epoch floats for events in the window.
_rate_windows: Dict[str, list] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_db_path() -> Path:
    try:
        from hermes_state import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH
    except ImportError:
        from pathlib import Path as _Path
        import os
        hermes_home = os.environ.get("HERMES_HOME", str(_Path.home() / ".hermes"))
        return _Path(hermes_home) / "state.db"


def _ensure_dedup_schema(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        conn.executescript(_DEDUP_DDL)
    finally:
        conn.close()


def _is_duplicate_event(db_path: Path, event_id: str, source: str) -> bool:
    """Return True if *event_id* has already been processed.

    Side effect: if not a duplicate, records the event_id in the dedup table
    (atomic INSERT OR IGNORE — safe under concurrent writers).
    Prunes expired entries on the same connection for bounded table growth.
    """
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=5.0, isolation_level=None)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Check for existing record
            row = conn.execute(
                "SELECT 1 FROM session_orchestration_event_dedup WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                return True
            # Record the new event_id
            conn.execute(
                "INSERT INTO session_orchestration_event_dedup (event_id, source, seen_at) "
                "VALUES (?, ?, ?)",
                (event_id, source, now),
            )
            # Prune expired entries (best-effort, inside same transaction)
            cutoff = now - _DEDUP_TTL_SECONDS
            conn.execute(
                "DELETE FROM session_orchestration_event_dedup WHERE seen_at < ?",
                (cutoff,),
            )
            conn.execute("COMMIT")
            return False
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
    finally:
        conn.close()


def _check_rate_limit(source: str, limit_per_minute: int) -> bool:
    """Return True if the *source* is within the rate limit.

    Uses an in-memory fixed 60-second sliding window.  Not persisted across
    restarts (acceptable: rate limiting is a temporary throttle, not a hard
    security control — the webhook adapter already rate-limits at the route
    level before calling us).
    """
    now = time.time()
    window = _rate_windows.setdefault(source, [])
    # Evict entries outside the 60-second window
    _rate_windows[source] = [t for t in window if now - t < 60]
    if len(_rate_windows[source]) >= limit_per_minute:
        return False
    _rate_windows[source].append(now)
    return True


def _validate_hmac(raw_body: bytes, secret: str, signature_header: str) -> bool:
    """Validate ``X-Z-Harness-Signature: sha256=<hex>`` header.

    Matches the signing used by notify-watchdog.sh:
        openssl dgst -sha256 -hmac "$SECRET" -hex | awk '{print $NF}'
    which produces a plain hex digest (no prefix in the openssl output; the
    shell script prepends ``sha256=``).
    """
    if not signature_header.startswith("sha256="):
        logger.debug("[ingest] Missing sha256= prefix in signature header")
        return False
    provided_hex = signature_header[len("sha256="):]
    expected_hex = hmac.new(
        secret.encode(), raw_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(provided_hex, expected_hex)


def _get_session_orchestration_config() -> Dict[str, Any]:
    """Load session_orchestration section from Hermes config.yaml."""
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        result = cfg_get(cfg, "session_orchestration", default={})
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def _is_enabled() -> bool:
    cfg = _get_session_orchestration_config()
    return bool(cfg.get("enabled", False))


# ---------------------------------------------------------------------------
# Feed push helper
# ---------------------------------------------------------------------------


async def _push_to_feed(
    message: str,
    *,
    gateway_runner: Any = None,
    feed_channel_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> None:
    """Push *message* to the unified feed Discord channel.

    Requires ``gateway_runner`` (set by the webhook adapter) and a configured
    ``feed_channel_id``.  Best-effort — logs and returns on failure.
    """
    if not gateway_runner or not feed_channel_id:
        logger.debug("[ingest] feed push skipped: no gateway_runner or feed_channel_id")
        return

    try:
        from gateway.config import Platform
        discord_adapter = gateway_runner.adapters.get(Platform.DISCORD)
        if not discord_adapter:
            logger.warning("[ingest] feed push: Discord adapter not connected")
            return
        metadata = {"thread_id": thread_id} if thread_id else None
        await discord_adapter.send(feed_channel_id, message, metadata=metadata)
    except Exception as exc:
        logger.warning("[ingest] feed push failed: %s", exc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def process_z_harness_alert(
    payload: Dict[str, Any],
    *,
    raw_body: Optional[bytes] = None,
    signature_header: Optional[str] = None,
    hmac_secret: Optional[str] = None,
    gateway_runner: Any = None,
) -> Dict[str, Any]:
    """Process a validated z-harness alert payload.

    Called by the webhook adapter after its own HMAC + rate-limit + in-memory
    dedup checks pass.  We apply additional persistent dedup and per-source
    rate limiting, then correlate or adopt.

    Parameters
    ----------
    payload:
        Parsed JSON body from the z-harness POST.
    raw_body:
        Raw bytes of the POST body (for HMAC re-validation when called
        outside the webhook adapter path).  Optional — adapter already
        validated before calling us.
    signature_header:
        Value of ``X-Z-Harness-Signature`` header.  Only used for
        defence-in-depth validation when ``raw_body`` is provided.
    hmac_secret:
        HMAC secret for re-validation.  Read from ingest config when absent.
    gateway_runner:
        Gateway runner reference for feed push.  Set externally by the
        webhook adapter.

    Returns a status dict with ``{"status": "accepted"|"duplicate"|"rate_limited"
    |"invalid_signature"|"disabled"|"missing_fields", ...}``.
    """
    if not _is_enabled():
        logger.debug("[ingest] session_orchestration disabled; skipping z-harness alert")
        return {"status": "disabled"}

    so_cfg = _get_session_orchestration_config()
    rate_limit = int(so_cfg.get("rate_limit_per_source", 20))
    feed_channel_id: Optional[str] = so_cfg.get("feed_channel_id") or None
    external_thread_id: Optional[str] = so_cfg.get("external_runs_thread_id") or None

    # Defence-in-depth HMAC re-validation (when raw bytes provided).
    if raw_body is not None and signature_header is not None:
        secret = hmac_secret or so_cfg.get("hmac_secret", "")
        if secret and not _validate_hmac(raw_body, secret, signature_header):
            logger.warning("[ingest] HMAC re-validation failed for z-harness alert")
            return {"status": "invalid_signature"}

    # Required fields
    source = payload.get("source", "unknown")
    event_id = payload.get("event_id", "")
    run_id = payload.get("run_id", "")
    repo_raw = payload.get("repo", "")  # may be a workdir path or remote URL

    if not event_id:
        logger.warning("[ingest] z-harness alert missing event_id; rejected")
        return {"status": "missing_fields", "detail": "event_id required"}

    # Per-source rate limit (in-memory, 60-second window)
    if not _check_rate_limit(source, rate_limit):
        logger.warning("[ingest] rate limit exceeded for source=%s", source)
        return {"status": "rate_limited", "source": source}

    db_path = _get_db_path()
    _ensure_dedup_schema(db_path)

    # Persistent event_id dedup
    if _is_duplicate_event(db_path, event_id, source):
        logger.info("[ingest] duplicate event_id=%s from source=%s; skipping", event_id, source)
        return {"status": "duplicate", "event_id": event_id}

    registry = SessionOrchestrationRegistry(db_path=db_path)

    # Determine canonical repo key from the raw repo field.
    # z-harness sends the git toplevel path (a local absolute path) as `repo`.
    # We pass it as workdir= so canonical_repo_id hashes it consistently.
    # If the payload includes a remote URL (future extension), it would be
    # in a `remote_url` field.
    repo_key = canonical_repo_id(
        workdir=repo_raw if repo_raw else None,
        remote_url=payload.get("remote_url") or None,
    )

    # Correlate: look for an existing registry row with (run_id, repo_key).
    existing_rows = registry.list(run_id=run_id) if run_id else []
    matched = next(
        (r for r in existing_rows if r.get("repo") == repo_key),
        None,
    )

    event_name = payload.get("event", "unknown")
    severity = payload.get("severity", "info")
    reason = payload.get("reason", "")
    ts = payload.get("ts", int(time.time()))

    if matched:
        # Known run: enqueue an update intent so cron can apply it.
        task_id = matched["task_id"]
        registry.enqueue_intent(
            "update",
            task_id=task_id,
            run_id=run_id,
            repo=repo_key,
            payload={
                "task_id": task_id,
                "last_alert_event": event_name,
                "last_alert_ts": float(ts),
                "last_alert_severity": severity,
            },
        )
        logger.info(
            "[ingest] correlated run_id=%s repo=%s → task_id=%s event=%s",
            run_id, repo_key, task_id, event_name,
        )
        action = "correlated"
        thread_id = matched.get("discord_thread_id")
    else:
        # Unknown run: adopt with a lightweight row.
        import uuid as _uuid
        new_task_id = f"adopted-{_uuid.uuid4().hex[:8]}"
        slug = payload.get("slug", "")
        adopt_payload: Dict[str, Any] = {
            "task_id": new_task_id,
            "agent": payload.get("source", "z-harness"),
            "run_id": run_id,
            "repo": repo_key,
            "project": slug or repo_raw or "unknown",
            "discord_thread_id": external_thread_id,
            "state": "RUNNING",
            "last_alert_event": event_name,
            "last_alert_ts": float(ts),
            "last_alert_severity": severity,
        }
        registry.enqueue_intent(
            "adopt",
            task_id=new_task_id,
            run_id=run_id,
            repo=repo_key,
            payload=adopt_payload,
        )
        logger.info(
            "[ingest] adopted unknown run_id=%s repo=%s → new task_id=%s event=%s",
            run_id, repo_key, new_task_id, event_name,
        )
        action = "adopted"
        thread_id = external_thread_id
        task_id = new_task_id

    # Feed push: notify the unified feed channel.
    initiator = payload.get("initiator_discord_user_id")
    mention = f"<@{initiator}> " if initiator else ""
    feed_message = (
        f"{mention}**z-harness alert** `{event_name}` | severity: `{severity}`"
        f"\nrun: `{run_id or 'unknown'}` | slug: `{payload.get('slug', '')}` | {action}"
        + (f"\n> {reason}" if reason else "")
    )
    # Use the task's own discord thread (if known) for feed context.
    payload_thread_id = payload.get("thread_id") or thread_id
    await _push_to_feed(
        feed_message,
        gateway_runner=gateway_runner,
        feed_channel_id=feed_channel_id,
        thread_id=payload_thread_id,
    )

    return {
        "status": "accepted",
        "action": action,
        "task_id": task_id,
        "event_id": event_id,
    }
