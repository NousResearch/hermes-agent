"""
Session-orchestration watcher — core loop (T006).

Architecture
------------
The watcher is the **sole mutator** of the ``session_orchestration`` registry.
It is designed to run as a Hermes ``--no-agent`` cron job at a 1–2 minute
cadence; each tick it:

1. Drains and applies the intent queue (adopt / drive / update intents
   enqueued by the webhook-ingest or Discord-drive paths).
2. Iterates all rows in the registry, skipping rows whose adapter is
   unavailable (determined once at startup by ``verify_adapters``).
3. For each active row, acquires the per-session lock BEFORE capture-pane.
   If the lock is held (by the relay), the row is **skipped this tick** —
   no read of a potentially half-rendered pane, no spurious state update.
4. Calls ``adapter.detect()`` on the captured pane text, updates registry
   state, increments counters atomically.
5. Computes transitions.  Actual push behaviours (turn-change, heartbeat,
   hang) are implemented in T007/T008/T009.  This module exposes hook seams
   (``_on_turn_change``, ``_on_heartbeat_tick``, ``_on_hang``) that those
   tasks will fill in.

Config gate
-----------
All orchestration is gated on ``session_orchestration.enabled`` in the Hermes
config (``~/.hermes/config.yaml``).  When the key is absent or falsy the
watcher exits immediately, producing zero side-effects and no network calls —
byte-identical to pre-feature behaviour.

Lock contract
-------------
``registry.acquire_lock(task_id, holder, ttl_seconds=300.0)`` BEFORE any
``capture-pane`` operation.  If it returns ``False``, skip this row.
``registry.release_lock`` in a ``finally`` block.
``lock_ts`` stores the float expiry epoch (``str(time.time() + ttl)``).

Seams for downstream tasks
---------------------------
- ``T007`` (turn-change push): implement ``_on_turn_change``
- ``T008`` (heartbeat edit): implement ``_on_heartbeat_tick``
- ``T009`` (hang detection + nudge): implement ``_on_hang``

Each hook receives the full registry row dict so the downstream implementation
has access to all state without further DB reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from session_orchestration.adapters.base import AgentAdapter
from session_orchestration.adapters.verify import verify_adapters
from session_orchestration.markers import marker_kind_to_lifecycle, read_markers_since
from session_orchestration.menu_parse import extract as extract_menu_context
from session_orchestration.registry import SessionOrchestrationRegistry
from session_orchestration.types import SessionHandle, SessionLifecycle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default lock TTL: 5x a 60 s cron interval.
_LOCK_TTL_SECONDS: float = 300.0

#: Heartbeat cadence: fire every N ticks (~5 min at 1-min cron).
_HEARTBEAT_CADENCE: int = 5

#: Recency window for marker-derived state (seconds).  Markers older than
#: this threshold are treated as stale and the watcher falls back to pane
#: detection.  Matches the lock TTL so a single missed cron tick does not
#: cause a spurious fallback.
_MARKER_RECENCY_SECONDS: float = 300.0

#: Tick interval when any session is RUNNING or WAITING_USER (active).
_FAST_TICK_SECONDS: float = 30.0

#: Tick interval when no active sessions exist.
_IDLE_TICK_SECONDS: float = 120.0

#: Active states — rows in these states are iterated by the watcher.
_ACTIVE_STATES = frozenset(
    {
        SessionLifecycle.RUNNING.value,
        SessionLifecycle.WAITING_USER.value,
        SessionLifecycle.PAUSED_HANDOFF.value,
        SessionLifecycle.STALLED.value,
    }
)

#: Attention states — sessions waiting for user input.  Used to stamp
#: ``attention_since`` on entry and to gate the re-nudge check.
_ATTENTION_STATES = frozenset(
    {
        SessionLifecycle.WAITING_USER.value,
        SessionLifecycle.PAUSED_HANDOFF.value,
    }
)

#: Terminal states — rows in these states are never re-iterated.
_TERMINAL_STATES = frozenset(
    {
        SessionLifecycle.DONE.value,
        SessionLifecycle.ERROR.value,
    }
)

#: User-attention states map directly to stable attention-item reasons.
_USER_ATTENTION_STATES = frozenset(
    {
        SessionLifecycle.WAITING_USER.value,
        SessionLifecycle.PAUSED_HANDOFF.value,
    }
)

#: Stale/frozen attention is a reason on a RUNNING row, not a registry state.
_ATTENTION_REASON_STALE_FROZEN = "STALE_FROZEN"


# ---------------------------------------------------------------------------
# Marker-recency helper
# ---------------------------------------------------------------------------


def _parse_marker_ts(ts_str: str) -> float:
    """Parse an ISO-8601 UTC timestamp string to a POSIX float.

    Returns 0.0 on any parse failure so malformed or missing timestamps are
    always treated as maximally stale (never recent).
    """
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Config helper for dead-tmux reap (avoids importing config at module level)
# ---------------------------------------------------------------------------


def _load_dead_tmux_reap_cfg() -> bool:
    """Return True if dead_tmux_reap is enabled in config (default True).

    Lazy-imported so the watcher module does not pull in config at import
    time.  Returns True (safe default) on any error.
    """
    try:
        from session_orchestration.config import load_session_orchestration_config
        return load_session_orchestration_config().dead_tmux_reap
    except Exception:  # config load must never crash the watcher
        return True


def _load_gc_after_seconds_cfg() -> int:
    """Return gc_after_seconds from config (default 86400 = 24 h).

    Lazy-imported so the watcher module does not pull in config at import
    time.  Returns the default on any error.
    """
    try:
        from session_orchestration.config import load_session_orchestration_config
        return load_session_orchestration_config().gc_after_seconds
    except Exception:  # config load must never crash the watcher
        return 86400


def _load_renudge_after_seconds_cfg() -> int:
    """Return renudge_after_seconds from config (default 1800 = 30 min).

    Lazy-imported so the watcher module does not pull in config at import
    time.  Returns the default on any error.  A value <= 0 disables re-nudging.
    """
    try:
        from session_orchestration.config import load_session_orchestration_config
        return load_session_orchestration_config().renudge_after_seconds
    except Exception:  # config load must never crash the watcher
        return 1800


# ---------------------------------------------------------------------------
# Default DM sender (injectable for tests so no network calls are made)
# ---------------------------------------------------------------------------


def _default_send_dm(user_id: str, msg: str) -> bool:
    """Send a DM to *user_id* with *msg*.

    Lazy-imports ``_get_bot_token`` and ``send_dm`` at call time.
    Returns ``False`` (never raises) on any error so the caller's try/except
    can still log the failure without crashing the watcher tick.

    This is the production default — tests inject a recording fake via the
    ``_send_dm_fn`` parameter on ``SessionWatcher``.
    """
    try:
        from tools.discord_tool import _get_bot_token  # type: ignore[import]
        from session_orchestration.dm_transport import send_dm

        token = _get_bot_token()
        if not token:
            return False
        return send_dm(user_id, msg, token)
    except Exception as exc:
        logger.debug("watcher._default_send_dm: failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Default tmux-liveness check (injectable for tests)
# ---------------------------------------------------------------------------


def _default_tmux_liveness(tmux_session: str) -> bool:
    """Return True if the tmux session is alive (exit 0), False otherwise.

    Uses ``tmux has-session -t <tmux_session>``.  Returns False on any error
    including binary-not-found so the reap logic degrades safely.
    """
    if not tmux_session:
        return False
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            capture_output=True,
        )
        return result.returncode == 0
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Fake-capture adapter (allows watcher to be unit-tested without tmux)
# ---------------------------------------------------------------------------


class _TmuxCapture:
    """Thin wrapper around ``tmux capture-pane``.

    Injected as ``tmux_capture`` in production; tests swap in a fake that
    returns pre-canned strings and records which sessions were captured.

    Protocol: callable ``(pane: str) -> str`` — returns the current pane text
    or an empty string if the pane does not exist / tmux is unavailable.
    """

    def __call__(self, pane: str) -> str:
        import subprocess  # local import so the protocol is clear in tests

        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-p", "-t", pane],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.warning("tmux capture-pane failed for pane=%s", pane)
            return ""


# ---------------------------------------------------------------------------
# Transition seam hooks (T007 / T008 / T009 will implement these)
# ---------------------------------------------------------------------------


def _reconcile_attention_digest(
    registry: Optional["SessionOrchestrationRegistry"],
    *,
    task_id: str,
    context: str,
) -> None:
    """Best-effort channel-level digest reconciliation for attention changes."""
    if registry is None:
        logger.debug(
            "watcher.%s: no registry available for digest reconcile task_id=%s",
            context,
            task_id,
        )
        return

    try:
        from session_orchestration.feed import reconcile_attention_digest

        reconcile_attention_digest(registry)
    except Exception as exc:
        logger.error(
            "watcher.%s: digest reconcile failed for task_id=%s: %s",
            context,
            task_id,
            exc,
        )


def _on_turn_change(
    task_id: str,
    row: Dict[str, Any],
    new_state: str,
    old_state: str,
    *,
    registry: Optional["SessionOrchestrationRegistry"] = None,
) -> None:
    """Called when a session transitions into a user-attention state.

    Reconciles the channel-level action digest and optionally posts a
    thread-local notice.  The digest is the sole ``feed_channel_id``
    projection for attention state.

    Parameters
    ----------
    task_id:    Registry key.
    row:        Full registry row dict at the time of the transition.
    new_state:  The new ``SessionLifecycle`` value (string).
    old_state:  The previous ``SessionLifecycle`` value (string).
    registry:   Registry instance (injected by ``_process_row``; defaults
                to a fresh production instance when absent).
    """
    logger.debug(
        "watcher._on_turn_change: task_id=%s %s -> %s",
        task_id,
        old_state,
        new_state,
    )
    _reconcile_attention_digest(
        registry,
        task_id=task_id,
        context="_on_turn_change",
    )
    try:
        from session_orchestration.feed import push_turn_change

        push_turn_change(task_id, row, new_state, old_state)
    except Exception as exc:
        # Non-fatal: a failed notification must not crash the watcher.
        logger.error(
            "watcher._on_turn_change: push failed for task_id=%s: %s",
            task_id,
            exc,
        )


def _on_heartbeat_tick(task_id: str, row: Dict[str, Any]) -> None:
    """Called every tick; fires heartbeat edit every ~5 min.

    Cadence gate: only acts when ``heartbeat_counter % _HEARTBEAT_CADENCE == 0``
    (where _HEARTBEAT_CADENCE = 5, matching a 1-min cron for a ~5-min interval).
    Non-cadence ticks are a no-op.

    First cadence tick (no ``status_message_id`` in row): POSTs a new status
    message to the feed channel and persists the returned message_id via the
    registry (watcher is sole mutator).

    Subsequent cadence ticks: PATCHes the existing status message in-place with
    current activity state and elapsed-since-last-change.  No new notification
    is fired.

    Parameters
    ----------
    task_id:  Registry key.
    row:      Full registry row dict (fresh — reflects counters just written).
    """
    counter = row.get("heartbeat_counter") or 0
    if counter == 0 or counter % _HEARTBEAT_CADENCE != 0:
        logger.debug(
            "watcher._on_heartbeat_tick: skipping (counter=%d) task_id=%s",
            counter,
            task_id,
        )
        return

    # Resolve feed channel from config
    try:
        from session_orchestration.config import load_session_orchestration_config

        cfg = load_session_orchestration_config()
        feed_channel_id = cfg.feed_channel_id
    except Exception as exc:
        logger.debug(
            "watcher._on_heartbeat_tick: could not read config: %s", exc
        )
        feed_channel_id = None

    if not feed_channel_id:
        logger.debug(
            "watcher._on_heartbeat_tick: no feed_channel_id; skipping task_id=%s",
            task_id,
        )
        return

    # Build status content: current state + elapsed since last output change
    state = row.get("state", "RUNNING")
    agent = row.get("agent", "unknown")
    project = row.get("project") or row.get("repo") or ""
    project_part = f" | {project}" if project else ""

    last_output_ts = row.get("last_output_ts")
    if last_output_ts:
        elapsed_s = int(time.time() - float(last_output_ts))
        elapsed_min, elapsed_sec = divmod(elapsed_s, 60)
        elapsed_str = f"{elapsed_min}m{elapsed_sec:02d}s" if elapsed_min else f"{elapsed_sec}s"
        elapsed_part = f" (no change for {elapsed_str})"
    else:
        elapsed_part = ""

    content = (
        f"⏱ **[{agent}] {task_id}** — `{state}`{project_part}{elapsed_part}"
    )

    # Determine whether to POST (first) or PATCH (subsequent)
    status_message_id: Optional[str] = row.get("status_message_id") or None

    try:
        if status_message_id is None:
            # First heartbeat: POST a new status message
            from session_orchestration.feed import _post_discord_message

            new_msg_id = _post_discord_message(feed_channel_id, content)
            if new_msg_id:
                logger.info(
                    "watcher._on_heartbeat_tick: posted status message id=%s task_id=%s",
                    new_msg_id,
                    task_id,
                )
                # Persist the message_id so subsequent heartbeats PATCH it
                _heartbeat_registry_ref(task_id, row, new_msg_id)
            else:
                logger.warning(
                    "watcher._on_heartbeat_tick: POST returned no message_id for task_id=%s",
                    task_id,
                )
        else:
            # Subsequent heartbeats: PATCH in-place — no new notification
            from session_orchestration.feed import edit_status_message

            ok = edit_status_message(feed_channel_id, status_message_id, content)
            if ok:
                logger.debug(
                    "watcher._on_heartbeat_tick: edited status message id=%s task_id=%s",
                    status_message_id,
                    task_id,
                )
            else:
                logger.warning(
                    "watcher._on_heartbeat_tick: PATCH failed for message_id=%s task_id=%s",
                    status_message_id,
                    task_id,
                )
    except Exception as exc:
        # Non-fatal: a failed heartbeat must not crash the watcher.
        logger.error(
            "watcher._on_heartbeat_tick: error for task_id=%s: %s",
            task_id,
            exc,
        )


def _heartbeat_registry_ref(
    task_id: str,
    row: Dict[str, Any],
    status_message_id: str,
) -> None:
    """Persist *status_message_id* to the registry row for *task_id*.

    Called only after the first heartbeat POST succeeds.  Uses the production
    registry via a late import (avoids circular deps at module level).

    Isolated into its own function so tests can patch it without touching
    the broader _on_heartbeat_tick logic.
    """
    try:
        from session_orchestration.registry import SessionOrchestrationRegistry

        # The watcher owns registry writes; no queue needed here.
        reg = SessionOrchestrationRegistry()
        reg.upsert(
            task_id,
            agent=row.get("agent", "unknown"),
            run_id=row.get("run_id"),
            repo=row.get("repo"),
            source=row.get("source", "spawn"),
            status_message_id=status_message_id,
        )
    except Exception as exc:
        logger.error(
            "watcher._heartbeat_registry_ref: failed to persist status_message_id"
            " for task_id=%s: %s",
            task_id,
            exc,
        )



def _load_hang_thresholds() -> tuple[int, int]:
    """Return ``(hang_idle_ticks, hang_stale_seconds)`` with safe defaults."""
    try:
        from session_orchestration.config import load_session_orchestration_config

        cfg = load_session_orchestration_config()
        return cfg.hang_idle_ticks, cfg.hang_stale_seconds
    except Exception as exc:
        logger.debug("watcher: could not read hang thresholds, using defaults: %s", exc)
        return 3, 300


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_non_negative_int(value: Any) -> int:
    try:
        n = int(value)
        return n if n >= 0 else 0
    except (TypeError, ValueError):
        return 0


def _adapter_activity_regex_matches(
    adapter: Optional["AgentAdapter"],
    pane_text: str,
) -> bool:
    """Return True when the adapter declares a positive active-work match."""
    if adapter is None or not pane_text:
        return False

    try:
        caps = adapter.capabilities()
    except Exception as exc:
        logger.debug("watcher: adapter.capabilities() failed during stale guard: %s", exc)
        return False

    regex = getattr(caps, "idle_indicator_regex", None)
    return bool(regex is not None and regex.search(pane_text))


def _is_stale_frozen_eligible(
    row: Dict[str, Any],
    *,
    previous_pane_hash: Optional[str],
    current_pane_hash: Optional[str],
    pane_text: str,
    adapter: Optional["AgentAdapter"],
    hang_idle_ticks: int,
    hang_stale_seconds: int,
    now: Optional[float] = None,
) -> bool:
    """Return True only when deterministic stale/frozen evidence is present.

    The guard is intentionally conservative: a stale/frozen episode requires
    an unchanged pane hash, enough idle ticks, a stale ``last_output_ts``, and
    no adapter-declared active-work regex match in the current pane.
    """
    if not current_pane_hash or previous_pane_hash != current_pane_hash:
        return False

    if _coerce_non_negative_int(row.get("idle_ticks")) < hang_idle_ticks:
        return False

    last_output_ts = _coerce_optional_float(row.get("last_output_ts"))
    if last_output_ts is None:
        return False

    observed_now = time.time() if now is None else now
    if observed_now - last_output_ts < hang_stale_seconds:
        return False

    if _adapter_activity_regex_matches(adapter, pane_text):
        return False

    return True


def _is_stale_episode_action_eligible(row: Dict[str, Any]) -> bool:
    """Return True iff no stale/frozen action has fired in this episode."""
    return _coerce_non_negative_int(row.get("nudge_count")) == 0


def _is_omp_nudge_checkin_eligible(
    row: Dict[str, Any],
    *,
    previous_pane_hash: Optional[str],
    current_pane_hash: Optional[str],
    pane_text: str,
    adapter: Optional["AgentAdapter"],
    hang_idle_ticks: int,
    hang_stale_seconds: int,
    now: Optional[float] = None,
) -> bool:
    """Return True when the OMP stale nudge/check-in may fire this tick."""
    return _is_stale_frozen_eligible(
        row,
        previous_pane_hash=previous_pane_hash,
        current_pane_hash=current_pane_hash,
        pane_text=pane_text,
        adapter=adapter,
        hang_idle_ticks=hang_idle_ticks,
        hang_stale_seconds=hang_stale_seconds,
        now=now,
    ) and _is_stale_episode_action_eligible(row)


def _intent_task_id(intent: Dict[str, Any]) -> Optional[str]:
    """Return a task id from an intent row without mutating its payload."""
    task_id = intent.get("task_id")
    if task_id:
        return str(task_id)
    try:
        payload = json.loads(intent.get("payload") or "{}")
    except (TypeError, ValueError):
        return None
    payload_task_id = payload.get("task_id")
    return str(payload_task_id) if payload_task_id else None


def _attention_detail(row: Dict[str, Any], reason: str) -> str:
    """Build compact, stable detail text for an attention item refresh."""
    agent = row.get("agent", "unknown")
    state = row.get("state", SessionLifecycle.RUNNING.value)
    if reason == _ATTENTION_REASON_STALE_FROZEN:
        idle_ticks = _coerce_non_negative_int(row.get("idle_ticks"))
        last_output_ts = row.get("last_output_ts")
        return (
            f"state={state}; agent={agent}; idle_ticks={idle_ticks}; "
            f"last_output_ts={last_output_ts}"
        )
    return f"state={state}; agent={agent}"


def _last_output_advanced(
    previous_last_output_ts: Optional[float],
    current_last_output_ts: Optional[float],
) -> bool:
    """Return True only for a real observed liveness timestamp advance."""
    return (
        previous_last_output_ts is not None
        and current_last_output_ts is not None
        and current_last_output_ts > previous_last_output_ts
    )


def _sync_attention_lifecycle(
    registry: "SessionOrchestrationRegistry",
    task_id: str,
    *,
    old_state: str,
    new_state: str,
    fresh_row: Dict[str, Any],
    stale_frozen_eligible: bool,
    previous_pane_hash: Optional[str],
    current_pane_hash: Optional[str],
    previous_last_output_ts: Optional[float],
    current_last_output_ts: Optional[float],
    pane_text: str,
    adapter: Optional["AgentAdapter"],
    user_drive_signal: bool = False,
) -> tuple[bool, bool]:
    """Sync attention items; return (stale/frozen actionable, digest dirty)."""
    is_terminal = new_state in _TERMINAL_STATES
    digest_dirty = False

    if new_state in _USER_ATTENTION_STATES:
        registry.open_attention_item(
            task_id,
            new_state,
            priority=100,
            detail=_attention_detail(fresh_row, new_state),
        )
        for reason in _USER_ATTENTION_STATES:
            if reason != new_state:
                registry.resolve_attention_item(
                    task_id,
                    reason,
                    resolution_reason=f"state_changed_to_{new_state}",
                )
    else:
        for reason in _USER_ATTENTION_STATES:
            resolved = registry.resolve_attention_item(
                task_id,
                reason,
                resolution_reason=f"state_changed_to_{new_state}",
            )
            if resolved and not (
                old_state in _USER_ATTENTION_STATES and new_state != old_state
            ):
                digest_dirty = True

    if is_terminal:
        if registry.resolve_attention_item(
            task_id,
            _ATTENTION_REASON_STALE_FROZEN,
            resolution_reason=f"state_changed_to_{new_state}",
        ):
            digest_dirty = True
        return False, digest_dirty

    if user_drive_signal and registry.resolve_attention_item(
        task_id,
        _ATTENTION_REASON_STALE_FROZEN,
        resolution_reason="user_drive_signal",
    ):
        return False, True

    if stale_frozen_eligible:
        registry.open_attention_item(
            task_id,
            _ATTENTION_REASON_STALE_FROZEN,
            priority=50,
            detail=_attention_detail(fresh_row, _ATTENTION_REASON_STALE_FROZEN),
        )
        return True, digest_dirty

    pane_hash_changed = (
        previous_pane_hash is not None
        and current_pane_hash is not None
        and previous_pane_hash != current_pane_hash
    )
    active_regex_match = _adapter_activity_regex_matches(adapter, pane_text)
    if (
        pane_hash_changed
        or _last_output_advanced(previous_last_output_ts, current_last_output_ts)
        or active_regex_match
    ):
        resolved = registry.resolve_attention_item(
            task_id,
            _ATTENTION_REASON_STALE_FROZEN,
            resolution_reason="liveness_signal_observed",
        )
        if resolved:
            digest_dirty = True
    return False, digest_dirty


def _on_hang(
    task_id: str,
    row: Dict[str, Any],
    *,
    registry: Optional["SessionOrchestrationRegistry"] = None,
    adapter: Optional["AgentAdapter"] = None,
    pane_text: str = "",
    previous_pane_hash: Optional[str] = None,
    current_pane_hash: Optional[str] = None,
) -> None:
    """Called when a session is potentially hung (RUNNING + pane-hash unchanged
    for N ticks + stale last_output_ts).

    This hook is ONLY called when state is RUNNING (never WAITING_USER
    or PAUSED_HANDOFF) — the call-site in ``_process_row`` gates on
    ``new_state == RUNNING``.

    Inside this function we apply the STATIC thresholds from config:
    - ``hang_idle_ticks``: minimum idle_ticks before hang is declared.
    - ``hang_stale_seconds``: minimum seconds since last output change.
    - ``idle_indicator_regex``: if the adapter declares an active-tool pattern
      and it matches the current pane text, the session is NOT hung.

    On confirmed hang, the stale episode may produce at most one action:
    when ``nudge_count == 0`` the watcher reconciles the channel-level digest,
    optionally posts a thread-local hang notice, sends one auto-nudge/check-in
    via the relay, and increments ``nudge_count``.  Later ticks in the same
    unchanged-pane episode return without another feed action or relay nudge.

    The heartbeat fast-path (accelerant) can only reset liveness when it
    carries fresh activity (its own freshness TTL tracked externally); this
    function makes no assumption about the accelerant — it reads thresholds
    from static config only and does NOT suppress detection based on any
    accelerant signal.

    Parameters
    ----------
    task_id:             Registry key.
    row:                 Full registry row dict (fresh — post-counter-increment).
    registry:            Registry instance (injected by _process_row; defaults to
                         a new production instance if absent).
    adapter:             The adapter for this session (for idle_indicator_regex).
    pane_text:           Current pane capture text (for active-tool indicator check).
    previous_pane_hash:  Pane hash observed before this tick's capture/update.
    current_pane_hash:   Pane hash for this tick's capture.
    """
    hang_idle_ticks, hang_stale_seconds = _load_hang_thresholds()
    if current_pane_hash is None:
        current_pane_hash = _pane_hash(pane_text) if pane_text else None
    if previous_pane_hash is None:
        previous_pane_hash = row.get("last_pane_hash")

    if not _is_stale_frozen_eligible(
        row,
        previous_pane_hash=previous_pane_hash,
        current_pane_hash=current_pane_hash,
        pane_text=pane_text,
        adapter=adapter,
        hang_idle_ticks=hang_idle_ticks,
        hang_stale_seconds=hang_stale_seconds,
    ):
        logger.debug(
            "watcher._on_hang: task_id=%s is not stale/frozen eligible",
            task_id,
        )
        return

    nudge_count = _coerce_non_negative_int(row.get("nudge_count"))
    logger.info(
        "watcher._on_hang: CONFIRMED HANG task_id=%s idle_ticks=%d nudge_count=%d",
        task_id,
        _coerce_non_negative_int(row.get("idle_ticks")),
        nudge_count,
    )

    if not _is_stale_episode_action_eligible(row):
        logger.info(
            "watcher._on_hang: stale episode already acted on for task_id=%s",
            task_id,
        )
        _reconcile_attention_digest(
            registry,
            task_id=task_id,
            context="_on_hang",
        )
        return

    # First action in this stale episode: optional thread notice + relay nudge,
    # then reconcile the digest after nudge_count is advanced.
    try:
        from session_orchestration.feed import push_hang_notification

        push_hang_notification(task_id, row, escalate=False)
    except Exception as exc:
        logger.error(
            "watcher._on_hang: push_hang_notification failed for task_id=%s: %s",
            task_id,
            exc,
        )

    _send_auto_nudge(task_id, row, registry=registry, adapter=adapter)

    # Increment nudge_count so later ticks in this episode do not act again.
    try:
        reg = registry or SessionOrchestrationRegistry()
        reg.increment_counter(task_id, "nudge_count", by=1)
    except Exception as exc:
        logger.error(
            "watcher._on_hang: failed to increment nudge_count for task_id=%s: %s",
            task_id,
            exc,
        )
    _reconcile_attention_digest(
        registry,
        task_id=task_id,
        context="_on_hang",
    )

def _send_auto_nudge(
    task_id: str,
    row: Dict[str, Any],
    *,
    registry: Optional["SessionOrchestrationRegistry"] = None,
    adapter: Optional["AgentAdapter"] = None,
) -> None:
    """Send exactly one auto-nudge via the relay (lock-gated).

    Called only when ``nudge_count == 0``.  Non-fatal: logs on failure
    and returns rather than raising so the watcher tick continues.

    Parameters
    ----------
    task_id:   Registry key.
    row:       Full registry row dict.
    registry:  Registry instance (injected for testability).
    adapter:   Adapter for this session (needed by SessionRelay).
    """
    if adapter is None:
        logger.warning(
            "watcher._send_auto_nudge: no adapter available for task_id=%s — cannot nudge",
            task_id,
        )
        return

    try:
        from session_orchestration.relay import SessionRelay
        from session_orchestration.watcher import _build_handle_from_row

        reg = registry or SessionOrchestrationRegistry()
        relay = SessionRelay(reg, adapter)
        handle = _build_handle_from_row(row)
        nudge_msg = (
            "Hermes nudge: your session appears to have stalled. "
            "If you are waiting for me, please continue — otherwise carry on."
        )
        relay.send_message(task_id, handle, nudge_msg, retry_on_conflict=True)
        logger.info(
            "watcher._send_auto_nudge: nudge sent for task_id=%s", task_id
        )
    except Exception as exc:
        logger.error(
            "watcher._send_auto_nudge: failed to send nudge for task_id=%s: %s",
            task_id,
            exc,
        )


def _build_handle_from_row(row: Dict[str, Any]) -> "SessionHandle":
    """Reconstruct a ``SessionHandle`` from a registry row dict.

    Standalone helper (mirrors ``SessionWatcher._build_handle``) so that
    ``_send_auto_nudge`` — which runs outside a watcher instance — can
    build a handle without importing the class.
    """
    from datetime import datetime, timezone

    tmux_session = row.get("tmux_session") or ""
    pane = tmux_session if tmux_session else ""
    return SessionHandle(
        session_id=row.get("task_id", ""),
        tmux_session=tmux_session,
        pane=pane,
        launch_ts=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Terminate adapter helper (PART B)
# ---------------------------------------------------------------------------


def _handle_terminate_adapter(
    intent: Dict[str, Any],
    registry: "SessionOrchestrationRegistry",
    adapter_map: Dict[str, "AgentAdapter"],
    *,
    _spawn_fn=None,
) -> None:
    """Adapter-side terminate: kill the tmux session and optionally re-spawn.

    Called AFTER _apply_intent has already marked the registry row terminal.
    This function handles the tmux kill and optional restart.

    Parameters
    ----------
    intent:      The raw intent dict (from drain_intents).
    registry:    The registry instance (for row lookup).
    adapter_map: Dict of available adapters keyed by agent name.
    _spawn_fn:   Injectable spawn callable (defaults to spawn_session).
                 Pass a fake in tests to avoid real tmux launches.
    """
    import json as _json

    payload: Dict[str, Any] = _json.loads(intent.get("payload", "{}"))
    task_id = intent.get("task_id") or payload.get("task_id")
    if not task_id:
        logger.warning("_handle_terminate_adapter: intent missing task_id")
        return

    restart = payload.get("restart", False)
    row = registry.get(task_id)
    if row is None:
        logger.warning(
            "_handle_terminate_adapter: row not found for task_id=%s", task_id
        )
        return

    agent_name = row.get("agent", "")
    adapter = adapter_map.get(agent_name)
    if adapter is None:
        logger.warning(
            "_handle_terminate_adapter: no adapter for agent=%s task_id=%s",
            agent_name,
            task_id,
        )
        # Still attempt restart if requested — adapter kill is best-effort
    else:
        handle = _build_handle_from_row(row)
        try:
            adapter.terminate(handle)
            logger.info(
                "_handle_terminate_adapter: terminated task_id=%s", task_id
            )
        except Exception as exc:
            logger.error(
                "_handle_terminate_adapter: adapter.terminate() failed for task_id=%s: %s",
                task_id,
                exc,
            )

    if restart:
        # LIMITATION: the registry row does not persist the original prompt.
        # Re-spawn with a placeholder "continue" prompt so the session is
        # restarted best-effort.  A future task could persist the prompt to
        # enable faithful restart.
        if _spawn_fn is None:
            from session_orchestration.spawn import SpawnRequest, spawn_session as _spawn_fn  # type: ignore[assignment]

        from session_orchestration.spawn import SpawnRequest

        workdir = row.get("workdir") or ""
        request = SpawnRequest(
            prompt="continue",  # placeholder — original prompt not persisted
            agent=agent_name,
            workdir=workdir,
        )
        try:
            _spawn_fn(request)
            logger.info(
                "_handle_terminate_adapter: re-spawned task_id=%s (placeholder prompt)",
                task_id,
            )
        except Exception as exc:
            logger.error(
                "_handle_terminate_adapter: re-spawn failed for task_id=%s: %s",
                task_id,
                exc,
            )


# ---------------------------------------------------------------------------
# Re-nudge helper
# ---------------------------------------------------------------------------


def _check_renudge(
    task_id: str,
    row: dict,
    now: float,
    *,
    registry: "SessionOrchestrationRegistry",
    send_dm_fn,
) -> None:
    """Fire a re-nudge DM if the attention interval has elapsed (non-fatal).

    Called from ``_process_row`` on non-transition ticks where the session
    remains in an attention state (WAITING_USER or PAUSED_HANDOFF) without
    user action.  Fires at most once per ``renudge_after_seconds``.

    The caller wraps this in a try/except so any unexpected error is logged
    but does not crash the watcher tick.

    Parameters
    ----------
    task_id:       Registry key.
    row:           Fresh registry row dict (must include ``attention_since``,
                   ``last_renudge_at``, and ``discord_user_id``).
    now:           Current epoch (injectable for tests; from ``SessionWatcher._now_fn``).
    registry:      Registry instance (for persisting ``last_renudge_at``).
    send_dm_fn:    Callable ``(user_id: str, msg: str) -> bool``.  The production
                   default is ``_default_send_dm``; tests inject a recording fake.
    """
    renudge_after = _load_renudge_after_seconds_cfg()
    if renudge_after <= 0:
        logger.debug(
            "watcher._check_renudge: disabled (renudge_after_seconds=%d) for task_id=%s",
            renudge_after, task_id,
        )
        return

    attention_since = row.get("attention_since")
    if attention_since is None:
        # No stamp yet — the entering-attention set_attention_stamps hasn't been
        # read by this fresh_row yet (race-free: stamp is written, then fresh_row
        # is read, so this branch is only hit when the DB row genuinely has no stamp).
        logger.debug(
            "watcher._check_renudge: no attention_since for task_id=%s — skipping",
            task_id,
        )
        return

    attention_since_f = float(attention_since)
    last_renudge_at = row.get("last_renudge_at")
    reference_ts = max(attention_since_f, float(last_renudge_at or 0.0))

    if now - reference_ts < renudge_after:
        logger.debug(
            "watcher._check_renudge: within interval (%.0fs remaining) for task_id=%s",
            renudge_after - (now - reference_ts), task_id,
        )
        return

    # Interval elapsed — fire re-nudge DM
    user_id = row.get("discord_user_id")
    if user_id:
        wait_min = max(1, int((now - attention_since_f) / 60))
        task_label = row.get("project") or row.get("repo") or task_id
        msg = f"{task_label} still needs your input — waiting {wait_min}m"
        try:
            send_dm_fn(user_id, msg)
        except Exception as exc:
            logger.error(
                "watcher._check_renudge: DM failed for task_id=%s: %s",
                task_id, exc,
            )
    else:
        logger.debug(
            "watcher._check_renudge: no discord_user_id for task_id=%s — skipping DM",
            task_id,
        )

    # Always persist last_renudge_at so the next fire waits a full interval,
    # regardless of whether the DM succeeded (best-effort).
    try:
        registry.set_attention_stamps(task_id, attention_since_f, now)
        logger.info(
            "watcher._check_renudge: re-nudge fired for task_id=%s "
            "(attention for %.0fs, last_renudge_at updated)",
            task_id, now - attention_since_f,
        )
    except Exception as exc:
        logger.error(
            "watcher._check_renudge: failed to persist last_renudge_at for task_id=%s: %s",
            task_id, exc,
        )


# ---------------------------------------------------------------------------
# Core watcher
# ---------------------------------------------------------------------------


class SessionWatcher:
    """Core watcher loop — sole mutator of the session-orchestration registry.

    Parameters
    ----------
    registry:
        The ``SessionOrchestrationRegistry`` instance.  Defaults to the
        production DB (hermes state.db) when not injected.
    adapters:
        Adapter dict *before* verification.  The watcher calls
        ``verify_adapters`` at startup and uses only the verified subset.
    tmux_capture:
        Callable ``(pane: str) -> str``.  Defaults to the real subprocess
        implementation.  Inject a fake in tests.
    lock_ttl_seconds:
        Lock TTL forwarded to ``registry.acquire_lock``.
    """

    def __init__(
        self,
        registry: Optional[SessionOrchestrationRegistry] = None,
        adapters: Optional[Dict[str, AgentAdapter]] = None,
        *,
        tmux_capture=None,
        lock_ttl_seconds: float = _LOCK_TTL_SECONDS,
        tmux_liveness_fn: Optional[Callable[[str], bool]] = None,
        _now_fn: Optional[Callable[[], float]] = None,
        _send_dm_fn=None,
    ) -> None:
        self._registry = registry or SessionOrchestrationRegistry()
        self._raw_adapters: Dict[str, AgentAdapter] = adapters or {}
        self._tmux_capture = tmux_capture or _TmuxCapture()
        self._lock_ttl = lock_ttl_seconds
        # Injectable tmux-liveness check (``tmux has-session``). Tests inject a
        # fake so no real subprocess is spawned; production uses the default.
        self._tmux_liveness_fn: Callable[[str], bool] = (
            tmux_liveness_fn if tmux_liveness_fn is not None else _default_tmux_liveness
        )
        # Injectable clock — defaults to ``time.time``.  Tests inject a fixed
        # value so the attention-since / re-nudge interval logic is deterministic.
        self._now_fn: Callable[[], float] = _now_fn if _now_fn is not None else time.time
        # Injectable DM sender — defaults to ``_default_send_dm``.  Tests inject
        # a recording fake so no real Discord API calls are made.
        self._send_dm_fn = _send_dm_fn if _send_dm_fn is not None else _default_send_dm
        # Available adapters are populated once in run() / tick()
        self._available_adapters: Dict[str, AgentAdapter] = {}
        self._verified = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def startup_verify(self, **verify_kwargs) -> Dict[str, AgentAdapter]:
        """Run verify_adapters once and cache the result.

        Safe to call multiple times — subsequent calls are no-ops if
        already verified (idempotent).  In tests, pass ``probe_runner``
        and ``probe_specs`` via ``verify_kwargs`` to inject fakes.

        Returns the dict of available adapters (may be a subset of
        ``self._raw_adapters`` if some failed verification).
        """
        if not self._verified:
            self._available_adapters = verify_adapters(
                self._raw_adapters, **verify_kwargs
            )
            self._verified = True
        return self._available_adapters

    def recommended_next_tick_interval(self) -> float:
        """Return the advised tick interval in seconds.

        Returns ``_FAST_TICK_SECONDS`` (30 s) when any registry row is in
        RUNNING or WAITING_USER state (active sessions need faster polling).
        Returns ``_IDLE_TICK_SECONDS`` (120 s) otherwise.

        This is an advisory value for the cron driver — the watcher itself
        does not self-schedule.  Best-effort: returns ``_IDLE_TICK_SECONDS``
        on any registry error so the driver falls back to a safe cadence.
        """
        try:
            rows = self._registry.list()
            fast_states = {
                SessionLifecycle.RUNNING.value,
                SessionLifecycle.WAITING_USER.value,
            }
            if any(r.get("state") in fast_states for r in rows):
                return _FAST_TICK_SECONDS
            return _IDLE_TICK_SECONDS
        except Exception as exc:
            logger.debug(
                "watcher.recommended_next_tick_interval: registry error: %s", exc
            )
            return _IDLE_TICK_SECONDS

    def tick(self) -> int:
        """Execute one watcher tick.

        1. Drain + apply intent queue.
        2. Iterate active registry rows.
        3. For each row with an available adapter:
           a. Acquire per-session lock — skip row if locked.
           b. Capture pane text.
           c. Release lock.
           d. Detect new state via ``adapter.detect()``.
           e. Update registry state + counters.
           f. Fire transition hooks.

        Returns the number of rows successfully processed (lock-skips
        and unavailable-adapter rows are not counted).
        """
        if not self._verified:
            self.startup_verify()

        # Step 1 — drain intent queue (single-writer responsibility)
        intents = self._registry.drain_intents()
        drive_signal_task_ids = {
            task_id
            for intent in intents
            if intent.get("intent") == "drive"
            for task_id in [_intent_task_id(intent)]
            if task_id
        }
        for intent in intents:
            try:
                self._registry._apply_intent(intent)
            except Exception as exc:
                logger.error(
                    "watcher.tick: intent application failed for %r: %s",
                    intent,
                    exc,
                )

        # Adapter-side terminate: kill tmux + optional restart (AFTER registry row
        # is already marked terminal by _apply_intent above).
        terminate_intents = [i for i in intents if i.get("intent") == "terminate"]
        terminate_reaped = False
        for term_intent in terminate_intents:
            try:
                _handle_terminate_adapter(
                    term_intent, self._registry, self._available_adapters
                )
            except Exception as exc:
                logger.error(
                    "watcher.tick: _handle_terminate_adapter failed for %r: %s",
                    term_intent,
                    exc,
                )
            # Reap attention/feed state for the terminated session. _apply_intent
            # marks the row terminal but terminal rows are skipped by the active
            # loop below, so _sync_attention_lifecycle never runs for them — a
            # WAITING_USER attention item + feed router line would otherwise
            # dangle after a thread-archive → terminate. Resolve every attention
            # reason and clear the ping debounce here.
            term_task_id = _intent_task_id(term_intent)
            if not term_task_id:
                continue
            try:
                for reason in (
                    *_USER_ATTENTION_STATES,
                    _ATTENTION_REASON_STALE_FROZEN,
                ):
                    if self._registry.resolve_attention_item(
                        term_task_id, reason, resolution_reason="terminated"
                    ):
                        terminate_reaped = True
                from session_orchestration.feed import clear_last_notified

                clear_last_notified(term_task_id)
            except Exception as exc:
                logger.error(
                    "watcher.tick: terminate reap failed for task_id=%s: %s",
                    term_task_id,
                    exc,
                )
        if terminate_reaped:
            # Drop the resolved item(s) from the single feed-channel digest.
            _reconcile_attention_digest(
                self._registry, task_id=None, context="terminate_reap"
            )

        # Step 2 — iterate active rows
        rows = self._registry.list()
        active_rows = [r for r in rows if r.get("state") in _ACTIVE_STATES]

        processed = 0
        for row in active_rows:
            task_id = row["task_id"]
            agent_name = row.get("agent", "")

            # Skip rows whose adapter is unavailable
            adapter = self._available_adapters.get(agent_name)
            if adapter is None:
                logger.debug(
                    "watcher.tick: skipping task_id=%s — adapter %r unavailable",
                    task_id,
                    agent_name,
                )
                continue

            try:
                did_process = self._process_row(
                    task_id,
                    row,
                    adapter,
                    user_drive_signal=task_id in drive_signal_task_ids,
                )
                if did_process:
                    processed += 1
            except Exception as exc:
                logger.error(
                    "watcher.tick: error processing task_id=%s: %s",
                    task_id,
                    exc,
                    exc_info=True,
                )

        # Step 3 — GC terminal rows
        gc_after = _load_gc_after_seconds_cfg()
        if gc_after > 0:
            try:
                deleted = self._registry.gc_terminal_rows(
                    now=time.time(), max_age_seconds=gc_after
                )
                if deleted > 0:
                    logger.info(
                        "watcher.tick: gc_terminal_rows deleted %d row(s) "
                        "(gc_after_seconds=%d)",
                        deleted,
                        gc_after,
                    )
            except Exception as exc:
                logger.error("watcher.tick: gc_terminal_rows failed: %s", exc)

        return processed

    # ------------------------------------------------------------------
    # Internal — single-row processing
    # ------------------------------------------------------------------

    def _process_row(
        self,
        task_id: str,
        row: Dict[str, Any],
        adapter: AgentAdapter,
        *,
        user_drive_signal: bool = False,
    ) -> bool:
        """Process a single registry row in one tick.

        Returns True if the row was fully processed, False if it was skipped
        (e.g. because the per-session lock was held by the relay).
        """
        # ------------------------------------------------------------------
        # 0. Read new marker lines (append-only; no lock needed)
        # ------------------------------------------------------------------
        workdir = row.get("workdir")
        old_marker_offset = int(row.get("marker_offset") or 0)
        new_offset = old_marker_offset
        new_markers: List[Any] = []

        if workdir:
            marker_file = f"{workdir}/.hermes/sessions/{task_id}.jsonl"
            try:
                new_markers, new_offset = read_markers_since(marker_file, old_marker_offset)
            except OSError as exc:
                logger.warning(
                    "watcher._process_row: read_markers_since failed for task_id=%s: %s",
                    task_id,
                    exc,
                )

        now = self._now_fn()
        recent_markers = [
            m for m in new_markers
            if _parse_marker_ts(m.get("ts", "")) >= now - _MARKER_RECENCY_SECONDS
        ]
        has_recent_markers = bool(recent_markers)

        # ------------------------------------------------------------------
        # 1. Acquire per-session lock — BEFORE any capture-pane call
        # ------------------------------------------------------------------
        # The registry has tmux_session but not a separate pane column;
        # derive the pane target using the same logic as _build_handle.
        tmux_session = row.get("tmux_session") or ""
        pane = tmux_session if tmux_session else ""
        holder = f"watcher:pid:{os.getpid()}:{time.time():.3f}"

        lock_acquired = self._registry.acquire_lock(
            task_id, holder, ttl_seconds=self._lock_ttl
        )
        if not lock_acquired:
            logger.debug(
                "watcher.tick: lock held for task_id=%s — skipping capture this tick",
                task_id,
            )
            return False  # SKIP; never read a half-rendered pane

        pane_text = ""
        # Pre-compute marker kind from in-memory recent_markers — append-only read, no lock needed.
        latest_marker_kind = recent_markers[-1].get("kind", "") if recent_markers else ""
        latest_marker_payload = recent_markers[-1].get("payload", {}) if recent_markers else {}

        try:
            # ------------------------------------------------------------------
            # Dead-tmux reap: if the tmux session is gone and there are no recent
            # markers (no heartbeat proving liveness), mark the row terminal
            # immediately rather than waiting out the hang-nudge ladder.
            # Gated on config knob dead_tmux_reap (default True).
            # Runs INSIDE the lock so the relay cannot be mid-write.
            # Grace: skipped when has_recent_markers (session may have just spawned
            # or be alive but with a silent pane).
            # ------------------------------------------------------------------
            if tmux_session and not has_recent_markers:
                _so_cfg = _load_dead_tmux_reap_cfg()
                if _so_cfg:
                    pane_gone = not self._tmux_liveness_fn(tmux_session)
                    if pane_gone:
                        # Determine terminal state: DONE if the last known marker
                        # was a 'done' kind, ERROR otherwise.
                        all_markers, _ = [], new_offset
                        try:
                            if workdir:
                                all_markers, _ = read_markers_since(
                                    f"{workdir}/.hermes/sessions/{task_id}.jsonl", 0
                                )
                        except OSError:
                            pass
                        last_kind = all_markers[-1].get("kind", "") if all_markers else ""
                        terminal_state = (
                            SessionLifecycle.DONE.value
                            if last_kind == "done"
                            else SessionLifecycle.ERROR.value
                        )
                        logger.info(
                            "watcher._process_row: dead-tmux reap task_id=%s "
                            "tmux_session=%s -> %s",
                            task_id,
                            tmux_session,
                            terminal_state,
                        )
                        self._registry.upsert(
                            task_id,
                            agent=row.get("agent", "unknown"),
                            run_id=row.get("run_id"),
                            repo=row.get("repo"),
                            source=row.get("source", "spawn"),
                            state=terminal_state,
                            terminated_at=now,
                        )
                        return True

            # Capture pane while we hold the lock
            if pane:
                pane_text = self._tmux_capture(pane)

            # ------------------------------------------------------------------
            # 2. Determine authoritative lifecycle state (inside lock so the
            #    handoff_continue resume runs atomically with state assignment)
            #    Priority: recent marker > pane detect
            # ------------------------------------------------------------------
            marker_lifecycle: Optional[SessionLifecycle] = None
            if has_recent_markers:
                latest_kind = recent_markers[-1]["kind"]
                marker_lifecycle = marker_kind_to_lifecycle(latest_kind)

            if marker_lifecycle is not None:
                new_lifecycle = marker_lifecycle
            else:
                # Fall back to pane-scraping via adapter.detect()
                handle = self._build_handle(row)
                try:
                    new_lifecycle = adapter.detect(handle)
                except Exception as exc:
                    logger.error(
                        "watcher.tick: adapter.detect() failed for task_id=%s: %s",
                        task_id,
                        exc,
                    )
                    return

            # ------------------------------------------------------------------
            # Handoff continue: resume INSIDE lock — avoids racing the relay's
            # PAUSED_HANDOFF resume path on the same tmux pane (lock contract).
            # Because we set RUNNING here under the lock, the relay won't
            # subsequently see PAUSED_HANDOFF and double-resume.
            # ------------------------------------------------------------------
            if latest_marker_kind == "handoff_continue":
                handle = _build_handle_from_row(row)
                try:
                    # Pane mutation runs under per-session lock (avoids race with relay's PAUSED_HANDOFF resume)
                    adapter.resume(handle, "")  # autonomous /clear+resume, no user reply
                    new_lifecycle = SessionLifecycle.RUNNING  # override PAUSED_HANDOFF
                except Exception as exc:
                    logger.error(
                        "watcher._process_row: handoff_continue resume failed: %s", exc
                    )
                    # new_lifecycle stays as marker-derived value on failure

        finally:
            # Always release — crash-safe because relay reclaims after TTL
            self._registry.release_lock(task_id, holder)

        # ------------------------------------------------------------------
        # Handoff decision: DM the user with the question — NOT a pane/tmux
        # mutation, so the per-session lock is NOT required here.
        # new_lifecycle stays PAUSED_HANDOFF; _on_turn_change fires for the transition.
        # ------------------------------------------------------------------
        if latest_marker_kind == "handoff_decision":
            question = (latest_marker_payload or {}).get(
                "question", "(handoff decision required)"
            )
            user_id = row.get("discord_user_id")
            if user_id:
                try:
                    from tools.discord_tool import _get_bot_token  # type: ignore[import]
                    from session_orchestration.dm_transport import send_dm

                    token = _get_bot_token()
                    if token:
                        send_dm(user_id, f"Handoff decision needed: {question}", token)
                except Exception as exc:
                    logger.error(
                        "watcher._process_row: handoff_decision DM failed: %s", exc
                    )
            # new_lifecycle stays PAUSED_HANDOFF; _on_turn_change fires for the transition

        # ------------------------------------------------------------------
        # omp free-form wait promotion. omp gives no ❯/menu-footer/marker when
        # it finishes a turn and idles at its composer, so detect() reports
        # RUNNING and the session trips the hang ladder instead of surfacing its
        # question. Promote a STABLE idle omp pane (unchanged since the prior
        # tick, not busy) to WAITING_USER so the question reaches the thread.
        # The stability gate (a prior hash exists AND equals this tick's hash)
        # avoids flapping on a momentary mid-turn idle. Only adapters that opt
        # in via idle_waiting() (omp) are affected; Claude uses ❯ and is already
        # detected directly.
        # ------------------------------------------------------------------
        if new_lifecycle == SessionLifecycle.RUNNING and pane_text:
            _idle_fn = getattr(adapter, "idle_waiting", None)
            if callable(_idle_fn):
                _cur_hash = hashlib.sha256(
                    pane_text.encode(errors="replace")
                ).hexdigest()[:16]
                _prev_hash = row.get("last_pane_hash")
                try:
                    _is_idle = bool(_idle_fn(pane_text))
                except Exception as exc:
                    logger.debug(
                        "watcher._process_row: idle_waiting() failed for %s: %s",
                        task_id, exc,
                    )
                    _is_idle = False
                if _prev_hash and _cur_hash == _prev_hash and _is_idle:
                    new_lifecycle = SessionLifecycle.WAITING_USER

        new_state = new_lifecycle.value
        old_state = row.get("state", SessionLifecycle.RUNNING.value)
        previous_last_output_ts = _coerce_optional_float(row.get("last_output_ts"))

        # Attention-state transition flags (used for stamp/clear and re-nudge below)
        entering_attention = (
            new_state in _ATTENTION_STATES
            and old_state not in _ATTENTION_STATES
        )
        leaving_attention = (
            old_state in _ATTENTION_STATES
            and new_state not in _ATTENTION_STATES
        )
        staying_attention = (
            new_state == old_state
            and new_state in _ATTENTION_STATES
        )

        # Compute pane hash for hang-detection (T009 consumes this)
        if pane_text:
            pane_hash = hashlib.sha256(pane_text.encode(errors="replace")).hexdigest()[:16]
        else:
            pane_hash = None

        previous_pane_hash = row.get("last_pane_hash")
        captured_pane_hash = pane_hash is not None
        pane_changed = captured_pane_hash and pane_hash != previous_pane_hash
        idle_tick_delta = 1 if captured_pane_hash and not pane_changed else 0

        # Build update fields
        update_fields: Dict[str, Any] = {
            "state": new_state,
            "updated_at": "datetime('now')",  # handled by upsert
        }
        if captured_pane_hash:
            update_fields["last_pane_hash"] = pane_hash
        if pane_changed:
            update_fields["last_output_ts"] = now
            update_fields["idle_ticks"] = 0
            update_fields["nudge_count"] = 0  # reset stale-episode action counter on activity
        else:
            # Atomic increment — done via registry.increment_counter
            pass  # handled below
        # Persist advanced marker offset (only when new lines were consumed)
        if new_offset != old_marker_offset:
            update_fields["marker_offset"] = new_offset

        # ------------------------------------------------------------------
        # Answerable needs-input: persist the question + option labels so the
        # feed can present them and a numeric reply can be resolved. Two paths:
        #   (a) marker path (Claude Code emits a needs_input marker with a
        #       structured question/options payload); authoritative when present.
        #   (b) pane path (omp emits NO markers) — parse the captured pane text.
        # These fields are only meaningful while WAITING_USER; clear
        # last_input_kind when the session moves on so a stale "menu" can't
        # mislead the answer route or the feed digest.
        # ------------------------------------------------------------------
        needs_input_markers = [m for m in new_markers if m.get("kind") == "needs_input"]
        if needs_input_markers:
            ni_payload = needs_input_markers[-1].get("payload") or {}
            question = ni_payload.get("question", "")
            options = ni_payload.get("options") or []
            if question:
                update_fields["last_question"] = question
            update_fields["last_options"] = json.dumps(options)
            update_fields["last_input_kind"] = "menu" if options else "prompt"
        elif new_state == SessionLifecycle.WAITING_USER.value and pane_text:
            question, options, is_menu = extract_menu_context(pane_text)
            if question:
                update_fields["last_question"] = question
            update_fields["last_options"] = json.dumps(options)
            update_fields["last_input_kind"] = "menu" if is_menu else "prompt"
        elif new_state not in _ATTENTION_STATES:
            # Left the waiting state — drop the menu marker so the next
            # free-form reply is not misrouted as a stale menu selection.
            update_fields["last_input_kind"] = ""
            update_fields["last_options"] = json.dumps([])

        # Write state update (single writer)
        self._registry.upsert(
            task_id,
            agent=row.get("agent", "unknown"),
            run_id=row.get("run_id"),
            repo=row.get("repo"),
            source=row.get("source", "spawn"),
            **{k: v for k, v in update_fields.items() if k not in ("updated_at",)},
        )

        # Atomic idle-tick increment (separate atomic SQL expression)
        if idle_tick_delta > 0:
            self._registry.increment_counter(task_id, "idle_ticks", by=idle_tick_delta)

        # Heartbeat counter — always bump (T008 gates on modulo)
        self._registry.increment_counter(task_id, "heartbeat_counter", by=1)

        # Stamp/clear attention_since and last_renudge_at.  Done BEFORE the
        # fresh_row read so fresh_row reflects the updated values for the hooks.
        # These writes are sequential within the single-writer tick — no race.
        if entering_attention:
            try:
                self._registry.set_attention_stamps(task_id, now, None)
            except Exception as exc:
                logger.error(
                    "watcher._process_row: set_attention_stamps (enter) failed "
                    "for task_id=%s: %s",
                    task_id, exc,
                )
        elif leaving_attention:
            try:
                self._registry.set_attention_stamps(task_id, None, None)
            except Exception as exc:
                logger.error(
                    "watcher._process_row: set_attention_stamps (leave) failed "
                    "for task_id=%s: %s",
                    task_id, exc,
                )

        # Re-read fresh row for hook calls (reflects counters + attention stamps)
        fresh_row = self._registry.get(task_id) or row
        current_last_output_ts = _coerce_optional_float(fresh_row.get("last_output_ts"))
        hang_idle_ticks, hang_stale_seconds = _load_hang_thresholds()
        stale_frozen_eligible = (
            new_state == SessionLifecycle.RUNNING.value
            # Marker recency proves liveness even if the pane hash is frozen —
            # a recent heartbeat/status marker suppresses stale-frozen detection
            # (and therefore both the digest entry and the hang nudge).
            and not has_recent_markers
            and _is_stale_frozen_eligible(
                fresh_row,
                previous_pane_hash=previous_pane_hash,
                current_pane_hash=pane_hash,
                pane_text=pane_text,
                adapter=adapter,
                hang_idle_ticks=hang_idle_ticks,
                hang_stale_seconds=hang_stale_seconds,
            )
        )
        stale_frozen_actionable, attention_digest_dirty = _sync_attention_lifecycle(
            self._registry,
            task_id,
            old_state=old_state,
            new_state=new_state,
            fresh_row=fresh_row,
            stale_frozen_eligible=stale_frozen_eligible,
            previous_pane_hash=previous_pane_hash,
            current_pane_hash=pane_hash,
            previous_last_output_ts=previous_last_output_ts,
            current_last_output_ts=current_last_output_ts,
            pane_text=pane_text,
            adapter=adapter,
            user_drive_signal=user_drive_signal,
        )
        transition_digest_reconciled = False

        if new_state != old_state and new_state in _USER_ATTENTION_STATES:
            _on_turn_change(
                task_id,
                fresh_row,
                new_state,
                old_state,
                registry=self._registry,
            )
            transition_digest_reconciled = True
        elif new_state != old_state and old_state in _USER_ATTENTION_STATES:
            # Transitioning OUT of an attention state — re-arm the debounce
            # so the NEXT transition back fires as a new notification.
            try:
                from session_orchestration.feed import clear_last_notified

                clear_last_notified(task_id)
            except Exception as exc:
                logger.debug(
                    "watcher._process_row: clear_last_notified failed for task_id=%s: %s",
                    task_id,
                    exc,
                )
            _reconcile_attention_digest(
                self._registry,
                task_id=task_id,
                context="_process_row",
            )
            transition_digest_reconciled = True

        if attention_digest_dirty and not transition_digest_reconciled:
            _reconcile_attention_digest(
                self._registry,
                task_id=task_id,
                context="_process_row",
            )
            transition_digest_reconciled = True
        if (
            new_state in _USER_ATTENTION_STATES
            or stale_frozen_actionable
            or transition_digest_reconciled
        ):
            logger.debug(
                "watcher._process_row: digest owns channel projection; skipping heartbeat task_id=%s",
                task_id,
            )
        else:
            # Heartbeat hook (T008 decides whether to actually edit based on counter)
            _on_heartbeat_tick(task_id, fresh_row)

        # Hang hook — only when deterministic stale/frozen evidence is present.
        if stale_frozen_actionable:
            _on_hang(
                task_id,
                fresh_row,
                registry=self._registry,
                adapter=adapter,
                pane_text=pane_text,
                previous_pane_hash=previous_pane_hash,
                current_pane_hash=pane_hash,
            )

        # Re-nudge check — fires at most once per renudge_after_seconds when
        # the session stays in an attention state without user action.
        # This is the COMPLEMENT of the turn-change hook: same state, no transition.
        if staying_attention:
            try:
                _check_renudge(
                    task_id, fresh_row, now,
                    registry=self._registry,
                    send_dm_fn=self._send_dm_fn,
                )
            except Exception as exc:
                logger.error(
                    "watcher._process_row: _check_renudge failed for task_id=%s: %s",
                    task_id, exc,
                )

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_handle(row: Dict[str, Any]) -> SessionHandle:
        """Reconstruct a ``SessionHandle`` from a registry row dict.

        The registry schema stores ``tmux_session`` (session name) but not
        the pane target.  We derive the pane as ``<tmux_session>:0.0``,
        which is the default for single-pane sessions created by adapters.
        """
        tmux_session = row.get("tmux_session") or ""
        pane = tmux_session if tmux_session else ""
        return SessionHandle(
            session_id=row.get("task_id", ""),
            tmux_session=tmux_session,
            pane=pane,
            launch_ts=datetime.now(tz=timezone.utc),
        )


# ---------------------------------------------------------------------------
# Convenience: run one tick (called by the --no-agent cron script)
# ---------------------------------------------------------------------------


def run_tick(
    registry: Optional[SessionOrchestrationRegistry] = None,
    adapters: Optional[Dict[str, AgentAdapter]] = None,
    tmux_capture=None,
    *,
    probe_runner=None,
    probe_specs=None,
    tmux_liveness_fn: Optional[Callable[[str], bool]] = None,
) -> int:
    """Run a single watcher tick; return number of rows processed.

    This is the entry point called by ``session-orchestration-watch.sh``.
    The shell script passes no arguments (uses defaults); tests inject fakes.
    """
    watcher = SessionWatcher(
        registry=registry,
        adapters=adapters,
        tmux_capture=tmux_capture,
        tmux_liveness_fn=tmux_liveness_fn,
    )
    verify_kwargs: Dict[str, Any] = {}
    if probe_runner is not None:
        verify_kwargs["probe_runner"] = probe_runner
    if probe_specs is not None:
        verify_kwargs["probe_specs"] = probe_specs
    watcher.startup_verify(**verify_kwargs)
    return watcher.tick()


# ---------------------------------------------------------------------------
# Config gate helper
# ---------------------------------------------------------------------------


def _is_session_orchestration_enabled() -> bool:
    """Return True iff ``session_orchestration.enabled`` is truthy in config.

    Safe to call without the Hermes config file present (returns False).
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        so_cfg = cfg.get("session_orchestration") or {}
        return bool(so_cfg.get("enabled", False))
    except Exception as exc:
        logger.debug("watcher: could not read config: %s", exc)
        return False


# ---------------------------------------------------------------------------
# __main__ entry-point (used by the cron shell script)
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point — gated on ``session_orchestration.enabled``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not _is_session_orchestration_enabled():
        logger.info("watcher: session_orchestration.enabled is false — exiting")
        return

    from session_orchestration.adapters.claude_code import ClaudeCodeAdapter
    from session_orchestration.adapters.omp import OmpAdapter

    adapters: Dict[str, AgentAdapter] = {
        "claude": ClaudeCodeAdapter(),
        "omp": OmpAdapter(),
    }

    n = run_tick(adapters=adapters)
    logger.info("watcher: tick complete — %d rows processed", n)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _pane_hash(text: str) -> str:
    """Return a short SHA-256 hex prefix of pane text for change detection."""
    return hashlib.sha256(text.encode(errors="replace")).hexdigest()[:16]
