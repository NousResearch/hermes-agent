"""Kanban board watcher methods for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3).
These are the background-loop methods that subscribe to kanban boards, deliver
notifications/artifacts, and drive the multi-agent dispatcher. They use only
``self`` state, so they live on a mixin that ``GatewayRunner`` inherits — the
``self._kanban_*`` call sites resolve identically via the MRO, making this a
behavior-neutral move that lifts ~1,000 LOC out of run.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from agent.i18n import t
from gateway.kanban_status_card import (
    render_kanban_active_task_index,
    render_kanban_status_card,
    user_facing_title,
)
from gateway.topic_anchors import get_topic_anchor as _get_topic_anchor

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")


# Completion manifests can contain implementation evidence as well as a human
# deliverable. Automatic uploads are therefore fail-closed and explicit.
_KANBAN_ARTIFACT_DENIED_SUFFIXES = frozenset({
    ".py", ".sh", ".bash", ".zsh", ".js", ".ts", ".swift", ".rb", ".pl",
    ".go", ".rs", ".java", ".kt", ".c", ".cc", ".cpp", ".h", ".sql",
    ".json", ".lock", ".log", ".cache", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".md", ".txt", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".7z", ".rar",
})
_KANBAN_EXPLICIT_TEXT_ARTIFACT_SUFFIXES = frozenset({".md", ".markdown", ".txt"})
_KANBAN_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".heic"})
_KANBAN_VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"})
_KANBAN_AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".opus"})
_KANBAN_DOCUMENT_SUFFIXES = frozenset({
    ".pdf", ".doc", ".docx", ".odt", ".rtf", ".xls", ".xlsx", ".ods",
    ".ppt", ".pptx", ".odp",
})

# Bump only when the card projection contract changes. Stored on each durable
# surface so an upgraded gateway can re-render stale cards once without posting
# a replacement message or repeating that edit on a later restart.
_KANBAN_STATUS_RENDERER_VERSION = "2026-07-15.5"
_KANBAN_ACTIVE_INDEX_RENDERER_VERSION = "2026-07-15.6"


def _active_index_link_label(item: tuple[Any, ...]) -> str:
    """Return the rendered, mobile-safe title span used by an index entry."""
    task, timeline = item[1], item[2]
    return " ".join(user_facing_title(task, timeline, str(getattr(task, "id", ""))).split())[:72]



def _classify_kanban_artifact(path: str, *, allow_text_document: bool = False) -> Optional[str]:
    """Return a human-facing upload kind for an explicitly requested item only."""
    name = Path(path).name.lower()
    suffix = Path(name).suffix
    if name.startswith("checksums"):
        return None
    if suffix in _KANBAN_EXPLICIT_TEXT_ARTIFACT_SUFFIXES and allow_text_document:
        return "document"
    if suffix in _KANBAN_ARTIFACT_DENIED_SUFFIXES:
        return None
    if suffix in _KANBAN_IMAGE_SUFFIXES:
        return "image"
    if suffix in _KANBAN_VIDEO_SUFFIXES:
        return "video"
    if suffix in _KANBAN_AUDIO_SUFFIXES:
        return "audio"
    if suffix in _KANBAN_DOCUMENT_SUFFIXES:
        return "document"
    return None


def _looks_like_telegram_private_chat_id(chat_id: Any) -> bool:
    """True for a Telegram 1:1 chat id (positive); groups/channels are negative."""
    try:
        return int(chat_id) > 0
    except (TypeError, ValueError):
        return False


def _is_telegram_private_topic_lane(chat_id: Any, thread_id: Any) -> bool:
    """True for a private Telegram topic, whose pins are chat-global."""
    return bool(thread_id) and _looks_like_telegram_private_chat_id(chat_id)


def _kanban_status_route_metadata(adapter: Any, sub: dict, task_status: str) -> dict[str, Any]:
    """Return the one exact route used by status-card create and recreation.

    Hermes private-topic lanes use their native ``message_thread_id`` plus a
    recorded reply anchor. They are not Telegram Bot API Direct Messages chats,
    so this helper never infers ``direct_messages_topic_id``.
    """
    metadata: dict[str, Any] = {}
    platform = str(sub.get("platform") or "").lower()
    if platform == "telegram":
        status_metadata = getattr(adapter, "kanban_status_metadata", None)
        if callable(status_metadata):
            resolved = status_metadata(task_status)
            if isinstance(resolved, dict):
                metadata.update(resolved)
    thread_id = sub.get("thread_id")
    if not thread_id:
        return metadata
    thread_id = str(thread_id)
    metadata["thread_id"] = thread_id
    if _is_telegram_private_topic_lane(sub.get("chat_id"), thread_id) and thread_id != "1":
        metadata["telegram_dm_topic_reply_fallback"] = True
        anchor = _get_topic_anchor(platform, str(sub["chat_id"]), thread_id)
        if anchor:
            metadata["telegram_reply_to_message_id"] = anchor
    return metadata


def _kanban_active_index_metadata(adapter: Any, items: list[tuple[Any, ...]]) -> dict[str, Any]:
    """Reuse each visible status card's Telegram custom emoji in the overview."""
    status_metadata = getattr(adapter, "kanban_status_metadata", None)
    if not callable(status_metadata):
        return {}
    metadata: dict[str, Any] = {}
    custom_emoji: dict[str, str] = {}
    statuses = sorted({str(getattr(item[1], "status", "") or "").lower() for item in items})
    for status in statuses:
        resolved = status_metadata(status)
        if not isinstance(resolved, dict):
            continue
        for key, value in resolved.items():
            if key != "telegram_custom_emoji":
                metadata.setdefault(key, value)
            elif isinstance(value, dict):
                custom_emoji.update({str(icon): str(emoji_id) for icon, emoji_id in value.items()})
    if custom_emoji:
        metadata["telegram_custom_emoji"] = custom_emoji
    return metadata


def _kanban_active_index_route_metadata(
    adapter: Any, lane: dict, items: list[tuple[Any, ...]],
) -> dict[str, Any]:
    """Keep topic routing and custom emoji identical on every index surface."""
    metadata = _kanban_status_route_metadata(adapter, lane, "")
    route_custom_emoji = metadata.pop("telegram_custom_emoji", {})
    metadata.update(_kanban_active_index_metadata(adapter, items))
    index_custom_emoji = metadata.get("telegram_custom_emoji", {})
    if isinstance(route_custom_emoji, dict) and isinstance(index_custom_emoji, dict):
        metadata["telegram_custom_emoji"] = {**route_custom_emoji, **index_custom_emoji}
    return metadata


_TRANSIENT_SEND_ERROR_MARKERS = (
    "flood_control", "flood control", "retry after", "rate limit",
    "timed out", "timeout", "network", "connection", "pool",
    # Private DM topic with no recorded reply anchor yet (topic_anchors.py
    # only learns an anchor from a live inbound message in that lane) —
    # waiting for one to arrive is not a dead chat, don't burn strikes on it.
    "requires a reply anchor",
)


def _is_transient_kanban_send_error(exc: Exception) -> bool:
    """Transient failures (flood control, network blips, no anchor yet) must
    not count toward the dead-chat drop threshold — a 30s Telegram flood
    wait spans several 5s notifier ticks and would otherwise unsubscribe a
    healthy chat, and a DM topic with no anchor yet just needs the user to
    send one message there. Permanent failures (forbidden, chat not found)
    still count."""
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_SEND_ERROR_MARKERS)


def _flood_cooldown_from_failure(error: Exception | str, state: Optional[dict]) -> int:
    """Return a durable Bot API cooldown only for an actual rate-limit error."""
    message = str(error).lower()
    if not any(marker in message for marker in (
        "flood_control", "flood control", "retry after", "rate limit",
    )):
        return 0
    return int((state or {}).get("next_retry_at") or 0)


def _is_deleted_telegram_edit(result: Any, platform: str) -> bool:
    """Return the adapter's narrow discriminator for one missing edit target.

    The watcher deliberately does not inspect provider error text: only the
    Telegram adapter may classify its own response as a vanished message.
    """
    return platform == "telegram" and getattr(result, "error_kind", None) == "edit_message_not_found"


async def _edit_kanban_status_message(
    adapter: Any,
    chat_id: str,
    message_id: str,
    content: str,
    *,
    finalize: bool,
    metadata: dict[str, Any],
) -> Any:
    """Edit a card while preserving route metadata for capable adapters.

    Status cards predate ``metadata`` on ``edit_message``. Keep older adapters
    callable, but pass the exact Telegram origin lane to current adapters so a
    create→edit sequence cannot lose its group-topic route.
    """
    kwargs: dict[str, Any] = {"finalize": finalize}
    try:
        parameters = inspect.signature(adapter.edit_message).parameters.values()
        accepts_metadata = any(
            parameter.name == "metadata" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_metadata = False
    if accepts_metadata:
        kwargs["metadata"] = metadata
    return await adapter.edit_message(chat_id, message_id, content, **kwargs)


def _resolve_auto_decompose_settings(
    load_config: Callable[[], Any],
) -> "tuple[bool, int]":
    """Resolve the live (enabled, per_tick) auto-decompose settings.

    Read fresh from config on every dispatcher tick (#49638) so that flipping
    ``kanban.auto_decompose: false`` to STOP runaway fan-out takes effect on the
    next tick instead of requiring a gateway restart. Auto-decompose is a
    safety toggle — a user who sees it create and launch tasks they didn't
    intend reaches for this flag to halt it, and a stale boot-captured value
    silently ignoring that change is the bug reported in #49638.

    Fails **safe**: if the config read raises, return ``(False, 3)`` — a
    transient read error must never re-enable a feature the user turned off,
    nor fall back to the burst-prone default-on behaviour. ``per_tick`` is
    clamped to ``>= 1``.
    """
    try:
        cfg = load_config()
    except Exception:
        return False, 3
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    enabled = bool(kcfg.get("auto_decompose", True))
    try:
        per_tick = int(kcfg.get("auto_decompose_per_tick", 3) or 3)
    except (TypeError, ValueError):
        per_tick = 3
    if per_tick < 1:
        per_tick = 1
    return enabled, per_tick


def _acquire_singleton_lock(lock_path) -> "tuple[Optional[object], str]":
    """Take an exclusive, non-blocking advisory lock for the sole dispatcher.

    Only one gateway process machine-wide may run the embedded kanban
    dispatcher: concurrent dispatchers double the reclaim frequency (each
    runs its own ``release_stale_claims`` → promote → dispatch loop), double
    claim-attempt events in the event log, and — with ``wal_autocheckpoint=0`` —
    concurrent manual WAL checkpoints can corrupt index pages. The
    ``dispatch_in_gateway`` config flag is the primary control; this lock is the
    backstop that survives config drift and same-profile restart races.

    Delegates to :func:`gateway.status._try_acquire_file_lock` (``fcntl`` on
    POSIX, ``msvcrt`` on Windows) so the guard is cross-platform.

    Returns ``(handle, "held")`` on success — the caller keeps the file handle
    for the process lifetime and **must** release it via
    :func:`_release_singleton_lock` when done. ``(None, "contended")`` when
    another process holds the lock (caller must NOT dispatch). ``(None,
    "unavailable")`` when locking cannot be performed (non-POSIX filesystem
    without flock, or the status.py helpers are unimportable) — caller falls
    back to config-only control.
    """
    try:
        from gateway.status import _try_acquire_file_lock  # deferred; same package
    except ImportError:
        return None, "unavailable"
    try:
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(str(lock_path), "a+", encoding="utf-8")
    except OSError:
        return None, "unavailable"
    if not _try_acquire_file_lock(handle):
        handle.close()
        return None, "contended"
    return handle, "held"


def _release_singleton_lock(handle) -> None:
    """Release a dispatcher singleton lock acquired via :func:`_acquire_singleton_lock`."""
    if handle is None:
        return
    try:
        from gateway.status import _release_file_lock
        _release_file_lock(handle)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    async def _kanban_notifier_watcher(self, interval: float = 15.0) -> None:
        """Poll ``kanban_notify_subs`` and maintain exact-origin status cards.

        For each exact-origin subscription, fetches actionable lifecycle events
        newer than its stored cursor. The first event creates one card in
        ``(platform, chat_id, thread_id)``; later events edit that same card.
        The cursor advances only after a verified receipt. Subscriptions remain
        as durable lane records after terminal status for recovery.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Subscriptions live inside each board's own DB and cannot
        cross boards, so delivery semantics are unchanged — this is
        purely a fan-out of the single-DB poll.
        """
        # Worker dispatch and remote status delivery are deliberately separate.
        # The durable surface lease elects one eligible gateway; a profile that
        # is not configured to dispatch workers must not strand its card.
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban notifier: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban notifier: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban notifier: cannot load config (%s); disabled", exc)
            return
        try:
            interval = max(1.0, float(cfg.get("kanban", {}).get("notification_interval_seconds", interval)))
        except (TypeError, ValueError):
            logger.warning("kanban notifier: invalid notification_interval_seconds=%r; using %.1fs", cfg.get("kanban", {}).get("notification_interval_seconds"), interval)
        active_task_index_enabled = bool(
            cfg.get("kanban", {}).get("active_task_index_enabled", True)
        )

        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        # "status" covers dashboard drag-drop and `_set_status_direct()`
        # writes — surface those transitions to subscribers too.
        CARD_EVENT_KINDS = (
            "created", "claimed", "spawned", "heartbeat", "progress", "status", "blocked",
            "unblocked", "completed", "review_requested", "review_rejected", "review_accepted",
            "review_retry_scheduled", "review_recovered", "review_job_reconciled",
            "auditor_review_claimed", "auditor_review_spawned", "needs_auditor",
            "crashed", "timed_out", "gave_up", "archived", "dependency_wait", "reclaimed",
            "superseded", "replaced",
        )
        # The card is a projection of the task's current state, not an event
        # log. A task can accumulate created -> claimed -> completed before one
        # notifier tick; rendering every event would issue identical Telegram
        # edits back-to-back and needlessly consume the chat flood budget.
        CARD_RENDER_KINDS = frozenset({
            "created", "claimed", "spawned", "heartbeat", "completed",
            "review_requested", "review_rejected", "review_accepted",
            "review_retry_scheduled", "review_recovered", "review_job_reconciled",
            "auditor_review_claimed", "auditor_review_spawned", "needs_auditor", "blocked", "gave_up",
            "crashed", "timed_out", "status", "unblocked", "dependency_wait",
            "reclaimed", "archived",
        })
        # The subscription is the immutable origin receipt and is retained even
        # after completion/archival. Cursor claims deduplicate retries while the
        # status-surface lease serializes remote edits across gateways.
        # Retry state is durable: a gateway restart cannot reset an exact-origin
        # failure loop into a five-second retry storm.
        notifier_profile = getattr(self, "_kanban_notifier_profile", None)
        if not notifier_profile:
            notifier_profile = self._active_profile_name()
            self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        logged_flood_cooldown_until = 0
        while self._running:
            try:
                def _active_flood_cooldown() -> int:
                    """Read all board cooldowns before claiming any event cursor."""
                    cooldown_until = 0
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: set[str] = set()
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            continue
                        seen_db_paths.add(resolved_db_path)
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                            continue
                        try:
                            cooldown_until = max(
                                cooldown_until,
                                _kb.notification_flood_cooldown_until(conn),
                            )
                        finally:
                            conn.close()
                    return cooldown_until

                flood_cooldown_until = await asyncio.to_thread(_active_flood_cooldown)
                if flood_cooldown_until > int(time.time()):
                    if flood_cooldown_until != logged_flood_cooldown_until:
                        logger.warning(
                            "kanban notifier: Telegram flood-control cooldown active until %s; skipping remote delivery",
                            flood_cooldown_until,
                        )
                        logged_flood_cooldown_until = flood_cooldown_until
                    await asyncio.sleep(min(max(1.0, interval), flood_cooldown_until - int(time.time())))
                    continue
                logged_flood_cooldown_until = 0

                def _collect():
                    deliveries: list[dict] = []
                    refreshes: list[dict] = []
                    recoverable_refreshes: list[dict] = []
                    terminal_notifications: list[dict] = []
                    index_refreshes: list[dict] = []
                    legacy_index_receipts: list[dict] = []
                    flood_cooldown_until = 0

                    # Enumerate every board on disk, but poll each resolved DB
                    # path once. Multiple slugs can point at the same DB when
                    # HERMES_KANBAN_DB pins the board path; without this guard
                    # one gateway could collect the same subscription/event
                    # more than once before advancing the cursor.
                    try:
                        boards = _kb.list_boards(include_archived=False)
                    except Exception:
                        boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
                    seen_db_paths: set[str] = set()
                    for board_meta in boards:
                        slug = board_meta.get("slug") or _kb.DEFAULT_BOARD
                        db_path = board_meta.get("db_path")
                        try:
                            resolved_db_path = str(Path(db_path).expanduser().resolve()) if db_path else str(_kb.kanban_db_path(slug).resolve())
                        except Exception:
                            resolved_db_path = f"slug:{slug}"
                        if resolved_db_path in seen_db_paths:
                            logger.debug(
                                "kanban notifier: skipping duplicate board slug %s for DB %s",
                                slug, resolved_db_path,
                            )
                            continue
                        seen_db_paths.add(resolved_db_path)
                        try:
                            conn = _kb.connect(board=slug)
                        except Exception as exc:
                            logger.debug("kanban notifier: cannot open board %s: %s", slug, exc)
                            continue
                        try:
                            # `connect()` runs the schema + idempotent migration
                            # on first open per process, so an explicit
                            # `init_db()` here would be redundant. Worse:
                            # `init_db()` deliberately busts the per-process
                            # cache and re-runs the migration on a *second*
                            # connection, which races the first and used to
                            # log a benign but noisy `duplicate column name`
                            # traceback (and intermittent "database is locked"
                            # — issue #21378) on every gateway start against
                            # a legacy DB. `_add_column_if_missing` now
                            # tolerates that race, but we still skip the
                            # redundant call to avoid the wasted work.
                            subs = _kb.list_notify_subs(conn)
                            flood_cooldown_until = max(
                                flood_cooldown_until,
                                _kb.notification_flood_cooldown_until(conn),
                            )
                            if not subs:
                                logger.debug("kanban notifier: board %s has no subscriptions", slug)
                            for sub in subs:
                                old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                    conn,
                                    task_id=sub["task_id"],
                                    platform=sub["platform"],
                                    chat_id=sub["chat_id"],
                                    thread_id=sub.get("thread_id") or "",
                                    kinds=CARD_EVENT_KINDS,
                                )
                                if not events:
                                    continue
                                # `created` is a card bootstrap for a newly
                                # subscribed waiting task. If this same replay
                                # already contains a later lifecycle event,
                                # preserve the established behavior: create or
                                # edit from that later state instead of sending
                                # an immediately stale extra render first.
                                render_events = (
                                    [event for event in events if event.kind != "created"]
                                    if len(events) > 1 else events
                                )
                                if not render_events:
                                    continue
                                task = _kb.get_task(conn, sub["task_id"])
                                timeline = _kb.list_events(conn, sub["task_id"])
                                comments = _kb.list_comments(conn, sub["task_id"])
                                parents = [
                                    parent for parent_id in _kb.parent_ids(conn, sub["task_id"])
                                    if (parent := _kb.get_task(conn, parent_id)) is not None
                                ]
                                logger.debug(
                                    "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                    len(events), sub["task_id"], slug, old_cursor, cursor,
                                )
                                deliveries.append({
                                    "sub": sub,
                                    "old_cursor": old_cursor,
                                    "cursor": cursor,
                                    "events": render_events,
                                    "task": task,
                                    "board": slug,
                                    "timeline": timeline,
                                    "current_run": _kb.current_run_progress(conn, sub["task_id"]),
                                    "latest_comment": comments[-1] if comments else None,
                                    "parents": parents,
                                })
                            terminal_notifications.extend(
                                {"notification": notification, "board": slug}
                                for notification in _kb.list_pending_terminal_notifications(conn)
                            )
                            for sub in _kb.list_due_status_surface_refreshes(
                                conn, renderer_version=_KANBAN_STATUS_RENDERER_VERSION,
                            ):
                                task = _kb.get_task(conn, sub["task_id"])
                                if task is None:
                                    continue
                                timeline = _kb.list_events(conn, sub["task_id"])
                                comments = _kb.list_comments(conn, sub["task_id"])
                                parents = [
                                    parent for parent_id in _kb.parent_ids(conn, sub["task_id"])
                                    if (parent := _kb.get_task(conn, parent_id)) is not None
                                ]
                                refreshes.append({
                                    "sub": sub, "task": task, "board": slug,
                                    "timeline": timeline,
                                    "current_run": _kb.current_run_progress(conn, sub["task_id"]),
                                    "latest_comment": comments[-1] if comments else None,
                                    "parents": parents,
                                })
                            for sub in _kb.list_recoverable_edit_missing_status_surfaces(conn):
                                task = _kb.get_task(conn, sub["task_id"])
                                if task is None:
                                    continue
                                timeline = _kb.list_events(conn, sub["task_id"])
                                comments = _kb.list_comments(conn, sub["task_id"])
                                parents = [
                                    parent for parent_id in _kb.parent_ids(conn, sub["task_id"])
                                    if (parent := _kb.get_task(conn, parent_id)) is not None
                                ]
                                recoverable_refreshes.append({
                                    "sub": sub, "task": task, "board": slug,
                                    "timeline": timeline,
                                    "current_run": _kb.current_run_progress(conn, sub["task_id"]),
                                    "latest_comment": comments[-1] if comments else None,
                                    "parents": parents,
                                })
                            if active_task_index_enabled:
                                index_refreshes.extend(
                                    {"lane": lane}
                                    for lane in _kb.list_uninitialized_active_task_index_lanes(conn)
                                )
                                legacy_index_receipts.extend(
                                    _kb.list_legacy_active_task_index_receipts(conn)
                                )
                        finally:
                            conn.close()
                    return (
                        deliveries, refreshes, recoverable_refreshes, terminal_notifications,
                        index_refreshes, legacy_index_receipts, flood_cooldown_until,
                    )

                (
                    deliveries, refreshes, recoverable_refreshes, terminal_notifications,
                    index_refreshes, legacy_index_receipts, flood_cooldown_until,
                ) = await asyncio.to_thread(_collect)
                if flood_cooldown_until > int(time.time()):
                    if flood_cooldown_until != logged_flood_cooldown_until:
                        logger.warning(
                            "kanban notifier: Telegram flood-control cooldown active until %s; skipping remote delivery",
                            flood_cooldown_until,
                        )
                        logged_flood_cooldown_until = flood_cooldown_until
                    await asyncio.sleep(min(max(1.0, interval), flood_cooldown_until - int(time.time())))
                    continue
                if active_task_index_enabled:
                    registry = _kb.connect_active_task_index_registry()
                    try:
                        adopted_receipts = _kb.adopt_active_task_index_receipts(registry, legacy_index_receipts)
                        for adoption in adopted_receipts:
                            if adoption["conflicted"]:
                                logger.warning(
                                    "kanban active index: conflicting legacy receipts for %s/%s; "
                                    "editing deterministic existing message %s without replacement send",
                                    adoption["lane"][0], adoption["lane"][1], adoption["message_id"],
                                )
                        index_refreshes.extend(
                            {"lane": lane}
                            for lane in _kb.list_due_active_task_indexes(
                                registry, renderer_version=_KANBAN_ACTIVE_INDEX_RENDERER_VERSION,
                            )
                        )
                    finally:
                        registry.close()
                index_lanes: dict[tuple[str, str, str, str], dict] = {
                    (
                        str(item["lane"]["platform"]), str(item["lane"]["chat_id"]),
                        str(item["lane"].get("thread_id") or ""),
                        str(item["lane"].get("notifier_profile") or ""),
                    ): item["lane"]
                    for item in index_refreshes
                }
                for d in deliveries:
                    sub = d["sub"]
                    if active_task_index_enabled:
                        index_lanes[(
                            str(sub["platform"]), str(sub["chat_id"]), str(sub.get("thread_id") or ""),
                            str(sub.get("notifier_profile") or ""),
                        )] = sub
                    task = d["task"]
                    board_slug = d.get("board")
                    platform_str = (sub["platform"] or "").lower()
                    try:
                        plat = _Platform(platform_str)
                    except ValueError:
                        # Unknown platform string; skip and advance cursor so
                        # we don't replay forever.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        continue
                    sub_profile = sub.get("notifier_profile") or ""
                    # Route via the SAME chokepoint the authorization path uses
                    # (gateway/authz_mixin.py::_authorization_adapter): a stamped
                    # profile with its own adapter-registry entry must be served
                    # by THAT profile's same-platform adapter and must NOT silently
                    # fall back to the default profile's adapter — otherwise a
                    # secondary profile's task notification is delivered by the
                    # wrong bot (the cross-profile mis-delivery this whole change
                    # exists to fix). The helper returns None only when the profile
                    # (or default) genuinely has no adapter for the platform.
                    adapter = self._authorization_adapter(plat, sub_profile or None)
                    if adapter is None:
                        logger.debug(
                            "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                            platform_str, sub["task_id"],
                        )
                        await asyncio.to_thread(
                            self._kanban_rewind,
                            sub,
                            d["cursor"],
                            d.get("old_cursor", 0),
                            board_slug,
                        )
                        continue
                    title = user_facing_title(task, d.get("timeline") or d["events"], sub["task_id"])
                    board_tag = f"[{board_slug}] " if board_slug else ""
                    # Keep one newest renderable event. The full claimed cursor
                    # is advanced below, so coalescing does not lose lifecycle
                    # history or redeliver it on the next tick.
                    render_events = []
                    for candidate in reversed(d["events"]):
                        if candidate.kind not in CARD_RENDER_KINDS:
                            continue
                        if (
                            candidate.kind == "heartbeat"
                            and not (candidate.payload or {}).get("note")
                            and self._kanban_status_message_exists(sub, board_slug)
                        ):
                            continue
                        render_events = [candidate]
                        break
                    terminal_event = next(
                        (event for event in reversed(d["events"])
                         if event.kind == "review_accepted"),
                        None,
                    )
                    for ev in render_events:
                        kind = ev.kind
                        # Identity prefix: attribute terminal pings to the
                        # worker that did the work. Makes fleets (where one
                        # chat subscribes to many tasks) legible at a glance.
                        who = (task.assignee if task and task.assignee else None)
                        tag = f"@{who} " if who else ""
                        if kind == "created":
                            msg = f"⏳ {board_tag}{tag}Kanban {sub['task_id']} created — {title}"
                        elif kind == "claimed":
                            msg = f"▶ {board_tag}{tag}Kanban {sub['task_id']} started — {title}"
                        elif kind == "spawned":
                            pid = (ev.payload or {}).get("pid")
                            suffix = f" (pid {pid})" if pid else ""
                            msg = f"⚙ {board_tag}{tag}Kanban {sub['task_id']} worker running{suffix}"
                        elif kind == "heartbeat":
                            note = str((ev.payload or {}).get("note") or "working")[:200]
                            msg = f"⏳ {board_tag}{tag}Kanban {sub['task_id']} progress — {note}"
                        elif kind == "completed":
                            # Prefer the run's summary (the worker's
                            # intentional human-facing handoff, carried
                            # in the event payload), then fall back to
                            # task.result for legacy rows written before
                            # runs shipped.
                            handoff = ""
                            payload_summary = None
                            if ev.payload and ev.payload.get("summary"):
                                payload_summary = str(ev.payload["summary"])
                            if payload_summary:
                                lines = payload_summary.strip().splitlines()
                                h = lines[0][:200] if lines else payload_summary[:200]
                                handoff = f"\n{h}"
                            elif task and task.result:
                                lines = task.result.strip().splitlines()
                                r = lines[0][:160] if lines else task.result[:160]
                                handoff = f"\n{r}"
                            msg = (
                                f"✔ {board_tag}{tag}Kanban {sub['task_id']} done"
                                f" — {title}{handoff}"
                            )
                        elif kind == "review_requested":
                            msg = f"🔎 {board_tag}{tag}Kanban {sub['task_id']} handed to auditor — {title}"
                        elif kind == "review_rejected":
                            msg = f"↩️ {board_tag}{tag}Kanban {sub['task_id']} returned for rework — {title}"
                        elif kind == "review_accepted":
                            msg = f"✅ {board_tag}{tag}Kanban {sub['task_id']} accepted after review — {title}"
                        elif kind == "review_retry_scheduled":
                            msg = f"🔁 {board_tag}{tag}Kanban {sub['task_id']} auditor retry scheduled — {title}"
                        elif kind == "review_recovered":
                            msg = f"🔎 {board_tag}{tag}Kanban {sub['task_id']} audit handoff recovered — {title}"
                        elif kind == "review_job_reconciled":
                            msg = f"🔁 {board_tag}{tag}Kanban {sub['task_id']} audit handoff reconciled — {title}"
                        elif kind == "auditor_review_claimed":
                            msg = f"🔎 {board_tag}{tag}Kanban {sub['task_id']} claimed by auditor — {title}"
                        elif kind == "auditor_review_spawned":
                            msg = f"🔎 {board_tag}{tag}Kanban {sub['task_id']} auditor running — {title}"
                        elif kind == "needs_auditor":
                            reason = str((ev.payload or {}).get("reason") or "automatic auditor unavailable")[:160]
                            msg = f"🔐 {board_tag}{tag}Kanban {sub['task_id']} needs auditor — {reason}"
                        elif kind == "blocked":
                            reason = ""
                            if ev.payload and ev.payload.get("reason"):
                                reason = f": {str(ev.payload['reason'])[:160]}"
                            msg = f"⏸ {board_tag}{tag}Kanban {sub['task_id']} blocked{reason}"
                        elif kind == "gave_up":
                            err = ""
                            if ev.payload and ev.payload.get("error"):
                                err = f"\n{str(ev.payload['error'])[:200]}"
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} gave up "
                                f"after repeated spawn failures{err}"
                            )
                        elif kind == "crashed":
                            msg = (
                                f"✖ {board_tag}{tag}Kanban {sub['task_id']} worker crashed "
                                f"(pid gone); dispatcher will retry"
                            )
                        elif kind == "timed_out":
                            limit = 0
                            if ev.payload and ev.payload.get("limit_seconds"):
                                try:
                                    limit = int(ev.payload["limit_seconds"])
                                except (TypeError, ValueError):
                                    logger.warning(
                                        "kanban notifier: invalid timed_out limit_seconds for %s: %r",
                                        sub["task_id"], ev.payload.get("limit_seconds"),
                                    )
                            msg = (
                                f"⏱ {board_tag}{tag}Kanban {sub['task_id']} timed out "
                                f"(max_runtime={limit}s); will retry"
                            )
                        elif kind == "status":
                            new_status = ""
                            if ev.payload and ev.payload.get("status"):
                                new_status = str(ev.payload["status"])
                            msg = f"🔄 {board_tag}{tag}Kanban {sub['task_id']} → {new_status}"
                        elif kind == "unblocked":
                            msg = f"▶ {board_tag}{tag}Kanban {sub['task_id']} resumed"
                        elif kind == "dependency_wait":
                            reason = str((ev.payload or {}).get("reason") or "waiting for dependency")[:160]
                            msg = f"⏸ {board_tag}{tag}Kanban {sub['task_id']} waiting — {reason}"
                        elif kind == "reclaimed":
                            msg = f"↻ {board_tag}{tag}Kanban {sub['task_id']} recovered; retrying"
                        elif kind == "archived":
                            msg = f"▣ {board_tag}{tag}Kanban {sub['task_id']} archived"
                        else:
                            # Unknown events are intentionally not rendered but
                            # are advanced by the enclosing cursor claim.
                            continue
                        msg = render_kanban_status_card(
                            sub=sub, task=task, timeline=d.get("timeline") or d["events"],
                            latest_comment=d.get("latest_comment"), parents=d.get("parents"),
                            current_run=d.get("current_run"),
                        )
                        metadata = _kanban_status_route_metadata(
                            adapter, sub, getattr(task, "status", ""),
                        )
                        # Cursor claiming serializes event selection; this separate,
                        # durable lease serializes the remote status-card write and
                        # survives restarts. Include PID so two processes for the
                        # same profile are still competing owners, not one owner.
                        delivery_owner = f"{notifier_profile}:{os.getpid()}"
                        surface = await asyncio.to_thread(
                            self._kanban_claim_status_surface,
                            sub, delivery_owner, int(ev.id), board_slug,
                            # A reply anchor only makes a *first* send valid in
                            # a DM topic; it is not an operator recovery action.
                            # A parked card stays parked until a new explicit
                            # subscription is created.
                            recover_parked=False,
                        )
                        if surface is None:
                            # A different gateway owns the live card write. Do
                            # not consume its event via our pre-send cursor;
                            # return it to the durable stream for recovery.
                            await asyncio.to_thread(
                                self._kanban_rewind,
                                sub, d["cursor"], d.get("old_cursor", 0), board_slug,
                            )
                            break
                        try:
                            if surface.get("message_id"):
                                send_result = await _edit_kanban_status_message(
                                    adapter,
                                    sub["chat_id"], surface["message_id"], msg,
                                    finalize=kind == "review_accepted", metadata=metadata,
                                )
                                if _is_deleted_telegram_edit(send_result, platform_str):
                                    send_result = await adapter.send(
                                        sub["chat_id"], msg, metadata=metadata,
                                    )
                            else:
                                send_result = await adapter.send(
                                    sub["chat_id"], msg, metadata=metadata,
                                )
                            from gateway.platforms.base import SendResult
                            if not isinstance(send_result, SendResult) or not send_result.success:
                                # Keep the verified remote identity even when
                                # Telegram says an edit target disappeared.
                                # Retrying an edit is bounded and can park the
                                # lane; clearing it would turn the next retry
                                # into a fresh send every notifier tick.
                                error = getattr(send_result, "error", None) if send_result is not None else None
                                raise RuntimeError(error or "kanban notifier requires canonical SendResult(success=True)")
                            receipt_id = send_result.message_id
                            if not receipt_id:
                                raise RuntimeError("kanban notifier missing canonical status-card receipt message_id")
                            if not await asyncio.to_thread(
                                self._kanban_record_status_delivery,
                                sub, delivery_owner, int(surface["lease_generation"]),
                                int(ev.id), receipt_id, board_slug,
                                _KANBAN_STATUS_RENDERER_VERSION,
                                hashlib.sha256(msg.encode("utf-8")).hexdigest(),
                            ):
                                raise RuntimeError("kanban notifier lost status-card lease before receipt")
                            if terminal_event is not None:
                                notification = await asyncio.to_thread(
                                    self._kanban_enqueue_terminal_notification,
                                    sub, int(terminal_event.id), title,
                                    getattr(terminal_event, "payload", None), board_slug,
                                )
                                if notification is not None:
                                    flood_cooldown_until = max(
                                        flood_cooldown_until,
                                        await self._deliver_terminal_notification(
                                            adapter, notification, board_slug, delivery_owner,
                                        ),
                                    )
                                    if flood_cooldown_until > int(time.time()):
                                        break
                            logger.debug(
                                "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                                kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                            )
                            # Handoff artifacts are internal by default. Upload
                            # only when the originating user explicitly asked
                            # for a file and that intent was persisted on the
                            # review event.
                            if (
                                kind == "review_accepted"
                                and isinstance(getattr(ev, "payload", None), dict)
                                and getattr(ev, "payload", {}).get("deliver_artifacts") is True
                            ):
                                try:
                                    await self._deliver_kanban_artifacts(
                                        adapter=adapter,
                                        chat_id=sub["chat_id"],
                                        metadata=metadata,
                                        event_payload=getattr(ev, "payload", None),
                                        task=task,
                                    )
                                except Exception as art_exc:
                                    logger.debug(
                                        "kanban notifier: artifact delivery for %s failed: %s",
                                        sub["task_id"], art_exc,
                                    )
                        except Exception as exc:
                            failure_state = await asyncio.to_thread(
                                self._kanban_record_status_failure,
                                sub, delivery_owner, int(surface["lease_generation"]),
                                str(exc), board_slug,
                            )
                            attempts = int((failure_state or {}).get("attempts") or 0)
                            parked = bool((failure_state or {}).get("dead_lettered_at"))
                            if attempts == 1:
                                logger.warning(
                                    "kanban notifier: send failed for %s on %s; retrying exact origin after durable backoff: %s",
                                    sub["task_id"], platform_str, exc,
                                )
                            elif parked:
                                logger.warning(
                                    "kanban notifier: parked exact-origin status card for %s after %d failures",
                                    sub["task_id"], attempts,
                                )
                            await asyncio.to_thread(
                                self._kanban_rewind,
                                sub, d["cursor"], d.get("old_cursor", 0), board_slug,
                            )
                            # Telegram rate limits are bot-wide.  Persisting the
                            # exact-lane retry above gives restart-safe recovery;
                            # stopping this tick keeps unrelated cards from
                            # hammering the same limited bot.
                            flood_cooldown_until = max(
                                flood_cooldown_until,
                                _flood_cooldown_from_failure(exc, failure_state),
                            )
                            break
                    else:
                        # All events delivered; advance cursor. The cursor
                        # is the dedup mechanism — it prevents re-delivery
                        # of the same event on subsequent ticks.
                        await asyncio.to_thread(
                            self._kanban_advance, sub, d["cursor"], board_slug,
                        )
                        # The exact-origin subscription and surface survive
                        # terminal states so recovery never has to infer a lane.
                        task_terminal = False
                        # The durable notifier has already updated the status
                        # surface and, for accepted outcomes, delivered the
                        # grouped summary. Waking an agent for routine
                        # completion or a retryable crash/timed-out run creates
                        # a second conversational producer for the same origin.
                        # Only outcomes that require human attention may wake a
                        # session: a real blocker or an exhausted retry budget.
                        _WAKE_KINDS = ("gave_up", "blocked")
                        _wake_kinds = {ev.kind for ev in d["events"] if ev.kind in _WAKE_KINDS}
                        if _wake_kinds:
                            try:
                                _session_key = getattr(task, "session_id", None) or ""
                                if _session_key:
                                    _title = (task.title if task else sub["task_id"])[:120]
                                    _assignee = task.assignee if task else ""
                                    _parts = []
                                    if "gave_up" in _wake_kinds: _parts.append(t("gateway.kanban.wake.gave_up"))
                                    if "blocked" in _wake_kinds: _parts.append(t("gateway.kanban.wake.blocked"))
                                    _status = t("gateway.kanban.wake.status_joiner").join(_parts) or t("gateway.kanban.wake.status_default")
                                    _synth = t(
                                        "gateway.kanban.wake.message",
                                        task_id=sub["task_id"],
                                        status=_status,
                                        title=_title,
                                        assignee=_assignee,
                                        board=board_slug,
                                    )
                                    from gateway.session import SessionSource
                                    from gateway.platforms.base import MessageEvent, MessageType
                                    # KNOWN LIMITATION (tracked follow-up): the
                                    # subscription row does not persist the
                                    # creator's chat_type, and it is not carried
                                    # on the session-context bridge, so we cannot
                                    # faithfully reconstruct the creator's real
                                    # session key here. build_session_key() keys
                                    # DMs (":dm:<chat_id>") on a wholly different
                                    # shape from group/thread, so any hardcoded
                                    # value mis-routes some creators. "group" is
                                    # the least-surprising default for the
                                    # dashboard/group flows this wake primarily
                                    # serves; DM-originated creators are handled
                                    # by the follow-up that stamps + persists
                                    # chat_type end-to-end. handle_message()
                                    # get_or_create_session's the target, so a
                                    # mismatch degrades to "wake lands in a fresh
                                    # group session" — never an exception.
                                    _source = SessionSource(
                                        platform=plat,
                                        chat_id=sub["chat_id"],
                                        chat_type="group",
                                        thread_id=sub.get("thread_id") or None,
                                        user_id=sub.get("user_id"),
                                        profile=sub_profile or None,
                                    )
                                    _synth_event = MessageEvent(
                                        text=_synth,
                                        message_type=MessageType.TEXT,
                                        source=_source,
                                        internal=True,
                                    )
                                    await adapter.handle_message(_synth_event)
                                    logger.info(
                                        "kanban notifier: woke agent for %s on %s/%s profile=%s events=%s",
                                        sub["task_id"], platform_str, sub["chat_id"], sub_profile or "default", _wake_kinds,
                                    )
                            except Exception as _wk_err:
                                # Best-effort: the notification itself already
                                # delivered and the cursor has advanced, so a
                                # broken wake path must not wedge the tick — but
                                # log at WARNING with a traceback rather than
                                # DEBUG so a persistently-failing wake is visible
                                # in normal logs instead of silently no-op'ing.
                                logger.warning(
                                    "kanban notifier: wakeup injection failed for %s: %s",
                                    sub["task_id"], _wk_err, exc_info=True,
                                )
                        if task_terminal:
                            await asyncio.to_thread(
                                self._kanban_unsub, sub, board_slug,
                            )
                if flood_cooldown_until > int(time.time()):
                    logger.warning(
                        "kanban notifier: Telegram flood-control received; stopping this tick until %s",
                        flood_cooldown_until,
                    )
                    await asyncio.sleep(min(max(1.0, interval), flood_cooldown_until - int(time.time())))
                    continue
                for sub in index_lanes.values():
                    flood_cooldown_until = max(
                        flood_cooldown_until,
                        await self._refresh_kanban_active_task_index(
                            sub, notifier_profile=notifier_profile,
                        ),
                    )
                    if flood_cooldown_until > int(time.time()):
                        break
                if flood_cooldown_until > int(time.time()):
                    logger.warning(
                        "kanban notifier: Telegram flood-control from active index; stopping this tick until %s",
                        flood_cooldown_until,
                    )
                    await asyncio.sleep(min(max(1.0, interval), flood_cooldown_until - int(time.time())))
                    continue
                # An event render already produces the newest card snapshot in
                # this tick. Do not race it with a separately collected pulse
                # refresh for the same exact lane: one remote edit per tick is
                # enough, including when the event edit fails and is retried.
                event_lanes = {
                    (
                        item["sub"]["task_id"], item["sub"]["platform"],
                        item["sub"]["chat_id"], item["sub"].get("thread_id") or "",
                    )
                    for item in deliveries
                }
                for refresh in refreshes:
                    refresh_sub = refresh["sub"]
                    refresh_lane = (
                        refresh_sub["task_id"], refresh_sub["platform"],
                        refresh_sub["chat_id"], refresh_sub.get("thread_id") or "",
                    )
                    if refresh_lane in event_lanes:
                        continue
                    flood_cooldown_until = max(
                        flood_cooldown_until,
                        await self._refresh_kanban_status_surface(
                            refresh, notifier_profile=notifier_profile,
                        ),
                    )
                    if flood_cooldown_until > int(time.time()):
                        break
                for refresh in recoverable_refreshes:
                    refresh_sub = refresh["sub"]
                    refresh_lane = (
                        refresh_sub["task_id"], refresh_sub["platform"],
                        refresh_sub["chat_id"], refresh_sub.get("thread_id") or "",
                    )
                    if refresh_lane in event_lanes:
                        continue
                    flood_cooldown_until = max(
                        flood_cooldown_until,
                        await self._refresh_kanban_status_surface(
                            refresh, notifier_profile=notifier_profile, recover_parked=True,
                        ),
                    )
                    if flood_cooldown_until > int(time.time()):
                        break
                for terminal in terminal_notifications:
                    notification = terminal["notification"]
                    try:
                        plat = _Platform((notification["platform"] or "").lower())
                    except ValueError:
                        continue
                    adapter = self._authorization_adapter(
                        plat, notification.get("notifier_profile") or None,
                    )
                    if adapter is not None:
                        flood_cooldown_until = max(
                            flood_cooldown_until,
                            await self._deliver_terminal_notification(
                                adapter,
                                notification,
                                terminal.get("board"),
                                f"{notifier_profile}:{os.getpid()}",
                            ),
                        )
                        if flood_cooldown_until > int(time.time()):
                            break
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            # Sleep with cancellation checks.
            for _ in range(int(max(1, interval))):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def _refresh_kanban_status_surface(
        self, refresh: dict, *, notifier_profile: str, recover_parked: bool = False,
    ) -> int:
        """Refresh one already-sent card without consuming lifecycle events."""
        from gateway.config import Platform as _Platform
        from gateway.platforms.base import SendResult

        sub = refresh["sub"]
        board = refresh.get("board")
        try:
            platform = _Platform((sub["platform"] or "").lower())
        except ValueError:
            return 0
        adapter = self._authorization_adapter(
            platform, sub.get("notifier_profile") or None,
        )
        if adapter is None:
            return 0
        now = int(time.time())
        text = render_kanban_status_card(
            sub=sub, task=refresh["task"], timeline=refresh["timeline"],
            latest_comment=refresh.get("latest_comment"), parents=refresh.get("parents"), now=now,
            current_run=refresh.get("current_run"),
        )
        render_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        metadata = _kanban_status_route_metadata(
            adapter, sub, getattr(refresh["task"], "status", ""),
        )
        owner = f"{notifier_profile}:{os.getpid()}"
        surface = await asyncio.to_thread(
            self._kanban_claim_status_surface, sub, owner, 0, board, recover_parked,
        )
        if surface is None:
            return 0
        try:
            receipt_id = surface["message_id"]
            # The durable hash avoids a redundant remote edit when a 60-second
            # tick happens to render byte-for-byte identical content. A version
            # migration intentionally edits once even when the visible text did
            # not change, then persists the new version for restart idempotency.
            # A recovery claim must always attempt delivery: the stored hash
            # can match while the target message is the one Telegram lost.
            version_changed = surface.get("renderer_version") != _KANBAN_STATUS_RENDERER_VERSION
            if recover_parked or version_changed or surface.get("render_hash") != render_hash:
                result = await _edit_kanban_status_message(
                    adapter,
                    sub["chat_id"], surface["message_id"], text,
                    finalize=False, metadata=metadata,
                )
                if _is_deleted_telegram_edit(result, (sub.get("platform") or "").lower()):
                    result = await adapter.send(sub["chat_id"], text, metadata=metadata)
                if not isinstance(result, SendResult) or not result.success:
                    error = getattr(result, "error", None) if result is not None else None
                    raise RuntimeError(error or "kanban status refresh requires SendResult(success=True)")
                if not result.message_id:
                    raise RuntimeError("kanban status refresh missing canonical receipt message_id")
                receipt_id = result.message_id
            if not await asyncio.to_thread(
                self._kanban_record_status_delivery,
                sub, owner, int(surface["lease_generation"]), 0,
                receipt_id, board,
                _KANBAN_STATUS_RENDERER_VERSION, render_hash,
            ):
                raise RuntimeError("kanban status refresh lost lease before receipt")
        except Exception as exc:
            failure_state = await asyncio.to_thread(
                self._kanban_record_status_failure,
                sub, owner, int(surface["lease_generation"]), str(exc), board,
            )
            logger.warning("kanban notifier: status refresh failed for %s: %s", sub["task_id"], exc)
            return _flood_cooldown_from_failure(exc, failure_state)
        return 0

    async def _refresh_kanban_active_task_index(
        self, lane: dict, *, notifier_profile: str,
    ) -> int:
        """Edit one durable per-topic index; it is a projection, never routing state."""
        from gateway.config import Platform as _Platform
        from gateway.platforms.base import SendResult

        try:
            platform = _Platform((lane["platform"] or "").lower())
        except ValueError:
            return 0
        # This projection is explicitly Telegram-only. Return before adapter
        # resolution, task lookup, or durable index receipt work so a
        # subscription on another platform can only receive its canonical
        # live status card, never a second index message.
        if platform != _Platform.TELEGRAM:
            return 0
        adapter = self._authorization_adapter(platform, lane.get("notifier_profile") or None)
        if adapter is None:
            return 0
        # The empty-thread lane is chat-wide; each active forum topic receives
        # its own durable projection too.
        if not callable(getattr(adapter, "pin_message", None)):
            return 0
        items = await asyncio.to_thread(self._kanban_active_index_items, lane)
        text = render_kanban_active_task_index(items)
        metadata = _kanban_active_index_route_metadata(adapter, lane, items)
        links = {
            _active_index_link_label(item): str(item[5])
            for item in items if len(item) > 5 and item[5]
        }
        if links:
            metadata["telegram_text_links"] = links
        render_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        owner = f"{notifier_profile}:{os.getpid()}"
        surface = await asyncio.to_thread(self._kanban_claim_active_task_index, lane, owner)
        if surface is None:
            return 0
        try:
            if surface.get("message_id"):
                if (
                    surface.get("renderer_version") == _KANBAN_ACTIVE_INDEX_RENDERER_VERSION
                    and surface.get("render_hash") == render_hash
                ):
                    result = SendResult(success=True, message_id=surface["message_id"])
                else:
                    result = await _edit_kanban_status_message(
                        adapter, lane["chat_id"], surface["message_id"], text,
                        finalize=False, metadata=metadata,
                    )
                    if _is_deleted_telegram_edit(result, (lane.get("platform") or "").lower()):
                        if not await asyncio.to_thread(
                            self._kanban_forget_active_task_index_message, lane, owner,
                            int(surface["lease_generation"]),
                        ):
                            raise RuntimeError("kanban active index lost lease before recreation")
                        surface["pinned"] = False
                        surface["pin_attempted_at"] = None
                        result = await adapter.send(lane["chat_id"], text, metadata=metadata)
            else:
                result = await adapter.send(lane["chat_id"], text, metadata=metadata)
            if not isinstance(result, SendResult) or not result.success or not result.message_id:
                error = getattr(result, "error", None) if result is not None else None
                raise RuntimeError(error or "kanban active index requires SendResult receipt")

            pin_attempted = False
            pinned = bool(surface.get("pinned"))
            if (
                platform == _Platform.TELEGRAM and not pinned
                and not surface.get("pin_attempted_at")
            ):
                pin_attempted = True
                pin_message = getattr(adapter, "pin_message", None)
                try:
                    pin_result = (
                        await pin_message(lane["chat_id"], str(result.message_id), disable_notification=True)
                        if callable(pin_message) else None
                    )
                    pinned = isinstance(pin_result, SendResult) and pin_result.success
                    if not pinned:
                        logger.warning(
                            "kanban active index: Telegram pin unavailable for exact lane %s/%s; keeping one unpinned index",
                            lane["chat_id"], lane.get("thread_id") or "",
                        )
                except Exception as exc:
                    logger.warning(
                        "kanban active index: Telegram pin unavailable for exact lane %s/%s; keeping one unpinned index: %s",
                        lane["chat_id"], lane.get("thread_id") or "", exc,
                    )
            if not await asyncio.to_thread(
                self._kanban_record_active_task_index_delivery, lane, owner,
                int(surface["lease_generation"]), str(result.message_id), render_hash,
                pin_attempted, pinned,
            ):
                raise RuntimeError("kanban active index lost lease before receipt")
        except Exception as exc:
            failure_state = await asyncio.to_thread(
                self._kanban_record_active_task_index_failure, lane, owner,
                int(surface["lease_generation"]), str(exc),
            )
            logger.warning("kanban active index refresh failed for %s/%s: %s", lane["platform"], lane["chat_id"], exc)
            return _flood_cooldown_from_failure(exc, failure_state)
        return 0

    def _kanban_active_index_items(self, lane: dict) -> list[tuple[str, Any, list[Any], list[Any]]]:
        from hermes_cli import kanban_db as _kb
        items = []
        for board_meta in _kb.list_boards(include_archived=False):
            conn = _kb.connect(board=board_meta.get("slug"))
            try:
                topic_ids = _kb.active_task_index_topic_ids(
                    conn, platform=lane["platform"], chat_id=lane["chat_id"],
                    notifier_profile=lane.get("notifier_profile") or "",
                )
                requested_thread_id = str(lane.get("thread_id") or "")
                topic_groups = (
                    {requested_thread_id: topic_ids.get(requested_thread_id, [])}
                    if requested_thread_id else topic_ids
                )
                for thread_id, task_ids in topic_groups.items():
                    board_name = str(board_meta.get("name") or board_meta.get("slug") or "Задачи")
                    for task_id in task_ids:
                        task = _kb.get_task(conn, task_id)
                        if task is None:
                            continue
                        parents = [
                            parent for parent_id in _kb.parent_ids(conn, task_id)
                            if (parent := _kb.get_task(conn, parent_id)) is not None
                        ]
                        surface = conn.execute(
                            "SELECT message_id FROM kanban_status_surfaces WHERE task_id=? AND platform=? "
                            "AND chat_id=? AND thread_id=?",
                            (task_id, lane["platform"], lane["chat_id"], thread_id),
                        ).fetchone()
                        status_card_url = self._telegram_status_card_url(
                            lane["chat_id"], thread_id,
                            str(surface["message_id"]) if surface and surface["message_id"] else "",
                        )
                        items.append((
                            board_name, task, _kb.list_events(conn, task_id), parents,
                            _kb.current_run_progress(conn, task_id), status_card_url,
                        ))
            finally:
                conn.close()
        return items

    @staticmethod
    def _telegram_status_card_url(chat_id: Any, thread_id: Any, message_id: str) -> str:
        """Return a Telegram deep link only for group/forum messages."""
        chat = str(chat_id or "")
        if not message_id or not chat.startswith("-100"):
            return ""
        path = chat[4:]
        return f"https://t.me/c/{path}/{thread_id}/{message_id}" if thread_id else f"https://t.me/c/{path}/{message_id}"

    def _kanban_claim_active_task_index(self, lane: dict, owner: str) -> Optional[dict]:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect_active_task_index_registry()
        try:
            return _kb.claim_active_task_index(
                conn, platform=lane["platform"], chat_id=lane["chat_id"],
                thread_id=lane.get("thread_id") or "", notifier_profile=lane.get("notifier_profile") or "", owner=owner,
            )
        finally:
            conn.close()

    def _kanban_record_active_task_index_delivery(
        self, lane: dict, owner: str, generation: int, message_id: str, render_hash: str,
        pin_attempted: bool, pinned: bool,
    ) -> bool:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect_active_task_index_registry()
        try:
            return _kb.record_active_task_index_delivery(
                conn, platform=lane["platform"], chat_id=lane["chat_id"], thread_id=lane.get("thread_id") or "",
                notifier_profile=lane.get("notifier_profile") or "", owner=owner, generation=generation,
                message_id=message_id, renderer_version=_KANBAN_ACTIVE_INDEX_RENDERER_VERSION,
                render_hash=render_hash, pin_attempted=pin_attempted, pinned=pinned,
                sender_profile=lane.get("notifier_profile") or None,
            )
        finally:
            conn.close()

    def _kanban_record_active_task_index_failure(
        self, lane: dict, owner: str, generation: int, error: str,
    ) -> Optional[dict]:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect_active_task_index_registry()
        try:
            return _kb.record_active_task_index_failure(
                conn, platform=lane["platform"], chat_id=lane["chat_id"], thread_id=lane.get("thread_id") or "",
                notifier_profile=lane.get("notifier_profile") or "", owner=owner, generation=generation, error=error,
            )
        finally:
            conn.close()

    def _kanban_forget_active_task_index_message(
        self, lane: dict, owner: str, generation: int,
    ) -> bool:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect_active_task_index_registry()
        try:
            return _kb.forget_active_task_index_message(
                conn, platform=lane["platform"], chat_id=lane["chat_id"], thread_id=lane.get("thread_id") or "",
                notifier_profile=lane.get("notifier_profile") or "", owner=owner, generation=generation,
            )
        finally:
            conn.close()

    async def _deliver_terminal_notification(
        self, adapter, notification: dict, board: Optional[str], owner: str,
    ) -> int:
        """Send one root outcome handoff under a lease independent of the card."""
        claimed = await asyncio.to_thread(
            self._kanban_claim_terminal_notification, notification, owner, board,
        )
        if claimed is None:
            return 0
        metadata = _kanban_status_route_metadata(adapter, {
            "platform": claimed["platform"],
            "chat_id": claimed["chat_id"],
            "thread_id": claimed.get("thread_id") or "",
        }, "done")
        try:
            result = await adapter.send(
                claimed["chat_id"],
                f"Итоги готовы:\n{claimed['outcome_summary']}",
                metadata=metadata,
            )
            from gateway.platforms.base import SendResult
            if not isinstance(result, SendResult) or not result.success or not result.message_id:
                error = getattr(result, "error", None) if result is not None else None
                raise RuntimeError(error or "kanban terminal notification requires canonical SendResult receipt")
            if not await asyncio.to_thread(
                self._kanban_record_terminal_delivery,
                claimed, owner, int(claimed["lease_generation"]), str(result.message_id), board,
            ):
                raise RuntimeError("kanban terminal notification lost lease before receipt")
        except Exception as exc:
            failure_state = await asyncio.to_thread(
                self._kanban_record_terminal_failure,
                claimed, owner, int(claimed["lease_generation"]), str(exc), board,
            )
            logger.warning(
                "kanban notifier: final-ready ping failed for %s; rich card receipt remains confirmed: %s",
                claimed["task_id"], exc,
            )
            return _flood_cooldown_from_failure(exc, failure_state)
        return 0

    def _kanban_enqueue_terminal_notification(
        self, sub: dict, event_id: int, title: str, event_payload: Any,
        board: Optional[str] = None,
    ) -> Optional[dict]:
        """Sync helper: enqueue a root handoff after the corresponding card receipt."""
        from hermes_cli import kanban_db as _kb
        payload = event_payload if isinstance(event_payload, dict) else {}
        outcome_key = str(payload.get("notification_key") or "").strip()
        outcome_summary = str(payload.get("notification_summary") or "").strip()
        if not outcome_key or not outcome_summary:
            return None
        conn = _kb.connect(board=board)
        try:
            if not _kb.enqueue_terminal_notification(
                conn, task_id=sub["task_id"], platform=sub["platform"],
                chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or "",
                notifier_profile=sub.get("notifier_profile"), event_id=event_id, title=title,
                outcome_key=outcome_key, outcome_summary=outcome_summary,
            ):
                return None
            return {
                "task_id": sub["task_id"], "platform": sub["platform"],
                "chat_id": sub["chat_id"], "thread_id": sub.get("thread_id") or "",
                "notifier_profile": sub.get("notifier_profile"), "event_id": event_id,
                "title": title[:140],
            }
        finally:
            conn.close()

    def _kanban_claim_terminal_notification(
        self, notification: dict, owner: str, board: Optional[str] = None,
    ) -> Optional[dict]:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.claim_terminal_notification(conn, notification=notification, owner=owner)
        finally:
            conn.close()

    def _kanban_record_terminal_delivery(
        self, notification: dict, owner: str, generation: int, message_id: str,
        board: Optional[str] = None,
    ) -> bool:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.record_terminal_notification_delivery(
                conn, notification=notification, owner=owner,
                generation=generation, message_id=message_id,
            )
        finally:
            conn.close()

    def _kanban_record_terminal_failure(
        self, notification: dict, owner: str, generation: int, error: str,
        board: Optional[str] = None,
    ) -> Optional[dict]:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.record_terminal_notification_failure(
                conn, notification=notification, owner=owner,
                generation=generation, error=error,
            )
        finally:
            conn.close()

    def _kanban_claim_status_surface(
        self, sub: dict, owner: str, event_id: int, board: Optional[str] = None,
        recover_parked: bool = False,
    ) -> Optional[dict]:
        """Claim the exact-origin status card before a remote render attempt."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.claim_status_surface(
                conn, task_id=sub["task_id"], platform=sub["platform"],
                chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or "",
                owner=owner, event_id=event_id, recover_parked=recover_parked,
            )
        finally:
            conn.close()

    def _kanban_status_message_exists(self, sub: dict, board: Optional[str] = None) -> bool:
        """Return whether the immutable origin lane already has a card id."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            row = conn.execute(
                "SELECT message_id FROM kanban_status_surfaces WHERE task_id=? AND platform=? "
                "AND chat_id=? AND thread_id=?",
                (sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or ""),
            ).fetchone()
            return bool(row and row["message_id"])
        finally:
            conn.close()

    def _kanban_record_status_delivery(
        self, sub: dict, owner: str, generation: int, event_id: int,
        message_id: Optional[str], board: Optional[str] = None,
        renderer_version: Optional[str] = None, render_hash: Optional[str] = None,
    ) -> bool:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.record_status_surface_delivery(
                conn, task_id=sub["task_id"], platform=sub["platform"],
                chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or "",
                owner=owner, generation=generation, event_id=event_id,
                message_id=str(message_id) if message_id is not None else None,
                renderer_version=renderer_version, render_hash=render_hash,
                sender_profile=sub.get("notifier_profile") or None,
            )
        finally:
            conn.close()

    def _kanban_record_status_failure(
        self, sub: dict, owner: str, generation: int, error: str,
        board: Optional[str] = None,
    ) -> Optional[dict]:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            return _kb.record_status_surface_failure(
                conn, task_id=sub["task_id"], platform=sub["platform"],
                chat_id=sub["chat_id"], thread_id=sub.get("thread_id") or "",
                owner=owner, generation=generation, error=error,
            )
        finally:
            conn.close()

    def _kanban_advance(
        self, sub: dict, cursor: int, board: Optional[str] = None,
    ) -> None:
        """Sync helper: advance a subscription's cursor. Runs in to_thread.

        ``board`` scopes the DB connection to the board that owns this
        subscription. Unsub cursors in one board can't touch another's.
        """
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.advance_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                new_cursor=cursor,
            )
        finally:
            conn.close()

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.remove_notify_sub(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
            )
        finally:
            conn.close()

    def _kanban_rewind(
        self,
        sub: dict,
        claimed_cursor: int,
        old_cursor: int,
        board: Optional[str] = None,
    ) -> None:
        """Sync helper: undo a claimed notification cursor after send failure."""
        from hermes_cli import kanban_db as _kb
        conn = _kb.connect(board=board)
        try:
            _kb.rewind_notify_cursor(
                conn,
                task_id=sub["task_id"],
                platform=sub["platform"],
                chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "",
                claimed_cursor=claimed_cursor,
                old_cursor=old_cursor,
            )
        finally:
            conn.close()

    async def _deliver_kanban_artifacts(
        self,
        *,
        adapter,
        chat_id: str,
        metadata: dict,
        event_payload: Optional[dict],
        task,
    ) -> None:
        """Upload artifacts only after an explicit user-requested opt-in.

        Workers preserve ``kanban_complete(artifacts=[...])`` for reviewers
        and downstream workers. The notifier uploads them only when the event
        also has ``deliver_artifacts=true``.

        Summary/result text is never scanned for paths. The explicit event
        manifest is classified through a fail-closed allowlist before upload.
        """
        candidates: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            if not path:
                return
            expanded = os.path.expanduser(path)
            if expanded in seen:
                return
            if not os.path.isfile(expanded):
                return
            seen.add(expanded)
            candidates.append(expanded)

        # The explicit completion manifest is the only automatic-delivery source.
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                for item in raw:
                    if isinstance(item, str):
                        _add(item)

        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        safe_candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        allow_text_document = bool(event_payload and event_payload.get("deliver_artifacts") is True)
        classified_candidates = [
            (path, kind) for path in safe_candidates
            if (kind := _classify_kanban_artifact(path, allow_text_document=allow_text_document)) is not None
        ]
        if not classified_candidates:
            return

        from urllib.parse import quote as _quote

        # Partition images so they ride a single send_multiple_images call
        # on platforms that support batch image uploads (Signal/Slack RPCs).
        image_paths = [path for path, kind in classified_candidates if kind == "image"]
        other_paths = [(path, kind) for path, kind in classified_candidates if kind != "image"]

        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(
                    chat_id=chat_id, images=batch, metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: image batch upload failed: %s", exc,
                )

        for path, artifact_type in other_paths:
            try:
                if artifact_type == "video":
                    await adapter.send_video(
                        chat_id=chat_id, video_path=path, metadata=metadata,
                    )
                else:
                    await adapter.send_document(
                        chat_id=chat_id, file_path=path, metadata=metadata,
                    )
            except Exception as exc:
                logger.warning(
                    "kanban notifier: artifact upload (%s) failed: %s",
                    path, exc,
                )

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
        When true, the gateway hosts the single dispatcher for this profile:
        no separate `hermes kanban daemon` process needed. When false, the
        loop exits immediately and an external daemon is expected.

        Each tick calls :func:`kanban_db.dispatch_once` inside
        ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
        event loop. Failures in one tick don't stop subsequent ticks —
        same pattern as `_kanban_notifier_watcher`.

        Shutdown: the loop checks ``self._running`` between ticks; gateway
        stop() flips it to False and cancels pending tasks, and the
        in-flight ``to_thread`` returns on its own after the current
        ``dispatch_once`` call finishes (typically <1ms on an idle board).
        """
        # Read config once at boot. If the user flips the flag later, they
        # restart the gateway; same pattern as every other background
        # watcher here. Honours HERMES_KANBAN_DISPATCH_IN_GATEWAY env var
        # as an escape hatch (false-y value disables without editing YAML).
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return

        # Single-dispatcher backstop. dispatch_in_gateway defaults to true, so a
        # new profile gateway (or a same-profile restart race) can silently
        # start a second dispatcher; concurrent dispatchers double reclaim
        # frequency, double claim-attempt events, and — with
        # wal_autocheckpoint=0 — concurrent manual WAL checkpoints can corrupt
        # index pages. The lock lives at the machine-global kanban root
        # (shared across profiles by design), so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info(
                "kanban dispatcher: another gateway already holds the dispatcher "
                "lock (%s); this gateway will NOT dispatch.", _lock_path,
            )
            return
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning(
                "kanban dispatcher: advisory lock unavailable at %s; proceeding "
                "on config control alone.", _lock_path,
            )

        try:
            interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
        except (ValueError, TypeError):
            logger.warning(
                "kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                kanban_cfg.get("dispatch_interval_seconds"),
            )
            interval = 60.0
        interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

        # Read max_spawn config to limit concurrent kanban tasks
        max_spawn = kanban_cfg.get("max_spawn", None)
        if max_spawn is not None:
            logger.info(f"kanban dispatcher: max_spawn={max_spawn}")

        # Cap the number of simultaneously running tasks so slow workers
        # (local LLMs, resource-constrained hosts) don't pile up and time
        # out. When set, the dispatcher skips spawning when the board
        # already has this many tasks in 'running' status.
        raw_max_in_progress = kanban_cfg.get("max_in_progress", None)
        max_in_progress = None
        if raw_max_in_progress is not None:
            try:
                max_in_progress = int(raw_max_in_progress)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress=%r; ignoring",
                    raw_max_in_progress,
                )
                max_in_progress = None
            else:
                if max_in_progress < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress=%r is below 1; ignoring",
                        raw_max_in_progress,
                    )
                    max_in_progress = None
                else:
                    logger.info(f"kanban dispatcher: max_in_progress={max_in_progress}")

        raw_failure_limit = kanban_cfg.get("failure_limit", _kb.DEFAULT_FAILURE_LIMIT)
        try:
            failure_limit = int(raw_failure_limit)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT
        if failure_limit < 1:
            logger.warning(
                "kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                raw_failure_limit,
                _kb.DEFAULT_FAILURE_LIMIT,
            )
            failure_limit = _kb.DEFAULT_FAILURE_LIMIT

        # Read stale_timeout_seconds — 0 disables stale detection.
        raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
        try:
            stale_timeout_seconds = int(raw_stale or 0)
        except (TypeError, ValueError):
            logger.warning(
                "kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                "disabling stale detection",
                raw_stale,
            )
            stale_timeout_seconds = 0

        # Read kanban.default_assignee — fallback profile for tasks
        # created without an explicit assignee (e.g. via the dashboard).
        # When set, the dispatcher applies it to unassigned ready tasks
        # instead of skipping them indefinitely (#27145). Empty string
        # (the schema default) means "no fallback, keep skipping" —
        # backward-compatible with existing installs.
        default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
        if default_assignee:
            logger.info(
                "kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                "will route to this profile)",
                default_assignee,
            )

        # Read kanban.max_in_progress_per_profile — per-profile concurrency
        # cap (#21582). When set, no single profile gets more than N
        # workers running at once, even if the global max_in_progress
        # would allow it. Prevents one profile's local model / API quota
        # / browser pool from being overwhelmed by a fan-out.
        raw_per_profile = kanban_cfg.get("max_in_progress_per_profile", None)
        max_in_progress_per_profile = None
        if raw_per_profile is not None:
            try:
                max_in_progress_per_profile = int(raw_per_profile)
            except (TypeError, ValueError):
                logger.warning(
                    "kanban dispatcher: invalid kanban.max_in_progress_per_profile=%r; ignoring",
                    raw_per_profile,
                )
                max_in_progress_per_profile = None
            else:
                if max_in_progress_per_profile < 1:
                    logger.warning(
                        "kanban dispatcher: kanban.max_in_progress_per_profile=%r is below 1; ignoring",
                        raw_per_profile,
                    )
                    max_in_progress_per_profile = None
                else:
                    logger.info(
                        "kanban dispatcher: max_in_progress_per_profile=%d",
                        max_in_progress_per_profile,
                    )

        # Initial delay so the gateway finishes wiring adapters before the
        # dispatcher spawns workers (those workers may hit gateway notify
        # subscriptions etc.). Matches the notifier watcher's delay.
        await asyncio.sleep(5)

        # Health telemetry mirrored from `_cmd_daemon`: warn when ready
        # queue is non-empty but spawns are 0 for N consecutive ticks —
        # usually means broken PATH, missing venv, or credential loss.
        HEALTH_WINDOW = 6
        bad_ticks = 0
        last_warn_at = 0
        # Avoid hot-looping corrupt-looking board DBs, but do not suppress
        # same-fingerprint retries forever: transient WAL/open races can
        # surface as "database disk image is malformed" for one tick.
        CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300
        disabled_corrupt_boards: dict[
            str, tuple[tuple[str, int | None, int | None], float]
        ] = {}

        def _board_db_fingerprint(slug: str) -> tuple[str, int | None, int | None]:
            path = _kb.kanban_db_path(slug)
            try:
                resolved = str(path.expanduser().resolve())
            except Exception:
                resolved = str(path)
            try:
                stat = path.stat()
            except OSError:
                return (resolved, None, None)
            return (resolved, stat.st_mtime_ns, stat.st_size)

        def _is_corrupt_board_db_error(exc: Exception) -> bool:
            corrupt_guard_error = getattr(_kb, "KanbanDbCorruptError", None)
            if corrupt_guard_error is not None and isinstance(exc, corrupt_guard_error):
                return True
            if not isinstance(exc, sqlite3.DatabaseError):
                return False
            msg = str(exc).lower()
            return (
                "file is not a database" in msg
                or "database disk image is malformed" in msg
            )

        def _tick_once_for_board(slug: str) -> "Optional[object]":
            """Run one dispatch_once for a specific board.

            Runs in a worker thread via `asyncio.to_thread`. `board=slug`
            is passed through `dispatch_once` so `resolve_workspace` and
            `_default_spawn` see the right paths. The per-board DB is
            opened explicitly so concurrent boards never share a
            connection handle or accidentally claim across each other.
            """
            conn = None
            fingerprint = _board_db_fingerprint(slug)
            disabled_entry = disabled_corrupt_boards.get(slug)
            if disabled_entry is not None:
                disabled_fingerprint, disabled_at = disabled_entry
                age = time.monotonic() - disabled_at
                if (
                    disabled_fingerprint == fingerprint
                    and age < CORRUPT_BOARD_RETRY_AFTER_SECONDS
                ):
                    return None
                if disabled_fingerprint == fingerprint:
                    logger.info(
                        "kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch",
                        slug,
                        age,
                    )
                else:
                    logger.info(
                        "kanban dispatcher: board %s database changed; retrying dispatch",
                        slug,
                    )
                disabled_corrupt_boards.pop(slug, None)
            try:
                conn = _kb.connect(board=slug)
                # `connect()` runs the schema + idempotent migration on
                # first open per process; the previous explicit
                # `init_db()` call here busted the per-process cache and
                # re-ran the migration on a second connection, racing
                # the first. See the matching comment in
                # `_kanban_notifier_watcher` and issue #21378.
                return _kb.dispatch_once(
                    conn,
                    board=slug,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    stale_timeout_seconds=stale_timeout_seconds,
                    default_assignee=default_assignee,
                    max_in_progress_per_profile=max_in_progress_per_profile,
                )
            except sqlite3.DatabaseError as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            except Exception as exc:
                if _is_corrupt_board_db_error(exc):
                    disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                    logger.error(
                        "kanban dispatcher: board %s database %s is not a valid "
                        "SQLite database; pausing dispatch for this board until "
                        "the file changes, the gateway restarts, or the "
                        "quarantine timer expires. Move or restore the file, "
                        "then run `hermes kanban init` if you need a fresh board.",
                        slug,
                        fingerprint[0],
                    )
                    return None
                logger.exception("kanban dispatcher: tick failed on board %s", slug)
                return None
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        def _tick_once() -> "list[tuple[str, Optional[object]]]":
            """Run one dispatch_once per board. Returns (slug, result) pairs.

            Enumerating boards on every tick keeps the dispatcher honest
            when users create a new board mid-run: no restart required,
            the next tick picks it up automatically.
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            out: list[tuple[str, "Optional[object]"]] = []
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                out.append((slug, _tick_once_for_board(slug)))
            return out

        def _ready_nonempty() -> bool:
            """Cheap probe: is there at least one ready+assigned+unclaimed
            task on ANY board whose assignee maps to a real Hermes profile
            (i.e. one the dispatcher would actually spawn for)?

            Tasks assigned to control-plane lanes (e.g. ``orion-cc``,
            ``orion-research``) are pulled by terminals via
            ``claim_task`` directly and never spawnable, so a queue full
            of those is "correctly idle", not "stuck". Filtering them out
            here keeps the stuck-warn fire only on real failures (broken
            PATH, missing venv, credential loss for a real Hermes profile).
            """
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                conn = None
                try:
                    conn = _kb.connect(board=slug)
                    if _kb.has_spawnable_ready(conn):
                        return True
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
            return False

        # Auto-decompose: turn fresh triage tasks into ready workgraphs
        # before the dispatcher fans out workers. Gated by
        # ``kanban.auto_decompose`` (default True). Capped by
        # ``kanban.auto_decompose_per_tick`` (default 3) so a bulk-load
        # of triage tasks doesn't burst-spend the aux LLM in one tick;
        # remainder defers to subsequent ticks.
        #
        # The flag is re-read from config EVERY tick (#49638) rather than
        # captured once at boot. Auto-decompose is a safety toggle: a user who
        # sees it fan out and run tasks they didn't intend reaches for
        # ``kanban.auto_decompose: false`` to STOP it — and that must take
        # effect on the next tick, not require a gateway restart. (Reported:
        # auto-decompose created and launched destructive tasks while the user
        # was still typing the task description, and the flag "couldn't be
        # disabled" because the gateway had captured its boot-time value.)
        def _read_auto_decompose_settings() -> tuple[bool, int]:
            """Re-resolve (enabled, per_tick) from current config each tick."""
            return _resolve_auto_decompose_settings(_load_config)

        def _auto_decompose_tick(auto_decompose_per_tick: int) -> int:
            """Run the auto-decomposer for up to N triage tasks across all
            boards. Returns the number of triage tasks that were
            successfully decomposed or specified this tick.
            """
            try:
                from hermes_cli import kanban_decompose as _decomp
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "kanban auto-decompose: import failed (%s); skipping", exc,
                )
                return 0
            try:
                boards = _kb.list_boards(include_archived=False)
            except Exception:
                boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
            attempted = 0
            successes = 0
            for b in boards:
                slug = b.get("slug") or _kb.DEFAULT_BOARD
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin this board for the duration of the call — same
                # pattern as the dashboard specify endpoint. The
                # decomposer module connects with no board kwarg and
                # relies on the env var.
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug(
                            "kanban auto-decompose: list_triage_ids failed on board %s (%s)",
                            slug, exc,
                        )
                        triage_ids = []
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        attempted += 1
                        try:
                            outcome = _decomp.decompose_task(
                                tid, author="auto-decomposer",
                            )
                        except Exception:
                            logger.exception(
                                "kanban auto-decompose: decompose_task crashed on %s",
                                tid,
                            )
                            continue
                        if outcome.ok:
                            successes += 1
                            if outcome.fanout and outcome.child_ids:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → %d children",
                                    slug, tid, len(outcome.child_ids),
                                )
                            else:
                                logger.info(
                                    "kanban auto-decompose [%s]: %s → single task (no fanout)",
                                    slug, tid,
                                )
                        else:
                            # Common no-op reasons (no aux client configured) shouldn't
                            # spam logs every tick. Log at debug.
                            logger.debug(
                                "kanban auto-decompose [%s]: %s skipped: %s",
                                slug, tid, outcome.reason,
                            )
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
            return successes

        logger.info(
            "kanban dispatcher: embedded in gateway (interval=%.1fs)", interval
        )
        while self._running:
            try:
                # Reap zombie children before per-board work so a board DB
                # failure cannot block cleanup of unrelated workers.
                pids = await asyncio.to_thread(_kb.reap_worker_zombies)
                if pids:
                    logger.info(
                        "kanban dispatcher: reaped %d zombie worker(s), pids=%s",
                        len(pids),
                        pids,
                    )
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Re-read the auto-decompose toggle live each tick so a user
                # flipping kanban.auto_decompose=false to STOP runaway fan-out
                # takes effect on the next tick, not on gateway restart (#49638).
                _ad_enabled, _ad_per_tick = _read_auto_decompose_settings()
                if _ad_enabled:
                    await asyncio.to_thread(_auto_decompose_tick, _ad_per_tick)
                results = await asyncio.to_thread(_tick_once)
                any_spawned = False
                for slug, res in (results or []):
                    if res is not None and getattr(res, "spawned", None):
                        any_spawned = True
                        # Quiet by default — only log when something actually
                        # happened, so an idle gateway stays silent.
                        logger.info(
                            "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
                            "crashed=%d timed_out=%d promoted=%d auto_blocked=%d",
                            slug,
                            len(res.spawned),
                            res.reclaimed,
                            len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
                            len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
                            res.promoted,
                            len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
                        )
                # Health telemetry (aggregate across boards)
                ready_pending = await asyncio.to_thread(_ready_nonempty)
                if ready_pending and not any_spawned:
                    bad_ticks += 1
                else:
                    bad_ticks = 0
                if bad_ticks >= HEALTH_WINDOW:
                    now = int(time.time())
                    if now - last_warn_at >= 300:
                        logger.warning(
                            "kanban dispatcher stuck: ready queue non-empty for "
                            "%d consecutive ticks but 0 workers spawned. Check "
                            "profile health (venv, PATH, credentials) and "
                            "`hermes kanban list --status ready`.",
                            bad_ticks,
                        )
                        last_warn_at = now
            except asyncio.CancelledError:
                logger.debug("kanban dispatcher: cancelled")
                _release_singleton_lock(self._kanban_dispatcher_lock_handle)
                self._kanban_dispatcher_lock_handle = None
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

        _release_singleton_lock(self._kanban_dispatcher_lock_handle)
        self._kanban_dispatcher_lock_handle = None
