"""Extracted kanban dispatcher and notifier functions from GatewayRunner.

These are standalone functions that take a ``runner`` parameter
(the GatewayRunner instance) instead of ``self``.
"""

import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as _kb
from gateway.config import Platform as _Platform

logger = logging.getLogger(__name__)


async def kanban_notifier_watcher(runner, interval: float = 5.0) -> None:
    """Poll ``kanban_notify_subs`` and deliver terminal events to users.

    For each subscription row, fetches ``task_events`` newer than the
    stored cursor with kind in the terminal set (``completed``,
    ``blocked``, ``gave_up``, ``crashed``, ``timed_out``). Sends one
    message per new event to ``(platform, chat_id, thread_id)``,
    then advances the cursor. When a task reaches a terminal state
    (``completed`` / ``archived``), the subscription is removed.

    Runs in the gateway event loop; all SQLite work is pushed to a
    thread via ``asyncio.to_thread`` so the loop never blocks on the
    WAL lock. Failures in one tick don't stop subsequent ticks.

    **Multi-board:** iterates every board discovered on disk per
    tick. Subscriptions live inside each board's own DB and cannot
    cross boards, so delivery semantics are unchanged — this is
    purely a fan-out of the single-DB poll.
    """
    try:
        from hermes_cli import kanban_db as _kb
    except Exception:
        logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
        return

    TERMINAL_KINDS = ("completed", "blocked", "gave_up", "crashed", "timed_out")
    MAX_SEND_FAILURES = 3
    sub_fail_counts: dict[tuple, int] = getattr(
        runner, "_kanban_sub_fail_counts", {}
    )
    runner._kanban_sub_fail_counts = sub_fail_counts
    notifier_profile = getattr(runner, "_kanban_notifier_profile", None)
    if not notifier_profile:
        notifier_profile = runner._active_profile_name()
        runner._kanban_notifier_profile = notifier_profile

    # Initial delay so the gateway can finish wiring adapters.
    await asyncio.sleep(5)

    while runner._running:
        try:
            def _collect():
                deliveries: list[dict] = []
                active_platforms = {
                    getattr(platform, "value", str(platform)).lower()
                    for platform in runner.adapters.keys()
                }
                if not active_platforms:
                    logger.debug("kanban notifier: no connected adapters; skipping tick")
                    return deliveries

                # Enumerate every board on disk
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
                        subs = _kb.list_notify_subs(conn)
                        if not subs:
                            logger.debug("kanban notifier: board %s has no subscriptions", slug)
                        for sub in subs:
                            owner_profile = sub.get("notifier_profile") or None
                            if owner_profile and owner_profile != notifier_profile:
                                logger.debug(
                                    "kanban notifier: subscription for %s owned by profile %s; current profile %s skipping",
                                    sub.get("task_id"), owner_profile, notifier_profile,
                                )
                                continue
                            platform = (sub.get("platform") or "").lower()
                            if platform not in active_platforms:
                                logger.debug(
                                    "kanban notifier: subscription for %s on %s skipped; adapter not connected",
                                    sub.get("task_id"), platform or "<missing>",
                                )
                                continue
                            old_cursor, cursor, events = _kb.claim_unseen_events_for_sub(
                                conn,
                                task_id=sub["task_id"],
                                platform=sub["platform"],
                                chat_id=sub["chat_id"],
                                thread_id=sub.get("thread_id") or "",
                                kinds=TERMINAL_KINDS,
                            )
                            if not events:
                                continue
                            task = _kb.get_task(conn, sub["task_id"])
                            logger.debug(
                                "kanban notifier: claimed %d event(s) for %s on board %s cursor %s→%s",
                                len(events), sub["task_id"], slug, old_cursor, cursor,
                            )
                            deliveries.append({
                                "sub": sub,
                                "old_cursor": old_cursor,
                                "cursor": cursor,
                                "events": events,
                                "task": task,
                                "board": slug,
                            })
                    finally:
                        conn.close()
                return deliveries

            deliveries = await asyncio.to_thread(_collect)
            for d in deliveries:
                sub = d["sub"]
                task = d["task"]
                board_slug = d.get("board")
                platform_str = (sub["platform"] or "").lower()
                try:
                    plat = _Platform(platform_str)
                except ValueError:
                    await asyncio.to_thread(
                        kanban_advance, runner, sub, d["cursor"], board_slug,
                    )
                    continue
                adapter = runner.adapters.get(plat)
                if adapter is None:
                    logger.debug(
                        "kanban notifier: adapter %s disconnected before delivery for %s; rewinding claim",
                        platform_str, sub["task_id"],
                    )
                    await asyncio.to_thread(
                        kanban_rewind,
                        runner,
                        sub,
                        d["cursor"],
                        d.get("old_cursor", 0),
                        board_slug,
                    )
                    continue
                title = (task.title if task else sub["task_id"])[:120]
                for ev in d["events"]:
                    kind = ev.kind
                    who = (task.assignee if task and task.assignee else None)
                    tag = f"@{who} " if who else ""
                    if kind == "completed":
                        handoff = ""
                        payload_summary = None
                        if ev.payload and ev.payload.get("summary"):
                            payload_summary = str(ev.payload["summary"])
                        if payload_summary:
                            h = payload_summary.strip().splitlines()[0][:200]
                            handoff = f"\n{h}"
                        elif task and task.result:
                            r = task.result.strip().splitlines()[0][:160]
                            handoff = f"\n{r}"
                        msg = (
                            f"✔ {tag}Kanban {sub['task_id']} done"
                            f" — {title}{handoff}"
                        )
                    elif kind == "blocked":
                        reason = ""
                        if ev.payload and ev.payload.get("reason"):
                            reason = f": {str(ev.payload['reason'])[:160]}"
                        msg = f"⏸ {tag}Kanban {sub['task_id']} blocked{reason}"
                    elif kind == "gave_up":
                        err = ""
                        if ev.payload and ev.payload.get("error"):
                            err = f"\n{str(ev.payload['error'])[:200]}"
                        msg = (
                            f"✖ {tag}Kanban {sub['task_id']} gave up "
                            f"after repeated spawn failures{err}"
                        )
                    elif kind == "crashed":
                        msg = (
                            f"✖ {tag}Kanban {sub['task_id']} worker crashed "
                            f"(pid gone); dispatcher will retry"
                        )
                    elif kind == "timed_out":
                        limit = 0
                        if ev.payload and ev.payload.get("limit_seconds"):
                            limit = int(ev.payload["limit_seconds"])
                        msg = (
                            f"⏱ {tag}Kanban {sub['task_id']} timed out "
                            f"(max_runtime={limit}s); will retry"
                        )
                    else:
                        continue
                    metadata: dict[str, Any] = {}
                    if sub.get("thread_id"):
                        metadata["thread_id"] = sub["thread_id"]
                    sub_key = (
                        sub["task_id"], sub["platform"],
                        sub["chat_id"], sub.get("thread_id") or "",
                    )
                    try:
                        await adapter.send(
                            sub["chat_id"], msg, metadata=metadata,
                        )
                        logger.debug(
                            "kanban notifier: delivered %s event for %s to %s/%s on board %s",
                            kind, sub["task_id"], platform_str, sub["chat_id"], board_slug,
                        )
                        if kind == "completed":
                            try:
                                await deliver_kanban_artifacts(
                                    runner,
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
                        sub_fail_counts.pop(sub_key, None)
                    except Exception as exc:
                        fails = sub_fail_counts.get(sub_key, 0) + 1
                        sub_fail_counts[sub_key] = fails
                        logger.warning(
                            "kanban notifier: send failed for %s on %s "
                            "(attempt %d/%d): %s",
                            sub["task_id"], platform_str, fails,
                            MAX_SEND_FAILURES, exc,
                        )
                        if fails >= MAX_SEND_FAILURES:
                            logger.warning(
                                "kanban notifier: dropping subscription "
                                "%s on %s after %d consecutive send failures",
                                sub["task_id"], platform_str, fails,
                            )
                            await asyncio.to_thread(kanban_unsub, runner, sub, board_slug)
                            sub_fail_counts.pop(sub_key, None)
                        else:
                            await asyncio.to_thread(
                                kanban_rewind,
                                runner,
                                sub,
                                d["cursor"],
                                d.get("old_cursor", 0),
                                board_slug,
                            )
                        break
                else:
                    await asyncio.to_thread(
                        kanban_advance, runner, sub, d["cursor"], board_slug,
                    )
                    task_terminal = task and task.status in {"done", "archived"}
                    if task_terminal:
                        await asyncio.to_thread(
                            kanban_unsub, runner, sub, board_slug,
                        )
        except Exception as exc:
            logger.warning("kanban notifier tick failed: %s", exc)
        for _ in range(int(max(1, interval))):
            if not runner._running:
                return
            await asyncio.sleep(1)


def kanban_advance(
    runner, sub: dict, cursor: int, board: Optional[str] = None,
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


def kanban_unsub(runner, sub: dict, board: Optional[str] = None) -> None:
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


def kanban_rewind(
    runner,
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


async def deliver_kanban_artifacts(
    runner,
    *,
    adapter,
    chat_id: str,
    metadata: dict,
    event_payload: Optional[dict],
    task,
) -> None:
    """Upload artifact files referenced by a completed kanban task.

    Workers passing ``kanban_complete(artifacts=[...])`` ship absolute
    file paths through the completion event so downstream humans get
    the deliverable as a native upload instead of a path printed in
    chat.

    Sources scanned, in priority order:
      1. ``event_payload['artifacts']`` (explicit list — preferred)
      2. ``event_payload['summary']`` (truncated first line)
      3. ``task.result`` (legacy fallback)

    Files are deduplicated, missing files are silently skipped (the
    path may have been mentioned for reference only), and delivery
    errors are logged but do not break the notifier loop.
    """
    from pathlib import Path as _Path

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

    # 1. Explicit artifacts list in payload.
    if isinstance(event_payload, dict):
        raw = event_payload.get("artifacts")
        if isinstance(raw, (list, tuple)):
            for item in raw:
                if isinstance(item, str):
                    _add(item)

        # 2. Paths embedded in the payload summary.
        summary = event_payload.get("summary")
        if isinstance(summary, str) and summary:
            paths, _ = adapter.extract_local_files(summary)
            for p in paths:
                _add(p)

    # 3. Legacy: paths embedded in task.result.
    if task is not None and getattr(task, "result", None):
        result_text = str(task.result)
        paths, _ = adapter.extract_local_files(result_text)
        for p in paths:
            _add(p)

    if not candidates:
        return

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}

    from urllib.parse import quote as _quote

    image_paths = [p for p in candidates if _Path(p).suffix.lower() in _IMAGE_EXTS]
    other_paths = [p for p in candidates if _Path(p).suffix.lower() not in _IMAGE_EXTS]

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

    for path in other_paths:
        ext = _Path(path).suffix.lower()
        try:
            if ext in _VIDEO_EXTS:
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


async def kanban_dispatcher_watcher(runner) -> None:
    """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

    Gated by `kanban.dispatch_in_gateway` in config.yaml (default True).
    When true, the gateway hosts the single dispatcher for this profile:
    no separate `hermes kanban daemon` process needed. When false, the
    loop exits immediately and an external daemon is expected.

    Each tick calls :func:`kanban_db.dispatch_once` inside
    ``asyncio.to_thread`` so the SQLite WAL lock never blocks the
    event loop. Failures in one tick don't stop subsequent ticks —
    same pattern as `kanban_notifier_watcher`.

    Shutdown: the loop checks ``runner._running`` between ticks; gateway
    stop() flips it to False and cancels pending tasks, and the
    in-flight ``to_thread`` returns on its own after the current
    ``dispatch_once`` call finishes (typically <1ms on an idle board).
    """
    # Read config once at boot.
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

    interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
    interval = max(interval, 1.0)

    max_spawn = kanban_cfg.get("max_spawn", None)
    if max_spawn is not None:
        logger.info(f"kanban dispatcher: max_spawn={max_spawn}")

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

    await asyncio.sleep(5)

    HEALTH_WINDOW = 6
    bad_ticks = 0
    last_warn_at = 0
    disabled_corrupt_boards: dict[str, tuple[str, int | None, int | None]] = {}

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
        if not isinstance(exc, sqlite3.DatabaseError):
            return False
        msg = str(exc).lower()
        return (
            "file is not a database" in msg
            or "database disk image is malformed" in msg
        )

    def _tick_once_for_board(slug: str) -> "Optional[object]":
        conn = None
        fingerprint = _board_db_fingerprint(slug)
        disabled_fingerprint = disabled_corrupt_boards.get(slug)
        if disabled_fingerprint == fingerprint:
            return None
        if disabled_fingerprint is not None:
            logger.info(
                "kanban dispatcher: board %s database changed; retrying dispatch",
                slug,
            )
            disabled_corrupt_boards.pop(slug, None)
        try:
            conn = _kb.connect(board=slug)
            return _kb.dispatch_once(
                conn,
                board=slug,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
            )
        except sqlite3.DatabaseError as exc:
            if _is_corrupt_board_db_error(exc):
                disabled_corrupt_boards[slug] = fingerprint
                logger.error(
                    "kanban dispatcher: board %s database %s is not a valid "
                    "SQLite database; disabling dispatch for this board "
                    "until the file changes or the gateway restarts. Move "
                    "or restore the file, then run `hermes kanban init` if "
                    "you need a fresh board.",
                    slug,
                    fingerprint[0],
                )
                return None
            logger.exception("kanban dispatcher: tick failed on board %s", slug)
            return None
        except Exception:
            logger.exception("kanban dispatcher: tick failed on board %s", slug)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _tick_once() -> "list[tuple[str, Optional[object]]]":
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
                if _kb.has_spawnable_review(conn):
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

    auto_decompose_enabled = bool(kanban_cfg.get("auto_decompose", True))
    try:
        auto_decompose_per_tick = int(
            kanban_cfg.get("auto_decompose_per_tick", 3) or 3
        )
    except (TypeError, ValueError):
        auto_decompose_per_tick = 3
    if auto_decompose_per_tick < 1:
        auto_decompose_per_tick = 1

    def _auto_decompose_tick() -> int:
        try:
            from hermes_cli import kanban_decompose as _decomp
        except Exception as exc:
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
    while runner._running:
        try:
            if auto_decompose_enabled:
                await asyncio.to_thread(_auto_decompose_tick)
            results = await asyncio.to_thread(_tick_once)
            any_spawned = False
            for slug, res in (results or []):
                if res is not None and getattr(res, "spawned", None):
                    any_spawned = True
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
            raise
        except Exception:
            logger.exception("kanban dispatcher: unexpected watcher error")

        slept = 0.0
        while slept < interval and runner._running:
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0
