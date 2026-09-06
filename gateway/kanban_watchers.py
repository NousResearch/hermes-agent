"""Kanban board watcher methods for GatewayRunner.

Background loops that subscribe to kanban boards, deliver notifications and
artifacts, and drive the multi-agent dispatcher. They use only ``self`` state,
so they live on a mixin ``GatewayRunner`` inherits. Per-tick work lives in
``kanban_watchers_notifier`` / ``kanban_watchers_dispatcher``; shared plumbing
in ``kanban_watchers_common``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional

from gateway.kanban_watchers_common import (
    _acquire_singleton_lock,
    _kanban_dispatch_allowed,
    _release_singleton_lock,
    _resolve_auto_decompose_settings,
    _gc_retention_days,
    _to_thread_process_service,
    logger,
)
from gateway.kanban_watchers_notifier import _KanbanNotification, _notifier_collect
from gateway.kanban_watchers_dispatcher import (
    _KanbanDispatcher,
    _log_spawn_results,
    _resolve_dispatcher_settings,
)
from agent.auxiliary_client import async_call_llm
from agent.i18n import t
from hermes_cli.config import cfg_get

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_GC_INTERVAL_SECONDS = 3600.0
_HEALTH_WINDOW = 6

_KANBAN_NOTIFIER_LLM_TIMEOUT_SECONDS = 8.0


def _build_notifier_format_prompt(raw_msg: str, *, kind: str) -> str:
    """Build the narrowly-scoped prompt used for user-facing notifications."""
    return (
        "Rewrite the following Kanban notification into concise, natural "
        "Traditional Chinese suitable for a Telegram notification.\n\n"
        "Rules:\n"
        "- Preserve every concrete fact exactly; do not paraphrase, drop, or "
        "invent facts.\n"
        "- Keep task IDs (for example, t_xxxxxxxx), lane names, skill names, "
        "counts, error text, and URLs exactly as written.\n"
        "- Translate and reformat prose only; do not translate identifiers, "
        "names, error text, or URLs.\n"
        "- Keep any emoji prefix already present in the raw notification (such "
        "as ⚠, ✖, or ✅).\n"
        "- Return only the finished notification text, with no explanation or "
        "commentary.\n\n"
        f"Event kind: {kind}\n"
        f"Raw notification:\n{raw_msg}"
    )


def _notifier_llm_text(response: Any) -> str:
    """Extract text from the response shape returned by ``async_call_llm``."""
    if isinstance(response, str):
        return response.strip()
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


async def _format_notifier_message_zh(raw_msg: str, *, kind: str) -> str:
    """Format one raw notifier message in Traditional Chinese.

    This is deliberately best-effort: a notification must still be delivered
    if the auxiliary LLM is unavailable, slow, or returns no usable content.
    """
    try:
        response = await asyncio.wait_for(
            async_call_llm(
                task="kanban_notifier_formatter",
                messages=[
                    {
                        "role": "user",
                        "content": _build_notifier_format_prompt(raw_msg, kind=kind),
                    },
                ],
                temperature=0.2,
                max_tokens=500,
                timeout=_KANBAN_NOTIFIER_LLM_TIMEOUT_SECONDS,
            ),
            timeout=_KANBAN_NOTIFIER_LLM_TIMEOUT_SECONDS,
        )
        formatted = _notifier_llm_text(response)
        if not formatted:
            raise ValueError("empty formatter response")
        return formatted
    except Exception as exc:
        logger.warning(
            "kanban notifier: LLM formatting failed for kind %s (%s); "
            "sending raw message",
            kind,
            type(exc).__name__,
        )
        return raw_msg


def _notifier_llm_enabled() -> bool:
    """Read the opt-out gate; formatter failures always fail open."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return bool(cfg_get(cfg, "kanban", "notifier_llm_format", default=True))
    except Exception:
        return True


def _resolve_gateway_max_in_progress(kanban_cfg: dict, kb: Any):
    """Use the shared resolver for the embedded dispatcher configuration."""
    return kb.resolve_max_in_progress(
        kanban_cfg.get("max_in_progress"), warn=logger.warning,
    )
_SWARM_SUPERVISOR_DEFAULT_INTERVAL = 30.0
_SWARM_SUPERVISOR_MIN_INTERVAL = 15.0
_SWARM_SUPERVISOR_SOURCE = "swarm_supervisor"


def _swarm_topology_node_ids(swarm: dict[str, Any]) -> list[str]:
    """Stable unique worker / verifier / synthesizer ids from a topology dict."""
    ids: list[str] = []
    ids.extend(str(x).strip() for x in (swarm.get("worker_ids") or []) if str(x).strip())
    verifier_id = str(swarm.get("verifier_id") or "").strip()
    if verifier_id:
        ids.append(verifier_id)
    synthesizer_id = str(swarm.get("synthesizer_id") or "").strip()
    if synthesizer_id:
        ids.append(synthesizer_id)
    seen: set[str] = set()
    out: list[str] = []
    for node_id in ids:
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append(node_id)
    return out


def _swarm_node_last_activity(row: sqlite3.Row) -> Optional[int]:
    """Heartbeat if present, else the task's start/create timestamp.

    There is no ``updated_at`` column; ``started_at`` is the status-change
    stamp for a running attempt and ``created_at`` is the fallback when the
    node has never started.
    """
    heartbeat = row["last_heartbeat_at"]
    if heartbeat is not None:
        return int(heartbeat)
    started = row["started_at"]
    if started is not None:
        return int(started)
    created = row["created_at"]
    if created is not None:
        return int(created)
    return None


def _swarm_node_runtime_budget(row: sqlite3.Row) -> int:
    """Per-node runtime cap actually recorded on the task, else the lane default."""
    raw = row["max_runtime_seconds"] if "max_runtime_seconds" in row.keys() else None
    if raw is not None:
        try:
            budget = int(raw)
        except (TypeError, ValueError):
            budget = 0
        if budget >= 1:
            return budget
    from hermes_cli.kanban_swarm import (
        _default_worker_max_runtime_seconds,
        extract_contract,
    )

    lane_id = None
    body = row["body"] if "body" in row.keys() else None
    contract = extract_contract(body) if body else None
    if isinstance(contract, dict):
        raw_lane = contract.get("expected_lane_id")
        if raw_lane is not None:
            lane_id = str(raw_lane).strip() or None
    return int(_default_worker_max_runtime_seconds(lane_id))


def _supervise_swarm_stalls(
    conn: sqlite3.Connection,
    *,
    now: Optional[int] = None,
) -> int:
    """Record stall diagnostics for overdue non-terminal swarm nodes.

    Read-only on task status: always calls ``record_swarm_stall_diagnostic``
    with ``block=False``. Returns the number of diagnostic calls made
    (including no-op dedup hits).
    """
    from hermes_cli import kanban_db as kb

    now_ts = int(now if now is not None else time.time())
    recorded = 0
    for swarm in kb.find_active_swarm_topologies(conn):
        node_ids = _swarm_topology_node_ids(swarm)
        if not node_ids:
            continue
        placeholders = ",".join("?" * len(node_ids))
        rows = conn.execute(
            f"SELECT id, status, last_heartbeat_at, started_at, created_at, "
            f"max_runtime_seconds, body FROM tasks WHERE id IN ({placeholders})",
            tuple(node_ids),
        ).fetchall()
        root_id = str(swarm.get("root_id") or "")
        for row in rows:
            if row["status"] in kb.TERMINAL_STATUSES:
                continue
            last_activity = _swarm_node_last_activity(row)
            if last_activity is None:
                continue
            elapsed = now_ts - last_activity
            budget = _swarm_node_runtime_budget(row)
            if elapsed < budget:
                continue
            node_id = str(row["id"])
            stall_key = f"swarm-stall:{root_id}:{node_id}:{last_activity}"
            reason = (
                f"Swarm node {node_id} in swarm {root_id} has no fresh "
                f"heartbeat/activity for {elapsed}s (budget {budget}s)"
            )
            kb.record_swarm_stall_diagnostic(
                conn,
                node_id,
                reason=reason,
                stall_key=stall_key,
                source=_SWARM_SUPERVISOR_SOURCE,
                block=False,
            )
            recorded += 1
            logger.debug(
                "kanban swarm supervisor: stall on %s swarm=%s elapsed=%ss budget=%ss",
                node_id,
                root_id,
                elapsed,
                budget,
            )
    return recorded


class GatewayKanbanWatchersMixin:
    """Kanban watcher / notifier / dispatcher loops for GatewayRunner."""

    def _owns_kanban_dispatcher_lock(self) -> bool:
        return getattr(self, "_kanban_dispatcher_lock_handle", None) is not None

    def _release_kanban_dispatcher_lock(self) -> None:
        """Clear notifier-visible ownership before releasing the OS lock."""
        handle = getattr(self, "_kanban_dispatcher_lock_handle", None)
        self._kanban_dispatcher_lock_handle = None
        _release_singleton_lock(handle)

    async def _sleep_between_ticks(self, interval: float) -> None:
        """Sleep *interval* (floored to 1s) in 1s slices so stop() never waits a full interval."""
        interval = max(interval, 1.0)
        slept = 0.0
        while slept < interval and self._running:
            await asyncio.sleep(min(1.0, interval - slept))
            slept += 1.0

    async def _kanban_notifier_watcher(self, interval: float = 5.0) -> None:
        """Poll ``kanban_notify_subs`` and deliver terminal events to users.

        Per subscription, claims ``task_events`` newer than the stored cursor
        (kinds in TERMINAL_KINDS), sends one message per event, then advances
        the cursor. The subscription is removed only when the task is
        ``archived``: ``done`` is reversible, so the cursor — not unsubscribing
        — is the dedup mechanism (unsub-on-terminal dropped users when the
        dispatcher respawned a crashed task). All SQLite work runs in a thread;
        one tick's failure never stops the next.
        For each subscription row, fetches ``task_events`` newer than the
        stored cursor with kind in the terminal set (``completed``,
        ``blocked``, ``gave_up``, ``crashed``, ``timed_out``,
        ``worker_excused_needs_input``, ``review_requested``,
        ``block_loop_detected``). Sends one
        message per new event to ``(platform, chat_id, thread_id)``,
        then advances the cursor. The subscription is removed only when the
        task is ``archived``. A ``done`` task can be reopened for review or
        continuation, so its subscription and origin-session ownership must
        survive completion. Cursor advancement prevents old events replaying
        when that happens.

        Runs in the gateway event loop; all SQLite work is pushed to a
        thread via ``asyncio.to_thread`` so the loop never blocks on the
        WAL lock. Failures in one tick don't stop subsequent ticks.

        **Multi-board:** iterates every board discovered on disk per
        tick. Each gateway polls only subscriptions owned by profiles whose
        adapters it hosts. The dispatch-owning gateway also handles legacy
        subscriptions without a profile stamp.
        """
        from gateway.config import Platform as _Platform
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban notifier: kanban_db not importable; notifier disabled")
            return

        sub_fail_counts: dict[tuple, int] = getattr(self, "_kanban_sub_fail_counts", {})
        self._kanban_sub_fail_counts = sub_fail_counts
        notifier_profile = getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name()
        self._kanban_notifier_profile = notifier_profile

        # Initial delay so the gateway can finish wiring adapters.
        await asyncio.sleep(5)

        # Stale done-sub GC: subs survive ``done``, so boards that never
        # archive would accumulate rows scanned every tick. One DELETE per
        # board, at startup (0 → first tick) and at most hourly.
        _gc_next_at = 0.0

        while self._running:
            try:
                _gc_due = time.monotonic() >= _gc_next_at
                _retention = 30
                if _gc_due:
                    _gc_next_at = time.monotonic() + _GC_INTERVAL_SECONDS
                    _retention = _gc_retention_days()

                deliveries = await asyncio.to_thread(
                    _notifier_collect, self, _kb,
                    notifier_profile=notifier_profile, gc_due=_gc_due, gc_retention_days=_retention,
                )
                for d in deliveries:
                    await _KanbanNotification(
                        self, d, platform_cls=_Platform, sub_fail_counts=sub_fail_counts,
                    ).deliver()
            except Exception as exc:
                logger.warning("kanban notifier tick failed: %s", exc)
            await self._sleep_between_ticks(interval)

    def _kanban_sub_op(self, board: Optional[str], op: str, sub: dict, **extra: Any) -> None:
        """Sync helper (runs in to_thread): call ``kanban_db_notify.<op>`` for one subscription on its board."""
        from hermes_cli import kanban_db_connect as _kbc
        from hermes_cli import kanban_db_notify as _kbn
        conn = _kbc.connect(board=board)
        try:
            getattr(_kbn, op)(
                conn, task_id=sub["task_id"], platform=sub["platform"], chat_id=sub["chat_id"],
                thread_id=sub.get("thread_id") or "", **extra,
            )
        finally:
            conn.close()

    def _kanban_advance(self, sub: dict, cursor: int, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "advance_notify_cursor", sub, new_cursor=cursor)

    def _kanban_unsub(self, sub: dict, board: Optional[str] = None) -> None:
        self._kanban_sub_op(board, "remove_notify_sub", sub)

    def _kanban_rewind(self, sub: dict, claimed_cursor: int, old_cursor: int, board: Optional[str] = None) -> None:
        """Undo a claimed notification cursor after send failure."""
        self._kanban_sub_op(board, "rewind_notify_cursor", sub, claimed_cursor=claimed_cursor, old_cursor=old_cursor)

    async def _deliver_kanban_artifacts(self, *, adapter, chat_id: str, metadata: dict, event_payload: Optional[dict], task) -> None:
        """Upload artifact files referenced by a completed kanban task.

        Sources, in priority order: ``event_payload['artifacts']``,
        ``event_payload['summary']``, then ``task.result`` (legacy). Paths are
        deduplicated, missing files are skipped (may be mentioned for
        reference only), and upload errors are logged, never raised.
        """
        raw_paths: list[str] = []
        if isinstance(event_payload, dict):
            raw = event_payload.get("artifacts")
            if isinstance(raw, (list, tuple)):
                raw_paths += [item for item in raw if isinstance(item, str)]
            summary = event_payload.get("summary")
            if isinstance(summary, str) and summary:
                raw_paths += adapter.extract_local_files(summary)[0]
        if task is not None and getattr(task, "result", None):
            raw_paths += adapter.extract_local_files(str(task.result))[0]
        candidates: list[str] = []
        for path in raw_paths:
            expanded = os.path.expanduser(path) if path else ""
            if expanded and expanded not in candidates and os.path.isfile(expanded):
                candidates.append(expanded)
        if not candidates:
            return

        from gateway.platforms.base import BasePlatformAdapter
        candidates = BasePlatformAdapter.filter_local_delivery_paths(candidates)
        if not candidates:
            return

        from urllib.parse import quote as _quote

        # Images ride one send_multiple_images call (batch uploads on Signal/Slack).
        image_paths = [p for p in candidates if Path(p).suffix.lower() in _IMAGE_EXTS]
        other_paths = [p for p in candidates if Path(p).suffix.lower() not in _IMAGE_EXTS]
        if image_paths:
            try:
                batch = [(f"file://{_quote(p)}", "") for p in image_paths]
                await adapter.send_multiple_images(chat_id=chat_id, images=batch, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: image batch upload failed: %s", exc)
        for path in other_paths:
            try:
                if Path(path).suffix.lower() in _VIDEO_EXTS:
                    await adapter.send_video(chat_id=chat_id, video_path=path, metadata=metadata)
                else:
                    await adapter.send_document(chat_id=chat_id, file_path=path, metadata=metadata)
            except Exception as exc:
                logger.warning("kanban notifier: artifact upload (%s) failed: %s", path, exc)

    def _kanban_dispatcher_boot(self) -> Optional[tuple]:
        """Resolve config, kanban_db and the singleton lock; None when the dispatcher must not run.

        Config is read once at boot (restart to apply), except the auto-decompose
        toggle which is re-read every tick. The env var is an escape hatch to
        disable without editing YAML.
        """
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban dispatcher: config loader unavailable; disabled")
            return None
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info("kanban dispatcher: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env")
            return None
        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning("kanban dispatcher: cannot load config (%s); disabled", exc)
            return None
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info("kanban dispatcher: disabled via config kanban.dispatch_in_gateway=false")
            return None
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban dispatcher: kanban_db not importable; dispatcher disabled")
            return None

        # Single-dispatcher backstop (see _acquire_singleton_lock). The lock
        # lives at the machine-global kanban root, so it serialises ALL gateways.
        self._kanban_dispatcher_lock_handle = None
        _lock_path = _kb.kanban_home() / "kanban" / ".dispatcher.lock"
        _lock_handle, _lock_state = _acquire_singleton_lock(_lock_path)
        if _lock_state == "contended":
            logger.info("kanban dispatcher: another gateway already holds the dispatcher "
                        "lock (%s); this gateway will NOT dispatch.", _lock_path)
            return None
        if _lock_state == "held":
            self._kanban_dispatcher_lock_handle = _lock_handle  # hold for process lifetime
            logger.info("kanban dispatcher: holding singleton dispatcher lock (%s)", _lock_path)
        else:
            logger.warning("kanban dispatcher: advisory lock unavailable at %s; proceeding "
                           "on config control alone.", _lock_path)
        return _load_config, _kb, kanban_cfg

    async def _kanban_dispatcher_watcher(self) -> None:
        """Embedded kanban dispatcher — one tick every `dispatch_interval_seconds`.

        Gated by `kanban.dispatch_in_gateway` (default True); when false the
        loop exits and an external `hermes kanban daemon` is expected. Each
        tick runs :func:`kanban_db_dispatch.dispatch_once` in a thread; one tick's
        failure never stops the next. Shutdown: ``self._running`` is checked
        between ticks and the in-flight ``to_thread`` returns on its own.
        """
        boot = self._kanban_dispatcher_boot()
        if boot is None:
            return
        _load_config, _kb, kanban_cfg = boot
        settings = _resolve_dispatcher_settings(kanban_cfg, _kb)
        interval = settings.interval

        # Initial delay so adapters are wired before workers spawn (matches the notifier).
        await asyncio.sleep(5)

        # Health telemetry (mirrors `_cmd_daemon`): warn when the ready queue
        # is non-empty but spawns are 0 for N consecutive ticks — usually a
        # broken PATH, missing venv, or credential loss.
        bad_ticks = 0
        last_warn_at = 0
        dispatcher = _KanbanDispatcher(_kb, settings)

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
                out.append((slug, dispatcher.tick_once_for_board(slug)))
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
            # Only probe the review column when autonomous review dispatch is
            # actually on. With ``review_dispatch`` off (the default — no
            # sdlc-review agent), a task parked in 'review' is "correctly idle"
            # waiting for a human, not a stuck dispatcher; probing it here would
            # fire a false "dispatcher stuck" warning that never clears. Shares
            # the exact gate the dispatcher uses so the two can't drift.
            _review_probe = _kb.review_dispatch_enabled()
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
                    if _review_probe and _kb.has_spawnable_review(conn):
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
        tick_number = 0
        while self._running:
            tick_number += 1
            tick_started = time.monotonic()
            tick_spawned = 0
            tick_reclaimed = 0
            tick_skipped_nonspawnable = 0
            tick_ready_spawnable = False
            tick_paused = False
            try:
                # Reap zombies before per-board work so a board DB failure
                # cannot block cleanup of unrelated workers.
                from hermes_cli import kanban_db_dispatch as _kbd
                pids = await _to_thread_process_service(_kbd.reap_worker_zombies)
                if pids:
                    logger.info("kanban dispatcher: reaped %d zombie worker(s), pids=%s", len(pids), pids)
            except Exception:
                logger.exception("kanban dispatcher: zombie reaper failed")

            try:
                # Emergency stop (`hermes pause`): no auto-decompose or
                # dispatch while paused; running workers finish naturally.
                if not _kanban_dispatch_allowed():
                    bad_ticks = 0
                    tick_paused = True
                else:
                    # Re-read the auto-decompose toggle live so disabling it
                    # takes effect on the next tick, not on restart.
                    _ad_enabled, _ad_per_tick = _resolve_auto_decompose_settings(_load_config)
                    # See #49638.
                    if _ad_enabled:
                        await _to_thread_process_service(_auto_decompose_tick, _ad_per_tick)
                    results = await _to_thread_process_service(_tick_once)
                    any_spawned = False
                    for slug, res in (results or []):
                        if res is not None:
                            tick_spawned += len(getattr(res, "spawned", ()) or ())
                            tick_reclaimed += int(getattr(res, "reclaimed", 0) or 0)
                            tick_skipped_nonspawnable += len(
                                getattr(res, "skipped_nonspawnable", ()) or ()
                            )
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
                    ready_pending = await _to_thread_process_service(_ready_nonempty)
                    tick_ready_spawnable = bool(ready_pending)
                    if ready_pending and not any_spawned:
                        bad_ticks += 1
                    else:
                        bad_ticks = 0
                if bad_ticks >= _HEALTH_WINDOW:
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
                self._release_kanban_dispatcher_lock()
                raise
            except Exception:
                logger.exception("kanban dispatcher: unexpected watcher error")

            # Keep a bounded, once-per-tick heartbeat.  Previously this loop
            # only logged when it spawned/reaped something, so an entirely
            # non-spawnable ready queue looked exactly like a dead dispatcher.
            # The supervisor still handles task death; this line makes a live
            # but intentionally idle/filtered dispatcher observable.
            logger.info(
                "kanban dispatcher heartbeat: tick=%d elapsed=%.2fs paused=%s "
                "ready_spawnable=%s spawned=%d reclaimed=%d "
                "skipped_nonspawnable=%d",
                tick_number,
                time.monotonic() - tick_started,
                tick_paused,
                tick_ready_spawnable,
                tick_spawned,
                tick_reclaimed,
                tick_skipped_nonspawnable,
            )

            # Sleep in 1s slices so shutdown is snappy — otherwise a stop()
            # waits up to `interval` seconds for the current sleep to finish.
            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0

        self._release_kanban_dispatcher_lock()

    def _kanban_swarm_supervisor_tick(self, *, now: Optional[int] = None) -> None:
        """One read-only stall-detection pass over every board.

        No-ops unless this process currently holds the dispatcher lock.
        Never mutates task status.
        """
        if not self._owns_kanban_dispatcher_lock():
            return
        try:
            from hermes_cli import kanban_db as _kb
        except Exception:
            logger.warning("kanban swarm supervisor: kanban_db not importable; skip tick")
            return
        try:
            boards = _kb.list_boards(include_archived=False)
        except Exception:
            boards = [_kb.read_board_metadata(_kb.DEFAULT_BOARD)]
        for board in boards:
            slug = board.get("slug") or _kb.DEFAULT_BOARD
            conn = None
            try:
                conn = _kb.connect(board=slug)
                _supervise_swarm_stalls(conn, now=now)
            except Exception:
                logger.exception(
                    "kanban swarm supervisor: tick failed on board %s", slug,
                )
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

    async def _kanban_swarm_supervisor_watcher(self) -> None:
        """Periodic read-only observer of active swarm node liveness.

        Shares the dispatcher singleton lock: only the process that owns
        ``.dispatcher.lock`` runs the stall scan. Each tick records
        ``record_swarm_stall_diagnostic(..., block=False)`` for nodes whose
        heartbeat/activity is older than the node's own
        ``max_runtime_seconds`` (lane default if unset). Task status is
        never mutated. Failures in one tick do not stop subsequent ticks.
        """
        try:
            from hermes_cli.config import load_config as _load_config
        except Exception:
            logger.warning("kanban swarm supervisor: config loader unavailable; disabled")
            return
        env_override = os.environ.get("HERMES_KANBAN_DISPATCH_IN_GATEWAY", "").strip().lower()
        if env_override in {"0", "false", "no", "off"}:
            logger.info(
                "kanban swarm supervisor: disabled via HERMES_KANBAN_DISPATCH_IN_GATEWAY env"
            )
            return

        try:
            cfg = _load_config()
        except Exception as exc:
            logger.warning(
                "kanban swarm supervisor: cannot load config (%s); disabled", exc,
            )
            return
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        if not kanban_cfg.get("dispatch_in_gateway", True):
            logger.info(
                "kanban swarm supervisor: disabled via config "
                "kanban.dispatch_in_gateway=false"
            )
            return

        try:
            from hermes_cli import kanban_db as _kb  # noqa: F401
        except Exception:
            logger.warning(
                "kanban swarm supervisor: kanban_db not importable; disabled"
            )
            return

        try:
            interval = float(
                kanban_cfg.get(
                    "swarm_supervisor_interval_seconds",
                    _SWARM_SUPERVISOR_DEFAULT_INTERVAL,
                )
                or _SWARM_SUPERVISOR_DEFAULT_INTERVAL
            )
        except (ValueError, TypeError):
            logger.warning(
                "kanban swarm supervisor: invalid swarm_supervisor_interval_seconds=%r, "
                "using default %.1f",
                kanban_cfg.get("swarm_supervisor_interval_seconds"),
                _SWARM_SUPERVISOR_DEFAULT_INTERVAL,
            )
            interval = _SWARM_SUPERVISOR_DEFAULT_INTERVAL
        interval = max(interval, _SWARM_SUPERVISOR_MIN_INTERVAL)

        logger.info(
            "kanban swarm supervisor: embedded in gateway (interval=%.1fs)", interval
        )
        await asyncio.sleep(5)

        while self._running:
            try:
                if self._owns_kanban_dispatcher_lock():
                    await _to_thread_process_service(self._kanban_swarm_supervisor_tick)
            except Exception:
                logger.exception("kanban swarm supervisor: unexpected watcher error")

            slept = 0.0
            while slept < interval and self._running:
                await asyncio.sleep(min(1.0, interval - slept))
                slept += 1.0
# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Callable  # noqa: F401,E402
from contextvars import Context  # noqa: F401,E402
import logging  # noqa: F401,E402
import re  # noqa: F401,E402
import sqlite3  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    't': ('agent.i18n', 't'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
