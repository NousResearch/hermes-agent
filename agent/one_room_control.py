"""One-room control tower: thin NL gate → Kanban (opt-in).

Salvages the PR #25429 control-plane *intent* (STOP / STATUS / WORKER / STEER)
but executes durable work through Hermes Kanban — not a parallel in-memory
worker registry or repo-root oneshot.

Hard rules:
- Opt-in only via ``orchestration.frontdesk_live_enabled`` in config.yaml
  (or an explicit session/owner attribute for tests). No user-facing
  ``HERMES_*`` behavior flags.
- Returned non-None results are *consumed*: surfaces must not enqueue or
  send them to the main model.
- NEW_TASK creates a Kanban row; the gateway dispatcher spawns workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from utils import is_truthy_value

__all__ = [
    "OneRoomResult",
    "one_room_control_enabled",
    "handle_one_room_control",
]


@dataclass(frozen=True, slots=True)
class OneRoomResult:
    """Local control-plane response. Surfaces deliver ``message`` as-is."""

    action: str  # stop | status | new_task | append | steered | ignored
    message: str
    task_id: str | None = None
    intent: str | None = None


def one_room_control_enabled(owner: Any = None, *, session: dict | None = None) -> bool:
    """Return whether the thin live gate is explicitly enabled."""
    for carrier in (session, owner):
        if not carrier:
            continue
        if isinstance(carrier, dict):
            if "frontdesk_live_enabled" in carrier:
                return bool(carrier.get("frontdesk_live_enabled"))
            if "one_room_control_enabled" in carrier:
                return bool(carrier.get("one_room_control_enabled"))
            cfg = carrier.get("config")
        else:
            for attr in (
                "frontdesk_live_enabled",
                "_frontdesk_live_enabled",
                "one_room_control_enabled",
                "_one_room_control_enabled",
            ):
                if hasattr(carrier, attr):
                    return bool(getattr(carrier, attr))
            cfg = getattr(carrier, "config", None)
        if isinstance(cfg, dict):
            orchestration = cfg.get("orchestration") or {}
            if isinstance(orchestration, dict):
                raw = orchestration.get("frontdesk_live_enabled")
                if raw is None:
                    raw = orchestration.get("one_room_control_enabled")
                if raw is not None:
                    return is_truthy_value(raw, default=False)
    # config.yaml default path
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        orchestration = (cfg or {}).get("orchestration") or {}
        if isinstance(orchestration, dict):
            raw = orchestration.get("frontdesk_live_enabled")
            if raw is None:
                raw = orchestration.get("one_room_control_enabled")
            if raw is not None:
                return is_truthy_value(raw, default=False)
    except Exception:
        pass
    return False


def _format_kanban_status(*, board: str | None = None, limit: int = 12) -> str:
    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=board)
    try:
        tasks = kanban_db.list_tasks(conn, limit=200)
    finally:
        conn.close()

    active_statuses = {"ready", "running", "todo", "blocked", "triage", "scheduled"}
    active = [t for t in tasks if str(getattr(t, "status", "") or "") in active_statuses]
    if not active:
        return "No active Kanban tasks on this board."

    order = {"running": 0, "ready": 1, "blocked": 2, "todo": 3, "triage": 4, "scheduled": 5}
    active.sort(
        key=lambda t: (
            order.get(str(getattr(t, "status", "")), 9),
            str(getattr(t, "id", "") or ""),
        )
    )

    lines = ["Kanban status (one-room control):"]
    for t in active[:limit]:
        tid = getattr(t, "id", None) or "?"
        st = getattr(t, "status", None) or "?"
        title = (getattr(t, "title", None) or "").strip() or "(untitled)"
        assignee = getattr(t, "assignee", None) or "-"
        lines.append(f"- {tid}  [{st}]  {assignee}  {title}")
    if len(active) > limit:
        lines.append(f"... +{len(active) - limit} more")
    lines.append("Tip: /kanban show <id> · /kanban list")
    return "\n".join(lines)


def _create_kanban_task(
    title: str,
    body: str,
    *,
    board: str | None = None,
    assignee: str | None = None,
    created_by: str | None = None,
    session_id: str | None = None,
) -> str:
    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=board)
    try:
        task_id = kanban_db.create_task(
            conn,
            title=title[:200],
            body=body,
            assignee=assignee or "default",
            created_by=created_by or "one-room-control",
            session_id=session_id,
            board=board,
            triage=False,
        )
        return task_id
    finally:
        conn.close()


def _append_kanban_comment(
    task_id: str,
    text: str,
    *,
    board: str | None = None,
    author: str | None = None,
) -> None:
    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=board)
    try:
        kanban_db.add_comment(
            conn,
            task_id,
            author or "one-room-control",
            text,
        )
    finally:
        conn.close()


def _find_active_task_id(*, board: str | None = None, session_id: str | None = None) -> str | None:
    from hermes_cli import kanban_db

    conn = kanban_db.connect(board=board)
    try:
        tasks = kanban_db.list_tasks(conn, limit=100)
    finally:
        conn.close()

    running = [t for t in tasks if str(getattr(t, "status", "")) == "running"]
    if session_id:
        for t in running:
            if str(getattr(t, "session_id", "") or "") == session_id:
                return str(t.id)
    if running:
        return str(running[0].id)
    ready = [t for t in tasks if str(getattr(t, "status", "")) == "ready"]
    if ready:
        return str(ready[0].id)
    return None


def _reclaim_running(*, board: str | None = None) -> list[str]:
    """Best-effort reclaim of running claims on the board."""
    from hermes_cli import kanban_db

    reclaimed: list[str] = []
    conn = kanban_db.connect(board=board)
    try:
        tasks = kanban_db.list_tasks(conn, status="running", limit=50)
        for t in tasks:
            tid = str(getattr(t, "id", "") or "")
            if not tid:
                continue
            try:
                if kanban_db.reclaim_task(conn, tid):
                    reclaimed.append(tid)
            except Exception:
                continue
    finally:
        conn.close()
    return reclaimed


def handle_one_room_control(
    text: Any,
    *,
    owner: Any = None,
    session: dict | None = None,
    session_key: str | None = None,
    board: str | None = None,
    main_in_flight: bool = False,
    active_task_id: str | None = None,
    cancel_callback: Callable[[str], Any] | None = None,
    steer_callback: Callable[[str], Any] | None = None,
    assignee: str | None = None,
) -> OneRoomResult | None:
    """Classify and act. ``None`` → caller continues normal main-model path."""
    if not one_room_control_enabled(owner, session=session):
        return None
    if not isinstance(text, str) or not text.strip():
        return None

    from agent.control_plane import Intent, classify

    decision = classify(text, frontdesk_mode_active=True)
    intent = decision.intent

    # --- STOP ---
    if intent is Intent.STOP:
        reclaimed = []
        try:
            reclaimed = _reclaim_running(board=board)
        except Exception:
            reclaimed = []
        if cancel_callback is not None:
            try:
                cancel_callback(text)
            except Exception:
                pass
        extra = f" Reclaimed: {', '.join(reclaimed)}." if reclaimed else ""
        return OneRoomResult(
            action="stop",
            message=f"Stopped. Control-plane cancel consumed (not queued as user text).{extra}",
            intent=intent.value,
        )

    # --- STATUS ---
    if intent is Intent.STATUS:
        try:
            msg = _format_kanban_status(board=board)
        except Exception as exc:
            msg = f"Status unavailable: {exc}"
        return OneRoomResult(action="status", message=msg, intent=intent.value)

    # --- STEER / APPEND when main in flight or active kanban task ---
    if intent is Intent.STEER or (
        main_in_flight
        and intent is Intent.NEW_TASK_MAIN
        and len(text.strip()) < 200
        and not decision.should_delegate
    ):
        # Prefer explicit steer into running main agent when callback works
        if main_in_flight and steer_callback is not None and intent is Intent.STEER:
            try:
                ok = steer_callback(text)
                if ok:
                    return OneRoomResult(
                        action="steered",
                        message="Steered into the active main run.",
                        intent=Intent.STEER.value,
                    )
            except Exception:
                pass

        tid = active_task_id or _find_active_task_id(board=board, session_id=session_key)
        if tid:
            try:
                _append_kanban_comment(tid, text, board=board, author=session_key or "user")
                return OneRoomResult(
                    action="append",
                    message=f"Appended follow-up to active task `{tid}`.",
                    task_id=tid,
                    intent=intent.value,
                )
            except Exception as exc:
                return OneRoomResult(
                    action="append",
                    message=f"Could not append to task: {exc}",
                    task_id=tid,
                    intent=intent.value,
                )
        # No target — fall through to main if not hard STEER
        if intent is not Intent.STEER:
            return None

    # --- NEW_TASK_WORKER (and explicit worker-shaped) ---
    if intent is Intent.NEW_TASK_WORKER or decision.should_delegate:
        title = text.strip().splitlines()[0][:120]
        try:
            task_id = _create_kanban_task(
                title=title,
                body=text.strip(),
                board=board,
                assignee=assignee,
                session_id=session_key,
            )
        except Exception as exc:
            return OneRoomResult(
                action="new_task",
                message=f"Failed to create Kanban task: {exc}",
                intent=intent.value,
            )
        return OneRoomResult(
            action="new_task",
            message=(
                f"Queued as Kanban task `{task_id}`.\n"
                f"Dispatcher will pick it up; ask “지금 뭐 하고 있어?” for status."
            ),
            task_id=task_id,
            intent=intent.value,
        )

    # ACK / NOISE / DUPLICATE → soft ignore with optional chrome
    if intent in {Intent.ACK, Intent.NOISE, Intent.DUPLICATE}:
        return OneRoomResult(
            action="ignored",
            message=f"control: {intent.value}",
            intent=intent.value,
        )

    # NEW_TASK_MAIN / default → main agent
    return None
