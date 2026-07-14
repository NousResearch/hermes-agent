"""Product name: Concierge (only).

Concierge mode: thin NL gate → Kanban (opt-in).

Product name: **Concierge** (user-facing). Historical aliases:

Hard rules:
- Opt-in only via ``orchestration.concierge_live_enabled`` in config.yaml

  No user-facing ``HERMES_*`` behavior flags.
- Returned non-None results are *consumed*: surfaces must not enqueue or
  send them to the main model.
- NEW_TASK creates a Kanban row; the gateway dispatcher spawns workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from utils import is_truthy_value

__all__ = [
    "ConciergeResult",
    "concierge_enabled",
    "handle_concierge",
    # temporary aliases during rename
]

# Session/owner/config keys that mean "concierge mode on", in priority order.
_ENABLE_KEYS = (
    "concierge_live_enabled",
)
_ENABLE_ATTRS = (
    "concierge_live_enabled",
    "_concierge_live_enabled",
)


@dataclass(frozen=True, slots=True)
class ConciergeResult:
    """Local control-plane response. Surfaces deliver ``message`` as-is."""

    action: str  # stop | status | new_task | append | steered | ignored
    message: str
    task_id: str | None = None
    intent: str | None = None


# Back-compat alias for imports/tests mid-rename.


def concierge_enabled(owner: Any = None, *, session: dict | None = None) -> bool:
    """Return whether Concierge mode is explicitly enabled."""
    for carrier in (session, owner):
        if not carrier:
            continue
        if isinstance(carrier, dict):
            for key in _ENABLE_KEYS:
                if key in carrier:
                    return bool(carrier.get(key))
            cfg = carrier.get("config")
        else:
            for attr in _ENABLE_ATTRS:
                if hasattr(carrier, attr):
                    return bool(getattr(carrier, attr))
            cfg = getattr(carrier, "config", None)
        if isinstance(cfg, dict):
            orchestration = cfg.get("orchestration") or {}
            if isinstance(orchestration, dict):
                for key in _ENABLE_KEYS:
                    raw = orchestration.get(key)
                    if raw is not None:
                        return is_truthy_value(raw, default=False)
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        orchestration = (cfg or {}).get("orchestration") or {}
        if isinstance(orchestration, dict):
            for key in _ENABLE_KEYS:
                raw = orchestration.get(key)
                if raw is not None:
                    return is_truthy_value(raw, default=False)
    except Exception:
        pass
    return False


# Back-compat


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

    lines = ["Kanban status (concierge):"]
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
            created_by=created_by or "concierge",
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
            author or "concierge",
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


def handle_concierge(
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
) -> ConciergeResult | None:
    """Classify and act. ``None`` → caller continues normal main-model path."""
    if not concierge_enabled(owner, session=session):
        return None
    if not isinstance(text, str) or not text.strip():
        return None

    from agent.control_plane import Intent, classify

    # Mode-active only when the product flag is on; still use mode=True for
    # classification of STOP/STATUS so those control intents stay sharp.
    decision = classify(text, concierge_mode_active=True)
    intent = decision.intent

    # Contract: no Ctrl+F interception of free text.
    # Only whole-body STOP / whole-body STATUS are consumed locally.
    # Everything else (including "진행해", long instructions, GitHub URLs)
    # falls through to the main model to *understand* then act.

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
        return ConciergeResult(
            action="stop",
            message=f"Stopped. Concierge cancel consumed (not queued as user text).{extra}",
            intent=intent.value,
        )

    if intent is Intent.STATUS:
        try:
            msg = _format_kanban_status(board=board)
        except Exception as exc:
            msg = f"Status unavailable: {exc}"
        return ConciergeResult(action="status", message=msg, intent=intent.value)

    if intent in {Intent.ACK, Intent.NOISE, Intent.DUPLICATE}:
        return ConciergeResult(
            action="ignored",
            message=f"control: {intent.value}",
            intent=intent.value,
        )

    # MAIN / STEER / NEW_TASK_* → main agent understands and proceeds.
    return None


# Back-compat
