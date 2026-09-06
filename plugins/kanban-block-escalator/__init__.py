"""Kanban block -> OVERWATCH escalation plugin (rewritten 2026-09-06, plan O1).

Subscribes to ``kanban_task_blocked`` and, for every block that is a real
fault signal, spawns ONE fresh Agent Smith session ("overwatch", profile id
``default``) with a machine-built situation brief. Smith assesses the card in
the context of the whole board, acts within a written authority, and hands the
hard stops to Richie.

Richie's decision, 2026-09-06: "a fresh chat with Smith is triggered, he
assesses and acts to get things moving or tidy up the mess ... as quickly,
hygienically and cheaply as possible, while still delivering the feature to
the agreed quality, including all the agreed review and testing gates. Smith
may approve up to another $0.50 per card. At $1.50 the card is blocked and
Smith texts me a full rundown."

This replaced three mechanisms that disagreed with each other: a per-kind
routing table (Jobsy vs Smith vs Steve-o by substring), Steve-o cost
adjudication with two extensions, and ``dependency-gate-watch`` minting
``[triage]`` cards for give-ups. Measured on 5-6 Sep: 85 blocks, of which 15
were ``operator_hold`` by design and 28 dependency waits that needed nobody;
14 cost breaches each spawned ~4 more cards.

Triggers (read from the board, never from substrings of the reason text):
  cost_cap, capability, needs_input          -> always
  transient                                  -> on the SECOND occurrence
Never: operator_hold (Richie's decision), dependency (self-resumes),
scheduled (a date, not a fault).

Hard stops (no overwatch; ceiling marker for escalation-watch -> Richie):
  * the card already carries TWO overwatch decisions;
  * a cost_cap block on a card that was already extended once, or whose cap
    is already at kanban.max_cost_hard_ceiling.
On a hard stop ONE more Smith session is spawned with the RUNDOWN prompt:
write the rundown as a comment; escalation-watch (15 min, iMessage + Slack)
carries it to Richie.

Guardrails: never break the block transaction (every probe is best-effort);
fire-and-forget spawn in its own session; ``switch`` is never a target.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["register"]

# The overwatch profile. Pinned here so it is versioned and testable; tests
# assert it is never ``switch`` and never a worker.
OVERWATCH = "default"          # Agent Smith's profile id
DEFAULT_ASSESSOR = OVERWATCH   # kept for older tools that import the name
RUNTIME_ASSESSOR = OVERWATCH
COST_ASSESSOR = OVERWATCH      # Steve-o no longer adjudicates spend (2026-09-06)

ALWAYS_TRIGGER_KINDS = frozenset({"cost_cap", "capability", "needs_input"})
NEVER_TRIGGER_KINDS = frozenset({"operator_hold", "dependency", "scheduled"})
TRANSIENT_TRIGGER_AFTER = 1     # first transient retries; second triggers

OVERWATCH_LIMIT = 2             # overwatch touches per card before Richie
OVERWATCH_MARKER = "overwatch:"  # Smith's decision comment must start with this
EXTENSION_MARKER = "cost-extension:"
CEILING_MARKER = "escalation-ceiling"
RUNDOWN_MARKER = "rundown:"
NON_COST_TRIAGE_LIMIT = OVERWATCH_LIMIT  # name kept for escalation-watch

HARD_CEILING_FALLBACK = 1.50
BRIEF_DIR = Path(os.path.expanduser("~/.hermes/logs/overwatch"))


def _hermes_bin() -> str:
    return os.path.expanduser("~/.hermes/hermes-agent/venv/bin/hermes")


def _board_db_path() -> str:
    return os.environ.get("HERMES_KANBAN_DB") or os.path.expanduser("~/.hermes/kanban.db")


def _ro():
    con = sqlite3.connect(f"file:{_board_db_path()}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def _card(task_id: str) -> dict | None:
    try:
        con = _ro()
        try:
            row = con.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        finally:
            con.close()
        return dict(row) if row else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("kanban-block-escalator: card probe failed: %s", exc)
        return None


def _is_truly_blocked(card: dict | None) -> bool:
    return bool(card) and card.get("status") in ("blocked", "triage")


def _count_comments(task_id: str, prefix: str, author: str | None = None) -> int:
    try:
        con = _ro()
        try:
            q = "SELECT COUNT(*) FROM task_comments WHERE task_id = ? AND body LIKE ?"
            p = [task_id, prefix + "%"]
            if author:
                q += " AND author = ?"
                p.append(author)
            return int(con.execute(q, p).fetchone()[0])
        finally:
            con.close()
    except Exception:  # noqa: BLE001
        return 0


def _hard_ceiling() -> float:
    try:
        from hermes_cli import kanban_db  # type: ignore

        return float(kanban_db.resolve_max_cost_hard_ceiling())
    except Exception:  # noqa: BLE001
        return HARD_CEILING_FALLBACK


def _recurrences(task_id: str) -> int:
    card = _card(task_id)
    try:
        return int((card or {}).get("block_recurrences") or 0)
    except Exception:  # noqa: BLE001
        return 0


def should_trigger(card: dict) -> tuple[bool, str]:
    """(trigger?, why). Pure function of the card row — unit-tested."""
    kind = (card.get("block_kind") or "").strip()
    rec = int(card.get("block_recurrences") or 0)
    if kind in NEVER_TRIGGER_KINDS:
        return False, f"{kind or 'unkinded'}: not a fault signal"
    if kind in ALWAYS_TRIGGER_KINDS:
        return True, kind
    if kind == "transient":
        return (rec >= TRANSIENT_TRIGGER_AFTER), f"transient recurrence {rec}"
    # Unkinded / unknown kinds: a block is a block — but only once it repeats.
    return (rec >= 1), f"{kind or 'unkinded'} recurrence {rec}"


def is_hard_stop(task_id: str, card: dict) -> tuple[bool, str]:
    """Second overwatch on one card, or a cost breach past the extension."""
    n = _count_comments(task_id, OVERWATCH_MARKER, author=OVERWATCH)
    if n >= OVERWATCH_LIMIT:
        return True, f"{n} overwatch decisions already on this card"
    if (card.get("block_kind") or "") == "cost_cap":
        if _count_comments(task_id, EXTENSION_MARKER) >= 1:
            return True, "cost cap breached again after the one allowed extension"
        try:
            cap = float(card.get("max_cost") or 0)
        except Exception:  # noqa: BLE001
            cap = 0.0
        if cap and cap >= _hard_ceiling() - 1e-9:
            return True, f"cap ${cap:.2f} is already at the hard ceiling"
    return False, ""


def _brief(task_id: str, card: dict) -> tuple[str, str]:
    """Assemble the situation brief ($0). Returns (path, short summary)."""
    lines = [f"# Overwatch brief — {task_id}", ""]
    try:
        con = _ro()
        try:
            lines += [f"title: {card.get('title')}",
                      f"status: {card.get('status')}  block_kind: {card.get('block_kind')}  recurrences: {card.get('block_recurrences')}",
                      f"assignee: {card.get('assignee')}  created_by: {card.get('created_by')}  tenant: {card.get('tenant')}",
                      f"cap: {card.get('max_cost')}  workspace: {card.get('workspace_kind')} {card.get('workspace_path')}", ""]
            body = (card.get("body") or "").strip()
            lines += ["## body (head)", body[:1500], ""]
            parents = [r[0] for r in con.execute("SELECT parent_id FROM task_links WHERE child_id=?", (task_id,))]
            children = [r[0] for r in con.execute("SELECT child_id FROM task_links WHERE parent_id=?", (task_id,))]
            lines += [f"parents: {parents}", f"children: {children}"]
            sib = []
            for p in parents:
                for r in con.execute("SELECT t.id,t.status,t.block_kind,t.assignee,t.title FROM task_links l JOIN tasks t ON t.id=l.child_id WHERE l.parent_id=? AND t.id!=?", (p, task_id)):
                    sib.append(f"  {r[0]} {r[1]} {r[2] or ''} {r[3]} | {(r[4] or '')[:60]}")
            if sib:
                lines += ["siblings under the same parent:"] + sib
            lines += ["", "## runs"]
            for r in con.execute("SELECT profile,status,outcome,started_at,ended_at,substr(error,1,200) FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)):
                lines.append(f"  {r[0]} {r[1]} {r[2]} {r[3]}->{r[4]} {r[5] or ''}")
            lines += ["", "## last comments"]
            for r in con.execute("SELECT author,created_at,substr(body,1,500) FROM task_comments WHERE task_id=? ORDER BY created_at DESC LIMIT 8", (task_id,)):
                lines.append(f"  [{r[0]} @{r[1]}] {r[2]}")
            lines += ["", "## last events"]
            for r in con.execute("SELECT kind,substr(payload,1,160),created_at FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 10", (task_id,)):
                lines.append(f"  {r[2]} {r[0]} {r[1]}")
            dup = [r[0] for r in con.execute("SELECT id FROM tasks WHERE id!=? AND status NOT IN ('done','archived') AND title=?", (task_id, card.get("title")))]
            if dup:
                lines += ["", f"LIVE CARDS WITH THE SAME TITLE (possible duplicates): {dup}"]
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        lines += [f"(brief partly unavailable: {exc})"]
    try:
        from hermes_cli import kanban_db  # type: ignore

        own = kanban_db._cumulative_session_cost(
            kanban_db._state_db_path_for_assignee(card.get("assignee")), card.get("workspace_path"),
            task_id=task_id, all_ledgers=False)
        life = kanban_db._cumulative_session_cost(None, card.get("workspace_path"), task_id=task_id, all_ledgers=True)
        lines += ["", f"spend: assignee ${own:.2f} / lifetime all ledgers ${life:.2f} / cap {card.get('max_cost')}"]
    except Exception:  # noqa: BLE001
        pass
    text = "\n".join(lines)
    path = ""
    try:
        BRIEF_DIR.mkdir(parents=True, exist_ok=True)
        p = BRIEF_DIR / f"{task_id}-{int(time.time())}.md"
        p.write_text(text)
        path = str(p)
    except Exception:  # noqa: BLE001
        pass
    return path, text[:6000]


def _mark_ceiling(task_id: str, reason: str | None, why: str) -> None:
    """Machine-readable comment so escalation-watch tells Richie. Best effort."""
    try:
        from hermes_cli import kanban_db  # type: ignore

        with kanban_db.connect_closing(_board_db_path()) as con:  # type: ignore[attr-defined]
            kanban_db.add_comment(
                con, task_id, "kanban-block-escalator",
                f"{CEILING_MARKER}: hard stop — {why}. Staying blocked for Richie. "
                f"Reason: {reason or '(none)'}",
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("kanban-block-escalator: could not mark ceiling on %s: %s", task_id, exc)


AUTHORITY = (
    "YOUR AUTHORITY (Richie, 2026-09-06). You MAY: comment, unblock, reassign, split the card "
    "via Jobsy, archive duplicates, rescope, put on hold, open a HELD platform card, and — for a "
    "cost_cap block — extend the cap ONCE by at most $0.50 with `hermes kanban set-cap <id> <cap> "
    "--reason ...` (hard ceiling $1.50). You may NOT, while cards are running: commit platform "
    "code, restart a gateway, edit a SOUL, waive a review or test gate, raise a cap past $1.50, "
    "or create a card assigned to yourself. A defective gate is a held platform card, not an "
    "exemption. Finish quickly, hygienically and cheaply, keeping every agreed review and test "
    "gate. Your FIRST comment on the card must start with `overwatch:` and state the decision "
    "and its evidence — that comment is how the board counts your interventions."
)


def _overwatch_prompt(task_id: str, card: dict, reason: str | None, why: str, brief_path: str, brief: str) -> str:
    return (
        f"OVERWATCH: kanban card {task_id} blocked ({why}). Block reason: {reason or '(none)'!r}.\n"
        f"Read the situation brief first ({brief_path or 'inline below'}); it already contains the card, "
        "its lineage, runs, comments, spend and any live duplicate. Ask FIRST whether this is a deeper "
        "error — duplicate card, wrong scope, dead or empty workspace, stale gate, false design premise, "
        "missing capability — and fix the cause rather than the symptom.\n\n"
        f"{AUTHORITY}\n\n=== BRIEF ===\n{brief}"
    )


def _rundown_prompt(task_id: str, reason: str | None, why: str, brief_path: str, brief: str) -> str:
    return (
        f"HARD STOP on kanban card {task_id}: {why}. Block reason: {reason or '(none)'!r}. "
        "Do NOT unblock, extend, split or archive it — this decision is Richie's. Write ONE comment "
        f"on the card starting with `{RUNDOWN_MARKER}` covering: (1) the situation, (2) what has been "
        "done about it so far and by whom, (3) whether there is a deeper issue and what it is, "
        "(4) the plan and estimated cost to remedy, with your recommendation. escalation-watch "
        "delivers it to Richie by iMessage and Slack. Keep it under 250 words.\n\n"
        f"Brief: {brief_path or 'inline below'}\n=== BRIEF ===\n{brief}"
    )


def _spawn(assessor: str, prompt: str, task_id: str) -> None:
    try:
        env = dict(os.environ)
        env["HERMES_OVERWATCH_TASK"] = task_id
        # 2026-09-07: this hook runs INSIDE the blocked worker's process, so
        # os.environ carries HERMES_KANBAN_TASK / HERMES_KANBAN_RUN_ID. Passing
        # them on made the overwatch assessor look like a kanban worker to
        # agent/kanban_checkpoint.py: it received the per-turn `[checkpoint]`
        # reminder and the forced terminal-only finalize turn, and was steered
        # toward a board call it must never make (overwatch assesses; it does
        # not close cards). Observed live on t_125dfa35 run 1143, where the
        # assessor reported "the checkpoint cut MY turn too".
        for _worker_var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID"):
            env.pop(_worker_var, None)
        subprocess.Popen(
            [_hermes_bin(), "-p", assessor, "--cli", "chat", "-q", prompt],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True, env=env,
        )
        logger.info("kanban-block-escalator: overwatch spawned for %s (%s)", task_id, assessor)
    except Exception as exc:  # noqa: BLE001
        logger.warning("kanban-block-escalator: failed to spawn overwatch for %s: %s", task_id, exc)


def on_block(task_id: str = "", assignee: str | None = None, reason: str | None = None, **kwargs) -> None:
    """kanban_task_blocked callback. Fire-and-forget overwatch trigger."""
    if not task_id:
        return
    card = _card(task_id)
    if not _is_truly_blocked(card):
        return
    # The hook may or may not carry the kind; the board always does.
    if kwargs.get("kind") and not card.get("block_kind"):
        card["block_kind"] = kwargs["kind"]
    trigger, why = should_trigger(card)
    if not trigger:
        logger.info("kanban-block-escalator: %s not escalated (%s)", task_id, why)
        return
    hard, hard_why = is_hard_stop(task_id, card)
    brief_path, brief = _brief(task_id, card)
    if hard:
        _mark_ceiling(task_id, reason, hard_why)
        _spawn(OVERWATCH, _rundown_prompt(task_id, reason, hard_why, brief_path, brief), task_id)
        return
    _spawn(OVERWATCH, _overwatch_prompt(task_id, card, reason, why, brief_path, brief), task_id)


def register(ctx) -> None:
    """Register the kanban_task_blocked lifecycle hook."""
    ctx.register_hook("kanban_task_blocked", on_block)
