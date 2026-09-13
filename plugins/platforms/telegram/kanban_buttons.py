"""Press side of the Kanban blocker-ask buttons (the ``kb:`` callback namespace).

The ask is sent by the homelab watcher (``homelab/scripts/alerts/kanban_block_watch.py``)
through ``send_alert(..., buttons=...)``; the Telegram sink attaches the inline keyboard.
Presses come back into this bot, so the press handler lives next to the other button
namespaces (``ea:``, ``sc:``, ``cl:``) in ``adapter.py`` — this module holds the board
side of it, keeping the adapter a thin (authorize, answer, edit) shell.

``callback_data`` carries only ``kb:<task_id>:<index>`` (Telegram caps it at 64 bytes),
so the chosen choice is resolved back from the card's own stored options — the same data
the keyboard was rendered from. Nothing is guessed: an index that does not resolve to a
choice on that card is answered and dropped, never applied.

Index space
-----------
``0 .. KB_MAX_DECLARED-1``
    the card's task-declared options, in the order they were declared.
``STANDARD_BASE .. STANDARD_BASE+2``
    the always-present ``Do it`` / ``Park it`` / ``Drop it`` row.
``n .. n+2`` (``n`` = the number of declared options on the card)
    also read as that same standard row, so a renderer that appends the standard
    row straight after the declared options resolves to the same three verbs.
    That reading is only reachable once the declared-option reading has already
    said "no such option", so the two schemes never disagree.

Where the options come from
---------------------------
:func:`hermes_cli.kanban_db.block_options_for_task` — the block event payload's
structured ``options`` list when the blocking worker declared one, else the
documented exact-form ``OPTIONS: a=<label> | b=<label>`` line. That helper *is*
the parser; this module never scrapes prose itself, so a press can only ever
resolve to a choice the card really declared.

Verbs (identical to the chat loop and to ``hermes kanban``)
----------------------------------------------------------
* **Do it** — ``unblock`` a blocked/scheduled card. A card that is merely staged
  in ``review`` (the work is done and only the human decision is missing) is
  ``complete``\\ d instead.
* **Park it** — ``schedule`` with a reason (waiting on a real-world change).
* **Drop it** — ``complete`` with ``abandoned: <reason>``.
* **a task-declared option** — the operator's answer: ``unblock`` with the answer
  recorded, which is exactly what replying in chat does.

Every applied press first records ``USER CONFIRMED (<date>, telegram button):
<label>`` on the card — the audit trail a typed decision leaves behind. A press
on a card that already moved is answered and changes nothing: no comment, no verb.
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

# --- callback namespace + index space -------------------------------------

KB_PREFIX = "kb:"
KB_MAX_DECLARED = 4
STANDARD_BASE = 100
STANDARD_LABELS = {
    STANDARD_BASE: "Do it",
    STANDARD_BASE + 1: "Park it",
    STANDARD_BASE + 2: "Drop it",
}

VERB_DO_IT = "do_it"
VERB_PARK_IT = "park_it"
VERB_DROP_IT = "drop_it"
VERB_ANSWER = "answer"

_VERB_BY_BASE = dict(zip(STANDARD_LABELS, (VERB_DO_IT, VERB_PARK_IT, VERB_DROP_IT)))
_VERB_BY_OFFSET = {0: VERB_DO_IT, 1: VERB_PARK_IT, 2: VERB_DROP_IT}

# A verb only applies to the statuses it can actually move; any other status means the
# card already moved and the press must be answered without a board effect.
_VERB_STATUSES = {
    VERB_DO_IT: ("blocked", "scheduled", "review"),
    VERB_PARK_IT: ("todo", "ready", "running", "blocked"),
    VERB_DROP_IT: ("todo", "ready", "running", "blocked", "review"),
    VERB_ANSWER: ("blocked", "scheduled"),
}

# Block-shaped events carry the card's declared options (kanban_db owns the read).
_BLOCK_EVENT_KINDS = ("blocked", "block_loop_detected", "dependency_wait")

# Fallback parse, used only when kanban_db.parse_block_options is absent (older tree):
# one exact-form line, deliberately not a prose scraper.
_OPTIONS_LINE_RE = re.compile(r"^OPTIONS:[ \t]*(?P<body>\S.*?)[ \t]*$", re.MULTILINE)
_OPTION_PART_RE = re.compile(r"^(?P<key>[A-Za-z0-9]{1,4})[ \t]*=[ \t]*(?P<value>.+?)[ \t]*$")


@dataclass
class PressOutcome:
    """What a press did, in the two shapes the adapter needs to render."""

    applied: bool
    label: str
    verb: Optional[str]
    task_id: Optional[str]
    board: Optional[str]
    status: Optional[str]
    answer_text: str
    edit_text: Optional[str] = None


def parse_callback_data(callback_data: Any) -> Optional[tuple[str, int]]:
    """``kb:<task_id>:<index>`` -> ``(task_id, index)``; None when malformed."""
    parts = str(callback_data or "").split(":")
    if len(parts) != 3 or parts[0] != KB_PREFIX.rstrip(":"):
        return None
    task_id = parts[1].strip()
    if not task_id:
        return None
    try:
        index = int(parts[2].strip())
    except (TypeError, ValueError):
        return None
    return task_id, index


def resolve_choice(options: Sequence[Mapping[str, Any]], index: int) -> Optional[dict[str, Any]]:
    """``{label, value, verb, declared}`` for a button index, or None (unknown index)."""
    if index in _VERB_BY_BASE:
        return {
            "label": STANDARD_LABELS[index], "value": "",
            "verb": _VERB_BY_BASE[index], "declared": False,
        }
    declared = _normalize_options(options)
    if 0 <= index < len(declared):
        option = declared[index]
        return {"label": option["label"], "value": option["value"], "verb": VERB_ANSWER, "declared": True}
    # Standard row appended straight after the declared options (no fixed base).
    offset = index - len(declared)
    if index >= len(declared) and offset in _VERB_BY_OFFSET:
        return {
            "label": STANDARD_LABELS[STANDARD_BASE + offset], "value": "",
            "verb": _VERB_BY_OFFSET[offset], "declared": False,
        }
    return None


def declared_options(payload: Optional[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Options carried by a block *payload*: structured ``options``, else the

    ``OPTIONS:`` line in its reason (parsed by ``kanban_db`` when available)."""
    payload = payload if isinstance(payload, Mapping) else {}
    for key in ("options", "kb_options"):
        options = _normalize_options(payload.get(key))
        if options:
            return options
    return _options_from_reason(str(payload.get("reason") or ""))


def apply_press(
    callback_data: Any, *, pressed_by: Optional[str] = None,
    author: Optional[str] = None, now: Optional[float] = None,
) -> PressOutcome:
    """An authorized button press -> board effect. Never raises.

    ``author`` is the comment author, ``pressed_by`` the display name rendered into
    the answer. The caller owns the authorization gate and the callback answer;
    this function only touches the board.
    """
    parsed = parse_callback_data(callback_data)
    if parsed is None:
        return PressOutcome(False, "", None, None, None, None, "Unrecognised task button.")
    task_id, index = parsed
    try:
        return _apply(task_id, index, pressed_by=pressed_by, author=author, now=now)
    except Exception as exc:  # pragma: no cover - defensive: a press must never 500
        logger.warning("kanban button press failed for %s: %s", task_id, exc, exc_info=True)
        return PressOutcome(
            False, "", None, task_id, None, None,
            "Could not apply that decision — check the card.", None)


# --- board side ------------------------------------------------------------

def _apply(task_id: str, index: int, *, pressed_by, author, now) -> PressOutcome:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    board, conn, task = _find_task(kb, kbc, task_id)
    if conn is None:
        return PressOutcome(False, "", None, task_id, None, None,
                            f"Card {task_id} is not on a board.", None)
    try:
        choice = resolve_choice(_card_options(kb, conn, task_id), index)
        if choice is None:
            return PressOutcome(False, "", None, task_id, board, task.status,
                                "This button no longer matches a choice on the card.", None)
        verb, label, status = choice["verb"], choice["label"], task.status
        byline = f" Clicked by {pressed_by}." if pressed_by else ""
        if status not in _VERB_STATUSES.get(verb, ()):
            # Stale press (second tap, or the ask answered elsewhere): answer gracefully,
            # no comment, no verb — re-applying would release a card that already moved.
            return PressOutcome(
                False, label, verb, task_id, board, status,
                f"{task_id} is already {status} — nothing applied.",
                f"⌛ {task_id} is already {status} — this ask was already answered.{byline}")
        # LOG first (the decision is the durable fact), then apply; the verb's own CAS is
        # the final authority and reports False when the card moved in between.
        body = f"USER CONFIRMED ({_stamp(now)}, telegram button): {label}"
        value = str(choice.get("value") or "").strip()
        if value and value != label:
            body += f"\nANSWER: {value}"
        kb.add_comment(conn, task_id, str(author or pressed_by or "operator"), body)
        applied, landed_note = _apply_verb(kb, conn, task_id, verb, label, status)
        landed = kb.get_task(conn, task_id)
        landed_status = landed.status if landed is not None else status
        if not applied:
            return PressOutcome(
                False, label, verb, task_id, board, landed_status,
                f"{task_id} moved already — nothing applied.",
                f"⌛ {task_id} is {landed_status} — this ask was already answered.{byline}")
        return PressOutcome(
            True, label, verb, task_id, board, landed_status,
            f"{label}: {task_id} {landed_status}",
            f"{_icon(verb)} {label} — {task_id} {landed_note} ({landed_status}).{byline}")
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def _icon(verb: str) -> str:
    return {VERB_PARK_IT: "⏸", VERB_DROP_IT: "🗑"}.get(verb, "✅")


def _apply_verb(kb, conn, task_id: str, verb: str, label: str, status: str) -> tuple[bool, str]:
    """Run the verb -> ``(applied, landed_note)``. Reasons match the chat loop's."""
    if verb == VERB_PARK_IT:
        return bool(kb.schedule_task(
            conn, task_id,
            reason=f"parked by the operator from the Telegram ask (button: {label})")), "parked"
    if verb == VERB_DROP_IT:
        return bool(kb.complete_task(
            conn, task_id,
            result=f"abandoned: operator dropped it from the Telegram ask (button: {label})")), \
            "closed as abandoned"
    if verb == VERB_DO_IT and status == "review":
        # Staged in review: nothing to unblock; the human decision IS the completion.
        return bool(kb.complete_task(
            conn, task_id,
            result=f"approved by the operator from the Telegram ask (button: {label})")), \
            "approved and closed"
    # Do it on a blocked/scheduled card, and a task-declared answer, both release the
    # card; the answer itself is already on the card as the USER CONFIRMED comment.
    return bool(kb.unblock_task(conn, task_id)), "released"


def _find_task(kb, kbc, task_id: str):
    """``(board_slug, conn, Task)`` for ``task_id`` — current board first, then the
    others. An unreadable/absent board is skipped, never created."""
    for slug in _candidate_boards(kb):
        conn = None
        try:
            conn = kbc.connect(board=slug)
            task = kb.get_task(conn, task_id)
        except Exception:
            logger.debug("kanban press: board %r unreadable", slug, exc_info=True)
            task = None
        if task is not None:
            return slug, conn, task
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    return None, None, None


def _candidate_boards(kb) -> list[Optional[str]]:
    """Boards worth looking in: the current one first, then every board with a DB."""
    if os.environ.get("HERMES_KANBAN_DB", "").strip():
        return [None]  # env pins one DB no matter which slug is asked for
    slugs: list[Optional[str]] = []
    with contextlib.suppress(Exception):
        current = kb.get_current_board()
        if current:
            slugs.append(current)
    with contextlib.suppress(Exception):
        for meta in kb.list_boards():
            slug = (meta or {}).get("slug")
            if slug and slug not in slugs:
                slugs.append(slug)
    return [slug for slug in slugs if _db_exists(kb, slug)]


def _db_exists(kb, slug: Optional[str]) -> bool:
    with contextlib.suppress(Exception):
        return kb.kanban_db_path(slug).exists()
    return False


def _card_options(kb, conn, task_id: str) -> list[dict[str, str]]:
    """The card's declared options, from the owner of that read: ``kanban_db``.

    ``block_options_for_task`` resolves both halves (the structured payload list,
    else the exact-form ``OPTIONS:`` line) and is what the ask renderer uses too,
    so a press can only resolve to a choice the card really declared.
    """
    reader = getattr(kb, "block_options_for_task", None)
    if callable(reader):
        try:
            return _normalize_options((reader(conn, task_id) or {}).get("options"))
        except Exception:
            logger.debug("kanban press: block_options_for_task failed for %s", task_id, exc_info=True)
    return declared_options(_latest_block_payload(kb, conn, task_id))


def _latest_block_payload(kb, conn, task_id: str) -> dict:
    """Payload of the newest block-shaped event on the card ({} when there is none)."""
    for event in reversed(kb.list_events(conn, task_id)):
        if event.kind in _BLOCK_EVENT_KINDS:
            return dict(event.payload) if isinstance(event.payload, Mapping) else {}
    return {}


def _normalize_options(raw: Any) -> list[dict[str, str]]:
    """``[{"label","value"}]`` from ``[{"label":...,"value":...}]`` or ``["…"]``."""
    if not isinstance(raw, (list, tuple)):
        return []
    options: list[dict[str, str]] = []
    for item in raw:
        if len(options) >= KB_MAX_DECLARED:
            break
        if isinstance(item, Mapping):
            label = str(item.get("label") or item.get("title") or item.get("text") or "").strip()
            value = str(item.get("value") or item.get("id") or "").strip()
        else:
            label, value = str(item or "").strip(), ""
        if not label:
            label = value
        if not label:
            continue
        options.append({"label": label, "value": value or label})
    return options


def _options_from_reason(reason: str) -> list[dict[str, str]]:
    """Exact-form ``OPTIONS: a=<text> | b=<text>`` only — never free prose.

    ``kanban_db.parse_block_options`` when the tree has it (the single source of
    truth for the recognised line), else the same rule applied locally.
    """
    try:
        from hermes_cli import kanban_db as kb
        parser = getattr(kb, "parse_block_options", None)
        if callable(parser):
            return _normalize_options(parser(reason))
    except Exception:
        logger.debug("kanban press: parse_block_options unavailable", exc_info=True)
    options: list[dict[str, str]] = []
    for match in _OPTIONS_LINE_RE.finditer(str(reason or "")):
        for chunk in match.group("body").split("|"):
            if len(options) >= KB_MAX_DECLARED:
                return options
            part = _OPTION_PART_RE.match(chunk.strip())
            if not part:
                continue
            options.append({"label": part.group("value").strip(), "value": part.group("key").strip()})
    return options


def _stamp(now: Optional[float] = None) -> str:
    return time.strftime(
        "%Y-%m-%d %H:%M %Z", time.localtime(now if now is not None else time.time())).strip()
