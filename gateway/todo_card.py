"""Cross-turn persistent todo plan card for the gateway.

ONE living message per session, edited in place as the plan evolves across
turns, instead of appending a new card per todo call. The seed/edit/skip
decision is a pure function (decide_todo_card_action) split from the IO
(publish_todo_card) so the in-place / no-stacking behavior is unit-testable
with a fake adapter, the same split subagent_roster.py uses.

Presentation-only: callers enqueue sentinels and this edits Telegram messages.
Nothing here writes to conversation history or mutates agent context, tools, or
the system prompt, so per-conversation prompt caching stays intact. The card's
message id lives on GatewayOrchestrator._todo_card_state, keyed by session_key,
so it survives the per-turn recreation of the progress consumer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from gateway.todo_progress import extract_todo_items

# A plan card untouched for longer than this is treated as belonging to a
# finished or abandoned task; the next plan seeds a fresh card instead of
# editing a far-scrolled, possibly-stale message. 24h also sidesteps editing
# very old messages on platforms with edit-age limits (Telegram ~48h).
TODO_CARD_MAX_AGE_S = 24 * 3600

# Minimum seconds between edits to one card. The gateway consumer enforces this
# (Telegram per-chat edit flood ceiling); kept here as the shared default.
TODO_CARD_EDIT_INTERVAL = 1.5

_TERMINAL_STATUSES = {"completed", "cancelled"}


def todo_list_finished(items: Any) -> bool:
    """True when the list is non-empty and every item is terminal.

    Terminal means completed or cancelled. Accepts a coerced item list or a
    raw result payload (JSON string / dict) via extract_todo_items.
    """
    if isinstance(items, (str, dict)):
        items = extract_todo_items(items)
    if not items:
        return False
    saw = False
    for it in items:
        if not isinstance(it, dict):
            continue
        saw = True
        if str(it.get("status") or "pending") not in _TERMINAL_STATUSES:
            return False
    return saw


def decide_todo_card_action(
    state: Optional[Dict[str, Any]],
    *,
    thread_sig: str,
    new_plan: bool,
    card_text: str,
    now: float,
    max_age_s: float = TODO_CARD_MAX_AGE_S,
) -> str:
    """Return 'seed', 'edit', or 'skip'.

    seed: no usable card, the thread/topic changed, a brand-new plan replaced a
          finished one (new_plan and state.finished), or the card is stale.
    skip: a live card exists and the text is identical (no-op edit avoided).
    edit: a live card exists and the text changed.

    new_plan is True only for a write call with merge=False. A merge=True write
    or a read is NOT a new plan, so it edits the same card even after the plan
    finished (reopen / refresh).
    """
    if state is None or not state.get("message_id"):
        return "seed"
    if state.get("thread_sig") != thread_sig:
        return "seed"
    if state.get("finished") and new_plan:
        return "seed"
    if (now - float(state.get("seeded_at") or 0.0)) > max_age_s:
        return "seed"
    if state.get("last_text") == card_text:
        return "skip"
    return "edit"


async def publish_todo_card(
    *,
    adapter: Any,
    store: Dict[str, Dict[str, Any]],
    session_key: str,
    card_text: str,
    finished: bool,
    new_plan: bool,
    chat_id: Any,
    metadata: Optional[dict],
    reply_to: Optional[str],
    edit_accepts_metadata: bool,
    thread_sig: str,
    now: Optional[float] = None,
) -> None:
    """Seed or edit the persistent plan card; reseed once on edit failure.

    Single-writer: only the gateway consumer coroutine (loop thread) calls this.
    Deliberately never appends to any cleanup list. The plan card is the
    deliverable, not a transient progress bubble, so cleanup_progress must not
    delete it; a deleted card would dangle its stored id and force a reseed
    (stacking) next turn.
    """
    if not card_text:
        return
    if now is None:
        now = time.time()

    state = store.get(session_key) if session_key else None
    action = decide_todo_card_action(
        state,
        thread_sig=thread_sig,
        new_plan=new_plan,
        card_text=card_text,
        now=now,
    )

    if action == "skip":
        # No-op edit avoided, but still refresh the finished flag so a later
        # new_plan after this completion correctly seeds fresh.
        if state is not None:
            state["finished"] = finished
        return

    def _save(message_id: Any) -> None:
        if not session_key:
            return
        store[session_key] = {
            "message_id": str(message_id),
            "last_text": card_text,
            "finished": finished,
            "thread_sig": thread_sig,
            "seeded_at": now,
        }
        # LRU touch; cap enforcement lives on the caller's OrderedDict, but
        # move_to_end here keeps recency correct when the store supports it.
        mte = getattr(store, "move_to_end", None)
        if callable(mte):
            try:
                mte(session_key)
            except Exception:
                pass

    if action == "seed":
        result = await adapter.send(
            chat_id=chat_id,
            content=card_text,
            reply_to=reply_to,
            metadata=metadata,
        )
        if getattr(result, "success", False) and getattr(result, "message_id", None):
            _save(result.message_id)
        return

    # action == "edit"
    if state is None or not state.get("message_id"):
        # decide() only returns "edit" for a live card, but guard defensively.
        return
    kwargs: Dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": state["message_id"],
        "content": card_text,
    }
    if getattr(adapter, "REQUIRES_EDIT_FINALIZE", False):
        kwargs["finalize"] = True
    if edit_accepts_metadata:
        kwargs["metadata"] = metadata

    edit_result = None
    edit_error = ""
    try:
        edit_result = await adapter.edit_message(**kwargs)
    except Exception as exc:
        edit_error = str(exc)
    else:
        edit_error = str(getattr(edit_result, "error", "") or "")

    if getattr(edit_result, "success", False):
        if session_key and state is not None:
            state["last_text"] = card_text
            state["finished"] = finished
            state["seeded_at"] = now
        return

    lowered = edit_error.lower()
    # Transient / flood: keep state, do NOT reseed (avoid duplicate spam). The
    # next completion retries. Telegram flood>5s returns retryable=False with
    # error="flood_control:{wait}", so the "flood" substring match is
    # load-bearing; "retry after" never appears in the returned error.
    if (
        getattr(edit_result, "retryable", False)
        or "flood" in lowered
    ):
        return

    # Non-retryable edit failure: message deleted by the user, too old, wrong
    # topic, or no longer editable. Forget the stale id and seed a fresh card
    # ONCE. A failed reseed leaves message_id unset and bails (no recursion).
    if session_key:
        store.pop(session_key, None)
    reseed = await adapter.send(
        chat_id=chat_id,
        content=card_text,
        reply_to=reply_to,
        metadata=metadata,
    )
    if getattr(reseed, "success", False) and getattr(reseed, "message_id", None):
        _save(reseed.message_id)
