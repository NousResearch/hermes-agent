"""Finalize-in-process machinery for kanban workers.

A kanban worker's only legitimate terminal states are ``kanban_complete`` and
``kanban_block``. Empirically (2026-09-01) a large share of worker crashes are
ONE event: the model finishes its work, emits a plain-text reply, and the
process exits rc=0 without ever calling a terminal board tool. The dispatcher
then records a ``protocol_violation`` and a whole cold respawn is discarded to
recover a single missing tool call.

Two root causes are addressable in-process, and this module is the machinery
for both:

1. RECENCY. The checkpoint obligation lives once in the opening system prompt.
   By late in a long tool loop it has drifted outside the model's effective
   instruction-following window. Remedy: inject a *short* reminder near the
   recent tail of every API request while the task is still running, so the
   obligation is continually re-asserted at a handful of tokens per turn.

2. NO LAST CHANCE. Nothing between ``finish_reason=stop`` and ``exit rc=0``
   forces a terminal tool. Remedy: when the loop is about to close a turn on a
   still-running kanban task with plain text, run ONE additional bounded turn
   whose toolset is restricted to the terminal kanban tools alone. The model
   then physically cannot make any non-terminal tool call; it must either call
   ``kanban_complete`` / ``kanban_block`` or return text (which falls through
   to the existing bounded-retry path exactly as today).

Design rules from the card:
* Do not change the meaning of a legitimate ``kanban_block``.
* Do not raise any retry limit — this is about not NEEDING the retry.
* The per-turn reminder must cost a handful of tokens per turn, not a
  paragraph, and must not stack or duplicate across turns.
* ``agent.tool_use_enforcement: strict`` is NOT the right lever: it is a
  one-shot static system-prompt line for select model families, not a per-turn
  recency notice and not a forced-turn mechanism.

The reminder is injected into the API-call message copy only (``api_messages``),
never into the durable ``messages`` transcript, so it cannot stack or persist
across turns and does not perturb prompt-cache prefix stability.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Optional

# The env var a dispatcher-spawned worker carries. The card prose writes
# ``HERMES_KANBAN_TASK_ID``; the real dispatcher sets ``HERMES_KANBAN_TASK``
# (see ``_spawn_worker_env`` in hermes_cli/kanban_db.py). The code follows the
# real env var — the worker's lifecycle tools already key off it.
_TASK_ENV = "HERMES_KANBAN_TASK"

# The terminal kanban tools a worker may end on. Imported (NOT duplicated)
# from ``agent.kanban_stop`` — the single source of truth — so the detection
# gate, the reminder, and the forced finalize-tool restriction can never
# diverge. Recognized terminal transitions: ``kanban_complete``,
# ``kanban_block``, ``kanban_request_review``, ``kanban_request_changes`` (a
# correct review/return handoff is also a clean close of the worker's run).
from agent.kanban_stop import _TERMINAL_KANBAN_TOOLS as _TERMINAL_TOOLS

# Flag stamped on the ephemeral reminder so `_is_ephemeral_scaffolding` in
# run_agent.py strips it from the durable transcript if it ever leaks into a
# persisted tail (defense in depth; the reminder is injected on the api copy,
# so this should never happen).
_REMINDER_FLAG = "_kanban_checkpoint_reminder_synthetic"

# Assistant turns that must have happened before the reminder is injected at
# all (2026-09-07). Defence in depth behind the wording fix, and it follows
# directly from this module's own stated purpose: the reminder addresses
# RECENCY DECAY — the opening system prompt's obligation drifting out of the
# model's instruction-following window "late in a long tool loop". On the first
# turns there has been no drift, so the reminder has nothing to do and can only
# mislead a worker that is still orienting. Override with
# HERMES_KANBAN_REMINDER_MIN_TURNS (0 restores the pre-2026-09-07 behaviour,
# which is how the negative control neuters this gate).
_REMINDER_MIN_ASSISTANT_TURNS_DEFAULT = 3


def reminder_min_assistant_turns() -> int:
    raw = (os.environ.get("HERMES_KANBAN_REMINDER_MIN_TURNS") or "").strip()
    if not raw:
        return _REMINDER_MIN_ASSISTANT_TURNS_DEFAULT
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return _REMINDER_MIN_ASSISTANT_TURNS_DEFAULT


def _assistant_turns_in(messages: Iterable[dict]) -> int:
    """Count assistant turns already in a message list.

    Never raises on an odd message shape — an accounting helper must not be
    able to fail a turn.
    """
    n = 0
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            n += 1
    return n


# ---------- Reminder (recency fix) ---------------------------------------

def kanban_checkpoint_enabled() -> bool:
    """True while this process is a dispatcher-spawned kanban worker."""
    return bool((os.environ.get(_TASK_ENV) or "").strip())


def worker_task_id() -> str:
    return (os.environ.get(_TASK_ENV) or "").strip()


def build_checkpoint_reminder(task_id: str) -> str:
    """Short (one/two sentence) recency notice stating the terminal obligation.

    Deliberately NOT a re-statement of the whole contract — a handful of tokens
    kept near the recent tail, not a paragraph.

    2026-09-07: the wording is CONDITIONAL, and says so first. The previous text
    opened with the bare imperative "End this worker with a terminal board tool"
    and a weak-hierarchy model read it as "end now": on t_125dfa35 a
    deepseek-v4-flash worker obeyed it 13–19s into orientation, three runs
    running, and called ``kanban_block`` with "Checkpoint requested a terminal
    board state before implementation could begin" — no code was ever written.
    The reminder exists to stop a FINISHED worker exiting on prose. It must
    never read as permission, let alone instruction, to stop early.
    """
    tid = (task_id or "").strip() or worker_task_id() or "this task"
    return (
        "[checkpoint] Reminder about how to finish task `%s` — NOT an "
        "instruction to finish now. Keep working until the task's own work is "
        "actually done or you are genuinely blocked. At that point a plain-text "
        "reply will not close it: call `kanban_complete`/`kanban_block`, or the "
        "handoffs `kanban_request_review`/`kanban_request_changes`." % tid
    )


def maybe_append_checkpoint_reminder(api_messages: list, task_id: str) -> list:
    """Append one short checkpoint reminder to an API-call message copy.

    Operates on a per-request copy so the reminder never persists, never stacks
    across turns, and never perturbs the durable transcript. Returns the input
    list (mutated in place) or a new list with the reminder appended.

    Skips when:
    * not a kanban worker (env unset), or
    * the session already called a terminal tool (no point re-minding), or
    * fewer than ``reminder_min_assistant_turns()`` assistant turns have
      happened yet — the worker is still orienting and there has been no
      recency decay to correct (2026-09-07), or
    * the current tail is a ``user`` message (appending would make invalid
      user→user alternation on strict wire providers).

    The session-terminal check needs the durable messages; this helper reads the
    api copy's own content only, so the caller passes ``terminal_already_called``
    to gate on real session state.
    """
    if not kanban_checkpoint_enabled():
        return api_messages
    if _session_called_terminal_in(api_messages):
        return api_messages
    if not api_messages:
        return api_messages
    if _assistant_turns_in(api_messages) < reminder_min_assistant_turns():
        return api_messages
    tail = api_messages[-1]
    if isinstance(tail, dict) and tail.get("role") == "user":
        return api_messages
    reminder = build_checkpoint_reminder(task_id)
    out = list(api_messages)
    out.append({"role": "user", "content": reminder, _REMINDER_FLAG: True})
    return out


def _session_called_terminal_in(messages: Iterable[dict]) -> bool:
    """Scan a message list for an already-issued terminal kanban tool call."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool" and str(msg.get("name") or "") in _TERMINAL_TOOLS:
            return True
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                if name in _TERMINAL_TOOLS:
                    return True
    return False


# ---------- Finalize turn (last chance before exit) ----------------------

# Per-run (process) state: whether the single finalize turn has fired and
# whether it succeeded, so the rate is measurable rather than inferred. Reset
# only when a worker turn starts under `HERMES_KANBAN_TASK`.
_finalize_fired = False
_finalize_succeeded = False


def reset_finalize_state() -> None:
    """Reset the per-run finalize flags. Called at kanban worker turn start."""
    global _finalize_fired, _finalize_succeeded
    _finalize_fired = False
    _finalize_succeeded = False


def mark_finalize_fired() -> None:
    global _finalize_fired
    _finalize_fired = True


def mark_finalize_succeeded() -> None:
    global _finalize_fired, _finalize_succeeded
    _finalize_fired = True
    _finalize_succeeded = True


def finalize_metrics() -> dict:
    """Snapshot of whether the finalize turn fired / succeeded this run."""
    return {"finalize_turn_fired": _finalize_fired,
            "finalize_turn_succeeded": _finalize_succeeded}


def should_fire_finalize_turn(*, attempts: int, terminal_tools_available: bool) -> bool:
    """Whether the bounded finalize turn should run now.

    Gates, in one place:
    * worker env active (non-kanban sessions are untouched),
    * at most ONE finalize turn per run (the fired latch, also carried on the
      agent as ``_kanban_finalize_turn_fired`` for the loop to read),
    * the terminal tool schemas are actually present in the toolset (nothing
      to force otherwise).
    """
    if not kanban_checkpoint_enabled():
        return False
    if _finalize_fired:
        return False
    # 2026-09-03: fire on the SECOND narrated text-exit, not the first. The
    # first gets the soft nudge (full toolset — "finish the deliverable, then
    # call a terminal tool"). Models that think out loud between tool calls
    # (deepseek-v4-flash, observed on t_9b77b7ac three minutes into a build)
    # were being forced into kanban_block with nothing written, because the
    # forced turn offers ONLY the terminal tools. One soft nudge costs a turn
    # on a genuine exit; the old order cost the whole card.
    return attempts >= 1 and terminal_tools_available


def build_finalize_instruction(task_id: str) -> str:
    """Decisive instruction for the one forced finalize turn."""
    tid = (task_id or "").strip() or worker_task_id() or "this task"
    return (
        "[System: forced finalize — task `%s` is still `running` and this is "
        "your last chance to close it in-process. The ONLY tools available now "
        "are the terminal board tools. If the work is done, call "
        "`kanban_complete(summary=..., artifacts=[...])` now; if you are "
        "genuinely blocked, call `kanban_block(reason=...)` now; if the work is "
        "finished and needs review, call `kanban_request_review(...)` now; if "
        "you are returning changes, call `kanban_request_changes(...)` now. Do "
        "not reply with prose — pick a terminal tool and call it.]" % tid
    )


def terminal_only_schemas(agent_tools: Optional[list]) -> Optional[list]:
    """Filter ``agent.tools`` (OpenAI-wire schema list) to the terminal kanban
    tools. Returns ``None`` when either tool is absent, so the caller knows a
    forced finalize turn is not possible."""
    if not agent_tools:
        return None
    out = [
        t for t in agent_tools
        if isinstance(t, dict)
        and t.get("function", {}).get("name") in _TERMINAL_TOOLS
    ]
    names = {t.get("function", {}).get("name") for t in out if isinstance(t, dict)}
    if not _TERMINAL_TOOLS.issubset(names):
        return None
    return out


def terminal_tools_present(agent_tools: Optional[list]) -> bool:
    return terminal_only_schemas(agent_tools) is not None


def serialize_finalize_metrics(metrics: dict) -> str:
    """JSON used when stamping finalize metrics onto the run row."""
    return json.dumps(metrics, ensure_ascii=False)


__all__ = [
    "build_checkpoint_reminder",
    "reminder_min_assistant_turns",
    "build_finalize_instruction",
    "finalize_metrics",
    "kanban_checkpoint_enabled",
    "mark_finalize_fired",
    "mark_finalize_succeeded",
    "maybe_append_checkpoint_reminder",
    "reset_finalize_state",
    "serialize_finalize_metrics",
    "should_fire_finalize_turn",
    "terminal_only_schemas",
    "terminal_tools_present",
    "worker_task_id",
]