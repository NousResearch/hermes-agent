"""Bounded stop guard for premature action promises (text stop, no tool call).

Some models on messaging surfaces end a turn with
``finish_reason=stop`` whose text announces the very next action — "Both writes
landed. Running the full verification battery now — …:", "…Fixing all of it
now.", "Moving them into the profile … all as writes now, then one verification
run:", "Applying them correctly instead, plus fixing the gate's
artifact-timestamp parsing:" — and never emits the tool call. No existing
detector covers that surface shape: ``trailing_continue_intent``
(``agent_runtime_helpers``) needs an explicit first-person tail ("let me now" /
"I'll now"), ``_looks_like_codex_intermediate_ack`` needs
``i['’]ll|i will|let me`` (and stays off the chat_completions path while
``tool_use_enforcement`` is ``"auto"``), and the kanban guard is kanban-only.
This sibling extends the stop-gate family (``verification_stop``,
``kanban_stop``): policy-only, it turns a clear immediate-action announcement
hanging at the tail of an otherwise finished answer into a bounded synthetic
nudge that ``agent/turn_stop_gates.py`` appends after the candidate row.

Deliberately a narrow surface-shape heuristic, NOT semantics: it fires only
when its final sentence is short and opens with an imperative/gerund
action verb and carries an immediate marker ("now"/"next"/"then") or hangs on
a colon/ellipsis — and never for questions, option lists, quoted/fenced text,
completed reports (done/finished/verified/green), or background/delegated
ownership. A false positive costs at most two bounded re-prompts (same budget
as the sibling gates); a false negative keeps the pre-guard behaviour.
The ``agent.promise_stop_guard`` config setting controls this gate.
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Budget matches the sibling gates (verify-on-stop, kanban, codex-ack): at most
# two continuations per turn; the counter resets per turn via
# ``agent/turn_context.py::_PER_TURN_RESET_STATE``.
MAX_PROMISE_STOP_NUDGES = 2

# Nudge-row metadata flag. Registered with the ephemeral-scaffolding tables
# (session_persistence / turn_final_response / turn_finalizer /
# conversation_compression) exactly like ``_verification_stop_synthetic`` so the
# row never persists, never replays as user context, and is stripped from the
# returned transcript.
PROMISE_STOP_SYNTHETIC_FLAG = "_promise_stop_synthetic"
# Provenance stamp on the withheld candidate row, distinct from the sibling
# gates' ``verification_required`` / ``verify_hook_continue`` /
# ``kanban_terminal_required``; registered with the supersede-not-union
# assistant-merge set in ``agent_runtime_helpers`` alongside them.
PROMISE_UNFULFILLED_FINISH_REASON = "promise_unfulfilled"

# Stable bytes: the nudge content never varies with the conversation, so replay
# stays prompt-cache safe (``_CODEX_ACK_CONTINUATION_NUDGE`` precedent).
_PROMISE_STOP_NUDGE = (
    "[System: Your last message ended the turn while promising an immediate "
    "action, but it carried no tool call. Do not repeat the promise. Either "
    "execute the announced action now with real tool calls, or deliver the "
    "final answer without promising future work.]"
)

_FALSY_TOKENS = {"0", "false", "no", "off", "none"}

# Imperative + gerund surface forms of the tool-backed action verbs the
# observed misses used. Keep this list about *executing actions*, not speech
# acts (say/tell/explain are deliberately absent: narration about talking is
# not an executable step the model is failing to take).
_ACTION_VERB_RE = re.compile(
    r"^(?:running|run|testing|test|verifying|verify|installing|install|"
    r"applying|apply|writing|write|patching|patch|editing|edit|creating|create|"
    r"moving|move|renaming|rename|copying|copy|fetching|fetch|downloading|download|"
    r"uploading|upload|starting|start|restarting|restart|launching|launch|"
    r"opening|open|closing|close|pushing|push|committing|commit|merging|merge|"
    r"deploying|deploy|building|build|shipping|ship|fixing|fix|debugging|debug|"
    r"searching|search|grepping|grep|scanning|scan|inspecting|inspect|reading|read|"
    r"updating|update|executing|execute|generating|generate|adding|add|removing|remove|"
    r"checking|check)\b",
    re.IGNORECASE,
)
# The sentence must hang on an immediate marker or a trailing colon/ellipsis:
# a plain declarative ending is a report, not a dangling promise.
_IMMEDIATE_MARKER_RE = re.compile(
    r"\b(now|immediately|straight ?away|at once|right ?away|next|then|directly)\b",
    re.IGNORECASE,
)
_TRAILING_HANG_RE = re.compile(r'''[:…]+\s*["'”’)\]]*\s*$''')
# A completed report, however action-flavoured, is terminal.
_COMPLETION_TOKEN_RE = re.compile(
    r"\b(done|complete|completed|finished|verified|all set|green|shipped)\b",
    re.IGNORECASE,
)
# Work already owned by a background/delegated executor is not this agent's
# next action (the delegate-only tool surface disables the guard separately).
_BACKGROUND_TOKEN_RE = re.compile(
    r"\b(background process|background task|backgrounded|delegated|delegating|"
    r"delegation|delegate[ds]? (?:task|worker|subagent)|subagent|polling|routed to|"
    r"dispatched)\b",
    re.IGNORECASE,
)
# Sentence boundaries for the tail probe; ';' counts so a semi-clause can hang.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…;]+|\n")
_QUOTE_STARTS = (">", '"', "'", "“", "”", "‘", "’", "`")
# Bound the action sentence, not its diagnostic preamble.
_PROMISE_MAX_CHARS = 320


def promise_stop_guard_enabled(agent: Any) -> bool:
    """Use the session-scoped config value bound by agent_init."""
    attr = getattr(agent, "promise_stop_guard", None)
    if attr is None:
        attr = getattr(agent, "_promise_stop_guard", True)
    if isinstance(attr, bool):
        return attr
    if isinstance(attr, str):
        return attr.strip().lower() not in _FALSY_TOKENS
    return bool(attr)


def promise_stop_nudge_text() -> str:
    """The stable nudge bytes (pinned by tests for the prompt-cache contract)."""
    return _PROMISE_STOP_NUDGE


def tool_surface_can_act(agent: Any) -> bool:
    """The guard only makes sense when this agent could actually act: tools
    exist, and the surface is not delegation-only (a delegate-task-only agent
    that narrates "the subagent will run it now" is correct — that work is
    owned elsewhere; kanban workers have their own terminal-tool guard)."""
    names = set(getattr(agent, "valid_tool_names", None) or ())
    if not names:
        return False
    return bool(names - {"delegate_task", "a2a_call", "process_manage"})


def _last_sentence(text: str) -> tuple[str, bool]:
    """``(last sentence, hangs on a colon/ellipsis)``. A dangling trailing
    colon/ellipsis is stripped first and itself counts as the hang signal."""
    hangs = bool(_TRAILING_HANG_RE.search(text))
    stripped = _TRAILING_HANG_RE.sub("", text).strip()
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(stripped) if p and p.strip()]
    return (parts[-1] if parts else stripped), hangs


def looks_like_immediate_action_promise(final_text: Any) -> bool:
    """Conservative surface-shape probe: a short final sentence that
    opens with an action verb and hangs on an immediate marker or colon,
    excluding questions, quotes/fences, completion reports, and
    background/delegated ownership. See the module docstring for limits."""
    if not isinstance(final_text, str):
        return False
    text = final_text.strip()
    if not text:
        return False
    if text.startswith(_QUOTE_STARTS):
        return False
    if text.endswith(("?", "？")):
        return False
    if _BACKGROUND_TOKEN_RE.search(text):
        return False
    last, hangs = _last_sentence(text)
    if not last or len(last) > _PROMISE_MAX_CHARS:
        return False
    if last.startswith(_QUOTE_STARTS):
        return False
    if _COMPLETION_TOKEN_RE.search(last):
        return False
    if not _ACTION_VERB_RE.match(last):
        return False
    return bool(_IMMEDIATE_MARKER_RE.search(last) or hangs)


def build_promise_stop_nudge(
    *,
    final_text: Any,
    attempts: int = 0,
    max_attempts: int = MAX_PROMISE_STOP_NUDGES,
) -> Optional[str]:
    """Synthetic continuation when a text-only stop announces an immediate
    action instead of performing it; ``None`` when the turn may end (not the
    promise shape, budget exhausted, or nothing was announced)."""
    if attempts >= max_attempts:
        return None
    if not looks_like_immediate_action_promise(final_text):
        return None
    return _PROMISE_STOP_NUDGE


__all__ = [
    "MAX_PROMISE_STOP_NUDGES",
    "PROMISE_STOP_SYNTHETIC_FLAG",
    "PROMISE_UNFULFILLED_FINISH_REASON",
    "build_promise_stop_nudge",
    "looks_like_immediate_action_promise",
    "promise_stop_guard_enabled",
    "promise_stop_nudge_text",
    "tool_surface_can_act",
]
