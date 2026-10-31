"""Stall detectors: whether a turn that ended with no tool call stopped on an announced next action.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet). Pure text checks with no
agent state; ``agent.turn_final_response`` decides what to do with the answer.
"""

import re


# Conservative "trailing continue-intent" detector for the said-continue-but-
# stopped stall guard (agent.stall_guards). Matches only when the message tail
# announces an immediate next action, including a non-conditional "I can
# proceed". Ordinary answers and conditional offers do not trip it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$", re.IGNORECASE,
)
_TRAILING_CAN_PROCEED_RE = re.compile(
    r"\bi can proceed\b(?P<action>[^.!?\n]{0,100}[.:\u2026]?\s*)$", re.IGNORECASE,
)
_PREPOSED_CONTINUE_CONDITION_RE = re.compile(
    r"(?:^|[,;:]\s*)(?:if|once|after|when|unless|until|assuming)\b"
    r"|(?:^|[,;:]\s*)(?:provided\s+that|subject\s+to)\b"
    r"|(?:^|[,;:]\s*)pending\s+"
    r"(?:(?:your|the|user|their)\s+)?"
    r"(?:confirmation|approval|permission|consent|decision|authori[sz]ation|go-ahead)\b"
    r"|(?:^|[,;:]\s*)(?:with|upon|awaiting|requires?|requiring|needs?|needing)\s+"
    r"(?:your\s+|the\s+)?"
    r"(?:confirmation|approval|permission|consent|decision|authori[sz]ation|go-ahead)\b"
    r"|(?:^|[,;:]\s*)(?:you|the user)\s+"
    r"(?:confirm(?:s|ed|ing)?|approve(?:s|d|ing)?|authori[sz]e(?:s|d|ing)?"
    r"|permit(?:s|ted|ting)?|consent(?:s|ed|ing)?|decid(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_POSTPOSED_CONTINUE_CONDITION_RE = re.compile(
    r"\b(?:if|once|after|when|unless|until|assuming)\b"
    r"|\bprovided\s+that\b"
    r"|\bsubject\s+to\b"
    r"|\bpending\s+(?:(?:your|the|user|their)\s+)?"
    r"(?:confirmation|approval|permission|consent|decision|authori[sz]ation|go-ahead)\b"
    r"|\b(?:with|upon|awaiting|requires?|requiring|needs?|needing)\s+"
    r"(?:your\s+|the\s+)?"
    r"(?:confirmation|approval|permission|consent|decision|authori[sz]ation|go-ahead)\b"
    r"|\b(?:you|the user)\s+"
    r"(?:confirm(?:s|ed|ing)?|approve(?:s|d|ing)?|authori[sz]e(?:s|d|ing)?"
    r"|permit(?:s|ted|ting)?|consent(?:s|ed|ing)?|decid(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)

# Allow a short diagnostic summary before the dangling action announcement.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 600


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action (stall-guard re-prompt trigger)."""
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    tail = t[-160:]
    if _TRAILING_CONTINUE_INTENT_RE.search(tail):
        return True
    proceed_match = _TRAILING_CAN_PROCEED_RE.search(t)
    if proceed_match is None:
        return False
    clause_start = max(
        t.rfind(mark, 0, proceed_match.start()) for mark in (".", "!", "?", "\n")
    ) + 1
    prefix = t[clause_start:proceed_match.start()]
    if _PREPOSED_CONTINUE_CONDITION_RE.search(prefix):
        return False
    return not _POSTPOSED_CONTINUE_CONDITION_RE.search(proceed_match.group("action"))


# Broader tail detector for PROMOTED REASONING only (reasoning-only clean stop with tools offered
# and no tool call). Visible content keeps the narrow ``let me now`` shape above because a real
# reply legitimately says "I'll" mid-text; chain-of-thought that ENDS on a first-person plan
# ("Let me batch the terminal calls and run them in parallel.", "I need to check the log.") is a
# stalled model whose turn would otherwise report "complete" with zero tool calls (#111761).
# Tail-only and anchored on the last sentence, so reasoning that merely mentions a plan before
# stating its answer ("...Let me check. The answer is 42.") still promotes.
# Thai (unsegmented script, so no \b after the trigger, unlike the English group) shares the same
# tail shape: a first-person future-action marker immediately followed by more Thai text, often
# preceded by an em/en dash rather than sentence punctuation (#116495). Trigger glosses, in
# pattern order: "I will give you" / "I will", "next I('ll)" + one of {start,try,check,fix,send,
# do,look}, "please let me" + one of {start,try,check,fix,send,do,look}, "I('ll)" + one of
# {start,try,check,fix,send,do,look,run,fire}.
_PROMOTED_REASONING_PLAN_TAIL_RE = re.compile(
    r"(?:^|[.!?:\u3002\uff01\uff1f\u2014\u2013\n]\s*|\u2026\s*)"
    r"(?:let(?:['\u2019]s| me)\b|i(?:['\u2019]ll| will| need to| should| am going to|['\u2019]m going to)\b"
    r"|next[,:]? i\b|now i(?:['\u2019]ll| will| need to)\b|first[,:]? i(?:['\u2019]ll| will| need to)\b"
    r"|\u0e08\u0e30\u0e43\u0e2b\u0e49\u0e1c\u0e21|\u0e1c\u0e21\u0e08\u0e30"
    r"|\u0e15\u0e48\u0e2d\u0e44\u0e1b(?:\u0e08\u0e30|\u0e1c\u0e21\u0e08\u0e30)"
    r"|\u0e02\u0e2d(?:\u0e40\u0e23\u0e34\u0e48\u0e21|\u0e25\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08|\u0e41\u0e01\u0e49|\u0e2a\u0e48\u0e07|\u0e17\u0e33|\u0e14\u0e39)"
    r"|\u0e08\u0e30(?:\u0e40\u0e23\u0e34\u0e48\u0e21|\u0e25\u0e2d\u0e07|\u0e15\u0e23\u0e27\u0e08|\u0e41\u0e01\u0e49|\u0e2a\u0e48\u0e07|\u0e17\u0e33|\u0e14\u0e39|\u0e23\u0e31\u0e19|\u0e22\u0e34\u0e07))"
    r"[^.!?\n\u3002\uff01\uff1f]{0,160}(?:[.:\u2026]+)?\s*$",
    re.IGNORECASE,
)


def promoted_reasoning_announces_action(text: str) -> bool:
    """Whether promoted reasoning ENDS on a first-person plan to act (stall, not an answer).

    No overall length cap: the reasoning block of a stalled model is often 300-1600 chars of
    planning monologue; only the tail decides.
    """
    t = (text or "").strip()
    if not t:
        return False
    return bool(_PROMOTED_REASONING_PLAN_TAIL_RE.search(t[-240:]))


# Visible-text variant of the plan-tail detector, used only when intent-ack continuation is
# enabled for the active model (``agent.intent_ack_continuation`` list/true). Local models often
# stop on "Let me write the parse script to a file first, then run it." with zero tool calls; the
# narrow ``let me now`` shape misses that. A real reply can end on a first-person sign-off, so the
# last sentence is screened for closings ("let me know", "I'll wait"), user-directed offers and
# conditions ("if you want", "once you confirm") before it counts as an announced action.
_VISIBLE_PLAN_TAIL_CLOSING_RE = re.compile(
    r"\blet (?:me|us) know\b"
    r"|\bi(?:['’]ll| will| am going to|['’]m going to)\s+"
    r"(?:wait|stop|leave|be (?:here|around|available|happy|glad)|keep (?:an eye|watching|monitoring))\b"
    r"|\byou(?:['’]d)? (?:like|want|prefer|need)\b|\b(?:from|with|to) you\b"
    r"|\bi (?:need to|should) (?:note|mention|add|stress|point out|flag|clarify)\b",
    re.IGNORECASE,
)
_LAST_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def visible_reply_announces_action(text: str) -> bool:
    """Whether a VISIBLE reply ends on a first-person plan to act (with no tool call it is a stall)."""
    t = (text or "").strip()
    if not t or not promoted_reasoning_announces_action(t):
        return False
    last = [s for s in _LAST_SENTENCE_SPLIT_RE.split(t) if s.strip()][-1]
    if _VISIBLE_PLAN_TAIL_CLOSING_RE.search(last):
        return False
    return not _POSTPOSED_CONTINUE_CONDITION_RE.search(last)


def announces_unfinished_action(text: str, *, promoted: bool, ack_mode: str) -> bool:
    """Whether a tool-less final reply ends on an announced next action (a stall, not an answer).

    Visible text gets the narrow tail check; promoted reasoning gets the broader plan-tail check;
    models opted in to intent-ack continuation (``ack_mode == "all"``) also get the visible
    plan-tail check ("Let me write the script to a file first, then run it.").
    """
    if promoted:
        return trailing_continue_intent(text) or promoted_reasoning_announces_action(text)
    return trailing_continue_intent(text) or (ack_mode == "all" and visible_reply_announces_action(text))
