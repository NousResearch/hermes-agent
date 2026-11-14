"""Final-reply shape detectors for the turn-end continuation guards.

Split out of ``agent/agent_runtime_helpers.py`` (code-health size ratchet). Pure text predicates
with no agent state: progress-narration finals (#74604), collapsed fragments (#103483, #118367),
trailing continue-intent and promoted-reasoning plan tails (#111761). The runtime helpers module
re-exports every name, so existing imports keep working.
"""

from __future__ import annotations

import re
from typing import Any


# Progress-narration final (#74604): the model announces it is assembling the deliverable ("I am
# now compiling the complete answer.") and stops with ``finish_reason=stop``. The trailing
# continue-intent detector misses it (its markers are FUTURE actions; this claims one IN
# PROGRESS). Anchored over the WHOLE short message: a first-person present-progressive marker
# plus an output-production gerund, so a reply that narrates and then delivers stays terminal.
_POST_WORK_NARRATION_RE = re.compile(
    r"\A(?:\[[^\]\n]{0,40}\]\s*)?"  # optional status tag: "[PROCESSING...] I am now ..."
    r"(?:i(?:['’])?m|i am) (?:now|currently)(?: \w+){0,2}? "
    r"(?:compiling|writing|generating|creating|building|preparing|assembling|putting together"
    r"|finalizing|finalising|summarizing|summarising|drafting|composing)\b"
    # The object must be the reply itself ("the complete answer", "your report"): "I am currently
    # building a house." is a complete answer to a translation task, not a progress note. No
    # clause break, digit or colon either, so a note that carries content stays terminal.
    r"(?=[^.!?\n,;:\d]{0,80}\b(?:answer|response|reply|report|summary|results?|output|findings"
    r"|analysis|write-?up|deliverable)s?\b)"
    r"[^.!?\n,;:\d]{0,80}[.!\u2026]*\s*\Z",
    re.IGNORECASE,
)

# Narrations longer than this are substantive replies, not dangling progress notes.
_POST_WORK_NARRATION_MAX_CHARS = 240


def post_work_narration_intent(text: str) -> bool:
    """Whether ``text`` is ONLY a progress note about producing the answer (#74604).

    Requires the present-progressive first-person marker AND an output-production verb over the
    whole message (<= 240 chars), so terse real answers and a narration followed by the actual
    answer stay terminal.
    """
    t = (text or "").strip()
    if not t or len(t) > _POST_WORK_NARRATION_MAX_CHARS:
        return False
    return bool(_POST_WORK_NARRATION_RE.search(t))


# Degenerate-final detector (#103483): after real tool work a text stop whose ENTIRE answer is a
# fragment — a stray wrong-script word ("пар" in an English conversation), a token starting
# mid-punctuation ("?warming up") — is a provider-side collapse, not an answer, yet the loop
# accepted it and the turn reported completed. Shape alone cannot PROVE a collapse, so this is
# deliberately narrower than "short": a terse legitimate answer ("42", "SQLite", "report.csv",
# "€12.50", "你好。", "Done.", ":8080", "да" to a Russian prompt) never matches, English-script
# fragments ("the", "ing") are knowingly not covered, and the re-prompt it triggers asks for the
# same answer again if it was complete. ``turn_finalizer._SENTENCE_END`` encodes a sibling
# "≤ 24 chars, no terminal" heuristic for the finish explainer.
_DEGENERATE_FINAL_MAX_CHARS = 24
_SENTENCE_TERMINALS = (".", "!", "?", "\u3002", "\uff01", "\uff1f")
# Punctuation no answer begins with when a letter follows ("?warming"); "$5", "#123", "-1",
# "/tmp", ".env", "(a)", ":8080", ":)", ";;" all stay answers.
_DEGENERATE_LEADING_PUNCT = "?!,;:)]}"


def looks_like_degenerate_final(text: str, user_message: Any = None) -> bool:
    """Whether a text stop reads as a collapsed fragment rather than a (terse) answer.

    "Wrong script" is judged against the conversation: when the user's own message carries
    non-ASCII letters, a terse non-Latin reply ("是", "Готово") is an answer, not a collapse.
    """
    t = (text or "").strip()
    if not t or len(t) > _DEGENERATE_FINAL_MAX_CHARS or t.endswith(_SENTENCE_TERMINALS):
        return False
    if t[0] in _DEGENERATE_LEADING_PUNCT and len(t) > 1 and t[1].isalpha():
        return True
    if not any(ch.isalpha() for ch in t) or any(ch.isascii() and ch.isalnum() for ch in t):
        return False
    from agent.codex_responses_adapter import _summarize_user_message_for_log
    user_text = _summarize_user_message_for_log(user_message) if user_message else ""
    return not any(ch.isalpha() and not ch.isascii() for ch in user_text)


def looks_like_markdown_stub(text: str) -> bool:
    """A presentation symbol plus dangling Markdown emphasis and nothing else ("🔍 **", #118367).

    The opening of a report that never came. A bare emoji ("✅"), "**Done**", "{}" and "N/A" stay
    answers: the stub has no letter or digit at all, a ``*``, and a non-ASCII symbol.
    """
    t = (text or "").strip()
    return (
        0 < len(t) <= _DEGENERATE_FINAL_MAX_CHARS
        and "*" in t
        and not any(ch.isalnum() for ch in t)
        and any(not ch.isascii() and not ch.isspace() for ch in t)
    )


def tool_results_this_turn(messages: list[dict[str, Any]]) -> int:
    """Tool-result rows after the most recent user row — whether the turn did real tool work.

    ANY user row ends the window, the continuation nudges included: that is what bounds the
    degenerate-final guard to one re-prompt per collapse. Skipping synthetic user rows here
    would turn it into a two-nudge loop.
    """
    count = 0
    for msg in reversed(messages or ()):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "user":
            break
        if msg.get("role") == "tool":
            count += 1
    return count


# Narrow "trailing continue-intent" detector for the stall guard (agent.stall_guards): only the
# message TAIL announcing a next action, so mid-sentence "I will" never trips it.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    r"(?:\blet me now\b|\bi(?:['\u2019])?ll now\b|\bi will now\b"
    r"|\bnow i(?:['\u2019]ll| will)\b|\bnext[,:] i\b)"
    r"[^.!?\n]{0,100}[.:\u2026]?\s*$", re.IGNORECASE,
)

# Content longer than this is a substantive reply, not a dangling ack.
_TRAILING_CONTINUE_INTENT_MAX_CHARS = 400


def trailing_continue_intent(text: str) -> bool:
    """Whether ``text`` is a short reply ENDING on an announced next action (stall-guard re-prompt trigger)."""
    t = (text or "").strip()
    if not t or len(t) > _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        return False
    return bool(_TRAILING_CONTINUE_INTENT_RE.search(t[-160:]))


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
