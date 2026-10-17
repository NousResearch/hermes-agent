"""Stall-guard tail detectors: does a reply (or promoted reasoning) END on an announced next action?

Consumed by ``agent/turn_final_response.py``, which re-prompts the model to act (bounded to two
continuations per turn) when a turn with tools offered stops on one of these shapes.
"""

from __future__ import annotations

import re

_APOS = r"['’]"
# One sentence's worth of text. A dot inside a token (``foo.py``, ``e.g``) does not end it.
_SENT = r"(?:[^.!?\n]|\.(?=\S))"

# Original "now"-qualified forms. Unanchored, as before: "Found it. Let me now update it."
_NOW_INTENT = (
    rf"\blet me now\b|\bi{_APOS}?ll now\b|\bi will now\b"
    rf"|\bnow i(?:{_APOS}ll| will)\b|\bnext[,:] i\b"
)

# Bare forms ("Let me search for the registry file.") are what stalled models write most, but the
# same openers are ordinary dialog ("I'll be happy to help", "Let's go with B", "I need to know
# which env you want"). So a bare form counts only when it OPENS a sentence and is followed by a
# tool-style action verb. Verbs that are mostly conversational (explain, provide, summarize, try,
# review, start) are left out on purpose: a miss costs one stalled turn, a false fire talks over
# the user.
_BARE_BOUNDARY = r"(?:^|[.!?:;…—–。！？\n]\s*)"
_BARE_FILLER = r"(?:(?:now|next|first|then|so)[,:]?\s+|[a-z]+,\s+){0,2}"
_BARE_OPENER = (
    rf"(?:let me|let{_APOS}s|i{_APOS}ll|i will|i need to|i{_APOS}m going to|i am going to"
    r"|proceeding to)\s+(?:(?:also|first|now|then|just|quickly|still|actually|go ahead and)\s+)*"
)
_BARE_SEQUENCED = r"(?:now|next|first|then)[,:]?\s+i\s+"  # "Next I run the linter"
_ACTION_VERB = (
    r"(?:read|re-?read|open|view|inspect|examine|look(?!\s+forward)|search|grep|find|locate|list"
    r"|scan|fetch|pull|clone|download|load|check|verify|test|run|re-?run|execute|build|compile"
    r"|install|lint|create|write|add(?!\s+that\b)|edit|update|modify|change|fix|patch|replace"
    r"|remove|delete|rename|move|refactor|implement|apply|commit|push|merge|rebase|deploy"
    r"|generate|configure|debug|investigate|explore|trace|profile|query|restart)\b(?!\s+you\b)"
)
_BARE_INTENT = (
    rf"{_BARE_BOUNDARY}(?:{_BARE_FILLER}{_BARE_OPENER}|{_BARE_FILLER}{_BARE_SEQUENCED}){_ACTION_VERB}"
)

# Handoff veto, applied to every form: the announced action is gated on the user, so stopping is
# the correct behaviour ("I'll run the migration once you confirm.", "Next, I will wait for your
# decision.", "Let me check with you first.").
_USER_HANDOFF_VETO = (
    rf"(?!{_SENT}*\b(?:if|once|unless|until|when|whenever|after|before|whether)\s+you\b)"
    rf"(?!{_SENT}*\byour\s+(?:go-?ahead|go\s+ahead|ok(?:ay)?|approval|confirmation|decision"
    r"|permission|input|call|answer|reply|response|sign-?off)\b)"
    rf"(?!{_SENT}*\b(?:wait(?:ing)?|hold(?:ing)?)\s+(?:for|on)\s+(?:you|your)\b)"
    rf"(?!{_SENT}*\bwith\s+you\b)"
    rf"(?!{_SENT}*\blet\s+me\s+know\b)"
)

# The intent must be the LAST sentence, with at most ~100 trailing characters, so an intent
# followed by the work itself ("Let me check. The answer is 42.") never matches.
_TRAILING_CONTINUE_INTENT_RE = re.compile(
    rf"(?:{_NOW_INTENT}|{_BARE_INTENT}){_USER_HANDOFF_VETO}{_SENT}{{0,100}}[.:…]?\s*$",
    re.IGNORECASE,
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
# and no tool call). Visible content uses the verb-gated detector above because a real reply
# legitimately says "I'll" in dialog; chain-of-thought that ENDS on a first-person plan
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
    r"(?:^|[.!?:。！？—–\n]\s*|…\s*)"
    r"(?:let(?:['’]s| me)\b|i(?:['’]ll| will| need to| should| am going to|['’]m going to)\b"
    r"|next[,:]? i\b|now i(?:['’]ll| will| need to)\b|first[,:]? i(?:['’]ll| will| need to)\b"
    r"|จะให้ผม|ผมจะ"
    r"|ต่อไป(?:จะ|ผมจะ)"
    r"|ขอ(?:เริ่ม|ลอง|ตรวจ|แก้|ส่ง|ทำ|ดู)"
    r"|จะ(?:เริ่ม|ลอง|ตรวจ|แก้|ส่ง|ทำ|ดู|รัน|ยิง))"
    r"[^.!?\n。！？]{0,160}(?:[.:…]+)?\s*$",
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
