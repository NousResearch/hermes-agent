"""Conversational approval replies for plaintext messaging surfaces.

On iMessage/SMS-style platforms there are no buttons and typing ``/approve`` reads
like operating a machine, so a pending approval must also accept the way a person
actually answers: "sure, go ahead", "yes please", "don't do that".

Matching is deliberately **whole-message and table-driven**, never substring:

* Substring matching is unsafe here. "no, don't go ahead" contains "go ahead",
  and a substring matcher would approve a refusal. Requiring the entire
  normalized message to equal a known phrase makes that class impossible.
* No LLM is involved. The decision to execute a flagged command must stay
  deterministic and auditable; a model must never be the thing that decides an
  approval was granted.
* Anything that does not match exactly returns ``None`` and falls through to
  normal busy handling, so the approval keeps blocking (fail-safe) instead of
  being resolved on a guess.

The caller is responsible for the security gate: these phrases may only be
consulted while ``tools.approval.has_blocking_approval(session_key)`` is true,
so a conversational "sure" in ordinary chat can never fire a command.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Optional, Tuple

# (verb, args) for the synthesized slash command, matching the /approve and /deny
# handlers' own argument grammar.
_APPROVE_ONCE = ("approve", "")
_APPROVE_SESSION = ("approve", "session")
_APPROVE_ALWAYS = ("approve", "always")
_DENY = ("deny", "")

#: Whole-message phrases → (verb, args). Keys MUST be pre-normalized (lowercase,
#: no punctuation, single-spaced) — :func:`_normalize` output is matched against
#: them directly. Add only unambiguous answers; anything conditional
#: ("yes but not the second one") must fall through to a human-readable re-ask.
_PHRASES: Dict[str, Tuple[str, str]] = {
    # -- affirmative, one operation ------------------------------------------
    **dict.fromkeys(
        (
            "approve", "approved", "yes", "y", "ya", "yah", "yeah", "yep", "yup",
            "ok", "okay", "k", "kk", "sure", "confirm", "confirmed", "affirmative",
            "go", "go ahead", "goahead", "go for it", "do it", "just do it",
            "please do", "please do it", "do it please", "yes do it",
            "yes please", "please yes", "sure go ahead", "sure do it",
            "ok go ahead", "okay go ahead", "yes go ahead", "yeah go ahead",
            "yep go ahead", "go ahead please", "please go ahead",
            "sounds good", "looks good", "fine", "thats fine", "that is fine",
            "fine by me", "approve it", "approve that", "allow it", "permitted",
            "you can", "you may", "you can do it", "green light", "proceed",
            "carry on", "continue", "send it", "run it", "lgtm",
        ),
        _APPROVE_ONCE,
    ),
    # -- affirmative, remainder of the session --------------------------------
    **dict.fromkeys(
        (
            "session", "approve session", "session approve",
            "approve for this session", "approve for the session",
            "yes for this session", "for this session",
            "approve this session", "ok for this session",
        ),
        _APPROVE_SESSION,
    ),
    # -- affirmative, permanently -------------------------------------------
    **dict.fromkeys(
        (
            "always", "always approve", "approve always", "always allow",
            "allow always", "approve permanently", "permanently",
            "yes always", "always yes", "dont ask again", "do not ask again",
            "stop asking", "stop asking me", "always do this",
        ),
        _APPROVE_ALWAYS,
    ),
    # -- refusal -------------------------------------------------------------
    **dict.fromkeys(
        (
            "deny", "denied", "no", "n", "nope", "nah", "negative",
            "reject", "cancel", "cancel that", "cancel it", "abort", "stop",
            "dont", "do not", "dont do it", "do not do it", "dont do that",
            "do not do that", "dont run it", "do not run it", "no dont",
            "no do not", "no dont do it", "no do not do it", "no dont do that",
            "no do not do that", "no thanks", "no thank you", "not now",
            "hold off", "hold on", "skip it", "skip that", "leave it",
            "never mind", "nevermind", "forget it", "dont bother",
            "do not bother", "no way", "denied for now", "not this time",
            "no go", "dont approve", "do not approve",
            # Compound refusals: a "no" prefix in front of an affirmative phrase is
            # still a refusal. These must be enumerated because whole-message
            # matching (deliberately) will not see the "no" in a substring pass.
            "no dont go ahead", "no do not go ahead",
            "no dont run it", "no do not run it",
            "no dont approve", "no do not approve",
            "no dont send it", "no do not send it",
            "no dont proceed", "no do not proceed",
            "dont go ahead", "do not go ahead",
            "dont send it", "do not send it",
            "dont proceed", "do not proceed",
            "dont cancel", "do not cancel",
            "no dont bother", "no do not bother",
        ),
        _DENY,
    ),
}

# Tapback emoji carry the same intent as a one-word reply on iMessage.
_EMOJI_PHRASES: Dict[str, Tuple[str, str]] = {
    "\U0001f44d": _APPROVE_ONCE,   # 👍
    "\u2705": _APPROVE_ONCE,       # ✅
    "\U0001f44c": _APPROVE_ONCE,   # 👌
    "\U0001f44e": _DENY,           # 👎
    "\u274c": _DENY,               # ❌
    "\U0001f6d1": _DENY,           # 🛑
}

# Longest phrase in the table; a message with more words cannot be a bare answer
# and is left to normal handling rather than approximately matched.
_MAX_PHRASE_WORDS = max(len(phrase.split()) for phrase in _PHRASES)

# Punctuation and symbols are dropped before matching so "Sure!", "yes." and
# "ok :)" all reduce to their bare phrase. Word characters and spaces survive.
_STRIP_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Reduce a reply to a bare comparison key.

    NFKC first so iOS smart punctuation folds (the curly apostrophe in "don't"
    would otherwise survive as a distinct character and miss the table), then
    strip punctuation/emoji, lowercase, and collapse whitespace.
    """
    folded = unicodedata.normalize("NFKC", text or "").replace("\u2019", "'").replace("\u02bc", "'")
    folded = folded.replace("'", "")  # dont == don't, after the curly fold above
    return _WS_RE.sub(" ", _STRIP_RE.sub(" ", folded)).strip().lower()


def match_conversational_approval(text: str) -> Optional[Tuple[str, str]]:
    """Map a plaintext reply to ``(verb, args)``, or ``None`` when it is not a
    clean answer.

    ``None`` is the safe outcome: the caller leaves the approval pending and the
    user can answer again. Only an exact whole-message match resolves it.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    # A bare tapback/emoji reply, before punctuation stripping removes it.
    if emoji_match := _EMOJI_PHRASES.get(raw):
        return emoji_match
    normalized = _normalize(raw)
    if not normalized:
        return None
    # A question is a question, not an answer ("are you sure?" must not approve).
    if "?" in raw:
        return None
    if len(normalized.split()) > _MAX_PHRASE_WORDS:
        return None
    return _PHRASES.get(normalized)


__all__ = ["match_conversational_approval"]
