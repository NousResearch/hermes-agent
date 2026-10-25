"""Title-shape rejection: titles that name nothing a reader could find a session by.

A model that answers instead of titling returns a title-shaped string that
names nothing: a refusal ("Unable to work with that request"), a negation of
the deliverable ("No content generated"), a first-person refusal ("I can't
write that scene"), or a third-person report of the request ("The user is
asking me to ..."). Persisting one writes a nonsense thread name — the
card's own bad shape, "no scene with mind control", is a negation of the
deliverable the opener asked for.

This module owns the title-shape guards: the prompt-example echo guard and the
refusal-shaped guard. The prompt's example lines render from the constants
here, so the reject set and the prompt cannot drift apart.

Every pattern below was measured against the fleet's own 19,651 stored session
titles (every ``sessions.title`` in every profile ``state.db``). Together they
fire on 29 titles, every one of them a refusal or a request report, and on none
of the legitimate titles: "No change on North Star pwa", "Nothing owns
simulator lifecycle", "Explicit sexual content", "Refused: sexualized content
involving minors", "Unable to reach the API endpoint" and "I can't reproduce
the ARM bug" all survive.

Rejection is not truncation. ``_request_title`` returns None for a rejected
title, so the caller falls back to ``derive_title`` and the derived title (a
slice of the user's own words) wins instead of the model's garbage.
"""

import re

# The example titles shown to the model in the prompt, and the echo-guard
# set: when the opening message carries little topical signal, a small model
# sometimes takes the cheapest schema-valid answer and parrots one of these
# back verbatim — most visibly "Fix login button on mobile" naming sessions
# that have nothing to do with a login button. The prompt's example lines are
# rendered from these constants so the guard set and the prompt cannot drift
# apart. Port of QwenLM/qwen-code#9709.
PROMPT_GOOD_EXAMPLES = (
    "Fix login button on mobile",
    "Postgres connection pool exhaustion",
    "Friendly greeting",
)
PROMPT_VAGUE_EXAMPLE = "Code changes"

# "Friendly greeting" is deliberately NOT in the reject set: the prompt
# instructs the model to produce it for bare greetings, so it is a legitimate
# output, not an echo failure. The too-vague example is rejected too — a model
# repeating the counter-example says nothing about the session, and the
# derived title the guard falls back to is strictly more informative.
EXAMPLE_ECHO_REJECT = frozenset(
    t.lower() for t in PROMPT_GOOD_EXAMPLES if t != "Friendly greeting"
) | {PROMPT_VAGUE_EXAMPLE.lower()}


def is_prompt_example_echo(title: str) -> bool:
    """Return True when *title* is one of the prompt's own example titles.

    Comparison is case-insensitive after stripping any leading/trailing run of
    non-letter/non-digit characters, so bracket/quote wrappers cannot bypass
    the guard — while ``_clean_title`` keeps brackets for real titles like
    "(WIP) Fix build". Unicode-aware so full-width wrappers are covered too.
    """
    normalized = re.sub(r"^[\W_]+|[\W_]+$", "", title.strip(), flags=re.UNICODE).lower()
    return normalized in EXAMPLE_ECHO_REJECT


# A negation of the deliverable the opener asked for: "No content generated",
# "no scene with mind control", "Nothing was produced".
_NO_DELIVERABLE = re.compile(
    r"^(?:no|nothing)\s+(?:title|titles|content|scene|output|text|answer|response|result|story|"
    r"article|summary|report|draft|image|code|file|files|list|notes|message|reply|description|"
    r"explanation)\b",
    re.IGNORECASE,
)
_NOTHING_GENERATED = re.compile(r"^nothing\s+(?:was|were)\s+\w+", re.IGNORECASE)

# A refusal to deliver: "Unable to work with that request", "Unable to provide
# title", "Cannot generate a title for this", "Can't help with that request".
# The object of the refusal must be a deliverable — "Unable to reach the API
# endpoint" and "Cannot reproduce on ARM64" name a real finding instead, and
# survive.
_UNABLE_TO_DELIVER = re.compile(
    r"^(?:unable|cannot|can't|won't|not able)\s+(?:to\s+)?(?:\w+\s+){0,4}?"
    r"(?:title|conversation|request|message|response|answer|output|content)\b",
    re.IGNORECASE,
)

# A first-person refusal: "I can't write that scene". The verb list keeps
# "I can't reproduce the ARM bug" — a real finding, not a decline.
_FIRST_PERSON_REFUSAL = re.compile(
    r"^i\s+can(?:no|')t\s+(?:help|continue|write|create|generate|produce|comply|do|assist|engage|fulfill)\b",
    re.IGNORECASE,
)

# A third-person report of the request instead of a title: "The user is asking
# me to ...". Every one of the fleet's 8 titles in this shape reports the
# request back instead of naming the work. The guard only ever sees MODEL
# output — a human-written title is set at ``user`` authority and never reaches
# it — so a model reporting "The user is asking about billing" instead of
# titling is the answer-shaped failure this guard exists for.
_REQUEST_REPORT = re.compile(
    r"^the\s+(?:user|assistant|model|agent)\s+(?:is\s+|was\s+|has\s+)?"
    r"(?:asking|requesting|wanting|wants|asked|requested|wanted)\b",
    re.IGNORECASE,
)

_REJECTION_PATTERNS = (
    _NO_DELIVERABLE,
    _NOTHING_GENERATED,
    _UNABLE_TO_DELIVER,
    _FIRST_PERSON_REFUSAL,
    _REQUEST_REPORT,
)


def is_rejection_shaped_title(title: str) -> bool:
    """True when *title* refuses or reports instead of naming the work.

    A title that answers the opener ("Unable to work with that request") names
    nothing a reader could find the session by, so it is rejected outright and
    the derived title wins. See this module's docstring for the corpus
    measurement behind each pattern.
    """
    if not title:
        return False
    return any(pattern.match(title) for pattern in _REJECTION_PATTERNS)
