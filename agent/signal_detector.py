"""Implicit signal detection & behavioral nudges.

Scans user messages for implicit emotional/cognitive signals (frustration,
confusion, fatigue, etc.) using lightweight regex patterns and prepends
a behavioral nudge to the message before the LLM sees it.

The nudge steers the model toward more context-appropriate responses without
modifying the system prompt or conversation history (cache-safe).

Inspired by NeuroSkill (MIT Media Lab, arXiv:2603.03212).
See: https://github.com/NousResearch/hermes-agent/issues/692
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Signal patterns — each key maps to a list of regex patterns that indicate
# a particular user state. Patterns use word boundaries and are matched
# case-insensitively against the full message text.
# ---------------------------------------------------------------------------

SIGNALS: Dict[str, List[str]] = {
    "frustration": [
        r"\b(stupid|annoying|ugh|wtf|damn|hate this|sick of)\b",
        r"\b(still (not|won't|doesn't|can't))\b",
        r"\b(for hours|all day|keeps? (failing|breaking|crashing))\b",
        r"\b(tried everything|nothing works|give up)\b",
    ],
    "confusion": [
        r"\b(confused|don't (understand|get it)|makes? no sense)\b",
        r"\b(lost|stuck|no idea|bewildered)\b",
    ],
    "urgency": [
        r"\b(asap|urgent|deadline|due (today|tomorrow|soon))\b",
        r"\b(quickly|fast|hurry|right now|immediately)\b",
    ],
    "fatigue": [
        r"\b(exhausted|tired|burned? out|drained|sleepy)\b",
        r"\b(been (at|doing|working on) this .{0,15}(hours|all day))\b",
        r"\b(can't think|brain.{0,5}(fried|dead|mush))\b",
    ],
    "learning": [
        r"\b(learning|studying|new to|beginner|first time)\b",
        r"\b(explain|teach|walk me through|eli5)\b",
    ],
    "exploration": [
        r"\b(curious|wondering|what if|explore|experiment)\b",
        r"\b(brainstorm|ideas?|possibilities)\b",
    ],
    "celebration": [
        r"\b(it works|finally|hell yeah|awesome|nailed it)\b",
        r"\b(fixed|solved|figured (it )?out|got it)\b",
    ],
    "anxiety": [
        r"\b(worried|nervous|anxious|scared|afraid)\b",
        r"\b(might (break|fail|crash)|what if .{0,20}(goes wrong|breaks))\b",
    ],
    "overwhelm": [
        r"\b(overwhelm|too (much|many)|where do I (start|begin))\b",
        r"\b(information overload|drowning in)\b",
    ],
    "deep_work": [
        r"\b(deep (dive|work)|focus|concentrate|in the zone)\b",
        r"\b(don't (interrupt|distract)|flow state)\b",
    ],
}

# ---------------------------------------------------------------------------
# Nudge strings — short behavioral guidance the LLM receives when a signal
# fires.  Kept concise to minimise token overhead (~20-40 tokens total).
# ---------------------------------------------------------------------------

NUDGES: Dict[str, str] = {
    "frustration": (
        "User sounds frustrated. Acknowledge briefly, pivot to a concrete "
        "alternative. Don't repeat what they've tried."
    ),
    "confusion": (
        "User seems confused. Use simpler language, break into clear steps, "
        "lead with a concrete example."
    ),
    "urgency": (
        "User is in a hurry. Lead with the answer, skip preamble, most "
        "direct solution first."
    ),
    "fatigue": (
        "User sounds tired/burned out. Keep response focused, avoid "
        "overload. Offer to handle more autonomously."
    ),
    "learning": (
        "User is learning. Be patient, explain concepts before "
        "implementation, point out beginner pitfalls."
    ),
    "exploration": (
        "User is exploring. Be creative, suggest multiple approaches, "
        "encourage experimentation."
    ),
    "celebration": (
        "User just succeeded. Match their energy briefly, then suggest "
        "next steps."
    ),
    "anxiety": (
        "User is anxious about risk. Reassure with specifics, suggest "
        "reversible approaches, offer dry runs."
    ),
    "overwhelm": (
        "User is overwhelmed. Give ONE clear next step, offer to break "
        "the problem down."
    ),
    "deep_work": (
        "User is in deep focus. Be precise and efficient, no small talk."
    ),
}


def detect_signals(message: str) -> Dict[str, bool]:
    """Scan *message* for implicit user-state signals.

    Returns a dict mapping each signal name to ``True`` if any of its
    patterns matched, ``False`` otherwise.  Matching is case-insensitive.
    """
    if not message:
        return {name: False for name in SIGNALS}

    lowered = message.lower()
    return {
        name: any(re.search(p, lowered) for p in patterns)
        for name, patterns in SIGNALS.items()
    }


def build_nudge_prefix(message: str) -> str:
    """Return a ``[Context: …]`` prefix for *message*, or ``""`` if no
    signals were detected.

    The prefix is meant to be **prepended** to the user message before
    the API call so the LLM receives behavioural guidance without any
    change to the system prompt or conversation history.
    """
    signals = detect_signals(message)
    active = [NUDGES[name] for name, fired in signals.items() if fired]
    if not active:
        return ""
    nudge_text = " ".join(active)
    return f"[Context: {nudge_text}]\n\n"
