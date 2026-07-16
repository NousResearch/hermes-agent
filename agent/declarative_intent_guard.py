"""Declarative Intent Guard — turn-end interception for text-without-action.

Detects when the agent emits text claiming imminent action ("Now I'll X", "Let me X")
but produces zero tool calls in the same turn. Injects a synthetic nudge to force
execution, identical in architecture to verify_on_stop.

Pattern: After verify_on_stop, before text_response break in conversation_loop.py.
"""

import re
from typing import Optional

# ── Declarative intent patterns ──────────────────────────────────────────
# These match text that CLAIMS action is being taken but requires tool calls.
DECLARATIVE_PATTERNS = [
    # "Now I'll X" / "Now I will X"
    re.compile(r"\bNow\s+I(?:'ll|\s+will)\s+(?:now\s+)?(?:run|test|check|verify|review|implement|fix|build|deploy|spawn|dispatch|investigate|scan|audit|create|add|update|patch|commit|push|merge|rebase)"),
    # "Let me X"
    re.compile(r"\bLet\s+me\s+(?:now\s+)?(?:run|test|check|verify|review|implement|fix|build|deploy|spawn|dispatch|investigate|scan|audit|create|add|update|patch|commit|push|merge|rebase)"),
    # "I'll X" / "I will X"  
    re.compile(r"\bI(?:'ll|\s+will)\s+(?:now\s+)?(?:run|test|check|verify|review|implement|fix|build|deploy|spawn|dispatch|investigate|scan|audit|create|add|update|patch|commit|push|merge|rebase)"),
    # "I'm going to X"
    re.compile(r"\bI(?:'|\s+a)m\s+going\s+to\s+(?:run|test|check|verify|review|implement|fix|build|deploy|spawn|dispatch|investigate|scan|audit)"),
    # "Following standard process" / "Following the fix"
    re.compile(r"\bFollowing\s+(?:standard\s+process|the\s+fix|up\s+with|through\s+with)"),
]

# ── Terminal markers (do NOT trigger guard) ──────────────────────────────
TERMINAL_PATTERNS = [
    re.compile(r"\b(?:done|complete|finished|all\s+set|good\s+to\s+go|ready\s+to\s+commit)\b.*[.!?]$", re.IGNORECASE),
    re.compile(r"\b(?:thank|appreciate|great|excellent|perfect)\b", re.IGNORECASE),
    re.compile(r"\b(?:waiting\s+for|when\s+it\s+completes|once\s+done|pending)\b", re.IGNORECASE),
]

MAX_NUDGES = 2  # Prevent infinite loops on false positives


def declarative_intent_guard_enabled() -> bool:
    """Check if the guard is enabled. Always True for now — can be config-gated."""
    return True


def build_declarative_intent_nudge(
    response_text: str,
    turn_tool_count: int,
    attempts: int = 0,
) -> Optional[str]:
    """Build a synthetic user nudge if the agent declared intent without action.

    Returns None if no nudge needed (no pattern match, already has tool calls,
    or max attempts exceeded).
    """
    if turn_tool_count > 0:
        return None  # Agent took action — legitimate turn
    if attempts >= MAX_NUDGES:
        return None  # Don't loop forever
    if not response_text or len(response_text.strip()) < 20:
        return None  # Too short to contain declarative intent

    # Check terminal markers first — don't nudge legitimate endings
    for pattern in TERMINAL_PATTERNS:
        if pattern.search(response_text):
            return None

    # Check declarative intent patterns
    matched = None
    for pattern in DECLARATIVE_PATTERNS:
        m = pattern.search(response_text)
        if m:
            matched = m.group(0)
            break

    if not matched:
        return None

    # Build nudge
    if attempts == 0:
        prefix = (
            "[System: Your last response indicated you would take action "
            f"(\"{matched}\") but you did not emit any tool calls. "
        )
    else:
        prefix = (
            "[System: You still haven't taken the action you declared "
            f"(\"{matched}\"). "
        )

    return (
        f"{prefix}"
        f"If you intend to perform this action, use the appropriate tool NOW. "
        f"If the task is complete, rephrase your response to clearly state what "
        f"was accomplished — do not describe future actions as if they have "
        f"already happened.\n\n"
        f"Declarative phrase detected: \"{matched}\"\n"
        f"Attempt {attempts + 1} of {MAX_NUDGES}."
    )
