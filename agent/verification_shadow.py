"""Shadow-mode verification detector.

Passively monitors assistant responses for verifiable claims that lack
backing evidence. In shadow mode (default) it only logs warnings. In
gated mode it can inject a nudge to force the model to actually run
the verification before claiming it did.

This is the "verification enforcement" layer described in the
VERIFICATION_ENFORCEMENT_GUIDANCE prompt constant.

Config:
    agent.verification_shadow_mode:  "shadow" (default), "gate", or "off"
    agent.verification_shadow_threshold: float, confidence threshold
        for flagging (default 0.6).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum number of shadow-check nudges per response before giving up.
# Prevents infinite loops when the model keeps fabricating.
MAX_SHADOW_ATTEMPTS = 3

# ── Verifiable claim patterns ────────────────────────────────────────
# Each tuple: (compiled_regex, claim_type, negation_hint)
# claim_type: category label for logging
# negation_hint: what the model should have done instead

_VERIFIABLE_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    # "I verified that X" / "I confirmed X" / "I checked X"
    (
        re.compile(
            r"(?:i\s+(?:verified|confirmed|checked|tested|validated|observed|saw|noted))\s+that?\s+(.+)",
            re.IGNORECASE,
        ),
        "claim_of_verification",
        "Show the command/tool output that proves this.",
    ),
    # "X works" / "X is working" / "X is fixed"
    # Tightened: require first-person or task-result framing to avoid
    # matching general prose like "the code works by..." or "this works
    # differently on Windows."  Terminal punctuation or end-of-string
    # anchors prevent matching explanatory continuations.  Comma is only
    # terminal when NOT followed by a space + lowercase letter (which would
    # indicate a continuation clause like "it works, and here is the output").
    # NOTE: We compile three separate patterns (no re.IGNORECASE) so that
    # the comma lookahead [a-z] stays case-sensitive — re.IGNORECASE would
    # make [a-z] match uppercase too, breaking continuation detection.
    # Each branch is case-insensitive via inline (?i) at its own start.
    (
        re.compile(
            r"(?i:the\s+(?:fix|change|patch|update|solution)\s+(?:is\s+(?:working|fixed|correct)|works?|passes?|succeeds?))\s*(?:now|already|again)?\s*(?:\.|!|(?!,\s+[a-z]),|$)|"
            r"(?i:now\s+(?:works?|is\s+fixed|is\s+working))\s*(?:\.|!|(?!,\s+[a-z]),|$)|"
            r"(?i:it\s+(?:is\s+(?:working|fixed|correct)|works?|passes?|succeeds?))\s*(?:now|already|again)?\s*(?:\.|!|(?!,\s+[a-z]),|$)",
        ),
        "claim_of_state",
        "Include the actual output or screenshot.",
    ),
    # "tests pass" / "all tests passed" / "test suite is green"
    (
        re.compile(
            r"(?:test(?:s)?\s+(?:pass(?:ed)?|succeed(?:ed)?|is\s+green|are\s+green|all\s+pass))",
            re.IGNORECASE,
        ),
        "claim_of_test_result",
        "Show the test runner output.",
    ),
    # "it turns red" / "it becomes red" / color change claims
    (
        re.compile(
            r"(?:it\s+(?:turns?|becomes?|changes?\s+to)\s+(?:red|green|blue|yellow|black|white|visible|hidden))",
            re.IGNORECASE,
        ),
        "claim_of_visual_change",
        "Include a screenshot or DOM inspection output.",
    ),
    # "renders correctly" / "displays properly"
    (
        re.compile(
            r"(?:renders?\s+(?:correctly|properly|as\s+expected|fine|well)|displays?\s+(?:correctly|properly|as\s+expected))",
            re.IGNORECASE,
        ),
        "claim_of_rendering",
        "Include a screenshot or browser_vision output.",
    ),
    # "no errors" / "no warnings" / "clean output"
    (
        re.compile(
            r"(?:no\s+(?:errors?|warnings?|failures?|issues?|problems?)\s+(?:found|detected|seen|in\s+(?:output|console|log))|clean\s+(?:output|console|log))",
            re.IGNORECASE,
        ),
        "claim_of_clean_state",
        "Show the actual output that was inspected.",
    ),
    # "I can see" / "I observe" / "looking at"
    # Tightened: exclude conversational filler ("I can see what you mean",
    # "I notice that's a good point") by requiring task-result vocabulary.
    (
        re.compile(
            r"(?:i\s+(?:can\s+see|observe|notice|see)\s+(?:that\s+)?(?:the\s+(?:output|result|page|console|log|screen|element|button|color|text|value|number|status|response|rendering|display|test|build))\s+.+?)(?:\.|$)",
            re.IGNORECASE,
        ),
        "claim_of_observation",
        "Back this up with a tool result or screenshot.",
    ),
]

# ── Evidence patterns ────────────────────────────────────────────────
# If any of these are present in the response, the claim is considered
# backed by evidence and won't be flagged.

_EVIDENCE_PATTERNS: List[re.Pattern] = [
    # Tool output markers
    re.compile(r"(?:terminal|execute_code|browser_|read_file|search_files)\s*\(", re.IGNORECASE),
    # Code block with command output
    re.compile(r"```(?:bash|shell|sh|console|output|text|py)\s*\n[^`]*\$?\s*\S+", re.MULTILINE),
    # Screenshot reference
    re.compile(r"(?:screenshot|SCREENSHOT|vision|browser_vision)", re.IGNORECASE),
    # Explicit "not yet verified"
    re.compile(r"(?:not\s+yet\s+verified|haven't\s+(?:tested|verified|checked)|need\s+to\s+verify)", re.IGNORECASE),
    # Prescription marker (from fix-fidelity guidance)
    re.compile(r"Prescription:\s+", re.IGNORECASE),
    # JSON output from a tool
    re.compile(r"(?:exit_code|output|status|content)\s*:", re.IGNORECASE),
    # "returned" / "output:" / "result:"
    re.compile(r"(?:returned|output:|result:)\s+", re.IGNORECASE),
    # "return code" / "traceback"
    re.compile(r"(?:return\s+code|traceback\s*\()", re.IGNORECASE),
]


def has_evidence(text: str) -> bool:
    """Check if the response text contains evidence markers."""
    return any(pat.search(text) for pat in _EVIDENCE_PATTERNS)


def detect_verifiable_claims(text: str) -> List[Dict[str, Any]]:
    """Scan assistant response text for verifiable claims without evidence.

    Returns a list of dicts with keys:
        - claim_type: str (category)
        - match: str (the matched text)
        - hint: str (what evidence should have been included)
        - confidence: float (0.0-1.0, heuristic)

    Only claims that lack evidence markers in the same response are
    returned.
    """
    if not text or not text.strip():
        return []

    # Quick check: if there's evidence in the response, skip detailed scanning
    # (avoids false positives on responses that include real output)
    if has_evidence(text):
        return []

    claims: List[Dict[str, Any]] = []
    for pattern, claim_type, hint in _VERIFIABLE_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            matched_text = match if isinstance(match, str) else match[0]
            # Confidence heuristic: longer matches = more specific = higher confidence
            confidence = min(0.4 + len(matched_text) * 0.01, 0.95)
            claims.append({
                "claim_type": claim_type,
                "match": matched_text[:200],  # Cap length
                "hint": hint,
                "confidence": round(confidence, 2),
            })

    return claims


def shadow_check(
    *,
    response_text: str,
    shadow_mode: str = "shadow",
    threshold: float = 0.6,
) -> Optional[Dict[str, Any]]:
    """Run the shadow-mode verification check on an assistant response.

    Args:
        response_text: The assistant's response text.
        shadow_mode: "shadow" (log only), "gate" (return nudge), or "off".
        threshold: Minimum confidence to flag a claim.

    Returns:
        None if no claims found or mode is "off".
        In "shadow" mode: dict with flagged claims + log warning.
        In "gate" mode: dict with a nudge string to inject.
    """
    if shadow_mode == "off":
        return None

    claims = detect_verifiable_claims(response_text)

    # Filter by confidence threshold
    flagged = [c for c in claims if c["confidence"] >= threshold]

    if not flagged:
        return None

    # Deduplicate by claim_type
    seen = set()
    unique_flagged = []
    for c in flagged:
        if c["claim_type"] not in seen:
            seen.add(c["claim_type"])
            unique_flagged.append(c)

    if shadow_mode == "shadow":
        logger.warning(
            "verification_shadow: %d unverified claim(s) in response: %s",
            len(unique_flagged),
            ", ".join(c["claim_type"] for c in unique_flagged),
        )
        return {
            "mode": "shadow",
            "flagged_claims": unique_flagged,
            "count": len(unique_flagged),
        }

    # "gate" mode — build a nudge
    claim_descriptions = "\n".join(
        f"  - [{c['claim_type']}] \"{c['match'][:80]}\" — {c['hint']}"
        for c in unique_flagged
    )
    nudge = (
        f"[VERIFICATION NUDGE] Your previous response contained {len(unique_flagged)} "
        f"verifiable claim(s) without backing evidence:\n"
        f"{claim_descriptions}\n\n"
        f"Please actually run the verification (execute the command, take the "
        f"screenshot, run the tests) and include the real output in your response. "
        f"If you cannot verify something, say 'not yet verified' instead of claiming "
        f"you did."
    )
    return {
        "mode": "gate",
        "flagged_claims": unique_flagged,
        "count": len(unique_flagged),
        "nudge": nudge,
    }


def get_shadow_mode(agent) -> str:
    """Resolve the verification shadow mode from agent config.

    Checks agent._verification_shadow_mode first, then falls back to
    config.yaml agent.verification_shadow_mode. Default is "shadow".
    """
    mode = getattr(agent, "_shadow_verification_mode", None)
    if mode is None:
        mode = getattr(agent, "_verification_shadow_mode", None)
    if mode:
        return str(mode).lower()
    # Fallback: check config.yaml. cfg_get takes the loaded config dict as its
    # first arg, then the key path; the previous cfg_get("agent", {}) passed
    # "agent" as the dict, so it never actually read config.
    try:
        from hermes_cli.config import cfg_get, load_config
        val = cfg_get(load_config(), "agent", "verification_shadow_mode", default="shadow")
        return str(val).lower()
    except Exception:
        pass
    return "shadow"


def get_shadow_threshold(agent) -> float:
    """Resolve the confidence threshold from agent config.

    Default is 0.6.
    """
    threshold = getattr(agent, "_verification_shadow_threshold", None)
    if threshold is not None:
        return float(threshold)
    try:
        from hermes_cli.config import cfg_get, load_config
        val = cfg_get(load_config(), "agent", "verification_shadow_threshold", default=0.6)
        return float(val)
    except Exception:
        pass
    return 0.6
