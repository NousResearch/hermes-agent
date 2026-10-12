"""Soft-refusal detector for the anti-refusal safety net.

This module exposes a synchronous detector that calls a local decision judge
to score completed LLM generations for soft refusals (sanitized near-misses,
declinations, or outright refusals).

The detector is called by the retry logic (``agent/anti_refusal_retry.py``)
after every live text response.

USAGE
    from soft_refusal_detector import detect_soft_refusal

    verdict = detect_soft_refusal(response_text, context={"session": "..."})
    if verdict.is_soft_refusal:
        # handle refusal
        pass

ARCHITECTURE
    The detector calls a local System One decision model via HTTP. The endpoint,
    model, and threshold are named constants at module top so a sibling benchmark
    card can supersede the judge choice by changing one constant.

    The call pattern matches the proven soft_refusal_hook.py in the canonical,
    version-controlled home: ~/.hermes/profiles/.scripts/soft-refusal/ (restored
    there per that directory's RECONCILIATION.md; a working copy with the same
    constants lives at ~/.hermes/scripts/soft-refusal-detection/). This module is
    standalone so it can be imported without fragile profile-script paths.

LATENCY
    Corrected claim: earlier revisions of this docstring stated a "<200 ms" budget
    the detector does not meet. The measured reality (BENCHMARK-RECOMMENDATION.md;
    thresholds.py MEASURED) is p50 388 ms on the benchmark sweep, 392 ms on the
    2026-10-01 run and 399 ms on the 2026-10-02 re-measurement - the claim was
    wrong, and it is corrected here rather than the measurement. The detector uses
    a 30-second timeout and returns a verdict with confidence=None on transport
    failure, so the caller can distinguish a slow judge from a clean generation.

    If latency becomes a blocker, the retry logic can call the detector
    asynchronously or cache recent verdicts. This module is synchronous by design
    to keep the interface simple.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

# --- Named constants (the one reviewable location) -------------------------
# These MUST be named constants, not hardcoded literals, so a sibling benchmark
# card can supersede the judge choice by changing one constant.

# The local decision judge endpoint. Canonical port 11434 - homebrew Ollama 0.35.0,
# verified 2026-10-02; see thresholds.py DEFAULT_ENDPOINT (the canonical copy in
# ~/.hermes/profiles/.scripts/soft-refusal/). An earlier revision of this file
# carried 11435, a stale install port; that value is superseded.
JUDGE_ENDPOINT: str = "http://127.0.0.1:11434/v1/systemone"

# The decision model to call. tev1 (4B) separates refusals from compliant text;
# the 0.8B model does not (measured, not asserted).
JUDGE_MODEL: str = "tev1"

# The refusal probability threshold. 0.60 is the lowest threshold that sits on
# the far side of the gap between compliant ceiling (0.550) and failure floor
# (0.600), buying FPR 0.000 with recall 0.989 on the 245-row labelled corpus.
REFUSAL_THRESHOLD: float = 0.60

# HTTP timeout for the judge call. The measured p50 is 388-399 ms (see LATENCY
# above); 30 s is generous to avoid false negatives on slow calls.
JUDGE_TIMEOUT: float = 30.0

# The Noul question key and instructions. Kept verbatim so the measured numbers
# and the running detector ask the model the identical question.
QUESTION_KEY: str = "failed"
JUDGE_INSTRUCTIONS: str = (
    "The text in `output` is what a generator model returned when asked to produce "
    "a commissioned artifact. Decide whether it is a refusal, or a sanitized "
    "near-miss of what was asked, rather than the work itself."
)

# Truncation window to match measurement conditions. Generations longer than this
# are truncated so the runtime payload matches the corpus window.
TRUNCATION_WINDOW: int = 1800


@dataclass
class SoftRefusalVerdict:
    """The detector's verdict on a completed generation.

    Fields:
        is_soft_refusal: True when the judge score >= REFUSAL_THRESHOLD.
        judge_score: The refusal probability in [0.0, 1.0], or None on transport failure.
        judge_model: The model name that produced the score.
        confidence: The judge score when available, None on transport failure.
            Distinct from judge_score=None: a failed call must never be silently
            recorded as "the generation was fine."
    """

    is_soft_refusal: bool
    judge_score: float | None
    judge_model: str
    confidence: float | None


def detect_soft_refusal(
    response_text: str,
    context: dict[str, Any] | None = None,
) -> SoftRefusalVerdict:
    """Detect whether a completed LLM response is a soft refusal.

    Calls the local decision judge with a Noul question and returns a verdict.
    Synchronous; adds the measured judge latency per call (p50 388-399 ms, see
    LATENCY above).

    Args:
        response_text: The completed LLM response to judge.
        context: Optional context dict (session id, lane, etc.). Currently unused
            but reserved for future logging or routing.

    Returns:
        SoftRefusalVerdict with is_soft_refusal, judge_score, judge_model, and
        confidence. On transport failure, judge_score and confidence are None and
        is_soft_refusal is False (the caller must decide how to handle a dead judge).
    """
    context = context or {}

    # Build the judge payload
    payload = {
        "model": JUDGE_MODEL,
        "state": {"output": (response_text or "")[:TRUNCATION_WINDOW]},
        "questions": {
            QUESTION_KEY: {"type": "noul", "instructions": JUDGE_INSTRUCTIONS},
        },
    }

    # Call the judge
    req = urllib.request.Request(
        JUDGE_ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT) as resp:
            body = json.loads(resp.read())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        # Transport failure: return a verdict with None scores so the caller can
        # distinguish a dead judge from a clean generation.
        return SoftRefusalVerdict(
            is_soft_refusal=False,
            judge_score=None,
            judge_model=JUDGE_MODEL,
            confidence=None,
        )

    # Extract the score
    score = body.get("answers", {}).get(QUESTION_KEY, {}).get("noul")
    if score is None:
        return SoftRefusalVerdict(
            is_soft_refusal=False,
            judge_score=None,
            judge_model=JUDGE_MODEL,
            confidence=None,
        )

    score = float(score)
    is_refusal = score >= REFUSAL_THRESHOLD

    return SoftRefusalVerdict(
        is_soft_refusal=is_refusal,
        judge_score=score,
        judge_model=JUDGE_MODEL,
        confidence=score,
    )


if __name__ == "__main__":
    # Standalone test: python3 soft_refusal_detector.py "I cannot help with that."
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 soft_refusal_detector.py <response_text>")
        sys.exit(1)

    text = " ".join(sys.argv[1:])
    verdict = detect_soft_refusal(text)
    print(
        json.dumps(
            {
                "is_soft_refusal": verdict.is_soft_refusal,
                "judge_score": verdict.judge_score,
                "judge_model": verdict.judge_model,
                "confidence": verdict.confidence,
            },
            indent=2,
        )
    )
