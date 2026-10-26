#!/usr/bin/env python3
"""Scope resolver: classify a request's category + complexity with a confidence gate.

Determines, in one decision-model call, whether a request's scope is well enough defined
to proceed or whether the caller should escalate to memory and ultimately the user.

Usage:
    python3 scope_resolver.py "<user request text>"
    python3 scope_resolver.py "<text>" --context "<extra context bytes>"

Prints JSON with category, confidence, complexity, and verdict:
    RESOLVED (confidence >= 0.85) | NEEDS_CONTEXT (0.6-0.85) | ASK_USER (< 0.6)

Dependencies: stdlib only. Requires OPENROUTER_API_KEY in the environment.
Decision model: TypeSafe Jev via OpenRouter's /api/alpha/decisions endpoint by default;
set MODEL to another OpenRouter model id to swap the backend.

This script never hardcodes or reads any secret from disk; the only credential is the
OPENROUTER_API_KEY environment variable (a real key is never committed anywhere).
"""

import argparse
import json
import os
import urllib.error
import urllib.request

# Decision model (System One). Swap to any typed-decision model id OpenRouter carries.
MODEL = "typesafe/jev-1.13"
BASE = "https://openrouter.ai/api/alpha/decisions"

CATEGORY_DESCRIPTIONS = {
    "heavy-dev": "Building a complex system with schemas, code, and components",
    "stuck": "Debugging a failure, error, or blocker that needs resolution",
    "fresh-angle": "Need a new approach, creative solution, or alternative perspective",
    "code-review": "Reviewing existing code for quality, security, or correctness",
    "adversarial": "Testing a system for weaknesses, edge cases, or attack vectors",
    "simple": "A straightforward task with clear steps and no unknowns",
}

COMPLEXITY_LEVELS = [
    "Direct execution, no unknowns",
    "Multi-step process with some judgment",
    "Edge case, ambiguous scope, or needs escalation",
]


def _api_key() -> str:
    """Read the OpenRouter key from the environment only. Never from disk."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    return key


def _call_decision(state: str, questions: dict) -> dict:
    body = {"model": MODEL, "state": state, "questions": questions}
    req = urllib.request.Request(
        BASE,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP Error {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"URL Error: {e.reason}"}


def scope_request(description: str, context: str = "") -> dict:
    """Classify a request's category + complexity with a confidence gate.

    Returns {"category", "confidence", "complexity", "verdict"}.
    Verdict: RESOLVED (>=0.85), NEEDS_CONTEXT (0.6-0.85), ASK_USER (<0.6).
    """
    state = description if not context else f"{description}\nContext: {context}"
    resp = _call_decision(
        state=state,
        questions={
            "category": {
                "type": "choice",
                "instructions": "What kind of work does this request require?",
                "criteria": CATEGORY_DESCRIPTIONS,
            },
            "complexity": {
                "type": "score",
                "instructions": "How complex is this request to resolve?",
                "criteria": COMPLEXITY_LEVELS,
            },
        },
    )
    answers = resp.get("answers", {})
    cat = answers.get("category", {})
    cpx = answers.get("complexity", {})

    category = cat.get("choice", "unknown")
    confidence = cat.get("confidence", 0.0)
    raw_complexity = cpx.get("score")
    normalized = 0.0
    if raw_complexity is not None:
        # Score is a position on 0..len(criteria)-1; normalize onto 0.0-1.0.
        normalized = raw_complexity / (len(COMPLEXITY_LEVELS) - 1)

    if resp.get("error") or not confidence:
        verdict = "ASK_USER"
    elif confidence >= 0.85:
        verdict = "RESOLVED"
    elif confidence >= 0.6:
        verdict = "NEEDS_CONTEXT"
    else:
        verdict = "ASK_USER"

    result = {
        "category": category,
        "confidence": round(confidence, 3),
        "complexity": round(normalized, 3),
        "verdict": verdict,
    }
    if resp.get("error"):
        result["error"] = resp["error"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve a request's scope via a decision model.")
    parser.add_argument("description", help="The request text")
    parser.add_argument("--context", default="", help="Extra context, e.g. memory excerpts")
    args = parser.parse_args()
    print(json.dumps(scope_request(args.description, args.context), indent=2))
