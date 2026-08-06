"""Editorial review layer for blog drafts.

A second-opinion LLM call, independent from the writer model, that scores a
draft against a strict rubric and returns claims to verify. Uses a free LLM
chain (not the longform writer chain) so the reviewer is always a different
voice from the writer.

Rubric (0-10 each, 10 = best):
  - accuracy_risk: are stated facts/numbers grounded or fabricated?
  - voice_fidelity: does the copy match the stream voice?
  - secret_sauce_leakage: (Builder) does it expose proprietary internals?
  - hype_honesty: is the hype-vs-reality honest, not oversold?
  - structure: British English, zero em-dashes, H2 structure, no AI-isms?

Returns: {passed, score, issues[], claims_to_verify[]}

Degradation: when the LLM is unavailable or returns malformed JSON, the
reviewer degrades to a neutral pass (score=5, no issues). It NEVER blocks
the pipeline on infrastructure failure. Blocking only happens on genuine
quality issues that the LLM surfaces.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm_generate import _call_llm, _llm_configs


def _build_rubric_prompt(draft: dict, stream: str) -> dict:
    """Build the system + user prompt for the review LLM call."""
    from blog.blog_streams import STREAMS

    s = STREAMS.get(stream, {})
    voice = s.get("voice", "")
    structure = s.get("structure", "")

    system = "\n".join([
        "You are a strict editorial reviewer for SahilBlog. You review a draft",
        "blog post against a rubric and return a JSON verdict. You are NOT the",
        "writer. You are an independent critic.",
        "",
        "## Stream voice (the target the draft should match)",
        voice,
        "",
        "## Stream structure rule",
        structure,
        "",
        "## Rubric (score each 0-10, 10 = best)",
        "- accuracy_risk: Are stated facts and numbers grounded in real context",
        "  or fabricated? 10 = all grounded, 0 = fabricated.",
        "- voice_fidelity: Does the copy match the stream voice above?",
        "- secret_sauce_leakage: (Builder stream especially) Does it expose",
        "  proprietary internals, API keys, or implementation secrets? 10 = no",
        "  leakage, 0 = full leak.",
        "- hype_honesty: Is the hype-vs-reality honest? 10 = candid, 0 = oversold.",
        "- structure: British English, zero em-dashes, proper H2 sections, no",
        "  AI-isms ('Let's dive in', 'Great question', etc.)?",
        "",
        "## Output format (STRICT JSON, no prose)",
        "Return exactly this JSON shape:",
        '{"score": <int 0-10>, "passed": <bool>, "issues": [<strings>],',
        ' "claims_to_verify": [<strings>], "rubric": {',
        '   "accuracy_risk": <int>, "voice_fidelity": <int>,',
        '   "secret_sauce_leakage": <int>, "hype_honesty": <int>,',
        '   "structure": <int>}}',
        "",
        "Score is the overall average of rubric dimensions. passed is true when",
        "score >= 6 AND issues is empty. claims_to_verify lists any specific",
        "factual claims (named events, statistics) that should be web-verified",
        "before publishing.",
    ])

    body = draft.get("body_md", "")
    title = draft.get("title", "")
    user = "\n".join([
        f"Title: {title}",
        f"Stream: {stream}",
        "",
        "## Draft body",
        body,
        "",
        "Review this draft. Return JSON only.",
    ])
    return {"system": system, "user": user}


def _call_review_llm(system: str, user: str) -> Optional[str]:
    """Call the LLM chain with a free (non-longform) config. Returns raw text
    or None on failure."""
    for cfg in _llm_configs(longform=False):
        body = _call_llm(system, user, cfg, timeout=90, max_tokens=2000)
        if body:
            return body
    return None


def _extract_json(raw: str) -> Optional[dict]:
    """Extract a JSON object from an LLM response that may have prose around it."""
    # Try direct parse first.
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find a JSON block in the text.
    m = re.search(r'\{[^{}]*"(?:score|passed|issues)"[^{}]*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Broader: find the outermost braces.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def review(draft: dict, stream: str) -> dict[str, Any]:
    """Review a blog draft and return a verdict dict.

    Returns: {"passed": bool, "score": int, "issues": list[str],
              "claims_to_verify": list[str], "degraded": bool}

    Degrades to neutral pass (score=5, passed=True, no issues, degraded=True)
    when the LLM is unavailable or returns malformed output. Never blocks on
    infra failure. ``degraded`` lets callers in strict mode halt rather than
    stage an unreviewed draft.
    """
    prompts = _build_rubric_prompt(draft, stream)
    raw = _call_review_llm(prompts["system"], prompts["user"])
    if not raw:
        return {"passed": True, "score": 5, "issues": [],
                "claims_to_verify": [], "degraded": True}

    parsed = _extract_json(raw)
    if not parsed:
        return {"passed": True, "score": 5, "issues": [],
                "claims_to_verify": [], "degraded": True}

    score = int(parsed.get("score", 5))
    issues = list(parsed.get("issues", []) or [])
    claims = list(parsed.get("claims_to_verify", []) or [])
    # Some models return a placeholder sentence instead of an empty list when
    # there is nothing to verify ("No specific factual claims requiring
    # verification", "None", etc.). Treat those as NO claims — otherwise every
    # clean draft triggers a web-verify round-trip that can fail when search
    # backends are down, forcing an unnecessary retry that ends in None.
    claims = [c for c in claims if not _is_no_claims_placeholder(c)]
    passed = bool(parsed.get("passed", score >= 6 and not issues))

    return {
        "passed": passed,
        "score": score,
        "issues": issues,
        "claims_to_verify": claims,
        "degraded": False,
    }


def _is_no_claims_placeholder(text: str) -> bool:
    """True when a claims_to_verify entry is a 'no claims' placeholder.

    Models that follow the rubric but return prose instead of an empty list
    commonly emit variants of "No specific factual claims requiring
    verification". These are NOT claims to verify; filtering them prevents a
    needless (and potentially failing) news-verify round-trip.
    """
    low = (text or "").strip().lower()
    if not low:
        return True
    # Match only at the START of the entry (after optional filler), so real
    # claims like "OpenAI's 47% adoption..." are never filtered. Bare "na"
    # is intentionally omitted — it matches substrings of real words.
    start = low.lstrip(" -–—:.,;")
    no_claim_starts = (
        "no ", "none", "n/a", "not applicable", "nothing",
        "none found", "no specific", "no factual", "no claims",
        "no particular", "no material", "no issues",
    )
    return start.startswith(no_claim_starts)