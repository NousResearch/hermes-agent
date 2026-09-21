"""Blog gate — quality checks for adhoc/manual blog posts.

Adhoc posts (written by hand or from non-pipeline sources) bypass the
generator's write_with_gate flow. This module provides the same quality
envelope for adhoc posts before they are staged.

adhoc_check(draft, stream) -> (status, issues)
  status: "ok" | "fail"
  issues: list[str] concrete failure reasons

Checks (in order):
  1. article_gates.check — slop, em-dash, length, secrets, fabricated numbers
  2. blog_reviewer.review — voice, accuracy, hype, structure (LLM second opinion)
  3. Em-dash Unicode scan (U+2014, U+2013)
  4. Minimum word count vs stream word_target
  5. Meaningful closing-section check (decision, implication, reflection, etc.)
  6. External link presence (AI/PM streams only; builder exempt)

The reviewer LLM call degrades to neutral pass on infra failure (never blocks
on infra). The deterministic checks (3-6) are the hard gate.
"""
from __future__ import annotations

import re
from typing import Optional

from article_gates import check as _article_check
from blog.blog_reviewer import review as _blog_review
from blog.blog_streams import STREAMS
from blog.source_grounding import _extract_paper_references, _verify_links


# Unicode em-dash and en-dash.
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"


def _has_em_dash(body: str) -> list[str]:
    """Scan for Unicode em-dash (U+2014) and en-dash (U+2013).

    Returns a list of issue strings (empty if clean). The shared article_gates
    check also catches the ASCII '--' variant, but Unicode dashes slip past
    text-based regex. This is a dedicated Unicode scan.
    """
    issues: list[str] = []
    em_count_2014 = body.count(_EM_DASH)
    em_count_2013 = body.count(_EN_DASH)
    if em_count_2014:
        issues.append(f"Found {em_count_2014} Unicode em-dash (U+2014) — use a comma or colon instead")
    if em_count_2013:
        issues.append(f"Found {em_count_2013} Unicode en-dash (U+2013) — use a hyphen instead")
    return issues


def _word_count(body: str) -> int:
    return len(re.findall(r"\b\w+\b", body))


def _has_required_section(body: str) -> bool:
    """Check that the final H2 gives the article a real closing move.

    Keep the helper name for callers, but do not require a ritual takeaway
    heading. A decision, implication, reflection, limitation or boundary is
    equally valid when it is the final section.
    """
    closing_terms = (
        "try next", "takeaway", "try this", "reflection", "implication",
        "decision", "boundary", "limitation", "limitations", "trade-off",
        "tradeoffs", "what changes", "next move", "bottom line",
    )
    headings = [line[3:].strip().lower() for line in body.splitlines()
                if line.startswith("## ")]
    return bool(headings and any(term in headings[-1] for term in closing_terms))


def _has_external_link(body: str) -> bool:
    """Check if the body contains at least one external markdown link.

    Matches [text](http...) or [text](https...) patterns. Bare URLs (not
    in markdown link syntax) also count.
    """
    # Markdown link with URL.
    if re.search(r"\[[^\]]*\]\(https?://", body):
        return True
    # Bare URL.
    if re.search(r"https?://[^\s)]+", body):
        return True
    return False


def adhoc_check(draft: dict, stream: str = "ai") -> tuple[str, list[str]]:
    """Run the full adhoc gate on a blog draft.

    draft: {title, description, body_md, tier, tags, format, source, stream, slug}
    stream: "ai" | "pm" | "builder"

    Returns (status, issues):
      status: "ok" if all checks pass, "fail" otherwise
      issues: list of concrete failure strings (empty on pass)
    """
    body = draft.get("body_md", "") or ""
    issues: list[str] = []

    # 1. Article gate: slop, em-dash (ASCII), length, secrets, fabricated numbers.
    gate_result = _article_check(draft)
    if not gate_result.passed:
        issues.extend(gate_result.issues)
    # Always carry the redacted body back into the draft.
    if gate_result.redacted_body:
        draft["body_md"] = gate_result.redacted_body

    # 2. Editorial reviewer (LLM second opinion). Degrades to neutral pass.
    review_result = _blog_review(draft, stream)
    if not review_result["passed"]:
        issues.extend(review_result.get("issues", []))

    # 3. Em-dash Unicode scan (U+2014, U+2013).
    em_issues = _has_em_dash(body)
    issues.extend(em_issues)

    # 4. Minimum word count vs stream target.
    s = STREAMS.get(stream, {})
    word_target = s.get("word_target", 1500)
    min_words = int(word_target * 0.7)  # 70% of target as floor for adhoc
    wc = _word_count(body)
    if wc < min_words:
        issues.append(f"Post too short: {wc} words (min {min_words} for {stream} stream)")

    # 5. Require a meaningful closing section, without forcing a ritual heading.
    if not _has_required_section(body):
        issues.append(
            "Missing meaningful closing section (e.g. decision, implication, "
            "reflection or boundary)"
        )

    # 6. External link presence + dead-link check (AI/PM/research only;
    # builder exempt). The research-roundup stream explicitly requires
    # multi-source synthesis, so dead-link detection is hard-required.
    if stream in ("ai", "pm", "research"):
        if not _has_external_link(body):
            issues.append(f"No external link found — {stream} stream posts must cite at least one primary source")
        else:
            dead = _verify_links(body)
            if dead:
                issues.append(f"Dead links detected ({len(dead)}): " + ", ".join(dead[:3]))

    status = "ok" if not issues else "fail"
    return status, issues

# -- Company case study gate (Block 5) ----------------------------------------

# Known AI/tech company names for regex matching.
_KNOWN_COMPANIES = (
    "OpenAI", "Anthropic", "Google", "DeepMind", "Meta", "Microsoft",
    "Nvidia", "Mistral", "Hugging Face", "Stability AI", "Cohere",
    "Apple", "Amazon", "AWS", "ByteDance", "Alibaba", "Tencent",
    "Baidu", "Inflection", "Adept", "Character.AI", "Perplexity",
    "Databricks", "Scale AI", "Replicate", "Together AI", "Anyscale",
    "Weights & Biases", "HuggingFace", "EleutherAI", "Meta AI",
    "ai[ -]?21", "AI21", "Llama", "GPT", "Claude", "Gemini",
    "Mistral", "Mixtral",
)

# Pattern for known company names (word-boundary, case-sensitive).
_COMPANY_RE = re.compile(
    r"\b(?:" + "|".join(_KNOWN_COMPANIES) + r")\b"
)

# Pattern for specific numbers: digits, $, %, GPU counts, parameter counts.
_NUMBER_RE = re.compile(
    r"(?:"
    r"\$\s?\d[\d,.]*\s?(?:million|billion|trillion|M|B|K)?"
    r"|£\s?\d[\d,.]*\s?(?:million|billion|M|K)?"
    r"|\d+\.\d+\s?(?:million|billion|trillion|thousand)"
    r"|\d+\s?(?:million|billion|trillion|thousand)"
    r"|\d+(?:\.\d+)?\s?(?:%|percent)"
    r"|\d+\s?(?:GPU|TPU|hour|day|node|parameter|param|token)\w*"
    r")"
)


def case_study_check(body: str, stream: str = "ai",
                     is_opinion: bool = False) -> tuple[str, list[str]]:
    """Check that AI posts include a named company/product with a specific number.

    - AI stream: soft block (hard fail) if missing company AND/OR number.
    - PM/builder: warning only (status ok, issues returned).
    - Opinion/exempt posts: skip entirely.

    Returns (status, issues):
      status: "ok" | "fail"  (PM/builder always "ok" with warnings)
      issues: list of failure/warning strings
    """
    if is_opinion:
        return "ok", []

    has_company = bool(_COMPANY_RE.search(body))
    has_number = bool(_NUMBER_RE.search(body))

    issues: list[str] = []

    if not has_company:
        issues.append("No named company/product found")
    if not has_number:
        issues.append("No specific number found (e.g. $63M, 1.8T params, 47%)")

    if stream == "ai":
        # Hard gate: both must be present.
        status = "ok" if not issues else "fail"
    else:
        # PM/builder: warning only, never block.
        status = "ok"

    return status, issues
