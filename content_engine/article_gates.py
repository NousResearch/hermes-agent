"""Article quality gate.

Reuses llm_generate.gate_post for the shared slop / stale-tech / unfilled
placeholders / em-dash / length checks, and adds article-only checks:
- min words (ARTICLE_MIN_WORDS)
- at least 3 H2 sections
- a takeaway / "what I'd try next" section
- data integrity (numbers in the body must appear in the context)
- secret strip (redact credential-shaped tokens in body and context)

check() returns a GateResult dataclass with the redacted body and context
so downstream code (illustrator, assembler, Discord send) never sees the
raw credentials even if the gate "passed" (it always flags the redaction).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List

import config as cfg_mod
from llm_generate import gate_post


SECRET_PATTERNS = [
    # OpenAI / generic API keys: sk-... with sufficient entropy.
    re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|live-)?[A-Za-z0-9._-]{8,}"),
    # GitHub tokens (classic + fine-grained prefixes).
    re.compile(r"ghp_[A-Za-z0-9]{16,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{16,}"),
    re.compile(r"gho_[A-Za-z0-9]{16,}"),
    # AWS access keys.
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Private keys.
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    # Generic bearer / api_key=value / .env-style.
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\b\s*[:=]\s*['\"]?([A-Za-z0-9._+/=_-]{8,})"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"),
    # Anomaly .env value leak (KEY=VALUE lines, when preceded by KEY=ALL_CAPS).
    re.compile(r"^[A-Z][A-Z0-9_]{2,}=[A-Za-z0-9._+/=_-]{8,}$", re.MULTILINE),
]


def _redact(text: str) -> tuple[str, int]:
    """Replace credential-shaped tokens with ***REDACTED***. Returns (text, count)."""
    count = 0
    out = text
    for pat in SECRET_PATTERNS:
        def repl(m):
            nonlocal count
            count += 1
            return "***REDACTED***"
        out = pat.sub(repl, out)
    return out, count


def _word_count(body: str) -> int:
    return len(re.findall(r"\b\w+\b", body))


def _h2_count(body: str) -> int:
    return sum(1 for line in body.splitlines() if line.startswith("## "))


def _has_takeaway(body: str) -> bool:
    """Return whether the final H2 is a meaningful closing section.

    The historical name is retained for compatibility, but a blog does not
    need a ritual ``What I'd try next`` heading. Accept headings that close an
    argument through a decision, implication, reflection or boundary.
    """
    closing_terms = (
        "try next", "takeaway", "try this", "reflection", "implication",
        "decision", "boundary", "limitation", "limitations", "trade-off",
        "tradeoffs", "what changes", "next move", "bottom line",
    )
    headings = [line[3:].strip().lower() for line in body.splitlines()
                if line.startswith("## ")]
    return bool(headings and any(term in headings[-1] for term in closing_terms))


def _fabricated_numbers(body: str, context: str) -> list[str]:
    """Find numbers in the body that read as STATISTICS (percentages or large
    metric-like figures) but have no counterpart in the context.

    Deliberately narrow: a long technical article is full of illustrative
    integers ("12 agents", "53 lines", "2am", "443'd") that are rhetorical, not
    data. Flagging those is noise. We only care about fabricated claims, so we
    check percentages and figures of 4+ digits (counts, sizes, money) and ignore
    small bare integers.
    """
    stats = set(re.findall(r"\b\d+(?:\.\d+)?%", body))           # percentages
    stats |= set(re.findall(r"\b\d{4,}\b", body))                # large bare figures
    stats |= set(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", body))    # comma-grouped (1,000)
    # audience/growth metric claims, e.g. "1,000 users", "200 downloads", "3k stars"
    stats |= {m.group(0) for m in re.finditer(
        r"\b\d[\d,.]*\s?[kKmM]?[\s-]*"
        r"(?:users?|downloads?|customers?|sign-?ups?|subscribers?|installs?|"
        r"stars?|MAU|DAU|members?|followers?)\b", body, re.I)}
    if not stats:
        return []
    # Compare on the digit core so "4,000" / "4000" / "4000ms" all match.
    def _core(tok: str) -> str:
        return re.sub(r"[^\d.]", "", tok)
    ctx_cores = {_core(t) for t in re.findall(r"\d[\d.,]*", context)}
    return sorted(s for s in stats if _core(s) not in ctx_cores)


@dataclass
class GateResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    redacted_body: str = ""
    redacted_context: str = ""
    slop_score: int = 0


def check(draft: dict) -> GateResult:
    """Run the full article gate. Returns a GateResult with redacted fields
    safe for downstream use (illustrator, disk write, Discord send).

    The redaction always runs even when other checks pass, so a secret in
    context alone still produces a clean redacted_context.
    """
    body_in = draft.get("body_md") or ""
    ctx_in = draft.get("context") or ""
    body_clean, body_secrets = _redact(body_in)
    ctx_clean, ctx_secrets = _redact(ctx_in)

    issues: List[str] = []
    slop_score = 0

    # 1. Shared gate: slop / stale-tech / placeholders / em-dash / length /
    #    specificity-vs-context. Pass the redacted context to the shared
    #    gate so any "spike" tokens are already gone.
    shared = gate_post(body_clean, platform="article", context=ctx_clean)
    if not shared.get("passed", False):
        issues.extend(shared.get("issues", []))
        slop_score = max(slop_score, shared.get("slop_score", 0))
    else:
        # Carry the score even on pass so the orchestrator can warn.
        slop_score = max(slop_score, shared.get("slop_score", 0))

    # 2. Article-only: min words.
    min_words = getattr(cfg_mod, "ARTICLE_MIN_WORDS", 450)
    wc = _word_count(body_clean)
    if wc < min_words:
        issues.append(f"Article too short: {wc} words (min {min_words})")
        slop_score += 3

    # 3. Article-only: at least 3 H2 sections.
    h2 = _h2_count(body_clean)
    if h2 < 3:
        issues.append(f"Need at least 3 H2 sections, found {h2}")
        slop_score += 2

    # 4. Article-only: meaningful final closing section.
    if not _has_takeaway(body_clean):
        issues.append(
            "Missing meaningful closing section (e.g. decision, implication, "
            "reflection or boundary)"
        )
        slop_score += 2

    # 5. Data integrity (fabricated numbers) now runs inside the shared gate_post
    #    so short posts get it too; no duplicate check here.

    # 6. Secret strip: surface the count even when everything else passed.
    stripped = body_secrets + ctx_secrets
    if stripped:
        issues.append(f"Secret-stripped: {stripped} credential-shaped token(s) redacted from body+context")
        # Don't fail the gate on redaction alone — the redacted body is safe
        # to ship. The flag is for Sahil to see in the Discord preview.

    return GateResult(
        passed=slop_score < 6,
        issues=issues,
        redacted_body=body_clean,
        redacted_context=ctx_clean,
        slop_score=slop_score,
    )
