---
name: slopless
description: Use when evaluating AI-generated prose, social copy, marketing text, or long-form content for low-quality or detectable slop patterns. Provides deterministic, zero-LLM linting with pass/fail thresholds, concrete issue detection, and actionable rewrite guidance.
version: 1.0.0
author: KENSEI
license: MIT
metadata:
  hermes:
    tags: [quality-control, linting, ai-text, slop-detection, content-review, deterministic]
    related_skills: [humanizer, brand-voices, sahil-twitter-voice]
---

# SLOPESS — Deterministic Prose Quality Linting for AI Output

## Overview

SLOPESS is a deterministic, zero-LLM heuristics engine that scores AI-generated text for detectable sloppiness ("slop") — the visual, structural, and lexical patterns that make prose immediately identifiable as AI-authored, even when it is grammatically correct.

It is not a style guide. It is a trap detector.

The score is 0–10. Higher = more slop detected. Threshold for pass: < 6.

Pattern categories:
- Boilerplate mantras recurring verbatim
- Template-itis (mad-libs sentence structures)
- Generic filler (no specific details)
- Repetitive structure (over-polished short lines)
- Slant hashtags and adverb bloat
- Corporate AI-isms ("delve into", "it's important to note")

## When to Use
- Any AI draft crosses your desk and you need to gate its quality before publishing
- Batch-linting content engine output before scheduling
- Regression testing: verify a template change did not re-introduce slop
- Benchmarking brand voice drift over time

Do NOT use for: evaluating human-written content (false positives on terse/professional copy), image/video slop (use visual-specific audits), or detecting factual errors (this is style/structure only).

## Core API

```python
import re
from typing import Dict, List

def audit_slop(body_text: str, *, context: str = "general") -> Dict:
    """
    Evaluate prose for AI-detectable slop patterns.

    Args:
        body_text: The draft text to evaluate.
        context: "general" | "twitter" | "linkedin" | "blog" | "marketing"
                 Adjusts weight of some signals.

    Returns:
        {
            "slop_score": int,   # 0–10 (capped)
            "issues": List[str],  # human-readable findings
            "passed": bool,       # True if slop_score < 6
            "threshold": 6,
        }
    """
```

## Detection Patterns & Weights

### 1. Boilerplate Mantras (weight: +2 each)
Verbatim repeated phrases that appear across many AI-generated drafts.

Examples found in the wild:
- "Shipping apps. Breaking things. Fixing them. Repeat."
- "Tap to play."
- "Make the call."
- "Your move."

How to tune: maintain a per-brand `BOILERPLATE_MANTRAS` list.

### 2. Template-itis (weight: +2)
Identical sentence skeleton with one slot swapped.

Example regex:
```python
r"Every AI tool now has a '[\w\s]+' feature"
r"(delve into|it's important to note|in today's digital age)"
```

Remediation: force the draft to include at least two specific details (names, tools, timeframes) that break the mad-lib structure.

### 3. Generic Filler (weight: +3 if zero specifics, +1 if one)
Text with no concrete anchors — numbers, tool names, company names, timeframes.

Specificity checks (at least 2 of these expected for "good" prose):
- Number: `r'\b\d+%|\b\d+x|\b\d+ hours|\b£[\d.]+|\b\d+ items'`
- Tool/tech name: `(Claude|Convex|Supabase|GitHub|Vercel|Postgres)`
- App/product name: `(Plenishd|MatchdayMaestro|CoachOS)`
- Timeframe: `r'\b\d+ (weeks?|months?|days?|years?)\b'`

If `specificity_count == 0` → +3 slop. If `1` → +1.

### 4. Over-Polished Structure (weight: +1)
Four or more short lines (< 30 chars each) with clean line breaks. Reads like stanzas.

```
Built it.

Broke it.

Fixed it.

Shipped it.
```

Remediation: vary line lengths, use a paragraph, or inject one longer explanatory sentence.

### 5. Slant Hashtags & Adverb Bloat (weight: +1)
Three or more hashtags, especially generic ones (`#AI`, `#Startup`, `#Tech`).

Adverb clusters: more than three adverbs ending in `-ly` in a single short paragraph.

### 6. Corporate AI-isms (weight: +1 each, up to +3 cap)
Overused phrases that signal instruction-following polish:
- "delve into"
- "it's important to note"
- "in today's digital age"
- "let's be honest"
- "at the end of the day"
- "a testament to"

Regex:
```python
AI_ISMS = re.compile(
    r"\b(delve into|it's important to note|let's be honest"
    r"|in today's digital age|at the end of the day|a testament to)\b",
    re.IGNORECASE,
)
```

## Context-Specific Tuning

### Twitter/X
- Allow shorter lines (lower threshold from 30 to 20 chars)
- Allow 1 hashtag per post
- Boost specificity demand: tools/names matter more on Twitter

### LinkedIn
- Corporate AI-isms count DOUBLE — the platform rewards them, but they are slop
- Long-form paragraphs allowed
- Generic inspirational quotes score +2 automatically

### Blog / Long-form
- Over-polished structure rule is RELAXED (not penalised)
- Generic filler rule is STRICTER (zero specifics in a 200+ word block = +3 still)

## Usage Recipe: Batch Lint

```python
from typing import List, Dict

def batch_lint(drafts: List[Dict]) -> List[Dict]:
    """
    drafts: [{"id": str, "body": str, "context": str}, ...]
    Returns: [{"id", "slop_score", "issues", "passed"}, ...]
    """
    results = []
    for d in drafts:
        audit = audit_slop(d["body"], context=d.get("context", "general"))
        results.append({"id": d["id"], **audit})
    return results

# Gate: reject anything that fails
for r in batch_lint(drafts):
    if not r["passed"]:
        print(f"REJECT {r['id']} (slop={r['slop_score']}): {r['issues']}")
```

## Usage Recipe: Content Engine Integration

When generating drafts:
1. Pick a template at random from the pool.
2. Run `audit_slop(template_body, context="twitter")`.
3. If `passed` → accept and store.
4. If `failed` → discard, pick next template (max 5 attempts).
5. If all templates fail → accept the lowest-scoring one and flag for manual review.

This is the pattern used in `_fallback_drafts()` in the KENSEI content engine.

## Database Schema Reference

```sql
CREATE TABLE draft_audits (
    id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL,
    slop_score INTEGER NOT NULL,
    slop_issues TEXT,             -- semicolon-separated issue strings
    passed INTEGER NOT NULL,      -- 0 or 1
    context TEXT DEFAULT 'general',
    audited_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Common Pitfalls

1. **False positives on terse technical copy.** A 40-word release note with zero adverbs and one version number will score specificity_count=1. This is by design — terse is not slop, but it is thin. Adjust thresholds with `context`.

2. **Treating slop score as an objective truth.** It is a proxy. The goal is to surface the drafts that FEEL most AI-generated, not to mathematically prove authorship. Use it as a gate, not a judge.

3. **Forgetting to update the boilerplate list.** As templates get rewritten, old mantras die and new ones appear. Audit your own boilerplate list quarterly.

4. **Over-tuning for one platform.** LinkedIn copy will ALWAYS score higher on AI-isms. If you lint cross-platform, use context parameters rather than a global threshold.

5. **Letting slop_score drift upward over time.** If your average score rises month-over-month, your templates are aging. Re-seed them.

## Verification Checklist

- [ ] `audit_slop("Shipping apps. Breaking things. Fixing them. Repeat.", context="twitter")` returns `slop_score >= 4`
- [ ] `audit_slop("Just shipped v2.3 of Plenishd. 47% faster pantry scan on Claude 3.5.", context="twitter")` returns `slop_score < 3`
- [ ] Boilerplate list is non-empty and brand-specific
- [ ] `passed` threshold is reviewed before each campaign cycle
- [ ] Monthly histogram of slop scores is checked for drift

## Changelog

- **v1.0.0** — Initial: six pattern categories, context-aware tuning, batch audit recipe, DB schema reference. Based on KENSEI content engine `_audit_slop()` (2026-05-16).
