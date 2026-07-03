---
name: wiki-quality-gate
description: Verify wiki pages are faithful to their source documents using the C/C/G/H four-dimension audit framework.
version: 1.0.0
author: Yuhao Lin (YuhaoLin2005)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [wiki, quality, verification, audit, knowledge-base]
    category: research
    related_skills: [llm-wiki]
---

# Wiki Quality Gate

Verify that wiki pages created by `llm-wiki` ingest operations are faithful to their source documents. Uses the **C/C/G/H four-dimension audit framework**: Completeness, Consistency, Groundedness, and Honesty. This is a **content quality** audit — complements the existing linter's structural checks (orphan pages, broken links, frontmatter validation).

Two operation modes:
- **Fast gate** (every ingest): Step 0 — verify written files exist on disk. Zero LLM cost. Covers Groundedness (does the file actually exist?).
- **Deep gate** (on-demand, weekly batch): Full C/C/G/H audit with LLM judgment + deterministic grounding pre-scan.

This skill does NOT check for contradictions with existing pages — that's the linter's Pass 2.A.

## When to Use

- After `llm-wiki` ingests a source — Fast gate runs automatically
- When user says "verify wiki quality", "audit wiki pages", "check wiki accuracy"
- Weekly batch: run deep gate over recent pages
- Before trusting the wiki as a decision-making reference

Do NOT use for:
- Structural checks (orphans, broken links) — use `llm-wiki` lint instead
- Contradiction detection with other pages — use `llm-wiki` lint Pass 2.A instead

## Prerequisites

- `llm-wiki` skill must be active (wiki directory must exist)
- `terminal` and `read_file` tools must be available
- No external API keys required
- Works on all platforms

## How to Run

**Fast gate (auto after ingest):**
```
terminal: ls -la $WIKI_PATH/<wiki-page-path>
If file missing → FAIL: page was claimed but not written.
```

**Deep gate (on-demand):**
```
1. Run verify_claims.py against the wiki page and its source
2. Review candidate issues from the script
3. Apply all four C/C/G/H dimensions: Completeness, Consistency, Groundedness, Honesty
4. File findings to log.md with ^Quality flag: callouts
```

## Quick Reference

| Operation | Checks | LLM Cost | When |
|-----------|--------|----------|------|
| Fast gate | Groundedness (file existence) | Zero | Every ingest |
| Deep gate | C/C/G/H: Completeness + Consistency + Groundedness + Honesty (4 dimensions) | ~3K tokens | User-triggered |

## C/C/G/H Audit Framework

All four dimensions apply to wiki quality verification:

| Dimension | Question | Wiki Application |
|-----------|----------|------------------|
| **Completeness** | Does the page cover the source's key content? | Main thesis, sections, critical data points present? |
| **Consistency** | Does the page contradict itself or wiki conventions? | Internal contradictions, format drift, frontmatter matches content? |
| **Groundedness** | Are claims traceable to the source? | File exists on disk, claims map to source text? |
| **Honesty** | Is confidence well-calibrated? | `confidence:` field accurate, limitations acknowledged? |

## Procedure

### Fast Gate: Step 0 — Groundedness (Mechanical File Verification)

After `llm-wiki` completes an ingest, verify that every claimed output file actually exists:

```
terminal: ls -la $WIKI_PATH/path/to/page.md
```

If any claimed page is missing → report "quality-gate: FAIL [file not found: <path>]" to user.
Do not proceed with any other checks until the file exists.

This is non-negotiable when files are claimed. No LLM judgment — just filesystem truth.

### Deep Gate: Completeness Audit

Triggered by user on-demand or weekly batch. Audit whether wiki pages adequately cover their source documents.

① **For each page to audit**, identify its source file(s) from frontmatter `sources:` field.

② **Read both** the wiki page and its source document(s).

③ **Check coverage**:
- Does the wiki page cover the source's main thesis or key findings?
- Are section headings from the source reflected (not necessarily one-to-one, but thematically)?
- Are critical claims or data points from the source present in the wiki page?

④ **Flag gaps**: For each significant omission, add a callout:
```markdown
> ^Quality flag: [completeness] Source covers [topic] but wiki page does not address it.
> Flagged by: wiki-quality-gate | Status: unresolved | Date: YYYY-MM-DD
```

### Deep Gate: Consistency Audit

Check whether the wiki page is internally consistent and follows wiki conventions.

① **Internal consistency**: Read the page and check for self-contradiction:
- Does the introduction claim one thing but a later section claim the opposite?
- Are dates, version numbers, or measurements consistent throughout?

② **Format consistency**: Check against wiki conventions:
- Frontmatter fields match the content (e.g., `confidence: high` but body uses hedging language)
- Section structure follows wiki templates
- Callout format matches `> ^Quality flag:` convention

③ **Flag inconsistencies**:
```markdown
> ^Quality flag: [consistency] Page states "supports X" in intro but "contradicts X" in Findings section.
> Flagged by: wiki-quality-gate | Status: unresolved | Date: YYYY-MM-DD
```

### Deep Gate: Groundedness Pre-Scan (Deterministic)

Before LLM adjudication, run `scripts/verify_claims.py` to identify potentially ungrounded claims:

```
terminal: python3 scripts/verify_claims.py $WIKI_PATH/path/to/wiki-page.md
```

The script performs keyword/substring overlap between the wiki page and its source documents, returning candidate claims with zero or weak source correlation. Only candidate issues are escalated to LLM for final adjudication — the script itself does not make quality decisions.

### Deep Gate: Honesty Audit

Audit whether the `confidence` frontmatter field is well-calibrated.

① **For each audited page**, read the `confidence:` value in frontmatter.

② **Check calibration**:
- Is `confidence: high` justified? → Page cites 2+ independent sources, claims are specific and verifiable.
- Is `confidence: low` missing where it should be? → Page is single-source, opinion-heavy, or makes speculative claims.
- Are weasel words present? → "likely", "probably", "may suggest" without qualification.

③ **Flag miscalibration**:
```markdown
> ^Quality flag: [honesty] confidence:high but page cites only one source and makes speculative claims.
> Suggested: confidence:medium or confidence:low. Flagged by: wiki-quality-gate | Status: unresolved | Date: YYYY-MM-DD
```

### Resolution Workflow

All flagged issues follow the same pattern as the existing `> Contradiction:` callouts in the wiki schema:

1. **Flag**: Add `^Quality flag:` callout with dimension, rationale, date, and `Status: unresolved`.
2. **User decides**: The user can mark as `resolved` (fixed), `contested` (disagree with the flag), or leave `unresolved` (needs more info).
3. **Track**: Quality flags are surfaced during `llm-wiki` lint. Unresolved flags older than 30 days are highlighted for review.

## Pitfalls

- **Don't audit mid-ingest.** Wait for the ingestor to complete all writes before running the gate.
- **False positives on groundedness.** Different phrasing doesn't mean ungrounded. The verify_claims.py script does substring overlap — legitimate paraphrasing will show as low match. That's why the LLM adjudicates, not the script.
- **Completeness is not 100% coverage.** A wiki page summarizing a 50-page paper will miss details. Flag only significant omissions (main thesis, key data points, central arguments).
- **Consistency ≠ cross-page contradiction.** Internal consistency checks the page against itself. Cross-page contradiction detection is the linter's job (Pass 2.A).
- **Don't audit pages you didn't create or update.** Batch deep gate targets recently modified pages only.
- **Quality flags are suggestions, not blocks.** The user decides. The gate provides evidence; the user makes the call.

## Verification

After running a quality audit, confirm:
1. Fast gate: all claimed output files verified on disk (Groundedness)
2. Deep gate: audit report appended to `log.md` with ^Quality flag: callouts covering all four C/C/G/H dimensions
3. Flagged pages have unresolved flags visible to the next lint run
4. The user can see what was checked and what was found
