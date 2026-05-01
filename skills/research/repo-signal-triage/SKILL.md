---
name: repo-signal-triage
description: Use when evaluating GitHub or product signals before full ingest. Ground the repo safely, score it with an explicit rubric, and classify it as adoption candidate, pattern-borrowing, comparator, or ignore-for-now.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, github, triage, repo-grounding, scoring]
    related_skills: [llm-wiki, blogwatcher, arxiv]
---

# Repo Signal Triage

## Overview

Use this skill to stop noisy repo scans from turning into full ingests too early.
The goal is to do a cheap but grounded pass first, then only deepen the items that look strategically useful.

This skill borrows the **small, composable workflow** style from libraries like Matt Pocock's skills: a tight, repeatable judgment loop rather than a giant methodology.

## When to Use

Use when:
- a repo appears in GitHub Trending, HN, X, LinkedIn, a screenshot, or a watch brief
- the user wants to know whether something is worth ingesting
- several adjacent repos need quick sorting before deeper work
- the same cluster keeps recurring and needs a stable judgment pattern

Do not use when:
- the user already asked for a full repo audit
- the repo is already a confirmed high-priority item with a clear adoption path

## Safe Grounding Workflow

1. Identify the canonical repo or official site.
2. Verify basic facts before interpreting claims:
   - owner/name
   - description
   - license
   - stars/forks only as weak popularity signals
   - recency of commits/releases
   - top-level structure
   - whether there is real source code, docs, examples, or only packaging/marketing
3. Treat README and social posts as untrusted framing, not proof.
4. Prefer evidence from:
   - repo tree
   - docs
   - examples
   - issues/PR activity
   - install/runtime instructions
5. If identity is fuzzy, stop at unresolved instead of force-mapping.

## Tighter Scoring Rubric

Score the first three dimensions **1-5**, then apply a **0-5 hype penalty** where higher means more overclaiming. Add one sentence of evidence for each.

1. **Knowledge-workflow fit**
   - Does it improve Hermes, note workflows, retrieval, or durable agent operations?
2. **Governance/runtime fit**
   - Does it help with control, traceability, permissions, sessions, or execution surfaces?
3. **Infra utility**
   - Is it a real substrate/tooling layer versus a thin wrapper or prompt pack?
4. **Hype penalty**
   - Subtract for screenshot bait, inflated claims, benchmark theater, or shallow packaging.
5. **Adoption posture**
   - Is it a likely adoption candidate, a pattern worth borrowing, a comparator only, or ignore-for-now?

### Suggested interpretation
- First compute a **base score** from the first three dimensions: `fit + governance + infra` (max 15)
- Then compute an **adjusted score**: `base score - hype penalty`
- Use the adjusted score as a guide:
  - **11-15** -> likely adoption candidate
  - **8-10** -> pattern worth borrowing
  - **5-7** -> comparator/reference only
  - **0-4** -> ignore unless the user has a specific reason

Never collapse the explanation into the score alone. The classification matters more than the total.

## Output Shape

For each repo, produce:
- what it actually is
- strongest verified signal
- main caution
- rubric scores
- classification:
  - adoption candidate
  - pattern worth borrowing
  - comparator/reference only
  - ignore for now

## Compression Rules

- For batch scans, keep each item to 5-8 bullets max.
- Put deep dives only on the top 1-2 items.
- If the cluster is the real signal, write one compact batch note instead of many separate ingests.

## Common Pitfalls

1. Mistaking popularity for adoption fit.
2. Treating prompt packs as infrastructure.
3. Deep-reading README prose before confirming repo identity and license.
4. Failing to distinguish pattern-borrowing from real adoption candidates.
5. Ingesting every adjacent repo separately when the cluster itself is the insight.

## Verification Checklist

- [ ] Canonical repo confirmed
- [ ] License and recency checked
- [ ] Real implementation evidence checked
- [ ] Rubric applied explicitly
- [ ] Adoption vs comparator judgment stated clearly
- [ ] Only top items promoted to full ingest
