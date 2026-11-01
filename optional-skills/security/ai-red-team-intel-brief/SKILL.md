---
name: ai-red-team-intel-brief
description: Produce threat-intelligence briefs on AI red teaming.
version: 1.0.0
author: 0xMrBlueOps, enhanced by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ai-security, red-teaming, threat-intelligence, adversarial-ml]
    category: security
    related_skills: [ai-intelligence-brief, ai-red-teaming-research, ai-paper-deep-dive]
---

# AI Red Team Intelligence Brief Skill

Produce a sourced threat-intelligence briefing about AI red teaming, adversarial
machine learning, jailbreaks, prompt injection, AI-related CVEs, and alignment
failures. Separate demonstrated risk from theoretical work and give concrete
defender actions without performing offensive testing.

## When to Use

- The user requests an AI red-team brief, AI security update, or adversarial-ML digest.
- A scheduled daily or weekly briefing needs current AI security developments.
- The user wants recent research ranked by operational relevance.

## Prerequisites

- Use `web_search` to discover current papers, disclosures, posts, and repositories.
- Use `web_extract` to verify claims against primary sources.
- Use `browser_navigate` only when a source requires browser rendering.
- Do not require external packages, credentials, or active testing of any target.

## How to Run

Ask for a coverage window and desired depth. If the user does not provide one,
use the previous seven days and the weekly length budget.

Example requests:

- "Create today's AI red-team intelligence brief."
- "Summarize the last week of adversarial-ML research."
- "Find critical prompt-injection and AI security developments since July 1."

## Quick Reference

| Priority | Sources |
|---|---|
| 1 | arXiv `cs.CR` and `cs.AI` |
| 2 | Anthropic, Google DeepMind, and OpenAI safety research |
| 3 | Recently created or updated GitHub repositories |
| 4 | Current web reporting and primary disclosures |
| 5 | Hacker News as a discovery fallback |

| Tier | Include |
|---|---|
| CRITICAL | Demonstrated attacks, broken assumptions, exploit chains, or AI-rooted CVEs |
| HIGH | Useful frameworks, benchmarks, tools, or directly applicable research |
| NOTABLE | Relevant early research, policy, or adjacent safety work |

Daily briefs contain at most 15 items. Weekly briefs contain at most 25. When
over budget, remove the lowest-scoring NOTABLE items first.

## Procedure

1. **Set the collection window.**
   - Record the start and end dates.
   - Prefer publication or disclosure dates over search-engine timestamps.

2. **Collect candidates from primary sources.**
   - Search arXiv for jailbreaks, prompt injection, backdoors, extraction,
     poisoning, model inversion, membership inference, and alignment failures.
   - Search lab safety blogs for red-team evaluations and security disclosures.
   - Search GitHub for new or actively maintained red-team tools and frameworks.
   - Use current web reporting and Hacker News to discover candidates, then
     trace each claim back to a primary source.

3. **Verify every candidate.**
   - Read the paper abstract, disclosure, lab post, or repository itself.
   - Confirm publication date, affected systems, empirical evidence, and limits.
   - Do not treat a search snippet or secondary headline as evidence.
   - For production incidents, seek two independent sources when available.

4. **Triage the evidence.**
   - Mark CRITICAL only for demonstrated production impact, a new attack class,
     a broken security assumption, a frontier-lab exploit chain, or an AI-rooted CVE.
   - Mark HIGH for adopted tools, useful benchmarks, agent-security studies, or
     alignment research with direct red-team implications.
   - Mark NOTABLE for relevant interpretability, manipulation, federated-learning,
     policy, or early-stage work.
   - Exclude marketing, unsupported claims, generic non-AI security, surveys
     without new results, and pure theory without operational relevance.

5. **Assess practical risk.**
   - Label exploitability as `Yes`, `Partial`, or `No`.
   - Name the evidence supporting that label and any laboratory constraints.
   - Identify affected models, harnesses, agents, or deployment patterns.
   - Do not inflate risk or suppress material limitations.

6. **Write the brief.**
   - Start with the ISO-8601 date, collection window, reachable sources, and
     counts by tier.
   - For each CRITICAL item include title, primary source, two-sentence summary,
     attack class, affected systems, exploitability, and defender action.
   - For each HIGH item include title, source, summary, relevance, and any
     applicable defender action.
   - For each NOTABLE item include title, source, and a one-line summary.
   - Mark safe, worthwhile reproduction work with `[REPLICATE]`; never provide
     unauthorized targeting instructions.
   - End with unreachable sources, trimmed-item count, and next-cycle follow-ups.

7. **Apply the fallback ladder.**
   - If `web_search` fails, navigate directly to arXiv, lab blogs, GitHub, and HN.
   - If browser rendering fails, use `web_extract` on known source URLs.
   - If a source is blocked or rate-limited, record it and continue; do not loop retries.

## Pitfalls

- arXiv cross-lists may not have AI security as their primary subject.
- A popular jailbreak catalogue is not necessarily a red-team framework.
- Capability benchmarks are not automatically safety evaluations.
- Repository stars do not replace recent commits, documentation, or adoption evidence.
- Blog categories may hide relevant safety posts under general research.
- A lab demonstration does not establish production exploitability.
- Dynamic year strings in canned searches become stale; derive dates from the requested window.

## Verification

Before delivering the brief, confirm:

- Every item links to a verified primary source.
- Every item falls inside the stated collection window or is labeled as context.
- Every CRITICAL item includes evidence, affected systems, exploitability, and a defender action.
- Claims distinguish demonstrated results from inference or theory.
- The item count stays within the applicable daily or weekly budget.
- Unreachable sources and collection gaps are disclosed.
