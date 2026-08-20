---
name: kirsanov-prompt-discipline
title: Kirsanov Prompt Discipline
description: Every prompt must pass through this 6-step formula. Enforce on every agent in the fleet — no exceptions.
tags: [prompt-engineering, standards, quality]
---

# Kirsanov Prompt-Writing Discipline

Every prompt you write — whether for another agent, a ticket, a design spec, or a client handoff — must pass through this 6-step formula in exact order:

1. **ACTION VERB** — start with "Please [verb]..." Never "I need you to..." or "Can you..." Polite commands, not requests.

2. **EXACT SCOPE** — comma-separated, exhaustive. List everything to cover. No ambiguity.

3. **METHOD PREFERENCE** — state how to approach it. "Prefer direct inspection over summaries." "Use GraphQL, not REST."

4. **OUTPUT CONSTRAINT** — size-limit the result. "400-500 words per skill." "No more than 500 lines." Agents need boundaries.

5. **EDGE CASE ESCAPE** — one escape clause. "If it's a SPA, try a best effort load." "If classification uncertain, route to triage."

6. **QUALITY GUARANTEE** — the final check. "Ensure full-stack coverage." "Human legible too."

## Hard Rules

- **Zero preamble.** Never "I need you to..." — start with the action verb every time.
- **Never hedge.** No "probably," "maybe," "I think." State the scope and let the agent handle edge cases.
- **Never leave output format unspecified.** Always give a size or structure constraint.
- **Never skip the escape clause.** Agents choke on edge cases without one.

## Examples

### Technical (infrastructure/code)

Please audit the config.yaml for drift between declared skills and on-disk directories. List: (a) declared-not-found, (b) present-not-declared, (c) broken symlinks. Prefer file-inspection over yq — verify with stat. Output a 3-column table (category, skill_name, issue_type), 1 row per problem. If a directory exists but has no SKILL.md, flag "empty" instead of skipping. Ensure every row has a concrete path — no "several issues" summaries.

### Operational (process/workflow)

Please install the Iris staging environment: Python 3.12, fal-client (pip install), Hermes gateway dev build, enforcer dry-run at /srv/hermes/designer-data/. Prefer brew for system packages, pipx for Python CLIs. Output a 6-step checklist: step number, command (or N/A), pass/fail. If brew is missing, fall back to python.org installer. Ensure every installed binary reports --version before declaring pass.

### Client-facing (update/handoff)

Please write the weekly PA gateway summary covering 2026-06-20 to 2026-06-26. Include: uptime %, message count, errors-by-class (network, auth, rate-limit, internal), mean latency + p95. Prefer absolute error counts over percentages. Output exactly 4 bullets matching "• Metric: value". If metrics are missing a day, mark "no data" not omit. Convert timestamps to Dubai timezone (UTC+4) before reporting.

## Verification Checklist

- [ ] Prompt starts with "Please [verb]..."
- [ ] Scope is comma-separated, exhaustive
- [ ] Method preference is explicit
- [ ] Output has size/structure constraint
- [ ] Edge case escape clause is present
- [ ] Quality guarantee closes the prompt
- [ ] Zero filler words, no hedging, no preamble
