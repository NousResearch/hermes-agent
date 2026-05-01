---
name: browser-agent-evaluation
description: Use when comparing browser-agent runtimes, surfaces, or plugins for Hermes. Separate browser control quality from packaging hype, and judge whether a tool is adoption-worthy, pattern-borrowing, or comparator-only.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, browser-agents, evaluation, runtime, automation]
    related_skills: [repo-signal-triage, browser-investigation-discipline]
---

# Browser-Agent Evaluation

## Overview

Browser-agent tools are easy to overrate because demos look impressive. This skill forces a runtime-first evaluation instead of a screenshot-first one.

It borrows the best part of browser-focused skill packs like Browserbase's: focus on **hard-site reality, debugging, escalation discipline, and runtime shape** rather than marketing glow.

## When to Use

Use when comparing:
- browser automation runtimes
- browser-agent plugins or marketplaces
- anti-bot / stealth browser layers
- cloud browser services
- local vs remote browser execution surfaces

## Core Evaluation Axes

1. **Interaction surface**
   - DOM/selectors, accessibility tree, screenshots, traces, state snapshots
2. **Execution model**
   - local, remote, hybrid, serverless, session persistence
3. **Governance & safety**
   - permissions, auditability, credential handling, session isolation, replay/debuggability
4. **Anti-bot realism**
   - what evidence exists beyond marketing words like stealth or residential proxies?
5. **Hermes fit**
   - how easily could the useful parts map into Hermes workflows?
6. **Token discipline**
   - does it encourage narrower state capture, traces, summaries, and decomposition?

## Fast Judgment Pattern

Classify each tool as one of:
- **Adoption candidate** — strong enough to use directly
- **Pattern worth borrowing** — good ideas but not the right substrate
- **Comparator/reference only** — useful benchmark, not a likely component
- **Demo-heavy / ignore for now**

## Questions to Answer

- Is the value in the runtime, the UX shell, or the skill packaging?
- Does it improve difficult sites, or only make happy-path demos look smoother?
- Is the browser state inspectable and replayable?
- Does it help reduce token burn, or increase it with screenshot-heavy loops?
- Are credentials handled safely?

## Hermes-Specific Heuristics

Prefer tools that show:
- stable element references or structured page-state capture
- trace/debug facilities for failures
- recoverable sessions
- support for hard sites, not just static pages
- clear separation between browser control and higher-level agent orchestration

Treat cautiously when the main value is:
- flashy UI without runtime depth
- generic "agent-native" positioning
- cloud coupling without strong governance benefits
- broad claims with little evidence of hard-site robustness

## Output Shape

Use:
- what it actually is
- strongest runtime signal
- what Hermes could borrow
- main risk/coupling concern
- final classification

## Common Pitfalls

1. Confusing browser demos with browser infrastructure.
2. Ignoring credential and session-governance concerns.
3. Comparing UI shells to runtimes as if they are the same layer.
4. Missing token-cost implications of screenshot-heavy workflows.

## Verification Checklist

- [ ] Official repo/site checked
- [ ] Runtime model identified
- [ ] Governance/credential posture noted
- [ ] Hermes borrow-vs-adopt judgment stated
- [ ] Token-discipline implications considered
