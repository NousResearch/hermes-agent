---
name: browser-investigation-discipline
description: Use when a browser task is getting expensive, brittle, or screenshot-heavy. Follow an escalation ladder: fetch/search first, structured state next, full browser only when needed, and trace/debug before repeated blind retries.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [browser, qa, debugging, token-discipline, investigation]
    related_skills: [dogfood, browser-agent-evaluation]
---

# Browser Investigation Discipline

## Overview

This skill keeps browser work from turning into a wasteful loop of screenshots, random clicks, and retries.

It borrows from browser-focused operational playbooks: separate cheap retrieval from expensive interactive browsing, collect structured evidence before escalating, and debug failures systematically.

## When to Use

Use when:
- the task involves a web page, browser automation, or UI debugging
- repeated screenshots are burning tokens
- a site is brittle, dynamic, or hard to automate
- a task may be solvable without a full browser session

## Escalation Ladder

### Level 1 — Cheapest path first
Before opening a full browser session, ask:
- can this be solved with web search?
- can this be solved with a direct fetch or static page read?
- can a single page read answer the question?

Do not jump to full browser control if a static or lightweight path is enough.

### Level 2 — Structured page-state capture
Once in the browser, prefer structured evidence first:
- accessibility / DOM snapshot
- console output
- targeted inspection via JS evaluation
- exact URL/state check

Use screenshots only when visual understanding is necessary.

### Level 3 — Targeted interaction
If the page must be interacted with:
- click/type only against clearly identified targets
- check console after important interactions
- confirm the state change after each step
- avoid long sequences of actions without verification

### Level 4 — Trace and debug
If the site is brittle or failing:
- capture the failure pattern
- compare expected vs actual page state
- inspect auth/session issues
- isolate timing, selector, and bot-detection problems
- write down a site-specific playbook instead of retrying blindly

### Level 5 — Decompose the job
If the task is still long or messy, split it into sub-problems:
- login / auth
- navigation
- extraction
- verification

Do not keep one giant browser session carrying all context forever.

## Token-Discipline Rules

- Prefer checkpoint summaries after major state changes.
- Prefer narrow snapshots over whole-page visual refreshes.
- Prefer targeted screenshots over repeated full-page screenshots.
- Prefer decomposition before the context gets bloated.
- When debugging, gather evidence before asking the model to speculate.

## Output Shape

For a hard browser task, report:
- current page/state
- exact blocker
- evidence gathered
- likely cause
- next best escalation step

## Common Pitfalls

1. Going straight to screenshots when console/snapshot data would answer faster.
2. Clicking through multiple steps without checking what changed.
3. Treating retries as progress.
4. Failing to separate browser-runtime failure from higher-level reasoning failure.
5. Keeping one massive browser session alive when the work should be decomposed.

## Verification Checklist

- [ ] Cheapest path considered first
- [ ] Structured state captured before visual escalation
- [ ] Interactions verified step-by-step
- [ ] Failure mode identified before retrying
- [ ] Task decomposed if context/token load is growing
