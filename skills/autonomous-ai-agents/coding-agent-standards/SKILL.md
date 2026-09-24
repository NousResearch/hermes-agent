---
name: coding-agent-standards
description: "Use when authoring guidelines or running autonomous coding agents. Codifies senior-engineer judgment: think before coding, simplicity first, surgical changes, goal-driven execution."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Standards, Guidelines, Quality, Autonomous]
    related_skills: [claude-code, codex, opencode, hermes-agent-skill-authoring]
---

# Coding Agent Standards

Behavioral guidelines to reduce common autonomous coding mistakes. Apply to any sub-agent, delegated task, or自动化 coding flow.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

When a delegator gives a vague task like "fix the auth bug" — don't guess scope. Clarify:
- Which auth flow? (login, token refresh, permission check?)
- What constitutes "fixed"? (no errors, specific test passes, manual verification?)
- Any files off-limits?

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

**The "seems excessive" test:** If a delegator says "this seems excessive" — you've already failed. Catch it before sending.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless explicitly asked.

**The test:** Every changed line should trace directly to the user's request.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan before starting:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5. Kanban Discipline (when applicable)

When working a task tracked on a Kanban board:
- Update card status explicitly (not just verbal update in chat)
- Move to in-progress when starting, done/awaiting-review when claiming complete
- If blocked, note the blocker on the card before asking — don't just say "blocked" in chat
- "Done" means: code written, tested, and board updated — not just "code written"

---

## 6. Delivery Discipline

Results go where specified, not wherever the agent decides.

- If the task specifies a delivery channel (Discord, Telegram, etc.), deliver there
- If not specified, deliver to the delegator's home channel
- Cron summaries and outputs go to the assigned destination channel, not the main comms channel
- Don't mix task results with status chatter — separate deliverables from updates

---

## Common Pitfalls

1. **"While I'm in here" syndrome** — Unrelated improvements snuck into a PR. Fight it. Every diff line must trace to the request.

2. **Overcomplicated abstractions** — Adding "flexibility" that wasn't asked for. Ship the simplest thing that works.

3. **Assuming context** — Sub-agents should not assume they have full context. Check before assuming the delegator knows something.

4. **Skipping verification** — Declaring done before testing. Use the goal-driven execution loop.

5. **Vague success criteria** — "Make it work" requires constant clarification. Define what "works" looks like before starting.

---

## Verification Checklist

- [ ] Task scope confirmed before writing code
- [ ] No speculative features or abstractions added
- [ ] Diff traces directly to the request — nothing more
- [ ] Own orphans removed (unused imports/variables from your changes)
- [ ] Tests written first (or alongside) — not as afterthought
- [ ] Board/card status updated if Kanban-tracked
- [ ] Results delivered to the correct channel
- [ ] "Seems excessive" test passed before sending

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.