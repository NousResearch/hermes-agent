---
name: grok-build-patterns
description: "Use when running on the Grok (xAI) provider in Hermes and you want to apply battle-tested patterns from xAI's Grok Build CLI: best-of-n tournaments, systematic review loops, plan-then-execute, check-work verification, and subagent orchestration. Port these disciplines for higher quality autonomous work."
version: 0.1.0
author: trefong (via Grok Build)
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [grok, xai, best-practices, subagents, review, best-of-n, planning, quality]
    related_skills: [grok-xai-oauth, subagent-driven-development, writing-plans, hermes-agent-skill-authoring]
---

# Grok Build Patterns for Hermes (on Grok provider)

When your Hermes session is using the xAI Grok OAuth provider (see the `grok-xai-oauth` skill), you have access to one of the strongest reasoning engines available. Pair it with the hard-won agentic software engineering disciplines from xAI's own Grok Build CLI.

This skill teaches you (the agent) how to run "Grok Build style" inside Hermes using `delegate_task`, fresh subagents, reviews, and verification.

## Core Principle
Grok is excellent at:
- Comparative judgment (best-of-n)
- Structured planning + execution with reviews
- Detecting issues via systematic checklists (`check-work`)
- High-agency implementation with verification

Don't waste it on rote file edits. Use it for the hard parts: exploration, decision making, review, synthesis. Delegate or mix with faster tools for the mechanical work.

## Pattern 1: Best-of-N (Parallel Candidates + Synthesis)
See the expanded example in the `grok-xai-oauth` skill under "Best-of-N + Grok Reasoning".

General recipe:

1. Break the uncertain decision into N distinct approaches.
2. `delegate_task` N times in parallel (different prompts or constraints, all on Grok).
3. Collect outputs.
4. One final delegate (or the main thread) on Grok does structured comparison and picks/merges.

Use when: architecture choices, refactor strategies, API designs, UX approaches, algorithm selection.

## Pattern 2: Plan → Review → Execute (with Gates)
1. Use or emulate `writing-plans` to produce a detailed plan with small tasks.
2. For each task (or groups of tasks):
   - Dispatch implementer subagent(s) on Grok (or mix).
   - Dispatch spec-compliance reviewer.
   - Dispatch quality/safety reviewer (`requesting-code-review` style or `check-work`).
   - Only proceed when both reviewers PASS.
3. Final integration review.

This is exactly the two-stage (spec then quality) review from `subagent-driven-development`, supercharged when the brain is Grok.

## Pattern 3: Systematic Verification (`check-work` style)
After any non-trivial change or at milestones:

- Run full relevant test suite (or the project's test command).
- Lint / type check.
- Manual or agent review of diffs for the original requirements.
- Security / edge case scan.
- Update docs / examples if user-facing.

Never declare victory until a checklist like the Verification sections in peer skills is green.

## Pattern 4: Handoff Between Grok Build CLI and Hermes
(See the Grok Build Synergy section in `grok-xai-oauth`.)

- Heavy exploration, initial implementation, and multi-review polishing → do in this Grok Build CLI (excellent MCPs, worktrees, precise diffs, your local `~/.grok/skills/`).
- Long-running, always-on, multi-platform (Telegram etc.), scheduled, or memory-persistent work → hand off to Hermes running on the same Grok subscription.
- Use the Evidence Ledger / concierge patterns for shared memory that both tools can read/write.

## How to Invoke These Patterns
Load this skill + `grok-xai-oauth` + `subagent-driven-development` + `writing-plans` when the user asks for high-quality work on the Grok provider.

Example user prompt that should trigger this skill:
"Using the Grok provider, best-of-n 3 different approaches to the caching layer, then implement the winner with full reviews."

## Adapting Specific Grok Build Skills
The following `~/.grok/skills/` map well and are worth porting or emulating as first-class Hermes skills:

- best-of-n
- check-work
- implement (the full loop)
- review / requesting-code-review
- design
- execute-plan
- best-of-n (already referenced)

If you're the human maintainer, consider extracting the best reusable pieces from your `~/.grok/skills/` into this Hermes category so autonomous agents on Grok get the same superpowers.

Example: the `check-work` skill in Grok Build uses subagents for verification. When on Grok in Hermes, you can replicate by delegating "review this change against the plan" tasks.

See the user's local `~/.grok/skills/` for the canonical implementations (many use the spawn_subagent / worktree model that maps to Hermes `delegate_task` + isolation).

## Verification (for any work done under this skill)
- [ ] Used fresh subagent(s) via delegate_task where isolation mattered.
- [ ] At least one explicit review/gate step happened (spec or quality).
- [ ] A structured comparison or checklist was produced (table or bullet list).
- [ ] Final output references which Grok Build pattern was applied.
- [ ] If visuals or X search were involved, they went through the Grok provider.

## References
- `grok-xai-oauth` — the foundation for running on Grok in Hermes.
- `subagent-driven-development` — the closest existing peer pattern in the repo.
- Your local `~/.grok/skills/best-of-n/SKILL.md`, `check-work/SKILL.md`, `review/SKILL.md` etc. for the source patterns.
- xAI Grok Build announcement and docs.

**This is early (v0.1).** Improve it by adding more concrete delegate_task transcripts from real runs on the Grok provider.

---

**Created while taking the reins on Grok Build + Hermes contribution work.**
