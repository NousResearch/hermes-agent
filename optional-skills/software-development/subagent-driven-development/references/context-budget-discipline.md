# Context Budget Discipline

Practical rules for keeping orchestrator context lean when spawning subagents or reading large artifacts. Use these whenever you're running a multi-step agent loop that will consume significant context — plan execution, subagent orchestration, review pipelines, multi-file refactors.

Adapted from the GSD (Get Shit Done) project's context-budget reference — MIT © 2025 Lex Christopherson ([gsd-build/get-shit-done](https://github.com/gsd-build/get-shit-done)).

## Universal rules

Every workflow that spawns agents or reads significant content must follow these:

1. **Never read agent definition files.** `delegate_task` auto-loads them — you reading them too just doubles the cost.
2. **Never inline large files into subagent prompts.** Tell the agent to read the file from disk with `read_file` instead. The subagent gets full content; your context stays lean.
3. **Read depth scales with context window.** See the table below.
4. **Delegate heavy work to subagents.** The orchestrator routes; it doesn't execute.
5. **Proactively warn** the user when you've consumed significant context ("Context is getting heavy — consider checkpointing progress before we continue").

## Read depth by context window

Check the model's actual context window (not "it's Claude so 200K"). Some Sonnet deployments are 1M, some are 200K. If you don't know, assume the smaller one — err toward leanness.

| Context window | Subagent output reading | Summary files | Verification files | Plans for other phases |
|----------------|-------------------------|---------------|--------------------|-----------------------|
| < 500k (e.g. 200k) | Frontmatter only | Frontmatter only | Frontmatter only | Current phase only |
| >= 500k (1M models) | Full body permitted | Full body permitted | Full body permitted | Current phase only |

"Frontmatter only" means: read enough to see the final status/verdict/conclusion. If the subagent wrote a 3000-line debug log, read the summary section it produced, not the log.

## Four-tier degradation model

Monitor your context usage and shift behavior as you climb the tiers. The point is to notice *before* you hit the wall, not when responses start truncating.

| Tier | Usage | Behavior |
|------|-------|----------|
| **PEAK** | 0 – 30% | Full operations. Read bodies, spawn multiple agents in parallel, inline results freely. |
| **GOOD** | 30 – 50% | Normal operations. Prefer frontmatter reads. Delegate aggressively. |
| **DEGRADING** | 50 – 70% | Economize. Frontmatter-only reads, minimal inlining, **warn the user** about budget. |
| **POOR** | 70%+ | Emergency mode. **Checkpoint progress immediately.** No new reads unless critical. Finish the current task and stop cleanly. |

## Early warning signs (before panic thresholds fire)

Quality degrades *gradually* before hard limits hit. Watch for these:

- **Silent partial completion.** Subagent claims done but implementation is incomplete. Self-checks catch file existence, not semantic completeness. Always verify subagent output against the plan's must-haves, not just "did a file appear?"
- **Increasing vagueness.** Agent starts using phrases like "appropriate handling" or "standard patterns" instead of specific code. This is context pressure showing up before budget warnings fire.
- **Skipped protocol steps.** Agent omits steps it would normally follow. If success criteria has 8 items and the report covers 5, suspect context pressure, not "the agent decided 5 was enough."

When these signs appear, checkpoint the work and either reset context or hand off to a fresh subagent.

## Fundamental limitation

When you orchestrate, you cannot verify semantic correctness of subagent output — only structural completeness ("did the file appear?", "does the test pass?"). Semantic verification requires either running the code yourself or delegating a review pass to another fresh subagent.

**Mitigation:** in every task you delegate, include explicit "must-have" truths the subagent must confirm in its response (e.g., "confirm your test actually tests X, not just that X was imported"). The subagent re-asserting concrete facts is evidence; vague summaries are not.
