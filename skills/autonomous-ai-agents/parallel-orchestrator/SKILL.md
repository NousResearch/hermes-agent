---
name: parallel-orchestrator
description: "Parallel fan-out for independent read-only subtasks."
version: 1.0.0
author: web3blind + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
aliases:
  - parallel
  - fanout
  - fan-out
  - parallelize
  - параллельно
  - распараллель
  - распараллелить
metadata:
  hermes:
    tags: [Delegation, Parallel, Orchestration, Research, Review, Fan-Out]
    related_skills: [hermes-agent, subagent-driven-development]
---

# Parallel Orchestrator

## Overview

Use this skill to speed up tasks that naturally split into independent parts. The pattern is simple:

1. Classify the request for parallel safety.
2. Decompose independent read-only subtasks.
3. Run them with `delegate_task(tasks=[...])` batch mode.
4. Synthesize the child results into one answer.
5. Verify that the answer covers the original request.

The goal is lower wall-clock time, not uncontrolled agent swarms. Parallelism is useful when each child can work without mutating the same files, touching shared state, or performing external side effects.

Hermes already supports parallel subagents through `delegate_task` batch mode. This skill teaches when to use that capability and when to refuse it.

## When to Use

Use this skill when the user asks for a broad task with multiple independent objects, sources, repos, PRs, products, chains, files, documents, or approaches.

Good triggers:

- “Compare these 5 protocols.”
- “Analyze governance across several blockchains.”
- “Review these PRs.”
- “Check these websites/docs/repos.”
- “Find signals from these independent sources.”
- “Generate several alternative designs/ideas.”
- “Audit independent modules read-only.”
- “Make this faster if it can be done in parallel.”

Do not use this skill for:

- A single sequential debugging chain where each step depends on the previous result.
- Code edits that touch the same files in parallel.
- Git operations such as rebase, merge, commit, push, or release unless each worker has an isolated worktree and the user explicitly wants that pattern.
- Wallet, trading, blockchain transaction, deploy, email/post/send-message, database migration, or payment actions.
- Tasks requiring live user decisions inside each child.

## Safety Contract

### Safe by Default: Read-Only Fan-Out

You may auto-parallelize when all of these are true:

- Subtasks are independent.
- The work is read-only: research, analysis, review, extraction, comparison, summarization, or classification.
- Children do not need credentials beyond normal read access already available to the parent.
- Children do not write files, commit, push, deploy, post, trade, send messages, or submit forms.
- The parent can synthesize results after all children complete.

### Ask or Stay Sequential: Shared Writes

Do not parallelize automatically if children may write to the same repo, directory, file, database, browser profile, config, or service.

For code-changing work, prefer one of these safer patterns:

1. Sequential execution in the parent.
2. One child implements a narrow task, then parent verifies.
3. Isolated git worktrees, one per independent change, with explicit user approval.
4. Read-only parallel review first, then sequential implementation.

### Hard Stop: External Side Effects

Never auto-parallelize external side effects:

- Blockchain transactions or wallet signing.
- Trading, swaps, withdrawals, limit orders, farming actions.
- Deployments, migrations, DNS changes, production config changes.
- Sending email, Telegram/Discord/Slack messages, posts, comments, reviews, or PR comments.
- Purchases, payments, account changes, API key creation, credential rotation.

If side effects are required, ask for explicit scope and run the actions sequentially unless the user has approved a safe isolated plan.

## Delegate Task Limitations

`delegate_task` is fast, but it is not durable infrastructure.

- Do not use this pattern for durable or long-running work that must survive interruption. `delegate_task` is synchronous and child work is cancelled if the parent turn is interrupted. Use cron jobs or background terminal processes for durable monitoring or long batch jobs.
- Children cannot clarify with the user. If a child would need a decision from the user, keep that decision in the parent before delegation or do not delegate that slice.
- Subagent results are leads, not verified facts. For high-impact claims, the parent must verify at least the key evidence directly: source URL, file path/line, command output, PR status, or API response. Do not report “verified” unless the parent checked it or the child returned concrete evidence that can be inspected.
- Subagents are isolated from the parent context. Put all required assumptions, constraints, paths, source preferences, output schema, and forbidden actions into each child prompt.

## Decomposition Rules

### Good Split Axes

Split by independent object:

- Blockchain: Ethereum, Solana, Cosmos, Polkadot, Near.
- Repo/PR: PR #1, PR #2, PR #3.
- Website/docs: site A, site B, site C.
- File/module: module A, module B, module C, if read-only.
- Source category: official docs, GitHub, forum, news, social.
- Design option: option A, option B, option C.

### Bad Split Axes

Avoid splits that require children to coordinate shared assumptions:

- “One child edits tests, one edits implementation” in the same files.
- “One child decides architecture, another implements before architecture is settled.”
- “One child updates package lock, another changes dependencies.”
- “Several children scrape/write the same output file.”
- “Several children drive the same browser/login session.”

### Chunking More Than the Concurrency Limit

`delegate_task` has a configurable child limit (`delegation.max_concurrent_children`, commonly 3). If there are more objects than the limit:

- Group related objects into balanced bundles.
- Run multiple waves only if the task is still worth it.
- Prefer 2–3 strong subagents over 8 tiny fragmented ones.

Example grouping for 8 protocols with max 3 children:

- Child 1: Ethereum + L2 governance.
- Child 2: Cosmos + Polkadot ecosystems.
- Child 3: Solana + Near + Sui/Aptos.

## Execution Pattern

### Step 1 — Classify

Before delegating, state the classification internally:

- `parallelizable`: yes/no
- `risk`: read-only / shared-write / external-side-effect
- `split_axis`: object/source/module/option/etc.
- `max_children`: from config/tool limit, default 3 unless known otherwise
- `synthesis_goal`: what the final answer must decide or explain

If `risk` is not read-only, do not auto-parallelize.

### Step 2 — Build Self-Contained Child Tasks

Each child must receive enough context to work without reading the parent conversation.

A child prompt should include:

- The specific object(s) assigned to the child.
- The original user question.
- Exact output schema.
- Read-only constraint.
- Source preferences and verification requirements.
- What not to do.

Do not tell every child to solve the full task. Give each child a bounded slice.

### Step 3 — Call `delegate_task` in Batch Mode

Use one tool call with `tasks=[...]` instead of serial child calls.

Template:

```python
delegate_task(
    tasks=[
        {
            "goal": "Analyze governance model for Ethereum and major L2s",
            "context": "Original request: compare governance across several blockchains. Read-only research only. Use only the tools already available to this parent run. Return: summary, key mechanisms, risks, sources, confidence."
        },
        {
            "goal": "Analyze governance model for Cosmos and Polkadot ecosystems",
            "context": "Original request: compare governance across several blockchains. Read-only research only. Use only the tools already available to this parent run. Return: summary, key mechanisms, risks, sources, confidence."
        },
        {
            "goal": "Analyze governance model for Solana, Near, and Move-based ecosystems",
            "context": "Original request: compare governance across several blockchains. Read-only research only. Use only the tools already available to this parent run. Return: summary, key mechanisms, risks, sources, confidence."
        }
    ]
)
```

For code review without edits:

```python
delegate_task(
    tasks=[
        {
            "goal": "Read-only review of authentication module",
            "context": "Inspect files under src/auth and tests/auth. Do not edit files. Use static reads and non-mutating diagnostics only. Return critical issues, important issues, minor issues, and evidence with paths/lines."
        },
        {
            "goal": "Read-only review of API routing module",
            "context": "Inspect files under src/api and tests/api. Do not edit files. Use static reads and non-mutating diagnostics only. Return critical issues, important issues, minor issues, and evidence with paths/lines."
        }
    ]
)
```

### Step 4 — Synthesize

After children finish, the parent must synthesize. Do not paste child summaries one after another without integration.

Synthesis should include:

- Direct answer to the original request.
- Cross-comparison and contradictions.
- Confidence level and source quality.
- Gaps or items not checked.
- Practical recommendation or next step if relevant.

### Step 5 — Verify Coverage

Before finalizing, check:

- Did every requested object/source get covered?
- Did any child exceed scope or perform side effects?
- Are claims grounded in child outputs or tool results?
- Are contradictions resolved or clearly labeled?
- Is the final answer shorter and more useful than raw child outputs?

## Output Schemas for Children

### Research Child

Ask each child to return:

```text
Object(s):
Summary:
Key facts:
Evidence / sources:
Risks or caveats:
Confidence: high / medium / low
Open questions:
```

### Code Review Child

Ask each child to return:

```text
Scope reviewed:
Critical issues:
Important issues:
Minor issues:
Evidence with paths/lines:
Suggested fixes:
Confidence:
```

### PR Triage Child

Ask each child to return:

```text
PR/repo:
State:
CI status:
Mergeability/conflicts:
Changed-file scope:
Likely action: merge-ready / fix CI / rebase / rebuild / close
Evidence:
Risks:
```

### Design Alternatives Child

Ask each child to return:

```text
Option:
Core idea:
Strengths:
Weaknesses:
Implementation complexity:
Accessibility implications:
Best use case:
```

## Parent Tool Availability

Batch `delegate_task` items do not accept per-child tool controls in the current model schema. Children inherit the parent run's enabled tool surface. Do not document or pass model-facing per-task tool restrictions in the batch objects.

Instead, keep fan-out safe with prompt-level constraints:

- Web research: tell children to use read-only public sources and return URLs.
- GitHub/repo read-only inspection: tell children to use static file reads, `git diff`, `git show`, `git status`, collection-only tests, and equivalent non-mutating diagnostics.
- Browser QA: use only when visual or interactive behavior matters, and keep actions read-only.
- Document extraction: distinguish URL extraction from local document reads in the child context.
- Mixed research + local files: name the allowed source classes and forbid writes explicitly.

Avoid giving children broad permission by default. Fewer available tools in the parent run reduce cost, risk, and context noise; child prompts still need explicit boundaries.

When children can use `terminal`, constrain commands to read-only inspection: `git diff`, `git show`, `git status`, `python -m pytest --collect-only`, static reads, and equivalent non-mutating diagnostics. Do not let children install packages, run formatters, update lockfiles, write reports, run migrations, start services that mutate state, or execute commands that modify the workspace.

## Output Contract

The parent response must be one synthesized answer, not a concatenation of child reports.

Include:

- The final answer or recommendation.
- Which slices were delegated and which completed.
- Key evidence that supports high-impact claims.
- Contradictions, uncertainty, and missing slices.
- Any follow-up action that should remain sequential or require explicit approval.

Do not claim a result is verified only because a child said so. Say “child-reported” or “needs parent verification” when evidence was not directly checked by the parent.

## Examples

### Multi-Blockchain Governance Analysis

User asks:

```text
Compare governance in Ethereum, Solana, Cosmos, Polkadot, and Near.
```

Good plan:

- Child 1: Ethereum and L2 governance.
- Child 2: Solana and Near governance.
- Child 3: Cosmos and Polkadot governance.
- Parent: compare upgrade process, token voting, councils/foundations, validator role, on-chain/off-chain split, decentralization risks.

### PR Backlog Triage

User asks:

```text
Check the status of my open PRs and tell me what needs action.
```

Good plan:

- Child 1: PRs 1–3.
- Child 2: PRs 4–6.
- Child 3: PRs 7–9.
- Parent: produce final action list grouped by ready/fix/rebase/close.

Do not let children comment on GitHub or close PRs. Triage is read-only unless the user explicitly asks for actions after the synthesis.

### Website Accessibility Spot Checks

User asks:

```text
Check these three landing pages for obvious accessibility issues.
```

Good plan:

- One child per page or group pages by domain.
- Children inspect read-only with browser/web tools.
- Parent deduplicates issues and prioritizes fixes.

### Codebase Architecture Read-Only Review

User asks:

```text
Review architecture risks in this repo.
```

Good plan:

- Child 1: entry points and routing.
- Child 2: persistence/config/secrets handling.
- Child 3: tests/CI/deployment scripts.
- Parent: combine into risk-ranked architecture review.

Do not let children edit files unless the user explicitly asked for implementation and isolation is safe.

## Common Pitfalls

1. **Parallelizing sequential debugging.** If each step depends on the previous result, stay sequential.

2. **Giving every child the full task.** That creates duplicate work and inconsistent assumptions. Give each child a slice.

3. **Skipping synthesis.** The parent must integrate results into one answer; raw child dumps are not enough.

4. **Parallel writes in one repo.** This causes conflicts and subtle regressions. Use read-only review or isolated worktrees.

5. **Too many children.** More children can be slower and more expensive. Use 2–3 meaningful children unless the task clearly benefits from more.

6. **Broad parent tool surface by habit.** Start the parent run with only the tools the children will actually need for their slices.

7. **Letting children perform side effects.** Subagents should not post, send, trade, deploy, merge, or mutate external systems during auto-parallel work.

8. **Forgetting source quality.** Parallel research can amplify weak sources. Require evidence and confidence from each child.

9. **Assuming speedup is free.** Wall-clock time may improve, but token/API usage rises. Use parallelism when the time savings matter.

10. **Ignoring child failures.** If one child fails, synthesize what is known and clearly label the missing slice, or rerun only that slice.

## Verification Checklist

Before final response:

- [ ] The task was classified as safe for parallel read-only work, or explicit approval/isolation was used.
- [ ] Each child had a distinct bounded scope.
- [ ] The parent run's enabled tool surface was narrow enough for the delegated slices.
- [ ] No child was asked to perform external side effects.
- [ ] Results were synthesized, not pasted.
- [ ] Coverage of the original request was checked.
- [ ] Contradictions and uncertainty were labeled.
- [ ] Final answer includes practical next steps when useful.

## Quick Test Checklist

Use this quick check before applying the skill to a real request:

- [ ] Can the request be split into at least two independent slices?
- [ ] Is the first wave read-only?
- [ ] Does each child have a self-contained prompt and output schema?
- [ ] Is the parent tool surface appropriate for every child slice?
- [ ] Are `terminal` commands constrained to non-mutating inspection when used?
- [ ] Is there a parent synthesis and evidence-verification step?
- [ ] Would a parent interruption be acceptable? If not, use cron/background processes instead.

## Done Criteria

The skill run is complete only when:

- All delegated slices have either completed or are explicitly marked missing/failed.
- The parent synthesized the results into one coherent answer.
- High-impact claims have inspectable evidence or are labeled as unverified child reports.
- No child performed external side effects or shared-state writes.
- The final answer addresses the original user request and states any remaining uncertainty.

## Quick Invocation Examples

After loading this skill, suitable user prompts include:

```text
parallel compare governance across Ethereum, Solana, Cosmos, Polkadot, and Near
```

```text
параллельно проверь эти 6 PR и скажи, какие готовы к merge
```

```text
распараллель анализ этих документаций и собери общий вывод
```

```text
быстрее: проверь три сайта на очевидные accessibility-проблемы
```
