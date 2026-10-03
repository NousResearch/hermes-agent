---
name: better-harness
description: "Use when reviewing a coding project's agent work loop (harness): task understanding, controlled execution, change validation, reliable delivery, learning capture. Runs three parallel evidence passes and outputs prioritized findings with evidence-bound scores. Adapted from QoderAI/better-harness."
version: 1.0.0
author: Hermes Agent (adapted from QoderAI/better-harness, MIT)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [harness, work-loop, review, evidence, meta-review, process, quality]
    related_skills: [requesting-code-review, systematic-debugging, test-driven-development, plan]
---

# Better Harness (Hermes adaptation)

Review the workflow *around the diff*, not the diff itself. Hermes agents change
code fast, but the loop around them is often the weak point: fuzzy goals,
improvised steps, "it works" without proof, speed over safeguards, lessons lost.
This skill turns project and session evidence into five-dimension scores and
prioritized findings — each tied to its evidence, smallest owner, expected
outcome, and validation route.

Adapted from [QoderAI/better-harness](https://github.com/QoderAI/better-harness)
(MIT). The original ships a Qoder/Codex CLI; this version runs on Hermes-native
tools only (`terminal`, `search_files`, `read_file`, `session_search`,
`skills_list`, `memory`, `delegate_task`).

## When to Use

- User asks to review the "harness", "work loop", "workflow", or "process" of a
  project — not the code itself
- Recurring friction: repeated agent errors, repeated user corrections, skipped
  review, setup/validation failures that keep coming back
- Before a long task, to establish a task-bounded baseline of the project's
  agent-readiness
- After a series of sessions on one project, to check what the loop learned
- **Don't use for:** reviewing a single diff (use `requesting-code-review`),
  debugging one failure (`systematic-debugging`), or one-off questions

**Scope modes:** `quick` (3 assets or Task Episodes, previous 7 days) or
`normal` (5, previous 30 days). Default `normal` unless the user asks for quick.

## Step 1 — Resolve Scope and Collect the Evidence Bundle

Resolve and state: the target repo, the decision/acceptance boundary, risks,
locale (user's request language), scope mode, and evidence window. Stop if the
target, its git root, or the requested owner is missing — never substitute
another project or guess a path.

Collect ONE versioned evidence bundle per scope:

```bash
git status --short && git log --oneline -30 --since="<window>"   # project state
git diff --stat HEAD~1 HEAD                                      # current change
```

Plus: read `README.md` / `AGENTS.md` / `CONTRIBUTING.md` (context map),
`search_files` for test/CI config, and — for the Hermes layer —
`skills_list` (categories), `memory` targets (which memories exist),
`cronjob list`, and the active profile's config. Record counts as *routing
leads only*: zero or high counts never create findings or scores by themselves.

Freeze the bundle: target, provider scope (Hermes profile), window, depth,
and authority. A configured capability is **not** observed use — inventory
proves presence at most.

## Step 2 — Run Three Independent Evidence Passes

Launch exactly three fresh, read-only leaf subagents **in parallel** with
`delegate_task(tasks=[...])`. Children get ONLY their own brief — no parent
conclusions, no other briefs. Children must not delegate.

Give each child: its reference file path, the scope, the relevant slice of the
bundle, and its output contract. **Each child must end its brief with "the
claims the lead must not make from this evidence."** Children never assign
final severity or scores.

| Agent | Evidence lane | Input | Read | Must not touch |
| --- | --- | --- | --- | --- |
| 1 | Session Evidence | `session_search` hits in window, episode limit | `references/evidence-passes.md` §1 | project files, configured assets, raw sessions beyond window |
| 2 | Project Harness | git history slice, current diff, repo files | `references/evidence-passes.md` §2 | session facts, user memory, other briefs |
| 3 | Agent Customize | skills/memory/cron/plugin inventory envelopes | `references/evidence-passes.md` §3 | app code, session facts, memory bodies beyond authority |

Stop if the bundle is failed or a lane is unavailable: in `normal` mode any
unavailable lane blocks the report; in `quick` mode it lowers confidence and
every unavailable lane stays explicit. Never replace missing evidence with an
`Unobserved`-disguised conclusion.

## Step 3 — Lead Reconciliation and Scoring

Read `references/agent-work-loop.md` for the five dimensions, fifteen checks,
evidence states, and scoring rules. Perform ONE reconciliation:

- Retain every supported candidate from all three lanes. Merge only candidates
  with the same target, observed consequence, owner, and repair route.
- Keep a working reason for every unsupported or deferred candidate. Never drop
  an eligible finding to reach a number, shorten the report, or simplify a
  score.
- The lead alone validates consequence, cause chain, smallest owner, evidence
  boundary, confidence, and verifier; assigns final severity and ONE primary
  check per finding; derives conservative dimension scores from the evidence
  ceiling ladder.
- Freeze severity and dimension scores BEFORE shaping priority moves, repair
  prompts, or reader copy.

Five findings is a coverage floor for an evidence-rich report, never a target;
fewer are valid when evidence is sparse. Counts, filenames, asset presence,
severity, age, or score alone are never findings.

## Step 4 — Findings and Quality Gates

Read `references/quality-gates.md` and apply its gates before rendering. Each
finding carries: `title` (a concrete observed consequence, not a number-laden
label), `reason` (facts separated from inference, owner, uncertainty),
`severity`, `dimensionRefs` + primary check, `owner`, `aiFixPrompt` (scoped
repair, every command already discovered in the target), `expectedOutput`
(verifiable), and `verifier`. Any failed gate blocks the report.

## Step 5 — Render the Report and Follow Up

Render a Markdown report from `templates/harness-report.md` (write to
`.hermes/harness-report-<date>.md` in the target or print inline for quick
mode). Include: five-dimension scorecard, the retained findings, and the
selected support track — Bootstrap (0→1), Operationalize (1→60),
Optimize (60→100), or Undetermined. The track shapes at most three priority
moves; it never adds findings or rescales scores.

**Finding-bound fix:** repairing one finding authorizes exactly that repair.
Apply it at the smallest owner, run owner-owned validation, then launch ONE
fresh read-only subagent to judge the repair `verified | partial | blocked`
from the locked pre-fix report + actual outputs + post-fix validation. This
updates Repair Progress only. Loop Effectiveness (the five scores) changes only
after a comparable later Task Episode shows the repaired mechanism was routed,
applied, and improved the result without guardrail regression. If the fix
target was a Hermes asset (skill/memory/cron), prefer `skill_manage` /
`memory` / `cronjob` APIs over hand-editing files.

## Common Pitfalls

1. **Reviewing the diff instead of the loop.** This skill judges the workflow
   system, not code quality. If the user wants code review, load
   `requesting-code-review` instead.
2. **Scoring above the evidence.** `Present ≤ 74`, `Wired ≤ 84`,
   `Exercised ≤ 94`, only a later comparable outcome reaches 100. A score
   above its evidence ceiling is the #1 review defect.
3. **Counts as findings.** "Project has zero skills" or "23 memories exist" is
   a routing lead, never a finding. A finding needs an inspected gap + bounded
   impact + smallest owner + validation route.
4. **Filling the quota.** Five findings is a floor for rich evidence, not a
   required count. Sparse evidence → fewer findings, and say why.
5. **Leaking private data.** Never copy session IDs, absolute user-home paths,
   memory bodies, prompts, or secrets into the report. Redact to semantic
   facets (role, duration, aggregate failures).
6. **Letting children score.** Evidence agents propose candidates; only the
   lead assigns severity and scores, after freezing.
7. **Repair credit in the same window.** A just-fixed finding updates Repair
   Progress, never Loop Effectiveness. Same-window repair is not evidence that
   later tasks improved.
8. **Silently dropping an unavailable lane.** Unavailable stays explicit
   (`Unobserved` with a reason) — in normal mode it blocks the report rather
   than being papered over.

## Verification Checklist

- [ ] Scope (target, window, depth, authority) resolved and stated before any
      delegation
- [ ] Exactly three leaf evidence agents, parallel, isolated briefs, no
      delegation, no scoring
- [ ] Every child ended with "claims the lead must not make"
- [ ] One reconciliation performed; every candidate kept, merged, or deferred
      with a reason
- [ ] Severity + dimension scores frozen before drafting priority moves
- [ ] Every finding: consequence + owner + evidence + repair + verifier;
      no count-only findings
- [ ] All quality gates passed (see `references/quality-gates.md`)
- [ ] Report rendered from template; support track selected; no score changes
      from track
- [ ] No private data in output (session ids, paths, memory bodies, secrets)
- [ ] Finding-bound fix (if any) applied at smallest owner + independent
      verified/partial/blocked review; Loop Effectiveness untouched

## Attribution

Methodology and evidence model adapted from
[QoderAI/better-harness](https://github.com/QoderAI/better-harness) (MIT):
Agent Work Loop (5 dimensions × 15 checks), evidence-state ceiling ladder,
three-lane evidence collection, findings quality gates, and support tracks.
Hermes adaptation replaces the Qoder/Codex CLI with Hermes-native tools and
renders Markdown instead of host Canvas/HTML reports.
