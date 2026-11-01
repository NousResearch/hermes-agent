---
name: team-of-thoughts
description: "Kanban-based multi-agent deliberation for RCA and feature design."
version: 1.0.0
author: Ramiz Mehran (ramizmehran) <mehranramiz93@gmail.com>
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [kanban, multi-agent, deliberation, rca, feature-design, squad]
    related_skills: [adversarial-debate, kanban-orchestrator, subagent-driven-development]
    config:
      delegation.max_concurrent_children: 4
---

# Team of Thoughts

Multi-turn kanban debate across 4 agents (Clio, Hephaestus, Solon, Talaria) to lock an RCA or feature plan before handing off to a coding CLI. The default protocol for medium-stakes decisions where structured async review is sufficient.

**Core principle:** Async kanban comments beat synchronous chat for routine deliberation. Each agent works independently, then Hermes synthesizes. For high-stakes irreversible decisions, see the `adversarial-debate` skill instead.

## When to Use

- A non-trivial bug needs rigorous RCA before coding
- A feature has ≥2 defensible designs
- The user wants the Squad to research, debate, and lock a plan before any code is written
- **Deliverable validation:** Docs, scripts, or configs produced need cross-validation before shipping
- The user explicitly asks for "team review", "run through ToT", or "ask the squad"

### When NOT to use (use adversarial-debate instead)

- The outcome has significant irreversible consequences (one-way door)
- A previous kanban ToT produced weak or conflicting synthesis
- The user explicitly asks for "full team debate", "adversarial review", or "cross-examine this"

### Protocol selector

| User says | Stakes | Protocol |
|-----------|--------|----------|
| "run through ToT" / "ask the team" | Low-medium | Kanban ToT (this skill) |
| "full team debate" / "adversarial review" | High | Adversarial Debate (`adversarial-debate` skill) |

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Kanban CLI (`hermes kanban`) | Available in all Hermes installations |
| Agent profiles | clio, hephaestus, solon, talaria must exist |
| `delegation.max_concurrent_children` | ≥3 for parallel dispatch |

## How to Run

Load this skill in Hermes. Follow the Procedure below to bootstrap kanban cards, dispatch agents, and synthesize. The `scripts/tot-start.sh` helper can automate card creation.

```bash
~/.hermes/skills/team-of-thoughts/scripts/tot-start.sh \
  "Topic" "rca|feature" "/path/to/project" "<optional-parent-task-id>"
```

## Quick Reference

```
0. Set working directory
1. Bootstrap board (create root + child cards)
2. Hermes opens the debate comment
3. Prepare child cards (assign, unlink, promote)
4. Dispatch agents (parallel rounds)
5. Agents post findings as kanban_comments on root card
6. Synthesis — locked plan comment on root card
7. Execution hand-off via coding CLI
```

## Procedure

### 0. Set the working directory

All kanban commands must run from the project directory being debated:

```bash
cd /path/to/project
hermes kanban boards  # verify active board
```

### 1. Bootstrap the board

Create or switch to the correct kanban board:

```bash
hermes kanban boards create <slug> --name "Display Name" --description "..." --switch
```

Create the root card with rich context (goal, evidence, current state):

```bash
ROOT=$(hermes kanban create "ToT: <Topic>" \
  --triage \
  --body "Full context here" \
  --json 2>&1 | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "ROOT=$ROOT"
```

Create 4 child cards — one per agent — with `--assignee` so the dispatcher auto-claims:

```bash
hermes kanban create "🏛️ Clio: <Topic>" \
  --body "**PERSPECTIVE:** Clio the historian — research and evidence grounding.

**GOAL:** Investigate <topic> with questions:
1. What already exists?
2. What are the relevant patterns/docs?
3. What evidence supports each approach?

Post full findings as **kanban_comment** on root card $ROOT before completing." \
  --assignee clio

hermes kanban create "🔨 Hephaestus: <Topic>" \
  --body "**PERSPECTIVE:** Hephaestus the builder — feasibility and integration cost.

**GOAL:** Evaluate <topic> for:
1. Implementation complexity
2. Integration cost with existing code
3. Testability and risk factors

Post full findings as **kanban_comment** on root card $ROOT before completing." \
  --assignee hephaestus

hermes kanban create "⚖️ Solon: <Topic>" \
  --body "**PERSPECTIVE:** Solon the strategist — trade-off analysis.

**GOAL:** Analyze <topic> for:
1. Trade-offs between approaches
2. Long-term maintenance impact
3. Strategic alignment

Post full findings as **kanban_comment** on root card $ROOT before completing." \
  --assignee solon

hermes kanban create "✉️ Talaria: <Topic>" \
  --body "**PERSPECTIVE:** Talaria the messenger — mechanical and UX lens.

**GOAL:** Review <topic> for:
1. User-facing impact and edge cases
2. Naming conventions and consistency
3. Small-scope improvements

Post full findings as **kanban_comment** on root card $ROOT before completing." \
  --assignee talaria
```

Cards created with `--assignee` land in `ready` status — the dispatcher may auto-claim them. Run a dispatch to catch any stragglers.

Do NOT include implementation instructions (file paths, function signatures) in debate card bodies. Research cards investigate — they do not code.

### 2. Hermes opens the debate

Move the root card from `triage` to `running`:

```bash
hermes kanban specify <root-id>
hermes kanban claim <root-id>
```

If `specify` fails with an LLM error, use SQLite to move directly to `ready`:

```bash
python3 -c "
import sqlite3
db = sqlite3.connect('$HOME/.hermes/kanban/boards/<board>/kanban.db')
db.execute(\"UPDATE tasks SET status='ready' WHERE id='<root-id>'\")
db.commit()
db.close()
"
hermes kanban claim <root-id>
```

Post the opening debate comment on the root card naming the topic, expected output, and round count (default 2):

```bash
hermes kanban comment <root-id> "## Opening — <Topic>

**Format:** <rca|feature>
**Rounds:** 2
**Expected output:** Locked plan with rationale and rejected alternatives."
```

### 3. Prepare child cards for dispatch

Assign each child card to its agent profile:

```bash
hermes kanban assign <child-clio-id> clio
hermes kanban assign <child-hephaestus-id> hephaestus
hermes kanban assign <child-solon-id> solon
hermes kanban assign <child-talaria-id> talaria
```

Promote children to `ready` (use `--force` if parent link blocks):

```bash
hermes kanban promote --force <child-clio-id>
hermes kanban promote --force <child-hephaestus-id>
hermes kanban promote --force <child-solon-id>
hermes kanban promote --force <child-talaria-id>
```

If parent-child links block the dispatcher (claim rejects with `parents_not_done`), remove them:

```bash
hermes kanban unlink <root-id> <child-clio-id>
hermes kanban unlink <root-id> <child-hephaestus-id>
hermes kanban unlink <root-id> <child-solon-id>
hermes kanban unlink <root-id> <child-talaria-id>
```

### 4. Dispatch agents (parallel rounds)

```bash
hermes kanban dispatch --max 4
```

For each round, each agent:
- Reads all prior comments on the root card
- Posts one `kanban_comment` with `### Round <N> — <Agent>` and their analysis
- Does NOT execute code — only debates

Run 2 rounds minimum for non-trivial tasks.

To run subsequent rounds, post a Round 2 challenge comment on the root card summarizing Round 1 findings, then create new child cards for Round 2.

### 5. Synthesis

After all rounds complete, read all agent comments and write the synthesis:

```bash
hermes kanban comment <root-id> "$(cat << 'SYNTHESIS'
### Synthesis — Locked Plan

**Decision:** <final approach>

**Rationale:** <why this approach wins>

**Rejected alternatives:**
- <alternative 1>: <why rejected>
- <alternative 2>: <why rejected>

**Execution plan:**
- [ ] Task 1: <description>
- [ ] Task 2: <description>
- [ ] Task 3: <description>

**Dependencies:** <any ordering constraints>
SYNTHESIS
)"
```

Complete the root card:

```bash
hermes kanban complete <root-id> --summary "Synthesis posted"
```

### 6. Execution hand-off

Two valid paths:

**Path A — Execution card (complex multi-step):**

```bash
hermes kanban create "Execute: <Topic>" \
  --assignee hephaestus \
  --body "Use synthesis from root card $ROOT. Implement all tasks."
```

**Path B — Direct CLI dispatch (single well-scoped fix):**

Write a self-contained executor prompt from the synthesis and dispatch via the active coding CLI:

```bash
codex exec --sandbox danger-full-access "Read /tmp/plan.md and implement all changes."
```

**Rule of thumb:** If the fix touches >3 files or multiple components, use Path A. If single-component with exact code already drafted, Path B is faster.

## Pitfalls

### Parent-child link blocks dispatcher claim
Even with `promote --force`, child cards with `--parent` links are rejected by the dispatcher at claim time (`parents_not_done`). Remove the parent link with `hermes kanban unlink <root-id> <child-id>` before dispatching. The card remains parented in the DB for display; only the dispatcher block is lifted.

### Cross-profile project context contamination
Agents dispatched in different profiles may have `environment_hint` or `terminal.cwd` pointing to a different project. Include explicit project context in every child card body:

```markdown
**IMPORTANT — WORK IN THIS PROJECT:** /path/to/correct/project
**DO NOT** work in /other/project — that is a different project.
```

### Agents may not post comments to root cards automatically
The kanban-worker skill handles card lifecycle (claim → work → complete) but does not enforce posting findings as `kanban_comment`. Every child card body MUST end with an explicit instruction:

```markdown
Post your full findings as a **kanban_comment** on root card <ROOT_ID> before completing.
```

### Root cards with `--triage` avoid auto-assignment
Without `--triage`, root cards can be auto-assigned by `kanban.default_assignee` to a debate agent — that agent then tries to execute instead of synthesize. Always create root cards with `--triage`.

### Dispatcher "Spawned: 0" output is unreliable
The stdout count may not reflect actual spawning. Verify with `ps aux | grep "hermes.*task"` and `hermes kanban ls | grep <child-id>`.

### Large card body causes `create` to hang
For bodies >3000 chars, use two-step bootstrap: create with short placeholder, then add context as a `kanban_comment`.

### No task gets dispatched without debate
This is Rule #0. Always run the full ToT review before dispatching execution cards. If kanban creation fails, solve the obstacle — do not fall back to raw `delegate_task` for investigations.

## Verification

After a ToT cycle completes, verify:
1. All 4 agents posted `kanban_comment`s on the root card
2. A `### Synthesis — Locked Plan` comment exists with rationale and rejected alternatives
3. Child cards show `done` status
4. Root card shows `done` or was explicitly completed

## References

- `templates/rca-deliberation.md` — template for RCA-style debate
- `templates/feature-deliberation.md` — template for feature-design debate
- `templates/execution-prompt.md` — template for executor hand-off
- `scripts/tot-start.sh` — helper script for card creation
- `references/kanban-db-workarounds.md` — SQLite tricks for card status
- `references/profile-context-contamination.md` — detecting project context leaks
- Related: `adversarial-debate` skill for high-stakes synchronous debate
