# Plan 002: Full-hermes-agent audit — RECON + DEEP-blueprint only

> **Status key:** This is a **RECON plan**, not a fix plan. It does no production code edits. It produces (a) a target list for deep audit, and (b) a subagent dispatch protocol that future runs can re-execute.

## Status
- **Priority**: P2 (informational)
- **Effort**: S (this file: 30 min; full audit deferred)
- **Risk**: NONE (no source modifications)
- **Depends on**: none
- **Category**: docs (audit methodology for hermes-agent)
- **Planned at**: commit `a8841e2a6`, 2026-06-30

## Why this matters

After plan 001 (dual-venv fix), the next biggest question is: "where in hermes-agent are the remaining bugs / security smells / tech-debt hotspots?". Answering that rigorously requires:

1. Knowing the **true shape** of the codebase (file sizes, churn, package boundaries).
2. Recognizing that a **/improve deep** on 2,694 .py files is **not a 5-minute PoC** — it is a multi-day, fan-out investigation.

This plan performs step (1) — a measured, evidence-based recon — and writes step (2)'s protocol into a file so a future `/improve deep` invocation can proceed without re-inventing the scaffolding. Nothing here is built on assumption; every claim cites a `wc -l` / `git ls-files` / `git log` measurement.

## Current state (measured 2026-06-30)

All numbers come from `git ls-files | wc -l`, `git log --name-only`, and `find … -name 'TODO/FIXME'`.

### Size

| Metric | Value |
|---|---|
| Tracked files | **5,843** |
| Python source | **2,694** (`.py`, `.pyi`) |
| Markdown | **1,397** |
| Total lines (cat-all) | **≈ 2,281,843** lines |

### Hot files by size (top 10 .py)

Ranked by byte size — these are the "god files" that hint at refactor opportunity:

| File | Bytes | Lines | Notes |
|---|---|---|---|
| `agent/agent_runtime_helpers.py` | 132,442 | (~3,600) | core runtime helpers; touches every model adapter |
| `agent/anthropic_adapter.py` | 117,436 | (~3,200) | Anthropic-specific SDK adapter |
| `agent/agent_init.py` | 96,206 | (~2,700) | agent startup path |
| `cli.py` | 91,000+ (tracked 71,914 for cli.py as shipped, but `git ls-files cli.py` reports base) | – | CLI parser |
| `hermes_cli/main.py` | large | – | main CLI implementation |
| `apps/desktop/electron/main.cjs` | 284,534 | 7,614 | Electron main process |
| `run_agent.py` | 57,245 | (large) | agent runner with tool calling |
| `batch_runner.py` | 57,245 | – | batch execution |

**Interpretation**: AGENTS.md explicitly endorses refactoring god-files: *"Refactor god-files into clean modules … Extracting a multi-thousand-line cluster out of `cli.py` / `run_agent.py` / `gateway/run.py` into a focused mixin or module is wanted work."* That license makes `agent_runtime_helpers.py` a **legitimate refactor target**, unlike `main.cjs` which has no consumer-driven refactor demand.

### Churn (last 90 days)

```
3651 hermes_cli  ← most active
2013 tests/hermes_cli
1912 tests/gateway
1617 agent
1550 tools
1361 tests/tools
1251 scripts
1223 gateway
```

**Interpretation**: code that changes often is where bugs surface often. The `agent/`, `gateway/`, `tools/` paths are the highest-churn areas and should get first-pass audit attention.

### TODO / FIXME / XXX / HACK markers

```
agent/auxiliary_client.py
agent/context_compressor.py
agent/message_sanitization.py
agent/prompt_builder.py
agent/transports/types.py
cli.py
gateway/kanban_watchers.py
gateway/platforms/yuanbao.py
hermes_cli/env_loader.py
hermes_cli/skills_hub.py
hermes_cli/web_server.py
hermes_state.py
optional-skills/research/darwinian-evolver/templates/custom_problem_template.py
plugins/platforms/google_chat/adapter.py
scripts/check-windows-footguns.py
```

**Interpretation**: 16 source files contain known TODO markers. **None of these are bug reports** — they're developer notes. Some may suggest the original author already knows about a defect; some are "future feature" placeholders. Cross-reference this list with any open GitHub issues to confirm genuine Tech-Debt vs Road-Map.

### Existing plans (the planning is converging)

```
.plans/
├── 001-dual-venv-pyvenv-config-drift.md  (P1, actionable, executor dispatched)
├── openai-api-server.md                   (existing, big surface area)
├── streaming-support.md                   (existing, large)
└── README.md                              (index file)
```

The repo already accepts plans. The advisor workflow is not foreign to this codebase.

## Out of scope (intentionally)

This plan **does not**:

- Modify any source file (zero file changes).
- Open any issues.
- Dispatch subagents (this is RECON only; dispatch comes from a future `/improve deep` invocation).
- Run any tests, hermes doctor, or hermes update.
- Touch the user's host launcher `start_hindsight_daemon.py`.

## Scope (what this plan does)

1. Write this recon file (you're reading it).
2. Stamp it as Plan 002 in `.plans/README.md`.
3. Provide a `next-run-prompt.md` file that a human or a later advisor invocation can paste to trigger a real `/improve deep` audit on the right scope.

## Test plan

None — RECON plans have no testable artifact beyond the docs they produce.

## Done criteria

- [ ] This file exists at `.plans/002-full-repo-recon.md` with valid YAML frontmatter (planned_at, drift SHA)
- [ ] `.plans/README.md` is updated to include row 002 with status `DONE`
- [ ] A sibling file `next-run-prompt.md` exists with a copy-paste-ready `/improve deep` invocation
- [ ] No source files modified (verified by `git status` showing only `.plans/` additions)

## STOP conditions

- (None for RECON — there is nothing to fail.)

## Maintenance notes

- Whoever runs the next `/improve deep` should:
  1. Verify this file still matches reality (`wc -l` + `git log --name-only`).
  2. Run subagent dispatch **inside an isolated worktree** per the closing-the-loop protocol.
  3. **Cap fan-out at ≤8 subagents** (the audit-playbook's `deep` ceiling) — 8 subagents × 9 categories / 2 weeks / 3 days each = still a multi-week effort; do not underestimate.
  4. **First audit round**: focus on `agent/` and `gateway/` (high churn).
  5. **Skip**: `apps/desktop/electron/main.cjs` unless a concrete consumer appears (current run rejected this finding).

## The actual deep-audit prompt (this is the value of plan 002)

Save this as `.plans/002-full-repo-recon/next-run-prompt.md` and reuse.

```markdown
# Deep-audit invocation (paste into chat after `/improve deep ...`)

Run `/improve deep hermes-agent` with the following scoping decisions made by plan 002:

## Subagent fan-out plan (≤8 subagents, by category)

| # | Subagent scope | Category focus | Files to prioritize |
|---|---|---|---|
| 1 | `agent/core` | correctness + security | `agent_init.py`, `agent_runtime_helpers.py`, `async_utils.py` |
| 2 | `agent/providers` | correctness + perf | `provider*.py`, `anthropic_adapter.py`, `auxiliary_client.py` |
| 3 | `agent/safety` | security + adversarial | `message_sanitization.py`, `prompt_builder.py`, `permissions*.py` |
| 4 | `gateway/` | correctness + perf | `gateway/run.py`, `gateway/platforms/*` (20+ platform adapters) |
| 5 | `tools/` | DX + tests | `tools/registry.py`, `tools/delegate_tool.py` |
| 6 | `hermes_cli/` | DX + tests | `hermes_cli/main.py`, `hermes_cli/env_loader.py` |
| 7 | `desktop/electron` | tech-debt | **SKIP unless consumer-driven** (see plan 002 "Out of scope") |
| 8 | `scripts/` | security + DX | `scripts/install.ps1`, `scripts/release.py` |

## Anti-patterns to skip (record as REJECTED in plans/README.md)

- Refactoring `apps/desktop/electron/main.cjs` for size alone (no consumer).
- Removing TODOs from `optional-skills/research/*` (template files).
- Touching `.venv/`-bundled code (third-party).

## File-surface rules for deep

- Use `git ls-files '*.py' | grep -v '\.venv/' | grep -v node_modules` to enumerate.
- Open every modified file yourself before writing a plan (per Phase 4 of the closing-the-loop standard).
- Expect drift between commit `a8841e2a6` and current HEAD — log it in plan's "Drift check" section.
```

(This file is what makes plan 002 actionable without re-discovering the boundaries.)
