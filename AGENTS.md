# Hermes Agent

## What Hermes Is

Hermes is a personal AI agent that runs the same core across a CLI, messaging
gateway, TUI, and desktop app. It learns across sessions, runs scheduled work,
and uses a real terminal and browser. It is extended primarily through
**plugins and skills**, not by growing the core.

This repository is a fork of Hermes Agent. Keep upstream changes easy to merge
and implement product behavior directly in the existing code. Preserve native
Hermes semantics and document intentional differences in `docs/downstream/`.

**Never give up on the right solution.**

These properties shape almost every design decision and are the lens for
reviewing any change:

- **Deletion is the first primitive.** Before building, delete: identify what
  the change makes obsolete and remove it at the root, then add. Never layer
  a shim, fallback, legacy path, or compatibility wrapper over code that
  should not exist — that is how dead plumbing accumulates. A change that
  leaves two mechanisms for one job is not done until the old one is gone.
  Any deliberately retained compatibility layer must be explicitly justified.
- **Per-conversation prompt caching is sacred.** Runtime mutations must preserve
  warm provider caches. See "Prompt Caching Must Not Break" below for the
  binding rules.
- **The core is a narrow waist; capability lives at the edges.** Every model
  tool we add is sent on every API call, so the bar for a new *core* tool is
  high. Most new capability should arrive as a CLI command + skill, a
  service-gated tool, or a plugin — not as core surface.
- **Reuse native Hermes first.** Before implementing a feature, find and use
  the existing Hermes code path. Extend that code directly; do not add a shim
  or parallel implementation.
- **Preserve native harness semantics.** Product code must not
  rewrite prompts, context injection, tool calls, tool results, errors, or
  message flow. Any intentional deviation must be explicitly justified.
- **Model-surface changes require explicit qualification.** For any surface in
  `docs/workflows/prompt-work.md`, read and follow that workflow, update the
  affected documentation, and document the exact before/after behavior. Clear
  bug fixes that restore unambiguous intended behavior may proceed
  without approval when downstream intent is preserved and validation proves the
  repair. Propose first and obtain approval only when the change creates a product
  choice, leaves downstream intent ambiguous, or carries a material behavior
  tradeoff.

## How to Work

In everything written for the user — chat replies, PRs, commit messages,
issue updates, docs — use plain, simple language and keep it brief: no
jargon, no filler, no restating what they already know.

### Scope the task

Handle implementation requests end-to-end: plan, implement, verify, review,
commit, push, and open or update the pull request when required by Landing
Changes.

Don't hide confusion. State your assumptions explicitly; if multiple
interpretations exist, name them, pick the most reasonable, and proceed —
flag the choice in the PR. If a simpler approach exists, say so. Push back
when warranted. Once underway, stop with a concrete blocker when progress
requires missing access, external setup, an unsafe production action, or an
explicit user decision.

### Design the change

Focus on making complex things as simple as possible.

Use repository documentation, local context, established patterns, and
documented workflows before inventing new approaches. When a change builds on
external APIs, libraries, or protocols, fetch current primary docs and real
implementation examples. Never design from memory alone.

Write the simplest code that solves the problem - nothing speculative: no
abstractions for single-use code, no error handling for impossible
scenarios. If you write 200 lines and it could be 50, rewrite it. Ask
yourself: "Would a senior engineer say this is overcomplicated?" If yes,
simplify.

### Make the change

Match existing style, even if you'd do it differently. Remove
imports/variables/functions that your changes made unused, and delete what
your change makes obsolete at the root (deletion is the first primitive).

Document durable behavior, contracts, workflows, and architecture decisions -
not routine small changes or temporary process notes.

### Verify and review

Define success criteria and loop until verified. Transform tasks into
verifiable goals: "fix the bug" means a test that reproduces it and then
passes; "refactor X" means tests pass before and after. Strong success
criteria let you loop independently; weak criteria ("make it work") require
constant clarification.

Keep `main` deployable and the codebase secure. Run the applicable checks
and review the complete change set. Triage every finding and fix valid ones
at the root cause; after fixes, rerun affected checks and review.
Run `./.codex/scripts/codex-review.sh` with the complete change staged; triage
findings, fix valid ones, rerun affected checks and repeat review before committing
and pushing. See [Codex Review](docs/workflows/codex-review.md). Review-only agents
follow [Reviewing](docs/workflows/reviewing.md) and must not launch another reviewer.

## Landing Changes

Never create, switch, or request branches for routine work in this repository.
Always stay on the branch already checked out in the current worktree. Assume
existing local changes are intentional and may come from the user or parallel
agents; work with them instead of treating a dirty worktree as a blocker.

Follow the user's instructions for commits, pushes, and pull requests. Before
creating or updating a pull request, verify the branch has no merge conflicts
with `main`. Use `.github/PULL_REQUEST_TEMPLATE.md` for PRs. See
[Landing Changes](docs/workflows/landing-changes.md) for merge and salvage rules.

## Required Reading

Read the nearest area guide before editing there. Shared rules in the linked
documents apply throughout the repository.

| Area | Guide |
| --- | --- |
| Agent loop, prompts, providers | `agent/AGENTS.md` |
| Tools and environments | `tools/AGENTS.md` |
| Messaging gateway | `gateway/AGENTS.md` |
| CLI, configuration, updater | `hermes_cli/AGENTS.md` |
| Plugin contracts | `plugins/AGENTS.md` |
| Skills and curator | `skills/AGENTS.md` |
| Scheduling and Kanban | `cron/AGENTS.md` |
| TUI and JSON-RPC | `tui_gateway/AGENTS.md` |
| Desktop | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` |
| Dashboard | `web/AGENTS.md` |
| New gateway platform | `gateway/platforms/ADDING_A_PLATFORM.md` |
| Profiles, multiplexing, secret scope | `gateway/AGENTS.md`, `website/docs/user-guide/multi-profile-gateways.md` |

- Before planning or making code changes, read [ARCHITECTURE.md](ARCHITECTURE.md).
- Scan `docs/` for relevant durable context before editing.
- For the planned employee behavior, read [the product specification](docs/specs/employee.md).
  It records deliberate future divergences; do not confuse planned behavior with
  the current runtime or remove unrelated upstream features during documentation work.
- Before implementation, read [Development](docs/development.md) and
  [Contribution rules](docs/contributing.md), including the footprint ladder.
- Before changing TypeScript, read [TypeScript style](docs/typescript.md).
- Before changing dependencies, read [Dependencies](docs/dependencies.md).
- Before changing or running tests, read [Testing](docs/testing.md).
- Before intentionally diverging from upstream or integrating upstream changes,
  read [Downstream Intent](docs/downstream/README.md).
- Before changing prompts, tools, or model-visible messages, read
  [Model-Facing Changes](docs/workflows/prompt-work.md).

Keep root guidance compact. Durable detail belongs in `ARCHITECTURE.md` or
focused `docs/`; subtree-only rules belong in the nearest `AGENTS.md`.
Long-form subsystem documentation lives in [the developer guide](website/docs/developer-guide/).

## Prompt Caching Must Not Break

Do not alter past context, change toolsets, reload memories, or rebuild system
prompts mid-conversation. The system prompt stays byte-stable for the life of
a conversation; context compression is the exception.

Slash commands that mutate system-prompt state (skills, tools, memory, etc.)
must be **cache-aware**: default to deferred invalidation (change takes
effect next session), with an opt-in `--now` flag for immediate
invalidation. See `/skills install --now` for the canonical pattern.

## Testing

**ALWAYS use `scripts/run_tests.sh`** — do not call `pytest` directly. It
isolates test files and credentials and enforces CI parity. Tests must not
write to real user state. Assert behavior, not source text or frozen values.

See [Testing](docs/testing.md) for environment setup, profile isolation, host
markers, placement, and validation requirements.
