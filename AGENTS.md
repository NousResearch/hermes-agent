# Hermes Agent - Development Guide

For AI coding assistants and developers working on hermes-agent. This root file is a hub: what
applies everywhere, then task and area routing. Each area's `AGENTS.md` loads automatically when you
work in that directory; read it before editing there. `python scripts/check` caps this file at 12k
chars and every root-to-area chain at 30k, so it loads whole on 128k+ models: long form goes in the
guide.

## Design invariants

Two invariants shape almost every design decision and are the lens for reviewing any change:

- **Per-conversation prompt caching is sacred.** Mutating past context, swapping toolsets,
  reloading memories or rebuilding the system prompt mid-conversation breaks the cached prefix and
  multiplies the user's cost; the ONE exception is context compression. Slash commands that change
  system-prompt state defer to the next session, with an opt-in `--now` (`/skills install --now`).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

## Contribution rubric

The project's intent layer, for contributors and for the triage sweeper (which may only close on
`implemented_on_main`, `cannot_reproduce` or `incoherent`; taste-based closes are a maintainer's
call, and when in doubt a PR stays open). Before proposing, reviewing or triaging a contribution,
read `website/docs/developer-guide/contributing.md` § Contribution rubric — the long form with
examples: wanted work, rejected-even-when-well-built patterns, the verify-the-premise bar, and the
footprint ladder for new capability.

**Security:** `SECURITY.md` is the scope authority. A §3.1 finding goes private (GitHub Security
Advisories or security@nousresearch.com), never into a public issue, PR, commit or comment; §3.2
hardening is ordinary public work. Name the §2 boundary crossed, with a repro on `main`.

## Before you write or review code

Read `CODING_STANDARDS.md` — the code conventions extracted from this hub. Jump to the section that
matches the task:

- activating this checkout, running `scripts/check`, the pre-push hook → [Environment and gates](CODING_STANDARDS.md#environment-and-gates)
- new code and refactors: defensive style, size/complexity ratchets, waivers → [Code quality and the ratchet](CODING_STANDARDS.md#code-quality-and-the-ratchet)
- god files: facades, `_topic` siblings, late imports, patch seams → [Facades and siblings](CODING_STANDARDS.md#facades-and-siblings)
- moving an internal symbol: no re-export shims, fixing its docs → [Moving internal symbols](CODING_STANDARDS.md#moving-internal-symbols)
- `~/.hermes` paths, profiles, process identity, host facts → [Paths, profiles and machine facts](CODING_STANDARDS.md#paths-profiles-and-machine-facts)
- `pyproject.toml`, `uv.lock`, pinned dependencies, PM environments → [Dependencies and PM environments](CODING_STANDARDS.md#dependencies-and-pm-environments)
- TypeScript (desktop, TUI, website) → [TypeScript](CODING_STANDARDS.md#typescript)
- running tests, test shapes → [Tests](CODING_STANDARDS.md#tests)
- rebase and merge readiness, red-on-base invariant tests → [Commits and pull requests](CODING_STANDARDS.md#commits-and-pull-requests)

## Project Structure

Counts shift constantly; the filesystem is canonical. The routing table below routes each area to its
governing docs.

## Routing Table — working in X → read X/AGENTS.md

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | turn phases, caching and message-flow invariants, compression, model/aux resolution |
| `cli.py`, `hermes_cli/` | `hermes_cli/AGENTS.md` | CLI mixins, slash registry, config system, skins, `hermes update`, profiles / multiplex |
| `gateway/` | `gateway/AGENTS.md` | adapters, message guards, streaming, notifications, token locks, § Profile scope |
| `gateway/platforms/` new adapter | `gateway/platforms/ADDING_A_PLATFORM.md` | step-by-step adapter guide |
| `tools/`, `toolsets.py`, `model_tools.py` | `tools/AGENTS.md` | adding tools, registry, toolsets, delegation, session-scoped surface tools |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | plugin kinds, native compat contract, in-tree policy |
| `plugin-catalog/` | `plugin-catalog/README.md` | catalog admission rules (mirrored in the developer guide; keep identical) |
| `tui_gateway/`, `ui-tui/` | `tui_gateway/AGENTS.md` | process model, JSON-RPC transport, slash flow |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | dashboard embeds the real TUI |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | `serve` backend, slash palette, Bot Mode |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | frontmatter, authoring standards, curator |
| `cron/`, kanban | `cron/AGENTS.md` | scheduler invariants, job fields, kanban dispatcher |
| `tests/` | `tests/AGENTS.md` | runner, placement, OS markers, `wine2e`, banned test shapes |
| `pm/`, `pyproject.toml` | `pm/AGENTS.md` | pinning policy, PM-owned environments, plugin quarantine |
| `hermes_platform/` | `hermes_platform/AGENTS.md` | host facts, resolvers |

Long-form background: `website/docs/developer-guide/`. Workflow rules (PR/issue/review/salvage
process) live in the `hermes-agent-dev` skill, not here.
