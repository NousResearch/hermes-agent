# Hermes Agent development guide

Read the area contract below before editing that area. This entrypoint keeps shared
invariants visible; conditional contribution, organization and testing details live in
linked references. Paths in those references are relative to the repository root.

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a messaging
gateway (Telegram, Discord, Slack, ~20 platforms), a TUI, and an Electron desktop app. It
learns across sessions (memory + skills), delegates to subagents, runs scheduled jobs, and
drives a real terminal and browser. It is extended primarily through **plugins and skills**,
not by growing the core.

Two invariants shape almost every design decision and are the lens for reviewing any change:

- **Per-conversation prompt caching is sacred.** A long-lived conversation reuses a cached
  prefix every turn. Anything that mutates past context, swaps toolsets, reloads memories, or
  rebuilds the system prompt mid-conversation invalidates that cache and multiplies the user's
  cost. We do not do it; the ONE exception is context compression. Slash commands that mutate
  system-prompt state (skills, tools, memory) must be **cache-aware**: default to deferred
  invalidation (takes effect next session) with an opt-in `--now` flag (`/skills install --now`
  is the canonical pattern).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

## Area routing

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | AIAgent + mixins, turn phases, caching integrity, message-flow invariants, compression, model/aux resolution |
| `cli.py`, `hermes_cli/`, `main.py` | `hermes_cli/AGENTS.md` | CLI mixins, `_SLASH_DISPATCH`, slash registry, config system + loaders, skins, `hermes update` pipeline, profiles / multiplex |
| `gateway/` | `gateway/AGENTS.md` | Adapters, two message guards, streaming contract, background notifications, gateway vs desktop lifecycle, token locks, scoped secrets |
| `tools/`, `toolsets.py`, `model_tools.py` | `tools/AGENTS.md` | Adding tools, registry, toolsets, delegation, cross-tool references, backends |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | Plugin kinds, native compat contract, in-tree policy, current compatibility policy |
| `tui_gateway/`, `ui-tui/` | `tui_gateway/AGENTS.md` | Process model, JSON-RPC transport, key surfaces, slash flow, dev commands |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | Dashboard embeds the real TUI; what React may and may not rebuild |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | Desktop judgment guide; `serve` backend, slash palette curation, Bot Mode canonical chat |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | Frontmatter, HARDLINE authoring standards, curator |
| `cron/`, kanban (`hermes_cli/kanban*.py`, `tools/kanban_tools.py`, `plugins/kanban/`) | `cron/AGENTS.md` | Scheduler invariants, job fields, kanban board/dispatcher |
| `gateway/platforms/` new adapter | `gateway/platforms/ADDING_A_PLATFORM.md` | Step-by-step adapter guide |

Long-form background lives in `website/docs/developer-guide/` (agent-loop, prompt-assembly,
context-compression-and-caching, gateway-internals, tools-runtime, plugins/, cron-internals,
session-storage, ...). Workflow rules (PR/issue/review/salvage process) live in the
`hermes-agent-dev` skill, not here.

## Task references

- When proposing capabilities, diagnosing intent, reviewing or salvaging contributions,
  read [the contribution rubric](docs/agents/contribution-rubric.md). It preserves the
  maintainer's acceptance criteria, footprint ladder and session-capability contract.
- When locating, moving or splitting code and facade/sibling imports, read
  [code organization](docs/agents/code-organization.md).
- When adding or running tests, read [the testing contract](docs/agents/testing.md).
  Always use `scripts/run_tests.sh`; it isolates credentials, timezone, HERMES_HOME and
  per-file subprocess state. Never write tests against the real user profile. Preserve
  real-path integration coverage and native OS markers; assert behavior, not source text.

## Code Shape Rules (all languages)

- No "defense-in-depth" wrappers, `try/except: pass` around code that cannot fail, or flags
  nobody sets. Docstrings/comments keep the WHY, cut the WHAT.
- **Never infer process identity from argv substrings** (`"serve" in cmdline`) — the bug class
  behind ~10 fleet-update issues (#90778, #87594, #78089, #76129, #91964). Use the canonical
  matchers `gateway.status.looks_like_gateway_command_line` and
  `hermes_cli.update_cmd._hermes_holder_subcommand`; flag sets are DERIVED from the parser
  (`_holder_value_flags()`), never hand-written; match FULL cmdlines and truncate only for
  display. Details: `hermes_cli/AGENTS.md`.
- **Never hardcode `~/.hermes`.** `get_hermes_home()` for code paths, `display_hermes_home()`
  for user-facing text (both from `hermes_constants`). Hardcoding breaks profiles (5 bugs in
  PR #3575). Module-level constants are fine — they cache after `_apply_profile_override()`
  sets `HERMES_HOME`. Profile operations themselves are HOME-anchored
  (`_get_profiles_root()` = `Path.home()/.hermes/profiles`) so `hermes -p x profile list`
  sees all profiles — intentional, not a bug.
- **Argparse alias dispatch:** `add_parser("list", aliases=["ls"])` sets `dest` to the literal
  the user typed (`"ls"`). Dispatch must accept both (caught PTY-testing `hermes webhook ls`).
- **Don't wire in dead code without E2E validation.** Unshipped code was dead for a reason;
  E2E the real resolution chain with real imports against a temp `HERMES_HOME` first.

### TypeScript style (desktop, TUI, website, future TS packages)

Small nanostores over component state when state is shared or read by distant UI; each
feature owns its atoms (chat near chat, shared in `src/store`); rendering components use
`useStore`, non-rendering actions read `$atom.get()`; never thread state through three
components when the leaf can subscribe; persistence sits beside the atom that owns it. Route
roots stay thin (compose routes + shell, never controllers). No monolithic hooks — one narrow
job each; colocated action modules over god hooks. Pure side-effect callbacks use the terse
void form `onState={st => void setGatewayState(st)}`; async handlers make intent explicit
`onClick={() => void save()}`. Interfaces for public props and shared object shapes (not
`type X = {...}`); extend React primitives (`React.ComponentProps<'button'>`, `Omit`, `Pick`).
Table-driven beats condition ladders for ids/routes/views. `src/app` owns routes/pages,
`src/store` shared atoms, `src/lib` pure helpers.

## Dependency Pinning Policy

All dependencies carry upper bounds (litellm compromise #2796/#2810; Mini Shai-Hulud worm,
May 2026). PyPI: `>=floor,<next_major` (`"httpx>=0.28.1,<1"`); pre-1.0: `<0.(minor+2)`
(`>=0.29,<0.32`). Git URLs: 40-char commit SHA. GitHub Actions: SHA + `# vN` comment. CI-only
pip: `==exact`. A bare `>=X.Y.Z` is rejected by CI and reviewers. Run `uv lock` after
changing `pyproject.toml`. Reference: #2810 (bounds), #9801 (SHA pinning + audit CI).

## Commits, Merges, PRs

- Before an authorized merge, inspect the exact branch, current base and existing work.
  Fetch the base and integrate it with a recoverable merge, or an authorized rebase in
  an isolated checkout. Preserve unrelated changes and inspect the PR diff against the
  updated base for unintended reversions. After merging, verify the merge result.
- Salvage by cherry-pick so contributor authorship survives (see rubric).
- Tests per fix: use the smallest sufficient behavioral coverage of the affected
  invariants, proven red on the base where practical. Cover independent failure paths
  when needed, without an arbitrary test-count cap or change-detectors.
  Reject/rewrite in salvaged diffs:
  appendages to facades, new god helpers, compat aliases, wrappers.
