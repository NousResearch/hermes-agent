# Hermes Agent - Development Guide

This root file is a hub: what applies everywhere, then a routing table. Each area's `AGENTS.md`
loads automatically when you work in that directory; read it before editing there.
`python scripts/check` caps this file at 12k chars and every root-to-area chain at 30k: long form
goes in `website/docs/developer-guide/`.

**Never give up on the right solution.**

## What Hermes Is

Hermes is a personal AI agent that runs one agent core across a CLI, a TUI, an Electron desktop
app, a web dashboard, ACP editors and a messaging gateway (25+ platform plugins). It is extended
through **plugins and skills**, not by growing the core.

Two invariants shape almost every design decision and are the lens for reviewing any change:

- **Per-conversation prompt caching is sacred.** Mutating past context, swapping toolsets,
  reloading memories or rebuilding the system prompt mid-conversation breaks the cached prefix and
  multiplies the user's cost; the ONE exception is context compression. Slash commands that change
  system-prompt state defer to the next session, with an opt-in `--now` (`/skills install --now`).
- **The core is a narrow waist; capability lives at the edges.** Every model tool is sent on
  every API call, so the bar for a new *core* tool is high. New capability should arrive as a
  CLI command + skill, a service-gated tool, or a plugin — not as core surface.

A change to the stock system prompt or `tools[]` fails `tests/ci/test_prompt_surface.py` until
`scripts/ci/prompt_surface.py render` is committed with it, and that snapshot
(`tests/fixtures/prompt_surface/`) needs a hermes-agent-core approval to merge.

## Contribution rubric

The project's intent layer, for contributors and for the triage sweeper (which may only close on
`implemented_on_main`, `cannot_reproduce` or `incoherent`; taste-based closes are a maintainer's
call, and when in doubt a PR stays open). Long form with examples:
`website/docs/developer-guide/contributing.md` § Contribution rubric.

**Wanted:** real bug fixes (repro on `main`, the exact line, the whole class incl. sibling paths);
reach at the edges (adapters, providers, models, UI features) wired into the existing setup UX;
god-file → module refactors; extending before duplicating (3+ PRs in one category → an ABC +
orchestrator); behaviour-contract tests; E2E with real imports against a temp `HERMES_HOME` for
resolution, config, security and I/O changes; salvage by cherry-pick so authorship survives.

**Rejected even when well-built:** hooks with no concrete consumer; new `HERMES_*` env vars for
non-secret config (`.env` is secrets only, behaviour goes in `config.yaml`); a new core tool when
terminal + file or a skill already does the job; `offset`/`limit` pagination on instructional tools;
"fixes" that destroy the feature they secure; outbound telemetry without an opt-in gate;
change-detector tests; plugins that touch core files; third-party products in the core tree (ship a
standalone plugin repo).

**Before you call it a bug,** verify the claim AND the intent (`git log -p -S "<symbol>"`): the
isolation is often the design (profiles are islands on purpose), an absence can be load-bearing,
and a fix that cannot point to the line where the bug manifests has an unverified premise.

**Security:** `SECURITY.md` is the scope authority. A §3.1 finding goes private (GitHub Security
Advisories or security@nousresearch.com), never into a public issue, PR, commit or comment; §3.2
hardening is ordinary public work. Name the §2 boundary crossed, with a repro on `main`.

**Footprint ladder** (take the highest rung that solves it): extend existing code → CLI command +
skill → service-gated tool (`check_fn` answers reachability/opt-in, never per-session surface) →
plugin → MCP server in the catalog → new core tool (fundamental, broadly useful, unreachable
otherwise).

## Development Environment

`source ./activate` (`activate.fish`, `activate.ps1`) provisions and activates the PM environment; pick an isolated `HERMES_HOME`/`HERMES_RUNTIME_DIR` first
(`website/docs/reference/package-management.md#developer-workflow`). Tests need the separate test
environment in `CONTRIBUTING.md`. **`python scripts/check`** (`just check`) runs every blocking
lint check CI runs, with CI's pinned tools; `--install-hook pre-push` runs it on every push (re-run
after pulling to refresh the hook). `just` lists every developer command (lefthook calls the same
recipes); a new command is a `scripts/` file plus a recipe. The `HX0xx`/`PS-P0x` rules and their
fixes are in `scripts/code_health/config.py`; `# noqa` does not waive a finding,
`# health: allow <RULE> -- <why>` does.

## Project Structure

The filesystem is canonical. Load-bearing entry points:

```
hermes-agent/
├── run_agent.py          # AIAgent facade; the turn loop lives in agent/turn_*.py
├── cli.py                # HermesCLI (REPL, slash dispatch) + hermes_cli/cli_*_mixin.py
├── hermes_agent/         # the Python package
│   ├── model_tools.py        # Tool orchestration, handle_function_call(); discovery in tools/registry.py
│   ├── toolsets.py           # TOOLSETS dict, _HERMES_CORE_TOOLS
│   └── hermes_state.py       # SessionDB facade; hermes_state_*.py siblings
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── scripts/              # check, run_tests.sh, code_health/, ci/
├── evals/                # live-model A/B and probe harnesses
└── website/              # Docusaurus docs (developer-guide/ holds the long-form area docs)
```

**User state:** `~/.hermes/config.yaml` (settings), `.env` (secrets only), `logs/` (`hermes logs`);
all profile-aware via `get_hermes_home()`.

### Facade + siblings layout

Every former god file is a **facade** (public entry points + the names other packages import) plus
**siblings** `<stem>_<topic>.py`, each owning one topic (the tree above, `hermes_agent/gateway/run.py`,
`hermes_agent/tools/mcp_tool.py`, `hermes_cli/kanban.py`, `hermes_cli/web_server.py`).

- **Find code by topic:** `grep -rn "def name" <dir>/<stem>_*.py`, not by reading the facade.
- **Siblings late-import the facade** inside functions; never a module-level cycle.
- **Patch where production reads:** a sibling doing `from <facade> import name` inside the function
  makes the facade the seam; a patch on the defining module passes silently.
- **Size and complexity are ratcheted per unit:** new functions CC ≤ 20, ≤ 300 lines, nesting ≤ 6;
  files ≤ 2,000 lines; units already over may only go down (move a function into a sibling to
  offset growth, or put new tests in a new test file). Behaviour goes in a sibling, never a facade;
  name ladders become a dict → handler.
- **No re-export shims for internal moves;** internal paths are not API. Moving a symbol means
  fixing its docs in the same PR (grep `website/docs`, `skills/`, every `AGENTS.md`).

## Rules that apply everywhere

- **No defensive clutter:** no wrappers or `try/except: pass` around code that cannot fail, no flags
  nobody sets, no dead code wired in without E2E proof. Comments keep the WHY, cut the WHAT.
- **Never infer process identity from argv substrings;** use
  `hermes_agent.gateway.status.looks_like_gateway_command_line` / `hermes_cli.update_cmd._hermes_holder_subcommand`
  (HX003).
- **Never hardcode `~/.hermes`:** `get_hermes_home()` for paths, `display_hermes_home()` for text
  (HX001). `_get_profiles_root()` is HOME-anchored on purpose.
- **One process serves many profiles.** Code that runs outside a turn (boot probes, eviction,
  tickers, deferred callbacks, RPC methods, thread hops, child spawns) binds the owning profile scope
  explicitly; `os.environ`, module globals and import-time values hold the launch profile's, so an
  unbound read is a silent default-profile leak. Binding points and seams: `hermes_agent/gateway/AGENTS.md`
  § Profile scope (HX002/HX004/HX005/HX012, PS-P05/P06).
- **Machine facts and executable lookup go through `hermes_platform`.**
- **Dependencies carry upper bounds** (`>=floor,<next_major`; git URLs and Actions pinned to a SHA);
  after editing `pyproject.toml` run `hermes pm lock` and commit `uv.lock`; never mutate a Hermes
  environment with raw pip/uv.
- **TypeScript (desktop, TUI):** feature-owned nanostores over threaded state, thin route roots,
  narrow hooks and colocated action modules, `interface` for props and shared shapes, table-driven
  dispatch over condition ladders; desktop keeps routes in `src/app`, atoms in `src/store`, pure
  helpers in `src/lib`.
- **Don't add core/plugin crossings:** core never imports a specific bundled plugin, and plugins
  use only the published plugin API (`hermes_cli/data/plugin_api.json`, `ctx`). Existing crossings
  are baselined; `scripts/boundary_lint.py` fails the test suite (not `scripts/check`) on new ones.
- **A new config key or slash command ships with its docs** (`tests/ci/test_docs_drift.py`).
- **Commits and PRs:** rebase onto `main` before merging (a squash from a stale branch silently
  reverts newer fixes); 1–2 invariant tests per fix, proven red on the base.
- **Tests:** always `scripts/run_tests.sh`, never bare `pytest`; behaviour contracts, never
  change-detectors or source-reading tests; host-specific behaviour is tested on that host with
  `@pytest.mark.platforms(...)`, never by faking `sys.platform`; tests never write to `~/.hermes/`.

## Routing Table

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | turn phases, message-flow invariants, compression, model/aux resolution, i18n |
| `cli.py`, `hermes_cli/` | `hermes_cli/AGENTS.md` | CLI mixins, slash registry, config system, skins, `hermes update`, profiles / multiplex |
| `hermes_agent/gateway/` | `hermes_agent/gateway/AGENTS.md` | adapters, message guards, streaming, notifications, token locks, § Profile scope |
| `plugins/platforms/` new adapter | `hermes_agent/gateway/platforms/ADDING_A_PLATFORM.md` | step-by-step adapter guide |
| `hermes_agent/tools/`, `hermes_agent/toolsets.py`, `hermes_agent/model_tools.py` | `hermes_agent/tools/AGENTS.md` | adding tools, registry, toolsets, delegation, session-scoped surface tools |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | plugin kinds, native compat contract, in-tree policy, boundary lint |
| `plugin-catalog/` | `plugin-catalog/README.md` | catalog admission rules (mirrored in the developer guide; keep identical) |
| `hermes_agent/tui_gateway/`, `ui-tui/` | `hermes_agent/tui_gateway/AGENTS.md` | process model, JSON-RPC transport, slash flow |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | dashboard embeds the real TUI |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | state authority, gateway backend contract, slash palette, Bot Mode |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | frontmatter, authoring standards, curator |
| `hermes_agent/cron/`, kanban | `hermes_agent/cron/AGENTS.md` | scheduler invariants, job fields, kanban dispatcher |
| `tests/` | `tests/AGENTS.md` | runner, placement, OS markers, `wine2e`, banned test shapes |
| `pm/`, `pyproject.toml` | `pm/AGENTS.md` | pinning policy, PM-owned environments, release-age rule |
| `hermes_platform/` | `hermes_platform/AGENTS.md` | host facts, resolvers |
| `hermes_agent/acp_adapter/` | `website/docs/developer-guide/acp-internals.md` | ACP server for editors |
| `providers/` | `providers/README.md` | `ProviderProfile` registry and ABC |

PR workflow: `CONTRIBUTING.md` (maintainers: the `hermes-agent-dev` skill, not shipped here).
