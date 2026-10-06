# Coding standards

Read this file after the root [`AGENTS.md`](AGENTS.md) for any source, test, dependency, refactor, or review change. The closest area guide adds local contracts. This file owns general implementation policy; it does not replace an area guide, `SECURITY.md`, a script, configuration, or the filesystem.

## Decide scope before coding

### Footprint ladder

Choose the first rung that solves the problem correctly, in this order:

1. Extend existing code.
2. Add a CLI command plus a skill when shell commands and existing tools express the capability.
3. Add a service-gated tool when structured input and output are required. Its `check_fn` answers process-wide reachability or opt-in, never a per-session surface. Session-varying capability belongs in a named toolset resolved from the session.
4. Add a plugin for third-party, niche, or user-specific capability.
5. Add an MCP server in the catalog when the capability is a tool but is not fundamental to the core.
6. Add a core tool only when it is fundamental, broadly useful, and unreachable through terminal plus file or MCP.

The core tool schema is sent on every model call. Extend an existing seam before adding a module, manager, hook, or core tool. When three or more changes integrate the same category, design an ABC and orchestrator, then make providers use that surface.

### Premise check

Before calling something a bug or restricting behavior, reproduce it on `main`, identify the exact line where it manifests, and inspect `git log -p -S "<symbol>"`. Treat isolation, an absence, or a rejected approach as possibly deliberate. Preserve the feature while fixing the boundary. A triage sweeper may close only `implemented_on_main`, `cannot_reproduce`, or `incoherent`; a taste-based close is a maintainer decision, and an uncertain PR stays open.

### Work worth accepting

Prefer real bug fixes with the whole class and sibling paths covered; new adapters, providers, models, and UI features integrated with existing setup UX; god-file to module refactors; behavior-contract tests; E2E tests with real imports against a temporary `HERMES_HOME` for resolution, configuration, security, and I/O changes; and salvage by cherry-pick so external authorship survives.

### Accepted and rejected shapes

- Keep comments for the WHY, trade-offs, or API quirks. The code supplies the WHAT.
- Do not add wrappers or `try/except: pass` around code that cannot fail. Do not add flags no caller sets or wire dead code without E2E proof.
- Reject hooks and extension points with no concrete consumer.
- Do not add new `HERMES_*` environment variables for non-secret configuration. Put credentials in `.env`; put behavioral settings in `config.yaml`.
- Do not add a core tool when terminal plus file or a skill already solves the task. Fix a remote file-visibility mount instead of adding a tool.
- Do not add `offset` or `limit` pagination to instructional tools. The model must read skills, prompts, and playbooks in full.
- Read the original intent before restricting behavior. A fix that destroys the feature it secures is not a fix.
- Gate outbound telemetry, attribution, and third-party identifiers behind a user-facing opt-in with a config gate, setup prompt, and tool toggle.
- Reject change-detector tests, prompt-cache-breaking changes, dead code wired without E2E proof, and plugins that modify core files. Widen a generic plugin surface when a plugin needs a missing capability.
- Ship observability backends, vendor SaaS connectors, analytics dashboards, paid-service integrations, and other third-party products as standalone plugins, not under the core `plugins/` tree.

## Keep seams explicit

### Facades and siblings

A former god file is a facade containing public entry points and names imported by other packages, plus topical siblings that own behavior. Put new behavior in a sibling, never in a facade. Find code by topic with `grep -rn "def name" <dir>/<stem>_*.py`, not by reading the facade as an inventory. Siblings may import the facade inside functions to preserve the seam, but they must not create a module-level cycle.

Patch the binding production reads. If a sibling imports a name from the facade inside a function, the facade is the patch seam; patching the defining module can pass without changing production behavior.

Keep each new function at CC <= 20, 300 lines, and nesting <= 6. Keep files at or below 2,000 lines. A unit already over a limit may only shrink. Move behavior into a sibling to offset growth, or put new tests in a new test file. Replace name ladders with a dictionary and handler table.

Internal paths are not API. Do not add re-export shims for internal moves. Fix references in `website/docs`, `skills`, and every `AGENTS.md` in the same change.

### One source of truth

Use the filesystem, package manifests, configuration, scripts, and `--help` output for current layout and command facts. Do not copy file counts, toolset lists, or directory inventories into instruction files. Keep a rule in one authoritative document and link to it from callers. Cache only a convention, rationale, or failure mode that the environment cannot reveal.

## Development and dependencies

### Hermes environments

Choose an isolated `HERMES_HOME` and `HERMES_RUNTIME_DIR` before setup so development cannot migrate production state. From the repository root, activate the PM environment with `source ./activate`; use `activate.fish` for fish and `activate.ps1` for PowerShell. Tests require the separate test environment documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

Run `python scripts/check` for the blocking lint checks CI runs. `python scripts/check --install-hook pre-push` installs the same check on pushes; run it again after pulling to refresh the hook. `# noqa` does not waive a ratchet finding. Use `# health: allow <RULE> -- <why>` for a justified health exception.

### Dependencies and host facts

Every dependency has an upper bound, normally `>=floor,<next_major`; git URLs and GitHub Actions use a full commit SHA. After editing `pyproject.toml`, run `hermes pm lock` and commit `pyproject.toml` with `uv.lock`. Never mutate a Hermes environment with raw `pip` or `uv`; use PM-owned setup and repair flows. The full policy is [`pm/AGENTS.md`](pm/AGENTS.md).

Machine facts and executable lookup go through [`hermes_platform`](hermes_platform/AGENTS.md). Do not add a separate `shutil.which`, known-path table, or environment-variable hardware probe outside that owner.

## Tests and verification

Always run `scripts/run_tests.sh`, never bare `pytest`. It supplies CI-like credential clearing, UTC and locale settings, per-test `HERMES_HOME` isolation, and per-file subprocess isolation. Prepare the separate test interpreter through the PM workflow and follow [`tests/AGENTS.md`](tests/AGENTS.md) for placement and options.

Test behavior contracts and relationships, not current enumerations, source text, or snapshots. Do not add change-detector tests. Do not read `.py`, `.ts`, `.tsx`, `.js`, or configuration source text in a test. Extract behavior into a callable unit or exercise the real path. For resolution chains, profile routing, configuration, security boundaries, remote backends, and file or network I/O, use real imports and a temporary home. When profile scope changes, exercise at least two homes in both directions.

Run host-specific behavior on that host with one `@pytest.mark.platforms(...)` marker. Never fake `sys.platform`, stack host markers, or use a bare `skipif` for host selection. Tests never write to the user's Hermes home. Keep JavaScript assertions in the owning vitest suite when CI classifies the change as JS-only. The branch guide owns the detailed `wine2e` workflow and test placement rules.

## TypeScript

For the desktop, TUI, and website, prefer feature-owned nanostores over threaded state, thin route roots, narrow hooks, and colocated action modules. Use `interface` for props and shared shapes. Use table-driven dispatch instead of condition ladders. Keep routes in `src/app`, atoms in `src/store`, and pure helpers in `src/lib`.

## Commits and handoff

Rebase onto `main` before merging. A squash from a stale branch can silently revert newer fixes. Add one or two invariant tests per fix and prove them red on the base before the fix. Run the relevant branch checks and report any unrun check. Follow the contributor workflow in [`skills/autonomous-ai-agents/hermes-agent/SKILL.md`](skills/autonomous-ai-agents/hermes-agent/SKILL.md) (`references/contributor-guide.md`) for PR, issue, review, and salvage work.

## Long form

Use [`website/docs/developer-guide/contributing.md`](website/docs/developer-guide/contributing.md) for contribution examples and [`website/docs/developer-guide/`](website/docs/developer-guide/) for subsystem rationale. If a branch guide and this file both mention a rule, the branch guide supplies the local details and this file remains the general policy. Do not create a second copy to resolve a conflict; fix the authoritative source and its links.
