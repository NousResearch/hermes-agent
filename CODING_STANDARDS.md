# Coding standards — load by task

This is the implementation companion to [AGENTS.md](AGENTS.md), not startup context. Read the matching sections for code changes and reviews; read the applicable area `AGENTS.md` as well. For product intent and contribution decisions use the [contribution rubric](website/docs/developer-guide/contributing.md). For setup and the full test workflow use [CONTRIBUTING.md](CONTRIBUTING.md).

## Task index

Read the two common sections for every code change or review: [Before changing behavior](#before-changing-behavior) and [Structure and code shape](#structure-and-code-shape).

Then read only the branches your task touches.

| Task | Section |
|---|---|
| Provisioning the checkout, `scripts/check`, the pre-push hook | [Setup and checks](#setup-and-checks) |
| Dependency, lockfile, Git URL, GitHub Actions or PM environment changes | [Dependencies](#dependencies) |
| TypeScript, desktop, TUI or website UI work | [TypeScript](#typescript) |
| Writing, moving or reviewing tests | [Tests](#tests) |
| Merge readiness, salvage and invariant tests | [Delivery](#delivery) |

Root modules and other unlisted paths follow their area owner: `run_agent.py` → [agent](agent/AGENTS.md); `cli.py` → [hermes_cli](hermes_cli/AGENTS.md); `toolsets.py`, `model_tools.py` → [tools](tools/AGENTS.md); `hermes_state*.py`, `hermes_constants.py` → [Structure and code shape](#structure-and-code-shape); `pyproject.toml`, `uv.lock` → [pm](pm/AGENTS.md); the plugin loader → [plugins](plugins/AGENTS.md); web routers → [web](web/AGENTS.md); `agent/curator*.py` → [skills](skills/AGENTS.md). Read every matching area guide; overlapping routes apply together.

## Before changing behavior

Reproduce the issue on current `main`, locate the failing path and check the intent/history before changing it. Fix the whole class, including sibling call paths. Prefer an existing extension point over a new manager, hook, or core tool. A new hook needs a concrete consumer; a setting belongs in `config.yaml` rather than a non-secret `.env` variable. Keep prompt caching and message-role alternation intact; never inject synthetic user messages mid-loop. A tool that depends on the active desktop/GUI session belongs in a session-selected toolset, not a process-wide environment gate. See [tools/AGENTS.md](tools/AGENTS.md).

For a new capability, take the highest rung of the footprint ladder that solves it: extend existing code; a CLI command + skill; a service-gated tool (`check_fn` answers reachability or opt-in, never per-session surface); a plugin; an MCP server from the catalog; a new core tool last, only when it is fundamental, broadly useful and unreachable through terminal, file, search or browser tools. Extending before duplicating: when three or more open PRs cover one category, design the shared abstraction and make the existing built-in the first provider. See [contribution rubric](website/docs/developer-guide/contributing.md).

## Setup and checks

Use `source ./activate` to provision the PM environment; select isolated `HERMES_HOME` and `HERMES_RUNTIME_DIR` first. The test runner uses a separate interpreter. Run `python scripts/check` for blocking CI checks, then `scripts/run_tests.sh` for relevant tests; do not substitute bare `pytest`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [tests/AGENTS.md](tests/AGENTS.md). A ratchet waiver requires `# health: allow <RULE> -- <why>`; `# noqa` does not waive it.

## Structure and code shape

Find an implementation by topic in `<stem>_<topic>.py` siblings, not by reading a facade top to bottom. Facades hold public entry points and imported names; behavior belongs in topical siblings. Avoid module-level import cycles; late imports may preserve a facade patch seam. Patch the name at its production lookup site, not automatically where it was defined. Do not add compatibility re-exports for internal moves; update code and references in `website/docs`, `skills/`, and area `AGENTS.md` in the same PR. [Codebase ownership](website/docs/developer-guide/codebase-ownership.md) gives long-form examples.

No defensive wrappers around impossible failures, swallowed exceptions, unused flags, or dead code without an exercised resolution path. Comments explain why, not what. Prefer table-driven dispatch to growing name/kind condition ladders. Follow `scripts/code_health/config.py` ratchets (new functions: CC ≤ 20, ≤ 300 lines, nesting ≤ 6; files ≤ 2,000 lines); existing over-limit units must shrink, not grow.

Never hardcode `~/.hermes` for profile-aware code: use `get_hermes_home()` and `display_hermes_home()`. Outside a turn (boot, eviction, callbacks, tickers, RPC, child spawn), bind the owning profile scope and test two homes A→B→A. Process-global `os.environ` and import-time constants belong to the launch profile. See [gateway/AGENTS.md](gateway/AGENTS.md) and [hermes_cli/AGENTS.md](hermes_cli/AGENTS.md).

For process identity, use canonical full-command matchers rather than argv substrings; [hermes_cli/AGENTS.md](hermes_cli/AGENTS.md) documents the seam. Host facts and executable lookup use `hermes_platform`; distinguish control host, terminal target, and remote desktop client. See [hermes_platform/AGENTS.md](hermes_platform/AGENTS.md).

## Dependencies

Use upper-bounded registry dependencies, full commit SHAs for Git sources, and pinned GitHub Actions. After `pyproject.toml` changes run `hermes pm lock` and include `uv.lock`. PM owns Hermes Python dependency updates: do not mutate its environments using raw pip or uv. Plugin packages have their own quarantine policy, distinct from core's `exclude-newer`. The exact bounds and exceptions live in [pm/AGENTS.md](pm/AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## TypeScript

For shared UI state prefer feature-owned nanostores over passing state through multiple components. Rendering components subscribe via `useStore`; actions use `$atom.get()`. Keep route roots thin, hooks single-purpose, and actions colocated. Use interfaces for public props and shared object shapes; extend React primitives where practical. Prefer table-driven ids/routes/views. In desktop, `src/app` owns routes, `src/store` shared atoms, and `src/lib` pure helpers; read both [desktop guides](apps/desktop/AGENTS.md) when touching the UI.

## Tests

Test observable behavior and relationships, not snapshots of catalogs/config versions or source-text regexes. Exercise resolution chains, config, security boundaries, and I/O through real imports under a temporary `HERMES_HOME`; profile changes need two homes A→B→A. Place tests under the matching `tests/<source-area>/` tree. A Python test of a JS-only change belongs in the JS test lane instead. Do not write to a real `~/.hermes` from tests. Do not fake `sys.platform`: host-specific tests use one `@pytest.mark.platforms(...)` marker per test and run on that host. See [tests/AGENTS.md](tests/AGENTS.md) for marker and runner details.

## Delivery

Preserve contributor authorship when salvaging work. Before merging a stale branch, rebase on current `main` and inspect the effective diff so a squash does not revert intervening fixes. For a bug fix, prefer one or two invariant tests proven red on the base. Do not claim CI status without reading it; a documentation-only PR should verify its links, routing, size caps, and the actual context-load path rather than only visual formatting.
