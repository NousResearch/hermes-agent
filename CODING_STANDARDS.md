# Hermes Agent — Coding Standards

The code conventions that apply to every change, extracted from the root `AGENTS.md` hub, which
points here before code work. Read the section that matches your task; each names the canonical
source for the full policy, and the area `AGENTS.md` files (routing table in the hub) own
area-specific rules.

## Environment and gates

`source ./activate` (fish: `activate.fish`, PowerShell: `activate.ps1`) provisions and activates the
PM environment; pick an isolated `HERMES_HOME`/`HERMES_RUNTIME_DIR` first
([Package management § Developer workflow](website/docs/reference/package-management.md#developer-workflow)).
Tests need the separate test environment in
[CONTRIBUTING.md § Manual development and test environment](CONTRIBUTING.md#manual-development-and-test-environment).

**`python scripts/check`** runs every blocking lint check CI runs, with CI's pinned tools;
`--install-hook pre-push` runs it on every push (re-run after pulling to refresh the hook). PM
commands and environment ownership: [pm/AGENTS.md](pm/AGENTS.md).

## Code quality and the ratchet

- **No defensive clutter:** no wrappers or `try/except: pass` around code that cannot fail, no flags
  nobody sets, no dead code wired in without E2E proof. Comments keep the WHY, cut the WHAT. Name
  ladders become a dict → handler.
- **Size and complexity are ratcheted per unit**
  ([`scripts/code_health/config.py`](scripts/code_health/config.py)): new functions CC ≤ 20, ≤ 300
  lines, nesting ≤ 6; files ≤ 2,000 lines; units already over may only go down (move a function into
  a sibling to offset growth, or put new tests in a new test file). `# noqa` does not waive a
  ratchet finding: `# health: allow <RULE> -- <why>` does. The `HX0xx` ids name code-health rules,
  defined with their fixes in the same config file.

## Facades and siblings

Every former god file is a **facade** (public entry points + the names other packages import) plus
**siblings** `<stem>_<topic>.py`, each owning one topic (`hermes_state.py`, `gateway/run.py`,
`tools/mcp_tool.py`, `hermes_cli/kanban.py`, `hermes_cli/web_server.py`, `cli.py` →
`hermes_cli/cli_*_mixin.py`, `run_agent.py` → `agent/turn_*.py`). New behaviour goes in a sibling,
never a facade.

- **Find code by topic:** `grep -rn "def name" <dir>/<stem>_*.py`, not by reading the facade.
- **Siblings late-import the facade** inside functions; never a module-level cycle.
- **Patch where production reads:** a sibling doing `from <facade> import name` inside the function
  makes the facade the seam; a patch on the defining module passes silently.

### Moving internal symbols

- **No re-export shims for internal moves;** internal paths are not API.
- Moving a symbol means fixing its docs in the same PR: grep `website/docs`, `skills/`, every
  `AGENTS.md`.

## Paths, profiles and machine facts

- **Never hardcode `~/.hermes`:** `get_hermes_home()` for paths, `display_hermes_home()` for text
  (HX001). `_get_profiles_root()` is HOME-anchored on purpose.
- **One process serves many profiles.** Code that runs outside a turn (boot probes, eviction,
  tickers, deferred callbacks, RPC methods, thread hops, child spawns) binds the owning profile
  scope explicitly; `os.environ`, module globals and import-time values hold the launch profile's,
  so an unbound read is a silent default-profile leak. Binding points and seams:
  [gateway/AGENTS.md § Profile scope](gateway/AGENTS.md#profile-scope-adapters-turns-and-everything-between-turns)
  (HX002/HX004/HX005/HX012, PS-P05/P06).
- **Never infer process identity from argv substrings;** use
  `gateway.status.looks_like_gateway_command_line` / `hermes_cli.update_cmd._hermes_holder_subcommand`
  (HX003; details `hermes_cli/AGENTS.md`).
- **Machine facts and executable lookup go through `hermes_platform`**
  ([hermes_platform/AGENTS.md](hermes_platform/AGENTS.md)).
- **User state:** `~/.hermes/config.yaml` (settings), `.env` (secrets only), `logs/` (`hermes logs`);
  all profile-aware via `get_hermes_home()`.

## Dependencies and PM environments

- **Dependencies carry upper bounds** (`>=floor,<next_major`; git URLs and Actions pinned to a SHA);
  after editing `pyproject.toml` run `hermes pm lock` and commit `uv.lock`; never mutate a Hermes
  environment with raw pip/uv. Full policy:
  [pm/AGENTS.md § Dependency pinning policy](pm/AGENTS.md#dependency-pinning-policy).

## TypeScript

Desktop, TUI and website: feature-owned nanostores over threaded state, thin route roots, narrow
hooks and colocated action modules, `interface` for props and shared shapes, table-driven dispatch
over condition ladders; `src/app` routes, `src/store` atoms, `src/lib` pure helpers. Workspace
setup and checks:
[contributing.md § JavaScript workspaces and website](website/docs/developer-guide/contributing.md#javascript-workspaces-and-website).

## Tests

- Always `scripts/run_tests.sh`, never bare `pytest`.
- Behaviour contracts, never change-detectors or source-reading tests.
- Host-specific behaviour is tested on that host with `@pytest.mark.platforms(...)`, never by faking
  `sys.platform`; tests never write to `~/.hermes/`.

Runner, placement, OS markers, `wine2e`, banned test shapes: [tests/AGENTS.md](tests/AGENTS.md) —
[don't fake the host OS](tests/AGENTS.md#dont-fake-the-host-os),
[no change-detector tests](tests/AGENTS.md#dont-write-change-detector-tests),
[no source-reading tests](tests/AGENTS.md#never-read-source-code-in-tests).

## Commits and pull requests

- Rebase onto `main` before merging — a squash from a stale branch silently reverts newer fixes.
- 1–2 invariant tests per fix, proven red on the base.

Process (PR/issue/review/salvage):
[contributing.md § Pull Request Process](website/docs/developer-guide/contributing.md#pull-request-process)
and the `hermes-agent-dev` skill.
