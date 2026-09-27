# Development

Read this before implementation. Paths in code examples are relative to the
repository root unless stated otherwise. See [Architecture](../ARCHITECTURE.md)
for the runtime map and [Testing](testing.md) for the independent test environment.

## Development Environment

```bash
source ./activate   # provisions/syncs PM tools + dependencies, then activates
```
Select an isolated development `HERMES_HOME` and `HERMES_RUNTIME_DIR` first;
see `website/docs/reference/package-management.md#developer-workflow`.
PowerShell: `. .\activate.ps1`. `deactivate` restores the prior environment.
For tests, use the independent test environment in `CONTRIBUTING.md` (or Nix);
PM activation's `PYTHONPATH` does not survive the test runner's environment scrub.
`scripts/run_tests.sh` probes `.venv`, then `venv`, then `$HOME/.hermes/hermes-agent/venv`
(worktrees sharing the main checkout's venv).

## Facade + siblings layout (Sep 2026 decomposition)

Every former god file is a **facade** (public entry points + the names other packages import)
plus **siblings** `<stem>_<topic>.py` in the same directory, each owning one topic. Largest
families: `hermes_state.py` (21), `gateway/run.py` (15), `tools/mcp_tool.py` (15),
`hermes_cli/kanban.py` (14), `hermes_cli/web_server.py` (13 + 24 routers), `hermes_cli/auth.py`
(12), `tools/browser_tool.py` (11), `cli.py` (12 `hermes_cli/cli_*_mixin.py`), `run_agent.py`
(`agent/turn_*.py`, `agent_init.py`, `conversation_loop.py`).

- **Find code by topic, not by facade:** `grep -rn "def name" <dir>/<stem>_*.py`. Reading the
  facade first is the expensive way (`evals/codebase_navigability/`).
- **Siblings may import each other and late-import the facade** inside functions. A facade
  never imports a sibling at module level *and* gets imported by that sibling at module level.
- **Patch where production reads.** Siblings often do `from <facade> import name` inside the
  function so `monkeypatch.setattr(facade, "name", ...)` is the seam; a patch on the defining
  module passes silently. Check the call site's binding before writing a patch target
  (blind repointing to defining modules broke 130+ tests).
- **Compat pointers are OFF LIMITS in-tree.** Old import paths kept alive for external plugins
  (`PLUGIN-COMPAT` blocks, `COMPAT_MANIFEST.md`, `compat_manifest.json`) must not be used by
  in-tree code or tests; `scripts/check_compat_pointers.py` runs in CI, and
  `-W error::hermes_cli.plugin_compat.HermesPluginCompatWarning` catches them in the suite.
  They are removed 2026-09-14 by reverting one commit. Import from the defining module.
- **Don't recreate god files.** A file passing ~2,000 lines or a function passing ~300 lines /
  cyclomatic complexity 30 is the signal to split along `<stem>_<topic>` FIRST, in its own
  commit. New behaviour goes in a new or topical sibling — never appended to a facade.
- **No `if/elif` ladders ≥ 4 branches keyed on a name/kind** — use a dict/table → handler
  (`_SLASH_DISPATCH` in `cli.py`, `_command_handler_table` in the gateway are the shape).
- **No re-export shims for internal moves** ("keep the old name importable"). Internal paths
  are not API; external compat is handled ONCE by the compat layer, not per PR.
- **Moving a symbol means fixing its docs in the same PR:** grep `website/docs`,
  `skills/`, and every `AGENTS.md` for the old `path.py` + symbol (23 doc files went stale
  after the refactor). `evals/codebase_navigability/static_metrics.py <tree> <label>` measures
  file/function/CC/elif distributions before/after a large PR in ~2 min.

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
  PR #3575). Profile operations themselves are HOME-anchored
  (`_get_profiles_root()` = `Path.home()/.hermes/profiles`) so `hermes -p x profile list`
  sees all profiles — intentional, not a bug.
- **One process may serve many profiles; code that runs outside a turn binds the owning
  profile scope explicitly.** A profile = home + secret scope + terminal scope, bound by
  `gateway/run.py::_profile_runtime_scope` (turn), `tui_gateway/server.py::@_profile_scoped` +
  `model_switch.py::_session_profile_runtime_scope` (RPC, teardown), `cron/scheduler_provider.py::
  _profile_cron_scope` (ticker), `gateway/run_agent_cache.py::_run_release_in_profile_scope`
  (eviction). `os.environ`, module globals and import-time values hold the *launch* profile's, so
  an unbound read is a silent default-profile leak, never an error: home/config/`.env`-derived
  module constants are a bug class — key slots by `hermes_home_key()` or resolve at call time.
  Needs a binding: boot probes (`check_fn`, MCP discovery, hooks), session end/eviction, tickers,
  deferred callbacks, RPC methods, config readers, thread hops (`spawn_context_thread`), child
  spawns (`served_profile_child_env`, never `os.environ.copy()`). Fail-closed reads exist only after
  `set_multiplex_active(True)`. Prove live with two homes (A→B→A) under multiplex, not one temp
  `HERMES_HOME`. Advisory lint: `scripts/check_profile_scope_patterns.py`.
- **Machine facts and resource lookup go through `hermes_platform`.** `hermes_platform.host` is the
  one answer for OS family, native architecture (`IsWow64Process2` → `platform.machine()`; never
  `PROCESSOR_ARCHITECTURE` alone, it reads AMD64 under x64-on-ARM64 emulation), CPU identity, and
  WSL/container/Termux. Facts are cached per process and take **no environment-variable input**, so
  a hardware recognizer (`host/products.py`) cannot be set from a shell. Distinguish the control
  host (where this Python runs) from the terminal execution target (SSH/container) and the Desktop
  client (another machine): `host.*` answers only the first. A new bare `shutil.which` or a
  hand-written known-path table outside `hermes_platform/` fails
  `tests/test_managed_runtime_resolution.py` unless allowlisted with a reason; resolvers land in
  `hermes_platform/resolver/`. Lookup never installs, downloads, or starts anything.
- **Argparse alias dispatch:** `add_parser("list", aliases=["ls"])` sets `dest` to the literal
  the user typed (`"ls"`). Dispatch must accept both (caught PTY-testing `hermes webhook ls`).
- **Don't wire in dead code without E2E validation.** Unshipped code was dead for a reason;
  E2E the real resolution chain with real imports against a temp `HERMES_HOME` first.
