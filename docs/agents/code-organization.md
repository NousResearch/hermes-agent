## Project Structure

Counts shift constantly; the filesystem is canonical. Load-bearing entry points:

```
hermes-agent/
├── run_agent.py          # AIAgent facade; the turn loop lives in agent/turn_*.py
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # TOOLSETS dict, _HERMES_CORE_TOOLS
├── cli.py                # HermesCLI (REPL, slash dispatch) + hermes_cli/cli_*_mixin.py
├── hermes_state.py       # SessionDB facade; hermes_state_*.py siblings
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── hermes_logging.py     # agent.log / errors.log / gateway.log (profile-aware)
├── batch_runner.py       # Parallel batch processing
├── agent/                # turn_*.py loop phases, providers, memory, compression, prompt builder
├── hermes_cli/           # CLI subcommands, setup, config, plugins loader, skins, updater
│   └── web_routers/      # Dashboard FastAPI routers (one per surface); web_server.py mounts them
├── tools/                # Tool implementations, auto-discovered via tools/registry.py
│   └── environments/     # Terminal backends (local, docker, ssh, modal, daytona, singularity)
├── gateway/              # run.py facade + run_*.py phases + session*.py + platforms/
│   ├── platforms/        # One adapter per platform; see platforms/ADDING_A_PLATFORM.md
│   └── builtin_hooks/    # Always-registered gateway hooks (extension point; none shipped)
├── plugins/              # memory/, context_engine/, model-providers/, kanban/, image_gen/, ...
├── skills/               # Built-in skills (by category)   optional-skills/: shipped, not active
├── ui-tui/               # Ink (React) terminal UI — `hermes --tui`
├── tui_gateway/          # Python JSON-RPC backend for TUI + Desktop — server.py + methods_*.py
├── apps/desktop/         # Electron desktop app (+ apps/shared JSON-RPC client)   web/: dashboard SPA
├── acp_adapter/          # ACP server (VS Code / Zed / JetBrains)
├── cron/                 # jobs.py + scheduler.py (+ scheduler_*.py)
├── evals/                # Offline benchmarks (codebase_navigability/, compaction/, ...)
├── scripts/              # run_tests.sh, release.py, check_compat_pointers.py, ci/
├── website/              # Docusaurus docs (developer-guide/ holds the long-form area docs)
└── tests/                # Pytest suite (~39k tests / ~3.7k files, Sep 2026)
```

**User state:** `~/.hermes/config.yaml` (settings), `~/.hermes/.env` (secrets only),
`~/.hermes/logs/` (`agent.log` INFO+, `errors.log` WARNING+, `gateway.log`); all
profile-aware via `get_hermes_home()`. Browse logs with `hermes logs [--follow] [--level] [--session]`.

**Dependency chain:** `tools/registry.py` (no deps) ← `tools/*.py` (register at import) ←
`model_tools.py` (discovery) ← `run_agent.py`, `cli.py`, `batch_runner.py`, `environments/`.

### Facade + siblings layout (Sep 2026 decomposition)

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
  Consult `hermes_cli/plugin_compat.py` and the compatibility manifests for the current
  removal policy. Import from the defining module in-tree.
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

Code paths in this reference are relative to the repository root.
