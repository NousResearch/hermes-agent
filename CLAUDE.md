# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Canonical references

- **`AGENTS.md`** — the long-form development guide (~1100 lines). It owns the authoritative rules on tools, toolsets, plugins, skills, curator, cron, kanban, delegation, profiles, prompt caching, and Windows footguns. **Read it before designing any non-trivial change.**
- **`CONTRIBUTING.md`** — contributor-facing version: setup, "skill vs tool" decision, cross-platform rules, dependency-pinning policy, PR/commit conventions.
- **`README.md`** — user-facing product description and install paths.
- **`pyproject.toml`** — every dep is exact-pinned (`==X.Y.Z`); see the comment block above `dependencies` for the supply-chain rationale before changing it.

When the rules below and `AGENTS.md` agree, follow either. When they disagree, `AGENTS.md` is authoritative — this file is just the launching pad.

## Common commands

```bash
./hermes                       # thin launcher — invokes hermes_cli.main:main under the current python
hermes doctor                  # diagnostics
hermes setup                   # interactive setup wizard
hermes gateway start           # run the messaging gateway

# Dev install (first time) — sets up uv + venv + .[all] + symlinks ~/.local/bin/hermes
./setup-hermes.sh
# Manual equivalent:
uv venv .venv --python 3.11 && source .venv/bin/activate && uv pip install -e ".[all,dev]"
```

`scripts/run_tests.sh` (not `./hermes`) is the script that probes `.venv` → `venv` → `$HOME/.hermes/hermes-agent/venv`. `./hermes` itself does no venv detection — activate the venv yourself or run it through `setup-hermes.sh`'s `~/.local/bin/hermes` symlink.

### Testing — **always use the wrapper**

```bash
scripts/run_tests.sh                                  # full suite (CI-parity)
scripts/run_tests.sh tests/agent/                     # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # one test
scripts/run_tests.sh -- -v --tb=long                  # pass-through pytest args (after `--`)
scripts/run_tests.sh --no-isolate tests/foo/          # disable subprocess isolation (debug only)
```

`run_tests.sh` wraps `scripts/run_tests_parallel.py` with `env -i` + `TZ=UTC LANG=C.UTF-8 PYTHONHASHSEED=0` and per-file subprocess isolation. Calling `pytest` directly diverges from CI in five documented ways (credentials leaking through, real `~/.hermes/`, local TZ, locale, worker count) — don't do it unless your IDE forces you to. The hermetic env is what catches "works locally, fails in CI" regressions.

### TUI dev (Ink + React, in `ui-tui/`)

```bash
cd ui-tui && npm install     # first time
npm run dev                  # watch mode
npm run build                # production build
npm run type-check           # tsc --noEmit
npm test                     # vitest
```

## Architecture in 30 seconds

```
hermes-agent/
├── run_agent.py            AIAgent — core sync conversation loop (~4.3k LOC, ~60-arg __init__)
├── cli.py                  HermesCLI — interactive prompt_toolkit CLI (~14.8k LOC)
├── model_tools.py          Tool orchestration; importing it triggers tool + plugin discovery
├── toolsets.py             TOOLSETS dict; _HERMES_CORE_TOOLS is the default bundle
├── hermes_state.py         SessionDB — SQLite + FTS5 session store
├── hermes_constants.py     get_hermes_home(), display_hermes_home() — profile-aware paths
├── batch_runner.py         Parallel trajectory generation
├── agent/                  Provider adapters, memory, compression, prompt builder, curator
├── hermes_cli/             Subcommands, setup wizard, slash-command registry, skin engine
├── tools/                  Self-registering tool impls; tools/environments/ = terminal backends
├── gateway/                Messaging gateway runner + per-platform adapters
├── plugins/                memory/, model-providers/, context_engine/, image_gen/, kanban/, ...
├── skills/                 Bundled skills (loaded by default)
├── optional-skills/        Heavier/niche skills (discoverable via hub, not active by default)
├── ui-tui/                 Ink terminal UI for `hermes --tui`
├── tui_gateway/            Python JSON-RPC backend for the Ink TUI
├── cron/                   Scheduler (jobs.py + scheduler.py)
└── tests/                  ~17k tests across ~900 files; never writes to real ~/.hermes/
```

Tool dependency chain: `tools/registry.py` → `tools/*.py` (self-register at import) → `model_tools.py` → `run_agent.py` / `cli.py` / `batch_runner.py`. Adding a tool requires editing **two** files: `tools/<name>.py` (registers handler + schema) and `toolsets.py` (wires the name into a toolset — without this the tool is registered but invisible to the agent).

Slash commands live in a single `COMMAND_REGISTRY` in `hermes_cli/commands.py`; CLI, gateway, Telegram, Slack, autocomplete, and help all derive from it — adding an alias is a one-line tuple edit.

## Load-bearing invariants (don't break these)

These have incident history — each one cost real debug time when violated.

1. **Prompt caching must not break mid-conversation.** Never alter past context, change toolsets, or rebuild the system prompt mid-conversation outside of explicit context compression. Slash commands that mutate prompt state (skills/tools/memory) default to deferred invalidation (next session); immediate invalidation requires an opt-in `--now` flag — see `/skills install --now` as the canonical pattern.

2. **Profile-aware paths.** Use `get_hermes_home()` for state paths and `display_hermes_home()` for user-facing messages — both from `hermes_constants`. **Never** hardcode `Path.home() / ".hermes"` or `~/.hermes` in code that reads/writes state; it silently breaks multi-profile installs. Exception: `_get_profiles_root()` is HOME-anchored on purpose so `hermes -p coder profile list` sees all profiles.

3. **Tests must not write to real `~/.hermes/`.** The `_isolate_hermes_home` autouse fixture in `tests/conftest.py` redirects `HERMES_HOME` to a tmpdir. For profile tests, also patch `Path.home()` — see `tests/hermes_cli/test_profiles.py`.

4. **Dependency pinning.** Every direct dep in `pyproject.toml` is `==X.Y.Z` (exact). When adding/bumping, also `uv lock` to regenerate transitives. Reason: PyPI can ship a fresh transitive version at any time without our review (Mini Shai-Hulud, May 2026, hit `mistralai 2.4.6` — pin saved us). Git URLs use full commit SHAs; GitHub Actions use SHA + version comment.

5. **Cross-platform.** Never `os.kill(pid, 0)` for liveness — on Windows it maps to `CTRL_C_EVENT` and kills the target's console group. Use `psutil.pid_exists(pid)` instead. `shutil.which()` before shelling out — Windows lacks most POSIX CLI tools (`ps`, `kill`, `grep`, `awk`, `fuser`, `lsof`, `pgrep`), and `wmic` was removed in Windows 10 21H1+. Guard `termios`/`fcntl`/`os.setsid`/POSIX signals with `platform.system()` checks. Run `scripts/check-windows-footguns.py` on your diff before pushing.

6. **No `\033[K` in spinner/display code** — leaks as literal `?[K` under `prompt_toolkit`'s `patch_stdout`. Use space-padding: `f"\r{line}{' ' * pad}"`.

7. **No new `simple_term_menu` callsites** — it ghost-duplicates in tmux/iTerm2. Use `hermes_cli/curses_ui.py` (canonical pattern: `hermes_cli/tools_config.py`).

8. **No new in-tree memory providers.** `plugins/memory/` is closed (May 2026 policy). New memory backends ship as standalone plugin repos that implement `MemoryProvider` and install into `~/.hermes/plugins/`. PRs adding a new `plugins/memory/<name>/` directory will be closed. Bug fixes to existing in-tree providers are still welcome.

9. **Plugins must not edit core files.** If a plugin needs a capability the framework doesn't expose, expand the generic plugin surface (new hook, new `ctx` method) — never hardcode plugin-specific logic into `run_agent.py` / `cli.py` / `gateway/run.py` / `hermes_cli/main.py`.

10. **The gateway has two message guards** — `gateway/platforms/base.py::_pending_messages` queues messages when an agent is active, and `gateway/run.py` intercepts `/stop`, `/new`, `/queue`, `/status`, `/approve`, `/deny` before they reach `interrupt()`. Any new command that must reach the runner while the agent is blocked MUST bypass both guards (dispatched inline, not via `_process_message_background()`).

11. **Don't write change-detector tests.** Tests that snapshot model catalogs, config version literals, or enumeration counts break every routine source update and add no behavioral coverage. Write invariants ("every model in the catalog has a context-length entry"), not snapshots ("`gemini-2.5-pro` is in `_PROVIDER_MODELS['gemini']`").

## Skill vs tool — the recurring question

When a new capability is requested, the answer is almost always **skill** (instructions + existing tools + maybe a helper script in `scripts/`). Reach for a new tool only when the work needs end-to-end API key/auth management baked into the agent harness, deterministic processing that can't be best-effort, or binary/streaming data that can't go through `terminal`.

Bundled skills go in `skills/`; heavier or niche official skills go in `optional-skills/` and are installed on demand via `hermes skills install`. Specialized/community skills go to the Skills Hub, not this tree.

`SKILL.md` description must be ≤60 characters, one sentence, ending in a period — no marketing words ("powerful", "comprehensive", "seamless"), no name repetition. Reference Hermes tools by backticked name (`` `terminal` ``, `` `search_files` ``, `` `patch` ``) — never shell utilities the agent already wraps. See `AGENTS.md` §Skills for the full hardline standards.

## Conventional Commits

`<type>(<scope>): <description>` — types: `fix`, `feat`, `docs`, `test`, `refactor`, `chore`. Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, etc.
