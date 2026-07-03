# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

**Never give up on the right solution.**

## AGENTS.md is Runtime Prompt Surface

This file is loaded into coding-agent context, so edits affect latency, cost,
and instruction salience for every session in this repo. Keep it as the compact
runtime contract: invariants, high-blast-radius pitfalls, core file map, and
verification rules. Move long rationale, history, exhaustive subsystem docs, and
rarely-needed examples to developer docs or references, leaving short summaries
and links here. Treat changes to this file as context-impacting, not ordinary
documentation cleanup.

Detailed rationale and long subsystem notes live in
`website/docs/developer-guide/agents-md-reference.md`; load that file only when
the current task needs the extra detail.

## What Hermes Is

Hermes is a personal AI agent that runs the same agent core across a CLI, a
messaging gateway (Telegram, Discord, Slack, and ~20 other platforms), a TUI,
and an Electron desktop app. It learns across sessions (memory + skills),
delegates to subagents, runs scheduled jobs, and drives a real terminal and
browser. It is extended primarily through **plugins and skills**, not by
growing the core.

Two properties shape almost every design decision and are the lens for
reviewing any change:

- **Per-conversation prompt caching is sacred.** A long-lived conversation
  reuses a cached prefix every turn. Anything that mutates past context,
  swaps toolsets, or rebuilds the system prompt mid-conversation invalidates
  that cache and multiplies the user's cost. We do not do it (the one
  exception is context compression).
- **The core is a narrow waist; capability lives at the edges.** Every model
  tool we add is sent on every API call, so the bar for a new *core* tool is
  high. Most new capability should arrive as a CLI command + skill, a
  service-gated tool, or a plugin — not as core surface.

## Important Policies

### Dependency Pinning Is Supply-Chain Control

When adding or changing dependencies, keep the supply-chain invariant inline:
PyPI ranges need `>=floor,<next_major`, Git URL dependencies pin a full commit
SHA, GitHub Actions pin a full SHA plus a version comment, and CI-only pip
installs may use `==exact`. Do not add unbounded `>=X.Y.Z` specs. See
[Dependency pinning policy](CONTRIBUTING.md#dependency-pinning-policy-supply-chain-hardening)
and [AGENTS.md Reference](website/docs/developer-guide/agents-md-reference.md#dependency-pinning-policy).

### Prompt Caching Must Not Break

Hermes-Agent ensures caching remains valid throughout a conversation. **Do NOT implement changes that would:**
- Alter past context mid-conversation
- Change toolsets mid-conversation
- Reload memories or rebuild system prompts mid-conversation

Cache-breaking forces dramatically higher costs. The ONLY time we alter context is during context compression.

Slash commands that mutate system-prompt state (skills, tools, memory, etc.)
must be **cache-aware**: default to deferred invalidation (change takes
effect next session), with an opt-in `--now` flag for immediate
invalidation. See `/skills install --now` for the canonical pattern.

### Background Process Notifications (Gateway)

When `terminal(background=true, notify_on_complete=true)` is used, the gateway runs a watcher that
detects process completion and triggers a new agent turn. Control verbosity of background process
messages with `display.background_process_notifications`
in config.yaml (or `HERMES_BACKGROUND_NOTIFICATIONS` env var):

- `all` — running-output updates + final message (default)
- `result` — only the final completion message
- `error` — only the final message when exit code != 0
- `off` — no watcher messages at all

## Known Pitfalls

### DO NOT hardcode `~/.hermes` paths
Use `get_hermes_home()` from `hermes_constants` for code paths. Use `display_hermes_home()`
for user-facing print/log messages. Hardcoding `~/.hermes` breaks profiles — each profile
has its own `HERMES_HOME` directory. This was the source of 5 bugs fixed in PR #3575.

### DO NOT introduce new `simple_term_menu` usage
Existing call sites in `hermes_cli/main.py` remain for legacy fallback only;
the preferred UI is curses (stdlib) because `simple_term_menu` has
ghost-duplication rendering bugs in tmux/iTerm2 with arrow keys. New
interactive menus must use `hermes_cli/curses_ui.py` — see
`hermes_cli/tools_config.py` for the canonical pattern.

### DO NOT use `\033[K` (ANSI erase-to-EOL) in spinner/display code
Leaks as literal `?[K` text under `prompt_toolkit`'s `patch_stdout`. Use space-padding: `f"\r{line}{' ' * pad}"`.

### `_last_resolved_tool_names` is a process-global in `model_tools.py`
`_run_single_child()` in `delegate_tool.py` saves and restores this global around subagent execution. If you add new code that reads this global, be aware it may be temporarily stale during child agent runs.

### DO NOT hardcode cross-tool references in schema descriptions
Tool schema descriptions must not mention tools from other toolsets by name (e.g., `browser_navigate` saying "prefer web_search"). Those tools may be unavailable (missing API keys, disabled toolset), causing the model to hallucinate calls to non-existent tools. If a cross-reference is needed, add it dynamically in `get_tool_definitions()` in `model_tools.py` — see the `browser_navigate` / `execute_code` post-processing blocks for the pattern.

### The gateway has TWO message guards — both must bypass approval/control commands
When an agent is running, messages pass through two sequential guards:
(1) **base adapter** (`gateway/platforms/base.py`) queues messages in
`_pending_messages` when `session_key in self._active_sessions`, and
(2) **gateway runner** (`gateway/run.py`) intercepts `/stop`, `/new`,
`/queue`, `/status`, `/approve`, `/deny` before they reach
`running_agent.interrupt()`. Any new command that must reach the runner
while the agent is blocked (e.g. approval prompts) MUST bypass BOTH
guards and be dispatched inline, not via `_process_message_background()`
(which races session lifecycle).

### Squash merges from stale branches silently revert recent fixes
Before squash-merging a PR, ensure the branch is up to date with `main`
(`git fetch origin main && git reset --hard origin/main` in the worktree,
then re-apply the PR's commits). A stale branch's version of an unrelated
file will silently overwrite recent fixes on main when squashed. Verify
with `git diff HEAD~1..HEAD` after merging — unexpected deletions are a
red flag.

### Don't wire in dead code without E2E validation
Unused code that was never shipped was dead for a reason. Before wiring an
unused module into a live code path, E2E test the real resolution chain
with actual imports (not mocks) against a temp `HERMES_HOME`.

### Tests must not write to `~/.hermes/`
The `_isolate_hermes_home` autouse fixture in `tests/conftest.py` redirects `HERMES_HOME` to a temp dir. Never hardcode `~/.hermes/` paths in tests.

**Profile tests**: When testing profile features, also mock `Path.home()` so that
`_get_profiles_root()` and `_get_default_hermes_home()` resolve within the temp dir.
Use the pattern from `tests/hermes_cli/test_profiles.py`:
```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

## Testing

**ALWAYS use `scripts/run_tests.sh`** — do not call `pytest` directly for
normal verification. The wrapper enforces CI-parity: credential env vars unset,
`HOME` / `HERMES_HOME` isolated, TZ=UTC, LANG=C.UTF-8, xdist, and the in-tree
subprocess-per-test isolation plugin.

```bash
scripts/run_tests.sh                                  # full suite, CI-parity
scripts/run_tests.sh tests/gateway/                   # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # one test
scripts/run_tests.sh -v --tb=long                     # pass-through pytest flags
scripts/run_tests.sh --no-isolate tests/foo/          # disable isolation only for debugging
```

If you must bypass the wrapper (e.g. IDE), activate `.venv` / `venv` and run
`python -m pytest ...`; the isolation plugin still loads unless `--no-isolate`
is passed.

Don't write change-detector tests for expected-to-change data (model catalogs,
config version literals, provider enumeration counts). Assert behavior and
relationships instead: catalog plumbing works, migrations bump to current,
plan-only models don't leak, and every catalog model has required metadata.
See `website/docs/developer-guide/agents-md-reference.md#testing` for examples.

## Contribution Rubric — Runtime Summary

Hermes ships a lot at the edges and stays conservative at the core waist.
Use this rubric to avoid wrong-premise fixes and permanent core footprint.
Full rationale and examples are in
`website/docs/developer-guide/agents-md-reference.md#contribution-rubric`.

What we want:
- Fix real, reproduced bugs on current `main`; point to the line where the bug
  manifests and fix the class, not just one call site.
- Expand product reach at the edges: platform adapters, providers, desktop/TUI,
  dashboard features, plugins, skills, and setup/config UX.
- Refactor god-files into focused modules when the request is explicitly a
  refactor and behavior is preserved.
- Keep the core narrow: new model tools are expensive because every tool ships
  on every API call.
- Extend existing infrastructure instead of duplicating managers/hooks.
- Preserve cache stability, strict message role alternation, and byte-stable
  system prompts for the life of a conversation.

What we don't want:
- Speculative hooks with no concrete consumer.
- New user-facing `HERMES_*` env vars for non-secret config; use config.yaml.
- New core tools when terminal/file/CLI+skill/plugin/MCP can solve it.
- Lazy-reading pagination on instructional tools the agent must read fully.
- Security "fixes" that destroy the protected feature's purpose.
- Outbound telemetry/attribution without explicit opt-in gating.
- Change-detector tests or cache-breaking mid-conversation changes.

Before calling something a bug, verify both the premise and the original intent
against the codebase (`git log -p -S` is often useful). If unsure, ask rather
than shipping a fix that fights the design.

Footprint ladder for new capability, least footprint first:
1. Extend existing code.
2. CLI command + skill.
3. Service-gated tool (`check_fn`) that appears only when configured.
4. Plugin.
5. MCP server in the catalog.
6. New core tool only as a last resort for broadly useful fundamentals.

Plugins MUST NOT modify core files; expose capability through generic plugin
surfaces instead.

## Development Environment

```bash
# Prefer .venv; fall back to venv if that's what your checkout has.
source .venv/bin/activate   # or: source venv/bin/activate
```

`scripts/run_tests.sh` probes `.venv` first, then `venv`, then
`$HOME/.hermes/hermes-agent/venv` (for worktrees that share a venv with the
main checkout).

## Project Structure

File counts shift constantly — the filesystem is canonical. Load detailed
subsystem reference from `website/docs/developer-guide/agents-md-reference.md`
when needed. Load-bearing entry points:

- `run_agent.py` — `AIAgent` core conversation loop.
- `agent/prompt_builder.py` / `agent/system_prompt.py` — prompt assembly,
  context files, truncation, skills prompt cache.
- `model_tools.py`, `toolsets.py`, `tools/registry.py` — tool discovery,
  schema shaping, execution, and toolset membership.
- `cli.py`, `hermes_cli/` — CLI, setup, config, subcommands, curses UI.
- `gateway/` — messaging gateway runner, sessions, adapters, platform guards.
- `plugins/` — plugin system; third-party/niche capability belongs here when
  possible instead of core.
- `skills/`, `optional-skills/` — bundled and optional procedural memory.
- `cron/`, `tools/cronjob_tools.py` — scheduled jobs.
- `tests/` — pytest suite; use `scripts/run_tests.sh`.

User config is under `get_hermes_home()` (`config.yaml`, `.env` for secrets
only). Logs are under `get_hermes_home() / "logs"`; browse with
`hermes logs [--follow] [--level ...] [--session ...]`.

## TypeScript Style

Applies to TypeScript across Hermes: desktop, TUI, website, and future TS packages.

- Prefer small nanostores over component state when state is shared, reused, or read by distant UI.
- Let each feature own its atoms. Chat state belongs near chat, shell state near shell, shared state in `src/store`.
- Components that render from an atom should use `useStore`. Non-rendering actions should read with `$atom.get()`.
- Do not pass state through three components when the leaf can subscribe to the atom.
- Keep persistence beside the atom that owns it.
- Keep route roots thin. They compose routes and shell; they should not become controllers.
- No monolithic hooks. A hook should own one narrow job.
- Prefer colocated action modules over hidden god hooks.
- If a callback is pure side effect, use the terse void form:
  `onState={st => void setGatewayState(st)}`.
- Async UI handlers should make intent explicit:
  `onClick={() => void save()}`.
- Prefer interfaces for public props and shared object shapes. Avoid `type X = { ... }` for object props.
- Extend React primitives for props: `React.ComponentProps<'button'>`, `React.ComponentProps<typeof Dialog>`, `Omit<...>`, `Pick<...>`.
- Table-driven beats condition ladders when mapping ids, routes, or views.
- `src/app` owns routes, pages, and page-specific components.
- `src/store` owns shared atoms.
- `src/lib` owns shared pure helpers.

## Architecture Reference Index

Keep runtime rules inline; open
`website/docs/developer-guide/agents-md-reference.md` for long subsystem detail.

- File dependency chain: CLI/gateway/TUI/API construct an `AIAgent`, which uses
  prompt builders, provider adapters, model tools, terminal/browser/file tools,
  memory, skills, and session storage. Trace definitions/usages before editing.
- AIAgent / CLI / TUI architecture: avoid god-file growth unless doing an
  explicit extraction; preserve prompt caching and role alternation.
- Adding tools: prefer existing tools, CLI+skill, service-gated tools, plugins,
  or MCP before core tools. New core schemas must avoid cross-tool references
  unless added dynamically in `get_tool_definitions()`.
- Configuration: user-facing non-secret behavior belongs in config.yaml, not
  `.env`; use profile-aware paths (`get_hermes_home()`, `display_hermes_home()`).
- Plugins/skills/toolsets/delegation/cron/kanban/curator: keep capability at the
  edges and load detailed references only for tasks touching that subsystem.

## Profiles: Multi-Instance Support

Hermes supports **profiles** — multiple fully isolated instances, each with its own
`HERMES_HOME` directory (config, API keys, memory, sessions, skills, gateway, etc.).

The core mechanism: `_apply_profile_override()` in `hermes_cli/main.py` sets
`HERMES_HOME` before any module imports. All `get_hermes_home()` references
automatically scope to the active profile.

### Rules for profile-safe code

1. **Use `get_hermes_home()` for all HERMES_HOME paths.** Import from `hermes_constants`.
   NEVER hardcode `~/.hermes` or `Path.home() / ".hermes"` in code that reads/writes state.
2. **Use `display_hermes_home()` for user-facing messages.** Import from `hermes_constants`.
   This returns `~/.hermes` for default or `~/.hermes/profiles/<name>` for profiles.
3. **Module-level constants are fine** — they cache `get_hermes_home()` at import time,
   which is AFTER `_apply_profile_override()` sets the env var. Just use `get_hermes_home()`,
   not `Path.home() / ".hermes"`.
4. **Tests that mock `Path.home()` must also set `HERMES_HOME`** since code now uses
   `get_hermes_home()`.
5. **Gateway platform adapters should use token locks** — if the adapter connects with
   a unique credential (bot token, API key), call `acquire_scoped_lock()` from
   `gateway.status` in `connect()`/`start()` and release it in `disconnect()`/`stop()`.
6. **Profile operations are HOME-anchored, not HERMES_HOME-anchored** — `_get_profiles_root()`
   returns `Path.home() / ".hermes" / "profiles"`, NOT `get_hermes_home() / "profiles"`.
   This is intentional so any active profile can list all profiles.
