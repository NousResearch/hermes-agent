# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.

**Never give up on the right solution.**

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

## Contribution Rubric — What We Want / What We Don't

This is the project's intent layer. Use it two ways:

1. **For humans and for your own work** — what gets merged and what gets
   rejected, so a contribution aims at the target.
2. **For automated review (the triage sweeper)** — guidance on when a PR is
   safe to close on the three allowed reasons (`implemented_on_main`,
   `cannot_reproduce`, `incoherent`) and, just as important, **when NOT to
   close** one. Taste-based "we don't want this / out of scope" closes are NOT
   an automated decision — those stay with a human maintainer. The sweeper's
   job here is to recognize design intent and *avoid wrongly closing a
   legitimate contribution*, not to make the won't-implement call itself.

Read the balance right: Hermes ships a **lot** — most merges are bug fixes to
real reported behavior, and the product surface (platforms, channels,
providers, models, desktop/TUI features) expands aggressively and on purpose.
The restraint below is aimed squarely at the **core agent + the model tool
schema**, the one place where every addition is paid for on every API call.
"Smallest footprint" governs *how a capability is wired into the core*, NOT
whether the product is allowed to grow. We are expansive at the edges and
conservative at the waist.

### What we want

- **Fix real bugs, well.** The bulk of what lands is `fix(...)` against an
  actual reported symptom. A good fix reproduces the symptom on current
  `main`, points to the exact line where it manifests, and fixes the whole bug
  class — sibling call paths included — not just the one site the reporter hit.
- **Expand reach at the edges.** New platform adapters, channels, providers,
  models, and desktop/TUI/dashboard features are welcome and land routinely,
  including large ones (a new messaging channel, a session-cap feature, a
  Windows PTY bridge). Breadth in the product is a goal, not a footprint
  concern — as long as it integrates with the existing setup/config UX
  (`hermes tools`, `hermes setup`, auto-install) rather than bolting on a raw
  env var.
- **Refactor god-files into clean modules.** Extracting a multi-thousand-line
  cluster out of `cli.py` / `run_agent.py` / `gateway/run.py` into a focused
  mixin or module is wanted work, even when the diff is huge and mechanical
  (large `+N/-N` refactors merge regularly). The "every line traces to the
  request" test applies to *feature* PRs; a declared refactor's request IS the
  extraction.
- **Keep the core narrow.** New *model tools* are the expensive exception —
  every tool ships on every API call. Prefer, in order: extend existing code →
  CLI command + skill → service-gated tool (`check_fn`) → plugin → MCP server
  in the catalog → new core tool (last resort). See "The Footprint Ladder."
- **Extend, don't duplicate.** Before adding a module/manager/hook, check
  whether existing infrastructure already covers the use case. When several PRs
  integrate the same *category*, design one shared interface instead of merging
  them one at a time (see the ABC + orchestrator note under the Footprint
  Ladder).
- **Behavior contracts over snapshots.** Tests should assert how two pieces of
  data must relate (invariants), not freeze a current value (model lists,
  config version literals, enumeration counts). See "Don't write
  change-detector tests."
- **E2E validation, not just green unit mocks.** For anything touching
  resolution chains, config propagation, security boundaries, remote
  backends, or file/network I/O, exercise the real path with real imports
  against a temp `HERMES_HOME`. Mocks hide integration bugs.
- **Cache-, alternation-, and invariant-safe.** Preserve prompt caching, strict
  message role alternation (never two same-role messages in a row; never a
  synthetic user message injected mid-loop), and a system prompt that is
  byte-stable for the life of a conversation.
- **Contributor credit preserved.** Salvage external work by cherry-picking
  (rebase-merge) so authorship survives in git history; don't reimplement from
  scratch when you can build on top.

### What we don't want (rejected even when well-built)

- **Speculative infrastructure.** Hooks, callbacks, or extension points with no
  concrete consumer. Adding a hook is easy; removing one after plugins depend
  on it is hard. A hook is NOT speculative if a contributor has a real, stated
  use case — even if the consumer ships separately.
- **New `HERMES_*` env vars for non-secret config.** `.env` is for secrets
  only (API keys, tokens, passwords). All behavioral settings — timeouts,
  thresholds, feature flags, display prefs — go in `config.yaml`. Bridge to an
  internal env var if the mechanism needs one, but user-facing docs point to
  `config.yaml`. Reject PRs that tell users to "set X in your .env" unless X
  is a credential.
- **A new core tool when terminal + file already do the job, or when a skill
  would.** If the only barrier is file visibility on a remote backend, fix the
  mount, not the toolset.
- **Lazy-reading escape hatches on instructional tools.** No `offset`/`limit`
  pagination on tools that load content the agent must read fully (skills,
  prompts, playbooks). Models will read page 1 and skip the rest.
- **"Fixes" that destroy the feature they secure.** A mitigation that kills the
  feature's purpose is the wrong mitigation. Read the original commit's intent
  (`git log -p -S`) before restricting behavior; find a fix that preserves the
  feature.
- **Outbound telemetry / usage attribution without opt-in gating.** No new
  analytics, third-party identifier tagging, or attribution tags until a
  generic user-facing opt-in (config gate + setup prompt + `hermes tools`
  toggle) exists. Park behind a label, do not merge.
- **Change-detector tests, cache-breaking mid-conversation, dead code wired in
  without E2E proof, and plugins that touch core files.** Plugins live in their
  own directory and work within the ABCs/hooks we provide; if a plugin needs
  more, widen the generic plugin surface, don't special-case it in core.

### Before you call it a bug — verify the premise (and when NOT to close)

The most common reason a well-written PR gets closed is not code quality — it
is that the change is built on a **wrong premise**, or it treats an
**intentional design as a gap**. These patterns cut both ways: they tell a
human reviewer what to scrutinize, and they tell the automated sweeper when a
PR is NOT safe to close as `implemented_on_main` / `cannot_reproduce` (when in
doubt, leave it open for a human). They are distilled from real closes.

- **"Intentional design, not a gap."** A limitation that looks like an
  oversight is often deliberate. Before "fixing" a missing link or a
  restriction, ask whether the isolation IS the design. Example: profiles are
  independent islands on purpose — a PR adding live config inheritance from the
  default profile was closed because coupling profiles together is exactly what
  the design prevents (the copy-at-creation `--clone` path already covers the
  legitimate "start from my default" case). Read the original commit's intent
  (`git log -p -S "<symbol>"`) before assuming something is unfinished.
- **"The premise doesn't hold against how X actually works."** A PR's
  justification frequently rests on a wrong mental model of an existing
  mechanism. Trace the real code/runtime before accepting the rationale. Two
  real closes: a rate-limit "re-probe during cooldown" PR (the breaker only
  trips on a *confirmed-empty* account bucket, so re-probing just hammers a
  bucket we've already proven empty); a usage-accumulation fix whose new branch
  **never executes at runtime** because an earlier guard already popped the
  state it depended on. If you can't point to the exact line where the bug
  manifests AND show the fix changes that line's behavior, you haven't verified
  the premise.
- **"This fix was wrong — the absence/omission was deliberate."** Adding the
  obvious-looking missing piece can break things the omission was protecting.
  Example: restoring "missing" `__init__.py` files made a test tree importable
  as a dotted package that shadowed the real plugin, deleting its `register()`
  at import time. The absence was load-bearing.
- **"Overreached / resurrected an approach we'd moved past."** Scope creep that
  supersedes an agreed-on base, or revives a direction the maintainers
  deliberately closed, gets rejected even when the code works. Keep the change
  to the narrow piece that was actually agreed; offer the rest as a focused
  follow-up.

The throughline: **verify the claim AND the intent against the codebase before
writing or merging a fix.** A confirmed reproduction on current `main` plus a
line-level account of where the fix acts beats a plausible-sounding rationale
every time. When in doubt about intent, it is cheaper to ask than to ship a
fix that fights the design.

### The Footprint Ladder (new capability decision)

Each rung adds more permanent surface than the one above. Choose the highest
(least-footprint) rung that correctly solves the problem:

1. **Extend existing code** — the capability is a variation of something that
   already exists. Zero new surface.
2. **CLI command + skill** — manages config/state/infra expressible as shell
   commands. The agent runs `hermes <subcommand>` guided by a skill. Zero
   model-tool footprint. Default choice for subscriptions, scheduled tasks,
   service setup. Examples: `hermes webhook`, `hermes cron`, `hermes tools`.
3. **Service-gated tool (`check_fn`)** — needs structured params/returns AND
   only appears when a prerequisite is configured. Zero footprint otherwise.
   Examples: Home Assistant tools (gated on token), memory-provider tools.
4. **Plugin** — third-party/niche/user-specific capability that doesn't ship in
   core. Lives in `~/.hermes/plugins/` or a pip package, discovered at runtime.
5. **MCP server (in the catalog)** — if the capability genuinely needs to be a
   tool (structured I/O the agent invokes) but isn't core-fundamental, prefer
   building it as an MCP server and adding it to the MCP catalog over growing
   the core toolset. The agent connects to it through the built-in MCP client;
   zero permanent core-schema footprint, and it's reusable by any MCP host.
6. **New core tool** — only when the capability is fundamental, broadly useful
   to nearly every user, and unreachable via terminal + file (or an MCP server).
   Examples of correct core tools: terminal, read_file, web_search,
   browser_navigate.

When 3+ open PRs try to integrate the same *category* of thing (memory
backends, providers, notifiers), don't merge them one at a time — design an
ABC + orchestrator, wrap the existing built-in as the first provider, and turn
the competing PRs into plugins against that interface.

## Development Environment

```bash
# Prefer .venv; fall back to venv if that's what your checkout has.
source .venv/bin/activate   # or: source venv/bin/activate
```

`scripts/run_tests.sh` probes `.venv` first, then `venv`, then
`$HOME/.hermes/hermes-agent/venv` (for worktrees that share a venv with the
main checkout).

## Project Structure

File counts shift constantly — don't treat the tree below as exhaustive.
The canonical source is the filesystem. The notes call out the load-bearing
entry points you'll actually edit.

```
hermes-agent/
├── run_agent.py          # AIAgent class — core conversation loop (~12k LOC)
├── model_tools.py        # Tool orchestration, discover_builtin_tools(), handle_function_call()
├── toolsets.py           # Toolset definitions, _HERMES_CORE_TOOLS list
├── cli.py                # HermesCLI class — interactive CLI orchestrator (~11k LOC)
├── hermes_state.py       # SessionDB — SQLite session store (FTS5 search)
├── hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
├── hermes_logging.py     # setup_logging() — agent.log / errors.log / gateway.log (profile-aware)
├── batch_runner.py       # Parallel batch processing
├── agent/                # Agent internals (provider adapters, memory, caching, compression, etc.)
├── hermes_cli/           # CLI subcommands, setup wizard, plugins loader, skin engine
├── tools/                # Tool implementations — auto-discovered via tools/registry.py
│   └── environments/     # Terminal backends (local, docker, ssh, modal, daytona, singularity)
├── gateway/              # Messaging gateway — run.py + session.py + platforms/
│   ├── platforms/        # Adapter per platform (telegram, discord, slack, whatsapp,
│   │                     #   homeassistant, signal, matrix, mattermost, email, sms,
│   │                     #   dingtalk, wecom, weixin, feishu, qqbot, bluebubbles,
│   │                     #   yuanbao, webhook, api_server, ...). See ADDING_A_PLATFORM.md.
│   └── builtin_hooks/    # Extension point for always-registered gateway hooks (none shipped)
├── plugins/              # Plugin system (see "Plugins" section below)
│   ├── memory/           # Memory-provider plugins (honcho, mem0, supermemory, ...)
│   ├── context_engine/   # Context-engine plugins
│   ├── model-providers/  # Inference backend plugins (openrouter, anthropic, gmi, ...)
│   ├── kanban/           # Multi-agent board dispatcher + worker plugin
│   ├── hermes-achievements/  # Gamified achievement tracking
│   ├── observability/    # Metrics / traces / logs plugin
│   ├── image_gen/        # Image-generation providers
│   └── <others>/         # disk-cleanup, google_meet, platforms, spotify,
│                         #   strike-freedom-cockpit, ...
├── optional-skills/      # Heavier/niche skills shipped but NOT active by default
├── skills/               # Built-in skills bundled with the repo
├── ui-tui/               # Ink (React) terminal UI — `hermes --tui`
│   └── src/              # entry.tsx, app.tsx, gatewayClient.ts + app/components/hooks/lib
├── tui_gateway/          # Python JSON-RPC backend for the TUI
├── acp_adapter/          # ACP server (VS Code / Zed / JetBrains integration)
├── cron/                 # Scheduler — jobs.py, scheduler.py
├── scripts/              # run_tests.sh, release.py, auxiliary scripts
├── website/              # Docusaurus docs site
└── tests/                # Pytest suite (~17k tests across ~900 files as of May 2026)
```

**User config:** `~/.hermes/config.yaml` (settings), `~/.hermes/.env` (API keys only).
**Logs:** `~/.hermes/logs/` — `agent.log` (INFO+), `errors.log` (WARNING+),
`gateway.log` when running the gateway. Profile-aware via `get_hermes_home()`.
Browse with `hermes logs [--follow] [--level ...] [--session ...]`.

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

## File Dependency Chain

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

## Dependency Pinning Policy

All dependencies must have upper bounds to limit supply-chain attack surface.
This policy was established after the litellm compromise (PR #2796, #2810) and
reinforced after the Mini Shai-Hulud worm campaign (May 2026).

| Source type | Treatment | Example |
|---|---|---|
| PyPI package | `>=floor,<next_major` | `"httpx>=0.28.1,<1"` |
| Git URL | Commit SHA | `git+https://...@<40-char-sha>` |
| GitHub Actions | Commit SHA + comment | `uses: actions/checkout@<sha>  # v4` |
| CI-only pip | `==exact` | `pyyaml==6.0.2` |

**When adding a new dependency to `pyproject.toml`:**
1. Pin to `>=current_version,<next_major` for post-1.0 (e.g. `>=1.5.0,<2`).
2. For pre-1.0 packages, use `<0.(current_minor + 2)` (e.g. `>=0.29,<0.32`).
3. Never commit a bare `>=X.Y.Z` without a ceiling — CI and reviewers will reject it.
4. Run `uv lock` to regenerate `uv.lock` with hashes.

Reference: #2810 (bounds pass), #9801 (SHA pinning + audit CI).

## Important Policies

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

**ALWAYS use `scripts/run_tests.sh`** — do not call `pytest` directly. The script enforces
hermetic environment parity with CI (unset credential vars, TZ=UTC, LANG=C.UTF-8,
`-n auto` xdist workers, in-tree subprocess-isolation plugin). Direct `pytest`
on a 16+ core developer machine with API keys set diverges from CI in ways
that have caused multiple "works locally, fails in CI" incidents (and the reverse).

```bash
scripts/run_tests.sh                                  # full suite, CI-parity
scripts/run_tests.sh tests/gateway/                   # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x  # one test
scripts/run_tests.sh -v --tb=long                     # pass-through pytest flags
scripts/run_tests.sh --no-isolate tests/foo/          # disable subprocess isolation (faster, for debugging)
```

### Subprocess-per-test isolation

Every test runs in a freshly-spawned Python subprocess via the in-tree plugin
at `tests/_isolate_plugin.py` (`multiprocessing.get_context("spawn")`). This
means module-level dicts/sets and ContextVars from one test cannot leak into
the next. Per-test overhead is ~0.5–1.0s; xdist parallelism amortizes this.
`isolate_timeout` (in `pyproject.toml`) caps each test at 30s. Pass
`--no-isolate` to disable for debugging. The plugin disables itself in child
processes (sentinel `HERMES_ISOLATE_CHILD=1`), so there's no fork-bomb risk.

`tests/conftest.py` also enforces hermetic behavior (unset keys, temp HOME,
UTC, C.UTF-8) as an autouse fixture so any pytest invocation gets parity.

### Running without the wrapper

If you can't use the wrapper (e.g. inside an IDE), at minimum activate the venv:

```bash
source .venv/bin/activate   # or: source venv/bin/activate
python -m pytest tests/ -q
```

For fast feedback while debugging: `python -m pytest tests/agent/test_foo.py -q --no-isolate`

Always run the full suite before pushing changes.

### Don't write change-detector tests

A test is a **change-detector** if it fails whenever data that is **expected
to change** gets updated — model catalogs, config version numbers,
enumeration counts, hardcoded lists of provider models. These tests add no
behavioral coverage; they just guarantee that routine source updates break
CI and cost engineering time to "fix."

**Do not write:**

```python
# catalog snapshot — breaks every model release
assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]

# config version literal — breaks every schema bump
assert DEFAULT_CONFIG["_config_version"] == 21

# enumeration count — breaks every time a skill/provider is added
assert len(_PROVIDER_MODELS["huggingface"]) == 8
```

**Do write:**

```python
# behavior: does the catalog plumbing work at all?
assert "gemini" in _PROVIDER_MODELS
assert len(_PROVIDER_MODELS["gemini"]) >= 1

# behavior: does migration bump the user's version to current latest?
assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]

# invariant: every model in the catalog has a context-length entry
for m in _PROVIDER_MODELS["huggingface"]:
    assert m.lower() in DEFAULT_CONTEXT_LENGTHS_LOWER
```

The rule: if the test reads like a snapshot of current data, delete it. If
it reads like a contract about how two pieces of data must relate, keep it.

Reviewers should reject new change-detector tests; authors should convert
them into invariants before re-requesting review.

## Module Documentation

The root `AGENTS.md` covers project-wide rules. Module-specific guides live
alongside the code they describe:

| Module | Path | Covers |
|--------|------|--------|
| Agent core | `agent/AGENTS.md` | `AIAgent` class, delegation, curator |
| CLI | `hermes_cli/AGENTS.md` | CLI architecture, config, skin system, profiles |
| Tools | `tools/AGENTS.md` | Adding new tools, toolsets |
| Plugins | `plugins/AGENTS.md` | Plugin system, kanban |
| Skills | `skills/AGENTS.md` | Skill authoring standards, frontmatter |
| TUI | `ui-tui/AGENTS.md` | Ink TUI, JSON-RPC transport, dashboard embedding |
| Desktop | `apps/desktop/AGENTS.md` | Electron desktop app, slash command curation |
| Cron | `cron/AGENTS.md` | Scheduler, job formats, hardening invariants |
