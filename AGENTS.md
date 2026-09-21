# Hermes Agent - Development Guide

Instructions for AI coding assistants and developers working on the hermes-agent codebase.
This root file holds only what applies everywhere. Each area has its own `AGENTS.md` (aim for
~8k chars; `agent/subdirectory_hints.py` delivers up to 32k and truncates head/tail with a warning
past that); see the **routing table** at the end and read the area file before editing in that area.

**Never give up on the right solution.**

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

## Contribution Rubric — What We Want / What We Don't

The project's intent layer. It serves humans aiming a contribution AND the automated triage
sweeper, which may only close on `implemented_on_main`, `cannot_reproduce`, or `incoherent`.
Taste-based "out of scope" closes are a human maintainer's call; the sweeper's job is to
recognize design intent and *avoid wrongly closing a legitimate contribution*.

Read the balance right: Hermes ships a **lot**. Most merges are bug fixes to reported
behavior, and the product surface (platforms, providers, models, desktop/TUI features)
expands aggressively on purpose. The restraint below targets the **core agent + model tool
schema**, the one place where every addition is paid for on every API call. "Smallest
footprint" governs *how a capability is wired into the core*, not whether the product may
grow: expansive at the edges, conservative at the waist.

### What we want

- **Fix real bugs, well.** Reproduce the symptom on current `main`, point to the exact line
  where it manifests, and fix the whole bug class — sibling call paths included.
- **Expand reach at the edges.** New adapters, channels, providers, models, desktop/TUI/
  dashboard features land routinely, including large ones — as long as they integrate with
  the existing setup/config UX (`hermes tools`, `hermes setup`, auto-install) rather than
  bolting on a raw env var.
- **Refactor god-files into clean modules.** Huge mechanical `+N/-N` extraction PRs are
  wanted work. "Every line traces to the request" applies to *feature* PRs; a declared
  refactor's request IS the extraction.
- **Keep the core narrow.** Prefer, in order: extend existing code → CLI command + skill →
  service-gated tool (`check_fn`) → plugin → MCP server in the catalog → new core tool (last
  resort). See the Footprint Ladder.
- **Extend, don't duplicate.** Check whether existing infrastructure covers the use case
  before adding a module/manager/hook. When 3+ open PRs integrate the same *category*
  (memory backends, providers, notifiers), design an ABC + orchestrator, wrap the existing
  built-in as the first provider, and turn the competing PRs into plugins against it.
- **Behavior contracts over snapshots.** Tests assert how two pieces of data relate, never
  freeze a current value (see Testing).
- **E2E validation, not just green unit mocks.** Anything touching resolution chains, config
  propagation, security boundaries, remote backends, or file/network I/O must exercise the
  real path with real imports against a temp `HERMES_HOME` — two of them (A→B→A) when the
  change touches profile scope. Mocks hide integration bugs.
- **Cache-, alternation-, and invariant-safe.** Preserve prompt caching, strict role
  alternation (never two same-role messages in a row; never a synthetic user message injected
  mid-loop), and a system prompt byte-stable for the life of a conversation.
- **Contributor credit preserved.** Salvage external work by cherry-picking (rebase-merge) so
  authorship survives; build on top rather than reimplementing.

### What we don't want (rejected even when well-built)

- **Speculative infrastructure.** Hooks/callbacks/extension points with no concrete consumer.
  Adding a hook is easy; removing one after plugins depend on it is hard. A hook with a real,
  stated use case is NOT speculative even if the consumer ships separately.
- **New `HERMES_*` env vars for non-secret config.** `.env` is for secrets only. Behavioral
  settings (timeouts, thresholds, flags, display prefs) go in `config.yaml`; bridge to an
  internal env var in code if the mechanism needs one. Reject "set X in your .env" docs
  unless X is a credential.
- **A new core tool when terminal + file (or a skill) already do the job.** If the only
  barrier is file visibility on a remote backend, fix the mount, not the toolset.
- **Lazy-reading escape hatches on instructional tools.** No `offset`/`limit` pagination on
  tools that load content the agent must read fully (skills, prompts, playbooks) — models
  read page 1 and skip the rest.
- **"Fixes" that destroy the feature they secure.** Read the original intent
  (`git log -p -S`) before restricting behavior; find a fix that preserves the feature.
- **Outbound telemetry / usage attribution without opt-in gating.** No analytics,
  third-party identifier tagging, or attribution tags until a generic user-facing opt-in
  (config gate + setup prompt + `hermes tools` toggle) exists. Park behind a label.
- **Change-detector tests, cache-breaking mid-conversation, dead code wired in without E2E
  proof, plugins that touch core files.** Plugins work within the ABCs/hooks we provide; if
  one needs more, widen the generic plugin surface, never special-case it in core.
- **Third-party products integrated into the core tree.** Observability backends, vendor
  SaaS connectors, analytics dashboards, and other "someone else's product" plugins do NOT
  land under `plugins/` — every one becomes our burden against a fast-moving core for a
  backend we don't own. Ship as a **standalone plugin repo** (`~/.hermes/plugins/` or pip
  entry point), promoted in the Nous Research Discord `#plugins-skills-and-skins`. This is a
  coupling decision, not a quality bar; such PRs are closed with a pointer to publish.

### Before you call it a bug — verify the premise (and when NOT to close)

The most common reason a well-written PR is closed is a **wrong premise** or treating an
**intentional design as a gap**. These patterns tell a reviewer what to scrutinize and tell
the sweeper when a PR is NOT safe to close (when in doubt, leave it open for a human):

- **"Intentional design, not a gap."** Ask whether the isolation IS the design. Profiles are
  independent islands on purpose: a PR adding live config inheritance from the default
  profile was closed because coupling profiles is exactly what the design prevents (`--clone`
  already covers "start from my default"). Read `git log -p -S "<symbol>"` before assuming
  something is unfinished.
- **"The premise doesn't hold against how X actually works."** Trace the real runtime before
  accepting a rationale. Real closes: a rate-limit "re-probe during cooldown" PR (the breaker
  trips only on a *confirmed-empty* bucket, so re-probing hammers a bucket proven empty); a
  usage fix whose new branch **never executes** because an earlier guard already popped the
  state. If you can't point to the exact line where the bug manifests AND show the fix changes
  that line's behavior, the premise is unverified.
- **"The absence was deliberate."** Restoring "missing" `__init__.py` files made a test tree
  importable as a dotted package that shadowed the real plugin and deleted its `register()`
  at import time. The omission was load-bearing.
- **"Overreached / resurrected an approach we moved past."** Scope creep beyond the agreed
  base, or reviving a direction maintainers closed, is rejected even when it works. Offer the
  rest as a focused follow-up.

Throughline: **verify the claim AND the intent against the codebase before writing or merging
a fix.** A reproduction on current `main` plus a line-level account beats a plausible
rationale. When unsure about intent, asking is cheaper than shipping a fix that fights the
design.

### The Footprint Ladder (new capability decision)

Choose the highest (least-footprint) rung that correctly solves the problem:

1. **Extend existing code** — a variation of something that exists. Zero new surface.
2. **CLI command + skill** — config/state/infra expressible as shell commands; the agent runs
   `hermes <subcommand>` guided by a skill. Default for subscriptions, scheduled tasks,
   service setup (`hermes webhook`, `hermes cron`, `hermes tools`).
3. **Service-gated tool (`check_fn`)** — needs structured params/returns AND only appears when
   a prerequisite is configured (Home Assistant tools, memory-provider tools). This rung gates
   reachability/opt-in process-wide; a capability that varies per SESSION (who is watching) is
   a named toolset folded in by the toolset resolver, not a `check_fn` — see "Surface capability
   is a property of the SESSION" below.
4. **Plugin** — third-party/niche/user-specific; lives in `~/.hermes/plugins/` or a pip
   package, discovered at runtime.
5. **MCP server (in the catalog)** — genuinely a tool but not core-fundamental. Zero permanent
   core-schema footprint, reusable by any MCP host, reached via the built-in MCP client.
6. **New core tool** — only when fundamental, broadly useful to nearly every user, and
   unreachable via terminal + file or an MCP server (terminal, read_file, web_search,
   browser_navigate).

### Surface capability is a property of the SESSION, never of the process env

A tool that works only because of *who is on the other end* (desktop panes, in-app browser,
message reactions, Projects) must resolve availability from the **session's own source**, not
from an env var on the backend. Client and backend are separate machines: the desktop app may
drive a locally spawned backend, one over SSH, one behind URL + token, or Hermes Cloud, and
only the first two carry `HERMES_DESKTOP=1`. An env-keyed gate is a silent no-op on the other
topologies — the tool is stripped from the schema while the platform hint tells the model it
is "inside the Hermes desktop app". The pattern:

- **The toolset is the surface gate.** Keep such tools off `_HERMES_CORE_TOOLS` and in a named
  toolset (`desktop_ui`, `project`); the GUI gateway's `_load_enabled_toolsets(platform)`
  folds it in when the session's platform says GUI. One resolver, every topology.
- **`check_fn` answers reachability or opt-in, not surface.** "Is the bridge wired?" — fine.
  "Was I spawned by Electron?" — not. `check_fn` results are TTL-cached process-wide
  (`tools/registry.py`); a per-session answer does not belong there.
- **Ask which identity you mean.** `HERMES_DESKTOP=1` legitimately means "this backend was
  spawned by the app" (cron ticker, web-dist handling). It does NOT mean "a GUI is watching";
  the embedded terminal pane (`hermes --tui` against that backend) is the counterexample.

Test: if the capability still makes sense with the client on another machine, it is
session-scoped. Assert the GUI session gets the tool **with the env var absent**.

## Development Environment

```bash
source .venv/bin/activate   # or: source venv/bin/activate
```
`scripts/run_tests.sh` probes `.venv`, then `venv`, then `$HOME/.hermes/hermes-agent/venv`
(worktrees sharing the main checkout's venv).

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

---

## Adding Configuration

### config.yaml options:
1. Add to `DEFAULT_CONFIG` in `hermes_cli/config.py`
2. Bump `_config_version` (check the current value at the top of `DEFAULT_CONFIG`)
   ONLY if you need to actively migrate/transform existing user config
   (renaming keys, changing structure). Adding a new key to an existing
   section is handled automatically by the deep-merge and does NOT require
   a version bump.

### Top-level `config.yaml` sections (non-exhaustive):

`model`, `agent`, `terminal`, `compression`, `display`, `stt`, `tts`,
`memory`, `security`, `delegation`, `checkpoints`,
`auxiliary`, `curator`, `skills`, `gateway`, `logging`, `cron`, `profiles`,
`plugins`, `honcho`.

`auxiliary` holds per-task overrides for side-LLM work (curator, vision,
embedding, title generation, session_search, etc.) — each task can pin
its own provider/model/base_url/max_tokens/reasoning_effort. See
`agent/auxiliary_client.py::_resolve_auto` for resolution order.

`curator` holds the background skill-maintenance config —
`enabled`, `interval_hours`, `min_idle_hours`, `stale_after_days`,
`archive_after_days`, `backup` (nested).

### .env variables (SECRETS ONLY — API keys, tokens, passwords):
1. Add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py` with metadata:
```python
"NEW_API_KEY": {
    "description": "What it's for",
    "prompt": "Display name",
    "url": "https://...",
    "password": True,
    "category": "tool",  # provider, tool, messaging, setting
},
```

Non-secret settings (timeouts, thresholds, feature flags, paths, display
preferences) belong in `config.yaml`, not `.env`. If internal code needs an
env var mirror for backward compatibility, bridge it from `config.yaml` to
the env var in code (see `gateway_timeout`, `terminal.cwd` → `TERMINAL_CWD`).

### Config loaders (three paths — know which one you're in):

| Loader | Used by | Location |
|--------|---------|----------|
| `load_cli_config()` | CLI mode | `cli.py` — merges CLI-specific defaults + user YAML |
| `load_config()` | `hermes tools`, `hermes setup`, most CLI subcommands | `hermes_cli/config.py` — merges `DEFAULT_CONFIG` + user YAML |
| Direct YAML load | Gateway runtime | `gateway/run.py` + `gateway/config.py` — reads user YAML raw |

If you add a new key and the CLI sees it but the gateway doesn't (or vice
versa), you're on the wrong loader. Check `DEFAULT_CONFIG` coverage.

### Working directory:
- **CLI** — uses the process's current directory (`os.getcwd()`).
- **Messaging** — uses `terminal.cwd` from `config.yaml`. The gateway bridges this
  to the `TERMINAL_CWD` env var for child tools. **`MESSAGING_CWD` has been
  removed** — the config loader prints a deprecation warning if it's set in
  `.env`. Same for `TERMINAL_CWD` in `.env`; the canonical setting is
  `terminal.cwd` in `config.yaml`.

---

## Skin/Theme System

The skin engine (`hermes_cli/skin_engine.py`) provides data-driven CLI visual customization. Skins are **pure data** — no code changes needed to add a new skin.

### Architecture

```
hermes_cli/skin_engine.py    # SkinConfig dataclass, built-in skins, YAML loader
~/.hermes/skins/*.yaml       # User-installed custom skins (drop-in)
```

- `init_skin_from_config()` — called at CLI startup, reads `display.skin` from config
- `get_active_skin()` — returns cached `SkinConfig` for the current skin
- `set_active_skin(name)` — switches skin at runtime (used by `/skin` command)
- `load_skin(name)` — loads from user skins first, then built-ins, then falls back to default
- Missing skin values inherit from the `default` skin automatically

### What skins customize

| Element | Skin Key | Used By |
|---------|----------|---------|
| Banner panel border | `colors.banner_border` | `banner.py` |
| Banner panel title | `colors.banner_title` | `banner.py` |
| Banner section headers | `colors.banner_accent` | `banner.py` |
| Banner dim text | `colors.banner_dim` | `banner.py` |
| Banner body text | `colors.banner_text` | `banner.py` |
| Response box border | `colors.response_border` | `cli.py` |
| Spinner faces (waiting) | `spinner.waiting_faces` | `display.py` |
| Spinner faces (thinking) | `spinner.thinking_faces` | `display.py` |
| Spinner verbs | `spinner.thinking_verbs` | `display.py` |
| Spinner wings (optional) | `spinner.wings` | `display.py` |
| Tool output prefix | `tool_prefix` | `display.py` |
| Per-tool emojis | `tool_emojis` | `display.py` → `get_tool_emoji()` |
| Agent name | `branding.agent_name` | `banner.py`, `cli.py` |
| Welcome message | `branding.welcome` | `cli.py` |
| Response box label | `branding.response_label` | `cli.py` |
| Prompt symbol | `branding.prompt_symbol` | `cli.py` |

### Built-in skins

- `default` — Classic Hermes gold/kawaii (the current look)
- `ares` — Crimson/bronze war-god theme with custom spinner wings
- `mono` — Clean grayscale monochrome
- `slate` — Cool blue developer-focused theme

### Adding a built-in skin

Add to `_BUILTIN_SKINS` dict in `hermes_cli/skin_engine.py`:

```python
"mytheme": {
    "name": "mytheme",
    "description": "Short description",
    "colors": { ... },
    "spinner": { ... },
    "branding": { ... },
    "tool_prefix": "┊",
},
```

### User skins (YAML)

Users create `~/.hermes/skins/<name>.yaml`:

```yaml
name: cyberpunk
description: Neon-soaked terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

Activate with `/skin cyberpunk` or `display.skin: cyberpunk` in config.yaml.

---

## Plugins

Hermes has two plugin surfaces. Both live under `plugins/` in the repo so
repo-shipped plugins can be discovered alongside user-installed ones in
`~/.hermes/plugins/` and pip-installed entry points.

### General plugins (`hermes_cli/plugins.py` + `plugins/<name>/`)

`PluginManager` discovers plugins from `~/.hermes/plugins/`, `./.hermes/plugins/`,
and pip entry points. Each plugin exposes a `register(ctx)` function that
can:

- Register Python-callback lifecycle hooks:
  `pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`,
  `on_session_start`, `on_session_end`
- Register new tools via `ctx.register_tool(...)`
- Register CLI subcommands via `ctx.register_cli_command(...)` — the
  plugin's argparse tree is wired into `hermes` at startup so
  `hermes <pluginname> <subcmd>` works with no change to `main.py`

Hooks are invoked from `model_tools.py` (pre/post tool) and `run_agent.py`
(lifecycle). **Discovery timing pitfall:** `discover_plugins()` only runs
as a side effect of importing `model_tools.py`. Code paths that read plugin
state without importing `model_tools.py` first must call `discover_plugins()`
explicitly (it's idempotent).

#### Native plugin compatibility policy

The canonical contract and deprecation policy live in
`website/docs/developer-guide/plugins/index.md#native-plugin-compatibility-contract`.
Compatibility is enforced as a behavior contract, not through a monolithic
`PLUGIN_API_VERSION`, a manifest-wide native `api:` match, or version literals
on unrelated payloads. Keep documented plugin surfaces additive:

- add hook payload data as keyword fields; signature-inspect callbacks so old
  narrow signatures receive only fields they declare, while `**kwargs`
  callbacks receive the complete payload;
- do not remove or rename `PluginContext` methods; make new parameters optional
  with defaults and keyword-only where possible;
- ignore unknown native manifest fields;
- give new provider methods default implementations, and signature-inspect
  optional callback kwargs rather than forwarding them unconditionally;
- use a local schema version only for a capability with a wire or persisted
  contract, and preserve old state/config/session replay or ship a migration.

Deprecations require a once-per-process warning, a documented replacement and
migration note, and at least two subsequent minor releases before removal.
Compatibility tests must load frozen plugins through the real discovery path
and assert outcomes. Do not replace these with exact registry/catalog counts,
source-reading tests, or assertions that a global version literal changed.

### Memory-provider plugins (`plugins/memory/<name>/`)

Separate discovery system for pluggable memory backends. Current built-in
providers include **honcho, mem0, supermemory, byterover, hindsight,
holographic, openviking, retaindb**.

Discovery covers the same four sources as the general `PluginManager` —
bundled, `$HERMES_HOME/plugins/`, `./.hermes/plugins/` (opt-in via
`HERMES_ENABLE_PROJECT_PLUGINS`), and `hermes_agent.memory_providers` entry
points — but with **bundled-first** precedence, the reverse of the general
system's later-wins order: a memory provider is activated by name, so a
dropped-in directory must not be able to shadow a shipped one. Discovery
enumerates without importing; nothing runs until `memory.provider` names it.

Each provider implements the `MemoryProvider` ABC (see `agent/memory_provider.py`)
and is orchestrated by `agent/memory_manager.py`. Lifecycle hooks include
`sync_turn(turn_messages)`, `prefetch(query)`, `shutdown()`, and optional
`post_setup(hermes_home, config)` for setup-wizard integration.

**CLI commands via `plugins/memory/<name>/cli.py`:** if a memory plugin
defines `register_cli(subparser)`, `discover_plugin_cli_commands()` finds
it at argparse setup time and wires it into `hermes <plugin>`. The
framework only exposes CLI commands for the **currently active** memory
provider (read from `memory.provider` in config.yaml), so disabled
providers don't clutter `hermes --help`.

**Rule (Teknium, May 2026):** plugins MUST NOT modify core files
(`run_agent.py`, `cli.py`, `gateway/run.py`, `hermes_cli/main.py`, etc.).
If a plugin needs a capability the framework doesn't expose, expand the
generic plugin surface (new hook, new ctx method) — never hardcode
plugin-specific logic into core. PR #5295 removed 95 lines of hardcoded
honcho argparse from `main.py` for exactly this reason.

**No new in-tree memory providers (policy, May 2026):** the set of
built-in memory providers under `plugins/memory/` is closed. New memory
backends must ship as **standalone plugin repos** that users install
into `~/.hermes/plugins/` (or via pip entry points) — they implement
the same `MemoryProvider` ABC, register through the same discovery
path, and integrate via `hermes memory setup` / `post_setup()` without
landing in this tree. PRs that add a new directory under
`plugins/memory/` will be closed with a pointer to publish the
provider as its own repo. Existing in-tree providers stay; bug fixes
to them are welcome.

**No new third-party-product plugins in-tree (policy, June 2026):** the
same rule applies beyond memory providers. Plugins that integrate
someone else's product or project — observability/metrics backends,
vendor SaaS connectors, analytics dashboards, paid-service tie-ins —
must ship as **standalone plugin repos** that users install into
`~/.hermes/plugins/` (or via pip entry points). They register through
the existing plugin discovery path and use the ABCs/hooks/ctx surface
we expose; nothing special is needed in core. The reason is
maintenance load: every product we absorb into the tree becomes our
burden to keep working against a fast-moving core, for a backend we
don't own. Promote standalone plugins in the Nous Research Discord
(`#plugins-skills-and-skins`). PRs that add such a directory under
`plugins/` are closed with a pointer to publish it as its own repo —
this is a coupling decision, not a quality judgment. (The
`observability/`, `kanban/`, `disk-cleanup/`, etc. directories already
in the tree are existing precedent, not an invitation to add more
third-party-product plugins alongside them.)

### Model-provider plugins (`plugins/model-providers/<name>/`)

Every inference backend (openrouter, anthropic, gmi, deepseek, nvidia, …)
ships as a plugin here. Each plugin's `__init__.py` calls
`providers.register_provider(ProviderProfile(...))` at module load.
`providers/__init__.py._discover_providers()` is a **lazy, separate
discovery system** — scanned on first `get_provider_profile()` or
`list_providers()` call, NOT by the general PluginManager.

Scan order:
1. Bundled: `<repo>/plugins/model-providers/<name>/`
2. User: `$HERMES_HOME/plugins/model-providers/<name>/`
3. Legacy: `<repo>/providers/<name>.py` (back-compat)

User plugins of the same name override bundled ones — `register_provider()`
is last-writer-wins. This lets third parties swap out any built-in
profile without a repo patch.

## Commits, Merges, PRs

- **Squash merges from stale branches silently revert recent fixes.** Before squash-merging,
  bring the branch to `main` (`git fetch origin main && git reset --hard origin/main`, re-apply
  the PR's commits). Verify with `git diff HEAD~1..HEAD` after merging — unexpected deletions
  are a red flag.
- Salvage by cherry-pick so contributor authorship survives (see rubric).
- Tests per fix: 1–2 INVARIANT tests (behaviour contract, proven red on base), never
  change-detectors; ≤ 2 tests is the salvage bar too. Reject/rewrite in salvaged diffs:
  appendages to facades, new god helpers, compat aliases, wrappers.

## Testing (applies everywhere)

**ALWAYS use `scripts/run_tests.sh`**, never bare `pytest`. It enforces CI parity: credential
vars unset, `TZ=UTC`, `LANG=C.UTF-8`, `HERMES_HOME` → temp dir, and per-file subprocess
isolation via `scripts/run_tests_parallel.py` (no xdist; workers scale with CPU count) so
module-level dicts/ContextVars cannot leak between files. Direct `pytest` on a big machine
with API keys set has caused repeated "works locally, fails in CI" incidents (and the reverse).

`plugins/context_engine/`, `plugins/image_gen/`, etc. follow the same
pattern (ABC + orchestrator + per-plugin directory). Context engines
plug into `agent/context_engine.py`; image-gen providers into
`agent/image_gen_provider.py`. Reference / docs-companion plugins
(`example-dashboard`, `strike-freedom-cockpit`, `plugin-llm-example`,
`plugin-llm-async-example`) live in the
[`hermes-example-plugins`](https://github.com/NousResearch/hermes-example-plugins)
companion repo, not in this tree.

### Bot Mode (`apps/desktop/src/plugins/hermes-bots/`)

The desktop "Bots" experience ships bundled in-tree. Each bot is a Hermes
agent **profile** with a persistent identity. Its design rests on one settled
invariant that has been regressed repeatedly, cost users real conversation
history each time, and is not open for re-litigation in a routine PR:

**One bot = ONE canonical forever-chat, identified by NAME.** The chat's one
and only identity is **(profile, session titled exactly "Bot Chat")** — the
state DB's UNIQUE(title) index makes that pair an exact registry of at most
one row. The full lifecycle when a bot row is clicked:

1. **Resolve the registry, every time.** Look up the profile's `Bot Chat`
   session by exact title via `session.list {title, include_hidden: true}`
   (indexed, window-free; hidden rows resolve because canonical chats are
   always hidden; compression lineages resolve to the live tip). Row exists →
   open it. That is the entire happy path.
2. **No row → create it,** titled `Bot Chat`, born hidden, kicked off with
   the bot's intro. Creation adopts-before-minting: it re-runs the registry
   lookup first, so a concurrent or pre-existing row is opened, never forked.
   (`set_session_title` silently drops conflicting titles — returns 0 rows —
   which is how the 2026-08 infinite fork loop started; adopt-before-mint is
   what kills it.)

**There is NO session-id pin.** The previous design stored a pointer in
`ui_meta['hermes-bots'].chat` and verified it per click; five hardening
waves (#88690, #90732, #90751, the #91791 revert, #92042) each guarded a new
way that pointer dangled or got stolen — rows[0] steals, `last_session`
adoptions, transient clears, drifted-title welds (a pin re-anchored onto a
cron session passed every guard). Name-as-identity removes the failure class:
a name cannot dangle, and a corrupted historical pointer simply never gets
read. Legacy `chat` keys in ui_meta are ignored and dropped from merges.

Why recency must never win (the #91791 → #92042 lesson): canonical Bot
Chats are **unconditionally hidden** from the Sessions sidebar, so the bot
row is the ONLY door to the forever-chat. A "newest visible session wins"
preference doesn't re-order two equivalent entry points — it walls the
entire relationship off behind a row that previews one session and opens
another, and any stray draft that catches a prompt captures the row.
Side-chats started via "New chat with this agent" are not plumbing-titled,
stay visible in the Sessions sidebar, and are reachable there; they are
never the bot row's target.

Corollaries for reviewers:

- There is no per-bot session browser, by explicit design (removed in
  #90732). Do not add one back.
- Reject any PR that reintroduces a stored session-id pointer as canonical
  identity — including "as a fallback tier" or "for verification". The
  registry lookup is the whole contract; pointers are how every prior
  incident started.
- Reject any PR that consults recency, visibility, or "where the user left
  off" for the bot row's target — reports that motivate such a change are
  almost always about side-chats, and the fix belongs in the Sessions
  sidebar (hide-sweep false positives), not in the bot row's target.
- The gateway reports the registry row per profile as `canonical_session`
  on `profiles.list` (resolved server-side by title); roster preview,
  activity signals, and the `/new`→`/compact` guard all read it, so preview
  identity and click identity are the same row by construction.

Regression tests encoding this contract:
`tests/canonical-chat-registry.test.mjs` (includes a tripwire asserting the
open path never reads or writes a stored pointer),
`tests/canonical-chat-creation.test.mjs`, `tests/hide-bot-chats.test.mjs`,
and `tests/tui_gateway/test_profiles_list_canonical_session.py`.

---

## Skills

Two parallel surfaces:

- **`skills/`** — built-in skills shipped and loadable by default.
  Organized by category directories (e.g. `skills/github/`, `skills/mlops/`).
- **`optional-skills/`** — heavier or niche skills shipped with the repo but
  NOT active by default. Installed explicitly via
  `hermes skills install official/<category>/<skill>`. Adapter lives in
  `tools/skills_hub.py` (`OptionalSkillSource`). Categories include
  `autonomous-ai-agents`, `blockchain`, `communication`, `creative`,
  `devops`, `email`, `health`, `mcp`, `migration`, `mlops`, `productivity`,
  `research`, `security`, `web-development`.

When reviewing skill PRs, check which directory they target — heavy-dep or
niche skills belong in `optional-skills/`.

### SKILL.md frontmatter

Standard fields: `name`, `description`, `version`, `author`, `license`,
`platforms` (OS-gating list: `[macos]`, `[linux, macos]`, ...),
`metadata.hermes.tags`, `metadata.hermes.category`,
`metadata.hermes.related_skills`, `metadata.hermes.config` (config.yaml
settings the skill needs — stored under `skills.config.<key>`, prompted
during setup, injected at load time).

Top-level `tags:` and `category:` are also accepted and mirrored from
`metadata.hermes.*` by the loader.

### Skill authoring standards (HARDLINE)

Every new or modernized skill — bundled, optional, or contributed —
must meet these standards before merge. Reviewers reject PRs that
violate them.

1. **`description` ≤ 60 characters, one sentence, ends with a period.**
   Long descriptions bloat skill listings and dilute the model's
   attention when many skills are loaded. State the capability, not
   the implementation. No marketing words ("powerful",
   "comprehensive", "seamless", "advanced"). Don't repeat the skill
   name. Verify with:
   ```python
   import re, pathlib
   m = re.search(r'^description: (.*)$',
                 pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
                 re.MULTILINE)
   assert len(m.group(1)) <= 60, len(m.group(1))
   ```

2. **Tools referenced in SKILL.md prose must be native Hermes tools or
   MCP servers the skill explicitly expects.** When the skill needs a
   capability, point at the proper tool by name in backticks
   (`` `terminal` ``, `` `web_extract` ``, `` `read_file` ``,
   `` `patch` ``, `` `search_files` ``, `` `vision_analyze` ``,
   `` `browser_navigate` ``, `` `delegate_task` ``, etc.). Do NOT
   name shell utilities the agent already has wrapped — `grep` →
   `search_files`, `cat`/`head`/`tail` → `read_file`, `sed`/`awk` →
   `patch`, `find`/`ls` → `search_files target='files'`. If the skill
   depends on an MCP server, name the MCP server and document the
   expected setup in `## Prerequisites`. Anything else (third-party
   CLIs, shell pipelines, etc.) is fair game inside script files but
   should not be the headline interaction surface in the prose.

3. **`platforms:` gating audited against actual script imports.**
   Skills that use POSIX-only primitives (`fcntl`, `termios`,
   `os.setsid`, `os.kill(pid, 0)` for liveness, `/proc`, `/tmp`
   hardcoded, `signal.SIGKILL`, bash heredocs, `osascript`, `apt`,
   `systemctl`) must declare their supported platforms. Default
   posture: try to fix it cross-platform first — `tempfile.gettempdir`,
   `pathlib.Path`, `psutil.pid_exists`, Python-level filtering instead
   of `grep`. Gate to a narrower set only when the dependency is
   genuinely platform-bound.

4. **`author` credits the human contributor first.** For external
   contributions, the contributor's real name + GitHub handle goes
   first; "Hermes Agent" is the secondary collaborator. If the
   contributor's commit shows "Hermes Agent" as author (because they
   used Hermes to draft the skill), replace it with their actual name
   — credit the human, not the tool.

5. **SKILL.md body uses the modern section order.** `# <Skill> Skill`
   title, 2-3 sentence intro stating what it does and doesn't do,
   `## When to Use`, `## Prerequisites`, `## How to Run`,
   `## Quick Reference`, `## Procedure`, `## Pitfalls`,
   `## Verification`. Target ~200 lines for a complex skill,
   ~100 lines for a simple one. Cut redundant intro fluff, marketing
   prose, and re-explanations of env vars already in
   `## Prerequisites`.

6. **Scripts go in `scripts/`, references in `references/`,
   templates in `templates/`.** Don't expect the model to inline-write
   parsers, XML walkers, or non-trivial logic every call — ship a
   helper script. Reference it from SKILL.md by path relative to the
   skill directory.

7. **Tests live at `tests/skills/test_<skill>_skill.py`** and use only
   stdlib + pytest + `unittest.mock`. No live network calls. Run via
   `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`.

8. **`.env.example` additions are isolated to a clearly delimited
   block.** Don't touch the surrounding file — contributor-supplied
   `.env.example` versions are usually stale and edits outside the
   skill's own block must be dropped during salvage.

The full salvage / modernization checklist for external skill PRs
lives in the `hermes-agent-dev` skill at
`references/new-skill-pr-salvage.md` — load it before polishing
contributor skill PRs.

---

## Toolsets

All toolsets are defined in `toolsets.py` as a single `TOOLSETS` dict.
Each platform's adapter picks a base toolset (e.g. Telegram uses
`"messaging"`); `_HERMES_CORE_TOOLS` is the default bundle most
platforms inherit from.

Current toolset keys: `browser`, `clarify`, `code_execution`, `cronjob`,
`debugging`, `delegation`, `discord`, `discord_admin`, `feishu_doc`,
`feishu_drive`, `file`, `homeassistant`, `image_gen`, `kanban`, `memory`,
`messaging`, `moa`, `rl`, `safe`, `search`, `session_search`, `skills`,
`spotify`, `terminal`, `todo`, `tts`, `video`, `vision`, `web`, `yuanbao`.

Enable/disable per platform via `hermes tools` (the curses UI) or the
`tools.<platform>.enabled` / `tools.<platform>.disabled` lists in
`config.yaml`.

---

## Delegation (`delegate_task`)

`tools/delegate_tool.py` spawns a subagent with an isolated
context + terminal session. By default the parent waits for the
child's summary before continuing its own loop. With `background=true`,
Hermes returns a delegation id immediately and the result re-enters the
conversation later through the async-delegation completion queue.

Two shapes:

- **Single:** pass `goal` (+ optional `context`).
- **Batch (parallel):** pass `tasks: [...]` — each gets its own subagent
  running concurrently. Concurrency is capped by
  `delegation.max_concurrent_children` (default 3).

Roles:

- `role="leaf"` (default) — focused worker. Cannot call `delegate_task`,
  `clarify`, `memory`, `send_message`, `cronjob`. Retains `execute_code`
  (programmatic tool calling).
- `role="orchestrator"` — retains `delegate_task` so it can spawn its
  own workers. Gated by `delegation.orchestrator_enabled` (default true)
  and bounded by `delegation.max_spawn_depth` (default 2).

Key config knobs (under `delegation:` in `config.yaml`):
`max_concurrent_children`, `max_spawn_depth`, `child_timeout_seconds`,
`orchestrator_enabled`, `subagent_auto_approve`, `inherit_mcp_toolsets`,
`max_iterations`.

Durability rule: background `delegate_task` is detached from the current
turn but still process-local. For work that must survive process restart, use
`cronjob` or `terminal(background=True, notify_on_complete=True)` instead.

---

## Curator (skill lifecycle)

Background skill-maintenance system that tracks usage on agent-created
skills and auto-archives stale ones. Users never lose skills; archives
go to `~/.hermes/skills/.archive/` and are restorable.

- **Core:** `agent/curator.py` (review loop, auto-transitions, LLM review
  prompt) + `agent/curator_backup.py` (pre-run tar.gz snapshots).
- **CLI:** `hermes_cli/curator.py` wires `hermes curator <verb>` where
  verbs are: `status`, `run`, `pause`, `resume`, `pin`, `unpin`,
  `archive`, `restore`, `prune`, `backup`, `rollback`.
- **Telemetry:** `tools/skill_usage.py` owns the sidecar
  `~/.hermes/skills/.usage.json` — per-skill `use_count`, `view_count`,
  `patch_count`, `last_activity_at`, `state` (active / stale /
  archived), `pinned`.

Invariants:
- Curator only touches skills with `created_by: "agent"` provenance —
  bundled + hub-installed skills are off-limits.
- Never deletes; max destructive action is archive.
- Pinned skills are exempt from every auto-transition and from the
  LLM review pass.
- `skill_manage(action="delete")` refuses pinned skills; patch/edit/
  write_file/remove_file go through so the agent can keep improving
  pinned skills.

Config section (`curator:` in `config.yaml`):
`enabled`, `interval_hours`, `min_idle_hours`, `stale_after_days`,
`archive_after_days`, `backup.*`.

Full user-facing docs: `website/docs/user-guide/features/curator.md`.

---

## Cron (scheduled jobs)

`cron/jobs.py` (job store) + `cron/scheduler.py` (tick loop). Agents
schedule jobs via the `cronjob` tool; users via `hermes cron <verb>`
(`list`, `add`, `edit`, `pause`, `resume`, `run`, `remove`) or the
`/cron` slash command.

Supported schedule formats:
- Duration: `"30m"`, `"2h"`, `"1d"`
- "every" phrase: `"every 2h"`, `"every monday 9am"`
- 5-field cron expression: `"0 9 * * *"`
- ISO timestamp (one-shot): `"2026-06-01T09:00:00Z"`

Per-job fields include `skills` (load specific skills), `model` /
`provider` overrides, `script` (pre-run data-collection script whose
stdout is injected into the prompt; `no_agent=True` turns the script
into the entire job), `context_from` (chain job A's last output into
job B's prompt), `workdir` (run in a specific directory with its
`AGENTS.md`/`CLAUDE.md` loaded), and multi-platform delivery.

Hardening invariants:
- **3-minute hard interrupt** on cron sessions — runaway agent loops
  cannot monopolize the scheduler.
- Catchup window: half the job's period, clamped to 120s–2h.
- Grace window: 120s for one-shot jobs whose fire time was missed.
- File lock at `~/.hermes/cron/.tick.lock` prevents duplicate ticks
  across processes.
- Cron sessions pass `skip_memory=True` by default; memory providers
  intentionally do not run during cron.

Cron deliveries are **not** mirrored into the target gateway session —
they land in their own cron session with a header/footer frame so the
main conversation's message-role alternation stays intact.

---

## Kanban (multi-agent work queue)

Durable SQLite-backed board that lets multiple profiles / workers
collaborate on shared tasks. Users drive it via `hermes kanban <verb>`;
workers spawned by the dispatcher drive it via a dedicated `kanban_*`
toolset so their schema footprint is zero when they're not inside a
kanban task.

- **CLI:** `hermes_cli/kanban.py` wires `hermes kanban` with verbs
  `init`, `create`, `list` (alias `ls`), `show`, `assign`, `link`,
  `unlink`, `comment`, `attach`, `attachments`, `attach-rm`, `complete`,
  `request-review`, `request-changes`, `reopen-review`, `block`, `unblock`, `archive`,
  `tail`, plus less-commonly-used `watch`, `stats`, `runs`, `log`,
  `assignees`, `heartbeat`, `notify-*`, `dispatch`, `daemon`, `gc`.
- **Worker/orchestrator toolset:** `tools/kanban_tools.py` exposes
  `kanban_show`, `kanban_complete`, `kanban_request_review`,
  `kanban_request_changes`, `kanban_block`,
  `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`,
  `kanban_attach`, `kanban_attach_url`, `kanban_attachments`; profiles that
  explicitly enable the `kanban` toolset outside a dispatcher-spawned
  task also get `kanban_list` and `kanban_unblock` for board routing.
- **Dispatcher:** long-lived loop that (default every 60s) reclaims
  stale claims, promotes ready tasks, atomically claims, and spawns
  assigned profiles. Runs **inside the gateway** by default via
  `kanban.dispatch_in_gateway: true`.
- **Plugin assets:** `plugins/kanban/dashboard/` (web UI) +
  `plugins/kanban/systemd/` (`hermes-kanban-dispatcher.service` for
  standalone dispatcher deployment).

Isolation model:
- **Board** is the hard boundary — workers are spawned with
  `HERMES_KANBAN_BOARD` pinned in their env so they can't see other
  boards.
- **Tenant** is a soft namespace *within* a board — one specialist
  fleet can serve multiple businesses with workspace-path + memory-key
  isolation.
- After `kanban.failure_limit` consecutive non-success attempts on the
  same task (default: 2), the dispatcher auto-blocks it to prevent spin
  loops.

Full user-facing docs: `website/docs/user-guide/features/kanban.md`.

---

## Update Pipeline (`hermes update`)

The updater is transactional in shape (fleet-update campaign, #91277 —
Aug 2026). Every stage exists because its absence was a real field
failure; PRs that weaken a stage need to answer for the failure class it
guards:

```
plan → snapshot → apply → restart-per-kind → verify → report
```

- **Plan** (`hermes_cli/update_inventory.py`, `hermes update --plan`):
  read-only inventory — install kind, all profiles, every live gateway
  with supervisor + running code version. Deployment kinds are
  first-class: `git` updates in place; `docker`/`nix`/`apt` are NOT
  in-place-updatable and the updater reports the correct external
  command instead of fighting the deployment model.
- **Snapshot** (`hermes_cli/backup.py`): pre-update quick snapshot for
  EVERY profile (the code swap + fleet restart touch all of them), each
  into its own `state-snapshots/`, identical file set + 1 GiB per-file
  cap + keep=1. **Never add a partial/tiered snapshot set** — mixed
  coverage creates torn-restore states across schema generations. Quick
  snapshots are FILE-LOSS RECOVERY (the per-profile cron-jobs safety
  net restores from them), NOT code-rollback insurance; `--backup` full
  mode owns rollback.
- **Apply**: git pull, or the Windows ZIP fallback — which fires ONLY
  when git itself failed (`_should_zip_fallback_on_update_error`,
  argv-classified; a dependency-install failure must never trigger a
  tree-clobbering re-download), REFUSES a dirty working tree
  (`-uall`, plus a pre-swap TOCTOU re-check), and grafts the live
  `apps/desktop/release/` into the staged swap (the GitHub source ZIP
  has no built desktop app; without the graft the swap deletes it).
- **Restart-per-kind**: systemd and launchd restarts are FLEET-WIDE
  (every `hermes-gateway*` unit / `ai.hermes.gateway*` LaunchAgent),
  drain-first (SIGUSR1) with per-unit/per-label failure isolation.
  Restarting only the invoking profile's service leaves siblings on
  stale `sys.modules` until they crash — the largest dupe-PR cluster in
  the repo's history came from that bug.
- **Verify**: gateways stamp their running `code_sha`/`code_version`
  into `gateway_state.json` on every runtime-status write
  (`gateway/status.py`); after the restart phase the updater compares
  each live gateway against the fresh checkout and prints a fleet
  version matrix. A provably-stale gateway fails the update (exit 1) —
  automation must never treat a mixed-version fleet as healthy.
- **Report**: every run writes a machine-readable receipt to
  `~/.hermes/logs/update_receipts/` (`latest.json` pointer; steps,
  skips WITH reasons, restart outcome, plan, fleet snapshot).
  Finalization is owned by the `cmd_update` command boundary — early
  `sys.exit` paths (preflight refusals, fetch failures) still persist
  a receipt with the real exit code. A begun-but-unwritten receipt is
  a bug: the refused/failed runs are the ones receipts exist for.

Architecture direction: process-scan-based coordination between the
updater, serve/dashboard, and the gateway is being replaced by a
gateway-owned control socket (#92091). Do not add new scan heuristics
without checking that design; scans are the fallback layer.

### Gateway lifecycle vs. the Desktop app

`hermes serve` (control plane, desktop-spawned child) dies with the app
— by design. The messaging gateway (`gateway run`) SURVIVES the app: the
serve backend's `/api/gateway/*` endpoints spawn it detached
(`_spawn_hermes_action` — `start_new_session` / `DETACHED_PROCESS`), so
`before-quit`'s backend SIGTERM never reaches it. Bots keep running
when the user closes the app. The known breach of this contract is the
Windows shim-unlock teardown (`taskkill /T /F` on venv-shim holders,
#85265) — it exists to let updates proceed, and its replacement is
#92091's `pause-for-update`. Do not "fix" gateway-dies-with-app reports
by re-parenting the gateway under the backend, and do not "fix" update
locks by widening the tree-kill.

---

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

- `concise` — one-line status message on completion; failures append a short output tail (default)
- `all` — running-output updates + final raw-output message
- `result` — only the final raw-output completion message
- `error` — only the final raw-output message when exit code != 0
- `off` — no watcher messages at all

---

## Profiles: Multi-Instance Support

Hermes supports **profiles** — multiple fully isolated instances, each with its own
`HERMES_HOME` directory (config, API keys, memory, sessions, skills, gateway, etc.).

The core mechanism: `_apply_profile_override()` in `hermes_cli/main.py` sets
`HERMES_HOME` before any module imports. All `get_hermes_home()` references
automatically scope to the active profile.

### Rules for profile-safe code

1. **Use `get_hermes_home()` for all HERMES_HOME paths.** Import from `hermes_constants`.
   NEVER hardcode `~/.hermes` or `Path.home() / ".hermes"` in code that reads/writes state.
   ```python
   # GOOD
   from hermes_constants import get_hermes_home
   config_path = get_hermes_home() / "config.yaml"

   # BAD — breaks profiles
   config_path = Path.home() / ".hermes" / "config.yaml"
   ```

2. **Use `display_hermes_home()` for user-facing messages.** Import from `hermes_constants`.
   This returns `~/.hermes` for default or `~/.hermes/profiles/<name>` for profiles.
   ```python
   # GOOD
   from hermes_constants import display_hermes_home
   print(f"Config saved to {display_hermes_home()}/config.yaml")

   # BAD — shows wrong path for profiles
   print("Config saved to ~/.hermes/config.yaml")
   ```

3. **Module-level constants are fine** — they cache `get_hermes_home()` at import time,
   which is AFTER `_apply_profile_override()` sets the env var. Just use `get_hermes_home()`,
   not `Path.home() / ".hermes"`.

4. **Tests that mock `Path.home()` must also set `HERMES_HOME`** — since code now uses
   `get_hermes_home()` (reads env var), not `Path.home() / ".hermes"`:
   ```python
   with patch.object(Path, "home", return_value=tmp_path), \
        patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
       ...
   ```

5. **Gateway platform adapters should use token locks** — if the adapter connects with
   a unique credential (bot token, API key), call `acquire_scoped_lock()` from
   `gateway.status` in the `connect()`/`start()` method and `release_scoped_lock()` in
   `disconnect()`/`stop()`. This prevents two profiles from using the same credential.
   See `plugins/platforms/irc/adapter.py` for the canonical pattern.

6. **Profile operations are HOME-anchored, not HERMES_HOME-anchored** — `_get_profiles_root()`
   returns `Path.home() / ".hermes" / "profiles"`, NOT `get_hermes_home() / "profiles"`.
   This is intentional — it lets `hermes -p coder profile list` see all profiles regardless
   of which one is active.

7. **Multiplex profile-scoped env reads MUST fail closed — never borrow from `os.environ`**
   (`agent/secret_scope.py` contract; #72348, #86905). Under `gateway.multiplex_profiles`,
   `os.environ` holds the **default profile's** values; a secondary profile's `.env` lives
   only in its secret scope (installed per-turn by `_profile_runtime_scope`). Any
   profile-level env config — credentials (`app_secret`, tokens) AND authorization
   (`FEISHU_ALLOWED_USERS`, `{PLATFORM}_ALLOW_ALL_USERS`, `GATEWAY_ALLOW_ALL_USERS`,
   `group_policy`, `allow_bots`, ...) — must be read scope-aware:
   - Adapters: `_get_scoped_secret()` (canonical fail-closed copy in
     `plugins/platforms/feishu/adapter.py`, #86905).
   - Gateway authz: `_auth_env()` / `_platform_gate_env()` (`gateway/authz_mixin.py`).
   Rules:
   - Scope installed + multiplex active → a scoped miss returns the **default**.
     NEVER fall through to `os.environ` — that leaks another profile's value and
     silently breaks routing/admission (a leaked default allowlist skips the
     allow-all check and rejects every secondary-profile sender, #86905).
   - Unscoped default-profile path (`UnscopedSecretError`) and single-profile
     deployments keep the `os.environ` read — there it IS the profile's own value.
   - Authorization config is the sharpest edge: allowlist/allow-all leaks cause
     silent rejections (or worse, fail-open) that only show up as missing replies.
   - The `_get_scoped_secret` wrapper is copy-pasted across ~15 platform adapters —
     when touching any of them, make sure the fail-closed semantics are present;
     do not reintroduce the `except _UnscopedSecretError: val = os.getenv(...)`
     fallback-after-miss shape.

## Known Pitfalls

### DO NOT infer process identity from argv substrings
The bug class behind ~10 fleet-update issues (#90778, #87594, #78089,
#76129, #91964, ...): classifying a process by `"serve" in cmdline` or
similar. `kanban --preserve-cache` contains "serve"; a flag VALUE can
equal a subcommand (`-m dashboard serve`); truncated cmdlines hide the
real subcommand. Rules:
- Use the canonical matchers: `gateway.status.looks_like_gateway_command_line`
  (gateway run), `hermes_cli.update_cmd._hermes_holder_subcommand`
  (top-level subcommand of any Hermes argv). Never hand-roll token scans.
- Flag sets must be DERIVED from the parser
  (`_holder_value_flags()` introspects `build_top_level_parser()`), never
  hand-written lists — they drift.
- Never blanket-exclude ancestors from process scans: when `/update` runs
  as the gateway's child, a gateway ancestor must stay visible to the
  pause machinery (#87594). Exclude interactive ancestry, carve out
  gateway-shaped ancestors.
- Match on FULL cmdlines; truncate only at display time (#78089).
- Before adding any new scan heuristic, read #92091 — the gateway control
  socket replaces scans as the primary coordination mechanism; scans are
  the fallback layer for old/crashed processes.

### DO NOT hardcode `~/.hermes` paths
Use `get_hermes_home()` from `hermes_constants` for code paths. Use `display_hermes_home()`
for user-facing print/log messages. Hardcoding `~/.hermes` breaks profiles — each profile
has its own `HERMES_HOME` directory. This was the source of 5 bugs fixed in PR #3575.

### All CLI menu-pickers MUST use curses.
Interactive menus must use `hermes_cli/curses_ui.py`. See `hermes_cli/tools_config.py` for an example.

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

### Streaming delivery contract (stream-is-the-message adapters) — duplicate-final class
Adapters with `draft_stream_is_message = True` (relay Slack native streaming)
keep ONE cumulative native stream per turn; the stream IS the final message.
Four invariants, each learned from a live duplicate-final incident (NS-658
canary ledger, hermes#85796 / gateway-gateway#210). Violating any of them
re-creates a duplicate or a frozen stream:

1. **Draft frames must be prefix-stable.** The connector computes append-only
   deltas: frame N must be a string prefix of frame N+1. NEVER mutate draft
   frames per-tick — no fence-closing (`ensure_closed_code_fences`), no cursor
   suffix, no segment-state resets at tool boundaries, no mrkdwn conversion.
   Any non-prefix frame triggers a whole-snapshot re-append on the platform
   ("stacked copies"). The finalize path may still transform the real final.
2. **The consumer declares the final; the adapter never guesses.**
   `finish(final_text)` carries the completed `final_response` (verifier
   footer, completion explainer included) as the authoritative finalize
   payload. New post-stream response augmentation MUST ride this payload —
   if it mutates `final_response` after the stream sealed, it re-opens the
   #11 bug (`delivered_final_matches` mismatch → corrective duplicate send).
3. **Interim sends must carry `_interim_send` metadata.** Any consumer-side
   `adapter.send()` that is NOT the turn-final (commentary, segment-tail
   flushes) must set `metadata["_interim_send"] = True`, or the relay
   adapter's seal-interception will seal the live stream with interim text.
   Seal-interception exists at BOTH egress doors (`send()` AND
   `send_for_platform()`); a new egress door needs the same two checks.
4. **Reconcile by edit, never by plain send.** Any lane that delivers a final
   beside an already-sealed stream (queued follow-ups, media-accompanied
   finals, future lanes) must first try `edit_message` on the consumer's
   `message_id`; plain `send()` is the fallback only when no editable message
   exists. A sealed native stream is a regular message — `chat.update` on it
   works (live-verified).

Contract tests: `tests/gateway/test_stream_final_contract.py` (all four
invariants, mutation-checked). Slack streaming API ground truth (live-probed,
also encoded in connector comments/tests): `chat.*Stream` speaks STANDARD
markdown, not mrkdwn; `stopStream.markdown_text` APPENDS (never replaces);
`startStream`/`stopStream` are rate-limit Tier 2 (~20/min).

Guard style note: check `draft_stream_is_message` with `is True` — MagicMock
adapters in older tests auto-create truthy attributes.

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

---

## Testing

### Python
**ALWAYS use `scripts/run_tests.sh`** — do not call `pytest` directly. The script enforces
hermetic environment parity with CI (unset credential vars, TZ=UTC, LANG=C.UTF-8,
per-file subprocess isolation via `scripts/run_tests_parallel.py` — no xdist,
worker count auto-scaled from CPU count). Direct `pytest`
on a 16+ core developer machine with API keys set diverges from CI in ways
that have caused multiple "works locally, fails in CI" incidents (and the reverse).

```bash
scripts/run_tests.sh                                    # full suite
scripts/run_tests.sh tests/gateway/                     # one directory
scripts/run_tests.sh tests/agent/test_foo.py -k test_x  # runner is file-granular; -k narrows
scripts/run_tests.sh -v --tb=long                       # pytest flags pass through
```

- **Flake policy:** a failing FILE is retried once in a fresh subprocess (`--file-retries`;
  `HERMES_TEST_FILE_RETRIES=0` disables); a worker killed by signal or the file timeout is never
  retried (relaunching a runaway doubles the damage). Pass-on-retry is green but printed under `⚠ FLAKY`
  with both outputs — a bug to fix, not noise. Timing tests must not assume a quiet runner:
  wall-clock bounds ≥ 2s, event-based sync, no `assert not _wait_until(...)` races.
- **Placement mirrors the source tree.** A test lives in `tests/<top-level source dir>/` (`tests/hermes_cli/`,
  `tests/agent/`, `tests/hermes_state/`, `tests/gateway/relay/`, ...); installer/updater script tests
  under `tests/scripts/{install,desktop_update}/`. Only tests of root-level modules (`batch_runner`,
  `utils`, `hermes_constants`, packaging) sit directly in `tests/`. No issue numbers in filenames —
  cite the issue in the module docstring (`test_89315_x.py` → `test_x.py`, "Regression for #89315").
- **Placement (CI lanes):** `scripts/ci/classify_changes.py` picks jobs by changed files. A Python test
  asserting about `package.json`, `package-lock.json`, `tsconfig.json`, or `.ts/.tsx/.js/
  .mjs/.cjs` sources will not run on a JS-only PR (green on PR, red on `main` where the
  classifier fails open). Such tests belong in the vitest suite, not `tests/*.py`.
- **Tests must not write to `~/.hermes/`.** The autouse `_isolate_hermes_home` fixture in
  `tests/conftest.py` redirects `HERMES_HOME`; never hardcode `~/.hermes/` in tests. Profile
  tests also mock `Path.home()` so `_get_profiles_root()` / `_get_default_hermes_home()` stay
  in the temp dir (pattern: `tests/hermes_cli/test_profiles.py`):
  ```python
  @pytest.fixture
  def profile_env(tmp_path, monkeypatch):
      home = tmp_path / ".hermes"; home.mkdir()
      monkeypatch.setattr(Path, "home", lambda: tmp_path)
      monkeypatch.setenv("HERMES_HOME", str(home))
      return home
  ```
  Tests that `patch.object(Path, "home", ...)` must ALSO set `HERMES_HOME` — code reads the
  env var, not `Path.home()/.hermes`.

### Don't fake the host OS

Behaviour that genuinely differs per host is tested ON that host with `@pytest.mark.linux_only`
/ `macos_only` / `windows_only`, never by patching `sys.platform`. Host-independent things stay
unmarked: pure functions that take the platform as data (`hidden_windows_child_options(opts,
is_windows=True)`) and declaration/packaging invariants ("pyproject declares `tzdata` with a
`sys_platform == 'win32'` marker"). Setting a module-level `IS_WINDOWS` flag and calling
`windows_detach_flags()` IS a fake. The line: **if the test needs the interpreter to believe it
is on another OS to pass, it belongs on that OS.** A test that walks several platforms in
sequence is split — host-native arm on Linux, other arms as their own marked tests.

**Use the marker, never a bare `skipif`.** `scripts/ci/list_os_marked_tests.py` finds files for
the macOS/Windows lanes by grepping the marker *name*, then filters with `-m <marker>`. A
`skipif(sys.platform != "win32")` test skips on Linux AND is never imported on Windows — it runs
nowhere, silently. A file-local alias (`windows_only = pytest.mark.skipif(...)`) is listed but
`-m windows_only` deselects everything: green over zero coverage. Don't `pytest.skip()` non-host
rows of a platform `@parametrize` — split into one marked test per OS.

**Live Windows process-topology E2E (`wine2e` lane):** `windows-venv-e2e.yml` runs
`tests/hermes_cli/test_venv_holder_windows_live.py` on a real `windows-latest` runner (real
processes, no mocked psutil) ONLY on pushes to `wine2e/**` branches. Workflow: write probes
pinning CORRECT behavior, push to `wine2e/` to reproduce live on unfixed code, fix, iterate to
green, then open the PR with the live receipt. Extend it when touching that subsystem; assert
against the gateway ANCESTOR found by argv, not the direct parent (the venv shim makes every
spawn a launcher/worker chain).

### Don't write change-detector tests

A change-detector fails whenever data *expected to change* is updated — model catalogs,
`_config_version`, enumeration counts, hardcoded model lists. It adds no coverage and taxes
every routine update. Don't: `assert "gemini-2.5-pro" in _PROVIDER_MODELS["gemini"]`,
`assert DEFAULT_CONFIG["_config_version"] == 21`, `assert len(models) == 8`. Do: `assert
"gemini" in _PROVIDER_MODELS and len(_PROVIDER_MODELS["gemini"]) >= 1` (plumbing works);
`assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]` (migration reaches
latest); `assert not (set(moonshot_models) & coding_plan_only_models)` (no leak); every
catalog model has a context-length entry (relationship). If it reads like a snapshot, delete
it; if it reads like a contract between two pieces of data, keep it. Reviewers reject new
change-detectors; authors convert them before re-review.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **KenseiAgent** (382394 symbols, 685423 relationships, 1114 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact analysis before editing.** Use `impact({target: "symbolName", direction: "upstream"})` (MCP) or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .` (CLI fallback); report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/KenseiAgent/context` | Codebase overview, check index freshness |
| `gitnexus://repo/KenseiAgent/clusters` | All functional areas |
| `gitnexus://repo/KenseiAgent/processes` | All execution flows |
| `gitnexus://repo/KenseiAgent/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
### Never read source code in tests

A test that reads a `.py`/`.ts`/`.tsx` file's text tests the *shape of the source*, not
behavior — banned outright. It passes when the implementation is subtly broken (regex matches
a mis-wired call site) and fails on correct refactors; it can't run against bundled/minified
artifacts; it blocks structural cleanup; it gives false confidence. Don't
`fs.readFileSync('main.ts')` + `assert.match(source, /spawn\(...hiddenWindowsChildOptions/)`.
Do extract the logic into a pure/DI-testable function and call it:
```ts
export function hiddenWindowsChildOptions(options = {}, isWindows = process.platform === 'win32') {
  if (!isWindows || 'windowsHide' in options) return options
  return { ...options, windowsHide: true }
}
```
If the logic lives inline in a god-file and extraction feels disruptive, that is the signal to
extract, not to regex around it.

## Routing Table — working in X → read X/AGENTS.md

| Area | Read | Covers |
|---|---|---|
| `run_agent.py`, `agent/` | `agent/AGENTS.md` | AIAgent + mixins, turn phases, caching integrity, message-flow invariants, compression, model/aux resolution |
| `cli.py`, `hermes_cli/`, `main.py` | `hermes_cli/AGENTS.md` | CLI mixins, `_SLASH_DISPATCH`, slash registry, config system + loaders, skins, `hermes update` pipeline, profiles / multiplex |
| `gateway/` | `gateway/AGENTS.md` | Adapters, two message guards, streaming contract, background notifications, gateway vs desktop lifecycle, token locks, scoped secrets |
| `tools/`, `toolsets.py`, `model_tools.py` | `tools/AGENTS.md` | Adding tools, registry, toolsets, delegation, cross-tool references, backends |
| `plugins/`, `hermes_cli/plugins*.py` | `plugins/AGENTS.md` | Plugin kinds, native compat contract, in-tree policy, Sep-2026 compat window |
| `tui_gateway/`, `ui-tui/` | `tui_gateway/AGENTS.md` | Process model, JSON-RPC transport, key surfaces, slash flow, dev commands |
| `web/`, `hermes_cli/web_routers/` | `web/AGENTS.md` | Dashboard embeds the real TUI; what React may and may not rebuild |
| `apps/desktop/` | `apps/desktop/AGENTS.md`, `apps/desktop/src/AGENTS.md` | Desktop judgment guide; `serve` backend, slash palette curation, Bot Mode canonical chat |
| `skills/`, `optional-skills/`, `agent/curator*.py` | `skills/AGENTS.md` | Frontmatter, HARDLINE authoring standards, curator |
| `cron/`, kanban (`hermes_cli/kanban*.py`, `tools/kanban_tools.py`, `plugins/kanban/`) | `cron/AGENTS.md` | Scheduler invariants, job fields, kanban board/dispatcher |
| `gateway/platforms/` new adapter | `gateway/platforms/ADDING_A_PLATFORM.md` | Step-by-step adapter guide |
| profiles / multiplex / secret scope (any area) | `gateway/AGENTS.md` § Profile scope, `website/docs/user-guide/multi-profile-gateways.md` § What is isolated per profile | which execution points bind scope, what is isolated per profile |

Long-form background lives in `website/docs/developer-guide/` (agent-loop, prompt-assembly,
context-compression-and-caching, gateway-internals, tools-runtime, plugins/, cron-internals,
session-storage, ...). Workflow rules (PR/issue/review/salvage process) live in the
`hermes-agent-dev` skill, not here.
