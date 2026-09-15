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
  real path with real imports against a temp `HERMES_HOME`. Mocks hide integration bugs.
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

Code paths in this reference are relative to the repository root.
