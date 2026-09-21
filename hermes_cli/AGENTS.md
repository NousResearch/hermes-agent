# hermes_cli/ + cli.py — CLI, slash commands, config, skins, updater, profiles

Applies on top of the root `AGENTS.md`. Long-form: `website/docs/developer-guide/cli-internals.md`.

## CLI architecture

`cli.py` holds `HermesCLI` (REPL loop, config, slash dispatch); behaviour lives in mixins
`hermes_cli/cli_commands_mixin.py`, `cli_stream_mixin.py`, `cli_status_bar_mixin.py`,
`cli_billing_mixin.py`, `cli_tui_mixin.py` (widgets, keybindings, panels), `cli_tui_runtime_mixin.py`
(run-loop phases: input dispatch, startup, signals, shutdown), `cli_init_mixin.py` (the `__init__`
phases), ... **Rich** renders banner/panels; **prompt_toolkit**
handles input + autocomplete; `KawaiiSpinner` (`agent/display.py`) animates API calls and prints
the `┊` activity feed. `load_cli_config()` in `cli.py` merges CLI defaults + user YAML.
`process_command()` resolves the canonical name via `resolve_command()` then dispatches through
`HermesCLI._SLASH_DISPATCH` (`canonical -> (method name, pass_arg)`), falling back to a
`_handle_<name>_command` method by naming convention. **There is no `elif` ladder — do not add one.**
Skill slash commands (`agent/skill_commands.py`) scan `~/.hermes/skills/` and inject as a **user
message**, never into the system prompt (prompt caching).

Rules: all interactive menu-pickers use curses (`hermes_cli/curses_ui.py`; example
`hermes_cli/tools_config.py`). Never emit `\033[K` (ANSI erase-to-EOL) in spinner/display code —
it leaks as literal `?[K` under prompt_toolkit's `patch_stdout`; space-pad instead:
`f"\r{line}{' ' * pad}"`. Wrapper CLIs extend via the protected hooks in `cli_tui_mixin.py`
(`website/docs/developer-guide/extending-the-cli.md`), not by overriding `run()`.

## Slash command registry (`hermes_cli/commands.py`)

`COMMAND_REGISTRY` (list of `CommandDef`) is the single source; everything derives from it: CLI
dispatch (`resolve_command()`), gateway `GATEWAY_KNOWN_COMMANDS` + dispatch, `gateway_help_lines()`,
`telegram_bot_commands()` (BotCommand menu), `slack_subcommand_map()`, `COMMANDS` (autocomplete),
`COMMANDS_BY_CATEGORY` (`show_help()`). Fields: `name` (no slash), `description`, `category`
(`Session | Configuration | Tools & Skills | Info | Exit`), `aliases` tuple, `args_hint`,
`cli_only`, `gateway_only`, `gateway_config_gate` (config dotpath; a `cli_only` command becomes
gateway-available when truthy — `GATEWAY_KNOWN_COMMANDS` always includes gated commands so the
gateway can dispatch them; help/menus show them only when the gate is open).

**Adding a command:** (1) `CommandDef("mycommand", "What it does", "Session", aliases=("mc",),
args_hint="[arg]")` in `COMMAND_REGISTRY`; (2) `_handle_mycommand_command(self, cmd_original)` on the
relevant `cli_*_mixin.py` (picked up by convention) or an explicit `_SLASH_DISPATCH` entry
`"mycommand": ("_handle_mycommand", True)` when the method name/arg-passing differs; (3) for the
gateway, `_handle_mycommand_command(self, event)` on the matching `gateway/slash_commands_*.py`
mixin and list it in `_IDLE_COMMANDS` (or `_PLAIN_COMMANDS` if it must work mid-run) in
`gateway/run_busy.py` — handlers resolve by name via `_command_handler_table`; (4) persistent
settings via `save_config_value()` in `cli.py`. **Adding an alias** = add to `aliases`; every
surface updates automatically. Commands that mutate system-prompt state default to deferred
invalidation with `--now` opt-in (root invariant).

### Shared goal commands

`hermes_cli/goal_command.py::dispatch_goal_command` owns `/goal` parsing and manager
mutations. CLI, messaging gateway, TUI/Desktop/dashboard and Desktop goal controls
all delegate there; adapters only resolve sessions, authorize gate creation, render
results, and schedule kickoff/continuation prompts. Async callers preserve ContextVars
when running dispatch off-loop (drafting uses profile-scoped auxiliary credentials).
Do not add a surface-specific goal parser. ACP has no goal command or goal loop yet.

## Config system (`hermes_cli/config.py`)

- **config.yaml option:** add to `DEFAULT_CONFIG`. Bump `_config_version` ONLY to actively
  migrate/transform existing config (rename keys, restructure); new keys deep-merge automatically.
  Top-level sections (non-exhaustive): `model, agent, terminal, compression, display, stt, tts,
  memory, security, delegation, smart_model_routing, checkpoints, auxiliary, curator, skills,
  gateway, logging, cron, profiles, plugins, honcho`. `auxiliary` = per-task side-LLM overrides
  (`agent/AGENTS.md`); `curator` = `enabled, interval_hours, min_idle_hours, stale_after_days,
  archive_after_days, backup.*`.
- **.env = SECRETS ONLY** (keys, tokens, passwords): add to `OPTIONAL_ENV_VARS` with
  `{"description", "prompt", "url", "password": True, "category": provider|tool|messaging|setting}`.
  Non-secret settings go in config.yaml; if internal code needs an env mirror, bridge it in code
  (`gateway_timeout`; `terminal.cwd` → `TERMINAL_CWD`). `MESSAGING_CWD` is removed and `TERMINAL_CWD`
  in `.env` is deprecated — the loader warns; canonical is `terminal.cwd`. `hermes config
  set/get/unset <NAME>` route any bare name registered in `OPTIONAL_ENV_VARS` / `_EXTRA_ENV_KEYS`
  (or carrying a `setup_hidden_env` platform suffix) to `.env` via `config_env_routing.py` — the
  file the platform setup flows write — never to the top level of config.yaml.
- **Three loaders — know which you're in:** `load_cli_config()` (CLI, `cli.py`); `load_config()`
  (`hermes tools/setup`, most subcommands, `hermes_cli/config.py`, merges `DEFAULT_CONFIG`);
  `hermes_cli/config_effective.py::load_user_config_effective()` (gateway runtime via
  `gateway/run.py::_load_gateway_config`, TUI gateway `_load_cfg`, cron, `hermes send`, doctor,
  `hermes_time`/`hermes_logging`: user file + managed overlay + `${VAR}` expansion + model-key
  canon, NO defaults — for presence-sensitive readers). If the CLI sees a key and the gateway
  doesn't (or vice versa), you're on the wrong loader — check `DEFAULT_CONFIG` coverage. Never
  hand-roll raw-read → overlay → expand; `read_user_config_raw` is for write-back round-trips only.
- **Every `DEFAULT_CONFIG` key has a runtime reader, and every reader a registry entry.** Both drift
  modes are silent: a registered key nothing reads (a knob that does nothing) and a reader of a key
  never registered (never shown, never migrated; new roots also go in `_KNOWN_ROOT_KEYS`). For a new
  key: `rg -n '"<key>"' hermes_cli/config_defaults.py` AND `rg -n '<key>' --glob '!tests' .` both hit,
  and one invariant test sets it in a temp `config.yaml` and asserts the behaviour through the loader
  the consuming surface uses. Under multiplex the process env never overrides a profile's YAML.
- **Working directory:** CLI uses `os.getcwd()`; messaging uses `terminal.cwd`, bridged to
  `TERMINAL_CWD` for child tools.

## Skin engine (`hermes_cli/skin_engine.py`)

Skins are **pure data** (`SkinConfig`); no code change to add one. `init_skin_from_config()` reads
`display.skin` at startup; `get_active_skin()` (cached), `set_active_skin(name)` (`/skin`),
`load_skin(name)` (user `~/.hermes/skins/*.yaml` → built-ins → default; missing values inherit
from `default`). Built-ins in `_BUILTIN_SKINS`: `default`, `ares`, `mono`, `slate`. Keys: `colors.*`
(banner border/title/accent/dim/text, response_border), `spinner.*` (waiting/thinking faces,
thinking_verbs, wings), `tool_prefix`, `tool_emojis`, `branding.*` (agent_name, welcome,
response_label, prompt_symbol). Consumers: `banner.py`, `display.py`, `cli.py`. Key-by-key table
and YAML template: `website/docs/user-guide/features/skins.md`.

## Update pipeline (`hermes update`) — transactional; every stage guards a real field failure

Fleet-update campaign #91277 (Aug 2026). A PR that weakens a stage must answer for the failure class
it guards. `plan → snapshot → apply → restart-per-kind → verify → report`

- **Plan** (`update_inventory.py`, `hermes update --plan`): read-only inventory — install kind, all
  profiles, every live gateway with supervisor + running code version. Deployment kinds are
  first-class: `git` updates in place; `docker`/`nix`/`apt` are NOT in-place-updatable and the
  updater reports the correct external command instead of fighting the deployment model.
- **Snapshot** (`backup.py`): pre-update quick snapshot for EVERY profile (the code swap + fleet
  restart touch all of them), each into its own `state-snapshots/`, identical file set, 1 GiB
  per-file cap, keep=1. **Never add a partial/tiered snapshot set** — mixed coverage creates
  torn-restore states across schema generations. Quick snapshots are FILE-LOSS RECOVERY (the
  per-profile cron-jobs safety net restores from them), NOT code-rollback insurance; `--backup`
  full mode owns rollback.
- **Apply**: git pull, or the Windows ZIP fallback — which fires ONLY when git itself failed
  (`_should_zip_fallback_on_update_error`, argv-classified; a dependency-install failure must never
  trigger a tree-clobbering re-download), REFUSES a dirty working tree (`-uall` + a pre-swap TOCTOU
  re-check — but classifies a `!!` line by whether the swap would destroy it: an ignored path under a
  root entry the ZIP does not ship (`.bytecode-fingerprint`, `.hermes-bootstrap-complete`,
  `hermes_agent.egg-info/`; tracked root entries stand in for the ZIP set before the download, the
  re-check gets the real one), a nested `__pycache__`/`node_modules`, or a `_ZIP_PRESERVED_NESTED`
  output is admitted; other ignored files under shipped dirs still block), and grafts the live nested
  build outputs (`_ZIP_PRESERVED_NESTED`: `apps/desktop/{release,dist,node_modules,build}`,
  `hermes_cli/web_dist`, `ui-tui/{dist,node_modules,packages/hermes-ink/dist}`, `web/node_modules`,
  `scripts/whatsapp-bridge/node_modules`) into the staged swap by hardlink (the GitHub source ZIP has
  none of them; without the graft the swap deletes them). Post-swap, the Desktop
  rebuild decision also trusts the build stamp under HERMES_HOME, so an install that already lost
  its artifacts in an earlier update is rebuilt instead of "forgotten" (#90495).
- **Restart-per-kind**: systemd and launchd restarts are FLEET-WIDE (every `hermes-gateway*` unit /
  `ai.hermes.gateway*` LaunchAgent), drain-first (SIGUSR1), with per-unit/per-label failure
  isolation. Restarting only the invoking profile's service leaves siblings on stale `sys.modules`
  until they crash — the largest dupe-PR cluster in the repo's history came from that bug.
- **Verify**: gateways stamp `code_sha`/`code_version` into `gateway_state.json` on every
  runtime-status write (`gateway/status.py`); the updater compares each live gateway against the
  fresh checkout and prints a fleet version matrix. A provably-stale gateway fails the update
  (exit 1) — automation must never treat a mixed-version fleet as healthy.
- **Report**: every run writes a machine-readable receipt to `~/.hermes/logs/update_receipts/`
  (`latest.json` pointer; steps, skips WITH reasons, restart outcome, plan, fleet snapshot).
  Finalization is owned by the `cmd_update` command boundary — early `sys.exit` paths (preflight
  refusals, fetch failures) still persist a receipt with the real exit code. A begun-but-unwritten
  receipt is a bug: refused/failed runs are the ones receipts exist for. The receipt is opened by
  the pre-swap process and finished by the post-swap child (below): `detach_update_receipt` /
  `resume_update_receipt` carry it across, so one run still yields exactly one receipt. A write
  failure prints `⚠ Update receipt not written` and logs at WARNING, never debug.
- **Nothing runs pulled code in the pre-pull interpreter** (`update_handoff.py`). The process that
  started `hermes update` imported the PRE-pull tree; once git (or the ZIP swap) has replaced the
  checkout it stops, writes the hand-off payload (open receipt, pre-update plan, pre-update
  version/active features, Windows pause token) and re-executes
  `hermes update <same flags> --post-swap <file>` under the venv interpreter, which imports only the
  pulled tree and owns the tail (deps, Node/web/Desktop, maintenance, config migration, fleet
  restart, verification, receipt); the parent relays the exit code. Every "purge `sys.modules`" /
  "reload this list of modules" / "isolate this one step" fix was a symptom of the old shape and is
  gone — do not reintroduce one: a phase that needs new code runs in the child, full stop. Mocked
  updater tests run the tail in-process via the `_inline_post_swap_handoff` autouse fixture
  (`@pytest.mark.real_post_swap_handoff` opts out). Live A/B:
  `evals/update_pipeline/post_swap_handoff_ab.sh`. Post-update steps still isolate their own
  failures (a crashed notice must not abort the fleet matrix and receipt finalize that follow).

Process-scan coordination between updater, serve/dashboard, and gateway is being replaced by a
gateway-owned control socket (#92091); scans are the fallback layer for old/crashed processes — read
#92091 before adding any heuristic. Process identity rules (never argv substrings; canonical
matchers; parser-derived flag sets; never blanket-exclude gateway ancestors, #87594): root
`AGENTS.md` and `website/docs/developer-guide/cli-internals.md`.

## Profiles (multi-instance)

`_apply_profile_override()` in `hermes_cli/main.py` sets `HERMES_HOME` before any module import for
single-profile commands (`hermes -p x <cmd>`), so there `get_hermes_home()` scopes to the active
profile. The multiplex gateway and the Desktop/dashboard `serve` backend instead bind the active
profile per activity via a contextvar override while `os.environ["HERMES_HOME"]` keeps the launch
profile — a module constant or import-time read there freezes to the launch profile (rules in
root). Profiles are independent
islands by design — no live config inheritance and no credential inheritance (a named profile reads
only its own `auth.json`/`.env`; the root store is never a fallback and never a write-through target,
#111724 — a profile without a provider gets the setup prompt); `--clone` copies at creation, minus messaging
channels (`profile_channels.py`: ownership-based inventory evaluated in the SOURCE's plugin scope —
adapter-declared keys + canonical/alias prefixes + `GATEWAY_ALLOW*`/`GATEWAY_RELAY_*`; prefixes shared
with tools (`HASS_`/`TWILIO_`/`EMAIL_`) are stripped only when the source runs that adapter; never a hand
list). `--clone-channels` opts in and its live-multiplexer refusal lives in `create_profile` (CLI, REST
and TUI all go through it). Clones are built in `profiles/.<name>.staging-<pid>` (hidden → invisible to
`_iter_named_profile_dirs` and the hot-serve rescan) and published by one `os.rename` after the strip;
symlinked `.env`/`config.yaml` are materialized first so a clone never writes through to its source. Multiplex
(`gateway.multiplex_profiles`) secret-scope rules: `gateway/AGENTS.md`. The served set is
`profiles.py::profiles_to_serve(multiplex=True)` = default + every live dir under `profiles/` — live =
carries an identity marker (`hermes_constants.named_profile_has_identity`: `config.yaml`/`.env`/`SOUL.md`/
`profile.yaml`/`auth.json`/`state.db`) and is not tombstoned. A marker-less dir (cron/log side-effect
shell, stray infrastructure dir) is never listed, served, ticked, `.env`-backfilled or resolvable via `-p`
(#95188, #99392); `profile create` replaces it only when it is also tombstoned (a live marker-less dir may hold user
files — fail closed, never rmtree). A dangling symlinked marker still counts as identity (`is_symlink()`).
`tools/bot_mode_probe._roster` (Bot Mode teammate roster, `bot_relay.deliver` target check) applies the same
predicate. There is no allowlist (`gateway.multiplex_profile_allowlist` was retired in config v43).
Enumeration is a pure read: never `mkdir` a profile home from a served path (`SessionDB`, logging,
cron all go through `mkdir_under_hermes_home` / `_ensure_cron_dir`, which refuse a deleted or
missing named profile, #94590). Process-global per-profile slots (MCP discovery in `mcp_startup.py`,
tool registry overlays) key on `hermes_constants.hermes_home_key()`, never a single flag.
`gateway.multiplex_profiles` defaults to **on**, but `GatewayConfig` keeps an unset flag `None` and
`gateway_multiplex_mode.resolve_multiplex_mode` settles it once per boot (called from
`load_gateway_config_for_runner`): default profile, >= 2 profiles, no standalone secondary gateway,
no preflight blocker, migratable host → `True`; else `False` + a logged reason. Explicit values pass
through. CLI/dashboard readers use `default_gateway_multiplexes` (live `served_profiles` record, then
the explicit flag) — never the merged default, which would guess a verdict only the gateway makes.
Migration from per-profile gateways: `hermes_cli/gateway_migrate.py` (`hermes gateway migrate
--multiplex|--standalone`, table-driven `_PREFLIGHT_CHECKS`, manifest `<default>/gateway_migration.json`);
`update_cmd_fleet._verify_fleet_after_update` calls `maybe_auto_migrate_after_update` on the success
path only; `gateway_migrate_guards.py` holds the auto-path-only refusals (table `_AUTO_MIGRATION_GUARDS`:
other service domain / UNIX user / HERMES_HOME outside `profiles/` — notices for the explicit command,
blockers for the hook) and the `gateway.auto_multiplex_migration` opt-out (#109954). Blockers reuse `GatewayRunner._adapter_credential_fingerprint` and `platform_binds_port`;
"has a `/p/<profile>/` ingress" is the adapter class attribute `serves_profile_prefix` — set it on a
new HTTP-inbound adapter when it answers the prefix, never extend a list here.
`hermes gateway restart` for a gateway Hermes did not install (custom launchd agent / unit running
`gateway run --external-supervisor`): `gateway_supervised_restart.py` — the gateway's SELF-declared
supervisor (control-socket `identify` answering anything but `manual`, OR the argv marker) decides; hand back via SIGUSR1 and wait
for a fresh supervised PID, never stop + foreground `run_gateway` (that stamps the CLI's PID and wedges
every KeepAlive respawn, #110637).

Service installs are a matrix, not a unit file: `gateway_service_unit.py::generate_systemd_unit(system=,
run_as_user=)` (systemd unit generation / `systemd_unit_is_current` / `refresh_systemd_unit_if_needed` live in that
sibling and read facade helpers late-bound through `hermes_cli.gateway`, so patch them on the facade; user unit AND `--system` unit with `User=`; an unresolvable `User=` is a blocker,
never a dir-owner fallback), `generate_launchd_plist` (`gui/<uid>` then `user/<uid>` domains, never a
`~/Library/LaunchAgents` glob), Windows Scheduled Task and the Desktop-spawned backend all carry the
profile's `HERMES_HOME` (and `HOME` for the service user) explicitly — a supervisor starts with an
empty environment, so the env override that makes `-p` work interactively does not exist there. A
change to install/restart/status regenerates and diffs every kind; both user and system units are
recorded when both exist. Process liveness is `(pid, start_time)` or the canonical matchers
(`gateway.status.live_gateway_pid_for_home`), never bare PID existence.

## Nous free tier (`hermes_cli/anon_auth.py`)

Sign-in completion is one function, `settle_after_upgrade`, called by every caller that persists an
account over a free-tier identity (CLI `upgrade_guest`, the desktop poller): it moves a config on the
welcome route to the account's host and the tier's recommended default
(`models.recommended_nous_default_model`, shared with `GET /api/model/recommended-default`).

The shared flow, states, and copy live in `anon_sign_in.py`; CLI rendering lives in
`anon_sign_in_cli.py`. `anon_auth.py` keeps identity, promotion polling, and settlement, and
re-exports the existing sign-in API. The flow resolves identity and persistence collaborators
through `anon_auth` at call time to preserve module-attribute monkeypatch seams.

The sign-in itself is one composition: `anon_auth.run_sign_in()` yields `SignInState`s (`Code`,
`Waiting`, `Completed`, `Declined`, `Superseded`, `TimedOut`, `Retired`, `Failed`,
`AlreadySignedIn`, `Unavailable`). It reads the current state itself, holds one absolute deadline
across both waits, persists only after a completed promotion **and** a token grant, runs
`settle_after_upgrade` exactly once per completion, and never lets a persist or settle failure
escape as an exception — it becomes `Failed`. Every state carries its own `.copy` (the chat form,
which never contains a raw exception, a URL or a `hermes` verb) and `.copy_terminal`, so no caller
maps a reason to a string. `cancelled()` stops an attempt; `cancel_wins_after_promotion` decides
what happens when the server had already completed the transfer — the desktop keeps `True` (a
DELETE means "not on this machine"), the gateway passes `False` (a supersede must not discard a
transfer the user actually approved). `scope` is entered only around the precondition and persist
blocks, never across a `yield` or a network wait, because `run_in_executor` does not carry
contextvars. `upgrade_guest` (`hermes auth upgrade`), the CLI `/login` handler and the desktop
promotion poller are renderers over it; a surface that needs the cancel check and the save to be
atomic passes `persist_guard`. The desktop's plain "connect another Nous account" device-code login
is a separate path (`_nous_plain_poller`) and must stay one.
