# hermes_cli/ + cli.py — CLI, slash commands, config, skins, updater, profiles

Applies on top of the root `AGENTS.md`. Long-form: `website/docs/developer-guide/cli-internals.md`.

## CLI architecture

`cli.py` holds `HermesCLI` (REPL loop, config, slash dispatch); behaviour lives in mixins
`hermes_cli/cli_commands_mixin.py`, `cli_stream_mixin.py`, `cli_status_bar_mixin.py`,
`cli_billing_mixin.py`, `cli_tui_mixin.py`, ... **Rich** renders banner/panels; **prompt_toolkit**
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
  in `.env` is deprecated — the loader warns; canonical is `terminal.cwd`.
- **Three loaders — know which you're in:** `load_cli_config()` (CLI, `cli.py`); `load_config()`
  (`hermes tools/setup`, most subcommands, `hermes_cli/config.py`, merges `DEFAULT_CONFIG`); raw
  YAML (gateway runtime, `gateway/run.py` + `gateway/config.py`). If the CLI sees a key and the
  gateway doesn't (or vice versa), you're on the wrong loader — check `DEFAULT_CONFIG` coverage.
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
  re-check), and grafts the live `apps/desktop/release/` into the staged swap (the GitHub source
  ZIP has no built desktop app; without the graft the swap deletes it).
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
  receipt is a bug: refused/failed runs are the ones receipts exist for.

Process-scan coordination between updater, serve/dashboard, and gateway is being replaced by a
gateway-owned control socket (#92091); scans are the fallback layer for old/crashed processes — read
#92091 before adding any heuristic. Process identity rules (never argv substrings; canonical
matchers; parser-derived flag sets; never blanket-exclude gateway ancestors, #87594): root
`AGENTS.md` and `website/docs/developer-guide/cli-internals.md`.

## Profiles (multi-instance)

`_apply_profile_override()` in `hermes_cli/main.py` sets `HERMES_HOME` before any module import, so
every `get_hermes_home()` scopes to the active profile (rules in root). Profiles are independent
islands by design — no live config inheritance; `--clone` copies at creation. Multiplex
(`gateway.multiplex_profiles`) secret-scope rules: `gateway/AGENTS.md`.
