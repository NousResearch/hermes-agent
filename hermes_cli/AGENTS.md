# CLI, configuration, updates, and profiles

The root guidance applies. Read `website/docs/developer-guide/cli-internals.md`, `website/docs/developer-guide/extending-the-cli.md`, and `website/docs/developer-guide/source-update-completion.md` for maintained architecture.

## CLI and commands

`cli.py` is the `HermesCLI` facade. Put behavior in the existing `hermes_cli/cli_*_mixin.py` or topical helper. Keep facade-owned mutable state and late-bound patch seams intact. Wrapper CLIs use the protected extension hooks rather than overriding `run()`.

Interactive menu pickers use `hermes_cli/curses_ui.py`. Spinner/display output space-pads lines. ANSI erase-to-end escapes leak through `prompt_toolkit` and are not valid cleanup.

`hermes_cli/commands.py::COMMAND_REGISTRY` is the source for names, aliases, help, completion, gateway exposure, and desktop disposition.

- Add a `CommandDef`, then the relevant CLI handler. Add a gateway handler and busy policy only when the gateway supports it.
- Commands that must work during a blocked turn use the gateway's plain/busy dispatch path described in `gateway/AGENTS.md`.
- Add aliases only to the registry. Dispatch accepts the literal alias returned by argparse.
- A command that changes system-prompt state defaults to next-session effect and offers an explicit immediate option when safe.
- Skill commands become user messages, never system-prompt mutations.
- Goal parsing and mutation stay in `hermes_cli/goal_command.py::dispatch_goal_command`. Clients only authorize, resolve sessions, render, and schedule.

## Configuration

Behavior belongs in `config.yaml`. Credentials belong in `.env` and register through `OPTIONAL_ENV_VARS`. Existing setup/config flows write both.

- Add normal options to `DEFAULT_CONFIG`. Bump `_config_version` only when existing files need a transformation.
- Every config key has both a registry/default entry and a runtime reader. Add an invariant test through the loader the consuming surface uses.
- All `config.yaml` writes go through `hermes_cli.config.atomic_config_write`. It preserves comments, ordering, quoting, deletion, and the unreadable-file guard. Do not use a generic YAML writer on a config path.
- Use the correct loader: `cli.py::load_cli_config` for interactive CLI defaults. `hermes_cli.config.load_config` for setup and most commands. `load_user_config_effective` for gateway, TUI gateway, cron, send, doctor, time, and logging when key presence matters. `read_user_config_raw` is only for round-trip writeback.
- The CLI working directory is `os.getcwd()`. Messaging uses `terminal.cwd`, bridged internally for child tools.

## Updates

`hermes update` is transactional: plan, snapshot, apply, restart by deployment kind, verify, and write a receipt.

- Plan is read-only and reports deployment kinds that require an external updater instead of modifying them.
- Snapshot every affected profile with the same file set before changing code. Quick snapshots recover files. Full backup mode owns rollback.
- Refuse a destructive source swap for a dirty tree. The Windows ZIP fallback runs only after Git itself fails, rechecks immediately before the swap, and preserves installer-owned nested build output.
- Restart every service tied to the updated installation, but leave other `HERMES_HOME` roots alone. Drain first and isolate per-service failures.
- Verify each live gateway's stamped code identity against the updated checkout. Mixed proven versions fail the update.
- Every started update writes a correlated machine-readable receipt, including early failure or missing completion child.
- New code runs in a new completion process. The pre-update interpreter never reloads pulled modules. Keep the frozen handoff module names importable for older updaters.
- Process scans are compatibility fallback. Use canonical full-command matchers, parser-derived flags, home bounds, and PID start times rather than substrings or bare PID checks.

## Profiles and services

Single-profile commands set `HERMES_HOME` before imports. Multiplex gateway and `serve` bind profile scope per activity while the process environment remains the launch profile.

Profiles are independent. `--clone` copies state once and strips messaging ownership unless `--clone-channels` is explicit. Build clones in hidden staging, materialize symlinked config/secret files, and publish by one rename.

A served named profile directory must contain an identity marker and must not be tombstoned. Enumeration is read-only: it never creates a profile home. A marker-less directory with user files fails closed instead of being removed. Process-global profile slots key on `hermes_home_key()`.

Gateway multiplex mode settles once at boot after preflight. Readers use the live served-profile record and explicit setting. They do not guess from merged defaults. Migration is resumable from its manifest and refuses unsafe automatic cases. Explicit service commands and update hooks share the same guards.

Service install, restart, and status cover systemd user/system units, launchd domains, Windows tasks, and Desktop launch. Generated services carry the intended `HERMES_HOME` and service-user `HOME`. Supervisors may start with an empty environment. A gateway that declares an external supervisor is handed back to that supervisor, not relaunched in the CLI foreground.

## Nous sign-in

Every free-tier promotion uses `anon_auth.run_sign_in()` and `settle_after_upgrade`. The state iterator owns one deadline, persists only after promotion and token grant, settles once, converts persistence errors into `Failed`, and supplies safe copy for each renderer. Keep cancellation semantics caller-specific. Enter profile scope only around preconditions and persistence, never across a network wait or yielded state. The plain connect-another-account device flow remains separate.

## Tests

Run `tests/hermes_cli/` through `scripts/run_tests.sh`. Exercise config through its real loader and writer, aliases through dispatch, updates through staged trees and correlated completion, and profiles with two homes. Use existing package tests for TUI or desktop consumers. Do not inspect JavaScript source from Python tests.