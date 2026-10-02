# Phase 7.2 — canonical command definitions

Status: complete on `refactor/phase7-commands-tools`, following 7.1 commit
`269fb0a2d9b20ebffbe73dbd0be6f2e633398f3b`. Verified 2026-10-01 on Windows,
Python 3.11.15. Phase 5.8 and Phase 6 were not integrated or modified.

## Ownership

The existing frozen `CommandDef`, all 102 built-in definitions, the one name/alias
lookup, completion metadata, busy-policy queries, and Desktop metadata now live
in `commands/__init__.py`. Definitions were moved intact, preserving order,
descriptions, aliases, argument hints, subcommands, busy policies/handlers,
configuration gate paths, execution keys, and Desktop dispositions.

| Responsibility | Canonical owner |
|---|---|
| CommandDef, COMMAND_REGISTRY, resolve_command | commands |
| available_commands, is_gateway_available | commands |
| SUBCOMMANDS, argument-mode inference, Desktop wire metadata | commands |
| Gateway known names and busy-policy metadata queries | commands |
| Lazy published plugin command metadata | commands consumes plugin_runtime.api |
| CLI help descriptions, grouped help, Session help subgroups | hermes_cli.commands_presentation |
| Gateway help rendering and configuration gate loading | gateway.command_presentation |
| Plugin host command-resolver binding | commands binds plugin_runtime.host_bindings |
| Manifest-listed external compatibility entries | hermes_cli.commands only |

`available_commands(surface, enabled_config_gates=...)` returns built-ins in
registry order. CLI and Desktop exclude Gateway-only definitions; Desktop
execution/presentation dispositions remain metadata for their existing consumer.
Gateway availability accepts canonical names whose configuration gates the
application has resolved. The domain does not load configuration, persist it,
render help text, or authorize requests. Gateway help still applies the caller's
allowed-command filter, while dispatch authorization remains in
`gateway/slash_access.py` and the existing session/execution boundaries.

Importing the canonical package uses stdlib, shared constants, and the existing
lightweight plugin host binding. Published plugin metadata remains lazy and uses
Phase 4's `plugin_runtime.api.get_plugin_commands()`; no second registry,
registration pass, or discovery lifecycle was introduced.

## Consumer cut

Agent skill collision lookup, CLI dispatch and help, Gateway lookup/busy handling,
Desktop/TUI discovery and resolution, platform adapters, the Desktop dump script,
and shared executors' definition lookup now use the canonical owner directly.
Dynamic `module_loader` and `_tools_mod` references were migrated too. Desktop
catalog formatting stays with its consumer, preserving the existing RPC shape.

Canonical registry, busy-policy, and Desktop metadata tests live in
`tests/commands/`. Existing CLI completion/menu tests remain at the presentation
boundary and query the new owner. The migration evidence script was retargeted,
without rewriting the frozen 7.1 baseline.

Shared execution bodies and contracts remain in `hermes_cli/slash_exec.py` for
7.3. This step changes their definition/help imports only. Tool capability policy
and its consumer migration remain for 7.4–7.5. The public CLI entrypoint is unchanged.

## External compatibility exception

The old command module contains only the preexisting PLUGIN-COMPAT block and its
minimal supporting imports/logger. It has no CommandDef, built-in registry,
lookup, availability implementation, or forwards to the new canonical metadata.

All 26 manifest entries are retained:

- Imported/supporting names: Any, Callable, Dict, Mapping, Optional, Sequence,
  Tuple, field, os, shutil, subprocess, time.
- Restored helper/body entries: _requires_argument, _CMD_NAME_LIMIT,
  _clamp_command_names, _collect_gateway_skill_entries, discord_skill_commands.
- Existing lazy external pointers: SlashCommandAutoSuggest,
  SlashCommandCompleter, discord_skill_commands_by_category, slack_app_manifest,
  slack_native_slashes, slack_subcommand_map, telegram_bot_commands,
  telegram_menu_commands, telegram_menu_max_commands.

These remain isolated external obligations, barred from first-party dependencies
by the existing compatibility checker. No new compatibility entries were added.

## Verification

Both test batches used `scripts/run_tests.sh -j 4` with its clean environment and
per-file isolation:

- Registry/availability, CLI menus/completion, execution mapping, Gateway discovery
  authorization and slash-access, Desktop description fidelity:
  **85 passed, 0 failed, 1 Linux-only skip**, 11 files.
- Canonical tests plus agent/CLI command dispatch, busy handling, Discord
  registration/access, routed profile scope, skill collisions, real plugin
  registration/reload/unload/scoping, ACP, and Phase 4 ownership boundaries:
  **251 passed, 0 failed**, 23 files.
- After removing the 60 repeated passing tests: **276 distinct tests passed,
  0 failed, 1 skipped**, 29 distinct files.

The new import-isolation test blocks CLI, Gateway, agent, formatting utilities,
terminal UI, and plugin discovery while importing/querying built-in metadata.
A real on-disk A→B→A profile check proves gate loading follows the active profile
and Gateway rendering receives those resolved inputs.

Executable 7.1 replay matched **all 102 command definitions and 2,100 tool
selection cases**, including aliases, ordering, Desktop metadata, CLI catalog,
closed/open availability, and execution contracts/mapping. Tool policy was not
changed.

The existing compatibility checker completed over **8,025 tracked/new first-party
Python files**, finding no in-tree dependency on its 2,084 protected entries.
A subsequent changed-file check passed over 46 Python files. A tracked-source
search for full and relative legacy command references, followed by AST inspection
of 31 referencing files, found zero imports/dynamic loads of the retired owner.
New files were separately inspected by the changed-file check and import-isolation
test. Logger names retained for existing log-capture contracts are not imports.

`commands` is included in setuptools distribution package discovery through
`pyproject.toml`. This is a package-discovery check; wheel installation/startup
verification remains part of 7.7.

The initial unrestricted compatibility walk reached its 120-second job timeout.
The bounded tracked-file checker completed; its redundant second full AST pass
was cancelled after collecting the successful compatibility result. The targeted
source audit then completed successfully.

## Exit gate and remaining verification

Every built-in definition has one canonical owner. Registry and availability
tests pass directly against it, and first-party consumers do not use the retired
command definition path. **7.2's exit gate is satisfied.**

This does not claim Phase 7 completion. 7.3 must remove CLI dependencies from
shared executors. Later gates must migrate tool policy, enforce the completed
boundary, build/install distribution artifacts, run host-appropriate end-to-end
discovery, and integrate adjacent phases. The unchanged Windows session-policy
fixture failure recorded in 7.1 remains a later verification item; it was neither
modified nor represented as passing here.
