# Phase 7.1 — baseline and dependency inventory

Status at 7.1 capture: inventory and executable baseline complete. The subsequent
[7.2 receipt](phase7.2.md) records the command definitions cut; the
[7.3 receipt](phase7.3.md) records the shared execution cut; the
[7.4 receipt](phase7.4.md) records the runtime capability-policy cut and the
approved explicit-empty correction. The
[7.5 receipt](phase7.5.md) records consumer migration and the shared application
configuration operation. The
[7.6 receipt](phase7.6.md) records the ownership hard cut and blocking CI gate.
The [7.7 receipt](phase7.7.md) records integration and verification.
Captured 2026-10-01 on Windows, Python 3.11.15.

## Foundation and isolation

- Baseline: `f04be5e51b781733edb839cc83059f66adaef1db`,
  **Close Phase 4 plugin runtime ownership**.
- Baseline branch: `refactor/gateway-cli-boundary`; the same commit is published as
  `fork/split/phase4-plugin-runtime`.
- New branch: `refactor/phase7-commands-tools`.
- Worktree: `C:\!bin\workspace\hermes-phase7-commands-tools`.
- Phase 5.8 (`hermes-phase5-8-runtime-consumers`) has existing uncommitted work.
  It was inspected for status only; no writes, merges, resets or cherry-picks there.
- Phase 6 (`refactor/phase6-auth-credentials`) remains independently maintained.
  None of its implementation was incorporated into this branch.

Phase 4 supplies the actual published APIs in `plugin_runtime/api.py`:
`get_plugin_commands()` performs idempotent discovery; `get_plugin_toolsets()`
reads manager/registry registrations. The current selection path also uses
`plugin_runtime.lifecycle.get_plugin_toolset_keys_nowait()`: it must retain its
nonblocking startup and persisted-key semantics, rather than replace them with a
blocking discovery call merely because another API is available.

## Evidence and reproduction

- `dependency-inventory.json`: direct source inventory, 419 import/reference
  locations (including tests, dynamic module-name strings, related CLI sibling
  modules and the evidence script), 36 compatibility entries, and current
  CLI/Gateway/Desktop/ACP execution mapping declarations.
- `behaviour-baseline.json`: all 102 built-in definitions, ordered metadata,
  aliases and resolution, CLI catalog, Gateway known-command names, availability
  with gates closed/open, Desktop dispositions and argument modes, shared
  execution contracts and executor mapping, and 2,100 tool-selection cases.
- `scripts/phase7_baseline.py`: executable capture/check/inventory aid.
  The evidence snapshots are temporary migration receipts, not permanent tests
  freezing product catalogs or enumeration counts.

Use a test interpreter with the existing dependencies and an isolated
`HERMES_HOME`/`HERMES_RUNTIME_DIR`, with credentials removed. From this checkout:

```text
python scripts/phase7_baseline.py inventory
python scripts/phase7_baseline.py capture
python scripts/phase7_baseline.py check
```

Capture refuses to overwrite the recorded baseline. Capture and check both
succeeded, in separate interpreter processes under a temporary home.
During 7.2–7.5 retarget this evidence script to the canonical owners together
with the tests. Do not keep the old imports alive to accommodate the script.
Rerun the structural inventory after the hard cut.

The matrix covers all 22 platform-registry entries plus ACP, raw GUI and an
unknown-platform diagnostic, with 21 configuration cases, credentials absent
and present, and default MCP inclusion on/off. It records the supplied config,
plugin keys, selected toolsets and actual composite expansion. Credential
presence and synthetic plugin metadata are controlled inputs; toolset definitions
and selection/expansion implementations are real. Actual plugin lifecycle,
profile scope and runtime availability still need the integration regressions in
7.7. Raw GUI is diagnostic: real GUI session policy maps the surface to Desktop
and folds in surface toolsets; the GUI regression tests exercise that path.

Cases include defaults, explicit empty and explicit native selections,
composites and mixed lists, new/declined/explicit plugin keys, MCP defaults,
explicit MCP allowlists, disabled servers, no_mcp, final global suppression
(including a composite and a list-literal string), active context engines,
legacy kanban opt-in, known built-ins, list-literal selections and invalid names.

## Ownership and consumers

Every located reference is in the JSON inventory, with file and line. The
following groups define the implementation batches; related presentation
consumers are intentionally distinguished from policy reads.

| Surface | Current consumers and responsibility | Phase 7 destination |
|---|---|---|
| Gateway | run/run_inbound/run_turn, platforms/base, session_commands: lookup, aliases, busy handling; slash_commands: shared executor calls | commands query/execution interfaces; orchestration stays Gateway |
| Gateway policy | run, run_turn, session, session_policy, hosted_room_execution_policy, platforms/api_server | tools platform policy; retain config loading at consumer |
| Desktop/TUI | command_discovery, server, methods_tools, methods_slash: catalog, completion, lookup and dispatch | commands; keep RPC/session execution and wire shapes |
| Desktop/TUI tools | agent_factory, methods_profiles, methods_tools | tools policy for reads; application operation for shared configuration writes |
| ACP | session imports _get_platform_tools and enabled_mcp_server_names; commands owns its curated command map | canonical policy; preserve ACP-specific command dispatch |
| Agent/plugin commands | agent/skill_commands; plugins/context_engine; platform adapters for Discord, Slack, Matrix, Telegram | commands metadata and published plugin APIs; lifecycle remains subsystem-owned |
| CLI | cli.py, cli_commands/info/loops/modal/model_switch mixins, commands_completion, commands_platforms | commands for definitions/shared execution; retain terminal presentation |
| Other policy consumers | cron scheduler/preflight, tools/kanban_tools, doctor_tools, kanban_db_dispatch, oneshot, prompt_size, memory_setup, web_server_config | tools runtime policy |
| Configuration/UI | config, config_migrations, toolset_validation, setup_quick, tools_config_mcp, web_routers/tools | scope/policy reads move; menus/providers/persistence remain application concerns |
| Generated Desktop fallback | scripts/dump_desktop_slash_registry.py and apps/desktop/src/lib/desktop-slash-registry.json | continue deriving from the one canonical registry |

Dynamic imports through `_tools_mod`, `_lazy` and `module_loader` are real
consumers and must migrate too. Test monkeypatch targets are not external API.

## Frozen command and execution contracts

`CommandDef` fields, ordering, aliases, subcommands, busy policies/handlers,
Desktop dispositions and argument modes are recorded verbatim. Configuration
gates are `verbose -> display.tool_progress_command` and
`skills -> skills.write_approval`. Gateway known-command dispatch includes
gated commands even when discovery hides them.

Six execution keys resolve to one implementation each:

| Key | Current implementation |
|---|---|
| version | _exec_version |
| egress | _exec_egress |
| profile | _exec_profile |
| bundles | _exec_bundles |
| gateway_help | _exec_help |
| gateway_commands | _exec_commands |

`CommandContext`: surface, args, options, config_get.
`CommandReply`: text, data, format. Both remain frozen dataclasses.
Shared result formatting, pagination, allowed-command filtering and errors retain
their current contracts. The shared executors currently depend on CLI banner,
proxy status, profile helpers and command help; these dependencies need existing
subsystem APIs or explicitly supplied application inputs in 7.3, not new facades.

Plugin host binding in commands must follow the canonical registry. Plugin
commands remain lazily obtained from plugin_runtime; built-ins win catalog
collisions, plugin entries precede skills, and profile/workspace binding stays
with the existing discovery owners. Do not relocate skill/plugin lifecycles.

Gateway `slash_access.py` owns authorization; execution approvals and session
policy retain their current enforcement sites. Availability and discovery do
not grant execution authority.

## Runtime policy extraction dependencies

The current selection call graph is `_get_platform_tools` plus explicit/composite
selection, default-off and recently-shipped rules, static configurable subset
inference, native recovery, plugin enablement, MCP merging and final disabled
selection pruning. It depends on:

- Existing `toolsets.TOOLSETS/resolve_toolset` and runtime registry, which remain.
- Platform default identifiers currently coupled to CLI platform display data.
- Configurable key membership/default-off/recently-shipped policy currently
  mixed with labels and provider menus.
- Platform restrictions currently in `hermes_cli.toolset_scope`.
- List-literal parsing currently in `hermes_cli.toolset_validation`.
- xAI/HASS credential-presence inputs, currently reached through CLI/auth/secret
  helpers. Supply resolved inputs or use actual subsystem APIs; do not import
  Phase 6 silently to satisfy them.
- `model_tools._apply_toolset_selection` for composite disabled-tool pruning.
  Extract shared mechanics only if necessary to avoid a circular dependency;
  retain model schema preparation and runtime availability in their owners.
- `_save_platform_tools`, consumed by Desktop, HTTP/UI and CLI. Its reconciliation
  of known plugin/built-in keys, preserved MCP entries and global disabled keys
  is a shared application write operation, not runtime policy.

## Compatibility audit

The manifest documents 26 entries on `hermes_cli.commands` and 10 on
`hermes_cli.tools_config`; none on slash_exec or toolset_scope. Exact entries
and original targets are recorded in the inventory.

The externally named command obligations are SlashCommandAutoSuggest,
SlashCommandCompleter, discord_skill_commands, discord_skill_commands_by_category,
slack_app_manifest, slack_native_slashes, slack_subcommand_map,
telegram_bot_commands, telegram_menu_commands and telegram_menu_max_commands.
The manifest also restores their private helper dependencies
(_requires_argument, _CMD_NAME_LIMIT, _clamp_command_names,
_collect_gateway_skill_entries) and leaked stdlib/typing imports. These are
supporting compatibility implementation, not independent internal APIs.

The tools_config entries are MANAGED_FEATURE_COVERAGE_CATEGORY,
NOUS_MANAGED_PROVIDER, base_url_hostname, fal_key_is_configured,
format_nous_portal_entitlement_message, is_truthy_value, save_env_value, shutil,
subprocess and sys. They concern existing presentation/provider/configuration
compatibility, not canonical runtime selection.

`COMPAT_MANIFEST.md` schedules removal on 2026-09-14, but these blocks still
exist at the actual baseline and the plugin opt-in escape hatch remains.
A past removal date alone does not prove the code has been removed or that
existing externally named obligations can be dropped here. Preserve only these
explicit obligations, isolated from canonical definitions and barred from
first-party use. The manifest does NOT establish an obligation to forward
CommandDef, COMMAND_REGISTRY, resolve_command, the shared execution package,
_get_platform_tools or toolset scope.

The public CLI entrypoint and the frozen old-updater tool installer stop/relaunch
surface (`_pip_install`, `install_cua_driver`; see tests/compat) are unrelated
exceptions. Keep genuine CLI presentation/configuration in place until Phase 11.
Do not mistake all of tools_config for obsolete runtime implementation.

## Verification and baseline discrepancies

Run through `scripts/run_tests.sh` (credential-clean, per-file isolation):

- Commands, shared execution mapping, CLI tools, Gateway API toolsets, slash
  access and busy-policy regressions: **69 passed**, 0 failed, 6 files.
- Tool policy, Gateway discovery/access dispatch, session policy, Desktop GUI
  toolsets/profile pinning and ACP commands, rerun with ACP/pytest-asyncio
  dependencies present: **79 passed, 1 failed, 7 skipped**, 7 files.
- Combined distinct tests: **148 passed, 1 failed, 7 skipped**.
- Executable migration baseline: **102 command definitions and 2,100 tool
  selections captured; replay matches exactly**.

The lightweight interpreter initially lacked ACP and pytest-asyncio, so its
collection/async errors were superseded by the fuller interpreter run. The
initial automatic activation could not verify its downloaded ripgrep executable;
it was not used as verification evidence. No dependency files were changed.

One unchanged baseline test remains red:
`tests/gateway/test_session_policy.py::test_launch_policy_reaches_real_turn_runner`.
Its fixture reaches the real filesystem effect and then fails at
`session_policy_peer.py:117`, looking for `str(cwd)` inside `json.dumps(m)`.
Windows paths contain backslashes, which JSON escapes, making that literal
substring comparison invalid for ordinary native Windows paths. This is a
baseline test portability defect, not proof of a Phase 7 runtime regression.
No fixture or runtime fix is included in 7.1. Seven host/configuration skips
remain explicit; Linux-only discovery coverage is not claimed on Windows.

**Specification conflict to resolve before 7.4 acceptance:** an explicit empty
Feishu list currently recovers feishu_doc and feishu_drive through
_recover_platform_native_toolsets. The captured matrix proves this. Strict
"empty remains empty" and unconditional preservation of native recovery cannot
both hold for that case. Preserve this evidence and resolve the intended rule
explicitly during the policy migration; do not silently normalize the baseline.

## Next gates

7.2 can start from this branch without integrating Phase 5.8 or Phase 6.
7.7 must revisit the baseline failure, run host-appropriate plugin/profile,
approval and lifecycle checks, verify installed packages/startup and combine
actual adjacent-phase changes. Packaging and combined integration are not claimed
by this inventory step. Arcana was not needed: direct source inspection is the
acceptance evidence and must be repeated after migration.
