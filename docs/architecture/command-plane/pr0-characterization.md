# PR 0 — command-plane inventory and executable characterization

**Architecture root:** [#96692](https://github.com/NousResearch/hermes-agent/issues/96692)  
**Source pin:** `c30ac90a92097058ddd6f9db3fa2e3182a7bfdcc`  
**Observed:** `2026-08-28T11:01:05Z`  
**Scope:** inventory and characterization only; no production behavior changes.

## Result

Hermes no longer has the exact pre-#96791 command topology described in the first draft of #96692. [#96791](https://github.com/NousResearch/hermes-agent/pull/96791), by `OutThisLife`, merged the first material convergence step:

- `CommandDef` now carries Desktop argument and availability metadata;
- `commands.catalog` projects registry-backed built-in and plugin metadata;
- Desktop consumes that backend metadata instead of maintaining the same shared semantic rows locally;
- Desktop retains only bounded client-local actions, pickers, dedicated RPCs, and unavailable rendering.

That merged ownership is current product truth and must be composed, not duplicated.

The command plane is nevertheless not unified. The remaining defect is a **distributed semantic godfile**: command identity, dynamic contribution, catalog assembly, policy, execution, result interpretation, and native registration still have multiple owners spread across modules. No one file contains the entire control plane, but several files can still independently legislate the same executable intent.

The machine-readable inventory is [`pr0-inventory.json`](./pr0-inventory.json). The executable characterization is in [`test_command_plane_characterization.py`](../../../tests/conformance/test_command_plane_characterization.py).

## Current ownership map

| Authority | Current owner | Characterized responsibility | State |
|---|---|---|---|
| Core definition | `hermes_cli/commands.py` | names, aliases, descriptions, legacy arguments, availability flags, busy policy, shared execute key, Desktop metadata | canonical for core metadata |
| Plugin contribution | `hermes_cli/plugins.py` | dynamic command metadata, handlers, result conversion | parallel dynamic contributor |
| Skill contribution | `agent/skill_commands.py` | profile/project-scoped discovery and invocation preparation | parallel dynamic contributor |
| Runtime catalog | `tui_gateway/methods_tools.py` | `commands.catalog`, quick/plugin/skill/TUI contributions, `command.dispatch`, `slash.exec` | split authority |
| Ink client registry | `ui-tui/src/app/slash/registry.ts` | local command definitions, aliases, handler lookup | split authority |
| Desktop-local bindings | `apps/desktop/src/lib/desktop-slash-commands.ts` | local actions, pickers, RPCs, unavailable rendering | composed after #96791 |
| Gateway execution | `gateway/slash_commands.py` | normal command handlers and confirmation paths | split authority |
| Busy/cancellation policy | `gateway/run.py` | busy admission, interruption, mid-run routing, session routing | split authority |
| Native Discord projection | `plugins/platforms/discord/adapter.py` | application-command registration, options, interaction normalization | split authority |
| Relay Discord projection | `gateway/relay/command_manifest.py` | handwritten names, descriptions, and options | handwritten mirror |
| Dashboard projection | `web/src/lib/slashExec.ts` | web execution and result handling | split authority |
| Classic CLI handlers | `hermes_cli/cli_commands_mixin.py` | interactive execution and rendering | split authority |

## What is already singular

PR 0 proves the current core boundary rather than assuming the old one:

1. Core canonical names and aliases are case-insensitively collision-free.
2. Every core alias resolves to the **same `CommandDef` object** as its canonical name.
3. Desktop metadata is derived from that resolved `CommandDef`.
4. `CommandDef` currently has one explicit fourteen-field schema:
   `name`, `description`, `category`, `aliases`, `args_hint`, `subcommands`,
   `cli_only`, `gateway_only`, `gateway_config_gate`, `busy_policy`,
   `busy_handler`, `execute`, `argument_mode`, and `desktop`.
5. Every authority named by the inventory exists in the pinned source topology.

These are executable invariants, not prose claims.

## What remains non-singular

The current core registry still does not provide the complete command-plane ABI proposed by #96692. The remaining unique work is:

- stable semantic identity independent of spelling (`command_id`);
- one versioned, context-scoped catalog schema and deterministic revision;
- one collision/override authority for core, quick commands, plugins, skills, bundles, and client-local contributions;
- one typed invocation carrying actor, profile, cwd/project, session/channel/thread, source interaction, capability snapshot, and idempotency identity;
- one policy boundary for authorization, confirmation, mutation scope, busy state, live-session requirements, retry, and idempotency;
- one execution binding per shared command;
- one structured result/settlement ABI;
- generation-fenced execution that returns `catalog_stale` rather than guessing against a newer catalog;
- native and relay projections generated from the same semantic object;
- deletion of compatibility shims only after parity evidence exists.

The target is not a second registry beside `COMMAND_REGISTRY`. It is to extend the current owner into a narrow semantic core and make every other surface a contributor, binding, transport, or presentation projection.

## Migration order after PR 0

1. **Stable identity and schema.** Add `command_id` and the minimum versioned metadata needed by a shared catalog, directly on the current core owner. Preserve #96791’s fields and authorship.
2. **Context-scoped catalog assembly.** Compose core, quick, plugin, skill, bundle, and client-local contributions through one fail-closed collision boundary.
3. **Invocation and result ABI.** Add one normalized request/result contract and a generation check.
4. **Client adoption.** Move Ink and Desktop shared execution to the versioned catalog while retaining bounded local bindings.
5. **Gateway and platform projection.** Route normal/busy execution through one dispatcher; generate native Discord, relay Discord, Telegram, Slack, Matrix, and later surfaces from the same snapshot.
6. **Compatibility deletion.** Remove duplicate tables and routing shims only after exact-head cross-surface parity tests pass.

## Provenance and interlocks

- [#96791](https://github.com/NousResearch/hermes-agent/pull/96791), `OutThisLife`: merged current owner for registry-backed Desktop metadata and `commands.catalog` projection.
- [#96705](https://github.com/NousResearch/hermes-agent/pull/96705): closed predecessor that introduced the stable-ID/revision direction before #96791 changed the topology; it is provenance, not a landing object.
- [#93338](https://github.com/NousResearch/hermes-agent/pull/93338): Discord alias projection lineage.
- [#93501](https://github.com/NousResearch/hermes-agent/pull/93501) and [#94073](https://github.com/NousResearch/hermes-agent/pull/94073): Desktop presentation lineage consumed by the current topology.
- [#95028](https://github.com/NousResearch/hermes-agent/issues/95028) / [#95101](https://github.com/NousResearch/hermes-agent/pull/95101): related Authority Policy ABI; later command policy should compose with it rather than reconstruct it.

Open semantic interlocks were re-read before this PR 0 write: #96990, #96955, #96462, #50054, #95388, #96361, #66163, #96243, and #86508. Their changed-file sets do not overlap this PR’s three files. They remain merge-order and semantic watch items because they change command contribution, registration, or execution behavior that later slices must absorb.

## PR 0 acceptance

- [x] Pinned current-main source object.
- [x] Reconciled merged #96791 ownership.
- [x] Recorded every current semantic/contribution/execution/projection owner.
- [x] Added machine-readable current fields, missing shared fields, typed-outcome target, provenance, and interlocks.
- [x] Added executable alias identity, collision, Desktop projection, schema-boundary, path, and inventory tests.
- [x] Zero production-code change.
- [x] Zero FILE-LIST collision with the related open PR set.
- [x] Every changed file remains below 2,000 lines.
