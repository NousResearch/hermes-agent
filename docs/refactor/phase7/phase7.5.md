# Phase 7.5 — consumer migration

Status: consumer migration complete on `refactor/phase7-commands-tools`, based on
`e08d42e1abd3cd61d1bb73b8f6171e4b5924c33f` (completed 7.4). Verified on Windows,
Python 3.11.15, 2026-10-01 America/Los_Angeles.

## Final owners and consumers

Earlier stages already migrated canonical command lookup/shared execution and
runtime policy reads. This stage audits those paths and completes the remaining
application and Gateway responsibilities.

| Responsibility | Canonical owner | Direct consumers |
|---|---|---|
| Definitions, aliases, metadata, availability | `commands` | Gateway, Desktop/TUI, ACP, CLI and skill collision checks |
| Shared reply/context contracts and six executors | `commands.execution` | Gateway, Desktop slash worker, CLI |
| Runtime tool selection and platform restrictions | `tools.platform_policy`, `tools.toolset_scope` | Gateway, Desktop/TUI, ACP, CLI, agent and cron |
| Shared tool-setting writes | `hermes_cli.config_toolsets` | CLI configuration, HTTP toolsets, Desktop tools.configure and profiles.configure |
| Telegram/Discord/Slack command-menu projections | `gateway.command_platforms` | Gateway runner/relay, platform adapters, Telegram inline picker and Slack CLI |

`config_toolsets` is an application operation beside the existing configuration
backend, which remains physically under `hermes_cli` until the later CLI cutover.
It uses the existing writer; runtime policy still neither loads nor persists
configuration. It preserves platform filtering, MCP entries, known plugin/builtin
records, global-disable reconciliation and MCP exclude-list mutation. The shared
configuration-section helper moved once, with provider menus consuming it directly.

The Gateway projections moved intact. Lazy plugin discovery still calls Phase 4's
published APIs. Menu ordering, collision resolution, truncation, scope and platform
filters retain their existing behavior. CLI completions and terminal presentation,
provider/setup menus and readiness display stay at their application boundaries.

Desktop RPC imports remain local where handler functions are rebound onto server
globals. Wire contracts, profile wrappers, session reset behavior, access control,
execution approval and deferred prompt-cache policy are unchanged. Empty profile
editor selections still clear the existing pin; explicit saved empty lists still
follow the approved 7.4 runtime rule.

## Hard cuts and external compatibility

Removed `hermes_cli/commands_platforms.py`; no forwarding module was added.
Removed the shared write implementations and their old private re-exports from
`tools_config`/`tools_config_mcp`. Consumers and tests call the final owners.

The seven already audited lazy `hermes_cli.commands` menu entries now target
`gateway.command_platforms`, with both compatibility manifests updated.
No new compatibility entry or internal fallback was introduced. Existing updater
and other externally required compatibility entries remain isolated.

The extended structural checks reject retired settings imports/definitions and
the old platform-command module. A fresh direct source inventory is recorded in
`phase7.5-dependency-inventory.json`; the 7.1 snapshots remain unchanged.
Remaining `tools_config` references concern CLI/application presentation,
provider configuration, readiness, menus or the isolated compatibility contract.

## Verification and limits

Full per-file results and focused Desktop selectors are in
`phase7.5-verification.json`. All completed final targeted checks pass.
The frozen evidence check covers 102 commands and 2,100 selection cases, retaining
the 104 previously approved explicit-empty corrections. Public CLI help and the
compatibility/config-writer audits pass.

The full Desktop server suite was not run; focused tool/configuration and
catalog/completion/dispatch selections were used. The Linux-only ordinary-daemon
discovery test is skipped on Windows. Existing unrelated baseline failures
documented in 7.3/7.4 were not reclassified or repaired here.

Phase 5.8 and Phase 6 worktrees remain independent and untouched. Installed
distribution verification and combined adjacent-phase integration remain 7.7
work, and broader ownership enforcement remains the 7.6 gate.
