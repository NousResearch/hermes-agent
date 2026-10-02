# Phase 7.6 — ownership hard cut

Status: ownership hard cut complete on `refactor/phase7-commands-tools`, based on
`de4680cbe81b6995bb60dabdcce66f6fd9ac20cb` (completed 7.5).
Verified on Windows, Python 3.11.15, 2026-10-01 America/Los_Angeles.

## Canonical owners and final cleanup

The single built-in command registry and query protocol remain in `commands`;
the frozen context/reply contracts and six shared executors remain in
`commands.execution`. Runtime capability selection, restrictions and shared
expansion remain in `tools.platform_policy`, `tools.toolset_scope` and
`tools.toolset_selection`. Existing tool registration, toolset definitions and
model-facing schema preparation retain their owners.

The previous stages deleted `hermes_cli/slash_exec.py`,
`hermes_cli/toolset_scope.py` and `hermes_cli/commands_platforms.py`.
No forwarding files, duplicate registries or fallback paths were introduced.
The final direct audit found one remaining shared configuration helper:
`_parse_enabled_flag` in CLI presentation. Its implementation moved intact to
the application operation `hermes_cli.config_toolsets.parse_enabled_flag`.
Configuration migration and Desktop profile editing now import that owner.
Its bool/int/string/default behavior is preserved, including unknown values.
The helper has no manifest-listed external import obligation.

The broader gate also caught a test importing the retired CLI platform filter;
it now imports `tools.toolset_scope`. Obsolete partial boundary tests were
replaced by the checker contract suite. Contributor, Desktop and published
documentation now name the canonical command registry and test locations.

Application configuration still uses the existing backend, physically under
`hermes_cli` until the later CLI cutover. Provider menus, completion, localized
presentation and readiness display remain application responsibilities. The
public CLI entrypoint is preserved. Gateway authorization, execution approvals,
plugin/skill lifecycles, wire contracts and session policy are unchanged.

## Blocking ownership gate

`scripts/check_phase7_boundaries.py` is wired into the lint workflow as a
blocking check. Its 44 contract tests exercise direct and relative imports,
module aliases, supported dynamic loaders, chained aliases, tuple unpacking,
attribute lookup, invalid Python, restored forwarding files and duplicate
canonical command definitions. The repository audit includes first-party tests;
the named existing manifest-contract test may deliberately resolve compatibility
entries. Commands and runtime policy cannot import CLI/Gateway/Desktop/ACP
implementations or the model schema pipeline.

This is a static dependency lint, not a general Python data-flow interpreter
or an authorization mechanism. Dependency directories, documentation, skills,
Desktop application source and evaluation trees are excluded from Python
runtime scanning. The existing compatibility checker separately enforces
manifest-listed first-party import prohibitions.

## Named external obligations

The unchanged manifest retains exactly 36 entries on the two retained facades:

- `hermes_cli.commands`: 26 existing external entries: completion classes,
  platform menu functions and their supporting private/stdlib/typing names.
  The module contains only isolated external compatibility support; built-in
  definitions, query APIs and shared execution are absent. Menu targets use
  `gateway.command_platforms`; completion targets use CLI presentation.
- `hermes_cli.tools_config`: 10 existing presentation/provider/configuration
  entries: `MANAGED_FEATURE_COVERAGE_CATEGORY`, `NOUS_MANAGED_PROVIDER`,
  `base_url_hostname`, `fal_key_is_configured`,
  `format_nous_portal_entitlement_message`, `is_truthy_value`,
  `save_env_value`, `shutil`, `subprocess` and `sys`.

Exact entries remain in the compatibility manifest and fresh
`phase7.6-dependency-inventory.json`. No compatibility obligation covers the
retired registry, shared execution, runtime selection, platform scope or parser.
No new compatibility entry was added. Unrelated frozen old-updater interfaces
remain outside this migration.

## Verification and limits

Per-file results are recorded in `phase7.6-verification.json`.
The 16 ownership/configuration/command/policy/dispatch files passed:
287 tests passed, six skipped. A broader unchanged configuration file added
171 passes and two Windows failures, making the complete run
458 passed, two failed, six skipped across 17 files.

Both failures are
`TestEnvWriteDenylist.test_non_exec_near_misses_still_writable` for
`git_config_parameters` and `ld_preload`. The fixture assumes POSIX
case-sensitive environment names, while the existing Windows writer correctly
normalizes names to uppercase and denies them. The test file and
`hermes_cli/config.py` are byte-identical to the completed 7.5 commit.
This establishes unchanged-source failures; a separate baseline checkout run
was not performed. No security behavior or unrelated fixture was changed.

Standalone ownership, compatibility and configuration-writer audits passed.
Frozen evidence replay and public CLI help passed under an isolated temporary
home. The original 7.1 evidence remains unchanged; the 104 approved explicit-empty
corrections from 7.4 remain the only selection differences. A fresh source inventory
records references, including compatibility declarations and checker/test strings;
those textual references do not imply runtime use of retired implementations.

Phase 5.8 and Phase 6 worktrees remain independent and untouched. Installed
distribution, host-appropriate broader regressions and combined adjacent-phase
integration remain 7.7 work. Phase 7 as a whole is not yet accepted.
