# Phase 7.3 — shared command execution

Status: exit gate passed; the broader Windows regression suite retains four confirmed baseline failures.
Branch: `refactor/phase7-commands-tools`.
Foundation: Phase 7.2 commit `a50d2b74a5e3430c5519fc7ab26f537824003ee8`.
Verified 2026-10-01 on Windows, Python 3.11.15.

## Ownership cut

`commands/execution.py` now owns the existing frozen `CommandContext` and `CommandReply`,
the single six-entry `EXECUTORS` mapping, `resolve_executor`, `run_execute`, and
`execute_command`. The execution keys and implementation names are unchanged.
No second registry or generalized Gateway handler framework was introduced.

The old `hermes_cli/slash_exec.py` is deleted. The Phase 7.1 compatibility audit
established no external obligation for that module, so it has no forwarding shim.
Every first-party caller and execution test now uses the canonical owner.
Desktop's existing CLI worker reaches the migrated CLI handlers; ACP retains its
existing curated dispatch. No RPC or cross-process reply shapes changed.

The data contracts remain exactly:

- `CommandContext`: surface, args, options, config_get.
- `CommandReply`: text, data, format.

Both dataclasses remain frozen. Surface-independent text, structured result data,
format hints, lookup errors, filtering, collision notes and pagination are preserved.

## Application inputs

The existing `options` mapping carries application-provided inputs. There are no
new context fields or configuration loaders in the shared execution layer.

| Executor | Supplied input | Application boundary |
|---|---|---|
| version | lazy version_label callable | Gateway supplies the existing banner label function |
| egress | lazy egress_status callable | CLI supplies the existing proxy status function |
| profile | profile_name, home_display, optional profile_label | CLI/Gateway resolve through profiles.profile_command_details |
| gateway_help / gateway_commands | translate and help_lines callables; existing allowed_commands/page_size | Gateway supplies its localization and catalog presentation |
| bundles | existing agent.skill_bundles API | Canonical executor consumes the existing subsystem |

Callers invoking the canonical version, egress, profile or catalog executor must
supply the corresponding inputs. Configuration loading, profile metadata reads,
localized catalog presentation and terminal formatting remain application concerns.
The small profile input operation lives beside existing profile display helpers;
it preserves the original active-profile fallback and metadata-error behavior.
It does not move profile lifecycle ownership.

Skill discovery, bundle discovery and collision handling still use their existing
subsystem APIs and scopes. Invalid page arguments still return usage before catalog
discovery. Restricted help still suppresses skill discovery. Collision warnings in
/commands retain their preexisting behavior, including restricted catalogs.

Gateway slash authorization, approval enforcement, session mutation and platform
mention decoration remain at their existing boundaries. Discovery grants no authority.

## Verification

All regression tests ran through `scripts/run_tests.sh -j 4` with the existing
credential-clean, per-file isolated runner.

- Canonical execution, Gateway discovery/access/approval, skills/bundles, collisions
  and Desktop worker regressions: **118 passed, 0 failed, 2 skipped**, 16 files.
- CLI presentation, banner labels, proxy status, profiles, plugin disablement,
  Desktop worker path and ACP commands: **171 passed, 4 failed, 4 skipped**, 10 files.
- Distinct total: **289 passed, 4 confirmed baseline failures, 6 skipped**, 26 files.
- All six executors execute with CLI imports blocked and produce equal replies on
  CLI, Gateway, TUI and ACP contexts with fixed supplied inputs.
- Execution keys exactly match the registry's registered execution keys, with one
  distinct implementation per key.
- `scripts/phase7_execution_parity.py` compares against the committed Phase 7.2
  implementation: **180 identical replies**, covering four surfaces, profile
  A/B/A changes, filtered catalogs, skills/collisions, invalid arguments, page sizes,
  navigation and clamping. It uses deterministic application/subsystem fixtures and
  records no installed legacy implementation.
- `scripts/phase7_baseline.py check`: **102 definitions and 2,100 tool selections identical**.
- `scripts/check_compat_pointers.py`: no dependency on **2,084** protected entries.
- Direct source search and AST inspection: zero imports of the retired execution
  module across seven referencing tracked files. The remaining old dotted name is
  the historical target list in the inventory aid.
- Structural tests forbid CLI/Gateway implementation imports in the canonical
  execution module and require the old file to be absent.
- Existing setuptools package configuration selects `commands.execution`;
  `build_py.find_package_modules` confirms inclusion. Installed-artifact/startup
  verification remains the Phase 7.7 gate.
- `git diff --check` passed.

The package-selection probe initially lacked setuptools' script_name setup; the
corrected probe passed. This was a verification harness error, not a package failure.

## Confirmed baseline failures

A temporary detached worktree at the exact Phase 7.2 commit reproduced all four
failing profile tests through the same test runner; 103 other tests were deselected.
The temporary worktree was removed after collecting the result.

| Test in tests/hermes_cli/test_profiles.py | Windows failure |
|---|---|
| TestCreateProfile.test_seeds_placeholder_env_file | POSIX 0600 assertion observes Windows 0666 |
| TestBackfillProfileEnvs.test_copies_default_env_into_envless_profiles | same POSIX mode assertion |
| TestWrapperScript.test_creates_sh_on_posix | expects a suffix-free shell wrapper; Windows returns .bat |
| TestFindAliasForProfile.test_list_profiles_surfaces_custom_alias | same wrapper filename expectation |

These tests and the profile lifecycle implementation were not changed to hide the
failures. Linux-only and other host skips remain explicit. The earlier 7.1 Windows
session-policy fixture failure remains recorded and was not rerun in this step.

## Remaining gates

Phase 7.4 tool capability policy has not started. The Feishu empty-list baseline
conflict remains for that step. Complete ownership enforcement, installed startup
and combined adjacent-phase integration remain for 7.6/7.7. No Phase 5.8 or Phase 6
implementation was merged or modified.
