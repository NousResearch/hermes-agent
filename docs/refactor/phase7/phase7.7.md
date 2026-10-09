# Phase 7.7 — integration and verification

Status: the Phase 7 ownership implementation is complete. Focused regressions,
package-content verification, frozen baseline replay, and combined Phase 5/6/7
integration checks are recorded below. A pre-existing Windows Gateway
file-effect failure prevents claiming an entirely green broad suite.

## Independent Phase 7 verification

- Phase 7.6's final 16-file ownership selection passed 287 tests, with six
  skips. Its two additional Windows configuration test failures were caused by
  POSIX case assumptions in an unchanged fixture.
- The corrected configuration tests now pass: **173 passed**.
- Model-facing tool-hook tests retargeted to the canonical plugin API:
  **23 passed**.
- The structural checker contract suite: **44 passed**. The ownership scanner
  is blocking in CI.
- Frozen baseline replay: all **102** command definitions and **2,100**
  tool-selection cases check successfully. The **104** approved explicit-empty
  corrections from Phase 7.4 are the only accepted changes to selection output.
- `git diff --check` passed for the test repairs.

## Package verification

The standalone Phase 7 wheel builds using the repository's documented Nix
build marker (`HERMES_NIX_BUILD=1`); unmarked wheel builds are deliberately
blocked by the repository. The resulting wheel has 2,192 members. After
extracting it and excluding the source checkout from Python's import path,
`commands`, `commands.execution`, `tools.platform_policy`,
`tools.toolset_scope`, and `plugin_runtime.api` import successfully.
The retired internal modules `hermes_cli/slash_exec.py`,
`hermes_cli/toolset_scope.py`, and
`hermes_cli/commands_platforms.py` are absent from the archive.
This exercises Python-wheel contents, not an OS installer or full Nix build.
The combined Phase 5/6/7 branch also builds with the Nix marker: its extracted
wheel contains 2,330 members. Eight essential auth/command/tool/route modules
are present; isolated imports and both Actual-provider and generic route
normalization checks pass. Retired Phase 7 internal CLI files remain absent.

## Combined Phase 5 / Phase 6 / Phase 7 verification

The independent Phase 7.6 hard cut and completed Phase 5 consumer branch were
combined, then Phase 6's completed auth branch was merged and reconciled in
`refactor/phase7-integration` at `144d5b0f`. The final reconciliation consumes provider aliases and
endpoint normalization
from the Phase 5 canonical provider owner. The restored static auth alias table
and duplicate CLI URL helpers have been removed; callers import providers
directly. MiniMax runtime dispatch also calls its canonical OAuth resolver.
Credential ownership remains in Phase 6's auth package.
The externally documented plugin import regression exercises real imports
rather than unavailable static-scanner helper functions.

Verification against the combined checkout:

- **65 passed** in `tests/auth`.
- **188 passed, one skipped** across command discovery/execution, CLI tooling,
  Gateway command discovery/authorization/toolsets, Desktop/TUI selection,
  ACP command dispatch and canonical tool-platform policy.
- Phase 7 structural ownership scanner: **passed** across 8,200 Python files.
- Plugin compatibility-pointer audit: **passed**; no first-party dependency
  on the 2,082 manifest entries (with UTF-8 console output).
- Syntax compilation of `auth`, `commands`, `tools`, and the merged route
  identity module: **passed**.
- Merged index: no unmerged entries; staged whitespace check passed.

## Existing regression limitation

`test_launch_policy_reaches_real_turn_runner` remains unreliable on this
Windows host. Its `policy-proof.txt` file can be absent even though the
terminal dispatch succeeds; that same missing-file assertion was reproduced
against the unchanged Phase 4 foundation under the same Python 3.11 runner.
The Phase 7 fixture now decodes actual terminal result payloads and compares
native Windows and MSYS working-directory representations instead of
searching JSON-escaped path strings. A canonical Python 3.14 run reached
that assertion for the CLI surface, but subsequent runs still exposed the
baseline file-effect failure, sometimes in a later surface.

Do not count that test as passing or attribute its baseline defect to this
ownership migration. The complete Desktop/TUI suite, OS-specific installer
validation, full Nix build, and the entire repository test suite are not
claimed here. Phase 6's independent closeout also documents its existing
Windows fixture failures and TUI-file timeout.

## Final recheck — 2026-10-02

The independent Phase 7 branch passed **582 tests across 41 files** on Python
3.14.7. The final combined branch passed **363 tests across 41 files**, with
two macOS-only skips. This includes both adjacent-phase provider ownership
guards, real plugin OAuth lifecycle and profile isolation, shared command
execution, platform policy, Gateway authorization and approval checks. A further **108 tests across three
files** passed for runtime provider resolution, late binding and profile-scoped
credential inputs, including the corrected MiniMax dispatch.

Both component wheels were also installed into fresh Python 3.14 virtual
environments and exercised outside the source checkout. Fourteen canonical
modules resolved from installed files; the six executor keys resolved; explicit
empty selection remained empty. The real installed console entrypoint and CLI,
tools and auth help passed. The PM-built test dependency set was reused through
a path-only .pth file, without processing its editable-project .pth files.
See the installed-package JSON receipts for wheel digests and exact inputs.

The final combined source audit passed across 8,200 Python files. The
compatibility-pointer audit passed for 2,082 manifest entries. The independent
Phase 5 and Phase 6 worktrees were not modified. Combined changes remain on
the separate refactor/phase7-integration branch; the final implementation
commit is `a75f79c78c`.

The previously documented Windows session-policy baseline defect remains open.
These focused green runs do not imply a green entire-repository suite, full
Desktop/TUI server run, full Nix distribution or signed installer verification.
