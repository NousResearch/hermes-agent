# Phase 5.9 — Deletion and boundary closeout

Phase 5 is closed on refactor/phase5-8-runtime-consumers. Credential acquisition,
OAuth refresh, pools and secret persistence remain Phase 6.

## Ownership result

- Canonical provider identity, configured matching, endpoint identity and route
  semantics remain under providers/.
- Model catalogue interpretation, aliases/deduplication, generation classification,
  billing facts and selection policy have one owner under models/.
- Application detection acquires facts and calls the canonical selector; the
  duplicate detection ladder and static-only selector are deleted.
- The current-provider live catalogue/native-vendor guard keeps its priority over
  reseller listings. No new choice or wire-selection policy was introduced.
- CLI acquisition, probe evidence, messages, persistence, locks, runtime application
  and rollback remain downstream. Pricing acquisition is shared application code.
- Pricing observations and negative retries use home/endpoint/credential scopes.
  Opaque generation observations are home-scoped. A → B → A checks pass.
- The shipped updater hook and existing external plugin names remain exact;
  compatibility targets point to final owners and internal use is prohibited.

The final inventory reconciles all 78 historical rows representing 90 references,
249 application definitions and every production Python file. Definitions have
explicit delete/consolidate/retain dispositions and final import consumers.
PHASE5_9_DELETION_MANIFEST.md and providers/README.md record retained exceptions
and distinct cache identities, refresh windows and stale behavior.

## Verification

All behavior suites used scripts/run_tests.sh with per-file subprocess isolation
and no retries.

| Check | Result |
| --- | --- |
| Broad integration: models/providers, ACP, CLI switching, gateway sessions, TUI profile/resume paths, provider/media plugins | 249 files; 2,338 passed, 14 initial failures, 18 skipped |
| Seven corrected broad-run cases: missing provider-prefix import, scoped credits fixtures, stale DeepSeek auxiliary import | 8-file recheck: 132 passed, zero failures |
| Remaining broad-run failures | Seven exact cases reproduced on unchanged Phase 5.8.8 baseline |
| TUI server model/profile/resume/context subset | 102 passed; three exact failures reproduced on baseline |
| Original ten-file representative regression set | 545 passed; same 24 inherited failures: 23 Actual/ACI cases and one short-alias expectation |
| Validation/import-gate recovery | 114 passed, zero failures |
| Whole-repository Python compile | 8,450 files, zero errors |
| Ruff across changed/new Python files | 88 files, passed |
| All-tree compatibility dependency lint | No internal use of 2,082 external pointers |
| Source ownership gates | Lower-domain direction, single semantic definitions, deleted modules, static/literal-dynamic imports and exact pointer targets passed |

The broad run occurred before the seven repairs. Only their affected suites and
ownership gates were rerun afterward; the table does not claim a second broad run.
Exact failure IDs and their baseline comparison are versioned in
phase5_9_inventory/final_integration.json. No newly introduced failure remains.

Inherited cases are not waived as new behavior: the native-Windows ACP symlink
cases, global compatibility sibling assertion, Copilot same-provider API mode,
short-alias expectation, CommandCode transport assertion, and bare-custom TUI
persistence assertion all reproduce on the unchanged entry baseline. The three
TUI server picker failures also reproduce exactly. Actual/ACI remains the
previously recorded transition/setup/key-reload baseline.

Fresh Lexicon/Arcana structural verification is supplementary; authoritative
source ownership gates and behavior comparisons determine the closeout.

The bounded Lexicon refresh command failed, so no fresh Arcana snapshot was
claimed. This tooling-state limitation does not weaken the all-production AST
import/definition gates or the exact baseline behavior comparisons.
