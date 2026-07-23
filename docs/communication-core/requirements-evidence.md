# Requirement-to-evidence matrix

Status is based on code/tests as of 2026-07-23. `Done` means deterministic
implementation evidence exists; it does not imply production writes were
enabled. The focused host command is
`scripts/run_tests.sh tests/communication -q` (Python 3.13.14, 29 passed).
The final 44-file relevant suite passed on Windows with 855 passed and seven
explained platform skips, and on Linux with 862 passed and no skips.

| ID | Status | Implementation / migration | Primary evidence |
| --- | --- | --- | --- |
| COM-001 | Done | Canonical v1/v2 schema and invariants | `schema.py`; schema/isolation tests; [schema](schema.md) |
| COM-002 | Done | Adapter ABC, capabilities, orchestrator | `adapters.py`; contract-completion tests; [matrix](adapters.md) |
| COM-003 | Done | Secret-ref-only connected-account registry | repository/CLI; CLI redaction test; [configuration](configuration.md) |
| COM-004 | Done | Policy, exact approvals, durable outbox above adapters | repository/service; sync/routes/outbox tests |
| COM-005 | Done | Restricted PII redaction, IDs-only diagnostics, retention config | search/timeline E2E; [privacy](privacy-safety.md) |
| COM-006 | Done | Adapter contracts and temp-`HERMES_HOME` integration | all `tests/communication` |
| COM-007 | Done | Read-only, stable, reconciled Facebook migration | Facebook migration tests; [guide](facebook-migration.md) |
| COM-008 | Done | Person/identity/account/endpoint separated | schema triggers, person-detail and isolation tests |
| COM-009 | Done | Episode/transition/preference state machine | channel-state tests; [routing](isolation-routing.md) |
| COM-010 | Done | Account-scoped keys/FKs/triggers | schema/isolation and required-scenario tests |
| COM-011 | Done | 2 Facebook + 2 Telegram + VK route scenario | required multi-account tests |
| ACC-001 | Done | `accounts list/show/add/disable/status/capabilities` | CLI catalog/redaction test; [usage](usage.md) |
| ACC-002 | Done | Honest health/auth/re-auth state | adapter health tests; [runbook](operations.md) |
| ACC-003 | Done | Multiple same-provider namespaces | same-external-ID and five-account tests |
| ACC-004 | Done | Account-owned auth/profile/state/cursor/lock/rate-limit fields | schema; parallel/failure isolation tests |
| ACC-005 | Done | Stable namespace and scoped external keys | schema/isolation tests |
| ADP-FB-001 | Done | Existing Facebook repository wrapped read-only | Facebook adapter/migration tests |
| ADP-TG-001 | Done | Account-explicit injectable Telegram read connector; News separate | adapter contract tests; [matrix](adapters.md) |
| SYNC-001 | Done | Full/incremental idempotent sync | sync/routes/outbox tests |
| SYNC-002 | Done | Runs, redacted issues, status, bounded retry | partial-failure/retry test; [runbook](operations.md) |
| SYNC-003 | Done | Cross-account/contact rejection | schema triggers and cross-contact test |
| SYNC-004 | Done | Same IDs/cursors isolated across accounts | required scenarios and sync tests |
| CHN-001 | Done | Person journey over separate platform conversations | timeline/channel tests |
| CHN-002 | Done | Evidence-bearing transitions and episodes | channel-state and merge-depth tests |
| CHN-003 | Done | No failure fallback | route disable/fallback tests |
| CHN-004 | Done | Exact inbound endpoint resume | channel-state test |
| RTE-001 | Done | Directed default-deny account links | route tests |
| RTE-002 | Done | Per-person endpoint routes | five-account/four-person matrix |
| RTE-003 | Done | Exact account selection, no sibling fallback | adapter/service signatures and negative tests |
| RTE-004 | Done | Audited dry-run without send | route tests |
| RTE-005 | Done | Disable pauses endpoints/routes without move | route matrix and disable tests |
| REL-001 | Done | Evidence-backed commitments/questions | analysis/greeting tests |
| REL-002 | Done | Topic/tone evidence labeled non-diagnostic | analysis test |
| CRM-001 | Done | Explainable next-action recommendations | daily brief implementation/tests |
| CRM-002 | Done | Daily relationship brief | CLI catalog and analysis/brief tests |
| TIM-001 | Done | Provenance timeline with person/endpoint/time filters | contract-completion timeline E2E |
| TIM-002 | Done | Episodes/transitions in same filtered view | timeline and merge-depth E2E |
| SRCH-001 | Done | People/identity/conversation/message/event search | restricted-PII search E2E |
| GRP-001 | Done | Explicit groups and hashed preview | group/segment preview E2E |
| GRP-002 | Done | Explainable smart-segment membership | group/segment preview E2E |
| ID-001 | Done | Explainable candidates and manual reversible merge | merge/unmerge tests |
| DRF-001 | Done | Draft from exact selected conversation route | outbox tests; CLI catalog |
| DRF-002 | Done | List/show/cancel draft queue | repository and CLI catalog tests |
| DRF-003 | Done | Draft binds current exact endpoint/route/recipient preview | outbox and frozen-preview tests |
| DRF-004 | Done | Payload/person/route/target mutations invalidate approval | trigger and approval tests |
| APR-001 | Done | Actor/person/accounts/endpoints/recipient/payload/TTL binding | exact approval tests |
| OUT-001 | Done | Atomic claim, uncertain expiry, verified sent | outbox tests; [privacy](privacy-safety.md) |
| GRT-001 | Done | Person-level contact events and greeting plan | greeting test |
| GRT-002 | Done | Timezone and per-person/event/date dedup | greeting test |
| GRT-003 | Done | Exclusion groups plus immutable previews | greeting and frozen-preview tests |
| ADP-VK-001 | Done | Same connector contract with synthetic/test fixtures; live connector honestly unconfigured | adapter contract tests; [matrix](adapters.md) |
| DATE-001 | Blocked | No pilot may be guessed | Needs user-named dating site |
| DATE-002 | Blocked | Generic/unnamed adapter construction is rejected | Needs confirmed provider, test account, isolated profile |
| DATE-003 | Blocked | Core analysis/draft path exists but no approved pilot connector | Needs read-only pilot fixture/E2E |
| DATE-004 | Blocked | Core forbids send/manipulation/mass automation, but pilot policy cannot be evidenced | Needs named pilot capability/policy review |
| DATE-005 | Blocked | Live smoke intentionally not run | Needs explicit safe test account/profile permission |
| NEWS-P1-001 | Done | Evolving story timelines | News/XDOM test |
| NEWS-P1-002 | Done | Claims and contradiction matrix | News/XDOM test |
| NEWS-P1-003 | Done | Explainable reliability history | News/XDOM test |
| NEWS-P2-001 | Done | Topic/entity/geography watchlists | News/XDOM test |
| NEWS-P2-002 | Done | Multi-source breaking confirmation | News/XDOM test |
| NEWS-P2-003 | Done | Normalization/translation records | News/XDOM test |
| NEWS-P3-001 | Done | Explainable select/dedup/reject decisions | News/XDOM test |
| NEWS-P3-002 | Done | Source health, quarantine, recovery | News/XDOM test |
| NEWS-P3-003 | Done | Daily/weekly digest and archive | News/XDOM test |
| XDOM-001 | Done | Only public topic/entity/story/source refs accepted | XDOM privacy test |
| XDOM-002 | Done | Private source-backed conversation suggestion | XDOM test |
| XDOM-003 | Done | News-derived output is draft-only | XDOM draft test |

The dating rows keep the overall verdict **NO-GO** until the named pilot and
safe test account are supplied and its read-only contract/smoke pass.
