# Hermes Communication Core completion report

Date: 2026-07-23
Repository baseline: `9de9c25f620ff7f1ce0fd5457d596052d5159596`
(`v2026.7.7.2-dirty`)
Hermes package: `0.18.2`
Final verdict: **NO-GO**

## Decision summary

All work that is independent of a real dating pilot is implemented and has
passing Windows and Linux evidence. The only open mandatory requirements are
`DATE-001` through `DATE-005`. They cannot be completed safely until the user:

1. selects one named dating provider;
2. explicitly confirms that provider for the pilot; and
3. supplies or identifies a safe read-only test account and isolated browser
   profile that may be used for the bounded pilot.

No provider or account was guessed. The external blocker therefore keeps the
goal active and the verdict at **NO-GO**. P5 production write is deliberately
outside this goal and does not affect this verdict.

No real Facebook/Messenger/VK/dating message, Telegram publication, or other
external write was performed. Facebook writes, production Communication Core
workers, and the test sink all remain disabled by default.

## Baseline and implementation boundary

- Windows host: Python `3.13.14`, isolated venv
  `temp/communication-core-py313`.
- Linux test/production images: Python `3.13.5`, Docker Engine `29.6.1`.
- Test image:
  `hermes-communication-core-test:20260723`
  (`sha256:579f53c2af31e7e40d06ca504cf8b0d97f20f65c3fc7ac616ef88004151a6f69`).
- Production image:
  `hermes-communication-core-production:20260723`
  (`sha256:6c1cc7ead5dbc0617768a96ca8e5d216d2a83a85c33dc49ed28b18a93a7edf94`).
- The pre-existing dirty worktree and unrelated user changes were preserved.
- No core model tool or permanent tool-schema entry was added. The capability
  is exposed through `hermes communication` and `$manage-communications`.

## Requirement-to-evidence matrix

`Done` means implementation plus deterministic evidence exists; it never means
production sending was enabled. Detailed links and test ownership are also in
[the maintained evidence matrix](docs/communication-core/requirements-evidence.md).

| ID | Status | Evidence |
| --- | --- | --- |
| COM-001 | Done | Versioned v1/v2 schema, atomic migrations and rollback; schema/isolation tests |
| COM-002 | Done | Adapter ABC, capability declarations and orchestrator; adapter contract tests |
| COM-003 | Done | Secret-reference-only account registry and redacted CLI output |
| COM-004 | Done | Core-owned exact approval and durable outbox above adapters |
| COM-005 | Done | Restricted-PII redaction, retention config and IDs-only diagnostics |
| COM-006 | Done | Temp-home integration and all 29 Communication Core tests |
| COM-007 | Done | Read-only, stable and reconciled Facebook migration plus rollback |
| COM-008 | Done | Person/identity/account/endpoint separation in schema, repository and CLI |
| COM-009 | Done | Episode/transition/preference state machine and inbound resume |
| COM-010 | Done | Account-scoped keys, FKs and cross-account rejection triggers |
| COM-011 | Done | Required 2 Facebook + 2 Telegram + VK route scenario |
| ACC-001 | Done | Account list/show/add/disable/status/capabilities CLI |
| ACC-002 | Done | Honest auth/health/re-auth states and failed unconfigured connectors |
| ACC-003 | Done | Multiple isolated namespaces for one provider |
| ACC-004 | Done | Account-owned auth/profile/cursor/lock/rate-limit/sync state |
| ACC-005 | Done | Stable namespaces and account-scoped external keys |
| ADP-FB-001 | Done | Existing Facebook repository wrapped as a read-only adapter |
| ADP-TG-001 | Done | Account-explicit Telegram connector; Telegram News remains separate |
| SYNC-001 | Done | Idempotent full and incremental sync |
| SYNC-002 | Done | Runs, redacted issues, status, bounded retry and partial failure |
| SYNC-003 | Done | Cross-account/contact ownership rejection |
| SYNC-004 | Done | Same external IDs and cursors isolated across accounts |
| CHN-001 | Done | Person journey above physically separate platform conversations |
| CHN-002 | Done | Evidence-bearing episodes and transitions |
| CHN-003 | Done | Channel/account failure never triggers delivery fallback |
| CHN-004 | Done | Exact inbound endpoint resumes only that endpoint |
| RTE-001 | Done | Directed default-deny account-link policy |
| RTE-002 | Done | Per-person exact endpoint routes |
| RTE-003 | Done | Explicit account arguments and no sibling-account fallback |
| RTE-004 | Done | Audited route dry-run without send |
| RTE-005 | Done | Disable pauses endpoints/routes without moving data |
| REL-001 | Done | Evidence-backed commitments and unanswered questions |
| REL-002 | Done | Topic/tone evidence labeled non-diagnostic |
| CRM-001 | Done | Explainable next-action recommendations |
| CRM-002 | Done | Daily relationship brief |
| TIM-001 | Done | Provenance timeline with person/endpoint/time filters |
| TIM-002 | Done | Episodes and transitions in the same filtered timeline |
| SRCH-001 | Done | Search over people, identities, conversations, messages and events |
| GRP-001 | Done | Explicit groups and immutable hashed recipient preview |
| GRP-002 | Done | Explainable smart-segment membership and frozen preview |
| ID-001 | Done | Explainable duplicate candidates and reversible manual merge |
| DRF-001 | Done | Draft created from an exact selected route |
| DRF-002 | Done | Draft list/show/cancel queue |
| DRF-003 | Done | Draft binds exact endpoint, route and recipient preview |
| DRF-004 | Done | Payload/person/route/target mutation invalidates approval |
| APR-001 | Done | Actor/person/accounts/endpoints/recipient/payload/TTL binding |
| OUT-001 | Done | Atomic claim, uncertain expiry and verified completion |
| GRT-001 | Done | Person-level contact events and greeting planning |
| GRT-002 | Done | Timezone and person/event/date deduplication |
| GRT-003 | Done | Exclusions and immutable greeting recipient previews |
| ADP-VK-001 | Done | Account-scoped VK connector contract; unconfigured live connector fails closed |
| DATE-001 | **Blocked** | Requires the user to name the pilot provider |
| DATE-002 | **Blocked** | Requires confirmed provider, safe test account and isolated profile |
| DATE-003 | **Blocked** | Requires a read-only pilot connector/fixture E2E |
| DATE-004 | **Blocked** | Requires provider-specific capability and policy review |
| DATE-005 | **Blocked** | Requires explicit permission for a bounded read-only smoke |
| NEWS-P1-001 | Done | Evolving story timelines |
| NEWS-P1-002 | Done | Claim extraction and contradiction matrix |
| NEWS-P1-003 | Done | Explainable historical source reliability |
| NEWS-P2-001 | Done | Topic/entity/geography watchlists |
| NEWS-P2-002 | Done | Multi-source breaking confirmation |
| NEWS-P2-003 | Done | Normalization and translation records |
| NEWS-P3-001 | Done | Explainable selection/dedup/reject decisions |
| NEWS-P3-002 | Done | Source health, quarantine and recovery |
| NEWS-P3-003 | Done | Daily/weekly digest and archive |
| XDOM-001 | Done | Only public topic/entity/story/source references enter News |
| XDOM-002 | Done | Private source-backed conversation suggestions |
| XDOM-003 | Done | News-derived output remains draft-only |

The maintained backlog has 63 completed tasks and exactly five open dating
tasks. Telegram News P1–P3 and XDOM evidence is also recorded in
[`TG_news/backlog.md`](TG_news/backlog.md); production publication remains off.

## Final ownership and schema model

```text
ConnectedAccount (our exact authenticated namespace)
  ├─ credentials/session/browser refs
  ├─ cursor, lock, rate-limit and health state
  └─ ContactEndpoint ── PlatformIdentity ── Person
                            │                  ├─ CRM/groups/events/commitments
                            │                  └─ CommunicationJourney
                            └─ raw account-scoped conversation/message provenance

AccountLinkPolicy (directed, default deny)
  + PersonChannelRoute (person-specific exact endpoints)
  → route dry-run → draft → exact approval → durable outbox → fake sink only
```

`Person` owns cross-platform relationship meaning. `PlatformIdentity` owns one
provider identity. `ConnectedAccount` owns the user's exact authentication and
namespace. `ContactEndpoint` is the exact account/identity pair. Raw platform
records never move between account namespaces when people are merged.

SQLite FKs and triggers reject cross-account conversations, participants,
messages, routes, drafts, approvals and outbox references. Migrations run
atomically and are reversible. Missing/read-only paths do not initialize data.
See [architecture](docs/communication-core/architecture.md) and
[schema](docs/communication-core/schema.md).

## Account isolation and route evidence

The required E2E route matrix creates two Facebook accounts, two Telegram
accounts and one VK account. It proves:

- identical provider external IDs remain distinct across namespaces;
- parallel sync cursors and failures are isolated;
- two people reached through the same source Facebook account route to
  different Telegram accounts/endpoints;
- changing one person's route does not affect another person's route;
- a link is unusable until its exact direction is allowlisted;
- disabling or rate-limiting a target does not select its sibling account;
- return-by-request and inbound resume require exact evidence;
- dry-run explains allow/deny state without sending; and
- route, account, endpoint, recipient or payload mutation invalidates approval.

See [account isolation and routing](docs/communication-core/isolation-routing.md).

## Adapter capability matrix

| Adapter | Read contract | Profile/group/receipt behavior | Write behavior |
| --- | --- | --- | --- |
| Facebook local CRM bridge | contacts, conversations, messages, events | profiles supported; groups/receipts explicitly unsupported | forbidden |
| Telegram Communication, configured | connector-declared, exact account required | connector-declared | forbidden |
| Telegram, unconfigured | failed health, zero capabilities | unsupported | forbidden |
| VK, configured/test fixture | connector-declared, exact account required | connector-declared | forbidden |
| VK, unconfigured | failed health, zero capabilities | unsupported | forbidden |
| Dating pilot | blocked pending named provider/account/profile | blocked | forbidden |
| Fake adapter | deterministic full read fixture | profile/group/receipt fixtures | in-memory fake sink only |

Unsupported operations raise explicitly; they are never replaced by a generic
browser guess. See [the full adapter reference](docs/communication-core/adapters.md).

## Facebook migration and rollback evidence

The migration bridge opens the source CRM through SQLite `mode=ro`, hashes the
source, maps stable IDs with the exact connected account in their derivation,
and reconciles source/destination counts. Replaying the same account/source
hash is idempotent. Importing the same external IDs to a second Facebook
account creates separate identities.

Messages and events retain provenance. Legacy settings, campaigns, birthdays,
approvals and outbox records are archived inertly in `legacy_records`; they do
not become active Core approvals or outbox work. Rollback deletes only rows
owned by the selected migration run, leaves other accounts intact, and a later
re-import restores the same stable IDs. The source hash and
`write_actions_enabled=0` remain unchanged.

CLI entry points:

```text
hermes communication migration facebook-import ...
hermes communication migration facebook-rollback <run-id>
```

See [migration](docs/communication-core/facebook-migration.md) and
[rollback](docs/communication-core/rollback.md).

## Legacy cutover inventory

No legacy file was deleted.

| Legacy family | Disposition | Canonical replacement |
| --- | --- | --- |
| Facebook local lookup/CRM helpers | retained read-only; skill consumer migrated | `people search/show`, timeline, Facebook adapter |
| Message/history dump and inspection scripts | retained for forensics | Core search/timeline/analyze |
| Browser sync/inbox/profile/timeline jobs | browser ownership retained; direct skill use retired | Facebook application service plus Core sync |
| Birthday and resend scripts | direct sender paths retired | greetings plan plus reviewed drafts |
| Dialogue/outreach campaigns | manipulative/autonomous path retired | explainable analysis/brief and reviewed drafts |
| Send/queue scripts | prohibited; never used as rollback | no production replacement in this goal |
| Diagnostics/export monitors | retained where read-only | account health, sync status/issues and reconciliation |

`skills/social-media/facebook/SKILL.md` is now a thin
`$manage-communications` shim. `skills/dialogue_campaigns/SKILL.md` is an
explicit retirement shim. Static tests reject direct SQL, Docker/direct sender
instructions and generic-browser escape hatches in these skill consumers.

## Skill validation

- Hermes metadata normalization check: `normalized 0 skill metadata file(s)`.
- Repository scanner: `SAFE`, `ALLOWED`.
- All skill discovery/guard/sync/tool/review tests in the final host and Linux
  suites passed.
- `agents/openai.yaml` supplies the display name, short description and default
  `$manage-communications` prompt.
- The generic Codex `quick_validate.py` was executed and reported that
  top-level `category` and `prerequisites` are unknown. Hermes deliberately
  requires these extended fields for its readiness/activation matrix, and its
  tests assert that contract. Removing them merely to satisfy the generic
  validator would make the Hermes artifact invalid, so the repository-native
  contract was preserved. This is a validator-schema incompatibility, not an
  ignored Hermes validation failure.

## Windows host verification

Canonical command (Git Bash with the explicit Python 3.13 venv):

```bash
scripts/run_tests.sh \
  tests/communication \
  tests/facebook \
  tests/news \
  tests/skills \
  tests/hermes_cli/test_config.py \
  tests/hermes_cli/test_browser_pool_config.py \
  tests/tools/test_skill_review_policy.py \
  tests/tools/test_skills_guard.py \
  tests/tools/test_skills_sync.py \
  tests/tools/test_skills_tool.py \
  tests/tools/test_mcp_stability.py \
  -q -rs
```

Result: **44 files, 855 passed, 0 failed, 7 skipped, exit 0, 28.2 s**.
The runner summary counts passed tests separately; the exact collected outcome
is 855 passed plus these seven platform skips:

| File/test | Count | Reason |
| --- | ---: | --- |
| `test_skills_guard.py`: bind-mount text execute bits; unknown executable payload | 2 | Windows does not preserve POSIX execute bits |
| `test_mcp_stability.py`: killpg tracked; own-pgroup guard; killpg failure; live grandchild | 4 | `os.killpg`/`getpgrp`/`setsid` are POSIX-only |
| `test_config.py`: corrupt symlink not backed up | 1 | Test symlink creation requires privileges on Windows |

The focused `tests/communication` total is 29 passed.

## Linux Docker verification

Build:

```bash
docker build --target test \
  --tag hermes-communication-core-test:20260723 .
```

Final run used the same 44 paths and pytest flags as the host run:

```bash
docker run --init --rm hermes-communication-core-test:20260723 \
  tests/communication tests/facebook tests/news tests/skills \
  tests/hermes_cli/test_config.py \
  tests/hermes_cli/test_browser_pool_config.py \
  tests/tools/test_skill_review_policy.py \
  tests/tools/test_skills_guard.py \
  tests/tools/test_skills_sync.py \
  tests/tools/test_skills_tool.py \
  tests/tools/test_mcp_stability.py -q -rs
```

Result: **44 files, 862 passed, 0 failed, 0 skipped, exit 0, 23.5 s**.
All seven Windows-skipped tests therefore executed on Linux. `--init` matches
the production init-reaper contract (production uses s6-overlay) for the real
MCP process-group/grandchild E2E test.

## Production image verification

Build:

```bash
docker build --target production \
  --tag hermes-communication-core-production:20260723 .
```

Inspection and startup evidence:

- `pytest_installed=false`;
- `/opt/hermes/tests` absent;
- `/opt/hermes/website/docs` absent;
- no project `test_*.py` files outside the venv;
- `communication.outbox_workers_enabled=false`;
- `communication.test_sink_enabled=false`;
- fresh `facebook_settings.write_actions_enabled=0`;
- `docker run --rm ... --version` completed with exit 0 under s6 and reported
  Hermes `0.18.2`, Python `3.13.5`;
- no Communication outbox/write worker is registered as a production service.

## Controlled local smoke

The bounded smoke ran only inside the production image with
`HERMES_HOME=/tmp/hermes-communication-smoke`:

```text
hermes communication init
hermes communication accounts add \
  --provider fake --namespace synthetic-smoke --label synthetic-smoke \
  --owner-profile test --write-policy disabled
hermes communication accounts list
```

Observed result: schema version 2 was initialized, one synthetic fake account
was registered and listed with `write_policy=disabled`. The container and its
temporary filesystem were removed. No credential, browser, network, external
ID, real message, approval consumption or outbox delivery was involved. No
test/production image container remained afterward.

Browser/network smoke for Facebook, Telegram Communication, VK and dating was
correctly not run because named safe targets were not supplied. Deterministic
fixtures cover their permitted contracts without crossing that safety gate.

## Production-write statement

Throughout implementation and verification:

- `facebook_settings.write_actions_enabled` stayed `0`;
- `communication.outbox_workers_enabled` stayed `false`;
- `communication.test_sink_enabled` stayed `false` in production defaults;
- fake-sink execution occurred only in deterministic tests;
- no real Facebook/Messenger/VK/dating action occurred;
- no Telegram production publication occurred; and
- P5 production write was not implemented or exercised.

## Residual risks and blocker

The remaining product risk is intentionally explicit: a generic dating adapter
cannot establish provider-specific authentication, policy, rate-limit,
capability and read-only behavior. The master goal forbids guessing those
facts. Until a named pilot and safe test target exist, the adapter must remain
blocked and `DATE-001`–`DATE-005` must remain open.

Telegram and VK connectors without a configured implementation also report
failed health and zero capabilities. This is fail-closed behavior, not a
successful empty sync.

Minimum user decision needed to continue to GO:

- the exact dating site/provider name;
- explicit confirmation that Hermes may use it for this read-only pilot; and
- identification of the safe test account and isolated browser profile, with
  permission for a bounded read-only smoke.

## Documentation index

- [Communication Core overview](docs/communication-core/README.md)
- [Architecture and ownership](docs/communication-core/architecture.md)
- [Schema](docs/communication-core/schema.md)
- [Account isolation and routing](docs/communication-core/isolation-routing.md)
- [Adapters](docs/communication-core/adapters.md)
- [Configuration](docs/communication-core/configuration.md)
- [Facebook migration](docs/communication-core/facebook-migration.md)
- [CLI and skill usage](docs/communication-core/usage.md)
- [Privacy and safety](docs/communication-core/privacy-safety.md)
- [Operations runbook](docs/communication-core/operations.md)
- [Rollback](docs/communication-core/rollback.md)
- [Requirement evidence](docs/communication-core/requirements-evidence.md)
- [`manage-communications` skill](skills/manage-communications/SKILL.md)
- [Communication backlog](plans/communication-skills-backlog.md)

## Final audit

| Criterion | Status |
| --- | --- |
| Every non-dating P0–P4 ID has implementation and evidence | PASS |
| News P1–P3 and XDOM evidence complete without publication | PASS |
| Person/account/identity/endpoint separation and account isolation | PASS |
| Facebook migration is stable, read-only and reversible | PASS |
| Route matrix, analysis, CRM, groups, greetings and outbox invariants | PASS |
| Host relevant suite green with explained skips | PASS |
| Linux relevant suite green with all platform tests executed | PASS |
| Production image minimal, starts, and write defaults disabled | PASS |
| Controlled local synthetic smoke and cleanup | PASS |
| Documentation, backlog, memory and this report current | PASS |
| Named dating pilot and safe test account/profile | **BLOCKED** |

Because one mandatory scope remains blocked, the only valid final verdict is
**NO-GO** and the active goal must not be marked complete.
