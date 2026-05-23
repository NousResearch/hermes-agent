# Hermes Execution Plan

Date: 2026-05-20

## Current Stopping Point

Phase 0 control-plane bootstrap is complete and judged PASS.

Phase 2 Tool Registry And Plugin System is complete and judged PASS as of
2026-05-20. Phase 1 remains planned because the Phase 0 architecture docs
already clarified the relevant boundaries; the safest implementation step was
a read-only inventory surface that improves tool, plugin, runtime, and
permission visibility without moving runtime code.

This Phase 2 pass adds:

- `hermes control inventory --json --redact`
- `hermes control inventory --markdown --redact`
- redacted schema version 1 inventory output
- risk, cost, approval, health-probe, and safe-next-action metadata
- focused tests for schema, redaction, plugin status, missing MCP binaries,
  quick-command risk, parser registration, and project metadata

Phase 3 Memory System is complete and judged PASS for the read-only memory
audit/deletion-readiness scaffold as of 2026-05-20. Phase 4 Reliability Layer
now has two completed, judged-PASS slices: read-only gateway startup validation
and redacted incident bundling. Phase 5 Security And Permissions now has six
completed, judged-PASS slices: private-artifact permissions,
approval/confirmation audit JSONL hardening, high-risk typed confirmation
policy, read-only Docker/container permission review, and high-severity Docker
execution enforcement scaffolding. The newest completed Phase 5 slice redacts
Docker backend diagnostic logs and startup/preflight failure surfaces.

Phase 6 UX/Ops now has two completed slices as of 2026-05-21:
`hermes ops status`, a read-only redacted operator status command, and
`hermes ops status --markdown`, a handoff-ready redacted Markdown receipt. The
status payload summarizes gateway validation, API health, cron metadata,
health-loop receipts, disk usage, log metadata/counts, and receipt paths
without printing raw logs, cron prompts, private memory, env values, or
secrets.

Phase 7 Skills And Reusable Workflows now has one completed slice as of
2026-05-21: `.agents/skills/hermes-ops-review/SKILL.md`, a reusable
ops/testing/review workflow that standardizes redacted status receipts,
wrapper-backed gateway checks, focused tests, secret scans, build-log updates,
and judge-ready evidence. A second Phase 7 slice adds
`.agents/skills/hermes-final-report/SKILL.md`, which defines the final report
anatomy, how-to guide, validation evidence, and gated Telegram delivery
workflow for the completed campaign.

Phase 8 Final Integration And Validation is complete and judged PASS as of
2026-05-21. The campaign now has an integrated final report at
`docs/HERMES_FINAL_REPORT.md`, final validation evidence in
`docs/HERMES_BUILD_LOG.md`, and a completed gated Telegram handoff that used
the existing Hermes/Operator environment without printing secrets, tokens, or
chat IDs.

Post-campaign cleanup is now active. The first completed cleanup slice resolves
the pre-existing `_raw_codex_guard` collection issue by adding a terminal
pre-exec guard that blocks raw `codex exec` and npm-based Codex exec calls
unless explicitly allowed, while preserving the Operator
`/Users/agent1/Operator/scripts/codex-run.sh` wrapper path. The second
completed cleanup slice adds a Telegram delivery dry-run/preflight path to
`hermes send` so final-report handoffs can prove readiness without calling the
Telegram API or printing private target/message data. The third completed
cleanup slice adds an explicit private-memory compaction approval plan without
reading or mutating live memory.

## Phase 0 - Repo Safety And Operating Contract

Status: completed for the control-plane bootstrap.

Goals:

- Check git status.
- Create rollback path.
- Create concise `AGENTS.md`.
- Create required `docs/HERMES_*` control-plane documents.
- Create judge rubric and build log.
- Validate docs and live startup path.

Validation:

- `git diff --check`
- `git diff --cached --check` after staging only Phase 0 files
- `hermes doctor`
- `hermes gateway status`
- focused docs/control-plane review
- judge cycle

Exit criteria:

- Docs exist.
- Build log records commands and known issues.
- Judge cycle returns PASS or documents fixes.

Phase 0 validation results:

- `git diff --check`: passed.
- `hermes doctor`: passed with expected optional-provider warnings.
- `hermes gateway status`: passed; active wrapper-backed service loaded.
- `scripts/run_tests.sh tests/test_project_metadata.py`: 6 passed.
- Secret-pattern scan over `AGENTS.md` and `docs/HERMES_*.md`: no matches.
- First judge cycle: Architecture PASS, Reliability/Security PASS,
  Tooling/UX FAIL.
- Required judge fixes: add developer guide, add operator quickstart, update
  stale phase status, stage Phase 0 docs for real diff validation, and rerun
  validation/judges.
- Judge fixes applied and revalidated:
  - `git diff --cached --check`: passed.
  - `git diff --check`: passed.
  - secret-pattern scan over `AGENTS.md` and `docs/HERMES_*.md`: no matches.
  - `scripts/run_tests.sh tests/test_project_metadata.py`: 6 passed.
  - `hermes gateway status`: passed.
  - `hermes doctor`: passed with expected optional-provider warnings.
- Second judge cycle: Architecture PASS, Reliability/Security PASS,
  Tooling/UX FAIL.
- Required second-cycle fix: expand `docs/HERMES_DEVELOPER_GUIDE.md` to
  preserve the old developer workflow material while keeping `AGENTS.md`
  concise.
- Second-cycle fix applied: developer guide now covers TUI/dashboard,
  dependency pinning, config, skins, plugins, skills, toolsets, delegation,
  curator, cron, kanban, runtime policies, memory, profile-safe paths,
  security pitfalls, known pitfalls, and testing-wrapper rationale.
- Second-cycle fix revalidated:
  - `git diff --cached --check`: passed.
  - `git diff --check`: passed.
  - secret-pattern scan over `AGENTS.md` and `docs/HERMES_*.md`: no matches.
  - `scripts/run_tests.sh tests/test_project_metadata.py`: 6 passed.
  - `hermes gateway status`: passed.
  - `hermes doctor`: passed with expected optional-provider warnings.
- Third judge cycle: Architecture PASS, Reliability/Security FAIL,
  Tooling/UX PASS.
- Required third-cycle fix: replace misleading rollback-patch reference with a
  real recovery path.
- Third-cycle fix applied: generated non-empty staged-doc rollback patch at
  `.codex-backups/phase0-control-plane-staged-20260520-validated.patch` and
  updated operator quickstart with explicit abandon/reapply commands.
- Third-cycle fix revalidated:
  - rollback patch regenerated and verified non-empty.
  - `git diff --cached --check`: passed.
  - `git diff --check`: passed.
  - secret-pattern scan over docs and rollback patch: no matches.
  - `scripts/run_tests.sh tests/test_project_metadata.py`: 6 passed.
  - `hermes gateway status`: passed.
  - `hermes doctor`: passed with expected optional-provider warnings.
- Fourth judge cycle: Architecture PASS, Reliability/Security PASS,
  Tooling/UX PASS.

Phase 0 final status:

- Complete for control-plane bootstrap.
- All three judges passed.
- No feature code or runtime config changed.
- Full optimization campaign is not complete; Phases 1-8 remain planned.
- Final validation after recording judge results passed:
  `git diff --cached --check`, `git diff --check`, rollback reverse-apply
  check, secret-pattern scan, focused metadata tests, `hermes gateway status`,
  and `hermes doctor`.

## Phase 1 - Architecture Cleanup

Status: planned.

Goal:

Clarify module boundaries without rewriting the agent runtime.

Planned work:

- Keep gateway as runtime nucleus.
- Define optimization control plane as plugin/CLI/API layer.
- Document gateway/core/tool/memory/runtime boundaries in code-adjacent docs.
- Add minimal architecture tests only if code changes are made.

Validation:

- Existing tests around gateway startup/API/tool registry.
- New tests only for changed seams.

Exit criteria:

- No behavior regression.
- No direct optimization business logic added to `gateway/run.py`.

## Phase 2 - Tool Registry And Plugin System

Status: completed and judged PASS.

Goal:

Build a read-only registry overlay before any enforcement.

Planned work:

- Add `hermes_cli/control.py`.
- Add `hermes control inventory --json --redact`.
- Inventory built-in tools, toolsets, plugins, MCP servers, quick commands,
  cron jobs, launchd labels, operator scripts, and credential presence.
- Split output into `observed_state` and `policy_overlay` so read-only
  discovery does not blur into enforcement.
- Add risk/cost/approval fields as policy metadata after read-only inventory
  is stable.

Implemented:

- `hermes_cli/control.py` builds a read-only, redacted inventory.
- `hermes_cli/main.py` registers the new `control` command and keeps startup
  plugin discovery gating in sync.
- `tests/hermes_cli/test_control_inventory.py` covers schema stability,
  secret redaction, missing MCP binary classification, plugin credential
  gating, high-risk quick-command classification, and CLI JSON output.
- Runtime probes are optional via `--no-runtime`; tool requirement probes are
  optional via `--no-tool-probe`.
- Credential fields report presence booleans only. Prompt bodies, env values,
  and raw launchd environment details are not included.

Validation:

- Unit tests for redaction.
- Unit tests for missing binary classification.
- Unit tests for plugin status classification.
- Stable JSON schema invariant test.
- Smoke: compare with generated inventory summary and `hermes gateway status`.

Phase 2 validation results so far:

- `scripts/run_tests.sh tests/hermes_cli/test_control_inventory.py tests/hermes_cli/test_startup_plugin_gating.py tests/test_project_metadata.py`:
  passed, 55 tests after remediation.
- `./venv/bin/python -m hermes_cli.main control inventory --json --redact`:
  passed and wrote parseable JSON.
- `./venv/bin/python -m hermes_cli.main control inventory --markdown --redact --no-runtime --no-tool-probe`:
  passed and wrote Markdown.
- JSON parse check with `python3 -m json.tool`: passed.
- Secret-pattern scan over generated inventory JSON/Markdown: no matches.
- `./venv/bin/python -m hermes_cli.main gateway status`: passed; active
  service label is `ai.hermes.gateway` and the program is
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- `./venv/bin/python -m hermes_cli.main doctor`: passed with expected
  optional-provider/tool-credential warnings.
- Later remediation tightened tool status semantics: plugin required envs gate
  plugin status, while built-in tool status follows registry/runtime
  availability and reports absent alternative backend envs as credential
  candidates instead of hard missing credentials.
- Final judge cycle: Architecture PASS, Reliability/Security PASS,
  Tooling/UX PASS.

Exit criteria:

- Read-only inventory runs without secrets.
- Registry identifies missing/gated/healthy tools accurately.

## Phase 3 - Memory System

Status: completed and judged PASS for a read-only audit/deletion-readiness
scaffold. Direct compaction of live `MEMORY.md` and `USER.md` remains blocked
until the user explicitly approves mutation of private persistent memory.

Goal:

Make memory durable, safe, compact, and deletable.

Planned work:

- Report `MEMORY.md` and `USER.md` capacity without printing contents.
- Compact `MEMORY.md` and `USER.md` to leave headroom when explicitly
  approved.
- Add a forget/deletion checklist covering markdown, structured DBs, session
  DBs, logs, backups, snapshots, response store, and profile stores.
- Reconcile holographic facts when built-in memory is replaced or removed.
- Document retention policy.

Implemented in the current Phase 3 pass:

- `hermes memory audit --json --redact`
- `hermes memory audit --markdown --redact`
- metadata-only store coverage for built-in markdown memory, holographic
  memory DB, session DB/transcripts, response store, logs, cache/media,
  screenshot, audio/video, document cache, backups, checkpoints, and profile
  homes
- capacity/headroom reporting for `MEMORY.md` and `USER.md`
- owner-only permission warnings for private memory/session/log stores
- explicit forget/deletion checklist with confirmation-required destructive
  commands separated from safe audit commands
- reconciliation notes for active external memory providers, especially
  `holographic`
- read-only CLI startup path for `hermes memory audit` that does not
  initialize a missing `HERMES_HOME`

Blocked:

- Live memory compaction and provider fact deletion are intentionally not
  automated in this phase. They require an explicit user-specific memory edit
  or forget request because they mutate private persistent runtime state.

Validation:

- Existing memory tests.
- New tests for audit schema, content non-leakage, missing stores, permission
  warnings, forget-domain coverage, provider reconciliation status, and CLI
  output.
- Manual smoke: `hermes memory status`.
- Manual smoke: `hermes memory audit --json --redact` and
  `hermes memory audit --markdown --redact`.
- Fresh missing `HERMES_HOME` smoke confirms `hermes memory audit --json
  --redact` does not initialize runtime directories.
- Final focused suite: 148 passed.
- Final judge cycle: Architecture PASS, Reliability/Security PASS,
  Tooling/UX PASS.

Exit criteria:

- Memory writes have capacity, or the compaction blocker is explicitly
  documented.
- Deletion path is documented and testable.

## Phase 4 - Reliability Layer

Status: two slices completed and judged PASS. Startup validation and redacted
incident bundling are in place. Broader structured runtime logging and
repair-flow work remain planned.

Goal:

Make startup, health checks, logs, and repair behavior explicit.

Planned work:

- Canonicalize `ai.hermes.gateway` in docs and status.
- Warn on legacy `com.agent1.hermes.gateway` if loaded.
- Preserve wrapper-backed launchd path.
- Add startup validation and incident bundle command.
- Improve health check treatment of authenticated detailed endpoints.

Implemented in the completed Phase 4 slice:

- `hermes gateway validate`
- `hermes gateway validate --json`
- `hermes gateway validate --markdown`
- read-only launchd startup validation for:
  - canonical active label
  - launchd plist presence and parseability
  - `/Users/agent1/Operator/scripts/hermes-gateway.sh` wrapper preservation
  - active `ai.hermes.gateway` loaded/running state
  - loaded-but-not-running active label failure handling
  - loaded legacy `com.agent1.hermes.gateway` warning
- local API health validation for:
  - unauthenticated `/health` HTTP 200
  - unauthenticated `/health/detailed` HTTP 200 or expected HTTP 401/403
  - local-only probing with graceful skip for non-local API hosts
- redacted JSON/Markdown/text output with no raw launchd environment output
  and no bearer token use.
- clean CLI errors for invalid negative or zero timeout values

Implemented in the second completed Phase 4 slice:

- `hermes gateway incident-bundle`
- `hermes gateway incident-bundle --json`
- optional `--output`, `--force`, `--no-health`, `--no-log-metadata`, and
  `--no-health-loop-metadata` controls
- local incident bundle files with private permissions:
  - `manifest.json`
  - `gateway_validation.json`
  - `artifact_metadata.json`
  - `summary.md`
- metadata-only receipts for known gateway logs and health-loop status files
- explicit bundle manifest fields proving:
  - no runtime mutation
  - no raw log content copied
  - no private memory read
  - no external side effects
- clean CLI errors for invalid negative or zero timeout values

Validation:

- Gateway status tests.
- API server health/auth tests.
- Launchd-wrapper preservation tests where feasible.
- Live smoke through `hermes gateway status`.
- Live smoke through `hermes gateway validate --json`.
- Incident bundle smoke through
  `hermes gateway incident-bundle --output /tmp/hermes-gateway-incident-phase4 --force --json`.
- Secret-pattern scan over generated incident bundle files.

Exit criteria:

- No broken startup path.
- Repair flow is pulse-first and drain-aware.
- Current slice exit: validator passes focused tests, live startup validation
  passes, and all three judges pass. The legacy label can remain as a warning
  because this slice is diagnostic only and intentionally does not mutate
  launchd services.
- Incident bundle slice exit: focused tests pass, generated bundle files are
  `0600` inside a `0700` directory, targeted secret-pattern scan is clean, and
  all three judges pass.

## Phase 5 - Security And Permissions

Status: six slices completed and judged PASS. Strict permissions for new
session export artifacts, subagent timeout diagnostic request dumps, and
approval/confirmation audit JSONL are in place. Broader high-risk typed
confirmation policy is now partially enforced for command/tool risk classes.
Docker-token forwarding/container permission review is now surfaced
read-only, and high-severity Docker findings now fail closed before container
execution. Historical live-artifact backfill remains planned.

Goal:

Harden default autonomy and evidence without weakening existing capabilities.

Planned work:

- Backfill strict permissions for session/request dump artifacts.
- Ensure new session/dump files are created `0600`.
- Add audit JSONL for gated actions.
- Define risk classes and map high-impact tools/commands.
- Make dangerous external actions require typed confirmation.
- Move broad Docker token forwarding to per-job opt-in.

Implemented in the first completed Phase 5 slice:

- Added `hermes_cli/private_artifacts.py` with reusable owner-only artifact
  helpers.
- `hermes sessions export <path>` now writes new JSONL session exports with
  file mode `0600`.
- Missing parent directories created for session exports are `0700`.
- Existing parent directories are not chmodded.
- Subagent timeout diagnostic dumps now write `0600` files.
- Newly created subagent diagnostic log directories are `0700`.
- Existing live logs, sessions, caches, provider facts, credentials, and
  private memory were not chmodded or mutated.

Implemented in the second completed Phase 5 slice:

- Added `hermes_cli/audit_log.py` for private, redacted, append-only JSONL
  audit events.
- Approval audit ledgers default to `~/.hermes/audit/YYYY-MM-DD.jsonl`.
- New audit directories are created `0700`.
- New and existing audit JSONL files opened by the helper are forced to
  `0600`, including under permissive `umask`.
- Dangerous-command approvals now record structured, redacted outcomes for
  requested, approved, denied, blocked, skipped, smart-approved,
  smart-denied, preapproved, yolo/mode-off bypass, cron policy, hardline
  block, sudo-stdin block, gateway timeout, and gateway notify failure paths.
- Slash confirmations now record structured, redacted outcomes for approved,
  cancelled/denied, invalid choice, stale confirmation ID, timeout, missing
  handler, and handler error paths.
- Session keys and comparable identifiers are hashed in audit events rather
  than written raw.
- Commands/descriptions are redacted and bounded before JSONL write.
- Dangerous-command and slash-confirm choices reject near-miss inputs through
  exact choice normalizers.
- Existing private memory, live logs, caches, provider facts, credentials, and
  historical artifacts were not backfilled or rewritten.

Implemented in the third completed Phase 5 slice:

- Added `hermes_cli/security_policy.py` with named risk classes:
  `read_only`, `local_write`, `private_data_access`, `credential_sensitive`,
  `external_side_effect`, `destructive`, `financial_or_account_action`, and
  `unknown_restricted`.
- Added explicit mappings for existing registered tools and command-like
  shortcuts where safe.
- Credential-sensitive command classification includes token/secret/key-like
  output destinations and Hermes/provider credential paths.
- Unknown/unmapped tools default to `unknown_restricted` and
  `typed_confirm` in policy metadata.
- `hermes control inventory` now reports both legacy `risk_class` (`R0`-`R5`)
  and named `risk_category`, plus typed-confirmation-required metadata in the
  policy overlay.
- Interactive CLI high-risk command approvals require exact typed phrases such
  as `CONFIRM DESTRUCTIVE`; loose yes/no, aliases, empty input, and near-miss
  text are denied.
- High-risk command confirmations remain one-shot and do not create session or
  permanent allowlist entries.
- High-risk typed-confirmation classes are not bypassed by
  `HERMES_YOLO_MODE`, session-scoped yolo, or `approvals.mode: off`.
- Noninteractive and gateway button approval paths block typed-confirmation
  risk classes because exact typed input is not available in those flows yet.
- Approval outcomes continue to use hardened private audit JSONL with hashed
  session identifiers and redacted command previews.
- Safe read-only commands and tools remain low-friction.
- Existing private memory, live logs, caches, provider facts, credentials, and
  historical artifacts were not mutated.

Implemented in the fourth completed Phase 5 slice:

- Added `hermes_cli/docker_security.py` as a pure read-only Docker/container
  permission analyzer.
- `hermes control inventory` now includes `container_backend.docker`, a
  metadata-only review of terminal Docker backend config.
- The review flags credential-sensitive Docker forwarding and isolation risks,
  including sensitive `docker_forward_env`/`docker_env` names, Docker socket
  mounts, host home/root/credential path mounts, host cwd workspace mounts,
  `--privileged`, host network/namespaces, env-file forwarding, all
  capabilities, and host device/group access.
- MCP and quick-command inventory items now include redacted Docker review
  findings when their command text invokes Docker or Podman.
- Review output includes only finding codes, severity, risk category, redacted
  details, counts, and env/key names already treated as credential metadata;
  it does not read raw Docker config, contact Docker, emit env values, or
  mutate live configuration.
- The live read-only inventory smoke found Docker review findings in the
  current config shape and reported only summary metadata; no live Docker
  config, private memory, live logs, caches, provider facts, or credentials
  were changed.

Implemented in the fifth completed Phase 5 slice:

- Added a high-severity enforcement scaffold for Docker backend startup.
- `DockerEnvironment` now evaluates user-supplied Docker backend options with
  `hermes_cli/docker_security.py` before Docker availability checks, sandbox
  directory creation, or `docker run`.
- High and critical findings fail closed through `DockerSecurityPolicyError`.
- Blocked cases include sensitive Docker env forwarding, sensitive `docker_env`
  keys, Docker socket mounts, host home/root/credential path mounts,
  `--privileged`, host network/namespaces, env-file forwarding, all
  capabilities.
- Medium findings such as explicit host cwd workspace mounting and host
  device/group access remain observational for this slice.
- Error messages include only finding codes and counts; they do not include
  env values, raw host paths, Docker config file content, command output, or
  private file contents.
- No live Docker config, private memory, live logs, caches, provider facts, or
  credentials were changed.

Implemented in the sixth Phase 5 slice:

- Docker backend diagnostic logs now redact user-supplied volume specs, host
  paths, env assignment values, env-file paths, credential-file mount host
  paths, skill/cache host paths, assembled Docker run args, and the debug
  startup command.
- Docker startup failure exceptions now use redacted argv and omit raw stderr
  so `CalledProcessError` cannot expose raw Docker args.
- Docker availability preflight failures omit raw `docker version` stderr to
  avoid exposing host socket/config paths.
- The runtime `docker run` argument list is not changed; redaction applies only
  to diagnostic log strings.
- Blocked high-risk Docker configs no longer emit raw host paths or sensitive
  mount specs before `DockerSecurityPolicyError`.
- Tests prove blocked high-risk volume config does not log raw host paths and
  safe Docker execution still receives original args while logs are redacted.
- No live Docker config, private memory, live logs, caches, provider facts, or
  credentials were changed.

Validation:

- Permission tests.
- Redaction tests.
- Approval/gate tests.
- Audit JSONL permission, redaction, approval, denial, block, and typed-choice
  tests.
- Risk-class policy and high-risk typed-confirmation tests.
- Manual security smoke with fake secrets only.
- Temporary `HERMES_HOME` session-export smoke.
- Temporary `HERMES_HOME` audit JSONL smoke under `umask 000`.
- Read-only Docker/container inventory smoke.
- Docker enforcement smoke using synthetic high-risk backend options.
- Docker diagnostic redaction smoke using synthetic paths and env values.
- Targeted secret-pattern scans over touched files and generated temp
  artifacts.

Exit criteria:

- No unresolved critical security issue.
- No secret-bearing output in docs/logs/tests.

## Phase 6 - UX/Ops

Status: partially complete.

Goal:

Make Hermes easier to run, inspect, and debug.

Completed slices:

- `2026-05-21`: added `hermes ops status` as a read-only, redacted operator
  status command. It preserves the wrapper-backed `ai.hermes.gateway` startup
  path and reports gateway validation, legacy-label state, API health, cron
  counts, health-loop receipt metadata, disk usage, log metadata/counts, and
  receipt paths without raw log lines, cron prompt bodies, private memory,
  provider facts, env values, or secrets.
- `2026-05-21`: added `hermes ops status --markdown` as a redacted
  handoff-ready receipt format using the same read-only status payload and
  redaction boundaries as text/JSON output.

Remaining planned work:

- Keep refining the compact local ops guide as new operator flows land.
- Treat dashboard as a manual local console unless service-managed later.
- Consider a follow-up `hermes ops quickstart` or `hermes ops doctor` only if
  it removes real operator friction without duplicating `doctor`.

Validation:

- CLI smoke.
- Status output redaction check.
- Gateway health smoke.
- Focused `hermes ops status` tests.
- Markdown output smoke and redaction check.

Exit criteria:

- User can run, inspect, restart, and troubleshoot Hermes from one documented
  path.

## Phase 7 - Skills And Reusable Workflows

Status: partially complete.

Goal:

Capture repeatable Hermes workflows as skills or workflow packs.

Planned work:

- Add `.agents/skills/` or repo-native skill entries only where useful.
- Add review workflow.
- Add testing workflow.
- Add tool-building workflow.
- Add safe research/report/content/coding workflow scaffolds.

Completed slices:

- `2026-05-21`: added `.agents/skills/hermes-ops-review/SKILL.md`, a reusable
  Hermes ops/testing/review workflow. It uses `hermes ops status --markdown`
  and `hermes ops status --json --no-health` as redacted receipts, preserves
  the wrapper-backed `ai.hermes.gateway` path, avoids private memory/log
  mutation, and defines focused tests, secret scans, build-log updates, and
  judge-cycle evidence.
- `2026-05-21`: added `.agents/skills/hermes-final-report/SKILL.md`, a reusable
  final-report workflow. It defines the final report anatomy, how-to guide,
  system anatomy, validation summary, known limitations, operator commands,
  and a gated Telegram delivery path that only runs after Phase 8/final
  validation and secret scanning.

Remaining planned work:

- Add a reusable code-review workflow if it removes repeated review setup.
- Add a reusable testing workflow for scoped phase validation.
- Add a reusable tool-building workflow after the tool interface stabilizes.
- Add safe research/report/content/coding workflow scaffolds only when they
  have a concrete operator use case.

Safe workflow lanes:

- Research scouts: public reads, local scoring, source receipts, no spend.
- Local file processing: local-only artifacts and redacted reports.
- Business intelligence: Mission Control-led reports and operational rollups.
- Content: draft and approval packets only; no account actions by default.
- Coding: scoped kanban/subagent tasks, reviews, tests, and evidence bundles.

Excluded:

- Spam, scams, fake engagement, abusive automation, unauthorized account
  actions, unsafe financial actions, live trades, and unapproved publishing.

Validation:

- Skill lint/inspection if available.
- Manual workflow dry-runs.
- Focused tests for skill metadata, required receipt commands, runtime
  boundaries, and absence of live-mutation commands.
- Focused tests for final-report anatomy, validation commands, Telegram
  delivery gates, redaction rules, and absence of token-dumping instructions.

Exit criteria:

- Reusable procedures are not buried in chat history.

## Phase 8 - Final Integration And Validation

Status: completed and judged PASS.

Goal:

Run integrated validation and judge cycle for the full campaign.

Final report requirement:

- Produce a full final report with executive summary, phase upgrade map,
  Hermes system anatomy, operator how-to guide, validation evidence, known
  limitations, exact commands, and next actions.
- Send the final report through the existing Hermes/Operator Telegram path only
  after Phase 8 passes or a final blocker is documented and the report draft
  passes targeted secret scanning.

Implemented:

- Generated `docs/HERMES_FINAL_REPORT.md`.
- Re-ran integrated validation over the control inventory, memory audit,
  gateway startup validation, incident bundle, ops status, approval/security
  gates, Docker security, gateway/CLI confirmation, and API health surfaces.
- Confirmed the wrapper-backed `ai.hermes.gateway` startup path remains active.
- Confirmed doctor passes with expected optional-provider warnings.
- Confirmed memory audit remains metadata-only and does not initialize a fresh
  missing `HERMES_HOME`.
- Confirmed incident bundle artifacts are private and do not copy raw logs or
  read private memory.
- Secret-scanned the final report and generated handoff artifacts.
- Ran final Architecture, Reliability/Security, and Tooling/UX judges; all
  passed.
- Completed gated Telegram delivery through the existing Operator environment
  wrapper after all final checks passed.

Validation:

- Focused unit tests.
- Relevant smoke tests.
- Startup verification.
- Tool loading verification.
- Memory behavior verification.
- Security gate verification.
- Docs completeness check.

Exit criteria:

- All three judges PASS. Completed.
- No critical security issue. Completed.
- No broken startup path. Completed.
- Tests or smoke tests pass, or failures are documented with next steps.
  Completed with focused integrated suite.
- Build log and execution plan are updated. Completed.

## Post-Campaign Cleanup

Status: eight cleanup/deployment-readiness/handoff slices completed and judged
PASS.

Completed slices:

- `2026-05-21`: resolved the stale
  `tests/tools/test_terminal_codex_guard.py` collection failure by adding
  `_raw_codex_guard` in `tools/terminal_tool.py`, wiring it before terminal
  environment creation, and expanding focused tests. Raw `codex exec`, `npx
  @openai/codex exec`, and package-manager Codex exec paths now fail closed
  unless `HERMES_ALLOW_RAW_CODEX=1` is set. The Operator wrapper
  `/Users/agent1/Operator/scripts/codex-run.sh` remains allowed.
- `2026-05-21`: added `hermes send --dry-run/--preflight` for Telegram
  delivery readiness checks. The preflight validates configured platform,
  credential presence, home-channel or explicit target shape, client library
  availability, and message metadata without invoking `send_message_tool`,
  contacting Telegram, printing message content, or printing raw chat/thread
  identifiers. `--output` writes the redacted receipt through the private
  artifact helper with `0700` parent directories and `0600` files.
- `2026-05-21`: added an explicit private-memory compaction approval plan to
  `docs/HERMES_MEMORY_PLAN.md`. The plan requires separate exact typed
  approvals for read-and-draft and apply steps, keeps raw private memory out of
  chat and repo artifacts, requires owner-only backups/drafts, and leaves all
  live private memory untouched until a future current-run approval.
- `2026-05-21`: after the exact current-run read-and-draft approval phrase was
  provided, created a private read-only compaction draft for only the default
  eligible memory files. Draft artifacts were written outside the repo under
  owner-only permissions, live memory files were not mutated, generated
  artifacts passed targeted secret-pattern scanning, and the slice passed the
  three-judge cycle.
- `2026-05-21`: after the exact current-run apply approval phrase was provided,
  created owner-only backups, applied the reviewed draft to only
  `~/.hermes/memories/MEMORY.md` and `~/.hermes/memories/USER.md`, verified
  live files match the draft and backups match pre-apply bytes, ran focused
  tests and smokes, scanned artifacts without printing private content, and
  passed the three-judge cycle.
- `2026-05-21`: configured fallback provider routing to use OpenRouter as the
  first backup provider with `google/gemini-3-flash-preview`, after creating an
  owner-only config backup and adding an idempotent
  `hermes fallback configure-openrouter` command with dry-run support. Focused
  fallback tests, gateway/doctor/ops smokes, secret scans, and the
  three-judge cycle passed.
- `2026-05-21`: refreshed the deployment-readiness state in the final report
  after private memory compaction and fallback routing were completed. The
  pass corrected stale limitations, revalidated gateway/doctor/ops/fallback
  surfaces, preserved the wrapper-backed gateway path, passed targeted secret
  scans, and judged PASS.
- `2026-05-21`: prepared the final repository handoff by reviewing the dirty
  working tree, documenting commit-candidate paths, excluding unrelated local
  artifacts, adding a PR-ready summary, preserving the wrapper-backed gateway
  path, and revalidating focused tests, smokes, secret scans, and diff checks.

Remaining cleanup candidates:

- Consider future parser modularization only as a small, well-tested slice.
- Commit or open a PR only after explicit current-run approval and an
  intentional staging review using `docs/HERMES_REPOSITORY_HANDOFF.md`.

## Recommended Next Prompt

Recommended next prompt after this campaign:

```text
Review docs/HERMES_REPOSITORY_HANDOFF.md and the current git status. If the
campaign-owned paths are still correct, intentionally stage only those paths,
run the documented validation commands, create a local commit, and stop before
any external push or PR creation unless explicitly approved in the current run.
```
