# Hermes Security Model

Date: 2026-05-20

## Security Goal

Hermes should be powerful locally while keeping external actions, credentials,
private data, destructive operations, and money/spend/trade paths gated,
auditable, and reversible where possible.

## Core Rules

- Never expose or commit secrets.
- Never print raw env, Keychain values, auth JSON, tokens, private keys, or
  private user data.
- Never perform external sends, publishing, purchases, trades, money movement,
  or account changes without explicit user confirmation.
- Prefer read-only inspection and scaffolds when a capability is unsafe or
  blocked.
- Keep live-service restarts drain-aware and wrapper-preserving.

## Existing Strong Controls

- Keychain-loaded gateway environment through `Operator/scripts/hermes-env.sh`.
- Wrapper-backed launchd startup through `Operator/scripts/hermes-gateway.sh`.
- Owner-only core config/auth files observed for `.env`, `auth.json`, and
  `config.yaml`.
- Redaction enabled by default in `agent/redact.py`.
- Dangerous command approval system in `tools/approval.py`.
- Hardline non-overridable command blocks for catastrophic shell actions.
- API server detailed endpoints require bearer auth when configured.
- Test runner unsets credential-shaped env variables.
- Cron redacts stdout/stderr before return paths.
- `hermes ops status` provides read-only, redacted operator metadata and counts
  without printing raw logs, cron prompt bodies, private memory, env values, or
  secrets.
- `hermes ops status --markdown` uses the same redacted metadata payload for
  shareable handoff receipts; it does not add new runtime reads beyond the
  existing ops-status path.

## Current Risks

High:

- Some historical runtime/session/request artifacts under `~/.hermes` are more
  readable than appropriate for private prompts/tool output.

Medium:

- Docker config can forward broad tokens and mount broad host paths. The
  current control inventory now reports this as read-only metadata; enforcement
  and live config cleanup remain separate, explicit steps.
- Smart approval mode can approve false negatives.
- Destructive slash confirmation is not the safest default.
- Quick commands are powerful exec shortcuts and need risk metadata.
- Non-interactive execution paths need clearer policy gates.

## Permission Boundary

Risk tiers:

- `R0`: read-only local status/search.
- `R1`: documentation and metadata changes.
- `R2`: workspace-scoped file/code changes.
- `R3`: local runtime/service changes.
- `R4`: external sends, account writes, publishing, deploys, credential edits.
- `R5`: spend, trading, money movement, destructive/system operations.

Default policy:

- `R0`: allow and log.
- `R1`: allow after repo safety check.
- `R2`: allow after scoped plan and validation.
- `R3`: require explicit restart/service procedure.
- `R4`: require typed confirmation.
- `R5`: deny by default unless the user gives explicit instruction and the
  action is policy-allowed; still require typed confirmation.

Named risk classes:

- `read_only`: inspection, search, status, and health checks.
- `local_write`: workspace-scoped writes and generated artifacts.
- `private_data_access`: private session/history/memory access or mutation.
- `credential_sensitive`: credential, token, keychain, auth, host-control, or
  provider-secret handling.
- `external_side_effect`: sends, posts, publishes, uploads, deploys, and
  provider/API side effects.
- `destructive`: broad deletion, destructive git, data-drop/truncate, and
  gateway/update actions that can interrupt active work.
- `financial_or_account_action`: spend, trading, transfers, purchases, and
  account ownership/provider changes.
- `unknown_restricted`: unmapped tools/actions that could mutate state.

## Sensitive Confirmation Gates

Require typed confirmation for:

- Publishing or posting.
- Sending email or external messages outside requested delivery.
- Purchases or paid API activation.
- Trading, transfers, bets, financial transactions.
- Credential/keychain changes.
- Launchd service mutation.
- Gateway restart/update when active work may be running.
- Force push, destructive git, broad deletion.
- AWS/account mutations.
- App Store/TestFlight actions.
- macOS permission grants.

Example phrases:

- `CONFIRM SEND <target>`
- `CONFIRM PUBLISH <target>`
- `CONFIRM SPEND <amount>`
- `CONFIRM DELETE <path>`
- `CONFIRM RESTART HERMES`

## Tool Policy

Allow by default:

- Read/search/status/doctor.
- Local non-mutating health checks.
- Safe web reads.
- Workspace-scoped docs/code edits after repo safety check.

Gate:

- Terminal.
- Execute-code.
- Browser/computer-use state mutation.
- Cron creation/editing.
- Quick commands.
- Docker/SSH/remote backends.
- Env passthrough.
- MCP reloads.
- Cross-platform messaging.
- Provider/account config changes.
- Raw Codex CLI execution from the terminal tool. Use the Operator
  `/Users/agent1/Operator/scripts/codex-run.sh` wrapper unless
  `HERMES_ALLOW_RAW_CODEX=1` is set after explicit review.

Deny by default:

- Raw secret reads.
- Credential file edits without confirmation.
- Raw `codex exec` and npm-based `@openai/codex exec` through the terminal
  tool, because they can inherit unsafe HOME/CODEX_HOME state outside the
  Operator wrapper.
- `sudo` and system directory writes.
- `curl|sh` installers unless reviewed and confirmed.
- Destructive shell commands.
- Unauthorized account automation.
- Spam, scams, abusive automation.
- Unsafe financial actions.

## Audit Log Plan

Append-only JSONL:

- Path: `~/.hermes/audit/YYYY-MM-DD.jsonl`
- Mode: `0600`
- Directory mode for newly created audit directories: `0700`
- Optional hash chain.

Record:

- Timestamp.
- Session ID/source.
- Actor/surface.
- Risk tier.
- Tool/command.
- Redacted target path or external target.
- Approval ID and decision.
- Exit code/status.
- Before/after hashes when feasible.
- Redaction status.
- Receipt paths.

Current Phase 5 audit implementation:

- `hermes_cli/audit_log.py` writes private JSONL events through the
  owner-only artifact helper.
- Audit files are forced to `0600` even when an existing file has broader
  mode bits or the process runs under permissive `umask`.
- Newly created audit directories are `0700`.
- Dangerous-command approval events cover request, approval, denial, block,
  skip/bypass, smart approval, cron policy, hardline block, sudo-stdin block,
  gateway timeout, and notify-failure outcomes.
- Slash-confirmation events cover approval, cancellation/denial, invalid
  choice, stale confirmation ID, timeout, missing handler, and handler error
  outcomes.
- Session keys, chat IDs, user IDs, and similar identifiers are hashed before
  write.
- Commands and descriptions are redacted with forced secret redaction and
  bounded before write.

Never record:

- Raw env values.
- Tokens or keys.
- Private keys.
- Full request bodies.
- Full private chat payloads.
- Raw command output when it may contain secrets.

## File Permission Plan

Phase 5 should:

- Set session/request dump creation to `0600`.
- Ensure containing directories are `0700`.
- Backfill old sensitive runtime artifacts.
- Add tests for file modes.
- Avoid chmodding unrelated user files outside Hermes-owned paths.

Current Phase 5 slice:

- New `hermes sessions export <path>` JSONL files are created `0600`.
- Missing parent directories created for session exports are created `0700`.
- Subagent timeout diagnostic request dumps are created `0600`.
- Missing diagnostic directories created for those request dumps are created
  `0700`.
- Approval/confirmation audit JSONL files are created or tightened to `0600`.
- Missing audit directories created by the audit helper are created `0700`.
- High-risk command approvals use named risk classes from
  `hermes_cli/security_policy.py`.
- Credential-sensitive classification includes token/secret/key-like output
  destinations such as provider token files under Hermes/provider config paths.
- Interactive CLI high-risk command approvals require exact typed phrases and
  reject aliases, default Enter, empty input, and near-miss text.
- High-risk command approvals are one-shot and do not add session/permanent
  allowlist entries.
- `HERMES_YOLO_MODE`, session-scoped yolo, and `approvals.mode: off` cannot
  bypass typed-confirmation-required risk classes.
- Noninteractive and gateway button approval paths block typed-confirmation
  risk classes until an exact typed-input gateway path exists.
- `hermes control inventory` reports named `risk_category` metadata while
  preserving the existing `R0`-`R5` `risk_class` field.
- Existing live directories are not chmodded by this slice; live backfill must
  be a separate, explicit review because it mutates private runtime state.

Current Docker/container review scaffold:

- `hermes_cli/docker_security.py` analyzes Docker/Podman command text and
  Hermes terminal Docker config without executing commands, reading Docker
  config files, emitting env values, or contacting Docker. Credential presence
  booleans use the existing control-inventory metadata pattern.
- `hermes control inventory` emits a `container_backend.docker` item with
  redacted finding codes, severity, risk category, counts, and safe-next-action
  metadata.
- The review flags sensitive `docker_forward_env` or `docker_env` variable
  names, Docker socket mounts, host root/home/credential path mounts, host cwd
  workspace mounts, `--privileged`, host networking, host namespaces, env-file
  forwarding, all capabilities, and host device/group access.
- MCP and quick-command inventory entries include redacted Docker review
  findings when configured command text uses Docker or Podman.
- The scaffold is observe-only. It does not change the Docker backend,
  historical artifacts, private memory, credentials, caches, logs, provider
  facts, or live Docker config.

Docker policy direction:

- Default to no broad host credential forwarding.
- Allow Docker credentials or host mounts only as explicit per-job opt-ins.
- Prefer narrow read-only mounts for non-secret fixtures.
- Treat Docker socket mounts, host home/root mounts, and credential directory
  mounts as high-risk and typed-confirmation candidates before any future
  enforcement step.

Current Docker/container enforcement scaffold:

- Docker backend startup now evaluates user-supplied Docker backend options
  with `hermes_cli/docker_security.py` before Docker availability checks,
  sandbox directory creation, or `docker run`.
- High and critical findings fail closed with `DockerSecurityPolicyError`.
- Blocked categories include sensitive Docker env forwarding, sensitive
  `docker_env` keys, Docker socket mounts, host root/home/credential path
  mounts, `--privileged`, host networking, host namespaces, env-file
  forwarding, and all capabilities.
- Medium findings remain observational in this slice; explicit host cwd
  workspace mounting and host device/group access are not blocked yet.
- Block messages include only finding codes/counts and a future typed-confirmed
  override hint. They do not include env values, raw host paths, Docker config
  file content, command output, or private file contents.
- Docker backend diagnostic logs redact host-side mount sources, env values,
  env-file paths, credential-file mount host paths, skills/cache host paths,
  assembled Docker run args, and the debug startup command. Execution args are
  unchanged; only log strings are redacted.
- Docker startup failure exceptions use redacted argv and omit raw stderr to
  avoid exposing raw Docker args through `CalledProcessError`.
- Docker availability preflight failures omit raw `docker version` stderr,
  which can include host socket or Docker config paths.
- This slice does not add a bypass or edit live Docker config. A future slice
  can design exact typed per-job override semantics.

## Provider Fallback Routing

Current post-campaign cleanup state:

- Hermes is configured with OpenRouter as the first fallback provider using
  `google/gemini-3-flash-preview`.
- The command `hermes fallback configure-openrouter` is the safe operator path
  for reapplying that policy. It supports `--dry-run`, preserves existing
  non-duplicate fallbacks after OpenRouter by default, and offers `--replace`
  for an explicit single-entry fallback chain.
- The command writes only non-secret provider routing metadata:
  provider, model, base URL, and API mode.
- The command does not read, print, validate, or mutate provider API keys.
- Before live config mutation, create an owner-only config backup and avoid
  printing raw config contents.
- Fallback routing does not change the wrapper-backed launchd gateway path.

Security boundaries:

- Do not print `.env`, Keychain output, API keys, bearer tokens, or raw provider
  auth files while inspecting fallback state.
- Do not replace the Operator gateway wrapper when validating fallback routing.
- Treat missing OpenRouter credentials as a doctor/runtime warning, not a reason
  to copy tokens into docs, logs, or chat.

## External Actions Policy

Safe scaffolds:

- Draft email content locally.
- Prepare social posts locally.
- Generate reports locally.
- Create deployment plans.
- Validate account state read-only.
- Queue approval requests.

Blocked without explicit confirmation:

- Send email.
- Post online.
- Make purchases.
- Execute trades.
- Transfer money.
- Change live accounts.
- Deploy mutable cloud resources.
- Upload release builds.

Phase 8 final-report Telegram handoff:

- Telegram delivery is the only approved Phase 8 external action.
- Delivery is gated on final validation, final report generation, targeted
  secret scans, passing gateway status, passing doctor, and three final judge
  PASS results.
- `hermes send --dry-run/--preflight` is the safe readiness check for future
  Telegram handoffs. It validates Telegram configuration, target shape, client
  library availability, and message metadata without calling Telegram or
  invoking the shared send tool.
- Dry-run receipts must not print message contents, raw chat IDs, thread IDs,
  bot tokens, credentials, or channel-private values. When persisted, use the
  `--output` path so the receipt is written through private artifact helpers.
- The delivery path uses the existing Operator environment wrapper so token
  and chat routing values stay in Keychain-backed environment variables.
- Commands, docs, build logs, and final responses must not print tokens,
  credentials, chat IDs, raw API responses, `.env`, auth files, Keychain
  output, launchd environments, private memory, or raw log lines.
- If delivery fails or cannot be proven safe, leave the report local and record
  the blocker instead of retrying through an ad hoc secret-exposing path.

## Security Validation

Required checks by phase:

- Redaction tests.
- Permission tests.
- Approval tests.
- API auth tests.
- Startup bind guard tests.
- Manual smoke for live gateway health without exposing secrets.
