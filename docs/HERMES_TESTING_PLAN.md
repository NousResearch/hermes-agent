# Hermes Testing Plan

Date: 2026-05-20

## Current Test System

Canonical local runner:

```bash
scripts/run_tests.sh
```

The runner:

- Finds `.venv`, `venv`, or shared Hermes venv.
- Ensures pytest-split is available.
- Unsets credential-shaped environment variables.
- Pins deterministic env.
- Forces live-gateway test guard if available.
- Runs pytest with xdist and skips integration/e2e unless explicitly requested.

CI:

- `.github/workflows/tests.yml`
- `.github/workflows/lint.yml`
- `.github/workflows/osv-scanner.yml`
- `.github/workflows/supply-chain-audit.yml`
- docs/site/lockfile workflows.

## Validation Tiers

### Tier 0 - Docs/Metadata

Use when only docs or AGENTS files change:

```bash
git diff --check
hermes doctor
hermes gateway status
```

Optional:

```bash
scripts/run_tests.sh tests/test_project_metadata.py
```

### Tier 1 - CLI/Config/Registry

Use when CLI, config, plugin loading, or inventory changes:

```bash
scripts/run_tests.sh tests/hermes_cli tests/tools/test_registry.py tests/test_project_metadata.py
```

Add targeted tests for any new command.

### Tier 2 - Gateway/API

Use when gateway, API server, platform routing, launchd, or health changes:

```bash
scripts/run_tests.sh tests/gateway tests/hermes_cli/test_gateway_service.py tests/hermes_cli/test_status.py
```

Live smoke:

```bash
hermes gateway status
curl -sS http://127.0.0.1:8642/health
```

Do not call authenticated detailed endpoints without proper redaction.

### Tier 3 - Tools/Execution

Use when tool dispatch, terminal, file, code execution, browser, or MCP changes:

```bash
scripts/run_tests.sh tests/tools tests/test_toolsets.py tests/test_model_tools_async_bridge.py
```

Add focused tool tests for redaction, permission gates, and output limits.

### Tier 4 - Memory

Use when memory, session search, context compression, or memory providers change:

```bash
scripts/run_tests.sh tests/agent/test_memory_provider.py tests/agent/test_memory_session_switch.py tests/tools/test_memory_tool.py tests/tools/test_session_search.py tests/plugins/memory
```

Manual smoke:

```bash
hermes memory status
```

### Tier 5 - Security

Use when redaction, approvals, env handling, file permissions, command guards,
or API auth changes:

```bash
scripts/run_tests.sh tests/agent/test_redact.py tests/tools/test_command_guards.py tests/tools/test_terminal_codex_guard.py tests/gateway/test_api_server.py tests/gateway/test_api_server_bind_guard.py
```

Add tests for:

- Secret redaction.
- File mode creation.
- Typed confirmation gates.
- Denied high-risk commands.
- No raw env leakage.

## Smoke Checks

Local Hermes health:

```bash
hermes doctor
hermes gateway status
hermes fallback list
hermes tools list
```

Launchd path:

```bash
launchctl print gui/$(id -u)/ai.hermes.gateway \
  | rg 'Label|PID|Program|ProgramArguments|StandardOutPath|StandardErrorPath|LastExitStatus'
```

Only inspect non-secret fields. Do not dump launchd environment.

Logs:

```bash
hermes logs gateway --since 30m --level WARNING
```

Use summaries, not raw secret-bearing dumps.

## Test Additions By Planned Phase

### Phase 2

- Control inventory JSON schema stability.
- Redaction of credential values, with credential/env names emitted only as
  presence booleans where inventory requires them.
- Missing binary detection.
- Plugin enabled/disabled/gated status.
- Quick command risk classification.

### Phase 3

- Built-in memory replace/remove mirrors or reconciles structured facts.
- Memory compaction does not exceed configured limits.
- Forget checklist covers all stores.
- Memory audit output is metadata-only and does not leak memory contents.
- Memory audit reports missing stores and broad permissions without creating
  files.

### Phase 4

- Active launchd label detection.
- Legacy label warning.
- Wrapper path preservation.
- Health detailed 401 treated as expected without auth.
- Redacted incident bundle emits metadata-only local evidence.
- Restart drain behavior where feasible.

Current Phase 4 gateway validation slice:

```bash
scripts/run_tests.sh tests/hermes_cli/test_gateway_validation.py \
  tests/hermes_cli/test_gateway_service.py \
  tests/hermes_cli/test_startup_plugin_gating.py \
  tests/test_project_metadata.py
./venv/bin/python -m pytest \
  tests/gateway/test_api_server.py::TestHealthEndpoint \
  tests/gateway/test_api_server.py::TestHealthDetailedEndpoint
./venv/bin/python -m hermes_cli.main gateway validate --json \
  > /tmp/hermes-gateway-validation.json
python3 -m json.tool /tmp/hermes-gateway-validation.json \
  > /tmp/hermes-gateway-validation.pretty.json
./venv/bin/python -m hermes_cli.main gateway validate --markdown --no-health \
  > /tmp/hermes-gateway-validation.md
rm -rf /tmp/hermes-gateway-incident-phase4
./venv/bin/python -m hermes_cli.main gateway incident-bundle \
  --output /tmp/hermes-gateway-incident-phase4 --force --json \
  > /tmp/hermes-gateway-incident-phase4-result.json
python3 -m json.tool /tmp/hermes-gateway-incident-phase4-result.json \
  > /tmp/hermes-gateway-incident-phase4-result.pretty.json
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
curl -sS -i --max-time 3 http://127.0.0.1:8642/health
curl -sS -i --max-time 3 http://127.0.0.1:8642/health/detailed
```

Expected:

- Focused gateway validation/service/startup tests pass.
- API health/auth tests pass through `./venv/bin/python` when the canonical
  runner selects a Python environment without `aiohttp`.
- `gateway validate --json` exits zero when there are no critical startup
  errors; a loaded legacy label is a warning, not a failure.
- `/health` returns HTTP 200.
- `/health/detailed` returns HTTP 200 when unauthenticated detailed health is
  allowed, or HTTP 401/403 when bearer auth is configured. The latter is
  expected and must not be treated as a broken gateway.
- Incident bundle smoke creates `manifest.json`, `gateway_validation.json`,
  `artifact_metadata.json`, and `summary.md` in a private output directory.
  Bundle output reports `runtime_mutation: false`,
  `raw_log_content_copied: false`, and `private_memory_read: false`.
- Secret-pattern scan over generated incident bundle files returns no matches;
  `rg` exit code 1 is expected when no matching secret pattern exists.
- No raw launchd environment, bearer token, or secret value is printed.

### Phase 5

- Session/request dump mode `0600`.
- Audit JSONL mode `0600`.
- Approval hooks record redacted audit entries.
- High-risk tools require typed confirmation.

Current Phase 5 private-artifact slice:

```bash
scripts/run_tests.sh \
  tests/hermes_cli/test_private_artifacts.py \
  tests/hermes_cli/test_sessions_export_permissions.py \
  tests/tools/test_delegate_subagent_timeout_diagnostic.py \
  tests/agent/test_redact.py \
  tests/tools/test_command_guards.py \
  tests/test_project_metadata.py
tmpdir=$(mktemp -d /tmp/hermes-phase5-export.XXXXXX)
HERMES_HOME="$tmpdir" ./venv/bin/python -m hermes_cli.main sessions export \
  "$tmpdir/export/sessions.jsonl"
stat -f '%Sp %N' "$tmpdir/export" "$tmpdir/export/sessions.jsonl"
```

Expected:

- Session export output file mode is `-rw-------`.
- Newly created session export parent directory mode is `drwx------`.
- Subagent timeout diagnostic files are `0600` in tests.
- Secret-pattern scan over touched files and generated temp artifacts returns
  no matches.

Current Phase 5 approval-audit slice:

```bash
./venv/bin/python -m py_compile hermes_cli/audit_log.py tools/approval.py tools/slash_confirm.py
scripts/run_tests.sh \
  tests/hermes_cli/test_audit_log.py \
  tests/tools/test_approval_audit.py \
  tests/tools/test_slash_confirm_audit.py
scripts/run_tests.sh \
  tests/hermes_cli/test_audit_log.py \
  tests/tools/test_approval_audit.py \
  tests/tools/test_slash_confirm_audit.py \
  tests/hermes_cli/test_private_artifacts.py \
  tests/hermes_cli/test_sessions_export_permissions.py \
  tests/tools/test_delegate_subagent_timeout_diagnostic.py \
  tests/agent/test_redact.py \
  tests/tools/test_command_guards.py \
  tests/tools/test_approval.py \
  tests/tools/test_approval_plugin_hooks.py \
  tests/tools/test_cron_approval_mode.py \
  tests/tools/test_yolo_mode.py \
  tests/test_project_metadata.py
scripts/run_tests.sh \
  tests/gateway/test_destructive_slash_confirm.py \
  tests/gateway/test_session_boundary_security_state.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/gateway/test_telegram_slash_confirm.py \
  tests/cli/test_destructive_slash_confirm.py \
  tests/cli/test_update_command.py \
  tests/hermes_cli/test_destructive_slash_confirm_gate.py
```

Expected:

- Focused audit tests pass.
- Expanded tracked Phase 5 security tests pass.
- Gateway/CLI slash-confirm tests pass.
- Audit JSONL parent directories are `drwx------`.
- Audit JSONL files are `-rw-------`, including under `umask 000`.
- Approval request, approval, denial, block, skipped/bypass, and slash-confirm
  events are logged as structured JSONL.
- Near-miss approval/confirmation choices fail closed.
- Audit events do not contain raw secret-shaped values or raw session keys.
- Secret-pattern scan over touched files and generated temp audit artifacts
  returns no matches. `rg` exit code 1 is expected when no match exists.

Current Phase 5 risk-policy typed-confirmation slice:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/security_policy.py \
  hermes_cli/control.py \
  tools/approval.py \
  cli.py \
  hermes_cli/callbacks.py \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py
./venv/bin/python -m pytest \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_approval_audit.py \
  tests/hermes_cli/test_control_inventory.py \
  -q
./venv/bin/python -m pytest \
  tests/tools/test_approval.py \
  tests/tools/test_command_guards.py \
  tests/tools/test_yolo_mode.py \
  tests/tools/test_hardline_blocklist.py \
  tests/tools/test_approval_plugin_hooks.py \
  tests/tools/test_slash_confirm_audit.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/cli/test_cli_approval_ui.py \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_approval_audit.py \
  tests/hermes_cli/test_control_inventory.py \
  -q
```

Expected:

- Read-only actions do not request typed confirmation.
- High-risk CLI command actions require exact typed confirmation.
- Credential/token/key-like output destinations require exact typed
  confirmation.
- Near-miss, empty, alias, and loose yes/no confirmations are denied.
- Noninteractive and gateway button approval paths block typed-confirmation
  classes until exact typed gateway input exists.
- `HERMES_YOLO_MODE`, session-scoped yolo, and `approvals.mode: off` do not
  bypass typed-confirmation-required classes.
- Approved, denied, and blocked outcomes are audited through private JSONL.
- Audit JSONL files remain `-rw-------` under permissive `umask`.
- Audit events do not contain raw fake secrets or raw session keys.

Current Phase 5 Docker/container permission review slice:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/docker_security.py \
  hermes_cli/control.py \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_control_inventory.py
./venv/bin/python -m pytest \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_control_inventory.py \
  -q
./venv/bin/python -m pytest \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_control_inventory.py \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_approval_audit.py \
  tests/hermes_cli/test_audit_log.py \
  tests/tools/test_yolo_mode.py \
  tests/tools/test_hardline_blocklist.py \
  tests/tools/test_docker_environment.py \
  tests/tools/test_terminal_config_env_sync.py \
  -q
./venv/bin/python -m hermes_cli.main control inventory \
  --json --redact --no-runtime --no-tool-probe \
  > /tmp/hermes-docker-control-inventory.json
python3 -m json.tool /tmp/hermes-docker-control-inventory.json \
  > /tmp/hermes-docker-control-inventory.pretty.json
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

Expected:

- Docker analyzer tests use synthetic command/config inputs only.
- Analyzer output includes finding codes and redacted details, not env values,
  Docker config content, Docker daemon output, or private files.
- `container_backend.docker` appears in control inventory with
  `docker_security` summary metadata.
- MCP and quick-command inventory entries include Docker review findings only
  when command text invokes Docker or Podman.
- High-severity Docker findings are marked typed-confirmation candidates in
  inventory policy metadata, but no Docker config or backend runtime behavior
  is mutated by this slice.
- Secret-pattern scan over changed code, docs, tests, generated inventory, and
  rollback patch returns no matches. `rg` exit code 1 is expected when no
  matching secret pattern exists.

Current Phase 5 Docker/container enforcement scaffold slice:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/docker_security.py \
  tools/environments/docker.py \
  tests/tools/test_docker_environment.py \
  tests/hermes_cli/test_docker_security.py
./venv/bin/python -m pytest \
  tests/tools/test_docker_environment.py \
  tests/hermes_cli/test_docker_security.py \
  -q
./venv/bin/python -m pytest \
  tests/tools/test_docker_environment.py \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_control_inventory.py \
  tests/tools/test_parse_env_var.py \
  tests/tools/test_file_tools_container_config.py \
  tests/tools/test_terminal_config_env_sync.py \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_approval_audit.py \
  tests/hermes_cli/test_audit_log.py \
  tests/tools/test_yolo_mode.py \
  tests/tools/test_hardline_blocklist.py \
  -q
./venv/bin/python -m hermes_cli.main control inventory \
  --json --redact --no-runtime --no-tool-probe \
  > /tmp/hermes-docker-enforcement-inventory.json
python3 -m json.tool /tmp/hermes-docker-enforcement-inventory.json \
  > /tmp/hermes-docker-enforcement-inventory.pretty.json
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

Expected:

- High/critical Docker findings raise before Docker is probed or `docker run`
  is built.
- Blocked error messages include finding codes/counts only.
- Sensitive env names and raw host paths are not emitted in block messages.
- Medium findings remain non-blocking for this slice.
- Existing safe Docker backend tests continue to pass.
- Generated inventory remains parseable and redacted.
- Secret-pattern scan over changed code, docs, tests, generated inventory, and
  rollback patch returns no matches.

Current Phase 5 Docker diagnostic redaction slice:

```bash
./venv/bin/python -m py_compile \
  tools/environments/docker.py \
  tests/tools/test_docker_environment.py \
  hermes_cli/docker_security.py \
  tests/hermes_cli/test_docker_security.py
./venv/bin/python -m pytest \
  tests/tools/test_docker_environment.py \
  tests/hermes_cli/test_docker_security.py \
  -q
./venv/bin/python -m pytest \
  tests/tools/test_docker_environment.py \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_control_inventory.py \
  tests/tools/test_parse_env_var.py \
  tests/tools/test_file_tools_container_config.py \
  tests/tools/test_terminal_config_env_sync.py \
  tests/hermes_cli/test_security_policy.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_approval_audit.py \
  tests/hermes_cli/test_audit_log.py \
  tests/tools/test_yolo_mode.py \
  tests/tools/test_hardline_blocklist.py \
  -q
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

Expected:

- Blocked high-risk Docker volume config does not log raw host paths before
  enforcement.
- Docker run diagnostics redact env values and host paths while preserving the
  actual execution argument list.
- Docker startup failure exceptions redact argv and omit raw stderr while the
  actual subprocess receives the original argv.
- Docker availability preflight failure logs omit raw `docker version` stderr.
- Secret-pattern scan over changed code, docs, tests, and rollback patch
  returns no matches.

### Phase 6

- `hermes ops status` or equivalent status command is redacted.
- Status includes gateway, health guardian, cron, API, disk, logs, and receipt
  paths.
- `hermes ops status --json --no-health` output parses as JSON, is read-only,
  and does not include raw log lines, cron prompt bodies, private memory,
  provider facts, env values, or secret canaries.
- `hermes ops status` with local health probing preserves the
  wrapper-backed `ai.hermes.gateway` path and treats authenticated
  `/health/detailed` responses as expected.
- Focused tests cover parser registration, invalid timeout failure messages,
  metadata-only cron/log/health-loop summaries, gateway-path redaction, and
  secret-like content embedded in logs, cron jobs, and receipts.
- `hermes ops status --markdown --no-health` produces a Markdown receipt from
  the same redacted payload as text/JSON and does not include raw logs, cron
  prompt bodies, private memory, env values, provider facts, or fake secret
  canaries.

### Phase 7

- Repo-native skills must have valid frontmatter with `name` and
  `description`.
- Hermes ops/review skills must reference `hermes ops status --markdown` and
  `hermes ops status --json --no-health` for redacted receipts.
- Workflow skills must preserve `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Workflow skills must explicitly avoid mutating private memory, live logs,
  caches, provider facts, credentials, Docker config, and launchd state.
- Focused tests should verify required commands and reject live-mutation
  command suggestions such as launchd restart or destructive git cleanup.
- Final-report workflow tests should verify the report anatomy, redacted final
  evidence commands, Telegram delivery gates, and absence of token-dumping
  instructions.

### Phase 8

Final integration validation should use the strongest practical focused suite
that avoids unsafe external effects and known unrelated collection failures:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/audit_log.py hermes_cli/control.py hermes_cli/docker_security.py \
  hermes_cli/gateway_incident.py hermes_cli/gateway_validation.py \
  hermes_cli/main.py hermes_cli/memory_audit.py hermes_cli/ops_status.py \
  hermes_cli/private_artifacts.py hermes_cli/security_policy.py \
  tools/approval.py tools/environments/docker.py tools/slash_confirm.py
scripts/run_tests.sh \
  tests/hermes_cli/test_phase7_hermes_final_report_skill.py \
  tests/hermes_cli/test_phase7_hermes_ops_skill.py \
  tests/hermes_cli/test_ops_status.py \
  tests/test_project_metadata.py
scripts/run_tests.sh \
  tests/hermes_cli/test_memory_audit.py \
  tests/hermes_cli/test_control_inventory.py \
  tests/hermes_cli/test_gateway_validation.py \
  tests/hermes_cli/test_gateway_incident.py
scripts/run_tests.sh \
  tests/hermes_cli/test_audit_log.py \
  tests/hermes_cli/test_private_artifacts.py \
  tests/hermes_cli/test_sessions_export_permissions.py \
  tests/hermes_cli/test_security_policy.py \
  tests/hermes_cli/test_docker_security.py
scripts/run_tests.sh \
  tests/tools/test_approval_audit.py \
  tests/tools/test_slash_confirm_audit.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_command_guards.py \
  tests/tools/test_yolo_mode.py \
  tests/tools/test_delegate_subagent_timeout_diagnostic.py \
  tests/tools/test_docker_environment.py
scripts/run_tests.sh \
  tests/gateway/test_destructive_slash_confirm.py \
  tests/gateway/test_session_boundary_security_state.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/gateway/test_telegram_slash_confirm.py \
  tests/cli/test_destructive_slash_confirm.py \
  tests/hermes_cli/test_destructive_slash_confirm_gate.py \
  tests/cli/test_update_command.py \
  tests/cli/test_cli_approval_ui.py \
  tests/cli/test_quick_commands.py
./venv/bin/python -m pytest \
  tests/gateway/test_api_server.py::TestHealthEndpoint \
  tests/gateway/test_api_server.py::TestHealthDetailedEndpoint -q
```

Expected:

- Focused integrated test groups pass.
- `hermes ops status --markdown` and `--json --no-health` pass.
- `hermes gateway status` confirms `ai.hermes.gateway` uses
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- `hermes doctor` passes, allowing expected optional-provider warnings.
- `hermes memory status` and `hermes memory audit --json/--markdown --redact`
  pass without printing private memory content.
- A fresh missing `HERMES_HOME` memory-audit smoke does not initialize runtime
  state.
- `hermes gateway validate --json/--markdown` passes.
- `hermes gateway incident-bundle --json` creates a private directory and
  `0600` files, with `runtime_mutation: false`, `raw_log_content_copied:
  false`, and `private_memory_read: false`.
- `/health` returns HTTP 200; `/health/detailed` may return 401/403 when
  detailed health auth is configured.
- Final report and handoff artifacts pass targeted secret-pattern scans.
- `git diff --check`, `git diff --cached --check`, and rollback reverse-check
  pass.
- All three final judges pass before Telegram delivery.

### Post-Campaign Cleanup

Raw Codex terminal guard cleanup validation:

```bash
./venv/bin/python -m py_compile \
  tools/terminal_tool.py \
  tests/tools/test_terminal_codex_guard.py
scripts/run_tests.sh tests/tools/test_terminal_codex_guard.py
scripts/run_tests.sh \
  tests/tools/test_terminal_codex_guard.py \
  tests/tools/test_terminal_foreground_timeout_cap.py \
  tests/tools/test_terminal_none_command_guard.py \
  tests/tools/test_command_guards.py \
  tests/tools/test_yolo_mode.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health
```

Expected:

- `tests/tools/test_terminal_codex_guard.py` collects and passes.
- Raw `codex exec`, `npx @openai/codex exec`, and package-manager Codex exec
  paths are blocked before terminal environment creation.
- `/Users/agent1/Operator/scripts/codex-run.sh` remains allowed.
- `HERMES_ALLOW_RAW_CODEX=1` remains the explicit operator override for this
  guard.
- Adjacent terminal foreground guidance, none-command guard, command guards,
  yolo semantics, and metadata tests continue to pass.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.

Telegram delivery dry-run/preflight cleanup validation:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/send_cmd.py \
  tests/hermes_cli/test_send_cmd.py \
  tests/hermes_cli/test_phase7_hermes_final_report_skill.py
scripts/run_tests.sh tests/hermes_cli/test_send_cmd.py
scripts/run_tests.sh \
  tests/hermes_cli/test_send_cmd.py \
  tests/hermes_cli/test_phase7_hermes_final_report_skill.py \
  tests/hermes_cli/test_ops_status.py \
  tests/test_project_metadata.py
bash -lc 'source /Users/agent1/Operator/scripts/hermes-env.sh >/dev/null 2>&1 && ./venv/bin/python -m hermes_cli.main send --to telegram --file docs/HERMES_FINAL_REPORT.md --dry-run --json --output /tmp/hermes-telegram-preflight/receipt.json --quiet'
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health
```

Expected:

- `hermes send --dry-run/--preflight` does not call `send_message_tool`, does
  not contact Telegram, and exits non-zero when required configuration is not
  present.
- JSON preflight output reports `dry_run: true`, `would_send: false`, and
  `network_performed: false`.
- Preflight output summarizes message size/source without printing message
  content.
- Preflight output hashes chat/thread identifiers and does not print raw target
  IDs, tokens, credentials, or private channel values.
- `--output` writes parent directories as `drwx------` and receipt files as
  `-rw-------`, even under permissive umask.
- The final-report skill requires a passing Telegram dry-run preflight before
  live delivery.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.

Private memory compaction approval-plan cleanup validation:

```bash
./venv/bin/python -m py_compile \
  tests/hermes_cli/test_memory_compaction_approval_plan.py
scripts/run_tests.sh \
  tests/hermes_cli/test_memory_compaction_approval_plan.py \
  tests/hermes_cli/test_memory_audit.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main memory audit --json --redact \
  > /tmp/hermes-memory-compaction-plan-audit.json
./venv/bin/python -m json.tool /tmp/hermes-memory-compaction-plan-audit.json \
  > /tmp/hermes-memory-compaction-plan-audit.pretty.json
./venv/bin/python -m hermes_cli.main memory status
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health
```

Expected:

- The cleanup pass does not read, rewrite, compact, delete, backfill, or
  reconcile live private memory.
- `docs/HERMES_MEMORY_PLAN.md` contains the exact current-run read-and-draft
  approval phrase and separate exact apply approval phrase.
- The plan requires owner-only private backups and draft artifacts before any
  approved apply step.
- The plan requires redacted summaries in chat and forbids pasting raw private
  memory contents.
- Memory audit remains metadata-only and redacted.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.

Private memory read-only compaction draft cleanup validation:

```bash
stat -f '%Sp %N' \
  /tmp/hermes-memory-compaction-draft-20260521T114154Z \
  /tmp/hermes-memory-compaction-draft-20260521T114154Z/*
stat -f '%m %z %Sp %N' \
  /Users/agent1/.hermes/memories/MEMORY.md \
  /Users/agent1/.hermes/memories/USER.md
./venv/bin/python -m py_compile \
  tests/hermes_cli/test_memory_compaction_approval_plan.py
scripts/run_tests.sh \
  tests/hermes_cli/test_memory_compaction_approval_plan.py \
  tests/hermes_cli/test_memory_audit.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main memory audit --json --redact \
  > /tmp/hermes-memory-draft-post-audit.json
./venv/bin/python -m hermes_cli.main memory status \
  > /tmp/hermes-memory-draft-memory-status.txt
./venv/bin/python -m hermes_cli.main gateway status \
  > /tmp/hermes-memory-draft-gateway-status.txt
./venv/bin/python -m hermes_cli.main doctor \
  > /tmp/hermes-memory-draft-doctor.txt
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health \
  > /tmp/hermes-memory-draft-ops-status.md
```

Expected:

- The draft step reads only the default approved memory files:
  `~/.hermes/memories/MEMORY.md` and `~/.hermes/memories/USER.md`.
- The draft step writes only private draft artifacts outside the repo and does
  not write back to either live source file.
- Draft parent directories are `drwx------` and draft files are `-rw-------`,
  including under permissive caller umask.
- Live source file mode, size, and mtime remain unchanged after draft
  generation.
- The manifest records `applied: false` and `live_memory_mutated: false`.
- Generated draft artifacts, redacted summaries, smoke outputs, changed docs,
  and rollback patches pass targeted secret-pattern scans without printing raw
  private memory contents.
- Applying the draft remains blocked until the second exact approval phrase is
  provided in a future current run.

Private memory compaction apply cleanup validation:

```bash
stat -f '%Sp %N' \
  /tmp/hermes-memory-compaction-apply-20260521T114955Z \
  /tmp/hermes-memory-compaction-apply-20260521T114955Z/* \
  /Users/agent1/.hermes/memories/MEMORY.md \
  /Users/agent1/.hermes/memories/USER.md
./venv/bin/python -m py_compile \
  tests/hermes_cli/test_memory_compaction_approval_plan.py
scripts/run_tests.sh \
  tests/hermes_cli/test_memory_compaction_approval_plan.py \
  tests/hermes_cli/test_memory_audit.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main memory audit --json --redact \
  > /tmp/hermes-memory-apply-post-audit.json
./venv/bin/python -m hermes_cli.main memory status \
  > /tmp/hermes-memory-apply-status.txt
./venv/bin/python -m hermes_cli.main gateway status \
  > /tmp/hermes-memory-apply-gateway-status.txt
./venv/bin/python -m hermes_cli.main doctor \
  > /tmp/hermes-memory-apply-doctor.txt
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health \
  > /tmp/hermes-memory-apply-ops-status.md
```

Expected:

- Apply proceeds only after the exact apply approval phrase appears in the
  current run.
- Before any live write, owner-only backups are created with `0700` parent
  directory and `0600` backup files.
- The current live source hash must match the draft manifest source hash
  before applying.
- The reviewed draft hash must match the draft manifest hash before applying.
- After applying, live files match the reviewed draft, backups match pre-apply
  bytes, and live files remain `-rw-------`.
- Memory audit remains metadata-only and redacted.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Secret scans over changed docs, rollback patches, private receipts, draft
  artifacts, backups, and generated smoke outputs report no credential-shaped
  matches without printing private memory content.

Fallback OpenRouter routing cleanup validation:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/fallback_cmd.py \
  hermes_cli/main.py \
  tests/hermes_cli/test_fallback_cmd.py
scripts/run_tests.sh \
  tests/hermes_cli/test_fallback_cmd.py \
  tests/run_agent/test_fallback_model.py \
  tests/run_agent/test_provider_fallback.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main fallback configure-openrouter --dry-run \
  > /tmp/hermes-fallback-openrouter-dry-run.txt
./venv/bin/python -m hermes_cli.main fallback configure-openrouter \
  > /tmp/hermes-fallback-openrouter-apply.txt
./venv/bin/python -m hermes_cli.main fallback list \
  > /tmp/hermes-fallback-openrouter-after.txt
./venv/bin/python -m hermes_cli.main gateway status \
  > /tmp/hermes-fallback-openrouter-gateway-status.txt
./venv/bin/python -m hermes_cli.main doctor \
  > /tmp/hermes-fallback-openrouter-doctor.txt
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health \
  > /tmp/hermes-fallback-openrouter-ops-status.md
```

Expected:

- `hermes fallback configure-openrouter --dry-run` shows the planned chain and
  writes no config changes.
- Before live config mutation, `~/.hermes/config.yaml` is backed up under an
  owner-only directory.
- The saved fallback chain has OpenRouter as the first fallback with
  `google/gemini-3-flash-preview` and no legacy `fallback_model` key.
- The operation is idempotent when re-run.
- Config file and backup permissions remain `-rw-------`.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Secret scans over changed code, docs, rollback patch, config backup,
  generated receipts, and the live config report no credential-shaped matches
  without printing raw config values.

Deployment-readiness refresh validation:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/fallback_cmd.py \
  hermes_cli/main.py \
  hermes_cli/ops_status.py \
  hermes_cli/memory_audit.py
scripts/run_tests.sh \
  tests/hermes_cli/test_fallback_cmd.py \
  tests/hermes_cli/test_ops_status.py \
  tests/hermes_cli/test_gateway_validation.py \
  tests/hermes_cli/test_memory_audit.py \
  tests/hermes_cli/test_memory_compaction_approval_plan.py \
  tests/hermes_cli/test_phase7_hermes_final_report_skill.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main fallback list \
  > /tmp/hermes-deploy-readiness-fallback.txt
./venv/bin/python -m hermes_cli.main gateway status \
  > /tmp/hermes-deploy-readiness-gateway-status.txt
./venv/bin/python -m hermes_cli.main doctor \
  > /tmp/hermes-deploy-readiness-doctor.txt
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health \
  > /tmp/hermes-deploy-readiness-ops-status.md
./venv/bin/python -m hermes_cli.main memory audit --json --redact \
  > /tmp/hermes-deploy-readiness-memory-audit.json
```

Expected:

- Final report no longer claims fallback providers are unconfigured.
- Final report records that default private memory compaction was applied and
  that additional stores require separate explicit approval.
- Fallback list shows OpenRouter first with `google/gemini-3-flash-preview`.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Doctor passes with only expected optional-provider warnings.
- Ops status is redacted and reports no raw log lines.
- Memory audit remains metadata-only and redacted.
- Secret scans over changed docs, rollback patch, and generated receipts report
  no credential-shaped matches.

Final repository handoff validation:

```bash
./venv/bin/python -m py_compile \
  hermes_cli/fallback_cmd.py \
  hermes_cli/main.py \
  hermes_cli/ops_status.py \
  hermes_cli/memory_audit.py \
  hermes_cli/control.py \
  hermes_cli/gateway_validation.py \
  hermes_cli/gateway_incident.py \
  hermes_cli/audit_log.py \
  hermes_cli/private_artifacts.py \
  hermes_cli/security_policy.py \
  hermes_cli/docker_security.py \
  hermes_cli/send_cmd.py
scripts/run_tests.sh \
  tests/hermes_cli/test_fallback_cmd.py \
  tests/hermes_cli/test_ops_status.py \
  tests/hermes_cli/test_gateway_validation.py \
  tests/hermes_cli/test_gateway_incident.py \
  tests/hermes_cli/test_memory_audit.py \
  tests/hermes_cli/test_memory_compaction_approval_plan.py \
  tests/hermes_cli/test_phase7_hermes_final_report_skill.py \
  tests/hermes_cli/test_phase7_hermes_ops_skill.py \
  tests/hermes_cli/test_security_policy.py \
  tests/hermes_cli/test_docker_security.py \
  tests/hermes_cli/test_audit_log.py \
  tests/hermes_cli/test_private_artifacts.py \
  tests/hermes_cli/test_sessions_export_permissions.py \
  tests/hermes_cli/test_send_cmd.py \
  tests/hermes_cli/test_control_inventory.py \
  tests/tools/test_approval_audit.py \
  tests/tools/test_slash_confirm_audit.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_terminal_codex_guard.py \
  tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main fallback list
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
./venv/bin/python -m hermes_cli.main ops status --markdown --no-health
./venv/bin/python -m hermes_cli.main memory audit --json --redact
git diff --check
git diff --cached --check
```

Expected:

- Focused tests pass.
- Gateway status preserves `ai.hermes.gateway` and
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- Doctor passes with only expected optional-provider warnings.
- Ops status and memory audit remain redacted and do not print raw logs or raw
  private memory.
- Handoff docs identify commit-candidate paths and excluded local artifacts.
- Targeted secret scans over changed docs, changed code/tests, generated smoke
  receipts, and rollback patch report no credential-shaped matches.

## Judge Validation

Before any phase is marked complete:

1. Run the relevant test tier.
2. Run Hermes smoke checks.
3. Update `docs/HERMES_BUILD_LOG.md`.
4. Run judge cycle.
5. Fix required judge failures.
6. Repeat judge cycle.

## Current Phase 2 Validation Plan

Commands:

```bash
scripts/run_tests.sh tests/hermes_cli/test_control_inventory.py tests/hermes_cli/test_startup_plugin_gating.py tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main control inventory --json --redact > /tmp/hermes-control-inventory.json
python3 -m json.tool /tmp/hermes-control-inventory.json > /tmp/hermes-control-inventory.pretty.json
./venv/bin/python -m hermes_cli.main control inventory --markdown --redact --no-runtime --no-tool-probe > /tmp/hermes-control-inventory.md
rg -n "sk-[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[0-9]{7,}:[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{12,}" /tmp/hermes-control-inventory.json /tmp/hermes-control-inventory.md
./venv/bin/python - <<'PY'
import json
from hermes_cli.control import _secret_scan_inventory

inventory = json.load(open("/tmp/hermes-control-inventory.json"))
findings = _secret_scan_inventory(inventory)
if findings:
    raise SystemExit(f"inventory secret scan findings: {findings}")
PY
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

Expected:

- Focused tests pass.
- Inventory JSON parses cleanly.
- Markdown inventory renders without executing runtime probes when requested.
- Secret-pattern scan returns no matches; `rg` exit code 1 is expected when no
  matching secret pattern exists.
- Internal inventory secret scan returns no findings for credential-shaped
  env assignments/options, bearer values, URL passwords, and known provider
  token patterns.
- Doctor does not report broken core setup.
- Gateway status shows active wrapper-backed service.

## Current Phase 3 Validation Plan

Commands:

```bash
scripts/run_tests.sh tests/hermes_cli/test_memory_audit.py tests/tools/test_memory_tool.py tests/agent/test_memory_provider.py tests/test_project_metadata.py
./venv/bin/python -m hermes_cli.main memory audit --json --redact > /tmp/hermes-memory-audit.json
python3 -m json.tool /tmp/hermes-memory-audit.json > /tmp/hermes-memory-audit.pretty.json
./venv/bin/python -m hermes_cli.main memory audit --markdown --redact > /tmp/hermes-memory-audit.md
tmp_home=$(mktemp -d /tmp/hermes-audit-home.XXXXXX); rm -rf "$tmp_home"; HERMES_HOME="$tmp_home" PYTHONDONTWRITEBYTECODE=1 ./venv/bin/python -m hermes_cli.main memory audit --json --redact > /tmp/hermes-memory-audit-fresh-home.json; test ! -e "$tmp_home"
rg -n "sk-[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[0-9]{7,}:[A-Za-z0-9_-]{20,}|Bearer [A-Za-z0-9._-]{12,}" /tmp/hermes-memory-audit.json /tmp/hermes-memory-audit.md
./venv/bin/python -m hermes_cli.main memory status
./venv/bin/python -m hermes_cli.main gateway status
./venv/bin/python -m hermes_cli.main doctor
```

Expected:

- Focused memory/audit tests pass.
- Audit JSON parses cleanly and stays metadata-only.
- Audit CLI does not initialize or create a missing `HERMES_HOME`.
- Secret-pattern scan returns no matches; `rg` exit code 1 is expected when no
  matching secret pattern exists.
- `memory status` remains functional.
- Gateway status shows active wrapper-backed service.
- Doctor does not report broken core setup.
