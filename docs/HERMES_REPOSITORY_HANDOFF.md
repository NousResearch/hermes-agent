# Hermes Repository Handoff

Date: 2026-05-21
Branch: `hermes-control-plane-20260520-182036`
Repo: `/Users/agent1/Code/hermes-agent`

## Purpose

This handoff prepares the Hermes optimization campaign for a clean commit or
PR review without exposing secrets, private memory, raw logs, credentials, or
unrelated local artifacts.

The initial handoff did not include external push or pull request creation.
After explicit approval, the campaign branch was pushed to the fork and opened
as upstream PR #29835.

## Current State

- Phase 0 through Phase 8 are complete and judged PASS.
- Post-campaign cleanup covered raw Codex guard collection, Telegram dry-run
  preflight, private-memory compaction planning/draft/apply, OpenRouter
  fallback routing, deployment-readiness refresh, and this repository handoff.
- The wrapper-backed `ai.hermes.gateway` path remains the required startup
  path:
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- The dirty tree contains intended campaign code/docs/tests plus unrelated
  local artifacts that should be excluded from a commit/PR.

## Commit Candidate Scope

Review and intentionally stage these campaign-owned paths for a future commit
or PR:

```text
AGENTS.md
.agents/skills/hermes-final-report/SKILL.md
.agents/skills/hermes-ops-review/SKILL.md
docs/HERMES_*.md
docs/HERMES_DEVELOPER_GUIDE.md
docs/HERMES_OPERATOR_QUICKSTART.md
agent/image_gen_provider.py
cli.py
gateway/platforms/api_server.py
gateway/run.py
hermes_cli/*.py
tests/cli/test_cli_approval_ui.py
tests/cli/test_quick_commands.py
tests/gateway/test_api_server.py
tests/gateway/test_approve_deny_commands.py
tests/hermes_cli/*.py
tests/plugins/image_gen/test_openai_codex_provider.py
tests/tools/test_approval_audit.py
tests/tools/test_command_guards.py
tests/tools/test_delegate_subagent_timeout_diagnostic.py
tests/tools/test_docker_environment.py
tests/tools/test_hardline_blocklist.py
tests/tools/test_risk_typed_confirmation.py
tests/tools/test_slash_confirm_audit.py
tests/tools/test_terminal_codex_guard.py
tests/tools/test_yolo_mode.py
tools/approval.py
tools/delegate_tool.py
tools/environments/docker.py
tools/image_generation_tool.py
tools/slash_confirm.py
tools/terminal_tool.py
```

The current index has a mixed staged/unstaged state from prior staged-safety
work. Do not commit the current index blindly. Refresh staging intentionally
after review.

## Excluded Local Artifacts

Keep these out of the commit/PR unless a future review explicitly decides
otherwise:

```text
.codex-backups/
.playwright-mcp/
citelocal-*.png
tinker-atropos/
```

Rationale:

- `.codex-backups/` contains local rollback patches for this workstation.
- `.playwright-mcp/` contains local browser automation logs/page snapshots.
- `citelocal-*.png` are unrelated local screenshot artifacts.
- `tinker-atropos/` is outside the Hermes campaign scope.

## Suggested Local Review Commands

```bash
cd /Users/agent1/Code/hermes-agent
git status --short --branch
git diff --name-status
git diff --cached --name-status
git ls-files --others --exclude-standard
git diff --check
git diff --cached --check
```

## Suggested Intentional Staging Command

Run this only after the reviewer confirms the campaign scope:

```bash
git add \
  AGENTS.md \
  .agents/skills/hermes-final-report/SKILL.md \
  .agents/skills/hermes-ops-review/SKILL.md \
  docs/HERMES_*.md \
  docs/HERMES_DEVELOPER_GUIDE.md \
  docs/HERMES_OPERATOR_QUICKSTART.md \
  agent/image_gen_provider.py \
  cli.py \
  gateway/platforms/api_server.py \
  gateway/run.py \
  hermes_cli/*.py \
  tests/cli/test_cli_approval_ui.py \
  tests/cli/test_quick_commands.py \
  tests/gateway/test_api_server.py \
  tests/gateway/test_approve_deny_commands.py \
  tests/hermes_cli/*.py \
  tests/plugins/image_gen/test_openai_codex_provider.py \
  tests/tools/test_approval_audit.py \
  tests/tools/test_command_guards.py \
  tests/tools/test_delegate_subagent_timeout_diagnostic.py \
  tests/tools/test_docker_environment.py \
  tests/tools/test_hardline_blocklist.py \
  tests/tools/test_risk_typed_confirmation.py \
  tests/tools/test_slash_confirm_audit.py \
  tests/tools/test_terminal_codex_guard.py \
  tests/tools/test_yolo_mode.py \
  tools/approval.py \
  tools/delegate_tool.py \
  tools/environments/docker.py \
  tools/image_generation_tool.py \
  tools/slash_confirm.py \
  tools/terminal_tool.py
```

Then verify the staged set:

```bash
git diff --cached --name-status
git diff --cached --check
```

## Suggested Commit Message

```text
Complete Hermes control plane, security, ops, and validation campaign
```

## PR-Ready Summary

Title:

```text
Complete Hermes control plane, security, ops, and validation campaign
```

Body:

```text
## Summary

- Add the Hermes optimization control plane docs, target architecture, judge
  rubric, build log, testing plan, security model, memory plan, tool registry,
  final report, and repository handoff.
- Add redacted control inventory, memory audit, gateway startup validation,
  incident bundle, ops status, final-report workflow, Telegram dry-run
  preflight, fallback configuration, private artifact helpers, audit JSONL
  hardening, typed confirmation policy, Docker/container security review, and
  Docker enforcement scaffolding.
- Preserve the wrapper-backed ai.hermes.gateway startup path at
  /Users/agent1/Operator/scripts/hermes-gateway.sh.
- Add focused tests and smoke coverage for the new CLI, security, memory,
  ops, Docker, approval, confirmation, and workflow surfaces.

## Validation

- py_compile for changed Hermes Python modules and focused tests
- focused Hermes handoff test suite
- hermes fallback list
- hermes gateway status
- hermes doctor
- hermes ops status --markdown --no-health
- hermes memory audit --json --redact
- targeted secret-pattern scan
- git diff --check
- git diff --cached --check
- rollback reverse-check for this handoff doc/update patch

## Safety Notes

- No secrets, tokens, credentials, private keys, raw private memory, raw log
  lines, raw Telegram identifiers, or provider facts are committed.
- Telegram delivery remains gated by dry-run preflight and explicit live-send
  approval.
- Private memory mutation remains approval-gated and out of scope for normal
  repository work.
- Exclude .codex-backups/, .playwright-mcp/, citelocal-*.png, and
  tinker-atropos/ from the PR.
```

## External Push Status

External push and PR creation were approved after this handoff was prepared.
The fork branch is `rke6693:hermes-control-plane-20260520-182036`, and the
upstream pull request is NousResearch/hermes-agent#29835. Do not merge the PR
or perform additional account-side actions without explicit approval and
available repository permissions.

## Latest Local Validation

Status: passed on 2026-05-21 during the repository handoff pass.

- `py_compile` for the handoff-relevant Hermes CLI modules and tests: passed.
- Focused handoff suite: 191 passed after adding the post-restart gateway
  validation case.
- Follow-up control-inventory fixture suite after static-scan hardening:
  18 passed.
- `hermes fallback list`: passed; OpenRouter fallback remains configured as
  the first fallback with `google/gemini-3-flash-preview`.
- `hermes gateway status`: passed; active label remains
  `ai.hermes.gateway`, wrapper backed by
  `/Users/agent1/Operator/scripts/hermes-gateway.sh`.
- `hermes doctor`: passed with expected optional-provider warnings only.
- `hermes gateway validate --markdown --no-health`: passed; a currently
  running wrapper-backed service with a previous non-zero exit status is a
  warning, not a startup failure.
- `hermes ops status --markdown --no-health`: passed with overall `PASS` and
  redacted log metadata/counts only.
- `hermes memory audit --json --redact`: passed with `metadata_only: true`,
  `redacted: true`, 22 stores, and no recommended actions.
- Targeted high-confidence credential scan over changed docs/code/tests,
  generated handoff receipts, and rollback patch: 53 files scanned, 0 matches.
- `git diff --check`: passed.
- `git diff --cached --check`: passed.
- Rollback reverse-check for
  `.codex-backups/final-repository-handoff-20260521.patch`: passed.

External push and PR creation were later performed with explicit approval.
Telegram send, private-memory mutation, live log rewrite, credential mutation,
Docker config mutation, and provider-fact mutation were not performed.
