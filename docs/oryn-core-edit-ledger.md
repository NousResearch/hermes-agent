# Oryn Core-Edit Ledger

Generated: 2026-05-29

Baseline: `upstream/main...HEAD`

Purpose: identify the Oryn fork's real upstream-merge conflict surface and keep
Oryn-owned harness logic behind documented seams.

## Summary

- Modified upstream-existing files: 39
- Oryn-new files: 78, including this ledger,
  `gateway/dev_control/routes.py`, `agent/tool_recovery.py`, and upstream PR
  preparation notes under `docs/upstream-prs/`
- Primary core conflict surface before this refactor:
  `gateway/platforms/api_server.py`
- Dev-control/project-dashboard route seam after this refactor:
  `gateway.dev_control.routes.register_dev_control_routes(self._app, self)`

## Modified Upstream-Core Files

These files exist in `upstream/main` and can collide with future upstream pulls.

| Path | Classification | Notes |
| --- | --- | --- |
| `agent/agent_init.py` | Modified core | Oryn integration hook. |
| `agent/agent_runtime_helpers.py` | Modified core | Oryn runtime/helper integration. |
| `agent/conversation_loop.py` | Modified core | Oryn runtime behavior integration; context usage is reduced to one documented telemetry call, and empty-tool recovery delegates to `agent.tool_recovery.empty_tool_result_recovery_response`. |
| `agent/credential_pool.py` | Modified core | Credential-pool hardening. |
| `agent/message_sanitization.py` | Modified core | Message redaction/sanitization hardening; chat-completions wire cleanup now lives here instead of inline transport logic. |
| `agent/tool_dispatch_helpers.py` | Modified core | Tool dispatch integration. |
| `agent/tool_executor.py` | Modified core | Tool execution integration. |
| `agent/transports/chat_completions.py` | Modified core | Chat-completions transport integration; transport retains a one-line call into `agent.message_sanitization.sanitize_chat_completion_messages_for_wire`. |
| `cli.py` | Modified core | CLI integration. |
| `gateway/platforms/api_server.py` | Modified core | Thin Oryn route/capability/read-model seams remain; `/v1/dev/*` and `/v1/oryn/project-dashboard` handlers moved to `gateway/dev_control/routes.py`. |
| `gateway/run.py` | Modified core | Gateway runtime seam; keep minimal and re-check on upstream pulls. |
| `gateway/status.py` | Modified core | Status/readiness integration. |
| `hermes_cli/commands.py` | Modified core | CLI command integration. |
| `hermes_cli/config.py` | Modified core | CLI config integration. |
| `hermes_cli/env_loader.py` | Modified core | Environment loading integration. |
| `hermes_cli/gateway.py` | Modified core | Gateway CLI integration. |
| `hermes_cli/kanban_db.py` | Modified core | Kanban integration. |
| `hermes_cli/main.py` | Modified core | CLI entrypoint integration. |
| `hermes_cli/models.py` | Modified core | CLI model integration. |
| `plugins/kanban/dashboard/plugin_api.py` | Modified core | Kanban dashboard integration. |
| `pyproject.toml` | Modified core | Test/lint/runtime dependency configuration. |
| `run_agent.py` | Modified core | Run-agent integration. |
| `tests/agent/transports/test_chat_completions.py` | Modified upstream test | Covers transport integration. |
| `tests/cli/test_surrogate_sanitization.py` | Modified upstream test | Covers sanitization integration. |
| `tests/gateway/test_api_server.py` | Modified upstream test | Covers API-server integration. |
| `tests/gateway/test_api_server_runs.py` | Modified upstream test | Covers API-server run/dev-control integration. |
| `tests/gateway/test_session_api.py` | Modified upstream test | Covers session API integration. |
| `tests/gateway/test_status.py` | Modified upstream test | Covers status integration. |
| `tests/hermes_cli/test_gateway_service.py` | Modified upstream test | Covers gateway CLI integration. |
| `tests/hermes_cli/test_kanban_core_functionality.py` | Modified upstream test | Covers kanban CLI integration. |
| `tests/hermes_cli/test_models.py` | Modified upstream test | Covers CLI model integration. |
| `tests/run_agent/test_run_agent.py` | Modified upstream test | Covers run-agent integration. |
| `tests/test_model_tools.py` | Modified upstream test | Covers tool/model integration. |
| `tests/test_toolsets.py` | Modified upstream test | Covers toolset integration. |
| `tests/tools/test_computer_use.py` | Modified upstream test | Covers tool integration. |
| `tools/delegate_tool.py` | Modified core | Delegation tool integration. |
| `toolsets.py` | Modified core | Toolset configuration integration. |
| `uv.lock` | Modified core metadata | Dependency lockfile. |
| `website/docs/user-guide/features/api-server.md` | Modified upstream docs | API-server documentation. |

## Oryn-New Files

These files do not exist in `upstream/main`; they are lower-risk for upstream
merges unless upstream later creates the same paths.

| Path | Classification |
| --- | --- |
| `.regent/.gitignore` | Oryn-new |
| `.regent/config.toml` | Oryn-new |
| `.regent/index.db` | Oryn-new generated state |
| `CLAUDE.md` | Oryn-new |
| `Makefile` | Oryn-new |
| `agent/context_usage.py` | Oryn-new |
| `agent/tool_recovery.py` | Oryn-new |
| `agent/secret_sources/onepassword.py` | Oryn-new |
| `config.py` | Oryn-new |
| `docs/oryn-core-edit-ledger.md` | Oryn-new |
| `docs/upstream-prs/chat-completions-message-sanitization.md` | Oryn-new upstream PR preparation |
| `docs/upstream-prs/empty-tool-result-recovery.md` | Oryn-new upstream PR preparation |
| `docs/superpowers/specs/2026-05-23-scout-evidence-pipeline-design.md` | Oryn-new |
| `gateway/ao_snapshot_cache.py` | Oryn-new |
| `gateway/dev_control/__init__.py` | Oryn-new |
| `gateway/dev_control/acceptance_criteria.py` | Oryn-new |
| `gateway/dev_control/acceptance_verification.py` | Oryn-new |
| `gateway/dev_control/ci_status.py` | Oryn-new |
| `gateway/dev_control/clarifications.py` | Oryn-new |
| `gateway/dev_control/events.py` | Oryn-new |
| `gateway/dev_control/harness_benchmarks.py` | Oryn-new |
| `gateway/dev_control/harness_observability.py` | Oryn-new |
| `gateway/dev_control/harness_recommendations.py` | Oryn-new |
| `gateway/dev_control/incidents.py` | Oryn-new |
| `gateway/dev_control/laminar_exporter.py` | Oryn-new |
| `gateway/dev_control/plan_artifacts.py` | Oryn-new |
| `gateway/dev_control/product_events.py` | Oryn-new |
| `gateway/dev_control/production_signals.py` | Oryn-new |
| `gateway/dev_control/read_models.py` | Oryn-new |
| `gateway/dev_control/repo_grounding.py` | Oryn-new |
| `gateway/dev_control/routes.py` | Oryn-new route seam module |
| `gateway/dev_control/runtime_capabilities.py` | Oryn-new |
| `gateway/dev_control/runtime_policy_evidence.py` | Oryn-new |
| `gateway/dev_control/runtime_selection.py` | Oryn-new |
| `gateway/dev_control/scm_lifecycle.py` | Oryn-new |
| `gateway/dev_control/signal_source.py` | Oryn-new |
| `gateway/dev_control/worker_output_contract.py` | Oryn-new |
| `gateway/dev_execution.py` | Oryn-new |
| `gateway/dev_worker_runtimes.py` | Oryn-new |
| `gateway/platforms/kanban_api_routes.py` | Oryn-new route seam module |
| `gateway/read_model_cache.py` | Oryn-new |
| `gateway/subagent_events.py` | Oryn-new |
| `hermes_cli/case.py` | Oryn-new |
| `hermes_cli/kanban_http.py` | Oryn-new |
| `infra/bootstrap/laminar/.env.example` | Oryn-new |
| `infra/bootstrap/laminar/README.md` | Oryn-new |
| `run.py` | Oryn-new wrapper entrypoint |
| `scripts/compact_trace.py` | Oryn-new |
| `scripts/run_dev_signal_digest.py` | Oryn-new |
| `scripts/smoke_ao_board.py` | Oryn-new |
| `tests/agent/test_ao_delegate_sequential.py` | Oryn-new test |
| `tests/agent/test_context_usage.py` | Oryn-new test |
| `tests/agent/test_message_sanitization.py` | Oryn-new test |
| `tests/agent/test_tool_recovery.py` | Oryn-new test |
| `tests/gateway/test_acceptance_criteria.py` | Oryn-new test |
| `tests/gateway/test_acceptance_verification.py` | Oryn-new test |
| `tests/gateway/test_ao_snapshot_cache.py` | Oryn-new test |
| `tests/gateway/test_api_server_kanban.py` | Oryn-new test |
| `tests/gateway/test_ci_status.py` | Oryn-new test |
| `tests/gateway/test_incidents.py` | Oryn-new test |
| `tests/gateway/test_product_events.py` | Oryn-new test |
| `tests/gateway/test_production_signals.py` | Oryn-new test |
| `tests/gateway/test_scm_lifecycle.py` | Oryn-new test |
| `tests/gateway/test_sse_approvals_integration.py` | Oryn-new test |
| `tests/gateway/test_worker_output_contract.py` | Oryn-new test |
| `tests/hermes_cli/test_case_cli.py` | Oryn-new test |
| `tests/plugins/memory/test_hindsight_consolidation_claims.py` | Oryn-new test |
| `tests/tools/test_ao_bridge.py` | Oryn-new test |
| `tests/tools/test_ao_delegate_tool.py` | Oryn-new test |
| `tests/tools/test_openhands_bridge.py` | Oryn-new test |
| `tests/tools/test_vault_publish_tool.py` | Oryn-new test |
| `tools/ao_bridge.mjs` | Oryn-new |
| `tools/ao_bridge.py` | Oryn-new |
| `tools/ao_delegate_tool.py` | Oryn-new |
| `tools/ao_shims/codex` | Oryn-new |
| `tools/dev_execution_tools.py` | Oryn-new |
| `tools/openhands_bridge.py` | Oryn-new |
| `tools/vault_publish_tool.py` | Oryn-new |

## Remaining Core Seams

`agent/conversation_loop.py` keeps these named seams:

- Import context-usage telemetry:
  `from agent.context_usage import emit_context_usage`
- Emit context usage from one documented post-usage accounting point. The
  previous four inline calls were reduced because existing hooks do not expose
  the live agent/callback state needed for streaming context usage.
- Import empty-tool recovery:
  `from agent.tool_recovery import empty_tool_result_recovery_response`
- Delegate empty final responses after tool calls through:
  `empty_tool_result_recovery_response(messages)`

`agent/transports/chat_completions.py` keeps this named seam:

- Delegate provider-facing message cleanup through:
  `sanitize_chat_completion_messages_for_wire(messages)`

`gateway/platforms/api_server.py` keeps these named seams:

- Import the Oryn route boundary:
  `from gateway.dev_control.routes import DevControlRouteMixin, dev_control_capabilities, register_dev_control_routes`
- Mix in relocated dev handlers:
  `class APIServerAdapter(DevControlRouteMixin, BasePlatformAdapter)`
- Advertise dev capabilities through:
  `**dev_control_capabilities()`
- Register `/v1/dev/*` and `/v1/oryn/project-dashboard` routes through:
  `register_dev_control_routes(self._app, self)`
- Start and cancel the dev supervisor-loop task through the relocated mixin method.
- Keep project-dashboard fingerprint/cache invalidation helpers in core until the
  shared read-model cache is split into a separate Oryn shell.

`gateway/run.py` remains a small Oryn runtime seam and should be checked during
every upstream pull.

## Upstream Compatibility Checklist

1. `git fetch upstream`
2. `git diff --stat upstream/main...HEAD`
3. Review this ledger and update any changed classifications.
4. Dry-run upstream integration:
   `git merge --no-commit --no-ff upstream/main`
5. If conflicts appear, verify they are outside `gateway/dev_control/routes.py`
   or update this ledger with the residual seam.
6. Abort the dry run: `git merge --abort`
7. Run dev-control route tests:
   `scripts/run_tests.sh tests/gateway/test_api_server_runs.py`
8. Run focused engine seam tests when engine files changed:
   `scripts/run_tests.sh tests/agent/test_tool_recovery.py tests/agent/test_message_sanitization.py tests/agent/transports/test_chat_completions.py`
9. Run lint: `.venv/bin/ruff check .`

## Upstream PR Preparation

Two Oryn-agnostic hardening changes are prepared for Felipe to submit upstream:

- `codex/upstream-empty-tool-recovery`: extracts and proposes empty-tool-result
  recovery. Description:
  `docs/upstream-prs/empty-tool-result-recovery.md`
- `codex/upstream-chat-message-sanitization`: consolidates chat-completions wire
  message sanitization in `agent/message_sanitization.py`. Description:
  `docs/upstream-prs/chat-completions-message-sanitization.md`

## 2026-05-29 Dry-Run Result

Command: `git merge --no-commit --no-ff upstream/main`

Result after route extraction and engine-seam reduction: conflicts remained in
four upstream-existing files:

- `agent/conversation_loop.py`
- `agent/transports/chat_completions.py`
- `gateway/platforms/api_server.py`
- `plugins/kanban/dashboard/plugin_api.py`

The `agent/conversation_loop.py` residual conflict is now at the empty-tool
recovery call site rather than an inline recovery helper body. The context-usage
footprint is one documented telemetry call.

The `agent/transports/chat_completions.py` residual conflict is now at imports;
the transport body delegates sanitization through
`sanitize_chat_completion_messages_for_wire(messages)` instead of carrying inline
normalization logic.

The `gateway/platforms/api_server.py` conflict was outside the extracted
`/v1/dev/*` handler/registration regions. It was in startup wiring around the
dev supervisor-loop task and upstream's stricter `API_SERVER_KEY` requirement.

Cleanup: dry-run was aborted with `git merge --abort`; no upstream merge was
committed.
