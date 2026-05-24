# Hermes update smoke packs

Purpose: pick a targeted, repeatable validation pack after upstream merges or Atlas-local patches without restarting the live gateway/dashboard. These packs are designed for guarded worktrees first; run them in the live checkout only for read-only verification.

## Non-negotiables

- Do **not** run `hermes gateway restart`, `hermes dashboard --stop`, systemd restarts, or any command that mutates live services unless Gabriel explicitly approves it.
- Run through `scripts/run_tests.sh` so each pytest file runs in an isolated subprocess with a clean environment.
- Prefer narrow packs first, then the protected pack, then the full suite if the changed surface is broad.
- If a pack fails, reproduce the failing test file directly with `python -m pytest <file> -q -o addopts=` before widening scope.
- Treat Blue/GHL, task-capsule deletion/removal, approvals, and customer-facing sends as review-gated even when tests pass.

## Quick command

```bash
# From the repo root. Runs all current high-risk update packs except optional npm TUI checks.
scripts/run_update_smoke_pack.sh all

# Narrow packs:
scripts/run_update_smoke_pack.sh dashboard-ws webhook-security streaming cross-profile kanban-promote codex-tui

# Include UI/TUI TypeScript tests when node_modules is present:
RUN_TUI_NPM=1 scripts/run_update_smoke_pack.sh codex-tui
```

## Pack map

### `dashboard-ws` — dashboard WebSocket auth

Run when dashboard auth/session routing, dashboard plugins, `/api/pty`, Kanban dashboard websocket routes, or browser dashboard token handling changes.

Files:

- `tests/plugins/test_kanban_dashboard_plugin.py`
- `tests/test_tui_gateway_server.py`
- `tests/hermes_cli/test_pty_bridge.py`
- `tests/hermes_cli/test_web_server.py`

Watchpoints:

- Query-param/session-token auth must fail closed.
- WebSocket close codes should surface useful client errors, not reconnect loops.
- Dashboard plugin API changes may need a dashboard restart later, but do not do it during this pack without approval.

### `webhook-security` — webhook route and signature handling

Run when webhook adapters, dynamic routes, HMAC/signature validation, API server routes, or webhook docs change.

Files:

- `tests/gateway/test_webhook_integration.py`
- `tests/gateway/test_webhook_dynamic_routes.py`
- `tests/gateway/test_api_server_runs.py`
- `tests/gateway/test_api_server.py` if present
- `tests/tools/test_url_safety.py`

Watchpoints:

- Raw request body must be what is signed; do not reserialize before verification.
- Missing, malformed, or mismatched signatures must fail closed.
- Provider-specific asymmetric webhooks should terminate in a verifier bridge before Hermes generic HMAC routes.

### `streaming` — gateway streaming transforms

Run when token streaming, draft edits, platform formatting, reasoning/thinking filters, or message batching changes.

Files:

- `tests/gateway/test_stream_consumer.py`
- `tests/gateway/test_stream_consumer_draft.py`
- `tests/gateway/test_text_batching.py`
- `tests/gateway/test_telegram_format.py`
- `tests/gateway/test_discord_send.py`
- `tests/agent/test_streaming_context_scrubber.py`
- `tests/run_agent/test_deepseek_reasoning_content_echo.py`

Watchpoints:

- Streaming transforms must not leak reasoning/thinking blocks to user-visible platform edits.
- Final response delivery must still happen after draft/edit failures.
- Platform-specific length/format guards should degrade by chunking or final resend, not silent truncation.

### `cross-profile` — profile/home/path/secret guards

Run when profiles, `HERMES_HOME`, worker env, auth fallback, cron profile execution, file write deny rules, or cross-profile notification paths change.

Files:

- `tests/test_hermes_constants.py`
- `tests/test_subprocess_home_isolation.py`
- `tests/hermes_cli/test_auth_profile_fallback.py`
- `tests/cron/test_cron_profile.py`
- `tests/tools/test_write_deny.py`
- `tests/tools/test_file_operations.py`
- `tests/tools/test_send_message_tool.py`
- `tests/hermes_cli/test_kanban_db.py`

Watchpoints:

- Default root and active profile home must not cross-contaminate.
- Global secrets such as root `.env` remain write-denied even from profile sessions.
- Kanban workers must read/write the dispatcher-pinned board DB, not a profile-local accidental DB.

### `kanban-promote` — Kanban parent/child promotion and review gates

Run when Kanban DB, dispatcher, dependency edges, task completion, blocked/review semantics, dashboard board moves, or promote/recompute code changes.

Files:

- `tests/hermes_cli/test_kanban_db.py`
- `tests/hermes_cli/test_kanban_blocked_sticky.py`
- `tests/hermes_cli/test_kanban_decompose_db.py`
- `tests/hermes_cli/test_kanban_cli.py`
- `tests/tools/test_kanban_tools.py`
- `tests/plugins/test_kanban_worker_runs.py`
- `tests/gateway/test_kanban_checkpoint_notifications.py`

Optional stress pass after DB/dispatcher rewrites:

- `tests/stress/test_property_fuzzing.py`
- `tests/stress/test_atypical_scenarios.py`

Watchpoints:

- A child promotes to `ready` only when **all** parents are `done`.
- Blocked/review-required tasks should not be auto-promoted or completed by a worker without review.
- Dashboard drag/drop and CLI status changes must preserve dependency invariants.

### `codex-tui` — Codex app-server and TUI behavior

Run when Codex app-server runtime, Codex OAuth/model routing, TUI/PTY bridge, slash command runtime switch, or UI streaming markdown changes.

Files:

- `tests/agent/transports/test_codex_app_server_runtime.py`
- `tests/agent/transports/test_codex_app_server_session.py`
- `tests/agent/transports/test_codex_event_projector.py`
- `tests/run_agent/test_codex_app_server_integration.py`
- `tests/hermes_cli/test_codex_runtime_switch.py`
- `tests/cron/test_codex_execution_paths.py`
- `tests/test_tui_gateway_server.py`
- `tests/hermes_cli/test_tui_resume_flow.py`
- `tests/hermes_cli/test_tui_npm_install.py`

Optional npm checks from `ui-tui/` when dependencies are installed:

```bash
npm run type-check
npm test -- --run
```

Watchpoints:

- App-server failures should produce actionable fallback guidance without corrupting the Hermes message transcript.
- Kanban worker env and sandbox overrides must propagate into Codex MCP/tool subprocesses.
- TUI-owned slash commands should render via overlays where expected and stay compatible with gateway command help.

## Protected update pack

Run this after any upstream merge that touches gateway, cron, Kanban, profile, Codex, tool safety, or dashboard paths:

```bash
scripts/run_update_smoke_pack.sh protected
```

The protected pack is intentionally broader than any single narrow pack and combines: Kanban DB/tools, blocked-review semantics, doctor/cron profile behavior, Codex runtime switching/app-server projection, dashboard WebSocket auth, webhook security, streaming transforms, cross-profile guards, and TUI bridge behavior.

## Live health checks after approved cutover only

After Gabriel approves live cutover/restarts, then run live checks such as:

```bash
venv/bin/hermes doctor
venv/bin/hermes status --all
venv/bin/hermes cron status
venv/bin/hermes kanban stats
ss -ltnp | grep -E '127\.0\.0\.1:18791|:9119'
```

These verify the live processes; they are not a substitute for the guarded smoke packs above.
