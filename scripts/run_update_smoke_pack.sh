#!/usr/bin/env bash
# Targeted smoke packs for guarded Hermes upstream/local updates.
# This script intentionally does not restart gateway/dashboard services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

run_py_pack() {
  local name="$1"
  shift
  local files=()
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      files+=("$path")
    else
      printf '  - skip missing %s\n' "$path" >&2
    fi
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    printf 'error: pack %s resolved to no existing test files\n' "$name" >&2
    return 2
  fi
  printf '\n▶ smoke pack: %s\n' "$name"
  scripts/run_tests.sh "${files[@]}" -- -q -o addopts=
}

run_tui_npm_checks() {
  if [[ "${RUN_TUI_NPM:-0}" != "1" ]]; then
    printf '\nℹ skipping ui-tui npm checks (set RUN_TUI_NPM=1 to enable)\n'
    return 0
  fi
  if [[ ! -d ui-tui/node_modules ]]; then
    printf '\nℹ skipping ui-tui npm checks: ui-tui/node_modules is absent\n'
    return 0
  fi
  printf '\n▶ smoke pack: ui-tui npm type/test\n'
  npm --prefix ui-tui run type-check
  npm --prefix ui-tui test -- --run
}

dashboard_ws() {
  run_py_pack dashboard-ws \
    tests/plugins/test_kanban_dashboard_plugin.py \
    tests/test_tui_gateway_server.py \
    tests/hermes_cli/test_pty_bridge.py \
    tests/hermes_cli/test_web_server.py
}

webhook_security() {
  run_py_pack webhook-security \
    tests/gateway/test_webhook_integration.py \
    tests/gateway/test_webhook_dynamic_routes.py \
    tests/gateway/test_api_server_runs.py \
    tests/gateway/test_api_server.py \
    tests/tools/test_url_safety.py
}

streaming() {
  run_py_pack streaming \
    tests/gateway/test_stream_consumer.py \
    tests/gateway/test_stream_consumer_draft.py \
    tests/gateway/test_text_batching.py \
    tests/gateway/test_telegram_format.py \
    tests/gateway/test_discord_send.py \
    tests/agent/test_streaming_context_scrubber.py \
    tests/run_agent/test_deepseek_reasoning_content_echo.py
}

cross_profile() {
  run_py_pack cross-profile \
    tests/test_hermes_constants.py \
    tests/test_subprocess_home_isolation.py \
    tests/hermes_cli/test_auth_profile_fallback.py \
    tests/cron/test_cron_profile.py \
    tests/tools/test_write_deny.py \
    tests/tools/test_file_operations.py \
    tests/tools/test_send_message_tool.py \
    tests/hermes_cli/test_kanban_db.py
}

kanban_promote() {
  run_py_pack kanban-promote \
    tests/hermes_cli/test_kanban_db.py \
    tests/hermes_cli/test_kanban_blocked_sticky.py \
    tests/hermes_cli/test_kanban_decompose_db.py \
    tests/hermes_cli/test_kanban_cli.py \
    tests/tools/test_kanban_tools.py \
    tests/plugins/test_kanban_worker_runs.py \
    tests/gateway/test_kanban_checkpoint_notifications.py
}

kanban_promote_stress() {
  run_py_pack kanban-promote-stress \
    tests/stress/test_property_fuzzing.py \
    tests/stress/test_atypical_scenarios.py
}

codex_tui() {
  run_py_pack codex-tui \
    tests/agent/transports/test_codex_app_server_runtime.py \
    tests/agent/transports/test_codex_app_server_session.py \
    tests/agent/transports/test_codex_event_projector.py \
    tests/run_agent/test_codex_app_server_integration.py \
    tests/hermes_cli/test_codex_runtime_switch.py \
    tests/cron/test_codex_execution_paths.py \
    tests/test_tui_gateway_server.py \
    tests/hermes_cli/test_tui_resume_flow.py \
    tests/hermes_cli/test_tui_npm_install.py
  run_tui_npm_checks
}

protected() {
  run_py_pack protected \
    tests/hermes_cli/test_kanban_db.py \
    tests/hermes_cli/test_kanban_blocked_sticky.py \
    tests/tools/test_kanban_tools.py \
    tests/hermes_cli/test_doctor.py \
    tests/hermes_cli/test_cron.py \
    tests/cron/test_cron_profile.py \
    tests/hermes_cli/test_codex_runtime_switch.py \
    tests/agent/transports/test_codex_app_server_runtime.py \
    tests/agent/transports/test_codex_event_projector.py \
    tests/plugins/test_kanban_dashboard_plugin.py \
    tests/test_tui_gateway_server.py \
    tests/gateway/test_webhook_integration.py \
    tests/gateway/test_webhook_dynamic_routes.py \
    tests/gateway/test_stream_consumer.py \
    tests/gateway/test_stream_consumer_draft.py \
    tests/agent/test_streaming_context_scrubber.py \
    tests/test_hermes_constants.py \
    tests/test_subprocess_home_isolation.py \
    tests/tools/test_write_deny.py
}

usage() {
  cat <<'USAGE'
Usage: scripts/run_update_smoke_pack.sh PACK [PACK...]

Packs:
  dashboard-ws       Dashboard/Kanban dashboard/TUI WebSocket auth
  webhook-security   Webhook routes, raw-body signatures, API routes
  streaming          Gateway streaming transforms and formatting
  cross-profile      HERMES_HOME/profile/path/secret/Kanban DB guards
  kanban-promote     Kanban dependency promotion and review gates
  kanban-stress      Optional Kanban promotion/property stress files
  codex-tui          Codex app-server runtime plus TUI/PTY behavior
  protected          Broad protected update pack
  all                dashboard-ws webhook-security streaming cross-profile kanban-promote codex-tui

Set RUN_TUI_NPM=1 to include optional ui-tui npm type/test checks.
USAGE
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for pack in "$@"; do
  case "$pack" in
    dashboard-ws) dashboard_ws ;;
    webhook-security) webhook_security ;;
    streaming) streaming ;;
    cross-profile) cross_profile ;;
    kanban-promote) kanban_promote ;;
    kanban-stress) kanban_promote_stress ;;
    codex-tui) codex_tui ;;
    protected) protected ;;
    all)
      dashboard_ws
      webhook_security
      streaming
      cross_profile
      kanban_promote
      codex_tui
      ;;
    *)
      printf 'error: unknown smoke pack: %s\n\n' "$pack" >&2
      usage >&2
      exit 2
      ;;
  esac
done
