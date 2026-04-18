#!/usr/bin/env bash
# Named chunk runner for hermes-agent verification suites.
# Delegates every chunk to scripts/run_tests.sh so CI-parity policy stays in one place.
#
# Usage:
#   scripts/run_test_chunks.sh list
#   scripts/run_test_chunks.sh list broad
#   scripts/run_test_chunks.sh broad
#   scripts/run_test_chunks.sh broad run-agent-a run-agent-b
#   scripts/run_test_chunks.sh run-agent-file -- -q --durations=20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TESTS="$SCRIPT_DIR/run_tests.sh"

if [ ! -x "$RUN_TESTS" ]; then
  echo "error: expected executable runner at $RUN_TESTS" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_test_chunks.sh list [suite]
  scripts/run_test_chunks.sh <suite> [chunk ...] [-- <extra pytest args>]

Suites:
  broad              Broad verification suite for current orchestration hardening targets.
  native-orchestration-broad
  run-agent-file     Chunked runner for tests/run_agent/test_run_agent.py.

Examples:
  scripts/run_test_chunks.sh broad
  scripts/run_test_chunks.sh broad run-agent-a run-agent-b
  scripts/run_test_chunks.sh run-agent-file -- -q --durations=20
EOF
}

suite_alias() {
  case "$1" in
    broad) echo "native-orchestration-broad" ;;
    run-agent) echo "run-agent-file" ;;
    *) echo "$1" ;;
  esac
}

suite_chunks() {
  case "$1" in
    native-orchestration-broad)
      printf '%s\n' \
        config-validation \
        delegate \
        delegate-toolset-scope \
        run-agent-a \
        run-agent-b \
        run-agent-c \
        run-agent-d \
        run-agent-e \
        run-agent-f
      ;;
    run-agent-file)
      printf '%s\n' \
        run-agent-a \
        run-agent-b \
        run-agent-c \
        run-agent-d \
        run-agent-e \
        run-agent-f
      ;;
    *)
      return 1
      ;;
  esac
}

chunk_nodes() {
  case "$1" in
    config-validation)
      printf '%s\n' tests/hermes_cli/test_config_validation.py
      ;;
    delegate)
      printf '%s\n' tests/tools/test_delegate.py
      ;;
    delegate-toolset-scope)
      printf '%s\n' tests/tools/test_delegate_toolset_scope.py
      ;;
    run-agent-a)
      printf '%s\n' \
        tests/run_agent/test_run_agent.py::test_aiagent_reuses_existing_errors_log_handler \
        tests/run_agent/test_run_agent.py::TestProviderModelNormalization \
        tests/run_agent/test_run_agent.py::TestHasContentAfterThinkBlock \
        tests/run_agent/test_run_agent.py::TestStripThinkBlocks \
        tests/run_agent/test_run_agent.py::TestExtractReasoning \
        tests/run_agent/test_run_agent.py::TestCleanSessionContent \
        tests/run_agent/test_run_agent.py::TestGetMessagesUpToLastAssistant \
        tests/run_agent/test_run_agent.py::TestMaskApiKey \
        tests/run_agent/test_run_agent.py::TestInit \
        tests/run_agent/test_run_agent.py::TestInterrupt \
        tests/run_agent/test_run_agent.py::TestHydrateTodoStore \
        tests/run_agent/test_run_agent.py::TestBuildSystemPrompt \
        tests/run_agent/test_run_agent.py::TestToolUseEnforcementConfig \
        tests/run_agent/test_run_agent.py::TestInvalidateSystemPrompt \
        tests/run_agent/test_run_agent.py::TestBuildApiKwargs \
        tests/run_agent/test_run_agent.py::TestBuildAssistantMessage
      ;;
    run-agent-b)
      printf '%s\n' \
        tests/run_agent/test_run_agent.py::TestFormatToolsForSystemMessage \
        tests/run_agent/test_run_agent.py::TestExecuteToolCalls \
        tests/run_agent/test_run_agent.py::TestConcurrentToolExecution \
        tests/run_agent/test_run_agent.py::TestPathsOverlap \
        tests/run_agent/test_run_agent.py::TestParallelScopePathNormalization \
        tests/run_agent/test_run_agent.py::TestHandleMaxIterations \
        tests/run_agent/test_run_agent.py::TestRunConversation
      ;;
    run-agent-c)
      printf '%s\n' \
        tests/run_agent/test_run_agent.py::TestRetryExhaustion \
        tests/run_agent/test_run_agent.py::TestFlushSentinelNotLeaked \
        tests/run_agent/test_run_agent.py::TestConversationHistoryNotMutated \
        tests/run_agent/test_run_agent.py::TestNousCredentialRefresh \
        tests/run_agent/test_run_agent.py::TestCredentialPoolRecovery \
        tests/run_agent/test_run_agent.py::TestMaxTokensParam \
        tests/run_agent/test_run_agent.py::TestSystemPromptStability \
        tests/run_agent/test_run_agent.py::TestBudgetPressure \
        tests/run_agent/test_run_agent.py::TestSafeWriter \
        tests/run_agent/test_run_agent.py::TestSaveSessionLogAtomicWrite
      ;;
    run-agent-d)
      printf '%s\n' \
        tests/run_agent/test_run_agent.py::TestBuildApiKwargsAnthropicMaxTokens \
        tests/run_agent/test_run_agent.py::TestAnthropicImageFallback \
        tests/run_agent/test_run_agent.py::TestFallbackAnthropicProvider \
        tests/run_agent/test_run_agent.py::test_aiagent_uses_copilot_acp_client \
        tests/run_agent/test_run_agent.py::test_quiet_spinner_allowed_with_explicit_print_fn \
        tests/run_agent/test_run_agent.py::test_quiet_spinner_allowed_on_real_tty \
        tests/run_agent/test_run_agent.py::test_quiet_spinner_suppressed_on_non_tty_without_print_fn \
        tests/run_agent/test_run_agent.py::test_is_openai_client_closed_honors_custom_client_flag \
        tests/run_agent/test_run_agent.py::test_is_openai_client_closed_handles_method_form \
        tests/run_agent/test_run_agent.py::test_is_openai_client_closed_falls_back_to_http_client \
        tests/run_agent/test_run_agent.py::TestAnthropicBaseUrlPassthrough \
        tests/run_agent/test_run_agent.py::TestAnthropicCredentialRefresh
      ;;
    run-agent-e)
      printf '%s\n' tests/run_agent/test_run_agent.py::TestStreamingApiCall
      ;;
    run-agent-f)
      printf '%s\n' \
        tests/run_agent/test_run_agent.py::TestInterruptVprintForceTrue \
        tests/run_agent/test_run_agent.py::TestAnthropicInterruptHandler \
        tests/run_agent/test_run_agent.py::TestStreamCallbackNonStreamingProvider \
        tests/run_agent/test_run_agent.py::TestPersistUserMessageOverride \
        tests/run_agent/test_run_agent.py::TestVprintForceOnErrors \
        tests/run_agent/test_run_agent.py::TestNormalizeCodexDictArguments \
        tests/run_agent/test_run_agent.py::TestOAuthFlagAfterCredentialRefresh \
        tests/run_agent/test_run_agent.py::TestFallbackSetsOAuthFlag \
        tests/run_agent/test_run_agent.py::TestMemoryNudgeCounterPersistence \
        tests/run_agent/test_run_agent.py::TestDeadRetryCode \
        tests/run_agent/test_run_agent.py::TestMemoryContextSanitization \
        tests/run_agent/test_run_agent.py::TestMemoryProviderTurnStart \
        tests/run_agent/test_run_agent.py::TestNativeOrchestrationHardening
      ;;
    *)
      return 1
      ;;
  esac
}

if [ "$#" -eq 0 ]; then
  usage
  exit 1
fi

if [ "$1" = "list" ]; then
  if [ "$#" -eq 1 ]; then
    echo "native-orchestration-broad"
    echo "run-agent-file"
    exit 0
  fi
  SUITE="$(suite_alias "$2")"
  suite_chunks "$SUITE"
  exit 0
fi

SUITE="$(suite_alias "$1")"
shift

DEFAULT_CHUNKS=()
while IFS= read -r chunk_name; do
  DEFAULT_CHUNKS+=("$chunk_name")
done < <(suite_chunks "$SUITE")
if [ "${#DEFAULT_CHUNKS[@]}" -eq 0 ]; then
  usage
  exit 1
fi

SELECTED_CHUNKS=()
PASSTHROUGH_ARGS=()
PARSING_CHUNKS=1
for arg in "$@"; do
  if [ "$arg" = "--" ]; then
    PARSING_CHUNKS=0
    continue
  fi
  if [ "$PARSING_CHUNKS" -eq 1 ]; then
    SELECTED_CHUNKS+=("$arg")
  else
    PASSTHROUGH_ARGS+=("$arg")
  fi
done

if [ "${#SELECTED_CHUNKS[@]}" -eq 0 ]; then
  SELECTED_CHUNKS=("${DEFAULT_CHUNKS[@]}")
fi

for chunk in "${SELECTED_CHUNKS[@]}"; do
  if ! chunk_nodes "$chunk" >/dev/null; then
    echo "error: unknown chunk '$chunk' for suite '$SUITE'" >&2
    echo "available chunks:" >&2
    suite_chunks "$SUITE" >&2
    exit 1
  fi
done

echo "▶ suite: $SUITE"
echo "▶ chunks: ${SELECTED_CHUNKS[*]}"
if [ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]; then
  echo "▶ pass-through pytest args: ${PASSTHROUGH_ARGS[*]}"
fi

for chunk in "${SELECTED_CHUNKS[@]}"; do
  nodes=()
  while IFS= read -r node; do
    nodes+=("$node")
  done < <(chunk_nodes "$chunk")
  echo
  echo "== chunk: $chunk =="
  if [ "${#PASSTHROUGH_ARGS[@]}" -gt 0 ]; then
    "$RUN_TESTS" "${PASSTHROUGH_ARGS[@]}" "${nodes[@]}"
  else
    "$RUN_TESTS" "${nodes[@]}"
  fi
done
