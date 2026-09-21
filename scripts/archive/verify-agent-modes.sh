#!/usr/bin/env bash
# ── KENSEI CUSTOM MODE VERIFICATION ──────────────────────────────────
# Run after any pipx install --force ., git pull upstream, or git merge
# to verify the custom mode system survived.  Exit code 0 = all good.
#
# If this fails, follow /home/kensei/.hermes/skills/devops/agent-modes/SKILL.md
# and /home/kensei/.hermes/skills/governance/SOUL.md "KENSEI CUSTOM MODE - RESTORE"
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail
FAIL=0
REPO="/home/kensei/repos/KenseiAgent"
HUMAN_GATE=1  # Set to 0 after human approves a mode change

# --- TUI TypeScript checks ---
check_ts() {
    local file="$1" pattern="$2" label="$3"
    if grep -q "$pattern" "$REPO/$file" 2>/dev/null; then
        echo "  ✓ $label"
    else
        echo "  ✗ MISSING: $label ($file should contain '$pattern')"
        FAIL=1
    fi
}

# --- File presence checks ---
check_file() {
    local file="$1" label="$2"
    if [[ -f "$file" ]]; then
        echo "  ✓ $label"
    else
        echo "  ✗ MISSING: $label ($file not found)"
        FAIL=1
    fi
}

# --- Direct grep check (bypasses check_ts $REPO prepend) ---
check_grep() {
    local file="$1" pattern="$2" label="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  ✓ $label"
    else
        echo "  ✗ MISSING: $label ($file should contain '$pattern')"
        FAIL=1
    fi
}

echo "── KENSEI agent-mode verification ──"

# 1) TypeScript: interfaces.ts
check_ts "ui-tui/src/app/interfaces.ts" "AGENT_MODES" "AgentMode type + AGENT_MODES array"
check_ts "ui-tui/src/app/interfaces.ts" "agentMode: AgentMode" "UiState.agentMode field"
check_ts "ui-tui/src/app/interfaces.ts" "AskUserQuestionsReq" "AskUserQuestionsReq import (new mode system)"

# 2) TypeScript: uiStore.ts
check_ts "ui-tui/src/app/uiStore.ts" "agentMode: 'auto'" "buildUiState default agentMode='auto'"

# 3) TypeScript: useInputHandlers.ts
# Shift+Tab is YOLO-only (not mode cycling).  Dead AgentMode/AGENT_MODES
# imports have been removed.  Mode is set via /mode slash command or
# config.set key=mode, not Shift+Tab.
check_ts "ui-tui/src/app/useInputHandlers.ts" "key: 'yolo'" "Shift+Tab toggles YOLO (not mode cycling)"

# 4) TypeScript: appLayout.tsx
check_ts "ui-tui/src/components/appLayout.tsx" "MODE_BADGE" "Mode badge emoji map"
check_ts "ui-tui/src/components/appLayout.tsx" "displayPrompt" "displayPrompt used in PromptPrefix"
check_ts "ui-tui/src/components/appLayout.tsx" "UltraPlan" "UltraPlan label in MODE_BADGE"

# 5) TypeScript: session.ts (/mode slash command)
check_ts "ui-tui/src/app/slash/commands/session.ts" "KENSEI CUSTOM: agent mode slash command" "/mode slash command"
check_ts "ui-tui/src/app/slash/commands/session.ts" "UltraPlan" "UltraPlan label in /mode help text"

# 6) TypeScript: askUserQuestionsTool.tsx (new component, item 4A)
check_file "$REPO/ui-tui/src/components/askUserQuestionsTool.tsx" "AskUserQuestionsTool.tsx (new batched prompt component)"
check_ts "ui-tui/src/components/askUserQuestionsTool.tsx" "Recommended" "Recommended label rendering"
check_ts "ui-tui/src/components/askUserQuestionsTool.tsx" "borderStyle" "Boxed design (borderStyle)"
# Q5 fix: parent-owned selections state
check_ts "ui-tui/src/components/askUserQuestionsTool.tsx" "setSelections" "Parent owns selections state (Q5 fix)"
# Q6 fix: __other__ sentinel for "Other" handoff
check_ts "ui-tui/src/components/askUserQuestionsTool.tsx" "selectionLabel" "selectionLabel helper for Other handling (Q6 fix)"

# 7) TypeScript: appOverlays.tsx wiring
check_ts "ui-tui/src/components/appOverlays.tsx" "AskUserQuestionsTool" "AskUserQuestionsTool imported in appOverlays"
check_ts "ui-tui/src/components/appOverlays.tsx" "overlay.askUserQuestions" "askUserQuestions overlay render check"

# 8) TypeScript: useMainApp.ts handler
check_ts "ui-tui/src/app/useMainApp.ts" "answerAskUserQuestions" "answerAskUserQuestions callback in useMainApp"
check_ts "ui-tui/src/app/useMainApp.ts" "ask_user_questions.respond" "Gateway RPC forward to ask_user_questions.respond"

# 9) TypeScript: overlayStore.ts
check_ts "ui-tui/src/app/overlayStore.ts" "askUserQuestions: null" "askUserQuestions field in initial state"

# 10) TypeScript: types.ts
check_ts "ui-tui/src/types.ts" "AskUserQuestionsReq" "AskUserQuestionsReq type defined"

# 11) TypeScript: createGatewayEventHandler.ts (B3 fix)
check_ts "ui-tui/src/app/createGatewayEventHandler.ts" "ask_user_questions.request" "TUI handles ask_user_questions.request event (B3 fix)"
check_ts "ui-tui/src/app/createGatewayEventHandler.ts" "patchOverlayState" "TUI populates overlay state on ask_user_questions.request"

# 12) Python: hermes_cli/mode_prompts.py (Q1 fix - single source of truth)
check_file "$REPO/hermes_cli/mode_prompts.py" "hermes_cli/mode_prompts.py (Q1: shared prompt source of truth)"
check_grep "$REPO/hermes_cli/mode_prompts.py" "PLAN_PROMPT" "PLAN_PROMPT constant"
check_grep "$REPO/hermes_cli/mode_prompts.py" "ULTRAPLAN_PROMPT" "ULTRAPLAN_PROMPT constant"
check_grep "$REPO/hermes_cli/mode_prompts.py" "RECON_PROMPT" "RECON_PROMPT constant"
check_grep "$REPO/hermes_cli/mode_prompts.py" "detect_mode" "detect_mode() function (Q2 fix)"
check_grep "$REPO/hermes_cli/mode_prompts.py" "mode_label" "mode_label() function"
check_grep "$REPO/hermes_cli/mode_prompts.py" "UltraPlan" "UltraPlan label"
check_grep "$REPO/hermes_cli/mode_prompts.py" "do NOT add it" "Recommended-instruction fix (Q4)"

# 13) Python: tui_gateway/server.py (Q1 dedup + B1-B3 wiring)
check_grep "$REPO/tui_gateway/server.py" "from hermes_cli.mode_prompts import" "tui_gateway imports shared mode_prompts (Q1 fix)"
check_grep "$REPO/tui_gateway/server.py" "ask_user_questions_callback" "_agent_cbs includes ask_user_questions_callback (B2 fix)"
check_grep "$REPO/tui_gateway/server.py" "ask_user_questions.request" "Gateway emits ask_user_questions.request event (B3 fix)"
check_grep "$REPO/tui_gateway/server.py" "@method(\"ask_user_questions.respond\")" "Gateway registers ask_user_questions.respond RPC handler (B1 fix)"
check_grep "$REPO/hermes_cli/mode_prompts.py" "delegate_task" "Recon prompt references delegate_task for profile dispatch (in shared module)"
check_grep "$REPO/tui_gateway/server.py" "UltraPlan" "UltraPlan mode prompt present"
check_grep "$REPO/tui_gateway/server.py" "if key == \"mode\"" "gateway config.set key=mode handler"

# 14) Python: cli.py (Q1 dedup + Q2 detect_mode + CLI callback)
check_grep "$REPO/cli.py" "from hermes_cli.mode_prompts import" "cli.py imports shared mode_prompts (Q1 fix)"
check_grep "$REPO/cli.py" "detect_mode" "cli.py uses detect_mode (Q2 fix)"
check_grep "$REPO/cli.py" "mode_label" "cli.py uses mode_label"
check_grep "$REPO/cli.py" "_ask_user_questions_callback" "cli.py defines _ask_user_questions_callback"
check_grep "$REPO/agent/agent_init.py" "agent.ask_user_questions_callback = ask_user_questions_callback" "agent_init.py wires ask_user_questions_callback to agent (B2 fix)"
check_grep "$REPO/cli.py" "UltraPlan" "UltraPlan label in CLI _handle_mode_command"
check_grep "$REPO/hermes_cli/mode_prompts.py" "delegate_task" "Recon prompt references delegate_task in shared module"

# 15) Python: agent wiring (B2 + B4)
check_grep "$REPO/agent/agent_init.py" "ask_user_questions_callback" "agent_init declares ask_user_questions_callback param (B2 fix)"
check_grep "$REPO/agent/agent_init.py" "agent.ask_user_questions_callback =" "agent_init sets agent.ask_user_questions_callback attr (B2 fix)"
check_grep "$REPO/agent/agent_runtime_helpers.py" "function_name == \"ask_user_questions\"" "agent_runtime_helpers dispatches ask_user_questions (B4 fix)"
check_grep "$REPO/agent/agent_runtime_helpers.py" "ask_user_questions_callback" "agent_runtime_helpers passes ask_user_questions_callback"
check_grep "$REPO/agent/tool_executor.py" "function_name == \"ask_user_questions\"" "tool_executor dispatches ask_user_questions (B4 fix)"
check_grep "$REPO/agent/tool_executor.py" "ask_user_questions_callback" "tool_executor passes ask_user_questions_callback"

# 16) hermes_cli/commands.py
check_ts "hermes_cli/commands.py" "gods_plan" "CommandDef for /mode with gods_plan subcommand"
check_ts "hermes_cli/commands.py" "UltraPlan" "UltraPlan label in CommandDef description"

# 17) Python: ask_user_questions tool (Q6 fix + invariant)
check_file "/home/kensei/repos/KenseiAgent/tools/ask_user_questions_tool.py" "ask_user_questions_tool.py (new batched clarification tool)"
for pat in "MAX_QUESTIONS_PER_CALL" "registry.register" "OTHER_SENTINEL" "needs_followup" "maxItems" "ask_user_questions"; do
    if grep -q "$pat" /home/kensei/repos/KenseiAgent/tools/ask_user_questions_tool.py 2>/dev/null; then
        echo "  ✓ ask_user_questions_tool.py contains '$pat'"
    else
        echo "  ✗ MISSING: ask_user_questions_tool.py should contain '$pat'"
        FAIL=1
    fi
done

# 18) Recon output directory
RECON_DIR="$HOME/.hermes/recon"
if [[ -d "$RECON_DIR" ]]; then
    echo "  ✓ Recon output directory exists: $RECON_DIR"
else
    echo "  ⚠ Recon output directory does not exist (will be created on first recon): $RECON_DIR"
    # Not a failure - just a warning.  The mode prompt tells the agent to create it.
fi

echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "  All mode components verified ✓"
else
    echo "  $FAIL component(s) MISSING - mode system is broken. Restore from skill doc."
fi
exit "$FAIL"
