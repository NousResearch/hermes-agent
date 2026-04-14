#!/usr/bin/env bash
set -euo pipefail

# Update dispatch intent with probe execution detection
# This script enhances the dispatch intent JSON with probe execution detection
# for quota-recovered workers that have been stuck in due_now state.

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
PYTHON="${PYTHON:-python3}"

DISPATCH_INTENT_PATH="$ROOT/state/continuity/latest/execution_supervisor_dispatch_intent_latest.json"
PROBE_PLAN_PATH="$ROOT/state/continuity/latest/execution_supervisor_probe_execution_plan_latest.json"
CANARY_SCHEDULE_PATH="$ROOT/state/continuity/latest/execution_supervisor_canary_probe_schedule_latest.json"
ENHANCED_PATH="$ROOT/state/continuity/latest/execution_supervisor_dispatch_intent_enhanced_latest.json"
ENHANCEMENT_SCRIPT="$ROOT/ops/openclaw/continuity/enhance_dispatch_intent_with_probe_execution.py"

# Check if all required files exist
if [[ ! -f "$DISPATCH_INTENT_PATH" ]]; then
    echo "Error: Dispatch intent not found at $DISPATCH_INTENT_PATH" >&2
    exit 1
fi

if [[ ! -f "$PROBE_PLAN_PATH" ]]; then
    echo "Error: Probe execution plan not found at $PROBE_PLAN_PATH" >&2
    exit 1
fi

if [[ ! -f "$CANARY_SCHEDULE_PATH" ]]; then
    echo "Error: Canary probe schedule not found at $CANARY_SCHEDULE_PATH" >&2
    exit 1
fi

# Run enhancement
echo "Enhancing dispatch intent with probe execution detection..."
"$PYTHON" "$ENHANCEMENT_SCRIPT" \
    "$DISPATCH_INTENT_PATH" \
    "$PROBE_PLAN_PATH" \
    "$CANARY_SCHEDULE_PATH" \
    "$ENHANCED_PATH"

# Check if enhancement was successful
if [[ $? -eq 0 ]] && [[ -f "$ENHANCED_PATH" ]]; then
    echo "Enhanced dispatch intent saved to: $ENHANCED_PATH"
    
    # Extract detection result
    DETECTION=$(jq -r '.probe_execution_detection // empty' "$ENHANCED_PATH" 2>/dev/null || echo '{}')
    SHOULD_TRIGGER=$(echo "$DETECTION" | jq -r '.should_trigger // false')
    TARGET_WORKER=$(echo "$DETECTION" | jq -r '.target_worker // ""')
    REASON=$(echo "$DETECTION" | jq -r '.reason // ""')
    
    if [[ "$SHOULD_TRIGGER" == "true" ]] && [[ -n "$TARGET_WORKER" ]]; then
        echo "PROBE EXECUTION DETECTED:"
        echo "  Worker: $TARGET_WORKER"
        echo "  Reason: $REASON"
        
        # Extract contract details
        CONTRACT=$(echo "$DETECTION" | jq -r '.probe_execution_contract // empty')
        EXPECTED_ARTIFACT=$(echo "$CONTRACT" | jq -r '.expected_artifact // ""')
        TIMEOUT_SEC=$(echo "$CONTRACT" | jq -r '.timeout_sec // 0')
        
        if [[ -n "$EXPECTED_ARTIFACT" ]]; then
            echo "  Expected artifact: $EXPECTED_ARTIFACT"
            echo "  Timeout: ${TIMEOUT_SEC}s"
        fi
        
        # Show enhancement metadata
        ENHANCEMENT=$(jq -r '._probe_execution_enhancement // empty' "$ENHANCED_PATH" 2>/dev/null || echo '{}')
        ORIGINAL_DECISION=$(echo "$ENHANCEMENT" | jq -r '.original_decision // ""')
        RECOMMENDED_DECISION=$(echo "$ENHANCEMENT" | jq -r '.recommended_decision // ""')
        STALE_TICKS=$(echo "$ENHANCEMENT" | jq -r '.stale_ticks // 0')
        
        echo "  Original decision: $ORIGINAL_DECISION"
        echo "  Recommended decision: $RECOMMENDED_DECISION"
        echo "  Stale ticks: $STALE_TICKS"
        
        # Create a simple action recommendation
        echo ""
        echo "RECOMMENDED ACTION:"
        echo "  Execute canary probe for $TARGET_WORKER using:"
        echo "  $PYTHON $ROOT/ops/openclaw/continuity/execution_supervisor_probe_executor.py \\"
        echo "    --probe-plan \"$PROBE_PLAN_PATH\" \\"
        echo "    --canary-schedule \"$CANARY_SCHEDULE_PATH\" \\"
        echo "    --output \"$ROOT/state/continuity/latest/probe_execution_result_latest.json\" \\"
        echo "    --execute"
    else
        echo "No probe execution needed: $REASON"
    fi
else
    echo "Error: Failed to enhance dispatch intent" >&2
    exit 1
fi