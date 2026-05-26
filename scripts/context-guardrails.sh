#!/usr/bin/env bash
# context-guardrails.sh - Context pressure monitoring and mitigation recommendations
#
# Usage:
#   ./context-guardrails.sh monitor <session_json>
#   ./context-guardrails.sh check_growth <history_json>
#   ./context-guardrails.sh detect_retry_storm <retry_history_json>
#   ./context-guardrails.sh recommend <pressure_json> <workload_classification>
#
# Exit codes:
#   0 = Success
#   1 = Invalid arguments
#   2 = Check failed

set -euo pipefail

VERSION="1.0.0"

# Source the budget estimator for shared functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED='' YELLOW='' GREEN='' BLUE='' NC=''
fi

#######################################
# Monitor session context state
# Arguments:
#   $1 - Session data JSON
# Outputs:
#   JSON with monitoring results
#######################################
monitor_session() {
    local session_json="$1"
    
    # Extract session metrics
    local current_tokens
    current_tokens=$(echo "$session_json" | jq -r '.current_tokens // 20000')
    
    local provider_limit
    provider_limit=$(echo "$session_json" | jq -r '.provider_limit // 128000')
    
    local growth_rate
    growth_rate=$(echo "$session_json" | jq -r '.growth_rate_pct // 0')
    
    local retry_count
    retry_count=$(echo "$session_json" | jq -r '.retry_count // 0')
    
    local compression_attempts
    compression_attempts=$(echo "$session_json" | jq -r '.compression_attempts // 0')
    
    local turn_number
    turn_number=$(echo "$session_json" | jq -r '.turn // 1')
    
    # Calculate pressure using budget estimator
    local pressure_json
    pressure_json=$("$SCRIPT_DIR/context-budget-estimator.sh" calculate_pressure \
        "$current_tokens" "$provider_limit" "$growth_rate" "$retry_count" "$compression_attempts")
    
    local state
    state=$(echo "$pressure_json" | jq -r '.state')
    
    local pressure_score
    pressure_score=$(echo "$pressure_json" | jq -r '.pressure_score')
    
    # Check for specific issues
    local issues='[]'
    
    # Retry amplification
    if [ "$retry_count" -ge 3 ]; then
        issues=$(echo "$issues" | jq '. + ["retry_amplification"]')
    fi
    
    # Rapid growth
    if (( $(awk "BEGIN {print ($growth_rate >= 25)}") )); then
        issues=$(echo "$issues" | jq '. + ["rapid_growth"]')
    fi
    
    # Ineffective compression
    if [ "$compression_attempts" -ge 2 ]; then
        issues=$(echo "$issues" | jq '. + ["compression_ineffective"]')
    fi
    
    # High turn count with high pressure
    if [ "$turn_number" -ge 15 ] && [ "$state" = "OVERSIZED" ]; then
        issues=$(echo "$issues" | jq '. + ["long_session_high_pressure"]')
    fi
    
    # Combine results
    cat <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "turn": $turn_number,
  "pressure": $(echo "$pressure_json" | jq -c .),
  "issues": $issues,
  "requires_action": $([ "$state" = "OVERSIZED" ] || [ "$state" = "CRITICAL" ] && echo true || echo false)
}
EOF
}

#######################################
# Check context growth pattern
# Arguments:
#   $1 - History JSON (array of {turn, tokens})
# Outputs:
#   JSON with growth analysis
#######################################
check_growth() {
    local history_json="$1"
    
    local turn_count
    turn_count=$(echo "$history_json" | jq 'length')
    
    if [ "$turn_count" -lt 2 ]; then
        echo '{"pattern": "insufficient_data", "healthy": true}'
        return 0
    fi
    
    # Get first and last points
    local first_tokens
    first_tokens=$(echo "$history_json" | jq '.[0].tokens')
    
    local last_tokens
    last_tokens=$(echo "$history_json" | jq '.[-1].tokens')
    
    # Calculate average growth per turn
    local total_growth=$((last_tokens - first_tokens))
    local avg_growth_per_turn=$(awk "BEGIN {printf \"%.2f\", $total_growth / ($turn_count - 1)}")
    
    # Calculate growth rate
    local growth_rate=$(awk "BEGIN {printf \"%.2f\", ($total_growth / $first_tokens) * 100}")
    
    # Detect pattern
    local pattern="linear"
    local healthy=true
    
    # Check for exponential growth (each turn grows more than previous)
    local exponential=true
    for ((i=1; i<turn_count-1; i++)); do
        local curr_tokens
        curr_tokens=$(echo "$history_json" | jq ".[$i].tokens")
        
        local next_tokens
        next_tokens=$(echo "$history_json" | jq ".[$(($i+1))].tokens")
        
        local prev_tokens
        prev_tokens=$(echo "$history_json" | jq ".[$(($i-1))].tokens")
        
        local curr_growth=$((curr_tokens - prev_tokens))
        local next_growth=$((next_tokens - curr_tokens))
        
        if [ "$next_growth" -le "$curr_growth" ]; then
            exponential=false
            break
        fi
    done
    
    if [ "$exponential" = true ]; then
        pattern="exponential"
        healthy=false
    fi
    
    # Check for staircase (sudden jumps)
    local max_single_turn_growth=0
    for ((i=0; i<turn_count-1; i++)); do
        local curr_tokens
        curr_tokens=$(echo "$history_json" | jq ".[$i].tokens")
        
        local next_tokens
        next_tokens=$(echo "$history_json" | jq ".[$(($i+1))].tokens")
        
        local single_growth=$((next_tokens - curr_tokens))
        if [ "$single_growth" -gt "$max_single_turn_growth" ]; then
            max_single_turn_growth=$single_growth
        fi
    done
    
    # If any single turn > 2x average, it's staircase
    local staircase_threshold=$(awk "BEGIN {printf \"%.0f\", $avg_growth_per_turn * 2}")
    if [ "$max_single_turn_growth" -gt "$staircase_threshold" ]; then
        pattern="staircase"
        healthy=false
    fi
    
    cat <<EOF
{
  "turn_count": $turn_count,
  "first_tokens": $first_tokens,
  "last_tokens": $last_tokens,
  "total_growth": $total_growth,
  "avg_growth_per_turn": $avg_growth_per_turn,
  "growth_rate_pct": $growth_rate,
  "pattern": "$pattern",
  "healthy": $healthy,
  "max_single_turn_growth": $max_single_turn_growth
}
EOF
}

#######################################
# Detect retry storm
# Arguments:
#   $1 - Retry history JSON (array of {timestamp, error_type})
# Outputs:
#   JSON with retry storm analysis
#######################################
detect_retry_storm() {
    local retry_history="$1"
    
    local retry_count
    retry_count=$(echo "$retry_history" | jq 'length')
    
    if [ "$retry_count" -eq 0 ]; then
        echo '{"storm_detected": false, "retry_count": 0}'
        return 0
    fi
    
    # Get first and last timestamps
    local first_ts
    first_ts=$(echo "$retry_history" | jq -r '.[0].timestamp')
    
    local last_ts
    last_ts=$(echo "$retry_history" | jq -r '.[-1].timestamp')
    
    # Calculate duration in seconds (simplified - assumes ISO8601)
    local duration=600  # Default 10 minutes if can't parse
    
    if command -v date >/dev/null 2>&1; then
        local first_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$first_ts" "+%s" 2>/dev/null || echo 0)
        local last_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%SZ" "$last_ts" "+%s" 2>/dev/null || echo 600)
        if [ "$first_epoch" -gt 0 ] && [ "$last_epoch" -gt 0 ]; then
            duration=$((last_epoch - first_epoch))
        fi
    fi
    
    local duration_minutes=$(awk "BEGIN {printf \"%.1f\", $duration / 60}")
    
    # Storm criteria: 3+ retries in 10 minutes
    local storm_detected=false
    local severity="none"
    
    if [ "$retry_count" -ge 3 ] && [ "$duration" -le 600 ]; then
        storm_detected=true
        
        if [ "$retry_count" -ge 7 ]; then
            severity="critical"
        elif [ "$retry_count" -ge 5 ]; then
            severity="high"
        else
            severity="moderate"
        fi
    fi
    
    # Calculate retry frequency (retries per minute)
    local frequency=0
    if [ "$duration" -gt 0 ]; then
        frequency=$(awk "BEGIN {printf \"%.2f\", $retry_count / ($duration / 60)}")
    fi
    
    cat <<EOF
{
  "storm_detected": $storm_detected,
  "retry_count": $retry_count,
  "duration_seconds": $duration,
  "duration_minutes": $duration_minutes,
  "frequency_per_minute": $frequency,
  "severity": "$severity"
}
EOF
}

#######################################
# Generate recommendations
# Arguments:
#   $1 - Pressure analysis JSON
#   $2 - Workload classification (LOCAL_SAFE, CLOUD_PREFERRED, etc.)
# Outputs:
#   JSON with recommendations
#######################################
generate_recommendations() {
    local pressure_json="$1"
    local workload="${2:-UNKNOWN}"
    
    local state
    state=$(echo "$pressure_json" | jq -r '.state')
    
    local retry_count
    retry_count=$(echo "$pressure_json" | jq -r '.retry_count')
    
    local compression_attempts
    compression_attempts=$(echo "$pressure_json" | jq -r '.compression_attempts')
    
    local recommendations='[]'
    
    case "$state" in
        SAFE)
            recommendations='["continue_normally"]'
            ;;
        LARGE)
            recommendations='["monitor_closely"]'
            
            # Add decomposition suggestion
            recommendations=$(echo "$recommendations" | jq '. + ["consider_decomposition_for_next_task"]')
            
            # If LOCAL_SAFE, suggest local routing as prep
            if [ "$workload" = "LOCAL_SAFE" ]; then
                recommendations=$(echo "$recommendations" | jq '. + ["prepare_local_routing_option"]')
            fi
            ;;
        OVERSIZED)
            recommendations='["reset_session_recommended"]'
            
            # Compression if not tried
            if [ "$compression_attempts" -eq 0 ]; then
                recommendations=$(echo "$recommendations" | jq '. + ["try_compression"]')
            fi
            
            # Decomposition
            recommendations=$(echo "$recommendations" | jq '. + ["decompose_remaining_work"]')
            
            # Local routing if safe
            if [ "$workload" = "LOCAL_SAFE" ]; then
                recommendations=$(echo "$recommendations" | jq '. + ["consider_local_ollama"]')
            fi
            
            # Provider escalation if available
            recommendations=$(echo "$recommendations" | jq '. + ["consider_longer_context_provider"]')
            ;;
        CRITICAL)
            recommendations='["immediate_session_reset"]'
            recommendations=$(echo "$recommendations" | jq '. + ["block_large_operations"]')
            recommendations=$(echo "$recommendations" | jq '. + ["preserve_current_session_for_reference"]')
            
            if [ "$retry_count" -ge 5 ]; then
                recommendations=$(echo "$recommendations" | jq '. + ["stop_retry_loop"]')
            fi
            
            if [ "$workload" = "CLOUD_REQUIRED" ]; then
                recommendations=$(echo "$recommendations" | jq '. + ["decompose_required_cannot_use_local"]')
            fi
            ;;
    esac
    
    # Retry-specific recommendations
    if [ "$retry_count" -ge 3 ]; then
        recommendations=$(echo "$recommendations" | jq '. + ["retry_amplification_mitigation"]')
    fi
    
    if [ "$retry_count" -ge 5 ]; then
        recommendations=$(echo "$recommendations" | jq '. + ["alternative_approach_required"]')
    fi
    
    # Compression-specific
    if [ "$compression_attempts" -ge 2 ]; then
        recommendations=$(echo "$recommendations" | jq '. + ["compression_ineffective_reset_instead"]')
    fi
    
    cat <<EOF
{
  "state": "$state",
  "workload": "$workload",
  "recommendations": $recommendations
}
EOF
}

#######################################
# Format recommendations as human-readable text
# Arguments:
#   $1 - Recommendations JSON
# Outputs:
#   Formatted text
#######################################
format_recommendations() {
    local rec_json="$1"
    
    local state
    state=$(echo "$rec_json" | jq -r '.state')
    
    local recommendations
    recommendations=$(echo "$rec_json" | jq -r '.recommendations[]')
    
    echo "Recommended Actions ($state):"
    echo ""
    
    while IFS= read -r rec; do
        case "$rec" in
            continue_normally)
                echo "  ✓ Continue normal operations"
                ;;
            monitor_closely)
                echo "  👁  Monitor context growth closely"
                ;;
            consider_decomposition_for_next_task)
                echo "  📦 Consider decomposing next major task into smaller chunks"
                ;;
            prepare_local_routing_option)
                echo "  🏠 Prepare local Ollama routing as backup option"
                ;;
            reset_session_recommended)
                echo "  🔄 Session reset recommended before next major operation"
                ;;
            try_compression)
                echo "  🗜️  Try context compression to reduce token count"
                ;;
            decompose_remaining_work)
                echo "  ✂️  Decompose remaining work into sequential subtasks"
                ;;
            consider_local_ollama)
                echo "  🏠 Consider local Ollama for this workload (context pressure + safe task)"
                ;;
            consider_longer_context_provider)
                echo "  ☁️  Consider switching to longer-context provider if available"
                ;;
            immediate_session_reset)
                echo "  🚨 IMMEDIATE session reset required"
                ;;
            block_large_operations)
                echo "  🛑 Block large operations in current session"
                ;;
            preserve_current_session_for_reference)
                echo "  💾 Preserve current session for reference (don't delete)"
                ;;
            stop_retry_loop)
                echo "  🔴 Stop retry loop - alternative approach needed"
                ;;
            decompose_required_cannot_use_local)
                echo "  ⚠️  Decomposition required (task too complex for local routing)"
                ;;
            retry_amplification_mitigation)
                echo "  ⚡ Retry amplification detected - context growing with each retry"
                ;;
            alternative_approach_required)
                echo "  🔀 Alternative approach required (retries not working)"
                ;;
            compression_ineffective_reset_instead)
                echo "  ❌ Compression ineffective - reset session instead"
                ;;
            *)
                echo "  • $rec"
                ;;
        esac
    done
}

#######################################
# Main command dispatcher
#######################################
main() {
    local command="${1:-}"
    
    case "$command" in
        monitor)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 monitor <session_json>" >&2
                exit 1
            fi
            monitor_session "$2"
            ;;
        check_growth)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 check_growth <history_json>" >&2
                exit 1
            fi
            check_growth "$2"
            ;;
        detect_retry_storm)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 detect_retry_storm <retry_history_json>" >&2
                exit 1
            fi
            detect_retry_storm "$2"
            ;;
        recommend)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 recommend <pressure_json> [workload_classification]" >&2
                exit 1
            fi
            generate_recommendations "$2" "${3:-UNKNOWN}"
            ;;
        format_recommendations)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 format_recommendations <recommendations_json>" >&2
                exit 1
            fi
            format_recommendations "$2"
            ;;
        version)
            echo "context-guardrails.sh version $VERSION"
            ;;
        help|--help|-h)
            cat <<EOF
context-guardrails.sh - Context pressure monitoring and mitigation

USAGE:
    $0 monitor <session_json>
        Monitor session context state and pressure
    
    $0 check_growth <history_json>
        Analyze context growth pattern
        history_json: array of {turn: N, tokens: M}
    
    $0 detect_retry_storm <retry_history_json>
        Detect retry amplification patterns
        retry_history_json: array of {timestamp: ISO8601, error_type: string}
    
    $0 recommend <pressure_json> [workload_classification]
        Generate mitigation recommendations
        workload: LOCAL_SAFE | CLOUD_PREFERRED | CLOUD_REQUIRED | UNKNOWN
    
    $0 format_recommendations <recommendations_json>
        Format recommendations as human-readable text
    
    $0 version
        Show version
    
    $0 help
        Show this help

EXAMPLES:
    # Monitor session
    $0 monitor '{"current_tokens": 78000, "provider_limit": 128000, "growth_rate_pct": 12, "retry_count": 2, "turn": 15}'
    
    # Check growth pattern
    $0 check_growth '[{"turn":1,"tokens":20000},{"turn":5,"tokens":35000},{"turn":10,"tokens":62000}]'
    
    # Detect retry storm
    $0 detect_retry_storm '[{"timestamp":"2026-05-23T10:00:00Z","error_type":"503"},{"timestamp":"2026-05-23T10:02:00Z","error_type":"503"}]'
    
    # Generate recommendations
    $0 recommend '{"state":"OVERSIZED","retry_count":3,"compression_attempts":1}' LOCAL_SAFE

OUTPUT:
    JSON objects for programmatic consumption

EXIT CODES:
    0 - Success
    1 - Invalid arguments
    2 - Check failed

SEE ALSO:
    docs/CONTEXT_GUARDRAILS.md
    docs/TOKEN_BUDGETING.md
    scripts/context-budget-estimator.sh
EOF
            ;;
        *)
            echo "Error: Unknown command '$command'" >&2
            echo "Run '$0 help' for usage" >&2
            exit 1
            ;;
    esac
}

main "$@"
