#!/usr/bin/env bash
# context-budget-estimator.sh - Estimate token consumption and budget allocation
#
# Usage:
#   ./context-budget-estimator.sh estimate "text content here"
#   ./context-budget-estimator.sh estimate_file <path>
#   ./context-budget-estimator.sh estimate_session <session_data_json>
#   ./context-budget-estimator.sh calculate_pressure <current_tokens> <provider_limit> <growth_rate> <retry_count> <compression_attempts>
#
# Exit codes:
#   0 = Success
#   1 = Invalid arguments
#   2 = Estimation failed

set -euo pipefail

VERSION="1.0.0"

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

# Token estimation multipliers (conservative)
ENGLISH_TEXT_MULTIPLIER=0.30
CODE_TEXT_MULTIPLIER=0.50
MIXED_TEXT_MULTIPLIER=0.40
OVERHEAD_MULTIPLIER=1.20  # 20% overhead for formatting

#######################################
# Estimate tokens from text
# Arguments:
#   $1 - Text content
#   $2 - Content type (text|code|mixed), default: mixed
# Outputs:
#   Estimated token count
#######################################
estimate_tokens() {
    local content="$1"
    local content_type="${2:-mixed}"
    
    local char_count=${#content}
    local multiplier=$MIXED_TEXT_MULTIPLIER
    
    case "$content_type" in
        text|english)
            multiplier=$ENGLISH_TEXT_MULTIPLIER
            ;;
        code)
            multiplier=$CODE_TEXT_MULTIPLIER
            ;;
        mixed)
            multiplier=$MIXED_TEXT_MULTIPLIER
            ;;
        *)
            multiplier=$MIXED_TEXT_MULTIPLIER
            ;;
    esac
    
    # Calculate: chars * multiplier * overhead
    local tokens=$(awk "BEGIN {printf \"%.0f\", $char_count * $multiplier * $OVERHEAD_MULTIPLIER}")
    
    echo "$tokens"
}

#######################################
# Estimate tokens from file
# Arguments:
#   $1 - File path
# Outputs:
#   JSON with file stats and token estimate
#######################################
estimate_file() {
    local file_path="$1"
    
    if [ ! -f "$file_path" ]; then
        echo "{\"error\": \"File not found: $file_path\"}"
        return 1
    fi
    
    local file_size
    file_size=$(wc -c < "$file_path" | tr -d ' ')
    
    local line_count
    line_count=$(wc -l < "$file_path" | tr -d ' ')
    
    # Detect content type by extension
    local content_type="mixed"
    case "$file_path" in
        *.py|*.js|*.ts|*.java|*.c|*.cpp|*.go|*.rs|*.sh)
            content_type="code"
            ;;
        *.txt|*.md|*.log)
            content_type="text"
            ;;
        *)
            content_type="mixed"
            ;;
    esac
    
    # Read file content (limit to 1MB for safety)
    if [ "$file_size" -gt 1048576 ]; then
        local estimated_tokens=$(awk "BEGIN {printf \"%.0f\", $file_size * 0.45}")
        cat <<EOF
{
  "file": "$file_path",
  "size_bytes": $file_size,
  "lines": $line_count,
  "content_type": "$content_type",
  "estimated_tokens": $estimated_tokens,
  "warning": "File > 1MB, used size-based estimation"
}
EOF
        return 0
    fi
    
    local content
    content=$(cat "$file_path")
    
    local estimated_tokens
    estimated_tokens=$(estimate_tokens "$content" "$content_type")
    
    cat <<EOF
{
  "file": "$file_path",
  "size_bytes": $file_size,
  "lines": $line_count,
  "content_type": "$content_type",
  "estimated_tokens": $estimated_tokens
}
EOF
}

#######################################
# Calculate pressure score
# Arguments:
#   $1 - Current tokens
#   $2 - Provider limit
#   $3 - Growth rate (percentage)
#   $4 - Retry count
#   $5 - Compression attempts
# Outputs:
#   JSON with pressure analysis
#######################################
calculate_pressure() {
    local current_tokens="$1"
    local provider_limit="$2"
    local growth_rate="${3:-0}"
    local retry_count="${4:-0}"
    local compression_attempts="${5:-0}"
    
    # Calculate usage percentage
    local usage_pct=$(awk "BEGIN {printf \"%.2f\", ($current_tokens / $provider_limit) * 100}")
    
    # Calculate pressure score
    # Formula: (usage_pct * 0.4) + (growth_rate * 0.3) + (retry_count * 5) + (compression_attempts * 10)
    local pressure_score=$(awk "BEGIN {printf \"%.2f\", ($usage_pct * 0.4) + ($growth_rate * 0.3) + ($retry_count * 5) + ($compression_attempts * 10)}")
    
    # Determine state
    local state="SAFE"
    if (( $(awk "BEGIN {print ($usage_pct >= 90)}") )); then
        state="CRITICAL"
    elif (( $(awk "BEGIN {print ($usage_pct >= 75)}") )); then
        state="OVERSIZED"
    elif (( $(awk "BEGIN {print ($usage_pct >= 50)}") )); then
        state="LARGE"
    fi
    
    # Pressure score can override state
    if (( $(awk "BEGIN {print ($pressure_score >= 76)}") )); then
        state="CRITICAL"
    elif (( $(awk "BEGIN {print ($pressure_score >= 51)}") )); then
        state="OVERSIZED"
    elif (( $(awk "BEGIN {print ($pressure_score >= 26)}") )); then
        state="LARGE"
    fi
    
    # Calculate remaining budget
    local remaining=$((provider_limit - current_tokens))
    local remaining_pct=$(awk "BEGIN {printf \"%.2f\", ($remaining / $provider_limit) * 100}")
    
    # Generate recommendations
    local recommendations='[]'
    case "$state" in
        SAFE)
            recommendations='["operate_normally"]'
            ;;
        LARGE)
            recommendations='["monitor_closely", "prepare_for_decomposition"]'
            ;;
        OVERSIZED)
            recommendations='["reset_session", "decompose_task", "compress_context"]'
            ;;
        CRITICAL)
            recommendations='["immediate_reset", "block_large_operations", "emergency_intervention"]'
            ;;
    esac
    
    # Check for retry amplification
    if [ "$retry_count" -ge 3 ]; then
        recommendations=$(echo "$recommendations" | jq '. + ["retry_amplification_detected"]')
    fi
    
    if [ "$compression_attempts" -ge 2 ]; then
        recommendations=$(echo "$recommendations" | jq '. + ["compression_ineffective"]')
    fi
    
    cat <<EOF
{
  "current_tokens": $current_tokens,
  "provider_limit": $provider_limit,
  "usage_pct": $usage_pct,
  "remaining_tokens": $remaining,
  "remaining_pct": $remaining_pct,
  "growth_rate_pct": $growth_rate,
  "retry_count": $retry_count,
  "compression_attempts": $compression_attempts,
  "pressure_score": $pressure_score,
  "state": "$state",
  "recommendations": $recommendations
}
EOF
}

#######################################
# Estimate session context size
# Arguments:
#   $1 - JSON with session data
# Outputs:
#   JSON with component breakdown
#######################################
estimate_session() {
    local session_json="$1"
    
    # Extract components (if not provided, use defaults)
    local system_prompt_tokens
    system_prompt_tokens=$(echo "$session_json" | jq -r '.system_prompt_tokens // 18000')
    
    local conversation_tokens
    conversation_tokens=$(echo "$session_json" | jq -r '.conversation_tokens // 20000')
    
    local tool_results_tokens
    tool_results_tokens=$(echo "$session_json" | jq -r '.tool_results_tokens // 5000')
    
    local reasoning_tokens
    reasoning_tokens=$(echo "$session_json" | jq -r '.reasoning_tokens // 0')
    
    local provider_limit
    provider_limit=$(echo "$session_json" | jq -r '.provider_limit // 128000')
    
    local reserve_pct
    reserve_pct=$(echo "$session_json" | jq -r '.reserve_pct // 15')
    
    # Calculate reserve
    local reserve_tokens=$(awk "BEGIN {printf \"%.0f\", $provider_limit * ($reserve_pct / 100)}")
    
    # Calculate total
    local total_tokens=$((system_prompt_tokens + conversation_tokens + tool_results_tokens + reasoning_tokens))
    
    local available_budget=$((provider_limit - reserve_tokens))
    local utilization_pct=$(awk "BEGIN {printf \"%.2f\", ($total_tokens / $available_budget) * 100}")
    
    cat <<EOF
{
  "components": {
    "system_prompt": $system_prompt_tokens,
    "conversation_history": $conversation_tokens,
    "tool_results": $tool_results_tokens,
    "reasoning": $reasoning_tokens,
    "reserve": $reserve_tokens
  },
  "total_tokens": $total_tokens,
  "available_budget": $available_budget,
  "provider_limit": $provider_limit,
  "utilization_pct": $utilization_pct
}
EOF
}

#######################################
# Detect giant prompt
# Arguments:
#   $1 - Token count for single turn
#   $2 - Provider context window size
# Outputs:
#   JSON with detection result
#######################################
detect_giant_prompt() {
    local turn_tokens="$1"
    local provider_limit="$2"
    
    # Thresholds based on context window
    local small_threshold=8000
    local medium_threshold=32000
    local large_threshold=64000
    
    local threshold=$medium_threshold
    local category="medium"
    
    if [ "$provider_limit" -lt 32000 ]; then
        threshold=$small_threshold
        category="small"
    elif [ "$provider_limit" -ge 128000 ]; then
        threshold=$large_threshold
        category="large"
    fi
    
    local is_giant=false
    local severity="normal"
    
    if [ "$turn_tokens" -gt "$threshold" ]; then
        is_giant=true
        
        local ratio=$(awk "BEGIN {printf \"%.2f\", $turn_tokens / $threshold}")
        
        if (( $(awk "BEGIN {print ($ratio >= 3)}") )); then
            severity="extreme"
        elif (( $(awk "BEGIN {print ($ratio >= 2)}") )); then
            severity="high"
        else
            severity="moderate"
        fi
    fi
    
    cat <<EOF
{
  "turn_tokens": $turn_tokens,
  "provider_limit": $provider_limit,
  "model_category": "$category",
  "threshold": $threshold,
  "is_giant": $is_giant,
  "severity": "$severity",
  "ratio": $(awk "BEGIN {printf \"%.2f\", $turn_tokens / $threshold}")
}
EOF
}

#######################################
# Generate advisory message
# Arguments:
#   $1 - Pressure analysis JSON
# Outputs:
#   Human-readable advisory
#######################################
generate_advisory() {
    local pressure_json="$1"
    
    local state
    state=$(echo "$pressure_json" | jq -r '.state')
    
    local current_tokens
    current_tokens=$(echo "$pressure_json" | jq -r '.current_tokens')
    
    local provider_limit
    provider_limit=$(echo "$pressure_json" | jq -r '.provider_limit')
    
    local usage_pct
    usage_pct=$(echo "$pressure_json" | jq -r '.usage_pct')
    
    local retry_count
    retry_count=$(echo "$pressure_json" | jq -r '.retry_count')
    
    local growth_rate
    growth_rate=$(echo "$pressure_json" | jq -r '.growth_rate_pct')
    
    local pressure_score
    pressure_score=$(echo "$pressure_json" | jq -r '.pressure_score')
    
    case "$state" in
        SAFE)
            echo -e "${GREEN}✓ Context: ${current_tokens} / ${provider_limit} tokens (${usage_pct}%) [SAFE]${NC}"
            ;;
        LARGE)
            echo -e "${YELLOW}⚠️  Context: ${current_tokens} / ${provider_limit} tokens (${usage_pct}%) [LARGE]${NC}"
            echo "   Approaching context limit. Consider decomposition for next major task."
            ;;
        OVERSIZED)
            echo -e "${YELLOW}⚠️  Context: ${current_tokens} / ${provider_limit} tokens (${usage_pct}%) [OVERSIZED]${NC}"
            echo "   High context pressure. Recommend starting fresh session."
            if [ "$retry_count" -gt 0 ]; then
                echo "   Retry count: $retry_count | Growth rate: +${growth_rate}% per turn"
            fi
            ;;
        CRITICAL)
            echo -e "${RED}🚨 CRITICAL: Context at ${current_tokens} / ${provider_limit} tokens (${usage_pct}%)${NC}"
            echo ""
            echo "Immediate action required:"
            echo "  • Start new session"
            echo "  • Current session preserved for reference"
            echo "  • Consider task decomposition"
            if [ "$retry_count" -ge 5 ]; then
                echo ""
                echo "Retry storm detected: $retry_count attempts"
            fi
            ;;
    esac
}

#######################################
# Main command dispatcher
#######################################
main() {
    local command="${1:-}"
    
    case "$command" in
        estimate)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 estimate \"text content\" [content_type]" >&2
                exit 1
            fi
            local tokens
            tokens=$(estimate_tokens "$2" "${3:-mixed}")
            echo "{\"estimated_tokens\": $tokens, \"content_type\": \"${3:-mixed}\"}"
            ;;
        estimate_file)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 estimate_file <path>" >&2
                exit 1
            fi
            estimate_file "$2"
            ;;
        estimate_session)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 estimate_session <session_json>" >&2
                exit 1
            fi
            estimate_session "$2"
            ;;
        calculate_pressure)
            if [ $# -lt 3 ]; then
                echo "Usage: $0 calculate_pressure <current_tokens> <provider_limit> [growth_rate] [retry_count] [compression_attempts]" >&2
                exit 1
            fi
            calculate_pressure "$2" "$3" "${4:-0}" "${5:-0}" "${6:-0}"
            ;;
        detect_giant)
            if [ $# -lt 3 ]; then
                echo "Usage: $0 detect_giant <turn_tokens> <provider_limit>" >&2
                exit 1
            fi
            detect_giant_prompt "$2" "$3"
            ;;
        advisory)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 advisory <pressure_json>" >&2
                exit 1
            fi
            generate_advisory "$2"
            ;;
        version)
            echo "context-budget-estimator.sh version $VERSION"
            ;;
        help|--help|-h)
            cat <<EOF
context-budget-estimator.sh - Token consumption and budget estimation

USAGE:
    $0 estimate "text content" [content_type]
        Estimate tokens from text
        content_type: text|code|mixed (default: mixed)
    
    $0 estimate_file <path>
        Estimate tokens from file
    
    $0 estimate_session <session_json>
        Estimate total session context size
    
    $0 calculate_pressure <current> <limit> [growth] [retries] [compressions]
        Calculate context pressure score and state
    
    $0 detect_giant <turn_tokens> <provider_limit>
        Detect giant prompt in single turn
    
    $0 advisory <pressure_json>
        Generate human-readable advisory message
    
    $0 version
        Show version
    
    $0 help
        Show this help

EXAMPLES:
    # Estimate tokens in text
    $0 estimate "Hello world" text
    
    # Estimate file size
    $0 estimate_file path/to/file.py
    
    # Calculate pressure
    $0 calculate_pressure 78000 128000 12 2 0
    
    # Detect giant prompt
    $0 detect_giant 85000 128000

OUTPUT:
    JSON objects for programmatic consumption

EXIT CODES:
    0 - Success
    1 - Invalid arguments
    2 - Estimation failed

SEE ALSO:
    docs/CONTEXT_GUARDRAILS.md
    docs/TOKEN_BUDGETING.md
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
