#!/usr/bin/env bash
# ollama-model-health.sh - Check Ollama service and model availability
#
# Usage:
#   ./ollama-model-health.sh check
#   ./ollama-model-health.sh models
#   ./ollama-model-health.sh recommend "workload_classification"
#
# Exit codes:
#   0 = Ollama healthy
#   1 = Invalid arguments
#   2 = Ollama unavailable
#   3 = Required models missing

set -euo pipefail

VERSION="1.0.0"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
CACHE_DIR="${HOME}/.hermes/cache/ollama"
CACHE_TTL=30  # seconds

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
# Check if Ollama service is running
# Returns:
#   0 if running, 1 otherwise
# Outputs:
#   JSON health status
#######################################
check_ollama_health() {
    local response
    local http_code
    
    # Try to reach Ollama API
    if ! response=$(curl -s -w "\n%{http_code}" --connect-timeout 2 --max-time 5 "$OLLAMA_URL/api/tags" 2>&1); then
        cat <<EOF
{
  "healthy": false,
  "url": "$OLLAMA_URL",
  "error": "Connection failed",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
        return 1
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" != "200" ]; then
        cat <<EOF
{
  "healthy": false,
  "url": "$OLLAMA_URL",
  "error": "HTTP $http_code",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
        return 1
    fi
    
    cat <<EOF
{
  "healthy": true,
  "url": "$OLLAMA_URL",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    return 0
}

#######################################
# Get list of available models with caching
# Returns:
#   0 if successful, 1 otherwise
# Outputs:
#   JSON array of models
#######################################
get_available_models() {
    mkdir -p "$CACHE_DIR"
    local cache_file="$CACHE_DIR/models.json"
    local cache_age=999999
    
    # Check cache freshness
    if [ -f "$cache_file" ]; then
        cache_age=$(( $(date +%s) - $(stat -f %m "$cache_file" 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null || echo 0) ))
    fi
    
    # Use cache if fresh
    if [ "$cache_age" -lt "$CACHE_TTL" ]; then
        cat "$cache_file"
        return 0
    fi
    
    # Fetch fresh data
    local response
    local http_code
    
    if ! response=$(curl -s -w "\n%{http_code}" --connect-timeout 2 --max-time 5 "$OLLAMA_URL/api/tags" 2>&1); then
        # Return empty array on failure
        echo '{"models": [], "cached": false, "error": "Connection failed"}'
        return 1
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" != "200" ]; then
        echo '{"models": [], "cached": false, "error": "HTTP '"$http_code"'"}'
        return 1
    fi
    
    # Parse and cache
    local result
    result=$(echo "$body" | jq -c '{models: [.models[] | {name: .name, size: .size, modified_at: .modified_at}], cached: false, error: null}')
    
    echo "$result" > "$cache_file"
    echo "$result"
    return 0
}

#######################################
# Check if a specific model is available
# Arguments:
#   $1 - Model name (e.g., "llama3.1:8b")
# Returns:
#   0 if available, 1 otherwise
#######################################
is_model_available() {
    local target_model="$1"
    local models_json
    
    if ! models_json=$(get_available_models); then
        return 1
    fi
    
    # Check if model exists in list
    echo "$models_json" | jq -e --arg model "$target_model" '.models[] | select(.name == $model)' >/dev/null 2>&1
}

#######################################
# Recommend models for workload classification
# Arguments:
#   $1 - Classification (LOCAL_SAFE|CLOUD_PREFERRED|CLOUD_REQUIRED|FALLBACK_ONLY)
# Outputs:
#   JSON object with model recommendations and availability
#######################################
recommend_models() {
    local classification="$1"
    local models_json
    
    if ! models_json=$(get_available_models); then
        cat <<EOF
{
  "classification": "$classification",
  "recommendations": [],
  "error": "Ollama unavailable",
  "ollama_healthy": false
}
EOF
        return 1
    fi
    
    # Define model recommendations per classification
    local primary_model=""
    local fallback_model=""
    local rationale=""
    
    case "$classification" in
        LOCAL_SAFE|FALLBACK_ONLY)
            primary_model="llama3.1:8b"
            fallback_model="llama3.2:3b"
            rationale="Fast, efficient for text tasks and summarization"
            ;;
        CLOUD_PREFERRED)
            primary_model="qwen2.5-coder:14b"
            fallback_model="qwen2.5-coder:7b"
            rationale="Code-aware, reasoning capable for review and planning tasks"
            ;;
        CLOUD_REQUIRED)
            cat <<EOF
{
  "classification": "$classification",
  "recommendations": [],
  "error": "CLOUD_REQUIRED workloads should not use local inference",
  "ollama_healthy": true
}
EOF
            return 1
            ;;
        *)
            cat <<EOF
{
  "classification": "$classification",
  "recommendations": [],
  "error": "Unknown classification",
  "ollama_healthy": true
}
EOF
            return 1
            ;;
    esac
    
    # Check availability
    local primary_available=false
    local fallback_available=false
    local primary_size="unknown"
    local fallback_size="unknown"
    
    if is_model_available "$primary_model"; then
        primary_available=true
        primary_size=$(echo "$models_json" | jq -r --arg m "$primary_model" '.models[] | select(.name == $m) | .size')
    fi
    
    if is_model_available "$fallback_model"; then
        fallback_available=true
        fallback_size=$(echo "$models_json" | jq -r --arg m "$fallback_model" '.models[] | select(.name == $m) | .size')
    fi
    
    # Build recommendations
    cat <<EOF
{
  "classification": "$classification",
  "recommendations": [
    {
      "model": "$primary_model",
      "available": $primary_available,
      "size": "$primary_size",
      "priority": "primary",
      "rationale": "$rationale"
    },
    {
      "model": "$fallback_model",
      "available": $fallback_available,
      "size": "$fallback_size",
      "priority": "fallback",
      "rationale": "Smaller, faster alternative"
    }
  ],
  "ollama_healthy": true,
  "any_available": $([ "$primary_available" = true ] || [ "$fallback_available" = true ] && echo true || echo false)
}
EOF
}

#######################################
# Format bytes to human-readable size
# Arguments:
#   $1 - Size in bytes
# Outputs:
#   Human-readable size (e.g., "4.7GB")
#######################################
format_size() {
    local bytes="$1"
    
    if [ "$bytes" = "unknown" ] || [ -z "$bytes" ]; then
        echo "unknown"
        return
    fi
    
    # Use numfmt if available, otherwise manual calculation
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec-i --suffix=B "$bytes" | sed 's/iB/B/'
    else
        # Manual calculation
        local units=("B" "KB" "MB" "GB" "TB")
        local unit_idx=0
        local size=$bytes
        
        while [ $size -gt 1024 ] && [ $unit_idx -lt ${#units[@]} ]; do
            size=$((size / 1024))
            unit_idx=$((unit_idx + 1))
        done
        
        echo "${size}${units[$unit_idx]}"
    fi
}

#######################################
# Generate advisory message for user
# Arguments:
#   $1 - Classification
#   $2 - Provider status
#   $3 - Outage minutes
# Outputs:
#   Human-readable advisory message
#######################################
generate_advisory_message() {
    local classification="$1"
    local provider_status="$2"
    local outage_minutes="$3"
    
    # Get model recommendations
    local recommendations_json
    if ! recommendations_json=$(recommend_models "$classification"); then
        echo -e "${RED}⚠️  Cloud provider ($provider_status).${NC}"
        echo ""
        echo "This is a $classification workload."
        echo ""
        if echo "$recommendations_json" | jq -e '.error' >/dev/null 2>&1; then
            local error_msg
            error_msg=$(echo "$recommendations_json" | jq -r .error)
            echo "Note: $error_msg"
        fi
        return
    fi
    
    local any_available
    any_available=$(echo "$recommendations_json" | jq -r .any_available)
    
    if [ "$any_available" = "false" ]; then
        echo -e "${RED}⚠️  Cloud provider ($provider_status). Ollama available but recommended models not installed.${NC}"
        echo ""
        local primary_model
        local fallback_model
        primary_model=$(echo "$recommendations_json" | jq -r '.recommendations[0].model')
        fallback_model=$(echo "$recommendations_json" | jq -r '.recommendations[1].model')
        echo "To use local inference, install one of:"
        echo "  • $primary_model (recommended)"
        echo "  • $fallback_model (lighter alternative)"
        echo ""
        echo "Install with: ollama pull <model>"
        return
    fi
    
    # Find best available model
    local recommended_model
    local model_size
    local rationale
    
    if echo "$recommendations_json" | jq -e '.recommendations[0].available == true' >/dev/null 2>&1; then
        recommended_model=$(echo "$recommendations_json" | jq -r '.recommendations[0].model')
        model_size=$(echo "$recommendations_json" | jq -r '.recommendations[0].size')
        rationale=$(echo "$recommendations_json" | jq -r '.recommendations[0].rationale')
    else
        recommended_model=$(echo "$recommendations_json" | jq -r '.recommendations[1].model')
        model_size=$(echo "$recommendations_json" | jq -r '.recommendations[1].size')
        rationale=$(echo "$recommendations_json" | jq -r '.recommendations[1].rationale')
    fi
    
    local human_size
    human_size=$(format_size "$model_size")
    
    # Generate advisory
    echo -e "${YELLOW}⚠️  Cloud provider ($provider_status)${NC}"
    if [ "$outage_minutes" -gt 0 ]; then
        echo "    Duration: ${outage_minutes} minute(s)"
    fi
    echo ""
    echo "This appears to be a $classification workload."
    echo ""
    echo -e "${GREEN}Recommendation: Consider using local Ollama for this task.${NC}"
    echo ""
    echo "Available model:"
    echo "  • $recommended_model ($human_size)"
    echo "    $rationale"
    echo ""
    echo "To use Ollama:"
    echo "  hermes config set provider ollama"
    echo "  hermes config set model $recommended_model"
    echo ""
    echo "Or wait for cloud recovery (monitoring continues)."
}

#######################################
# Main command dispatcher
#######################################
main() {
    local command="${1:-check}"
    
    case "$command" in
        check|health)
            check_ollama_health
            ;;
        models|list)
            get_available_models
            ;;
        recommend)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 recommend <classification>" >&2
                exit 1
            fi
            recommend_models "$2"
            ;;
        advisory)
            if [ $# -lt 4 ]; then
                echo "Usage: $0 advisory <classification> <provider_status> <outage_minutes>" >&2
                exit 1
            fi
            generate_advisory_message "$2" "$3" "$4"
            ;;
        version)
            echo "ollama-model-health.sh version $VERSION"
            ;;
        help|--help|-h)
            cat <<EOF
ollama-model-health.sh - Ollama service and model health checks

USAGE:
    $0 check
        Check if Ollama service is running
    
    $0 models
        List available models (with 30s cache)
    
    $0 recommend <classification>
        Get model recommendations for workload classification
        classification: LOCAL_SAFE | CLOUD_PREFERRED | FALLBACK_ONLY
    
    $0 advisory <classification> <provider_status> <outage_minutes>
        Generate human-readable advisory message
    
    $0 version
        Show version
    
    $0 help
        Show this help

EXAMPLES:
    # Check Ollama health
    $0 check
    
    # List available models
    $0 models
    
    # Get recommendations for summarization task
    $0 recommend LOCAL_SAFE
    
    # Generate advisory message
    $0 advisory LOCAL_SAFE degraded 7

CONFIGURATION:
    OLLAMA_URL - Ollama API URL (default: http://localhost:11434)

OUTPUT:
    JSON objects for programmatic consumption

EXIT CODES:
    0 - Success
    1 - Invalid arguments
    2 - Ollama unavailable
    3 - Required models missing

SEE ALSO:
    docs/OLLAMA_ROUTING_STRATEGY.md
    scripts/ollama-routing-policy.sh
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
