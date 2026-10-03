#!/usr/bin/env bash
# ollama-routing-policy.sh - Workload classification for Ollama routing recommendations
#
# Usage:
#   ./ollama-routing-policy.sh classify "user message here"
#   ./ollama-routing-policy.sh recommend "user message" "provider_status"
#
# Exit codes:
#   0 = Success
#   1 = Invalid arguments
#   2 = Classification failed

set -euo pipefail

VERSION="1.0.0"

# Color output (optional, graceful degradation)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    YELLOW='\033[1;33m'
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED='' YELLOW='' GREEN='' BLUE='' NC=''
fi

# Keyword arrays for classification
# CLOUD_REQUIRED: Security-sensitive, production-critical, high-stakes
declare -a CLOUD_REQUIRED_KEYWORDS=(
    "security" "vulnerability" "vulnerabilities" "CVE"
    "auth" "authentication" "authorization" "permissions"
    "secrets" "secret" "api key" "api keys" "token" "tokens" "credentials" "credential"
    "password" "passwords" "oauth" "jwt" "saml"
    "deploy" "deployment" "production" "prod" "release" "ship" "go live" "cutover"
    "migration" "migrate" "schema change" "alter table" "drop table" "truncate"
    "database" "postgres" "mysql" "mongodb" "sql"
    "invoice" "billing" "payment" "payments" "pricing" "revenue" "cost" "tax" "taxes"
    "financial" "money" "transaction" "transactions"
    "legal" "compliance" "gdpr" "hipaa" "soc2" "audit" "auditing" "regulatory"
    "customer-facing" "user-facing" "public" "marketing" "announcement" "press release"
    "customer email" "client email" "support email"
)

# CLOUD_PREFERRED: Quality matters but not critical
declare -a CLOUD_PREFERRED_KEYWORDS=(
    "review" "code review" "pr review" "pull request review"
    "plan" "planning" "design" "architecture" "approach" "strategy"
    "write tests" "generate tests" "test cases" "test coverage"
    "refactor" "refactoring" "improve" "optimize" "clean up" "cleanup"
    "feature" "implement" "implementation" "build"
)

# LOCAL_SAFE: Low-risk, easily verified, informational
declare -a LOCAL_SAFE_KEYWORDS=(
    "summarize" "summary" "tl;dr" "tldr" "overview" "digest"
    "changelog" "change log" "status report" "log summary"
    "format" "formatting" "lint" "linting" "style" "prettier" "black" "indent" "indentation"
    "how to" "command for" "what's the command" "which command"
    "ls" "grep" "find" "cat" "ps" "df" "du" "top" "htop"
    "explain" "what is" "what are" "how does" "documentation" "docs"
    "explain this code" "explain code" "what does this do" "walk me through"
    "parse ticket" "extract issue" "summarize thread" "summarize issue"
)

# FALLBACK_ONLY: Emergency diagnostics
declare -a FALLBACK_ONLY_KEYWORDS=(
    "check" "verify" "validate" "status" "health" "healthcheck"
    "parse logs" "extract errors" "find pattern" "log analysis"
    "validate config" "check syntax" "verify format" "syntax check"
)

# Production context keywords (upgrade to CLOUD_REQUIRED)
declare -a PRODUCTION_CONTEXT=(
    "production" "prod" "customer" "client" "live"
)

# Development context keywords (downgrade risk)
declare -a DEV_CONTEXT=(
    "staging" "dev" "development" "local" "test" "testing"
)

#######################################
# Normalize user message for classification
# Arguments:
#   $1 - User message
# Outputs:
#   Normalized message (lowercase, trimmed)
#######################################
normalize_message() {
    local msg="$1"
    # Lowercase, trim whitespace, collapse multiple spaces
    echo "$msg" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | tr -s ' '
}

#######################################
# Check if message contains any keyword from array
# Arguments:
#   $1 - Normalized message
#   $@ - Array of keywords (passed as separate args)
# Returns:
#   0 if match found, 1 otherwise
# Outputs:
#   Matched keywords (one per line)
#######################################
contains_keywords() {
    local msg="$1"
    shift
    local keywords=("$@")
    local matches=()
    
    for keyword in "${keywords[@]}"; do
        if [[ "$msg" == *"$keyword"* ]]; then
            matches+=("$keyword")
        fi
    done
    
    if [ ${#matches[@]} -gt 0 ]; then
        printf '%s\n' "${matches[@]}"
        return 0
    else
        return 1
    fi
}

#######################################
# Detect production or development context
# Arguments:
#   $1 - Normalized message
# Outputs:
#   "production", "development", or "unknown"
#######################################
detect_context() {
    local msg="$1"
    
    # Check production context first (higher priority)
    if contains_keywords "$msg" "${PRODUCTION_CONTEXT[@]}" >/dev/null; then
        echo "production"
        return 0
    fi
    
    if contains_keywords "$msg" "${DEV_CONTEXT[@]}" >/dev/null; then
        echo "development"
        return 0
    fi
    
    echo "unknown"
}

#######################################
# Classify workload
# Arguments:
#   $1 - User message
# Outputs:
#   JSON object with classification result
#######################################
classify_workload() {
    local raw_message="$1"
    local normalized
    normalized="$(normalize_message "$raw_message")"
    
    local context
    context="$(detect_context "$normalized")"
    
    local classification="CLOUD_REQUIRED"  # Conservative default
    local matched_keywords=()
    local reasoning=""
    
    # Priority 1: Check CLOUD_REQUIRED (highest priority)
    if keyword_output=$(contains_keywords "$normalized" "${CLOUD_REQUIRED_KEYWORDS[@]}" 2>/dev/null); then
        # Read keywords into array (portable method)
        matched_keywords=()
        while IFS= read -r line; do
            matched_keywords+=("$line")
        done <<< "$keyword_output"
        
        classification="CLOUD_REQUIRED"
        reasoning="Security-sensitive, production-critical, or high-stakes operation detected"
        
        # Emit JSON result
        cat <<EOF
{
  "classification": "$classification",
  "context": "$context",
  "matched_keywords": $(printf '%s\n' "${matched_keywords[@]}" | jq -R . | jq -s .),
  "reasoning": "$reasoning",
  "confidence": "high"
}
EOF
        return 0
    fi
    
    # Priority 2: Check CLOUD_PREFERRED
    if keyword_output=$(contains_keywords "$normalized" "${CLOUD_PREFERRED_KEYWORDS[@]}" 2>/dev/null); then
        matched_keywords=()
        while IFS= read -r line; do
            matched_keywords+=("$line")
        done <<< "$keyword_output"
        
        classification="CLOUD_PREFERRED"
        reasoning="Quality-sensitive operation, cloud preferred but local fallback acceptable"
        
        # Upgrade to CLOUD_REQUIRED if production context
        if [ "$context" = "production" ]; then
            classification="CLOUD_REQUIRED"
            reasoning="$reasoning (upgraded due to production context)"
        fi
        
        cat <<EOF
{
  "classification": "$classification",
  "context": "$context",
  "matched_keywords": $(printf '%s\n' "${matched_keywords[@]}" | jq -R . | jq -s .),
  "reasoning": "$reasoning",
  "confidence": "medium"
}
EOF
        return 0
    fi
    
    # Priority 3: Check LOCAL_SAFE
    if keyword_output=$(contains_keywords "$normalized" "${LOCAL_SAFE_KEYWORDS[@]}" 2>/dev/null); then
        matched_keywords=()
        while IFS= read -r line; do
            matched_keywords+=("$line")
        done <<< "$keyword_output"
        
        classification="LOCAL_SAFE"
        reasoning="Low-risk operation, easily verified, informational"
        
        # Upgrade to CLOUD_REQUIRED if production context with no dev indicators
        if [ "$context" = "production" ]; then
            classification="CLOUD_REQUIRED"
            reasoning="$reasoning (upgraded due to production context)"
        fi
        
        cat <<EOF
{
  "classification": "$classification",
  "context": "$context",
  "matched_keywords": $(printf '%s\n' "${matched_keywords[@]}" | jq -R . | jq -s .),
  "reasoning": "$reasoning",
  "confidence": "high"
}
EOF
        return 0
    fi
    
    # Priority 4: Check FALLBACK_ONLY
    if keyword_output=$(contains_keywords "$normalized" "${FALLBACK_ONLY_KEYWORDS[@]}" 2>/dev/null); then
        matched_keywords=()
        while IFS= read -r line; do
            matched_keywords+=("$line")
        done <<< "$keyword_output"
        
        classification="FALLBACK_ONLY"
        reasoning="Emergency diagnostic operation"
        
        cat <<EOF
{
  "classification": "$classification",
  "context": "$context",
  "matched_keywords": $(printf '%s\n' "${matched_keywords[@]}" | jq -R . | jq -s .),
  "reasoning": "$reasoning",
  "confidence": "low"
}
EOF
        return 0
    fi
    
    # No matches: Conservative default to CLOUD_REQUIRED
    cat <<EOF
{
  "classification": "CLOUD_REQUIRED",
  "context": "$context",
  "matched_keywords": [],
  "reasoning": "No classification patterns matched; conservative default",
  "confidence": "low"
}
EOF
}

#######################################
# Generate routing recommendation
# Arguments:
#   $1 - User message
#   $2 - Provider status (healthy|degraded|unavailable)
#   $3 - Outage duration minutes (optional, default 0)
# Outputs:
#   JSON object with recommendation
#######################################
generate_recommendation() {
    local message="$1"
    local provider_status="${2:-healthy}"
    local outage_minutes="${3:-0}"
    
    # Get classification
    local classification_json
    classification_json="$(classify_workload "$message")"
    
    local classification
    classification="$(echo "$classification_json" | jq -r .classification)"
    
    local should_recommend_local=false
    local recommendation_text=""
    local recommendation_level="none"
    
    case "$provider_status" in
        healthy)
            should_recommend_local=false
            recommendation_text="Use configured cloud provider (normal operation)"
            recommendation_level="none"
            ;;
        degraded)
            if [ "$outage_minutes" -lt 5 ]; then
                should_recommend_local=false
                recommendation_text="Cloud degraded but recent; retry cloud provider"
                recommendation_level="none"
            else
                case "$classification" in
                    LOCAL_SAFE)
                        should_recommend_local=true
                        recommendation_text="Cloud degraded >5min, LOCAL_SAFE workload: recommend local Ollama"
                        recommendation_level="strong"
                        ;;
                    CLOUD_PREFERRED)
                        should_recommend_local=true
                        recommendation_text="Cloud degraded, CLOUD_PREFERRED workload: suggest local with caveat"
                        recommendation_level="weak"
                        ;;
                    CLOUD_REQUIRED)
                        should_recommend_local=false
                        recommendation_text="CLOUD_REQUIRED workload: wait for cloud recovery"
                        recommendation_level="block"
                        ;;
                    FALLBACK_ONLY)
                        if [ "$outage_minutes" -ge 15 ]; then
                            should_recommend_local=true
                            recommendation_text="Cloud degraded >15min, FALLBACK_ONLY: emergency local use acceptable"
                            recommendation_level="weak"
                        else
                            should_recommend_local=false
                            recommendation_text="FALLBACK_ONLY requires >15min outage; wait for cloud"
                            recommendation_level="none"
                        fi
                        ;;
                esac
            fi
            ;;
        unavailable)
            case "$classification" in
                LOCAL_SAFE)
                    should_recommend_local=true
                    recommendation_text="Cloud unavailable, LOCAL_SAFE workload: recommend local Ollama"
                    recommendation_level="strong"
                    ;;
                CLOUD_PREFERRED)
                    should_recommend_local=true
                    recommendation_text="Cloud unavailable, CLOUD_PREFERRED workload: suggest local or wait"
                    recommendation_level="weak"
                    ;;
                CLOUD_REQUIRED)
                    should_recommend_local=false
                    recommendation_text="CLOUD_REQUIRED workload: block operation, manual review required"
                    recommendation_level="block"
                    ;;
                FALLBACK_ONLY)
                    should_recommend_local=true
                    recommendation_text="Cloud unavailable, FALLBACK_ONLY: emergency local use acceptable"
                    recommendation_level="weak"
                    ;;
            esac
            ;;
        *)
            should_recommend_local=false
            recommendation_text="Unknown provider status; conservative default to cloud"
            recommendation_level="none"
            ;;
    esac
    
    # Combine classification and recommendation
    cat <<EOF
{
  "classification": $(echo "$classification_json" | jq -c .),
  "provider_status": "$provider_status",
  "outage_minutes": $outage_minutes,
  "recommendation": {
    "use_local": $should_recommend_local,
    "level": "$recommendation_level",
    "message": "$recommendation_text"
  }
}
EOF
}

#######################################
# Main command dispatcher
#######################################
main() {
    local command="${1:-}"
    
    case "$command" in
        classify)
            if [ $# -lt 2 ]; then
                echo "Usage: $0 classify \"user message\"" >&2
                exit 1
            fi
            classify_workload "$2"
            ;;
        recommend)
            if [ $# -lt 3 ]; then
                echo "Usage: $0 recommend \"user message\" \"provider_status\" [outage_minutes]" >&2
                exit 1
            fi
            generate_recommendation "$2" "$3" "${4:-0}"
            ;;
        version)
            echo "ollama-routing-policy.sh version $VERSION"
            ;;
        help|--help|-h)
            cat <<EOF
ollama-routing-policy.sh - Workload classification for Ollama routing

USAGE:
    $0 classify "user message"
        Classify a workload into LOCAL_SAFE, CLOUD_PREFERRED, CLOUD_REQUIRED, or FALLBACK_ONLY
    
    $0 recommend "user message" "provider_status" [outage_minutes]
        Generate routing recommendation based on classification and provider health
        provider_status: healthy | degraded | unavailable
        outage_minutes: optional, defaults to 0
    
    $0 version
        Show version
    
    $0 help
        Show this help

EXAMPLES:
    # Classify a summarization task
    $0 classify "Summarize the git log for the last week"
    
    # Get recommendation during cloud outage
    $0 recommend "Summarize changelog" "degraded" 7
    
    # Check security-sensitive operation
    $0 classify "Review authentication logic for vulnerabilities"

OUTPUT:
    JSON object with classification/recommendation details

EXIT CODES:
    0 - Success
    1 - Invalid arguments
    2 - Classification failed

SEE ALSO:
    docs/OLLAMA_ROUTING_STRATEGY.md
    docs/LOCAL_INFERENCE_POLICY.md
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
