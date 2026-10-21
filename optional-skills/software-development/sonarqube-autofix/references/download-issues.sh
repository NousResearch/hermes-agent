#!/usr/bin/env bash
set -euo pipefail

# Download SonarQube issues with automatic edition detection
# ⚠️ SECURITY: This script uses environment variables directly in bash.
# NEVER inspect or print the value of $SONAR_TOKEN or other token variables.
# Bash expands variables automatically - you do not need (and must not) read their values.

PROJECT_NAME="${1:-}"
ISSUES_CSV="${2:-}"
NEW_CODE_ONLY="${3:-false}"

error_exit() {
  echo "[ERRO] $1" >&2
  exit 1
}

[ -n "$PROJECT_NAME" ] || error_exit "PROJECT_NAME not provided."

# Detect edition and build dynamic URL
if [ -n "${SONARQUBE_CUSTOM_URL:-}" ]; then
  SONAR_BASE_URL="$SONARQUBE_CUSTOM_URL"
  SONAR_TOKEN="${SONARQUBE_CUSTOM_TOKEN:-}"
  SONAR_EDITION="${SONARQUBE_CUSTOM_EDITION:-open}"
elif [ -n "${SONARQUBE_ENTERPRISE_TOKEN:-}" ]; then
  [ -n "${SONARQUBE_ENTERPRISE_URL:-}" ] || error_exit "SONARQUBE_ENTERPRISE_URL not configured."
  SONAR_BASE_URL="$SONARQUBE_ENTERPRISE_URL"
  SONAR_TOKEN="$SONARQUBE_ENTERPRISE_TOKEN"
  SONAR_EDITION="enterprise"
elif [ -n "${SONARQUBE_OPEN_TOKEN:-}" ] || [ -n "${SONAR_TK:-}" ]; then
  SONAR_BASE_URL="${SONARQUBE_OPEN_URL:-https://sonarcloud.io}"
  SONAR_TOKEN="${SONARQUBE_OPEN_TOKEN:-$SONAR_TK}"
  SONAR_EDITION="open"
else
  error_exit "No SonarQube configuration found. Set SONARQUBE_CUSTOM_URL, or SONARQUBE_ENTERPRISE_TOKEN + SONARQUBE_ENTERPRISE_URL, or SONARQUBE_OPEN_TOKEN/SONAR_TK (defaults to https://sonarcloud.io)."
fi

# Detect branch for Enterprise/Custom enterprise editions
if [ "$SONAR_EDITION" = "enterprise" ]; then
  CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    BRANCH_PARAM="&branch=$CURRENT_BRANCH"
  else
    BRANCH_PARAM=""
  fi
else
  BRANCH_PARAM=""
fi

# Build API URL
if [ -n "$ISSUES_CSV" ]; then
  # Download specific issues by CSV
  IFS=',' read -ra ISSUE_IDS <<< "$ISSUES_CSV"
  ISSUES_PARAM=""
  for id in "${ISSUE_IDS[@]}"; do
    ISSUES_PARAM="${ISSUES_PARAM}&issues=${id}"
  done
  API_URL="${SONAR_BASE_URL}/api/issues/search?resolved=false&components=${PROJECT_NAME}${ISSUES_PARAM}${BRANCH_PARAM}"
else
  # Download all unresolved issues
  API_URL="${SONAR_BASE_URL}/api/issues/search?resolved=false&components=${PROJECT_NAME}"

  if [ "$NEW_CODE_ONLY" = "true" ]; then
    API_URL="${API_URL}&inNewCodePeriod=true"
  fi

  API_URL="${API_URL}${BRANCH_PARAM}"
fi

# Download issues
AUTH_HEADER=""
if [ -n "$SONAR_TOKEN" ]; then
  AUTH_HEADER="Authorization: Bearer $SONAR_TOKEN"
fi

if [ -n "$AUTH_HEADER" ]; then
  curl -fs "$API_URL" -H "$AUTH_HEADER"
else
  curl -fs "$API_URL"
fi
