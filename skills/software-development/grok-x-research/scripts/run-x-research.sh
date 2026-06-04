#!/usr/bin/env bash
# Helper for grok-x-research skill.
# Usage: bash ${HERMES_SKILL_DIR}/scripts/run-x-research.sh "your research query here" [num_results]

set -euo pipefail

QUERY="${1:-latest discussion around Hermes Agent and Grok}"
NUM="${2:-5}"

echo "=== Grok X Research Session ==="
echo "Query: $QUERY"
echo "Results: $NUM"
echo

# This is meant to be called from within a Hermes session on the Grok provider.
# The skill will use this as a template for the agent to invoke terminal tool
# with actual Hermes X search capabilities.

echo "In Hermes on Grok provider, the agent should run something like:"
echo "hermes chat -q \"Using grok-x-research: $QUERY. Return top $NUM results with citations and synthesis.\""
echo
echo "For cron/automation, combine with 'hermes cron add' using the skill guidance."
