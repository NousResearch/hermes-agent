#!/bin/bash
# DuckDuckGo Search Helper Script
# Wrapper around ddgs CLI with sensible defaults
# Usage: ./duckduckgo.sh <query> [max_results]

set -e

QUERY="$1"
MAX_RESULTS="${2:-5}"

if [ -z "$QUERY" ]; then
    echo "Usage: $0 <query> [max_results]"
    echo ""
    echo "Examples:"
    echo "  $0 'python async programming' 5"
    echo "  $0 'latest AI news' 10"
    echo ""
    echo "Requires: pip install ddgs"
    exit 1
fi

# Check if ddgs is available
if ! command -v ddgs &> /dev/null; then
    echo "Error: ddgs not found. Install with: pip install ddgs"
    exit 1
fi

ddgs text -q "$QUERY" -m "$MAX_RESULTS"
