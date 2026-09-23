#!/bin/bash
# Usage: ./searxng.sh <query> [max_results] [engines]
# Example: ./searxng.sh "python async" 10 "google,bing"

QUERY="${1:-}"
MAX="${2:-5}"
ENGINES="${3:-google,bing}"

if [ -z "$SEARXNG_URL" ]; then
    echo "Error: SEARXNG_URL is not set"
    exit 1
fi

if [ -z "$QUERY" ]; then
    echo "Usage: $0 <query> [max_results] [engines]"
    exit 1
fi

ENCODED_QUERY=$(echo "$QUERY" | sed 's/ /+/g')

curl -s --max-time 10 \
    "${SEARXNG_URL}/search?q=${ENCODED_QUERY}&format=json&limit=${MAX}&engines=${ENGINES}"
