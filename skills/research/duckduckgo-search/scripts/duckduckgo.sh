#!/bin/bash
# DuckDuckGo Search Helper Script
# Fallback for when ddgs library is unavailable
# Usage: ./duckduckgo.sh [text|images|news|videos] <query> [limit]

set -e

MODE="${1:-text}"
QUERY="$2"
LIMIT="${3:-5}"

if [ -z "$QUERY" ]; then
    echo "Usage: $0 [text|images|news|videos] <query> [limit]"
    echo "Examples:"
    echo "  $0 text 'python async' 5"
    echo "  $0 images 'cat' 10"
    exit 1
fi

# URL encode query
ENCODED_QUERY=$(echo "$QUERY" | sed 's/ /+/g' | sed 's/&/%26/g' | sed 's/=/%3D/g')

case "$MODE" in
    text|images|news|videos)
        curl -s "https://api.duckduckgo.com/?q=${ENCODED_QUERY}&format=json&limit=${LIMIT}"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: text, images, news, videos"
        exit 1
        ;;
esac
