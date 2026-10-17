#!/bin/bash
# Check Ollama is running and bge-m3:567m is available
set -e

if ! curl -s --max-time 5 http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama not running"
    exit 1
fi

if ollama list 2>/dev/null | grep -q "bge-m3:567m"; then
    echo "OK"
else
    echo "MISSING"
fi
