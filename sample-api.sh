#!/usr/bin/env bash
# Sample OpenAI-compatible API calls to Hermes Agent (no auth)

echo "=== Non-streaming Chat Completion ==="
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hermes-agent",
    "messages": [
      {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
  }' | python3 -m json.tool

echo ""
echo "=== Streaming Chat Completion ==="
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hermes-agent",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hi in 3 words."}
    ]
  }'
echo ""
