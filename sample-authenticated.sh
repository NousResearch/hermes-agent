#!/usr/bin/env bash
# Sample OpenAI-compatible API calls to Hermes Agent (with Bearer auth)

API_KEY="NONPRIVATEKEYPROTECTENDPOINT"

echo "=== Non-streaming Chat Completion (Authenticated) ==="
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "hermes-agent",
    "messages": [
      {"role": "user", "content": "Say hello in exactly 5 words."}
    ]
  }' | python3 -m json.tool

echo ""
echo "=== Streaming Chat Completion (Authenticated) ==="
curl -s http://127.0.0.1:8642/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "hermes-agent",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Say hi in 3 words."}
    ]
  }'
echo ""
