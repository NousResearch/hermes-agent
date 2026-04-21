"""Test 2: Patchright Grok adapter - OpenAI-compatible interface.

Run: cd ~/Projects/hermes-agent && python3 tests/grok/test_adapter.py

Tests:
  - PatchrightGrokClient creates with correct interface
  - client.chat.completions.create() returns OpenAI-compatible response
  - response.choices[0].message.content has Grok's text
  - response.choices[0].message.role is "assistant"
  - response.choices[0].finish_reason is "stop"
  - response.model matches requested model
  - response.usage has token counts
  - Client close works cleanly
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PASS = 0
FAIL = 0

def check(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")

print("=== Test 2: Adapter Interface ===\n")

# Import
print("[1] Import...")
from agent.patchright_client import PatchrightGrokClient
check("Import successful", True)

# Create client
print("[2] Create client...")
client = PatchrightGrokClient("grok-auto")
check("Client created", client is not None)
check("Has chat.completions", hasattr(client.chat, "completions"))
check("Has create method", hasattr(client.chat.completions, "create"))
check("Has api_key", hasattr(client, "api_key"))
check("Has base_url", hasattr(client, "base_url"))

# Send message
print("[3] Send message (this opens browser, wait for response)...")
start = time.time()
response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what is 2+2? answer with just the number"},
    ],
    model="grok-auto",
)
elapsed = time.time() - start
print(f"  Response received in {elapsed:.1f}s")

# Validate response structure
print("[4] Validate response...")
check("Has choices", hasattr(response, "choices") and len(response.choices) > 0)
check("Has message", hasattr(response.choices[0], "message"))
check("Has content", response.choices[0].message.content is not None)
check("Content not empty", len(response.choices[0].message.content) > 0)
check("Role is assistant", response.choices[0].message.role == "assistant")
check("Finish reason is stop", response.choices[0].finish_reason == "stop")
check("Model matches", response.model == "grok-auto")
check("Has usage", hasattr(response, "usage"))
check("Has prompt_tokens", hasattr(response.usage, "prompt_tokens"))
check("Has completion_tokens", hasattr(response.usage, "completion_tokens"))

print(f"\n  Content: '{response.choices[0].message.content}'")
print(f"  Model: {response.model}")
print(f"  Usage: prompt={response.usage.prompt_tokens} completion={response.usage.completion_tokens}")

# Second message (test conversation works)
print("\n[5] Second message...")
start = time.time()
response2 = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "now multiply that by 3"},
    ],
    model="grok-auto",
)
elapsed = time.time() - start
print(f"  Response received in {elapsed:.1f}s")
check("Second response has content", response2.choices[0].message.content is not None)
print(f"  Content: '{response2.choices[0].message.content}'")

# Close
print("\n[6] Close...")
client.close()
check("Client closed", True)

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL > 0 else 0)
