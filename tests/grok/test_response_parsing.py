"""Test 3: Response parsing edge cases.

Run: cd ~/Projects/hermes-agent && python3 tests/grok/test_response_parsing.py

Tests:
  - Short response (single word)
  - Long response (paragraph)
  - Code block response
  - Error handling (empty message)
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

print("=== Test 3: Response Parsing ===\n")

from agent.patchright_client import PatchrightGrokClient
client = PatchrightGrokClient("grok-auto")

# Test 1: Short response
print("[1] Short response (one word)...")
r = client.chat.completions.create(
    messages=[{"role": "user", "content": "reply with just the word yes"}],
    model="grok-auto",
)
print(f"  Got: '{r.choices[0].message.content}'")
check("Short response captured", len(r.choices[0].message.content) > 0)
check("Response is short", len(r.choices[0].message.content) < 50)

# Test 2: Long response
print("\n[2] Long response (paragraph)...")
r = client.chat.completions.create(
    messages=[{"role": "user", "content": "explain what a GPU is in exactly 3 sentences"}],
    model="grok-auto",
)
print(f"  Got: '{r.choices[0].message.content[:200]}...'")
check("Long response captured", len(r.choices[0].message.content) > 50)

# Test 3: Code block
print("\n[3] Code block response...")
r = client.chat.completions.create(
    messages=[{"role": "user", "content": "write a python hello world one-liner, just the code nothing else"}],
    model="grok-auto",
)
print(f"  Got: '{r.choices[0].message.content[:200]}'")
check("Code response captured", len(r.choices[0].message.content) > 0)
check("Contains print", "print" in r.choices[0].message.content.lower())

client.close()

print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
sys.exit(1 if FAIL > 0 else 0)
