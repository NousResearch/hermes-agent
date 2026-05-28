#!/usr/bin/env python3
"""Send full state to DeepSeek via OpenRouter for gap analysis."""
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen, Request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4"
API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

auto_trim = Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/auto_trim.py").read_text()
test_auto_trim = Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/test_auto_trim.py").read_text()
context_orch = Path("/Users/lumenhubai/.hermes/scripts/context_orchestrator.py").read_text()[:8000]

gateway = Path("/Users/lumenhubai/.hermes/hermes-agent/gateway/run.py").read_text()
gateway_imports = gateway[2800:3100]
gateway_trim_section = gateway[8135:8160]

prompt = """You are performing a GAP ANALYSIS on the Hermes Agent context trimming system.
All 29 auto_trim tests PASS on both Mac and Linux. context_orchestrator.py imports successfully on Linux.
Find TEST GAPS and REMAINING INTEGRATION BUGS.

--- AUTO_TRIM.PY ---
AUTO_TRIM_PLACEHOLDER

--- CONTEXT_ORCHESTRATOR.PY ---
CONTEXT_ORCHESTRATOR_PLACEHOLDER

--- GATEWAY INTEGRATION (run.py) ---
IMPORTS:
GATEWAY_IMPORTS_PLACEHOLDER
TRIM_CALL:
GATEWAY_TRIM_PLACEHOLDER

--- TEST_AUTO_TRIM.PY (29 cases) ---
TEST_AUTO_TRIM_PLACEHOLDER

Focus areas:
1. Test gaps (untested scenarios)
2. Integration bugs between auto_trim and context_orchestrator
3. Production edge cases under real Telegram traffic
4. Concurrency / race conditions

Format each finding as:
### GAP N [CRITICAL|HIGH|MEDIUM|LOW]
Category: ...
Affected: file:line
Risk: ..."""

# Substitute file contents into prompt
prompt = prompt.replace("AUTO_TRIM_PLACEHOLDER", auto_trim)
prompt = prompt.replace("CONTEXT_ORCHESTRATOR_PLACEHOLDER", context_orch)
prompt = prompt.replace("GATEWAY_IMPORTS_PLACEHOLDER", gateway_imports)
prompt = prompt.replace("GATEWAY_TRIM_PLACEHOLDER", gateway_trim_section)
prompt = prompt.replace("TEST_AUTO_TRIM_PLACEHOLDER", test_auto_trim)

# Truncate to fit within model context
if len(prompt) > 30000:
    prompt = prompt[:30000] + "\n...[TRUNCATED]"

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a meticulous code reviewer. Find every test gap and bug. Be specific with line numbers."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 4096,
    "temperature": 0.1,
}

req = Request(OPENROUTER_URL, data=json.dumps(payload).encode(),
              headers={"Content-Type": "application/json",
                       "Authorization": f"Bearer {API_KEY}",
                       "HTTP-Referer": "https://hermes-agent.nousresearch.com"})
try:
    resp = urlopen(req, timeout=120)
    data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    print(content)
    # Save to file
    Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/_deepseek_gap_analysis.txt").write_text(content)
    print("\n\n[SAVED to _deepseek_gap_analysis.txt]")
except Exception as e:
    print(f"OPENROUTER ERROR: {e}", file=sys.stderr)