#!/usr/bin/env python3
"""Send auto_trim.py + test_auto_trim.py to Ollama qwen3-coder:30b for board review."""
import json, sys, textwrap
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3-coder:30b-a3b-q4_k_M"

auto_trim = Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/auto_trim.py").read_text()
test_auto_trim = Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/test_auto_trim.py").read_text()

# Read existing board reviews to avoid repeating findings
reviews = json.loads(Path("/Users/lumenhubai/.hermes/hermes-agent/scripts/_board_reviews.json").read_text())

prompt = f"""You are performing a rigorous code review of the Hermes Agent's context trimming engine.

The code is at:
- auto_trim.py (860 lines) — the core engine
- test_auto_trim.py (619 lines, 44 tests, all currently passing)

PREVIOUS REVIEWS ALREADY ADDRESSED:
1. Inconsistent return structures → FIXED via _response_base() helper
2. DRY_RUN block removal bug → FIXED: blocks stay in remaining during dry-run
3. Format string crash in compress_block() → FIXED: .replace() instead of .format()
4. Float priority silently dropped → FIXED: isinstance(pri, float) branch added
5. Missing "reason" key in active-trim path → FIXED
6. Dead TRIM_THRESHOLD env var in main() → FIXED: now uses TRIM_THRESHOLD_TOKENS
7. MIN_BLOCKS_KEPT=3 floor → FIXED with lookahead in Phase 1
8. Pause logic inverted → FIXED
9. Response dict contract unified → FIXED via _response_base()

Your task: Find ANY remaining bugs, edge cases, logic errors, test gaps, or improvements.
Focus especially on things the previous reviewers might have missed. Be specific with line numbers.

--- AUTO_TRIM.PY ---
{auto_trim}

--- TEST_AUTO_TRIM.PY ---
{test_auto_trim}

Return findings in this format:
### Finding N [CRITICAL|HIGH|MEDIUM|LOW|INFO]
Description, line numbers, suggested fix.
"""

payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"num_predict": 4096, "temperature": 0.1},
}

try:
    req = Request(f"{OLLAMA_HOST}/api/generate", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=180)
    data = json.loads(resp.read())
    print(data.get("response", "NO RESPONSE"))
except URLError as e:
    print(f"OLLAMA ERROR: {e}", file=sys.stderr)
    sys.exit(1)