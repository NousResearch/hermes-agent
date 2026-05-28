#!/usr/bin/env python3
"""Board review v3 — shorter prompts to avoid timeouts."""
import sys, os, json
import requests

sys.path.insert(0, '/Users/lumenhubai/.hermes/hermes-agent')
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv()

# Read files
with open('/Users/lumenhubai/.hermes/hermes-agent/scripts/auto_trim.py') as f:
    code = f.read()
with open('/Users/lumenhubai/.hermes/hermes-agent/scripts/test_auto_trim.py') as f:
    tests = f.read()

# Shorter prompt — reference key areas by line number instead of embedding full code
REVIEW_PROMPT = """Review the auto_trim.py code at:
/Users/lumenhubai/.hermes/hermes-agent/scripts/auto_trim.py

And the test suite at:
/Users/lumenhubai/.hermes/hermes-agent/scripts/test_auto_trim.py

Key areas to review (recently refactored):
1. _response_base() helper at line ~340 — does it correctly default all 16 fields? Are any overrides missing defaults?
2. trim_context() has 4 return paths (empty, paused, below-threshold, active-trim) at lines ~399, ~406, ~435, ~544 — are they all consistent?
3. The DRY_RUN fix at line ~486 — blocks should be kept in remaining but not counted as deleted. Is this correct?
4. compress_block() at line ~291 — the .replace() fix for curly braces. Is it safe?
5. _block_priority() at line ~338 — float handling. Is int() cast sufficient?
6. main() threshold fix at line ~808 — uses TRIM_THRESHOLD_TOKENS. Is this correct?
7. Any test edge cases that are NOT covered? (particularly: parse_trigger_signal, handle_cli_pause_resume, main() e2e, validate_inputs edge cases)
8. Any remaining bugs, security issues, or inconsistencies?

Give specific line numbers and severity (CRITICAL/MEDIUM/LOW/INFO) for each finding. Return at least 5 findings."""

reviews = {}
headers_base = {'User-Agent': 'Hermes-AutoTrim-Review'}

def post_json(url, headers, payload, label):
    r = requests.post(url, json=payload, headers=headers, timeout=90)
    r.raise_for_status()
    return r.json()

# ─── Claude ────────────────────────────────────────────────────────────
print("📡 Claude ...", flush=True)
try:
    r = post_json(
        'https://api.anthropic.com/v1/messages',
        {**headers_base, 'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01'},
        {
            'model': 'claude-sonnet-4-6',
            'max_tokens': 8000,
            'temperature': 0.3,
            'system': 'You are an expert Python code reviewer at a top security firm.',
            'messages': [{'role': 'user', 'content': REVIEW_PROMPT}],
        },
        "Claude",
    )
    review = r['content'][0]['text']
    reviews['claude'] = review
    print(f"  ✅ Claude: {len(review)} chars")
except Exception as e:
    print(f"  ❌ Claude: {e}")

# ─── DeepSeek via OpenRouter ───────────────────────────────────────────
print("📡 DeepSeek ...", flush=True)
try:
    r = post_json(
        'https://openrouter.ai/api/v1/chat/completions',
        {**headers_base, 'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}', 'HTTP-Referer': 'https://github.com/nousresearch/hermes'},
        {
            'model': 'deepseek/deepseek-v4-flash',
            'messages': [
                {'role': 'system', 'content': 'You are an expert code reviewer.'},
                {'role': 'user', 'content': REVIEW_PROMPT},
            ],
            'max_tokens': 8000,
            'temperature': 0.3,
        },
        "DeepSeek",
    )
    review = r['choices'][0]['message']['content']
    reviews['deepseek'] = review
    print(f"  ✅ DeepSeek: {len(review)} chars")
except Exception as e:
    print(f"  ❌ DeepSeek: {e}")

# ─── Ring-2.6-1t via OpenRouter ────────────────────────────────────────
print("📡 Ring ...", flush=True)
try:
    r = post_json(
        'https://openrouter.ai/api/v1/chat/completions',
        {**headers_base, 'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}', 'HTTP-Referer': 'https://github.com/nousresearch/hermes'},
        {
            'model': 'inclusionai/ring-2.6-1t',
            'messages': [
                {'role': 'system', 'content': 'You are an expert code reviewer.'},
                {'role': 'user', 'content': REVIEW_PROMPT},
            ],
            'max_tokens': 8000,
            'temperature': 0.3,
        },
        "Ring",
    )
    review = r['choices'][0]['message']['content']
    reviews['ring'] = review
    print(f"  ✅ Ring: {len(review)} chars")
except Exception as e:
    print(f"  ❌ Ring: {e}")

# Save
with open('/Users/lumenhubai/.hermes/hermes-agent/scripts/_board_reviews.json', 'w') as f:
    json.dump(reviews, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"📋 {len(reviews)}/3 complete")
for k, v in reviews.items():
    print(f"  {k}: {len(v)} chars")
print("="*60)