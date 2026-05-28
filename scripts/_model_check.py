#!/usr/bin/env python3
"""Check which models are available through each provider."""
import sys, os, json, urllib.request

sys.path.insert(0, '/Users/lumenhubai/.hermes/hermes-agent')
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv()

results = {}

# ─── OpenRouter ────────────────────────────────────────────────────────
print("Checking OpenRouter for Ring models...")
try:
    req = urllib.request.Request(
        'https://openrouter.ai/api/v1/models',
        headers={'Authorization': f'Bearer {os.environ["OPENROUTER_API_KEY"]}'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        all_models = [m['id'] for m in data.get('data', [])]
        ring_models = [m for m in all_models if 'ring' in m.lower()]
        deepseek_models = [m for m in all_models if 'deepseek' in m.lower()]
        results['openrouter_ring'] = ring_models
        results['openrouter_deepseek'] = deepseek_models
        print(f"  Ring models: {ring_models}")
        print(f"  DeepSeek models: {deepseek_models}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ─── Anthropic Claude ──────────────────────────────────────────────────
print("\nChecking Anthropic for Claude models...")
try:
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/models',
        headers={
            'x-api-key': os.environ['ANTHROPIC_API_KEY'],
            'anthropic-version': '2023-06-01',
        }
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        claude_models = [m['id'] for m in data.get('data', []) if 'sonnet' in m['id'].lower() or 'opus' in m['id'].lower()]
        results['anthropic'] = claude_models
        print(f"  Claude models: {claude_models}")
except Exception as e:
    print(f"  ❌ Error: {e}")

# ─── DeepSeek Direct ──────────────────────────────────────────────────
print("\nChecking DeepSeek API directly...")
try:
    req = urllib.request.Request(
        'https://api.deepseek.com/v1/models',
        headers={'Authorization': f'Bearer {os.environ["DEEPSEEK_API_KEY"]}'}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        ds_models = [m['id'] for m in data.get('data', [])]
        results['deepseek'] = ds_models
        print(f"  DeepSeek models: {ds_models}")
except Exception as e:
    print(f"  ❌ Error: {e}")

print(f"\nAll confirmed: {sum(len(v) for v in results.values())} model endpoints reachable")

with open('/Users/lumenhubai/.hermes/hermes-agent/scripts/_model_check.json', 'w') as f:
    json.dump(results, f, indent=2)