#!/usr/bin/env python3
"""Provider API Reachability Check — tests each configured provider endpoint.

Silent when healthy (exit 0, empty stdout).
Outputs alert text when a provider is unreachable or responding slowly.
"""
import subprocess
import sys
import json
import time
from pathlib import Path

HERMES = Path("/home/kensei/.hermes")

# Providers to check (name, base_url, endpoint) — only actively configured providers
PROVIDERS = [
    ("ollama-cloud", "https://ollama.com/v1", "/models"),
    ("openrouter", "https://openrouter.ai/api/v1", "/models"),
    ("nvidia", "https://integrate.api.nvidia.com/v1", "/models"),
    ("opencode-zen", "https://opencode.ai/zen/v1", "/models"),
]

TIMEOUT_SECONDS = 15
SLOW_THRESHOLD_SECONDS = 10

def check_provider(name, base_url, endpoint):
    """Check if a provider endpoint is reachable and responsive."""
    url = f"{base_url}{endpoint}"
    start = time.time()
    
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "10", "--max-time", str(TIMEOUT_SECONDS),
             url],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS + 5
        )
        elapsed = time.time() - start
        status_code = result.stdout.strip()
        
        if status_code in ("000", ""):
            return "unreachable", elapsed, "Connection failed"
        elif status_code.startswith("4") or status_code.startswith("5"):
            return "error", elapsed, f"HTTP {status_code}"
        elif elapsed > SLOW_THRESHOLD_SECONDS:
            return "slow", elapsed, f"Response in {elapsed:.1f}s (threshold: {SLOW_THRESHOLD_SECONDS}s)"
        else:
            return "ok", elapsed, None
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - start, "Request timed out"
    except Exception as e:
        return "error", time.time() - start, str(e)[:100]

def main():
    alerts = []
    
    for name, base_url, endpoint in PROVIDERS:
        status, elapsed, detail = check_provider(name, base_url, endpoint)
        
        if status == "unreachable":
            alerts.append(f"**{name}** — unreachable ({detail})")
        elif status == "error":
            alerts.append(f"**{name}** — error ({detail})")
        elif status == "timeout":
            alerts.append(f"**{name}** — timed out after {elapsed:.1f}s")
        elif status == "slow":
            alerts.append(f"**{name}** — slow ({detail})")
    
    if not alerts:
        sys.exit(0)  # Silent when healthy
    
    print("**🟡 Provider Health Alert — DD/MM/YY HH:MM**\n")
    for alert in alerts:
        print(f"• {alert}")
    print(f"\n**Action:** Check provider status pages. If persistent, check API keys and rate limits.")
    sys.exit(0)

if __name__ == "__main__":
    main()
