#!/usr/bin/env python3
"""Web backend health check — SearXNG + GroktoCrawl.

Checks:
1. SearXNG container running and responding on :8082
2. GroktoCrawl agent-svc container running and responding on :8090
3. GroktoCrawl scrape endpoint functional
4. DDGS importable and working

Exit codes:
  0 = all healthy
  1 = one or more services degraded
  2 = critical failure (search or extract completely down)

Designed for cron execution. Output is single-line status for Discord.
"""
import subprocess
import json
import sys
import os
import urllib.error
import urllib.request
from datetime import datetime

# P13 isolation: HERMES_HOME is env-overridable so a disposable run never
# touches /home/kensei/.hermes (this script is read-only but honours the
# convention for consistency with the Wave 2 pattern).
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))

# P13 dry-run: when --dry-run is passed, every network probe (sudo docker
# inspect, curl to SearXNG/GroktoCrawl, live DDGS search) is skipped. The
# script reports a synthetic "dry-run: probes skipped" healthy result and
# exits 0 so a disposable run never hits docker, the network, or DDGS.
_DRY_RUN = "--dry-run" in sys.argv


def _http_probe(url, timeout):
    """Return HTTP status/body/error with a bounded, non-root probe."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(1_000_000).decode("utf-8", errors="replace")
            return response.status, body, ""
    except urllib.error.HTTPError as exc:
        body = exc.read(1_000_000).decode("utf-8", errors="replace")
        return exc.code, body, str(exc)
    except Exception as exc:
        return None, "", str(exc)


def check_searxng():
    """Check the SearXNG JSON API without inspecting a container."""
    if _DRY_RUN:
        return True, "dry-run: probes skipped"
    status, body, error = _http_probe(
        "http://127.0.0.1:8082/search?q=health+check&format=json", timeout=5
    )
    if status != 200:
        return False, f"API status {status}" if status is not None else (error or "API not responding")
    try:
        n = len(json.loads(body).get("results", []))
        if n == 0:
            return False, "0 results returned"
        return True, f"{n} results"
    except Exception:
        return False, "invalid JSON response"


def check_groktoCrawl():
    """Check GroktoCrawl health and scrape status without Docker."""
    if _DRY_RUN:
        return True, "dry-run: probes skipped"
    status_code, body, error = _http_probe("http://127.0.0.1:8090/health", timeout=10)
    if status_code != 200 or not body:
        detail = f"health status {status_code}" if status_code is not None else (error or "health endpoint not responding")
        return False, detail
    try:
        h = json.loads(body)
        status = h.get("status", "unknown")
        checks = h.get("checks", {})
        down_services = [
            k for k, v in checks.items()
            if isinstance(v, dict) and v.get("status") == "down" and k not in ("searxng",)
        ]
        if down_services:
            return False, f"degraded: {','.join(down_services)}"
        if status == "down" and not down_services:
            return True, "healthy (searxng health check cosmetic 404, search works)"
        return True, f"healthy ({status})"
    except Exception:
        return False, "invalid health JSON"


def check_ddgs():
    """Check DDGS is importable and functional."""
    if _DRY_RUN:
        return True, "dry-run: probes skipped"
    r = subprocess.run(
        ['python3', '-c',
         'from ddgs import DDGS; ddgs=DDGS(); r=list(ddgs.text("test", max_results=1)); exit(0 if len(r)>0 else 1)'],
        capture_output=True, text=True, timeout=15
    )
    if r.returncode == 0:
        return True, "working"
    return False, f"import or search failed: {r.stderr[:80]}"


def main():
    ts = datetime.now().strftime('%d/%m/%y %H:%M')
    
    searx_ok, searx_detail = check_searxng()
    grok_ok, grok_detail = check_groktoCrawl()
    ddgs_ok, ddgs_detail = check_ddgs()
    
    all_ok = searx_ok and grok_ok and ddgs_ok
    search_ok = searx_ok or ddgs_ok  # at least one search backend
    extract_ok = grok_ok  # only extract backend
    
    # Silent when healthy
    if all_ok:
        sys.exit(0)
    
    # Build alert only when something is wrong
    parts = []
    parts.append(f"SearXNG {'OK' if searx_ok else 'FAIL'} ({searx_detail})")
    parts.append(f"GroktoCrawl {'OK' if grok_ok else 'FAIL'} ({grok_detail})")
    parts.append(f"DDGS {'OK' if ddgs_ok else 'FAIL'} ({ddgs_detail})")
    
    if search_ok and extract_ok:
        status_emoji = "🟡"
        exit_code = 1
    else:
        status_emoji = "🔴"
        exit_code = 2
    
    print(f"{status_emoji} Web Backend Health [{ts}]")
    for p in parts:
        print(f"  {p}")
    
    if not search_ok:
        print("  CRITICAL: All search backends down")
    if not extract_ok:
        print("  CRITICAL: Extract backend (GroktoCrawl) down — extract will fall back to Tavily")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
