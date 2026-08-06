#!/usr/bin/env python3
"""Web backend health check — SearXNG + GroktoCrawl.

Checks:
1. SearXNG container running and responding on :8082
2. GroktoCrawl agent-svc container running and responding on :8090
3. GroktoCrawl scrape endpoint functional
4. DDGS importable and working

Exit codes:
  0 = check completed (healthy or degraded; findings are emitted as output)

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


def _searxng_base_url() -> str:
    """Resolve the SearXNG base URL the same way the Hermes provider does.

    Priority: process env (SEARXNG_URL), then Hermes config-aware env lookup
    (~/.hermes/.env), matching agent/web_search_provider._searxng_url().
    """
    url = os.environ.get("SEARXNG_URL", "")
    if not url:
        try:
            from hermes_cli.config import get_env_value
            url = get_env_value("SEARXNG_URL") or ""
        except Exception:
            pass
    return url.strip().rstrip("/")


def check_searxng():
    """Check the effective SearXNG JSON search boundary."""
    if _DRY_RUN:
        return True, "dry-run: probes skipped"
    base_url = _searxng_base_url()
    if not base_url:
        return False, "SEARXNG_URL is not configured"
    # The provider calls /search?q=...&format=json&pageno=1. Verify JSON
    # returns 200 and parses so a config regression (e.g. HTML-only
    # container) is caught the same way Hermes would hit it.
    status, body, error = _http_probe(
        f"{base_url}/search?q=health+check&format=json", timeout=5
    )
    if status != 200:
        return False, f"API status {status}" if status is not None else (error or "API not responding")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return False, "search returned non-JSON body (JSON format not enabled?)"
    if not isinstance(data, dict) or "results" not in data:
        return False, "search JSON missing results key"
    return True, f"JSON search returned {len(data['results'])} results"


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
            if isinstance(v, dict) and v.get("status") == "down"
        ]
        if down_services:
            return False, f"degraded: {','.join(down_services)}"
        if status != "ok":
            return False, f"health status {status}"
        return True, f"healthy ({status})"
    except Exception:
        return False, "invalid health JSON"


_DDGS_PROBE = r'''import json
try:
    from ddgs import DDGS
    with DDGS(timeout=10) as client:
        results = list(client.text("Hermes Agent", max_results=1))
    if results:
        print(json.dumps({"ok": True, "count": len(results)}))
    else:
        print(json.dumps({"ok": False, "kind": "empty", "message": "search returned no results"}))
except ModuleNotFoundError as exc:
    print(json.dumps({"ok": False, "kind": "missing_dependency", "message": str(exc)}))
except Exception as exc:
    print(json.dumps({"ok": False, "kind": "search_error", "message": str(exc)}))
'''


def check_ddgs():
    """Check DDGS through a bounded subprocess and return safe diagnostics."""
    if _DRY_RUN:
        return True, "dry-run: probes skipped"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _DDGS_PROBE],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "search timed out after 15s"
    except OSError as exc:
        return False, f"probe could not start: {type(exc).__name__}: {exc}"

    output = result.stdout.strip()
    try:
        envelope = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        stderr = " ".join(result.stderr.strip().splitlines())
        detail = stderr or output or f"worker exited {result.returncode} without output"
        return False, f"probe protocol error: {detail[:500]}"

    if envelope.get("ok"):
        return True, f"working ({envelope.get('count', 0)} result)"

    kind = envelope.get("kind", "unknown_error")
    message = str(envelope.get("message") or "no detail")
    if kind == "missing_dependency":
        return False, (
            "dependency unavailable: ddgs is not installed in "
            f"{sys.executable}; install with `{sys.executable} -m pip install ddgs`"
        )
    if kind == "empty":
        return False, message
    return False, f"search error: {message[:500]}"


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
    else:
        status_emoji = "🔴"
    
    print(f"{status_emoji} Web Backend Health [{ts}]")
    for p in parts:
        print(f"  {p}")
    
    if not search_ok:
        print("  CRITICAL: All search backends down")
    if not extract_ok:
        print("  CRITICAL: Extract backend (GroktoCrawl) down — extract will fall back to Tavily")
    
    # A detected dependency outage is a successful watchdog execution.  The
    # alert is the result; reserve non-zero exits for unexpected script faults.
    sys.exit(0)


if __name__ == '__main__':
    main()
