#!/usr/bin/env python3
"""Ping healthchecks.io heartbeat monitor.

Used as --script for kensei-heartbeat cron job.
Gateway crash = no ping = healthchecks.io alerts after grace period.

Output (stdout) is injected into the cron prompt as context.
We emit a single status word so KENSEI's heartbeat reflects whether
external monitoring is healthy.
"""
import os
import sys
import urllib.request
import urllib.error

url = os.environ.get('HEALTHCHECKS_PING_URL')
if not url:
    print("uptime_status: not_configured")
    sys.exit(0)

try:
    with urllib.request.urlopen(url, timeout=10) as resp:
        if 200 <= resp.status < 300:
            print("uptime_status: ok")
        else:
            print(f"uptime_status: bad_response_{resp.status}")
except urllib.error.URLError as e:
    print(f"uptime_status: error_{type(e).__name__}")
except Exception as e:
    print(f"uptime_status: error_{type(e).__name__}")

# Always exit 0 — never break the heartbeat itself
sys.exit(0)
