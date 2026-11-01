#!/usr/bin/env python3
"""Unauthenticated reachability check for the Cafe24 AI SPACE MCP server.

Verifies (1) OAuth discovery responds with a registration endpoint and
(2) the MCP endpoint itself answers. No credentials are sent or stored.
"""
import json
import sys
import urllib.request

BASE = "https://aih-proxy.cafe24.com"
OK = True

def check(name, url, expect_json_key=None, accepted_codes=(200, 400, 401, 405, 406)):
    global OK
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", "replace")
            code = r.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        code = e.code
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        OK = False
        return
    if code not in accepted_codes:
        print(f"[FAIL] {name}: HTTP {code}")
        OK = False
        return
    if expect_json_key:
        try:
            if expect_json_key not in json.loads(body):
                print(f"[FAIL] {name}: missing '{expect_json_key}'")
                OK = False
                return
        except ValueError:
            print(f"[FAIL] {name}: non-JSON response")
            OK = False
            return
    print(f"[PASS] {name}: HTTP {code}")

check("OAuth discovery", f"{BASE}/.well-known/oauth-authorization-server",
      expect_json_key="registration_endpoint", accepted_codes=(200,))
check("MCP endpoint", f"{BASE}/mcp")

print("RESULT:", "OK - proceed to OAuth (see references/headless-oauth.md)" if OK else "NOT REACHABLE")
sys.exit(0 if OK else 1)
