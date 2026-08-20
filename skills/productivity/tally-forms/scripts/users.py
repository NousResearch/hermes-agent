#!/usr/bin/env python3
"""Tally API — Users endpoint.

Standalone script to fetch the authenticated user's profile.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

def get_me():
    r = requests.get(f"{BASE}/users/me", headers=HEADERS)
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    me = get_me()
    print(json.dumps(me, indent=2))
