#!/usr/bin/env python3
"""Tally API — Organizations endpoints.

Standalone script demonstrating organization user and invite operations.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

# ── List organization users ─────────────────────────────────
def list_users(org_id):
    r = requests.get(f"{BASE}/organizations/{org_id}/users", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Remove organization user ────────────────────────────────
def remove_user(org_id, user_id):
    r = requests.delete(f"{BASE}/organizations/{org_id}/users/{user_id}", headers=HEADERS)
    r.raise_for_status()

# ── List invites ────────────────────────────────────────────
def list_invites(org_id):
    r = requests.get(f"{BASE}/organizations/{org_id}/invites", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Create invite ───────────────────────────────────────────
def create_invite(org_id, email, role="MEMBER"):
    r = requests.post(
        f"{BASE}/organizations/{org_id}/invites",
        headers=HEADERS,
        json={"email": email, "role": role},
    )
    r.raise_for_status()
    return r.json()

# ── Cancel invite ───────────────────────────────────────────
def cancel_invite(org_id, invite_id):
    r = requests.delete(f"{BASE}/organizations/{org_id}/invites/{invite_id}", headers=HEADERS)
    r.raise_for_status()

# ── Demo ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python organizations.py <organization_id>")
        sys.exit(1)

    org_id = sys.argv[1]

    users = list_users(org_id)
    print(f"Users: {len(users.get('items', []))}")
    for u in users.get("items", []):
        print(f"  {u['email']}  role={u['role']}")

    invites = list_invites(org_id)
    print(f"Pending invites: {len(invites.get('items', []))}")
