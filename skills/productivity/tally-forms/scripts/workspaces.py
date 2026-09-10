#!/usr/bin/env python3
"""Tally API — Workspaces endpoints.

Standalone script demonstrating all workspace operations.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

# ── List workspaces ─────────────────────────────────────────
def list_workspaces():
    r = requests.get(f"{BASE}/workspaces", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Get workspace ───────────────────────────────────────────
def get_workspace(workspace_id):
    r = requests.get(f"{BASE}/workspaces/{workspace_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Create workspace ────────────────────────────────────────
def create_workspace(name):
    r = requests.post(f"{BASE}/workspaces", headers=HEADERS, json={"name": name})
    r.raise_for_status()
    return r.json()

# ── Update workspace ────────────────────────────────────────
def update_workspace(workspace_id, name):
    r = requests.patch(f"{BASE}/workspaces/{workspace_id}", headers=HEADERS, json={"name": name})
    r.raise_for_status()
    return r.json()

# ── Delete workspace ────────────────────────────────────────
def delete_workspace(workspace_id):
    r = requests.delete(f"{BASE}/workspaces/{workspace_id}", headers=HEADERS)
    r.raise_for_status()

# ── Demo ────────────────────────────────────────────────────
if __name__ == "__main__":
    data = list_workspaces()
    for w in data.get("items", []):
        print(f"  {w['id']}  {w['name']}")
