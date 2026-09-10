#!/usr/bin/env python3
"""Tally API — Forms endpoints.

Standalone script demonstrating all forms operations.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, uuid, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

def _uuid():
    return str(uuid.uuid4())

# ── List forms ──────────────────────────────────────────────
def list_forms(page=1, limit=50, workspace_ids=None):
    params = {"page": page, "limit": limit}
    if workspace_ids:
        params["workspaceIds"] = workspace_ids
    r = requests.get(f"{BASE}/forms", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

# ── Get form ────────────────────────────────────────────────
def get_form(form_id):
    r = requests.get(f"{BASE}/forms/{form_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Create form ─────────────────────────────────────────────
def create_form(title, status="PUBLISHED", description=""):
    blocks = [
        {
            "uuid": _uuid(),
            "type": "FORM_TITLE",
            "groupUuid": _uuid(),
            "groupType": "FORM_TITLE",
            "payload": {"html": f"<h1>{title}</h1>"},
        }
    ]
    if description:
        blocks.append({
            "uuid": _uuid(),
            "type": "TEXT",
            "groupUuid": _uuid(),
            "groupType": "TEXT",
            "payload": {"html": f"<p>{description}</p>"},
        })
    r = requests.post(f"{BASE}/forms", headers=HEADERS, json={"status": status, "blocks": blocks})
    r.raise_for_status()
    return r.json()

# ── Update form ─────────────────────────────────────────────
def update_form(form_id, **kwargs):
    """Pass any combination of: name, status, blocks, settings."""
    r = requests.patch(f"{BASE}/forms/{form_id}", headers=HEADERS, json=kwargs)
    r.raise_for_status()
    return r.json()

# ── Delete form ─────────────────────────────────────────────
def delete_form(form_id):
    r = requests.delete(f"{BASE}/forms/{form_id}", headers=HEADERS)
    r.raise_for_status()

# ── Demo ────────────────────────────────────────────────────
if __name__ == "__main__":
    # List
    data = list_forms(limit=5)
    print(f"Found {data.get('total', len(data.get('items', [])))} forms")
    for f in data.get("items", []):
        print(f"  {f['id']}  {f['name']}  ({f['status']})")

    # Create
    form = create_form("API Test Form", description="Created via script")
    print(f"\nCreated: https://tally.so/r/{form['id']}")

    # Get
    detail = get_form(form["id"])
    print(f"Blocks: {len(detail.get('blocks', []))}")

    # Update
    updated = update_form(form["id"], name="Renamed Form")
    print(f"Renamed to: {updated['name']}")

    # Delete
    delete_form(form["id"])
    print("Deleted.")
