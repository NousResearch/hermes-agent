#!/usr/bin/env python3
"""Tally API — Webhooks endpoints.

Standalone script demonstrating all webhook operations.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

# ── List webhooks ───────────────────────────────────────────
def list_webhooks(form_id=None):
    params = {}
    if form_id:
        params["formId"] = form_id
    r = requests.get(f"{BASE}/webhooks", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

# ── Create webhook ──────────────────────────────────────────
def create_webhook(form_id, url, event_types=None):
    payload = {
        "formId": form_id,
        "url": url,
        "eventTypes": event_types or ["FORM_RESPONSE"],
    }
    r = requests.post(f"{BASE}/webhooks", headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

# ── Update webhook ──────────────────────────────────────────
def update_webhook(webhook_id, **kwargs):
    """Pass any of: url, isActive, eventTypes."""
    r = requests.patch(f"{BASE}/webhooks/{webhook_id}", headers=HEADERS, json=kwargs)
    r.raise_for_status()
    return r.json()

# ── Delete webhook ──────────────────────────────────────────
def delete_webhook(webhook_id):
    r = requests.delete(f"{BASE}/webhooks/{webhook_id}", headers=HEADERS)
    r.raise_for_status()

# ── List webhook events ─────────────────────────────────────
def list_events(webhook_id):
    r = requests.get(f"{BASE}/webhooks/{webhook_id}/events", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Retry webhook event ─────────────────────────────────────
def retry_event(webhook_id, event_id):
    r = requests.post(f"{BASE}/webhooks/{webhook_id}/events/{event_id}/retry", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Demo ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python webhooks.py <form_id> [webhook_url]")
        sys.exit(1)

    form_id = sys.argv[1]
    url = sys.argv[2] if len(sys.argv) > 2 else "https://example.com/webhook"

    # List existing
    data = list_webhooks(form_id)
    print(f"Existing webhooks: {len(data.get('items', []))}")

    # Create
    wh = create_webhook(form_id, url)
    print(f"Created webhook {wh['id']} -> {wh['url']}")

    # List events
    events = list_events(wh["id"])
    print(f"Events: {len(events.get('items', []))}")

    # Cleanup
    delete_webhook(wh["id"])
    print("Deleted.")
