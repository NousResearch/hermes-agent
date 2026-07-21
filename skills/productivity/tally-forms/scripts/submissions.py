#!/usr/bin/env python3
"""Tally API — Submissions endpoints.

Standalone script demonstrating all submissions operations.
Set TALLY_API_KEY environment variable before running.
"""

import os, requests, json

BASE = "https://api.tally.so"
HEADERS = {
    "Authorization": f"Bearer {os.environ['TALLY_API_KEY']}",
    "Content-Type": "application/json",
}

# ── List submissions ────────────────────────────────────────
def list_submissions(form_id, page=1, limit=50, start_date=None, end_date=None, after_id=None, filter_expr=None):
    params = {"page": page, "limit": limit}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if after_id:
        params["afterId"] = after_id
    if filter_expr:
        params["filter"] = filter_expr
    r = requests.get(f"{BASE}/forms/{form_id}/submissions", headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

# ── Get single submission ───────────────────────────────────
def get_submission(form_id, submission_id):
    r = requests.get(f"{BASE}/forms/{form_id}/submissions/{submission_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json()

# ── Delete submission ───────────────────────────────────────
def delete_submission(form_id, submission_id):
    r = requests.delete(f"{BASE}/forms/{form_id}/submissions/{submission_id}", headers=HEADERS)
    r.raise_for_status()

# ── Fetch all (cursor pagination) ──────────────────────────
def fetch_all_submissions(form_id):
    all_items = []
    after_id = None
    while True:
        data = list_submissions(form_id, limit=100, after_id=after_id)
        items = data.get("items", [])
        all_items.extend(items)
        if not data.get("hasMore"):
            break
        after_id = items[-1]["id"]
    return all_items

# ── Demo ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python submissions.py <form_id>")
        sys.exit(1)

    form_id = sys.argv[1]

    data = list_submissions(form_id, limit=5)
    print(f"Total submissions: {data.get('total', '?')}")
    for s in data.get("items", []):
        print(f"  {s['id']}  completed={s.get('isCompleted')}  {s.get('createdAt', '')[:10]}")
        for field in s.get("fields", []):
            print(f"    {field['label']}: {field['value']}")
