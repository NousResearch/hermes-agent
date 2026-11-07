#!/usr/bin/env python3
"""
zotero_setup.py — Bootstrap the hermes-agent workspace inside Zotero.

Creates (idempotently):
  hermes-agent/
  ├── Books/
  ├── Papers/
  ├── Notes/
  └── Reading List/

Usage:
  python zotero_setup.py              # create/verify collection tree
  python zotero_setup.py --show       # print keys without modifying anything
  python zotero_setup.py --add-collection "Philosophy"
  python zotero_setup.py --add-collection "Philosophy" --parent hermes-agent
"""

import argparse
import json
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")


API_BASE = "https://api.zotero.org"
HEADERS = {}

ROOT_NAME = "hermes-agent"
DEFAULT_CHILDREN = ["Books", "Papers", "Notes", "Reading List"]


def get_env() -> tuple[str, str]:
    api_key = os.environ.get("ZOTERO_API_KEY", "")
    user_id = os.environ.get("ZOTERO_USER_ID", "")
    if not api_key or not user_id:
        sys.exit(
            "Set ZOTERO_API_KEY and ZOTERO_USER_ID environment variables.\n"
            "  Get them at: https://www.zotero.org/settings/keys"
        )
    return api_key, user_id


def build_headers(api_key: str) -> dict:
    return {
        "Zotero-API-Version": "3",
        "Zotero-API-Key": api_key,
        "Content-Type": "application/json",
    }


def get_all_collections(user_id: str) -> list[dict]:
    """Fetch all collections, handling pagination."""
    url = f"{API_BASE}/users/{user_id}/collections"
    collections = []
    start = 0
    while True:
        resp = requests.get(url, headers=HEADERS, params={"limit": 100, "start": start})
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        collections.extend(batch)
        total = int(resp.headers.get("Total-Results", len(batch)))
        start += len(batch)
        if start >= total:
            break
    return collections


def find_collection_by_name(collections: list[dict], name: str, parent_key: str | None = None) -> dict | None:
    """Find a collection matching name (case-insensitive) and optional parent."""
    for col in collections:
        data = col.get("data", {})
        col_name = data.get("name", "")
        col_parent = data.get("parentCollection", False)
        name_match = col_name.lower() == name.lower()
        parent_match = (
            (parent_key is None) or
            (parent_key is False and not col_parent) or
            (col_parent == parent_key)
        )
        if name_match and parent_match:
            return col
    return None


def create_collection(user_id: str, name: str, parent_key: str | None = None) -> dict:
    """Create a collection and return the created object."""
    url = f"{API_BASE}/users/{user_id}/collections"
    payload = [{"name": name}]
    if parent_key:
        payload[0]["parentCollection"] = parent_key
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    result = resp.json()
    created = result.get("successful", {}).get("0", {})
    if not created:
        sys.exit(f"Failed to create collection '{name}': {resp.text}")
    return created


def ensure_collection(
    user_id: str,
    collections: list[dict],
    name: str,
    parent_key: str | None = None,
    dry_run: bool = False,
) -> tuple[str, bool]:
    """Return (key, was_created). Creates if missing."""
    existing = find_collection_by_name(collections, name, parent_key=parent_key or False)
    if existing:
        return existing["data"]["key"], False

    if dry_run:
        return f"[would create '{name}']", False

    print(f"  Creating collection: {name}")
    created = create_collection(user_id, name, parent_key)
    key = created.get("data", {}).get("key") or created.get("key", "")
    # Append to local list so child lookups work in same run
    collections.append(created)
    return key, True


def print_tree(collections: list[dict], root_key: str, indent: int = 0) -> None:
    prefix = "  " * indent
    root = next((c for c in collections if c["data"]["key"] == root_key), None)
    if not root:
        return
    print(f"{prefix}├── {root['data']['name']}  [{root_key}]")
    children = [
        c for c in collections
        if c["data"].get("parentCollection") == root_key
    ]
    for child in sorted(children, key=lambda c: c["data"]["name"]):
        child_key = child["data"]["key"]
        print(f"{prefix}│   ├── {child['data']['name']}  [{child_key}]")


def cmd_show(user_id: str) -> None:
    collections = get_all_collections(user_id)
    root = find_collection_by_name(collections, ROOT_NAME, parent_key=False)
    if not root:
        print(f"'{ROOT_NAME}' collection not found. Run without --show to create it.")
        return
    root_key = root["data"]["key"]
    print(f"\nhermes-agent collection tree:")
    print_tree(collections, root_key)
    print()

    # Print env-style key map
    print("Collection keys (for use with other scripts):")
    print(f"  HERMES_KEY={root_key}")
    children = [
        c for c in collections
        if c["data"].get("parentCollection") == root_key
    ]
    for child in sorted(children, key=lambda c: c["data"]["name"]):
        name_env = child["data"]["name"].upper().replace(" ", "_")
        print(f"  {name_env}_KEY={child['data']['key']}")


def cmd_setup(user_id: str) -> None:
    print("Fetching existing collections...")
    collections = get_all_collections(user_id)

    print(f"\nEnsuring collection: {ROOT_NAME}")
    root_key, root_created = ensure_collection(user_id, collections, ROOT_NAME, parent_key=None)
    if not root_created:
        print(f"  Already exists  [{root_key}]")
    else:
        print(f"  Created  [{root_key}]")

    child_keys: dict[str, str] = {}
    for child_name in DEFAULT_CHILDREN:
        key, created = ensure_collection(user_id, collections, child_name, parent_key=root_key)
        child_keys[child_name] = key
        if not created:
            print(f"  {child_name}: already exists  [{key}]")
        else:
            print(f"  {child_name}: created  [{key}]")

    print("\n✓ hermes-agent workspace ready\n")
    print(f"{'hermes-agent':20s}  {root_key}")
    for name, key in child_keys.items():
        print(f"  {name:18s}  {key}")
    print()


def cmd_add_collection(user_id: str, name: str, parent_name: str) -> None:
    collections = get_all_collections(user_id)

    # Resolve parent
    if parent_name.lower() in ("hermes-agent", "root", "hermes"):
        parent = find_collection_by_name(collections, ROOT_NAME, parent_key=False)
        if not parent:
            sys.exit(f"'{ROOT_NAME}' collection not found. Run setup first.")
        parent_key = parent["data"]["key"]
    else:
        parent = find_collection_by_name(collections, parent_name)
        if not parent:
            sys.exit(f"Parent collection '{parent_name}' not found.")
        parent_key = parent["data"]["key"]

    key, created = ensure_collection(user_id, collections, name, parent_key=parent_key)
    if created:
        print(f"Created '{name}' under '{parent_name}'  [{key}]")
    else:
        print(f"'{name}' already exists under '{parent_name}'  [{key}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap hermes-agent Zotero workspace")
    parser.add_argument("--show", action="store_true", help="Print collection keys without modifying")
    parser.add_argument("--add-collection", metavar="NAME", help="Add a new sub-collection")
    parser.add_argument("--parent", default="hermes-agent", metavar="NAME",
                        help="Parent for --add-collection (default: hermes-agent)")
    args = parser.parse_args()

    api_key, user_id = get_env()
    global HEADERS
    HEADERS = build_headers(api_key)

    if args.show:
        cmd_show(user_id)
    elif args.add_collection:
        cmd_add_collection(user_id, args.add_collection, args.parent)
    else:
        cmd_setup(user_id)


if __name__ == "__main__":
    main()
