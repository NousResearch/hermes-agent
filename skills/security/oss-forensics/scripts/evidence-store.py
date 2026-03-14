#!/usr/bin/env python3
import json
import argparse
import os
import datetime
import hashlib
from datetime import UTC

def get_now_iso():
    return datetime.datetime.now(UTC).isoformat() + "Z"

def calculate_sha256(content):
    return hashlib.sha256(content.encode()).hexdigest()

class EvidenceStore:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = {"evidence": [], "metadata": {"created_at": get_now_iso(), "last_updated": get_now_iso()}}
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.data = json.load(f)

    def save(self):
        self.data["metadata"]["last_updated"] = get_now_iso()
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

    def add_evidence(self, source, content, evidence_type, timestamp=None, actor=None, url=None, meta=None):
        evidence_id = f"EV-{len(self.data['evidence']) + 1:04d}"
        entry = {
            "id": evidence_id,
            "type": evidence_type,
            "source": source,
            "content_sha256": calculate_sha256(content),
            "content": content,
            "timestamp": timestamp or get_now_iso(),
            "collected_at": get_now_iso(),
            "actor": actor,
            "url": url,
            "metadata": meta or {}
        }
        self.data["evidence"].append(entry)
        self.save()
        return evidence_id

    def list_evidence(self):
        return self.data["evidence"]

def main():
    parser = argparse.ArgumentParser(description="OSS Forensics Evidence Store Manager")
    parser.add_argument("--store", default="evidence.json", help="Path to evidence JSON file")
    subparsers = parser.add_subparsers(dest="command")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add evidence to the store")
    add_parser.add_argument("--source", required=True)
    add_parser.add_argument("--content", required=True)
    add_parser.add_argument("--type", required=True, choices=["git_commit", "gh_api", "web_archive", "manual", "ioc", "analysis"])
    add_parser.add_argument("--timestamp")
    add_parser.add_argument("--actor")
    add_parser.add_argument("--url")

    # List command
    subparsers.add_parser("list", help="List all evidence")

    args = parser.parse_args()
    store = EvidenceStore(args.store)

    if args.command == "add":
        eid = store.add_evidence(args.source, args.content, args.type, args.timestamp, args.actor, args.url)
        print(f"Added evidence: {eid}")
    elif args.command == "list":
        for e in store.list_evidence():
            print(f"[{e['id']}] {e['type']} from {e['source']} - {e['timestamp']}")

if __name__ == "__main__":
    main()
