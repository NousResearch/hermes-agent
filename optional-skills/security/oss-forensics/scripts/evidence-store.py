#!/usr/bin/env python3
"""
OSS Forensics Evidence Store Manager
Manages a JSON-based evidence store for forensic investigations.

Commands:
  add      - Add a piece of evidence
  list     - List all evidence (optionally filter by type or actor)
  verify   - Re-check SHA-256 hashes for integrity
  query    - Search evidence by keyword
  export   - Export evidence as a Markdown table
  summary  - Print investigation statistics

Usage example:
  python3 evidence-store.py --store evidence.json add \
    --source "git fsck output" --content "dangling commit abc123" \
    --type git --actor "malicious-user" --url "https://github.com/owner/repo/commit/abc123"

  python3 evidence-store.py --store evidence.json list --type git
  python3 evidence-store.py --store evidence.json verify
  python3 evidence-store.py --store evidence.json export > evidence-table.md
"""

import json
import argparse
import os
import datetime
import hashlib
import sys

EVIDENCE_TYPES = [
    "git",           # Local git repository data (commits, reflog, fsck)
    "gh_api",        # GitHub REST API responses
    "gh_archive",    # GitHub Archive / BigQuery query results
    "web_archive",   # Wayback Machine snapshots
    "ioc",           # Indicator of Compromise (SHA, domain, IP, package name, etc.)
    "analysis",      # Derived analysis / cross-source correlation result
    "manual",        # Manually noted observation
    "vendor_report", # External security vendor report excerpt
]

VERIFICATION_STATES = ["unverified", "single_source", "multi_source_verified"]

IOC_TYPES = [
    "COMMIT_SHA", "FILE_PATH", "API_KEY", "SECRET", "IP_ADDRESS",
    "DOMAIN", "PACKAGE_NAME", "ACTOR_USERNAME", "MALICIOUS_URL",
    "WORKFLOW_FILE", "BRANCH_NAME", "TAG_NAME", "RELEASE_NAME", "OTHER",
]


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds") + "Z"


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class EvidenceStore:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = {
            "metadata": {
                "version": "2.0",
                "created_at": _now_iso(),
                "last_updated": _now_iso(),
                "investigation": "",
                "target_repo": "",
            },
            "evidence": [],
            "chain_of_custody": [],
        }
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading evidence store '{filepath}': {e}", file=sys.stderr)
                print("Hint: The file might be corrupted. Check for manual edits or syntax errors.", file=sys.stderr)
                sys.exit(1)

    def _save(self):
        self.data["metadata"]["last_updated"] = _now_iso()
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def _next_id(self) -> str:
        return f"EV-{len(self.data['evidence']) + 1:04d}"

    def add(
        self,
        source: str,
        content: str,
        evidence_type: str,
        actor: str = None,
        url: str = None,
        timestamp: str = None,
        ioc_type: str = None,
        verification: str = "unverified",
        notes: str = None,
    ) -> str:
        evidence_id = self._next_id()
        entry = {
            "id": evidence_id,
            "type": evidence_type,
            "source": source,
            "content": content,
            "content_sha256": _sha256(content),
            "actor": actor,
            "url": url,
            "event_timestamp": timestamp,
            "collected_at": _now_iso(),
            "ioc_type": ioc_type,
            "verification": verification,
            "notes": notes,
        }
        self.data["evidence"].append(entry)
        self.data["chain_of_custody"].append({
            "action": "add",
            "evidence_id": evidence_id,
            "timestamp": _now_iso(),
            "source": source,
        })
        self._save()
        return evidence_id

    def list_evidence(self, filter_type: str = None, filter_actor: str = None):
        results = self.data["evidence"]
        if filter_type:
            results = [e for e in results if e.get("type") == filter_type]
        if filter_actor:
            results = [e for e in results if e.get("actor") == filter_actor]
        return results

    def verify_integrity(self):
        """Re-compute SHA-256 for all entries and report mismatches."""
        issues = []
        for entry in self.data["evidence"]:
            expected = _sha256(entry["content"])
            stored = entry.get("content_sha256", "")
            if expected != stored:
                issues.append({
                    "id": entry["id"],
                    "stored_sha256": stored,
                    "computed_sha256": expected,
                })
        return issues

    def query(self, keyword: str):
        """Search for keyword in content, source, actor, or url."""
        keyword_lower = keyword.lower()
        return [
            e for e in self.data["evidence"]
            if keyword_lower in (e.get("content", "") or "").lower()
            or keyword_lower in (e.get("source", "") or "").lower()
            or keyword_lower in (e.get("actor", "") or "").lower()
            or keyword_lower in (e.get("url", "") or "").lower()
        ]

    def export_markdown(self) -> str:
        lines = [
            "# Evidence Registry",
            "",
            f"**Store**: `{self.filepath}`",
            f"**Last Updated**: {self.data['metadata'].get('last_updated', 'N/A')}",
            f"**Total Evidence Items**: {len(self.data['evidence'])}",
            "",
            "| ID | Type | Source | Actor | Verification | Event Timestamp | URL |",
            "|----|------|--------|-------|--------------|-----------------|-----|",
        ]
        for e in self.data["evidence"]:
            url = e.get("url") or ""
            url_display = f"[link]({url})" if url else ""
            lines.append(
                f"| {e['id']} | {e.get('type','')} | {e.get('source','')} "
                f"| {e.get('actor') or ''} | {e.get('verification','')} "
                f"| {e.get('event_timestamp') or ''} | {url_display} |"
            )
        lines.append("")
        lines.append("## Chain of Custody")
        lines.append("")
        lines.append("| Evidence ID | Action | Timestamp | Source |")
        lines.append("|-------------|--------|-----------|--------|")
        for c in self.data["chain_of_custody"]:
            lines.append(
                f"| {c.get('evidence_id','')} | {c.get('action','')} "
                f"| {c.get('timestamp','')} | {c.get('source','')} |"
            )
        return "\n".join(lines)

    def summary(self) -> dict:
        by_type = {}
        by_verification = {}
        actors = set()
        for e in self.data["evidence"]:
            t = e.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            v = e.get("verification", "unverified")
            by_verification[v] = by_verification.get(v, 0) + 1
            if e.get("actor"):
                actors.add(e["actor"])
        return {
            "total": len(self.data["evidence"]),
            "by_type": by_type,
            "by_verification": by_verification,
            "unique_actors": sorted(actors),
        }


def main():
    parser = argparse.ArgumentParser(
        description="OSS Forensics Evidence Store Manager v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--store", default="evidence.json", help="Path to evidence JSON file (default: evidence.json)")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- add ---
    add_p = subparsers.add_parser("add", help="Add a new evidence entry")
    add_p.add_argument("--source", required=True, help="Where this evidence came from (e.g. 'git fsck', 'GH API /commits')")
    add_p.add_argument("--content", required=True, help="The evidence content (commit SHA, API response excerpt, etc.)")
    add_p.add_argument("--type", required=True, choices=EVIDENCE_TYPES, dest="evidence_type", help="Evidence type")
    add_p.add_argument("--actor", help="GitHub handle or email of associated actor")
    add_p.add_argument("--url", help="URL to original source")
    add_p.add_argument("--timestamp", help="When the event occurred (ISO 8601)")
    add_p.add_argument("--ioc-type", choices=IOC_TYPES, help="IOC subtype (for --type ioc)")
    add_p.add_argument("--verification", choices=VERIFICATION_STATES, default="unverified")
    add_p.add_argument("--notes", help="Additional investigator notes")
    add_p.add_argument("--quiet", action="store_true", help="Suppress success message")

    # --- list ---
    list_p = subparsers.add_parser("list", help="List all evidence entries")
    list_p.add_argument("--type", dest="filter_type", choices=EVIDENCE_TYPES, help="Filter by type")
    list_p.add_argument("--actor", dest="filter_actor", help="Filter by actor")

    # --- verify ---
    subparsers.add_parser("verify", help="Verify SHA-256 integrity of all evidence content")

    # --- query ---
    query_p = subparsers.add_parser("query", help="Search evidence by keyword")
    query_p.add_argument("keyword", help="Keyword to search for")

    # --- export ---
    subparsers.add_parser("export", help="Export evidence as a Markdown table (stdout)")

    # --- summary ---
    subparsers.add_parser("summary", help="Print investigation statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    store = EvidenceStore(args.store)

    if args.command == "add":
        eid = store.add(
            source=args.source,
            content=args.content,
            evidence_type=args.evidence_type,
            actor=args.actor,
            url=args.url,
            timestamp=args.timestamp,
            ioc_type=args.ioc_type,
            verification=args.verification,
            notes=args.notes,
        )
        if not getattr(args, "quiet", False):
            print(f"✓ Added evidence: {eid}")

    elif args.command == "list":
        items = store.list_evidence(
            filter_type=getattr(args, "filter_type", None),
            filter_actor=getattr(args, "filter_actor", None),
        )
        if not items:
            print("No evidence found.")
        for e in items:
            actor_str = f" | actor: {e['actor']}" if e.get("actor") else ""
            url_str = f" | {e['url']}" if e.get("url") else ""
            print(f"[{e['id']}] {e['type']:12s} | {e['verification']:20s} | {e['source']}{actor_str}{url_str}")

    elif args.command == "verify":
        issues = store.verify_integrity()
        if not issues:
            print(f"✓ All {len(store.data['evidence'])} evidence entries passed SHA-256 integrity check.")
        else:
            print(f"✗ {len(issues)} integrity issue(s) detected:")
            for i in issues:
                print(f"  [{i['id']}] stored={i['stored_sha256'][:16]}... computed={i['computed_sha256'][:16]}...")
            sys.exit(1)

    elif args.command == "query":
        results = store.query(args.keyword)
        print(f"Found {len(results)} result(s) for '{args.keyword}':")
        for e in results:
            print(f"  [{e['id']}] {e['type']} | {e['source']} | {e['content'][:80]}")

    elif args.command == "export":
        print(store.export_markdown())

    elif args.command == "summary":
        s = store.summary()
        print(f"Total evidence items : {s['total']}")
        print(f"By type              : {json.dumps(s['by_type'], indent=2)}")
        print(f"By verification      : {json.dumps(s['by_verification'], indent=2)}")
        print(f"Unique actors        : {s['unique_actors']}")


if __name__ == "__main__":
    main()
