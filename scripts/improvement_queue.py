#!/usr/bin/env python3
"""improvement_queue.py — the human-approval gate for Friday's self-authorship.

A receipt layer, not an auto-mutator. The identity-reflection loop (and the very
first SOUL.md self-description amendment) *propose* file changes here; nothing
touches a target file until a human runs ``approve``. On approve, the current
target is snapshotted into ``identity/audit-backups/`` before the proposed
content is written, so every applied change is reversible and auditable.

Proposals live under ``~/.hermes/identity/queue/<id>/``:
    meta.json   — id, target, source, risk, summary, body, status, timestamps
    proposed    — the full proposed file content
    diff        — unified diff vs the target at creation time

Subcommands: create, list, show, digest, approve, reject.
Deployed to ``~/.hermes/scripts/``.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from hermes_constants import get_hermes_home
except ImportError:  # pragma: no cover - fallback for detached deployments
    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"


def _identity_dir() -> Path:
    d = get_hermes_home() / "identity"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _queue_dir() -> Path:
    d = _identity_dir() / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _backups_dir() -> Path:
    d = _identity_dir() / "audit-backups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _proposal_dir(pid: str) -> Path:
    return _queue_dir() / pid


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _next_id() -> str:
    existing = [
        int(p.name) for p in _queue_dir().iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return f"{(max(existing) + 1) if existing else 1:04d}"


def _load_meta(pid: str) -> Optional[dict]:
    mp = _proposal_dir(pid) / "meta.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_meta(pid: str, meta: dict) -> None:
    (_proposal_dir(pid) / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def cmd_create(args: argparse.Namespace) -> int:
    target = Path(args.target).expanduser()
    proposed_path = Path(args.proposed_file).expanduser()
    if not proposed_path.exists():
        print(f"improvement_queue: proposed file not found: {proposed_path}",
              file=sys.stderr)
        return 2

    proposed = proposed_path.read_text(encoding="utf-8")
    current = _read_text_or_empty(target)

    diff = "".join(difflib.unified_diff(
        current.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=f"a/{target.name}",
        tofile=f"b/{target.name}",
    ))
    if not diff.strip():
        print("improvement_queue: proposed content is identical to the target; "
              "nothing to propose.", file=sys.stderr)
        return 2

    pid = _next_id()
    pdir = _proposal_dir(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "proposed").write_text(proposed, encoding="utf-8")
    (pdir / "diff").write_text(diff, encoding="utf-8")

    meta = {
        "id": pid,
        "target": str(target),
        "source": args.source or "unknown",
        "risk": args.risk or "low",
        "summary": args.summary or "",
        "body": args.body or "",
        "status": "pending_review",
        "created_at": int(time.time()),
        "target_sha256_at_create": _sha256(current),
    }
    _save_meta(pid, meta)
    print(f"improvement_queue: created proposal {pid} "
          f"(target {target}, status pending_review)")
    return 0


# ---------------------------------------------------------------------------
# list / show
# ---------------------------------------------------------------------------

def _iter_proposals():
    for p in sorted(_queue_dir().iterdir(), key=lambda x: x.name):
        if p.is_dir() and p.name.isdigit():
            meta = _load_meta(p.name)
            if meta:
                yield meta


def cmd_list(args: argparse.Namespace) -> int:
    found = False
    for meta in _iter_proposals():
        if args.status and meta.get("status") != args.status:
            continue
        found = True
        print(f"{meta['id']}  [{meta.get('status')}]  "
              f"{meta.get('source')}  ->  {meta.get('target')}")
        if meta.get("summary"):
            print(f"      {meta['summary']}")
    if not found:
        print("(no proposals)" if not args.status
              else f"(no proposals with status {args.status})")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    meta = _load_meta(args.id)
    if meta is None:
        print(f"improvement_queue: no such proposal: {args.id}", file=sys.stderr)
        return 2
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    diff = _read_text_or_empty(_proposal_dir(args.id) / "diff")
    if diff.strip():
        print("\n--- diff ---")
        print(diff)
    return 0


def cmd_digest(args: argparse.Namespace) -> int:
    pending = [
        meta for meta in _iter_proposals()
        if meta.get("status") == "pending_review"
    ]
    if not pending and not args.force:
        return 0
    if not pending:
        print("Verdict — Identity improvement queue is clear.")
        print("Action: none.")
        return 0

    print(f"Verdict — {len(pending)} identity proposal(s) need review.")
    print("Evidence: ~/.hermes/identity/queue")
    for meta in pending[:args.limit]:
        pid = meta["id"]
        summary = meta.get("summary") or "(no summary)"
        print(
            f"- {pid}: {summary} | risk={meta.get('risk', 'unknown')} | "
            f"source={meta.get('source', 'unknown')} | target={meta.get('target')}"
        )
        print(f"  show: python3 ~/.hermes/scripts/improvement_queue.py show {pid}")
        print(f"  approve: python3 ~/.hermes/scripts/improvement_queue.py approve {pid}")
        print(f"  reject: python3 ~/.hermes/scripts/improvement_queue.py reject {pid}")
    return 0


# ---------------------------------------------------------------------------
# approve / reject
# ---------------------------------------------------------------------------

def cmd_approve(args: argparse.Namespace) -> int:
    meta = _load_meta(args.id)
    if meta is None:
        print(f"improvement_queue: no such proposal: {args.id}", file=sys.stderr)
        return 2
    if meta.get("status") != "pending_review":
        print(f"improvement_queue: proposal {args.id} is not pending "
              f"(status: {meta.get('status')}).", file=sys.stderr)
        return 2

    target = Path(meta["target"]).expanduser()
    proposed = _read_text_or_empty(_proposal_dir(args.id) / "proposed")
    current = _read_text_or_empty(target)

    # Drift detection: warn (but proceed) if the target changed since the
    # proposal was created. The human is approving with eyes open.
    if _sha256(current) != meta.get("target_sha256_at_create"):
        print(f"improvement_queue: WARNING — {target} changed since this "
              f"proposal was created; the diff may not apply as previewed.",
              file=sys.stderr)

    # Snapshot the current target before overwriting (audit-backup convention).
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = _backups_dir() / f"{ts}-{target.name}-{args.id}.bak"
    backup.write_text(current, encoding="utf-8")
    with (_backups_dir() / "MANIFEST.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{ts}\tproposal={args.id}\ttarget={target}\t"
                 f"backup={backup.name}\tsource={meta.get('source')}\n")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(proposed, encoding="utf-8")

    meta["status"] = "approved"
    meta["approved_at"] = int(time.time())
    meta["backup"] = backup.name
    _save_meta(args.id, meta)
    print(f"improvement_queue: approved {args.id} — wrote {target} "
          f"(backup: {backup.name})")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    meta = _load_meta(args.id)
    if meta is None:
        print(f"improvement_queue: no such proposal: {args.id}", file=sys.stderr)
        return 2
    if meta.get("status") != "pending_review":
        print(f"improvement_queue: proposal {args.id} is not pending "
              f"(status: {meta.get('status')}).", file=sys.stderr)
        return 2
    meta["status"] = "rejected"
    meta["rejected_at"] = int(time.time())
    if args.reason:
        meta["reject_reason"] = args.reason
    _save_meta(args.id, meta)
    print(f"improvement_queue: rejected {args.id} (target left untouched)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command_name", required=True)

    c = sub.add_parser("create", help="Stage a proposed change to a target file.")
    c.add_argument("--target", required=True, help="Path of the file to change.")
    c.add_argument("--proposed-file", required=True,
                   help="Path to the file holding the full proposed content.")
    c.add_argument("--source", help="Who proposed this (e.g. identity-reflection).")
    c.add_argument("--risk", help="Risk label (default: low).")
    c.add_argument("--summary", help="One-line summary.")
    c.add_argument("--body", help="Longer rationale / cited receipts.")
    c.set_defaults(func=cmd_create)

    li = sub.add_parser("list", help="List proposals.")
    li.add_argument("--status", help="Filter by status (e.g. pending_review).")
    li.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="Show a proposal's metadata and diff.")
    sh.add_argument("id")
    sh.set_defaults(func=cmd_show)

    dg = sub.add_parser("digest", help="Print pending proposals for review delivery.")
    dg.add_argument("--force", action="store_true",
                    help="Print a clear message even when no proposals are pending.")
    dg.add_argument("--limit", type=int, default=8)
    dg.set_defaults(func=cmd_digest)

    ap = sub.add_parser("approve", help="Apply a proposal (with audit backup).")
    ap.add_argument("id")
    ap.set_defaults(func=cmd_approve)

    rj = sub.add_parser("reject", help="Reject a proposal (target untouched).")
    rj.add_argument("id")
    rj.add_argument("--reason")
    rj.set_defaults(func=cmd_reject)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
