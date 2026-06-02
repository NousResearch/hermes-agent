#!/usr/bin/env python3
"""One Feishu doc per canonical arXiv paper on board paper-nexus."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_VERSION_SUFFIX = re.compile(r"v\d+$", re.I)


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


def canonical_paper_id(paper_id: str) -> str:
    """2402.03300v3 and 2402.03300 → same canonical id; s2:<hash> unchanged."""
    pid = (paper_id or "").strip()
    if pid.lower().startswith("s2:"):
        return pid.lower()
    return _VERSION_SUFFIX.sub("", pid)


def registry_path(board: str = "paper-nexus") -> Path:
    return hermes_home() / "kanban" / "boards" / board / "paper_doc_registry.json"


def load_registry(board: str = "paper-nexus") -> dict:
    path = registry_path(board)
    if not path.is_file():
        return {"version": 1, "papers": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("papers", {})
    return data


def save_registry(data: dict, board: str = "paper-nexus") -> Path:
    path = registry_path(board)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def lookup(paper_id: str, board: str = "paper-nexus") -> dict | None:
    key = canonical_paper_id(paper_id)
    entry = load_registry(board)["papers"].get(key)
    if not entry:
        return None
    return {"canonical_id": key, **entry}


def register(
    paper_id: str,
    doc_url: str,
    *,
    document_id: str | None = None,
    title: str | None = None,
    title_zh: str | None = None,
    board: str = "paper-nexus",
) -> dict:
    key = canonical_paper_id(paper_id)
    now = datetime.now(timezone.utc).isoformat()
    data = load_registry(board)
    papers = data["papers"]
    prev = papers.get(key) or {}
    papers[key] = {
        "doc_url": doc_url,
        "document_id": document_id or prev.get("document_id"),
        "title": title or prev.get("title"),
        "title_zh": title_zh or prev.get("title_zh"),
        "paper_id_latest": paper_id,
        "created_at": prev.get("created_at") or now,
        "updated_at": now,
    }
    save_registry(data, board)
    return papers[key]


def resolve(paper_id: str, board: str = "paper-nexus") -> dict:
    """Return {action: create|update, canonical_id, doc_url?}."""
    key = canonical_paper_id(paper_id)
    entry = lookup(paper_id, board)
    if entry and entry.get("doc_url"):
        return {
            "action": "update",
            "canonical_id": key,
            "doc_url": entry["doc_url"],
            "document_id": entry.get("document_id"),
        }
    return {"action": "create", "canonical_id": key}


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "Usage: paper_doc_registry.py <lookup|register|resolve|list> <paper_id> [doc_url]",
            file=sys.stderr,
        )
        return 2
    cmd, paper_id = sys.argv[1], sys.argv[2]
    if cmd == "lookup":
        json.dump(lookup(paper_id) or {}, sys.stdout, ensure_ascii=False, indent=2)
    elif cmd == "resolve":
        json.dump(resolve(paper_id), sys.stdout, ensure_ascii=False, indent=2)
    elif cmd == "register":
        if len(sys.argv) < 4:
            print("register requires doc_url", file=sys.stderr)
            return 2
        json.dump(register(paper_id, sys.argv[3]), sys.stdout, ensure_ascii=False, indent=2)
    elif cmd == "list":
        json.dump(load_registry()["papers"], sys.stdout, ensure_ascii=False, indent=2)
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
