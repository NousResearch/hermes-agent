#!/usr/bin/env python3
"""Provenance inventory for files copied from verified-oss-loop.

Kit-owned paths update on re-run when unmodified. source: local is never
overwritten, including under --force. Stdlib only. Does not merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "verified-oss-loop.inventory.v1"
INV_DIR = ".verified-oss-loop"
INV_NAME = "inventory.yml"
DEFAULT_KIT = "https://github.com/kvnloo/verified-oss-loop"
SKILL_ROOTS = ("skills", ".agents/skills", ".claude/skills")
SOURCES = ("kit", "local", "modified")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inventory_path(root: Path) -> Path:
    return root / INV_DIR / INV_NAME


def posix(path: str) -> str:
    p = path.replace("\\", "/").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def load_inventory(root: Path) -> dict[str, Any]:
    path = inventory_path(root)
    data: dict[str, Any] = {
        "schema": SCHEMA,
        "kit": DEFAULT_KIT,
        "kit_revision": "",
        "synced_at": "",
        "files": {},
    }
    if not path.is_file():
        return data
    current: dict[str, str] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        dashed = line.strip()
        if dashed.startswith("- path:"):
            rel = posix(dashed.split(":", 1)[1].strip())
            current = {"path": rel, "source": "kit", "sha256": "", "kit_sha256": ""}
            data["files"][rel] = current
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if current is not None and key in ("source", "sha256", "kit_sha256"):
            current[key] = val
        elif current is None and key in ("schema", "kit", "kit_revision", "synced_at"):
            data[key] = val
    return data


def emit_inventory(data: dict[str, Any]) -> str:
    lines = [
        "# Provenance for files copied from kvnloo/verified-oss-loop.",
        "# source: kit = synced from the standard; local = developer-owned, never overwrite;",
        "#         modified = started as kit, local edits (re-run skips unless --force).",
        "# Re-run oss-onboard to receive new kit skills. Mark a path local with:",
        "#   python3 scripts/kit-inventory.py mark --root . --path skills/NAME/SKILL.md --source local",
        f"schema: {data.get('schema', SCHEMA)}",
        f"kit: {data.get('kit') or DEFAULT_KIT}",
        f"kit_revision: {data.get('kit_revision') or ''}",
        f"synced_at: {data.get('synced_at') or ''}",
        "files:",
    ]
    files = data.get("files") or {}
    for rel in sorted(files):
        entry = files[rel]
        lines.append(f"  - path: {rel}")
        lines.append(f"    source: {entry.get('source') or 'kit'}")
        if entry.get("sha256"):
            lines.append(f"    sha256: {entry['sha256']}")
        if entry.get("kit_sha256"):
            lines.append(f"    kit_sha256: {entry['kit_sha256']}")
    lines.append("")
    return "\n".join(lines)


def save_inventory(root: Path, data: dict[str, Any]) -> Path:
    dest = inventory_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(emit_inventory(data), encoding="utf-8")
    return dest


def append_session(session: Path, rec: dict[str, str]) -> None:
    session.parent.mkdir(parents=True, exist_ok=True)
    with session.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def decide(root: Path, rel: str, incoming: Path, dest: Path, force: bool) -> tuple[bool, str, str]:
    """Return (write?, action, reason). Never writes source=local."""
    rel = posix(rel)
    inv = load_inventory(root)
    entry = (inv.get("files") or {}).get(rel)
    source = (entry or {}).get("source") or ""

    if source == "local":
        return False, "skip", "local"

    if not dest.exists():
        return True, "write", "new"

    incoming_hash = sha256_file(incoming)
    dest_hash = sha256_file(dest)
    if dest_hash == incoming_hash:
        return False, "unchanged", "already current"

    if force:
        return True, "force", "force"

    if not entry:
        # First inventory: existing kit-path content may be an older template
        # or a local edit. Do not clobber; next run can --force or mark kit.
        return False, "skip", "modified"

    recorded = entry.get("sha256") or ""
    if source == "kit" and recorded and dest_hash == recorded:
        return True, "update", "kit file unmodified since last sync"

    if source == "modified" and dest_hash == incoming_hash:
        return True, "update", "matches new kit; reclaim"

    return False, "skip", "modified"


def cmd_apply(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    rel = posix(args.path)
    incoming = Path(args.incoming)
    dest = root / rel
    if not incoming.is_file():
        print(f"incoming missing: {incoming}", file=sys.stderr)
        return 2
    force = bool(args.force)
    write, action, reason = decide(root, rel, incoming, dest, force)
    dest.parent.mkdir(parents=True, exist_ok=True)
    incoming_hash = sha256_file(incoming)
    if write:
        shutil.copyfile(incoming, dest)
        if args.chmod:
            mode = int(args.chmod, 8)
            os.chmod(dest, stat.S_IMODE(dest.stat().st_mode) | mode)
        print(f"{action}: {rel}")
    else:
        print(f"skip {rel} ({reason})")
    if args.session:
        append_session(
            Path(args.session),
            {
                "path": rel,
                "action": action,
                "reason": reason,
                "incoming_sha256": incoming_hash,
            },
        )
    return 0


def iter_extra_skills(root: Path, kit_paths: set[str]) -> list[Path]:
    found: list[Path] = []
    for rel_root in SKILL_ROOTS:
        base = root / rel_root
        if not base.is_dir():
            continue
        for p in base.rglob("SKILL.md"):
            rel = p.relative_to(root).as_posix()
            if rel not in kit_paths:
                found.append(p)
    return found


def cmd_finalize(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    old = load_inventory(root)
    files: dict[str, dict[str, str]] = dict(old.get("files") or {})
    session_rows: list[dict[str, str]] = []
    if args.session and Path(args.session).is_file():
        for line in Path(args.session).read_text(encoding="utf-8").splitlines():
            if line.strip():
                session_rows.append(json.loads(line))

    kit_paths: set[str] = set()
    if args.kit_paths:
        for line in Path(args.kit_paths).read_text(encoding="utf-8").splitlines():
            line = posix(line.strip())
            if line:
                kit_paths.add(line)
    for row in session_rows:
        kit_paths.add(posix(row["path"]))

    by_path = {posix(r["path"]): r for r in session_rows}

    for rel in sorted(kit_paths):
        dest = root / rel
        prev = files.get(rel) or {}
        row = by_path.get(rel)
        action = (row or {}).get("action") or ""
        reason = (row or {}).get("reason") or ""
        incoming_hash = (row or {}).get("incoming_sha256") or ""

        if prev.get("source") == "local" or reason == "local" or action == "skip" and reason == "local":
            entry = {"path": rel, "source": "local", "sha256": "", "kit_sha256": prev.get("kit_sha256") or ""}
            if dest.is_file():
                entry["sha256"] = sha256_file(dest)
            files[rel] = entry
            continue

        if not dest.is_file():
            files.pop(rel, None)
            continue

        dest_hash = sha256_file(dest)
        if action in ("write", "update", "force", "unchanged"):
            files[rel] = {
                "path": rel,
                "source": "kit",
                "sha256": dest_hash,
                "kit_sha256": incoming_hash or dest_hash,
            }
        elif reason == "modified" or action == "skip":
            files[rel] = {
                "path": rel,
                "source": "modified",
                "sha256": dest_hash,
                "kit_sha256": incoming_hash or prev.get("kit_sha256") or "",
            }
        else:
            files[rel] = {
                "path": rel,
                "source": prev.get("source") or "kit",
                "sha256": dest_hash,
                "kit_sha256": incoming_hash or prev.get("kit_sha256") or dest_hash,
            }

    for extra in iter_extra_skills(root, kit_paths):
        rel = extra.relative_to(root).as_posix()
        prev = files.get(rel) or {}
        files[rel] = {
            "path": rel,
            "source": "local",
            "sha256": sha256_file(extra),
            "kit_sha256": prev.get("kit_sha256") or "",
        }

    # Keep previously recorded local files that are not skills (rare).
    for rel, entry in list(files.items()):
        if rel in kit_paths:
            continue
        if entry.get("source") != "local":
            continue
        dest = root / rel
        if dest.is_file():
            entry["sha256"] = sha256_file(dest)
            files[rel] = entry
        else:
            files.pop(rel, None)

    data = {
        "schema": SCHEMA,
        "kit": args.kit_url or old.get("kit") or DEFAULT_KIT,
        "kit_revision": args.kit_revision or old.get("kit_revision") or "",
        "synced_at": now_rfc3339(),
        "files": files,
    }
    dest = save_inventory(root, data)
    print(f"wrote {dest.relative_to(root).as_posix()}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    inv = load_inventory(root)
    if not inventory_path(root).is_file():
        print("no inventory; run oss-onboard to create .verified-oss-loop/inventory.yml")
        return 1
    grouped: dict[str, list[str]] = {"kit": [], "modified": [], "local": []}
    for rel, entry in (inv.get("files") or {}).items():
        src = entry.get("source") or "kit"
        grouped.setdefault(src, []).append(rel)
    if args.json:
        print(json.dumps({"kit": inv.get("kit"), "kit_revision": inv.get("kit_revision"), "synced_at": inv.get("synced_at"), "files": inv.get("files")}, indent=2))
        return 0
    print(f"kit: {inv.get('kit')} @ {inv.get('kit_revision') or '?'}")
    print(f"synced_at: {inv.get('synced_at')}")
    for src in ("kit", "modified", "local"):
        paths = grouped.get(src) or []
        print(f"\n{src}:")
        if not paths:
            print("  (none)")
            continue
        for p in sorted(paths):
            print(f"  {p}")
    return 0


def cmd_source(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    rel = posix(args.path)
    entry = (load_inventory(root).get("files") or {}).get(rel)
    if not entry:
        print("missing")
        return 0
    print(entry.get("source") or "missing")
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    rel = posix(args.path)
    source = args.source
    if source not in SOURCES:
        print(f"source must be one of {SOURCES}", file=sys.stderr)
        return 2
    dest = root / rel
    if not dest.is_file():
        print(f"missing file: {rel}", file=sys.stderr)
        return 1
    inv = load_inventory(root)
    files = inv.setdefault("files", {})
    prev = files.get(rel) or {}
    files[rel] = {
        "path": rel,
        "source": source,
        "sha256": sha256_file(dest),
        "kit_sha256": prev.get("kit_sha256") or "",
    }
    inv["synced_at"] = inv.get("synced_at") or now_rfc3339()
    save_inventory(root, inv)
    print(f"marked {rel} source={source}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verified OSS Loop kit inventory (provenance + sync)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("apply", help="copy incoming to dest if inventory allows")
    ap.add_argument("--root", required=True)
    ap.add_argument("--path", required=True, help="path relative to root")
    ap.add_argument("--incoming", required=True)
    ap.add_argument("--session", help="JSONL session file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--chmod", help="octal bits to OR onto mode, e.g. 755")
    ap.set_defaults(func=cmd_apply)

    fin = sub.add_parser("finalize", help="write inventory.yml after a sync")
    fin.add_argument("--root", required=True)
    fin.add_argument("--session")
    fin.add_argument("--kit-paths", help="newline list of kit-relative paths attempted this run")
    fin.add_argument("--kit-revision", default="")
    fin.add_argument("--kit-url", default=DEFAULT_KIT)
    fin.set_defaults(func=cmd_finalize)

    sh = sub.add_parser("show", help="print inventory grouped by source")
    sh.add_argument("--root", required=True)
    sh.add_argument("--json", action="store_true")
    sh.set_defaults(func=cmd_show)

    src = sub.add_parser("source", help="print source for one path")
    src.add_argument("--root", required=True)
    src.add_argument("--path", required=True)
    src.set_defaults(func=cmd_source)

    mk = sub.add_parser("mark", help="set source=local|kit|modified for a path")
    mk.add_argument("--root", required=True)
    mk.add_argument("--path", required=True)
    mk.add_argument("--source", required=True, choices=SOURCES)
    mk.set_defaults(func=cmd_mark)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
