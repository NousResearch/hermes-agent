#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.wiki_lifecycle import WikiLifecycleStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Compatibility CLI for wiki lifecycle tools")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init")
    init_p.add_argument("--wiki-dir", default=str(Path.home() / "wiki"))
    init_p.add_argument("--db", default="")
    init_p.add_argument("--domain", default="ops")

    lint_p = sub.add_parser("lint")
    lint_p.add_argument("--db", required=True)
    lint_p.add_argument("--json-out", default="")

    recompute_p = sub.add_parser("recompute")
    recompute_p.add_argument("--db", required=True)

    export_p = sub.add_parser("export")
    export_p.add_argument("--db", required=True)
    export_p.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.command == "init":
        db = args.db or str(Path(args.wiki_dir) / "wiki_lifecycle.db")
        store = WikiLifecycleStore(db)
        store.close()
        print(json.dumps({"ok": True, "db": db, "domain": args.domain}, ensure_ascii=False))
        return 0

    store = WikiLifecycleStore(args.db)
    try:
        if args.command == "lint":
            issues = store.lint()
            if args.json_out:
                out = Path(args.json_out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"ok": True, "issue_count": len(issues), "issues": issues[:15]}, ensure_ascii=False))
            return 0

        if args.command == "recompute":
            print(json.dumps(store.recompute(), ensure_ascii=False))
            return 0

        if args.command == "export":
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(store.export_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(str(out))
            return 0
    finally:
        store.close()

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
