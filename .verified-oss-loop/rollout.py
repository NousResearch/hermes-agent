#!/usr/bin/env python3
"""Read .verified-oss-loop/rollout.yml. Stdlib only. Does not merge."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCHEMES = ("rolling", "staged", "stable")


def load_scheme(root: Path) -> str:
    path = root / ".verified-oss-loop" / "rollout.yml"
    scheme = "rolling"
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("scheme:"):
                scheme = line.split(":", 1)[1].strip().strip("'\"")
                break
    if scheme not in SCHEMES:
        raise SystemExit(f"unknown rollout scheme: {scheme}")
    return scheme


def policy(scheme: str) -> dict[str, str | bool]:
    if scheme == "stable":
        return {
            "scheme": scheme,
            "worker_base": "main",
            "feature_target": "main",
            "overnight_target": "main",
            "automerge_preview": False,
            "automerge_nightly": False,
            "promote_preview_to_nightly": False,
        }
    if scheme == "staged":
        return {
            "scheme": scheme,
            "worker_base": "nightly",
            "feature_target": "preview",
            "overnight_target": "nightly",
            "automerge_preview": True,
            "automerge_nightly": False,
            "promote_preview_to_nightly": True,
        }
    return {
        "scheme": scheme,
        "worker_base": "nightly",
        "feature_target": "preview",
        "overnight_target": "nightly",
        "automerge_preview": True,
        "automerge_nightly": True,
        "promote_preview_to_nightly": True,
    }


def cmd_show(root: Path) -> int:
    p = policy(load_scheme(root))
    for k in (
        "scheme",
        "worker_base",
        "feature_target",
        "overnight_target",
        "automerge_preview",
        "automerge_nightly",
        "promote_preview_to_nightly",
    ):
        print(f"{k}={p[k]}".lower() if isinstance(p[k], bool) else f"{k}={p[k]}")
    return 0


def cmd_get(root: Path, key: str) -> int:
    p = policy(load_scheme(root))
    if key not in p:
        print(f"unknown key: {key}", file=sys.stderr)
        return 2
    val = p[key]
    print(str(val).lower() if isinstance(val, bool) else val)
    return 0


def cmd_allow(root: Path, channel: str) -> int:
    if channel not in ("preview", "nightly"):
        return 1
    p = policy(load_scheme(root))
    ok = bool(p[f"automerge_{channel}"])
    return 0 if ok else 1


def cmd_allow_promote(root: Path, what: str) -> int:
    if what != "preview-to-nightly":
        return 1
    p = policy(load_scheme(root))
    return 0 if p["promote_preview_to_nightly"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verified OSS Loop rollout scheme")
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show")
    g = sub.add_parser("get")
    g.add_argument("key")
    a = sub.add_parser("allow-automerge")
    a.add_argument("channel", choices=("preview", "nightly"))
    pr = sub.add_parser("allow-promote")
    pr.add_argument("what", choices=("preview-to-nightly",))
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    if args.cmd == "show":
        return cmd_show(root)
    if args.cmd == "get":
        return cmd_get(root, args.key)
    if args.cmd == "allow-promote":
        return cmd_allow_promote(root, args.what)
    return cmd_allow(root, args.channel)


if __name__ == "__main__":
    sys.exit(main())
