#!/usr/bin/env python3
"""Build the TUI and package a prepared bionic payload as a Termux .deb.

Tool staging and wheelhouse preparation remain separate prerequisites. This
sequence owns only the shared frontend build and the native .deb handoff.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.releases.commit_build import require_commit
from scripts.termux.deb_version import deb_version_for_tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--tag")
    identity.add_argument("--commit")
    args = parser.parse_args(argv)

    try:
        if args.tag is not None:
            deb_version_for_tag(args.tag)
        else:
            require_commit(args.commit)
    except ValueError as exc:
        parser.error(str(exc))

    repo, payload, out = (path.resolve() for path in (args.repo, args.payload, args.out))
    for option, path in (("--repo", repo), ("--payload", payload)):
        if not path.is_dir():
            parser.error(f"{option} must name an existing directory: {path}")
    product = repo / ".build/termux/tui"
    revision = ["--tag", args.tag] if args.tag is not None else ["--commit", args.commit]
    commands = [
        ["node", str(repo / "scripts/build/node-deps.mjs"),
         "--source", str(repo), "--workspace", "ui-tui"],
        ["node", str(repo / "scripts/build/tui.mjs"),
         "--source", str(repo), "--out", str(product)],
        ["bash", str(repo / "scripts/termux/build_deb.sh"),
         "--repo", str(repo), "--payload", str(payload), "--out", str(out),
         "--tui-product", str(product), *revision],
    ]
    try:
        for command in commands:
            subprocess.run(command, cwd=repo, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    except OSError as exc:
        print(f"termux build: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
