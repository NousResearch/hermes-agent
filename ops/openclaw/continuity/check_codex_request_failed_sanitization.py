#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import subprocess
import sys

RAW = "Codex request failed (unknown_error)."
EXPECTED = "The AI service is temporarily unavailable. Please try again in a moment."


def resolve_active_helper(root: pathlib.Path) -> pathlib.Path:
    dispatch_files = sorted(glob.glob(str(root / "dist/plugin-sdk/dispatch-*.js")))
    if not dispatch_files:
        raise RuntimeError(f"No dispatch bundle found under {root}/dist/plugin-sdk")

    dispatch_path = pathlib.Path(dispatch_files[0])
    head = dispatch_path.read_text(encoding="utf-8")[:20000]
    m = re.search(r'from "\./(pi-embedded-helpers-[^"]+\.js)";', head)
    if not m:
        raise RuntimeError(f"Could not resolve helper import from {dispatch_path}")

    helper_name = m.group(1)
    helper_path = root / "dist/plugin-sdk" / helper_name
    if not helper_path.exists():
        raise RuntimeError(f"Resolved helper does not exist: {helper_path}")
    return helper_path


def run_node(helper_path: pathlib.Path) -> dict:
    code = f"""
const mod = await import('file://{helper_path}');
const raw = {json.dumps(RAW)};
const msg = {{ stopReason: 'error', errorMessage: raw }};
console.log(JSON.stringify({{ rawOut: mod.y(raw), assistantOut: mod._(msg, {{}}) }}));
"""
    proc = subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"node exited {proc.returncode}")
    return json.loads(proc.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that Codex request-failed errors are user-sanitized.")
    parser.add_argument("--root", default="/usr/lib/node_modules/openclaw", help="OpenClaw package root")
    parser.add_argument("--expect-leak", action="store_true", help="Expect current vulnerable behavior (for baseline checks)")
    args = parser.parse_args()

    root = pathlib.Path(args.root).resolve()
    helper = resolve_active_helper(root)
    out = run_node(helper)

    print(json.dumps({"helper": str(helper), **out}, ensure_ascii=False))

    if args.expect_leak:
        ok = out.get("rawOut") == RAW and out.get("assistantOut") == RAW
        return 0 if ok else 1

    ok = out.get("rawOut") == EXPECTED and out.get("assistantOut") == EXPECTED
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
