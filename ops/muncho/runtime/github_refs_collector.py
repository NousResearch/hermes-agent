#!/usr/bin/env python3
"""Collect the three exact Git refs used by the production parity job.

The operation identifier is selected outside this module.  This module only
executes a fixed endpoint table through the packaged credential-aware wrapper;
neither response text nor model prose can change dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence


MAX_RESPONSE_BYTES = 1024 * 1024
ENDPOINTS = {
    "fork_ref": "repos/lomliev/hermes-agent/git/ref/heads/main",
    "upstream_ref": "repos/NousResearch/hermes-agent/git/ref/heads/main",
    "compare": "repos/NousResearch/hermes-agent/compare/main...lomliev:main",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def collect(asset: Path) -> tuple[dict[str, Any], int]:
    if not asset.is_absolute() or asset.name != "gh-hermes":
        raise ValueError("github_wrapper_path_invalid")
    results: dict[str, Any] = {}
    success = True
    for label, endpoint in ENDPOINTS.items():
        try:
            completed = subprocess.run(
                [str(asset), "api", endpoint],
                cwd="/",
                env={
                    "HOME": os.environ.get("HOME", "/var/empty"),
                    "HERMES_HOME": os.environ.get(
                        "HERMES_HOME",
                        "/opt/adventico-ai-platform/hermes-home",
                    ),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "TZ": "UTC",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            results[label] = {
                "endpoint": endpoint,
                "return_code": None,
                "response": None,
                "stdout_sha256": _sha256(bytes(exc.stdout or b"")),
                "stderr_sha256": _sha256(bytes(exc.stderr or b"")),
                "timed_out": True,
            }
            success = False
            continue
        if (
            len(completed.stdout) > MAX_RESPONSE_BYTES
            or len(completed.stderr) > MAX_RESPONSE_BYTES
        ):
            raise ValueError("github_response_oversized")
        try:
            response = json.loads(completed.stdout.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError):
            response = None
        row_ok = completed.returncode == 0 and isinstance(response, dict)
        success = success and row_ok
        results[label] = {
            "endpoint": endpoint,
            "return_code": completed.returncode,
            "response": response if row_ok else None,
            "stdout_sha256": _sha256(completed.stdout),
            "stderr_sha256": _sha256(completed.stderr),
            "timed_out": False,
        }
    return {
        "status": "PASS" if success else "BLOCKED",
        "operation": "github_refs",
        "results": results,
        "secret_material_recorded": False,
    }, 0 if success else 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="muncho-github-refs-collector")
    parser.add_argument("--asset", required=True, type=Path)
    args = parser.parse_args(argv)
    value, code = collect(args.asset)
    print(json.dumps(value, ensure_ascii=True, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
