"""Remove only run-isolated caches after a PM Toolchain smoke run completes."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess


def cleanup_run_caches(run_id: str, request, *, page_size: int = 100) -> list[int]:
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("invalid run ID")
    run = request("GET", f"actions/runs/{run_id}")
    if run["status"] != "completed" or run["path"] != ".github/workflows/pm-toolchain.yml":
        raise ValueError("cleanup requires a completed PM Toolchain run")
    pattern = re.compile(
        rf"^(?:(?:setup-pm-tools-|node-cache-).*-smoke(?:-prune|-consumers)?-{run_id}-[1-9][0-9]*"
        rf"|setup-pm-uv-v2-smoke(?:-prune|-consumers)?-{run_id}-[1-9][0-9]*-.+)$"
    )

    def collect():
        # Finish pagination before deleting: deletion shifts the next page.
        rows = []
        page = 1
        while True:
            data = request("GET", f"actions/caches?per_page={page_size}&page={page}")
            rows.extend(data["actions_caches"])
            if len(rows) >= data["total_count"]:
                return [row for row in rows if pattern.fullmatch(row["key"])]
            if not data["actions_caches"]:
                raise RuntimeError("cache inventory ended before its declared total")
            page += 1

    selected = collect()
    for row in selected:
        request("DELETE", f"actions/caches/{row['id']}")
        print(f"deleted smoke cache {row['id']}: {row['key']}")
    remaining = collect()
    if remaining:
        raise RuntimeError(f"smoke caches remain after cleanup: {[row['id'] for row in remaining]}")
    return [row["id"] for row in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    repository = os.environ["GITHUB_REPOSITORY"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("invalid repository")
    def request(method, path):
        result = subprocess.run(
            ["gh", "api", "--method", method, f"repos/{repository}/{path}"],
            check=True, capture_output=True, text=True, encoding="utf-8", timeout=90,
        )
        return json.loads(result.stdout) if result.stdout.strip() else None

    removed = cleanup_run_caches(args.run_id, request)
    print(f"removed {len(removed)} run-isolated caches")


if __name__ == "__main__":
    main()
