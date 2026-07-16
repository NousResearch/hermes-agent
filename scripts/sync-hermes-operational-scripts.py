#!/usr/bin/env python3
"""Synchronize the approved operational scripts between the repo and Hermes home.

The first run imports deployed-only operational scripts into the repository.
After that, the repository scripts are authoritative and are copied to Hermes home.
Only the explicit allowlist below is touched.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO_SCRIPTS = Path(r"C:\Users\downl\Documents\New project\hermes-agent\scripts")
HERMES_SCRIPTS = Path(r"C:\Users\downl\.hermes\scripts")
REPORT_DIR = Path(r"C:\Users\downl\.hermes\sync-reports")

# Operational cron scripts only; do not broaden without an explicit review.
ALLOWLIST = (
    "cross-platform-memory-sleep-fallback.py",
    "daily_moa_provider_selector.py",
    "daily_vrchat_post.py",
    "disaster-news-jp.py",
    "lm-twitterer-post.py",
    "lm-twitterer-replies.py",
    "mhlw-designated-check.py",
    "osint-agent-evening.py",
    "osint-agent-morning.py",
    "warashibe-hourly-arb-scan.py",
    "warashibe-x-niche-price-scan.py",
    "wm-osint-pdb-evening.py",
    "wm-osint-pdb-morning.py",
    "worldmonitor-fusion-jp-security-noagent.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_checked(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    REPO_SCRIPTS.mkdir(parents=True, exist_ok=True)
    HERMES_SCRIPTS.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    imported: list[str] = []
    deployed: list[str] = []
    unchanged: list[str] = []
    missing: list[str] = []

    for name in ALLOWLIST:
        repo = REPO_SCRIPTS / name
        deployed_path = HERMES_SCRIPTS / name
        repo_exists = repo.is_file()
        deployed_exists = deployed_path.is_file()

        if not repo_exists and not deployed_exists:
            missing.append(name)
            continue
        if not repo_exists and deployed_exists:
            # Bootstrap the repository from the known deployed operational copy.
            copy_checked(deployed_path, repo)
            imported.append(name)
            continue
        if repo_exists and not deployed_exists:
            copy_checked(repo, deployed_path)
            deployed.append(name)
            continue

        if digest(repo) == digest(deployed_path):
            unchanged.append(name)
            continue

        # Repository is authoritative after bootstrap.
        copy_checked(repo, deployed_path)
        deployed.append(name)

    now = datetime.now(timezone.utc).astimezone().isoformat()
    report = {
        "timestamp": now,
        "repo_scripts": str(REPO_SCRIPTS),
        "hermes_scripts": str(HERMES_SCRIPTS),
        "allowlist_count": len(ALLOWLIST),
        "imported_to_repo": imported,
        "deployed_to_hermes": deployed,
        "unchanged": unchanged,
        "missing": missing,
        "ok": not missing,
    }
    report_path = REPORT_DIR / "latest-operational-script-sync.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"operational script sync: imported={len(imported)} deployed={len(deployed)} unchanged={len(unchanged)} missing={len(missing)}")
    print(f"report: {report_path}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
