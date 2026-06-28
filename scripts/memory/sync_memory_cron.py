#!/usr/bin/env python3
"""Cron-safe wrapper for ``sync_memory.py`` (must live under ``~/.hermes/scripts/``).

Runs the repo orchestrator with cwd = cron ``--workdir``, then prints a
redacted JSON summary suitable for no-agent delivery.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _safe_summary(payload: dict[str, Any]) -> dict[str, Any]:
    social = payload.get("social") or {}
    obsidian = payload.get("obsidian") or {}
    return {
        "success": bool(payload.get("success")),
        "social": {
            "sources": social.get("sources"),
            "sessions_seen": social.get("sessions_seen"),
            "x_events_seen": social.get("x_events_seen"),
            "memories_written": social.get("memories_written"),
            "incremental": social.get("incremental"),
            "index_path": social.get("index_path"),
            "sleep": social.get("sleep"),
        },
        "obsidian": {
            "success": obsidian.get("success"),
            "skipped": obsidian.get("skipped"),
            "dry_run": obsidian.get("dry_run"),
            "items": obsidian.get("items"),
            "groups": obsidian.get("groups"),
            "wiki_root": obsidian.get("wiki_root"),
            "error": obsidian.get("error"),
        },
        "index_updated": payload.get("index_updated"),
    }


def main() -> int:
    repo_root = Path.cwd()
    sync_script = repo_root / "sync_memory.py"
    if not sync_script.is_file():
        print(json.dumps({"success": False, "error": f"sync_memory.py not found under {repo_root}"}))
        return 1

    proc = subprocess.run(
        [sys.executable, str(sync_script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"sync_memory exited {proc.returncode}",
                    "stderr_tail": (proc.stderr or "")[-500:],
                },
                ensure_ascii=False,
            )
        )
        return proc.returncode

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(json.dumps({"success": False, "error": "invalid JSON from sync_memory.py"}))
        return 1

    print(json.dumps(_safe_summary(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
