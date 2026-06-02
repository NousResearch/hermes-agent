#!/usr/bin/env python3
"""Initialize paper-nexus Feishu live updates in one shot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_LIVE = Path(__file__).resolve().parents[3] / "devops" / "kanban-feishu-live" / "scripts"


def _run_step(args: list[str]) -> dict:
    proc = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    raw = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0:
        raise RuntimeError(f"step failed ({proc.returncode}): {' '.join(args)} :: {raw[:1000]}")
    start = raw.find("{")
    return json.loads(raw[start:]) if start >= 0 else {"ok": True, "raw": raw}


def initialize_live(
    canonical_id: str,
    tasks_inline: str,
    *,
    title_zh: str = "",
    chat_id: str = "",
) -> dict:
    notify = str(_LIVE / "kanban_feishu_stage_notify.py")
    subscribe = str(_LIVE / "kanban_feishu_subscribe.py")
    init_args = [
        notify,
        "--board",
        "paper-nexus",
        "init",
        canonical_id,
        "--tasks-inline",
        tasks_inline,
    ]
    if title_zh:
        init_args.extend(["--title-zh", title_zh])
    if chat_id:
        init_args.extend(["--chat-id", chat_id])
    init_out = _run_step(init_args)
    subscribe_out = _run_step([subscribe, "--board", "paper-nexus", canonical_id])
    started_out = _run_step([
        notify,
        "--board",
        "paper-nexus",
        "notify",
        "--entity-id",
        canonical_id,
        "--event",
        "pipeline_started",
    ])
    return {
        "ok": True,
        "canonical_id": canonical_id,
        "init": init_out,
        "subscribe": subscribe_out,
        "pipeline_started": started_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("canonical_id")
    ap.add_argument("--tasks-inline", required=True)
    ap.add_argument("--title-zh", default="")
    ap.add_argument("--chat-id", default="")
    ns = ap.parse_args()
    out = initialize_live(
        ns.canonical_id,
        ns.tasks_inline,
        title_zh=ns.title_zh,
        chat_id=ns.chat_id,
    )
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
