#!/usr/bin/env python3
"""Format ranked papers and send Feishu IM via lark-cli (no online doc)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
_LIVE = _DIR.parents[2] / "devops" / "kanban-feishu-live" / "scripts"
if str(_LIVE) not in sys.path:
    sys.path.insert(0, str(_LIVE))

from paper_search_rank import run_search  # noqa: E402


def _slug(query: str) -> str:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "-", query.strip())[:40].strip("-")
    return s or "query"


def _trim(t: str, n: int = 100) -> str:
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def format_top_list(result: dict) -> str:
    lines = [
        f"📚 文献检索结果 · {_trim(result['query'], 48)}",
        f"权重：相关性35% + 引用30% + 高影响引用15% + 时效15% + 可下载5%",
        f"候选 {result['candidate_count']} 篇 → Top {len(result['papers'])}",
        "",
    ]
    for i, p in enumerate(result["papers"], 1):
        sc = p.get("scores", {})
        cite = p.get("citation_count") or 0
        infl = p.get("influential_citation_count") or 0
        year = p.get("year") or "?"
        link = p.get("arxiv_abs") or p.get("url") or ""
        lines.append(
            f"{i}. [{sc.get('display', '?')}] {_trim(p.get('title', ''), 72)} ({year})"
        )
        lines.append(f"   引用 {cite} / 高影响 {infl} | {link}")
        if p.get("venue"):
            lines.append(f"   venue: {_trim(p['venue'], 40)}")
    lines.append("")
    lines.append("精读某篇：/kanban-paper-nexus <arXiv id>")
    return "\n".join(lines)


def send_text(chat_id: str, text: str, *, dry_run: bool = False) -> dict:
    if dry_run:
        return {"dry_run": True, "chars": len(text)}
    proc = subprocess.run(
        [
            "lark-cli",
            "im",
            "+messages-send",
            "--as",
            "bot",
            "--chat-id",
            chat_id,
            "--text",
            text,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"lark-cli failed: {err}")
    raw = proc.stdout.strip()
    start = raw.find("{")
    if start >= 0:
        return json.loads(raw[start:])
    return {"ok": True}


def deliver_progress(
    chat_id: str,
    query: str,
    *,
    board: str = "paper-search",
    event: str,
    stage: str = "",
    summary: str = "",
    dry_run: bool = False,
) -> None:
    """Use kanban-feishu-live notify when session exists; else direct send."""
    slug = _slug(query)
    script = _LIVE / "kanban_feishu_stage_notify.py"
    if script.is_file():
        args = [
            sys.executable,
            str(script),
            "--board",
            board,
            "notify",
            "--entity-id",
            slug,
            "--event",
            event,
        ]
        if stage:
            args.extend(["--stage", stage])
        if summary:
            args.extend(["--summary", summary])
        if dry_run:
            args.append("--dry-run")
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
    # fallback one-liner
    msg = f"📚 paper-search · {query}\n{event} {stage}\n{summary}"
    send_text(chat_id, msg, dry_run=dry_run)


def init_session(
    chat_id: str,
    query: str,
    result: dict,
    *,
    board: str = "paper-search",
    dry_run: bool = False,
) -> None:
    slug = _slug(query)
    script = _LIVE / "kanban_feishu_stage_notify.py"
    if not script.is_file() or dry_run:
        return
    tasks = json.dumps({"T0": "search", "T1": "rank", "T2": "deliver"})
    # meta via env not supported; title carries query text
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--board",
            board,
            "init",
            slug,
            "--chat-id",
            chat_id,
            "--title",
            _trim(query, 60),
            "--tasks-inline",
            tasks,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--chat-id", default=os.environ.get("HERMES_SESSION_CHAT_ID", ""))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--profile", default="ml")
    ap.add_argument("--boost-recency", action="store_true")
    ap.add_argument("--min-citations", type=int, default=0)
    ap.add_argument("--json-in", help="skip search, use ranked json file")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chat_id = (args.chat_id or "").strip()
    if not chat_id and not args.dry_run:
        print("chat-id required", file=sys.stderr)
        return 2

    if args.json_in:
        result = json.loads(Path(args.json_in).read_text(encoding="utf-8"))
    else:
        result = run_search(
            args.query,
            top=args.top,
            profile=args.profile,
            boost_recency=args.boost_recency,
            min_citations=args.min_citations,
        )

    if not args.dry_run:
        init_session(chat_id, args.query, result)

    deliver_progress(
        chat_id,
        args.query,
        event="pipeline_started",
        summary=f"检索式：{args.query}",
        dry_run=args.dry_run,
    )
    deliver_progress(
        chat_id,
        args.query,
        event="stage_done",
        stage="T0",
        summary=f"候选 {result['candidate_count']} 篇（S2+arXiv）",
        dry_run=args.dry_run,
    )
    deliver_progress(
        chat_id,
        args.query,
        event="stage_done",
        stage="T1",
        summary=f"已排序 Top {len(result['papers'])}",
        dry_run=args.dry_run,
    )

    body = format_top_list(result)
    if args.dry_run:
        print(body)
        return 0

    deliver_progress(
        chat_id,
        args.query,
        event="pipeline_done",
        summary=f"Top {len(result['papers'])} 已列出",
        dry_run=False,
    )
    send_text(chat_id, body)
    json.dump({"ok": True, "query": args.query, "delivered": len(result["papers"])}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
