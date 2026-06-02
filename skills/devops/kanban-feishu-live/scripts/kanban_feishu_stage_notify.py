#!/usr/bin/env python3
"""Feishu live Kanban stage updates via lark-cli (no Hermes core changes)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from kanban_feishu_boards import BOARD_CONFIG, get_board_config  # noqa: E402
from kanban_feishu_session import load_session, new_session, save_session  # noqa: E402

_STATUS_ICON = {
    "triage": "📋",
    "todo": "📝",
    "ready": "⏳",
    "running": "🔄",
    "blocked": "⏸",
    "done": "✅",
}


def _import_kanban_db():
    try:
        from hermes_cli import kanban_db as kb  # type: ignore

        return kb
    except ImportError as exc:
        raise RuntimeError(
            "hermes_cli.kanban_db not importable; run with Hermes venv active"
        ) from exc


def _trim(text: str, limit: int = 160) -> str:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t
    return t[: limit - 1].rstrip() + "…"


def _entity_id(session: dict, board: str) -> str:
    cfg = get_board_config(board)
    return str(session.get(cfg["entity_key"]) or "")


def _display_title(session: dict, board: str) -> str:
    if board == "stock-nexus":
        return session.get("stock_name") or session.get("title") or _entity_id(session, board)
    if board == "paper-search":
        return session.get("query_text") or session.get("title") or _entity_id(session, board)
    return session.get("title_zh") or session.get("title") or _entity_id(session, board)


def _dag_line(kb, conn, tasks: dict[str, str], stages: tuple[str, ...]) -> str:
    parts: list[str] = []
    for stage in stages:
        tid = (tasks or {}).get(stage)
        if not tid:
            continue
        row = kb.get_task(conn, tid)
        if not row:
            parts.append(f"{stage}?")
            continue
        icon = _STATUS_ICON.get(row.status, row.status)
        parts.append(f"{stage}{icon}")
    return " ".join(parts) if parts else "(no tasks)"


def render_message(
    session: dict,
    *,
    board: str,
    event: str,
    stage: str = "",
    summary: str = "",
    kb=None,
    conn=None,
) -> str:
    cfg = get_board_config(board)
    eid = _entity_id(session, board)
    title = _display_title(session, board)
    tasks = session.get("tasks") or {}
    labels = cfg["stage_labels"]
    lines = [
        f"{cfg['header_icon']} {cfg['header']} · [{eid}] {_trim(title, 72)}",
    ]

    ev = (event or "").strip().lower()
    st = (stage or "").strip()
    if board == "paper-nexus":
        st = st.upper()
    if ev == "pipeline_started":
        lines.append("流水线已建卡，Dispatcher 将按依赖派工。")
    elif ev == "stage_started" and st:
        lines.append(f"▶ {st} {labels.get(st, st)} 开始")
    elif ev == "stage_done" and st:
        lines.append(f"✔ {st} {labels.get(st, st)} 完成")
        if summary:
            lines.append(f"摘要：{_trim(summary, 200)}")
    elif ev == "stage_blocked" and st:
        lines.append(f"⏸ {st} {labels.get(st, st)} 阻塞")
        if summary:
            lines.append(f"原因：{_trim(summary, 200)}")
    elif ev == "pipeline_done":
        tail = "T_deep" if session.get("deep") else "T6"
        lines.append(f"✔ 流水线已收尾（至 {tail}）")
        if summary:
            lines.append(f"建议：{_trim(summary, 200)}")
    else:
        lines.append(f"更新：{ev or 'status'}")

    if kb is not None and conn is not None and tasks:
        lines.append(f"看板：{_dag_line(kb, conn, tasks, cfg['stages'])}")
    doc = session.get("feishu_doc_url") or ""
    if doc:
        lines.append(f"📎 文档：{doc}")
    return "\n".join(lines)


def send_feishu_text(
    chat_id: str,
    text: str,
    *,
    thread_id: str = "",
    as_identity: str = "bot",
    dry_run: bool = False,
) -> dict:
    if dry_run:
        return {"dry_run": True, "chat_id": chat_id, "thread_id": thread_id, "text": text}
    if thread_id:
        args = [
            "lark-cli",
            "im",
            "+messages-reply",
            "--as",
            as_identity,
            "--message-id",
            thread_id,
            "--reply-in-thread",
            "--text",
            text,
        ]
    else:
        args = [
            "lark-cli",
            "im",
            "+messages-send",
            "--as",
            as_identity,
            "--chat-id",
            chat_id,
            "--text",
            text,
        ]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"lark-cli im send failed: {err}")
    return {"ok": True, "chat_id": chat_id}


def _paper_lookup_doc(board: str, entity_id: str) -> str:
    if board != "paper-nexus":
        return ""
    try:
        _paper_dir = Path(__file__).resolve().parents[2] / "research" / "kanban-paper-nexus" / "scripts"
        if not _paper_dir.is_dir():
            _paper_dir = Path(__file__).resolve().parent
        if str(_paper_dir) not in sys.path:
            sys.path.insert(0, str(_paper_dir))
        from paper_doc_registry import lookup  # noqa: E402

        hit = lookup(entity_id, board)
        return (hit or {}).get("doc_url") or ""
    except Exception:
        return ""


def cmd_init(args: argparse.Namespace) -> int:
    board = args.board
    cfg = get_board_config(board)
    eid = args.entity_id.strip()
    if board == "paper-nexus":
        try:
            _paper_dir = Path(__file__).resolve().parents[2] / "research" / "kanban-paper-nexus" / "scripts"
            if _paper_dir.is_dir():
                sys.path.insert(0, str(_paper_dir))
                from paper_doc_registry import canonical_paper_id  # noqa: E402

                eid = canonical_paper_id(eid)
        except Exception:
            pass
    chat_id = (args.chat_id or os.environ.get("HERMES_SESSION_CHAT_ID") or "").strip()
    if not chat_id:
        print("chat_id required", file=sys.stderr)
        return 2
    tasks = {}
    if args.tasks_json:
        tasks = json.loads(Path(args.tasks_json).read_text(encoding="utf-8"))
    elif args.tasks_inline:
        tasks = json.loads(args.tasks_inline)
    extra = args.doc_url or _paper_lookup_doc(board, eid)
    session = new_session(
        board=board,
        entity_id=eid,
        title=args.title or "",
        chat_id=chat_id,
        thread_id=args.thread_id or os.environ.get("HERMES_SESSION_THREAD_ID") or "",
        platform=args.platform or "feishu",
        tasks=tasks,
        extra_url=extra,
        meta={
            "paper_id_latest": args.entity_id,
            "stock_name": args.stock_name or args.title,
            "deep": args.deep,
            "title_zh": args.title_zh or "",
        },
    )
    if args.title_zh:
        session["title_zh"] = args.title_zh.strip()
    path = save_session(session, board)
    out = {"ok": True, "session_path": str(path), "board": board, cfg["entity_key"]: eid}
    print(json.dumps(out, ensure_ascii=False))
    return 0


def cmd_notify(args: argparse.Namespace) -> int:
    board = args.board
    cfg = get_board_config(board)
    eid = (args.entity_id or "").strip()
    if board == "paper-nexus" and eid:
        try:
            _paper_dir = Path(__file__).resolve().parents[2] / "research" / "kanban-paper-nexus" / "scripts"
            if _paper_dir.is_dir():
                sys.path.insert(0, str(_paper_dir))
                from paper_doc_registry import canonical_paper_id  # noqa: E402

                eid = canonical_paper_id(eid)
        except Exception:
            pass
    session = load_session(eid, board)
    if session is None:
        print(f"no feishu session for {eid} on {board}; run init first", file=sys.stderr)
        return 1
    kb = _import_kanban_db()
    conn = kb.connect(board=board)
    try:
        msg = render_message(
            session,
            board=board,
            event=args.event,
            stage=args.stage,
            summary=args.summary or "",
            kb=kb,
            conn=conn,
        )
    finally:
        conn.close()
    if args.update_doc_url:
        session["feishu_doc_url"] = args.update_doc_url.strip()
        save_session(session, board)
    result = send_feishu_text(
        session["chat_id"],
        msg,
        thread_id=session.get("thread_id") or "",
        as_identity=args.as_identity,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "board": board,
                "event": args.event,
                "stage": args.stage,
                cfg["entity_key"]: eid,
                **result,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Kanban Feishu live IM (finance-nexus style)")
    p.add_argument("--board", default="paper-nexus", choices=tuple(BOARD_CONFIG.keys()))
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("entity_id", help="canonical arXiv id or 6-digit stock symbol")
    pi.add_argument("--chat-id", default="")
    pi.add_argument("--thread-id", default="")
    pi.add_argument("--platform", default="feishu")
    pi.add_argument("--title", default="", help="display title (stock name / EN title)")
    pi.add_argument("--title-zh", default="", help="paper Chinese title for session")
    pi.add_argument("--stock-name", default="")
    pi.add_argument("--deep", action="store_true")
    pi.add_argument("--doc-url", default="")
    pi.add_argument("--tasks-json", default="")
    pi.add_argument("--tasks-inline", default="")

    pn = sub.add_parser("notify")
    pn.add_argument("--entity-id", "--canonical-id", "--paper-id", "--symbol", default="")
    pn.add_argument("--event", required=True, choices=[
        "pipeline_started",
        "stage_started",
        "stage_done",
        "stage_blocked",
        "pipeline_done",
    ])
    pn.add_argument("--stage", default="")
    pn.add_argument("--summary", default="")
    pn.add_argument("--update-doc-url", default="")
    pn.add_argument("--as", dest="as_identity", default="bot")
    pn.add_argument("--dry-run", action="store_true")

    args = p.parse_args()
    if args.cmd == "init":
        return cmd_init(args)
    if args.cmd == "notify":
        return cmd_notify(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
