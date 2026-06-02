#!/usr/bin/env python3
"""Send Feishu chat messages as rich post (markdown) via lark-cli — tables render correctly.

Hermes core send() downgrades markdown tables to plain text. Use this for important
reports with headings, lists, and tables. Does not modify Hermes gateway code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Feishu post body practical limit; lark-cli may split — keep one message reasonable.
_MAX_CHARS = 24_000


def _read_markdown(args: argparse.Namespace) -> str:
    if args.markdown is not None:
        return args.markdown
    if args.markdown_file:
        return Path(args.markdown_file).read_text(encoding="utf-8")
    if args.markdown_stdin:
        return sys.stdin.read()
    raise SystemExit("Provide --markdown, --markdown-file, or --markdown-stdin")


def _resolve_chat_id(args: argparse.Namespace) -> str:
    chat_id = (args.chat_id or os.environ.get("HERMES_SESSION_CHAT_ID") or "").strip()
    if not chat_id:
        raise SystemExit("chat_id required: --chat-id or env HERMES_SESSION_CHAT_ID")
    return chat_id


def _resolve_thread_id(args: argparse.Namespace) -> str:
    return (
        args.thread_id
        or os.environ.get("HERMES_SESSION_THREAD_ID")
        or os.environ.get("HERMES_SESSION_MESSAGE_ID")
        or ""
    ).strip()


def build_lark_args(
    *,
    chat_id: str,
    markdown: str,
    thread_id: str = "",
    as_identity: str = "bot",
) -> list[str]:
    base = ["lark-cli", "im", "--as", as_identity]
    if thread_id:
        return [
            *base,
            "+messages-reply",
            "--message-id",
            thread_id,
            "--reply-in-thread",
            "--markdown",
            markdown,
        ]
    return [
        *base,
        "+messages-send",
        "--chat-id",
        chat_id,
        "--markdown",
        markdown,
    ]


def send_markdown(
    chat_id: str,
    markdown: str,
    *,
    thread_id: str = "",
    as_identity: str = "bot",
    dry_run: bool = False,
) -> dict:
    body = markdown.strip()
    if not body:
        raise ValueError("empty markdown body")
    if len(body) > _MAX_CHARS:
        raise ValueError(
            f"markdown too long ({len(body)} chars > {_MAX_CHARS}); "
            "split into parts or attach a doc link"
        )
    args = build_lark_args(
        chat_id=chat_id,
        markdown=body,
        thread_id=thread_id,
        as_identity=as_identity,
    )
    if dry_run:
        return {"dry_run": True, "argv": args, "chars": len(body)}
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"lark-cli failed: {err}")
    raw = proc.stdout.strip()
    start = raw.find("{")
    if start >= 0:
        return json.loads(raw[start:])
    return {"ok": True, "raw": raw}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send Feishu rich markdown (post) via lark-cli; tables supported."
    )
    parser.add_argument("--chat-id", help="oc_xxx; default HERMES_SESSION_CHAT_ID")
    parser.add_argument(
        "--thread-id",
        help="om_xxx to reply in thread; default HERMES_SESSION_THREAD_ID",
    )
    parser.add_argument("--as", dest="as_identity", default="bot", choices=("bot", "user"))
    parser.add_argument("--markdown", help="Markdown body (use $'...' in shell)")
    parser.add_argument("--markdown-file", type=Path, help="Read markdown from file")
    parser.add_argument(
        "--markdown-stdin",
        action="store_true",
        help="Read markdown from stdin",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    chat_id = _resolve_chat_id(args)
    thread_id = _resolve_thread_id(args)
    body = _read_markdown(args)
    try:
        out = send_markdown(
            chat_id,
            body,
            thread_id=thread_id,
            as_identity=args.as_identity,
            dry_run=args.dry_run,
        )
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
