#!/usr/bin/env python3
"""CLI helper for Hermes numbered mobile replies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_profile_env() -> None:
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(
            hermes_home=Path(os.environ["HERMES_HOME"]) if os.environ.get("HERMES_HOME") else None,
            project_env=PROJECT_ROOT / ".env",
        )
    except Exception:
        return


def _print_json(data) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _send_message(target: str, message: str) -> dict:
    _load_profile_env()
    from tools.send_message_tool import send_message_tool

    raw = send_message_tool({"action": "send", "target": target, "message": message})
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"raw": parsed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes mobile ask/reply/watch bridge")
    parser.add_argument("--hermes-home", help="Override HERMES_HOME for this command")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Create a numbered mobile ask card")
    ask.add_argument("--title", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--source-kind", default="codex")
    ask.add_argument("--source-session")
    ask.add_argument("--deliver", help="Optional send target, e.g. weixin or weixin:filehelper")
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    reply = sub.add_parser("reply", help="Record a reply text manually")
    reply.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Reply text. Quotes are optional; extra words are joined with spaces.",
    )
    reply.add_argument("--platform", default="manual")
    reply.add_argument("--chat-id")
    reply.add_argument("--user-id")
    reply.add_argument("--json", action="store_true")

    watch = sub.add_parser("watch", help="Wait for a reply to a request id")
    watch.add_argument("request_id")
    watch.add_argument("--timeout", type=float, default=0)
    watch.add_argument("--poll", type=float, default=2)
    watch.add_argument("--json", action="store_true")

    show = sub.add_parser("show", help="Show latest reply for a request id")
    show.add_argument("request_id")
    show.add_argument("--json", action="store_true")

    paths = sub.add_parser("paths", help="Show bridge storage paths")
    paths.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.hermes_home:
        os.environ["HERMES_HOME"] = args.hermes_home

    from gateway.mobile_reply_bridge import (
        bridge_paths,
        create_mobile_ask,
        format_mobile_ask_card,
        handle_mobile_reply_text,
        load_latest_reply,
        wait_for_mobile_reply,
    )

    if args.command == "ask":
        record = create_mobile_ask(
            title=args.title,
            question=args.question,
            source_kind=args.source_kind,
            source_session=args.source_session,
        )
        card = format_mobile_ask_card(record)
        output = {"request": record, "card": card}
        if args.deliver:
            output["delivery"] = _send_message(args.deliver, card)
        if args.json:
            _print_json(output)
        else:
            print(card)
            if args.deliver:
                print()
                print("delivery:")
                _print_json(output["delivery"])
        return 0

    if args.command == "reply":
        reply_text = " ".join(args.text)
        result = handle_mobile_reply_text(
            reply_text,
            platform=args.platform,
            chat_id=args.chat_id,
            user_id=args.user_id,
        )
        if result is None:
            print("not a mobile numbered reply")
            return 1
        payload = {
            "action": result.action,
            "request_id": result.request_id,
            "reply_path": str(result.reply_path) if result.reply_path else None,
            "message": result.message,
        }
        if args.json:
            _print_json(payload)
        else:
            print(result.message)
            if result.reply_path:
                print(result.reply_path)
        return 0 if result.action == "recorded" else 2

    if args.command == "watch":
        data = wait_for_mobile_reply(
            args.request_id,
            timeout_seconds=args.timeout,
            poll_seconds=args.poll,
        )
        if data is None:
            print("no reply")
            return 1
        if args.json:
            _print_json(data)
        else:
            print(data.get("content", ""))
        return 0

    if args.command == "show":
        data = load_latest_reply(args.request_id)
        if data is None:
            print("no reply")
            return 1
        if args.json:
            _print_json(data)
        else:
            print(data.get("content", ""))
        return 0

    if args.command == "paths":
        payload = {name: str(path) for name, path in bridge_paths().items()}
        if args.json:
            _print_json(payload)
        else:
            for name, path in payload.items():
                print(f"{name}: {path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
