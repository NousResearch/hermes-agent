#!/usr/bin/env python3
"""Render and publish the Slack companion surface for payments ops."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python"
if (
    not os.environ.get("VIRTUAL_ENV")
    and VENV_PYTHON.exists()
    and Path(sys.executable) != VENV_PYTHON
):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__, *sys.argv[1:]])

from hermes_cli import payments as payments_store
from hermes_cli.config import load_env

MOBILE_INBOX_TEXT = "Payments mobile inbox"


def _env_value(name: str) -> str:
    try:
        env = load_env()
    except Exception:
        env = {}
    return str(os.environ.get(name) or env.get(name) or "").strip()


def _default_dashboard_url() -> str:
    return _env_value("HERMES_PAYMENTS_DASHBOARD_URL")


def _slack_token() -> str:
    token = _env_value("SLACK_BOT_TOKEN")
    if not token:
        raise SystemExit("SLACK_BOT_TOKEN is not configured")
    return token


def _optional_slack_token() -> str:
    return _env_value("SLACK_BOT_TOKEN")


def render_canvas_spec(args: argparse.Namespace) -> int:
    print(
        payments_store.render_slack_canvas_spec(
            channel_key=args.channel_key,
            channel_name=args.channel_name,
            canvas_title=args.canvas_title,
            dashboard_url=args.dashboard_url,
            per_status_limit=args.per_status_limit,
        ),
        end="",
    )
    return 0


def publish_canvas(args: argparse.Namespace) -> int:
    spec = payments_store.render_slack_canvas_spec(
        channel_key=args.channel_key,
        channel_name=args.channel_name,
        canvas_title=args.canvas_title,
        dashboard_url=args.dashboard_url,
        per_status_limit=args.per_status_limit,
    )
    wrapper = Path(__file__).resolve().parent / "slack-surfaces-run.sh"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        handle.write(spec)
        temp_path = Path(handle.name)
    try:
        cmd = [str(wrapper), "sync", "--spec", str(temp_path), "--apply"]
        subprocess.run(cmd, check=True)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
    return 0


def _mobile_blocks(args: argparse.Namespace) -> list[dict[str, Any]]:
    return payments_store.build_slack_mobile_blocks(
        statuses=tuple(args.statuses),
        per_status_limit=args.per_status_limit,
    )


def _find_existing_mobile_inbox_ts(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if str(message.get("text") or "").strip() == MOBILE_INBOX_TEXT:
            ts = str(message.get("ts") or "").strip()
            if ts:
                return ts
    return ""


async def _post_mobile_inbox_async(args: argparse.Namespace) -> int:
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=_slack_token())
    blocks = _mobile_blocks(args)
    result = await client.chat_postMessage(
        channel=args.channel,
        text=MOBILE_INBOX_TEXT,
        blocks=blocks,
    )
    ts = result.get("ts") if isinstance(result, dict) else None
    print(f"posted payments mobile inbox to {args.channel} ts={ts or 'unknown'}")
    return 0


def post_mobile_inbox(args: argparse.Namespace) -> int:
    return asyncio.run(_post_mobile_inbox_async(args))


async def _upsert_mobile_inbox_async(args: argparse.Namespace) -> int:
    from slack_sdk.web.async_client import AsyncWebClient

    client = AsyncWebClient(token=_slack_token())
    blocks = _mobile_blocks(args)
    history = await client.conversations_history(channel=args.channel, limit=20)
    messages = history.get("messages", []) if isinstance(history, dict) else []
    existing_ts = _find_existing_mobile_inbox_ts(messages)
    if existing_ts:
        result = await client.chat_update(
            channel=args.channel,
            ts=existing_ts,
            text=MOBILE_INBOX_TEXT,
            blocks=blocks,
        )
        ts = result.get("ts") if isinstance(result, dict) else existing_ts
        print(f"updated payments mobile inbox in {args.channel} ts={ts or existing_ts}")
        return 0

    return await _post_mobile_inbox_async(args)


def upsert_mobile_inbox(args: argparse.Namespace) -> int:
    return asyncio.run(_upsert_mobile_inbox_async(args))


def _sync_gmail(args: argparse.Namespace) -> dict[str, Any]:
    return payments_store.sync_gmail_payment_requests(
        query=args.query,
        max_results=args.max_results,
    )


def _print_sync_summary(result: dict[str, Any]) -> None:
    print(payments_store.format_sync_summary(result))


async def _sync_and_upsert_async(args: argparse.Namespace) -> int:
    result = _sync_gmail(args)
    _print_sync_summary(result)

    if not args.channel.strip():
        print("skipped slack inbox refresh: no channel configured")
        return 0
    if not _optional_slack_token():
        print("skipped slack inbox refresh: SLACK_BOT_TOKEN is not configured")
        return 0

    return await _upsert_mobile_inbox_async(args)


def sync_gmail(args: argparse.Namespace) -> int:
    _print_sync_summary(_sync_gmail(args))
    return 0


def sync_and_upsert(args: argparse.Namespace) -> int:
    return asyncio.run(_sync_and_upsert_async(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    canvas = sub.add_parser("canvas-spec", help="Render a slack-surfaces YAML spec")
    canvas.add_argument("--channel-key", default="payments-ops")
    canvas.add_argument("--channel-name", default="payments-ops")
    canvas.add_argument("--canvas-title", default="Payments Ops")
    canvas.add_argument("--dashboard-url", default=_default_dashboard_url())
    canvas.add_argument("--per-status-limit", type=int, default=8)
    canvas.set_defaults(func=render_canvas_spec)

    publish = sub.add_parser("publish-canvas", help="Publish the canvas via slack-surfaces")
    publish.add_argument("--channel-key", default="payments-ops")
    publish.add_argument("--channel-name", default="payments-ops")
    publish.add_argument("--canvas-title", default="Payments Ops")
    publish.add_argument("--dashboard-url", default=_default_dashboard_url())
    publish.add_argument("--per-status-limit", type=int, default=8)
    publish.set_defaults(func=publish_canvas)

    inbox = sub.add_parser("post-mobile-inbox", help="Post the compact mobile action inbox")
    inbox.add_argument("--channel", required=True, help="Slack channel or DM ID")
    inbox.add_argument(
        "--statuses",
        nargs="+",
        default=["needs_review", "ready_to_pay", "paid"],
        choices=["needs_review", "ready_to_pay", "paid", "ignored"],
    )
    inbox.add_argument("--per-status-limit", type=int, default=5)
    inbox.set_defaults(func=post_mobile_inbox)

    upsert = sub.add_parser(
        "upsert-mobile-inbox",
        help="Update the latest compact mobile action inbox in-place, or post it if missing",
    )
    upsert.add_argument("--channel", required=True, help="Slack channel or DM ID")
    upsert.add_argument(
        "--statuses",
        nargs="+",
        default=["needs_review", "ready_to_pay", "paid"],
        choices=["needs_review", "ready_to_pay", "paid", "ignored"],
    )
    upsert.add_argument("--per-status-limit", type=int, default=5)
    upsert.set_defaults(func=upsert_mobile_inbox)

    sync = sub.add_parser(
        "sync-gmail",
        help="Sync Gmail payment candidates into the canonical payments review store",
    )
    sync.add_argument("--query", default=payments_store.DEFAULT_GMAIL_QUERY)
    sync.add_argument("--max-results", type=int, default=payments_store.DEFAULT_GMAIL_MAX_RESULTS)
    sync.set_defaults(func=sync_gmail)

    sync_upsert = sub.add_parser(
        "sync-and-upsert",
        help="Sync Gmail into the canonical store and refresh the Slack mobile inbox when credentials are available",
    )
    sync_upsert.add_argument("--channel", default="", help="Slack channel or DM ID")
    sync_upsert.add_argument("--query", default=payments_store.DEFAULT_GMAIL_QUERY)
    sync_upsert.add_argument("--max-results", type=int, default=payments_store.DEFAULT_GMAIL_MAX_RESULTS)
    sync_upsert.add_argument(
        "--statuses",
        nargs="+",
        default=["needs_review", "ready_to_pay", "paid"],
        choices=["needs_review", "ready_to_pay", "paid", "ignored"],
    )
    sync_upsert.add_argument("--per-status-limit", type=int, default=5)
    sync_upsert.set_defaults(func=sync_and_upsert)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
