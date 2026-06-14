#!/usr/bin/env python3
"""X/Twitter draft + publish helpers with confirmation gates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.local_secretary.write_action_gate import check_write_action

MAX_TWEET_LENGTH = 280


def _count_chars(text: str, url: str | None = None) -> dict[str, Any]:
    base = len(text)
    url_len = 23 if url else 0
    total = base + (0 if not url else url_len)
    return {
        "text_chars": base,
        "url_weight": url_len,
        "total_weighted": total,
        "limit": MAX_TWEET_LENGTH,
        "within_limit": total <= MAX_TWEET_LENGTH,
    }


def x_draft_post(
    text: str,
    *,
    url: str | None = None,
    media_paths: list[str] | None = None,
    account: str | None = None,
) -> str:
    gate = check_write_action("x_draft_post")
    if not gate.ok:
        return json.dumps(gate.to_json())
    payload = {
        "success": True,
        "action": "x_draft_post",
        "text": text,
        "url": url,
        "media_paths": media_paths or [],
        "account": account or os.getenv("X_ACCOUNT_HANDLE", ""),
        "char_count": _count_chars(text, url),
        "dry_run": os.getenv("X_DRY_RUN", "").strip() in {"1", "true", "yes"},
    }
    return json.dumps(payload, ensure_ascii=False)


def x_publish_post(
    text: str,
    *,
    url: str | None = None,
    media_paths: list[str] | None = None,
    account: str | None = None,
    confirmed: bool = False,
    dry_run: bool = False,
) -> str:
    counts = _count_chars(text, url)
    preview = {
        "text": text,
        "url": url,
        "media_paths": media_paths or [],
        "account": account or os.getenv("X_ACCOUNT_HANDLE", ""),
        "char_count": counts,
    }
    gate = check_write_action(
        "x_publish_post",
        confirmed=confirmed,
        detail=json.dumps(preview, ensure_ascii=False),
    )
    if not gate.ok:
        out = gate.to_json()
        out["preview"] = preview
        return json.dumps(out, ensure_ascii=False)

    if dry_run or os.getenv("X_DRY_RUN", "").strip() in {"1", "true", "yes"}:
        return json.dumps(
            {
                "success": True,
                "action": "x_publish_post",
                "published": False,
                "dry_run": True,
                "preview": preview,
            },
            ensure_ascii=False,
        )

    missing = [
        name
        for name in (
            "X_API_KEY",
            "X_API_SECRET",
            "X_ACCESS_TOKEN",
            "X_ACCESS_TOKEN_SECRET",
        )
        if not os.getenv(name)
    ]
    if missing:
        return json.dumps(
            {
                "success": False,
                "action": "x_publish_post",
                "error": f"Missing credentials in .env: {', '.join(missing)}",
                "preview": preview,
            },
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "success": True,
            "action": "x_publish_post",
            "published": True,
            "dry_run": False,
            "preview": preview,
            "note": "Wire to xurl CLI or X API v2 in production deployments.",
        },
        ensure_ascii=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="X poster draft/publish helper")
    sub = parser.add_subparsers(dest="command", required=True)

    draft = sub.add_parser("draft", help="Create a draft payload (auto-allowed)")
    draft.add_argument("--text", required=True)
    draft.add_argument("--url", default="")
    draft.add_argument("--account", default="")
    draft.add_argument("--media", action="append", default=[])

    publish = sub.add_parser("publish", help="Publish with confirmation gate")
    publish.add_argument("--text", required=True)
    publish.add_argument("--url", default="")
    publish.add_argument("--account", default="")
    publish.add_argument("--media", action="append", default=[])
    publish.add_argument("--confirmed", action="store_true")
    publish.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.command == "draft":
        print(
            x_draft_post(
                args.text,
                url=args.url or None,
                media_paths=args.media,
                account=args.account or None,
            )
        )
        return 0
    print(
        x_publish_post(
            args.text,
            url=args.url or None,
            media_paths=args.media,
            account=args.account or None,
            confirmed=args.confirmed,
            dry_run=args.dry_run,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
