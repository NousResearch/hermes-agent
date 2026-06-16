"""Hermes-side YouTube Live comment monitor for Hakua."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from . import core


def _log(payload: dict[str, Any]) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _comment_prompt(comment: dict[str, str]) -> str:
    author = comment.get("author") or "viewer"
    text = comment.get("text") or ""
    return (
        "YouTube Liveの視聴者コメントに、はくあとして短く自然に返答してください。\n"
        f"投稿者: {author}\n"
        f"コメント: {text}"
    )


def run(args: argparse.Namespace) -> int:
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        _log({"event": "fatal", "error": "api key env is not set", "env": args.api_key_env})
        return 2

    try:
        live_chat_id = core.youtube_live_chat_id(args.live_id, api_key)
    except Exception as exc:
        _log(
            {
                "event": "fatal",
                "error": str(exc),
                "live_id": args.live_id,
                "api_key_env": args.api_key_env,
            }
        )
        return 3
    if not live_chat_id:
        _log(
            {
                "event": "fatal",
                "error": "active live chat id was not found",
                "live_id": args.live_id,
            }
        )
        return 4

    _log(
        {
            "event": "started",
            "live_id": args.live_id,
            "api_key_env": args.api_key_env,
            "poll_seconds": args.poll_seconds,
            "skip_existing": args.skip_existing,
            "play": args.play,
        }
    )

    seen_ids: set[str] = set()
    next_page_token = ""
    first_poll = True

    while True:
        try:
            result = core.fetch_youtube_live_comments(
                live_chat_id=live_chat_id,
                api_key=api_key,
                page_token=next_page_token,
            )
            if not result.get("ok"):
                _log({"event": "fetch_error", "error": result.get("error")})
                time.sleep(args.poll_seconds)
                continue

            next_page_token = str(result.get("next_page_token") or "")
            comments = [
                item
                for item in result.get("comments", [])
                if isinstance(item, dict) and item.get("id") not in seen_ids
            ]
            for item in result.get("comments", []):
                if isinstance(item, dict) and item.get("id"):
                    seen_ids.add(str(item["id"]))

            if first_poll and args.skip_existing:
                _log({"event": "seeded_existing_comments", "count": len(comments)})
                comments = []
            first_poll = False

            if comments:
                selected = comments[-1]
                _log(
                    {
                        "event": "comment_selected",
                        "id": selected.get("id"),
                        "author": selected.get("author"),
                        "text": selected.get("text"),
                    }
                )
                reply = core.run_hakua_once(
                    {
                        "prompt": _comment_prompt(selected),
                        "speak": True,
                        "play": args.play,
                        "tts_provider": "auto",
                    }
                )
                _log(
                    {
                        "event": "hakua_reply",
                        "ok": reply.get("ok"),
                        "reply": reply.get("reply"),
                        "tts_ok": (reply.get("tts") or {}).get("ok")
                        if isinstance(reply.get("tts"), dict)
                        else None,
                        "error": reply.get("error"),
                    }
                )

            api_interval_ms = int(result.get("polling_interval_ms") or 0)
            sleep_seconds = max(args.poll_seconds, api_interval_ms / 1000)
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            _log({"event": "stopped"})
            return 0
        except Exception as exc:
            _log({"event": "loop_error", "error": str(exc)})
            time.sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-id", required=True)
    parser.add_argument("--api-key-env", default="AITUBER_ONAIR_YOUTUBE_API_KEY")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--play", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
