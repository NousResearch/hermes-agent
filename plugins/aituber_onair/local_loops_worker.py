"""Local autonomous talk and comment reaction loops for Hakua."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import core


def _log(payload: dict[str, Any]) -> None:
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **payload}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _autonomous_prompt(topic: str) -> str:
    context = topic.strip() or "Galaxy S9に表示されているVRM配信画面"
    return (
        "配信の沈黙を避けるため、AIVTuberとして短く自然に一言話してください。\n"
        "コメントがない時の自律発話なので、視聴者に圧をかけず、落ち着いてください。\n"
        f"現在の話題: {context}"
    )


def _comment_prompt(comment: dict[str, Any]) -> str:
    author = str(comment.get("author") or "viewer")
    text = str(comment.get("text") or "")
    source = str(comment.get("source") or "local")
    return (
        "配信コメントにAIVTuberとして短く反応してください。\n"
        "日本語で、コメント内容を踏まえ、自然に一言返してください。\n"
        f"source: {source}\n"
        f"author: {author}\n"
        f"comment: {text}"
    )


def _read_queue(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    comments: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("text"):
            comments.append(item)
    return comments


def _read_seen_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    ids = data.get("ids") if isinstance(data, dict) else []
    if not isinstance(ids, list):
        return set()
    return {str(item) for item in ids if str(item)}


def _write_seen_ids(path: Path | None, seen_ids: set[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"ids": sorted(seen_ids)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def _reply(prompt: str, *, play: bool) -> dict[str, Any]:
    return core.run_hakua_once(
        {
            "prompt": prompt,
            "reply_backend": "hermes",
            "with_runtime_context": True,
            "speak": True,
            "play": play,
            "tts_provider": "auto",
        }
    )


def _backoff_seconds(result: dict[str, Any], default_seconds: float) -> float:
    detail = " ".join(
        str(result.get(key) or "") for key in ("error", "provider_error", "stderr")
    ).lower()
    if "usage limit" in detail or "try again" in detail or "rate limit" in detail:
        return max(default_seconds, 3600.0)
    return default_seconds


def run_autonomous(args: argparse.Namespace) -> int:
    _log(
        {
            "event": "started",
            "mode": "autonomous",
            "interval_seconds": args.interval_seconds,
            "topic": args.topic,
            "play": args.play,
        }
    )
    while True:
        try:
            result = _reply(_autonomous_prompt(args.topic), play=args.play)
            _log(
                {
                    "event": "autonomous_reply",
                    "ok": result.get("ok"),
                    "reply": result.get("reply"),
                    "tts_ok": (result.get("tts") or {}).get("ok")
                    if isinstance(result.get("tts"), dict)
                    else None,
                    "error": result.get("error"),
                    "provider_error": result.get("provider_error"),
                    "fallback_from_model": result.get("fallback_from_model"),
                }
            )
            sleep_seconds = _backoff_seconds(result, max(10.0, args.interval_seconds))
        except KeyboardInterrupt:
            _log({"event": "stopped", "mode": "autonomous"})
            return 0
        except Exception as exc:
            _log({"event": "loop_error", "mode": "autonomous", "error": str(exc)})
            sleep_seconds = max(10.0, args.interval_seconds)
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            _log({"event": "stopped", "mode": "autonomous"})
            return 0


def run_comments(args: argparse.Namespace) -> int:
    queue_file = Path(args.queue_file)
    processed_file = Path(args.processed_file) if args.processed_file else None
    _log(
        {
            "event": "started",
            "mode": "comments",
            "queue_file": str(queue_file),
            "processed_file": str(processed_file) if processed_file else "",
            "poll_seconds": args.poll_seconds,
            "play": args.play,
        }
    )
    seen_ids = _read_seen_ids(processed_file)
    while True:
        try:
            comments = _read_queue(queue_file)
            for comment in comments:
                comment_id = str(comment.get("id") or "")
                if comment_id and comment_id in seen_ids:
                    continue
                if comment_id:
                    seen_ids.add(comment_id)
                _log(
                    {
                        "event": "comment_selected",
                        "id": comment_id,
                        "author": comment.get("author"),
                        "text": comment.get("text"),
                    }
                )
                result = _reply(_comment_prompt(comment), play=args.play)
                _log(
                    {
                        "event": "comment_reply",
                        "ok": result.get("ok"),
                        "reply": result.get("reply"),
                        "tts_ok": (result.get("tts") or {}).get("ok")
                        if isinstance(result.get("tts"), dict)
                        else None,
                        "error": result.get("error"),
                        "provider_error": result.get("provider_error"),
                        "fallback_from_model": result.get("fallback_from_model"),
                    }
                )
                if comment_id:
                    seen_ids.add(comment_id)
                    _write_seen_ids(processed_file, seen_ids)
                if not result.get("ok"):
                    time.sleep(_backoff_seconds(result, 0.0))
        except KeyboardInterrupt:
            _log({"event": "stopped", "mode": "comments"})
            return 0
        except Exception as exc:
            _log({"event": "loop_error", "mode": "comments", "error": str(exc)})
        try:
            time.sleep(max(1.0, args.poll_seconds))
        except KeyboardInterrupt:
            _log({"event": "stopped", "mode": "comments"})
            return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["autonomous", "comments"], required=True)
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--queue-file", default="")
    parser.add_argument("--processed-file", default="")
    parser.add_argument("--topic", default="")
    parser.add_argument("--play", action="store_true")
    args = parser.parse_args()
    if args.mode == "autonomous":
        return run_autonomous(args)
    if not args.queue_file:
        parser.error("--queue-file is required for comments mode")
    return run_comments(args)


if __name__ == "__main__":
    raise SystemExit(main())
