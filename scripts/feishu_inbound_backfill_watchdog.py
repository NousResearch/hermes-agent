#!/usr/bin/env python3
"""Feishu inbound backfill watchdog.

Polls recent Feishu chat history and compares message_id values with Hermes
local evidence (gateway logs + Feishu dedup state). It is intentionally
read-only: it reports suspected websocket misses but does not replay messages.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def _home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()

def _dedup_path() -> Path:
    return _home() / "feishu_seen_message_ids.json"

def _env_path() -> Path:
    return _home() / ".env"

def _state_path() -> Path:
    return _home() / "state" / "feishu_inbound_backfill_watchdog.json"

def _queue_path() -> Path:
    return _home() / "state" / "feishu_inbound_backfill_queue.jsonl"

MESSAGE_ID_RE = re.compile(r"\bom_[A-Za-z0-9_\-]+")
DEFAULT_WINDOW_SECONDS = 15 * 60
DEFAULT_GRACE_SECONDS = 45
DEFAULT_PAGE_SIZE = 50
RECALLED_PREVIEWS = {"this message was recalled", "message was recalled", "消息已撤回"}


def _load_env(path: Path | None = None) -> None:
    path = path or _env_path()
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _domain_base() -> str:
    return "https://open.larksuite.com" if os.getenv("FEISHU_DOMAIN", "feishu").strip().lower() == "lark" else "https://open.feishu.cn"


def _request_json(method: str, url: str, *, token: str | None = None, body: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _tenant_token() -> str:
    app_id = os.getenv("FEISHU_APP_ID", "").strip()
    app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise RuntimeError("FEISHU_APP_ID/FEISHU_APP_SECRET missing")
    payload = _request_json(
        "POST",
        f"{_domain_base()}/open-apis/auth/v3/tenant_access_token/internal",
        body={"app_id": app_id, "app_secret": app_secret},
    )
    if payload.get("code") != 0 or not payload.get("tenant_access_token"):
        raise RuntimeError(f"tenant token request failed: code={payload.get('code')} msg={payload.get('msg')}")
    return str(payload["tenant_access_token"])


def _load_log_ids() -> set[str]:
    ids: set[str] = set()
    for path in _home().glob("logs/gateway.log*"):
        if not path.is_file():
            continue
        # Bound read volume per file. Rotated gateway logs may hold the relevant window.
        with path.open("rb") as fh:
            try:
                fh.seek(-5 * 1024 * 1024, os.SEEK_END)
            except OSError:
                fh.seek(0)
            text = fh.read().decode("utf-8", errors="replace")
        ids.update(MESSAGE_ID_RE.findall(text))
    return ids


def _load_dedup_ids(path: Path | None = None) -> set[str]:
    path = path or _dedup_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()
    data = payload.get("message_ids", {}) if isinstance(payload, dict) else {}
    if isinstance(data, dict):
        return {str(k) for k in data.keys() if str(k).startswith("om_")}
    if isinstance(data, list):
        return {str(k) for k in data if str(k).startswith("om_")}
    return set()


def _load_state() -> dict[str, Any]:
    try:
        return json.loads(_state_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"reported": {}}


def _save_state(state: dict[str, Any]) -> None:
    state_path = _state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_path)


def _list_messages(token: str, chat_id: str, start_time: int, end_time: int, page_size: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token = ""
    while True:
        query = {
            "container_id_type": "chat",
            "container_id": chat_id,
            "start_time": str(start_time),
            "end_time": str(end_time),
            "page_size": str(page_size),
        }
        if page_token:
            query["page_token"] = page_token
        payload = _request_json(
            "GET",
            f"{_domain_base()}/open-apis/im/v1/messages?{urlencode(query)}",
            token=token,
        )
        if payload.get("code") != 0:
            raise RuntimeError(f"list messages failed: code={payload.get('code')} msg={payload.get('msg')}")
        data = payload.get("data") or {}
        batch = data.get("items") or []
        if isinstance(batch, list):
            items.extend(x for x in batch if isinstance(x, dict))
        if not data.get("has_more"):
            break
        page_token = str(data.get("page_token") or "")
        if not page_token:
            break
    return items


def _message_id(item: dict[str, Any]) -> str:
    return str(item.get("message_id") or item.get("messageId") or "")


def _sender_type(item: dict[str, Any]) -> str:
    sender = item.get("sender") or {}
    return str(sender.get("sender_type") or sender.get("type") or "")


def _message_create_time(item: dict[str, Any]) -> int:
    raw = item.get("create_time") or item.get("update_time") or 0
    try:
        # Feishu often returns milliseconds as a string.
        val = int(str(raw))
        return val // 1000 if val > 10_000_000_000 else val
    except (TypeError, ValueError):
        return 0


def _preview(item: dict[str, Any], limit: int = 80) -> str:
    content = item.get("body", {}).get("content") or item.get("content") or ""
    text = ""
    try:
        parsed = json.loads(content) if isinstance(content, str) else content
        if isinstance(parsed, dict):
            text = parsed.get("text") or parsed.get("title") or json.dumps(parsed, ensure_ascii=False)
        else:
            text = str(parsed)
    except Exception:
        text = str(content)
    text = " ".join(text.split())
    return text[:limit]


def _is_recalled_message(item: dict[str, Any]) -> bool:
    msg_type = str(item.get("msg_type") or item.get("message_type") or "").lower()
    if "recall" in msg_type or "withdraw" in msg_type:
        return True
    preview = _preview(item, limit=120).strip().lower()
    return preview in RECALLED_PREVIEWS


def _queue_record(item: dict[str, Any], *, chat_id: str, detected_at: int) -> dict[str, Any]:
    return {
        "detected_at": detected_at,
        "chat_id": chat_id,
        "message_id": _message_id(item),
        "create_time": _message_create_time(item),
        "sender_type": _sender_type(item) or "<unknown>",
        "msg_type": str(item.get("msg_type") or item.get("message_type") or ""),
        "preview": _preview(item, limit=240),
    }


def _append_queue(records: list[dict[str, Any]], path: Path | None = None) -> None:
    path = path or _queue_path()
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only Feishu/Lark inbound backfill watchdog")
    parser.add_argument("--chat-id", default=os.getenv("FEISHU_HOME_CHANNEL", ""), help="Feishu chat/container id")
    parser.add_argument("--window-seconds", type=int, default=int(os.getenv("FEISHU_BACKFILL_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS)))
    parser.add_argument("--grace-seconds", type=int, default=int(os.getenv("FEISHU_BACKFILL_GRACE_SECONDS", DEFAULT_GRACE_SECONDS)))
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--include-bots", action="store_true", help="Also check app/bot-originated messages")
    parser.add_argument("--include-recalled", action="store_true", help="Also report recalled/withdrawn messages")
    parser.add_argument("--no-queue", action="store_true", help="Do not append suspected misses to the local review queue")
    parser.add_argument("--dry-run", action="store_true", help="Print OK summary even when no misses are found")
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_env()
    chat_id = (args.chat_id or os.getenv("FEISHU_HOME_CHANNEL", "")).strip()
    if not chat_id:
        raise RuntimeError("chat id missing; set FEISHU_HOME_CHANNEL or pass --chat-id")

    now = int(time.time())
    start_time = now - max(args.window_seconds, 60)
    end_time = now - max(args.grace_seconds, 0)
    if end_time <= start_time:
        raise RuntimeError("window too small after grace period")

    token = _tenant_token()
    messages = _list_messages(token, chat_id, start_time, end_time, max(1, min(args.page_size, 50)))
    log_ids = _load_log_ids()
    dedup_ids = _load_dedup_ids()
    seen_ids = log_ids | dedup_ids

    state = _load_state()
    reported = state.setdefault("reported", {})
    # Prune old report suppression entries.
    cutoff = now - 24 * 60 * 60
    state["reported"] = {
        mid: ts for mid, ts in reported.items()
        if isinstance(ts, int) and ts >= cutoff
    }
    reported = state["reported"]

    checked = 0
    missing: list[dict[str, Any]] = []
    for item in messages:
        mid = _message_id(item)
        if not mid:
            continue
        stype = _sender_type(item).lower()
        if not args.include_bots and stype in {"app", "bot"}:
            continue
        if not args.include_recalled and _is_recalled_message(item):
            continue
        checked += 1
        if mid not in seen_ids and mid not in reported:
            missing.append(item)
            reported[mid] = now

    state["last_run"] = now
    state["last_window"] = {"chat_id": chat_id, "start_time": start_time, "end_time": end_time, "checked": checked, "fetched": len(messages)}
    _save_state(state)

    queue_records = [_queue_record(item, chat_id=chat_id, detected_at=now) for item in missing]
    if not args.no_queue:
        _append_queue(queue_records)

    if missing:
        print("Feishu inbound watchdog: suspected missed message(s)")
        queue_note = " disabled" if args.no_queue else f" appended={len(queue_records)} path={_queue_path()}"
        print(f"chat_id={chat_id} window={start_time}..{end_time} fetched={len(messages)} checked={checked} missing={len(missing)} queue{queue_note}")
        for record in queue_records[:10]:
            print(
                f"- id={record['message_id']} create_time={record['create_time']} "
                f"sender_type={record['sender_type']} preview={record['preview']!r}"
            )
        if len(missing) > 10:
            print(f"... {len(missing) - 10} more")
        # Cron semantics: suspected misses are an alert condition, not a script
        # failure. Keep stdout non-empty so no_agent cron delivers the alert,
        # but reserve non-zero exit codes for API/config/runtime failures.
        return 0

    if args.dry_run:
        print(f"Feishu inbound watchdog OK: chat_id={chat_id} fetched={len(messages)} checked={checked} window={start_time}..{end_time}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
        print(f"Feishu inbound watchdog ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
