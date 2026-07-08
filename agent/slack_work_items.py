from __future__ import annotations

import fcntl
import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlparse

SourceType = Literal["thread", "canvas", "list", "file", "unknown"]

_SLACK_LINK_RE = re.compile(r"<(?P<url>https?://[^>|]+)(?:\|[^>]+)?>")
_ARCHIVE_RE = re.compile(r"/archives/(?P<channel>[A-Z0-9]+)/p(?P<pstamp>\d{16})")
_FILE_ID_RE = re.compile(r"\bF[A-Z0-9]{8,}\b")


@dataclass(frozen=True)
class SlackSourceRef:
    source_type: SourceType
    permalink: str
    channel_id: str | None
    message_ts: str | None
    thread_ts: str | None
    file_id: str | None


def _clean_link(text: str) -> str:
    raw = str(text or "").strip()
    match = _SLACK_LINK_RE.search(raw)
    if match:
        return unquote(match.group("url").replace("&amp;", "&"))
    return unquote(raw.replace("&amp;", "&"))


def _pstamp_to_ts(pstamp: str) -> str:
    return f"{pstamp[:10]}.{pstamp[10:]}"


def parse_slack_source_ref(text: str) -> SlackSourceRef | None:
    link = _clean_link(text)
    file_match = _FILE_ID_RE.search(link)
    archive_match = _ARCHIVE_RE.search(link)

    if archive_match:
        parsed = urlparse(link)
        query = parse_qs(parsed.query)
        channel_id = archive_match.group("channel")
        message_ts = _pstamp_to_ts(archive_match.group("pstamp"))
        thread_ts = (query.get("thread_ts") or [message_ts])[0]
        return SlackSourceRef(
            source_type="thread",
            permalink=link,
            channel_id=channel_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            file_id=file_match.group(0) if file_match else None,
        )

    if file_match:
        return SlackSourceRef(
            source_type="file",
            permalink=link if link.startswith("http") else "",
            channel_id=None,
            message_ts=None,
            thread_ts=None,
            file_id=file_match.group(0),
        )

    return None


def make_work_id(now: datetime | None = None, seed: str = "") -> str:
    dt = now or datetime.now(timezone.utc)
    stamp = dt.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(f"{dt.isoformat()}:{seed}".encode("utf-8")).hexdigest()[:12]
    return f"sw_{stamp}_{digest}"


@contextmanager
def _locked_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            yield handle
        finally:
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parse_jsonl_lines(lines: list[str]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


def load_work_items(path: Path) -> list[dict[str, Any]]:
    try:
        return _parse_jsonl_lines(path.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return []


def latest_work_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        work_id = item.get("work_id")
        if isinstance(work_id, str) and work_id:
            by_id[work_id] = item
    return list(by_id.values())


def append_work_item(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    record = dict(item)
    with _locked_file(path) as handle:
        handle.seek(0, 2)
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def find_work_item(items: list[dict[str, Any]], ref: SlackSourceRef) -> dict[str, Any] | None:
    for item in reversed(latest_work_items(items)):
        if ref.file_id and item.get("file_id") == ref.file_id:
            return item
        if ref.channel_id and ref.thread_ts:
            if item.get("channel_id") == ref.channel_id and item.get("thread_ts") == ref.thread_ts:
                return item
        if ref.permalink and item.get("source_permalink") == ref.permalink:
            return item
    return None


def update_work_item(path: Path, work_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    current = None
    for item in reversed(latest_work_items(load_work_items(path))):
        if item.get("work_id") == work_id:
            current = item
            break
    if current is None:
        raise KeyError(work_id)
    merged = dict(current)
    merged.update(updates)
    return append_work_item(path, merged)
