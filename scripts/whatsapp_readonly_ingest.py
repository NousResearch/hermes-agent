#!/usr/bin/env python3
"""Read-only WhatsApp ingestion for daily digests.

This script consumes inbound events from Hermes' local WhatsApp Baileys bridge
(`GET /messages`), filters them by group allowlist, and appends only the allowed
messages to a JSONL file for a later summarization cron job.

Security model:
- This process never calls bridge outbound endpoints (/send, /send-media, /edit,
  /typing). It only performs HTTP GET requests to /messages.
- DMs are ignored by default.
- Group ingestion is deny-by-default: `group_allowlist` must be non-empty unless
  `group_policy: open` is explicitly configured.
- Filtering happens before persistence and before any LLM sees content.

This is intentionally a narrow ingestor, not a Hermes gateway adapter. It keeps
WhatsApp collection separate from the normal bot/reply path.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_CONFIG_PATH = Path.home() / ".hermes" / "whatsapp-readonly-ingest.yaml"
DEFAULT_OUTPUT_PATH = Path.home() / ".hermes" / "whatsapp-ingest" / "messages.jsonl"

SENSITIVE_RAW_KEYS = {
    "botIds",
    "mentionedIds",
    "quotedParticipant",
    "quotedRemoteJid",
}


@dataclass(frozen=True)
class IngestConfig:
    bridge_url: str = "http://127.0.0.1:3000"
    output_path: Path = DEFAULT_OUTPUT_PATH
    group_policy: str = "allowlist"  # allowlist | open | disabled
    group_allowlist: frozenset[str] = frozenset()
    dm_policy: str = "disabled"  # disabled | open
    poll_interval_seconds: float = 5.0
    max_body_chars: int = 20000
    include_raw: bool = False


@dataclass(frozen=True)
class StoredMessage:
    ingested_at: str
    message_id: str | None
    chat_id: str
    chat_name: str | None
    sender_id: str | None
    sender_name: str | None
    body: str
    timestamp: Any
    has_media: bool
    media_type: str | None
    media_urls: list[str]
    raw: dict[str, Any] | None = None


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is present in Hermes dev env
        raise SystemExit("PyYAML is required to read YAML config files") from exc
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return loaded


def _load_config_file(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh) or {}
        if not isinstance(loaded, dict):
            raise SystemExit(f"Config file must contain an object: {path}")
        return loaded
    return _load_yaml(path)


def build_config(args: argparse.Namespace) -> IngestConfig:
    raw = _load_config_file(args.config)
    # Support either a top-level file or a section inside the main Hermes config.
    section = raw.get("whatsapp_readonly_ingest", raw)
    if not isinstance(section, dict):
        raise SystemExit("whatsapp_readonly_ingest config must be a mapping")

    group_allowlist = _coerce_list(args.group or section.get("group_allowlist") or section.get("groups"))
    group_policy = str(args.group_policy or section.get("group_policy") or "allowlist").strip().lower()
    dm_policy = str(args.dm_policy or section.get("dm_policy") or "disabled").strip().lower()
    if group_policy not in {"allowlist", "open", "disabled"}:
        raise SystemExit("group_policy must be one of: allowlist, open, disabled")
    if dm_policy not in {"disabled", "open"}:
        raise SystemExit("dm_policy must be one of: disabled, open")
    if group_policy == "allowlist" and not group_allowlist:
        raise SystemExit(
            "Refusing to ingest WhatsApp groups without a group_allowlist. "
            "Pass --group <jid> or set group_policy: open explicitly."
        )

    bridge_url = str(args.bridge_url or section.get("bridge_url") or "http://127.0.0.1:3000").rstrip("/")
    output_path = Path(args.output or section.get("output_path") or DEFAULT_OUTPUT_PATH).expanduser()
    poll_interval = float(args.poll_interval or section.get("poll_interval_seconds") or 5.0)
    max_body_chars = int(args.max_body_chars or section.get("max_body_chars") or 20000)
    include_raw = bool(args.include_raw or section.get("include_raw") or False)

    return IngestConfig(
        bridge_url=bridge_url,
        output_path=output_path,
        group_policy=group_policy,
        group_allowlist=frozenset(group_allowlist),
        dm_policy=dm_policy,
        poll_interval_seconds=poll_interval,
        max_body_chars=max_body_chars,
        include_raw=include_raw,
    )


def message_allowed(message: Mapping[str, Any], config: IngestConfig) -> bool:
    chat_id = str(message.get("chatId") or "")
    is_group = bool(message.get("isGroup")) or chat_id.endswith("@g.us")
    if not is_group:
        return config.dm_policy == "open"
    if config.group_policy == "disabled":
        return False
    if config.group_policy == "open":
        return True
    return chat_id in config.group_allowlist


def sanitize_message(message: Mapping[str, Any], config: IngestConfig) -> StoredMessage:
    body = str(message.get("body") or "")
    if config.max_body_chars > 0 and len(body) > config.max_body_chars:
        body = body[: config.max_body_chars] + "…[truncated]"

    raw: dict[str, Any] | None = None
    if config.include_raw:
        raw = {k: v for k, v in dict(message).items() if k not in SENSITIVE_RAW_KEYS}

    return StoredMessage(
        ingested_at=datetime.now(timezone.utc).isoformat(),
        message_id=message.get("messageId"),
        chat_id=str(message.get("chatId") or ""),
        chat_name=message.get("chatName"),
        sender_id=message.get("senderId"),
        sender_name=message.get("senderName"),
        body=body,
        timestamp=message.get("timestamp"),
        has_media=bool(message.get("hasMedia")),
        media_type=message.get("mediaType"),
        media_urls=[str(url) for url in (message.get("mediaUrls") or [])],
        raw=raw,
    )


def fetch_bridge_messages(bridge_url: str, timeout: float = 30.0) -> list[dict[str, Any]]:
    url = bridge_url.rstrip("/") + "/messages"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Bridge /messages returned {type(payload).__name__}, expected list")
    return [item for item in payload if isinstance(item, dict)]


def append_messages(path: Path, messages: Sequence[StoredMessage]) -> int:
    if not messages:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for message in messages:
            fh.write(json.dumps(asdict(message), ensure_ascii=False, separators=(",", ":")) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return len(messages)


def ingest_once(config: IngestConfig) -> tuple[int, int]:
    fetched = fetch_bridge_messages(config.bridge_url)
    allowed = [sanitize_message(msg, config) for msg in fetched if message_allowed(msg, config)]
    append_messages(config.output_path, allowed)
    return len(fetched), len(allowed)


def _print_sample_config() -> None:
    print(
        """# ~/.hermes/whatsapp-readonly-ingest.yaml
bridge_url: http://127.0.0.1:3000
output_path: ~/.hermes/whatsapp-ingest/messages.jsonl

# Deny-by-default: only these WhatsApp group JIDs are persisted.
group_policy: allowlist
group_allowlist:
  - "1234567890@g.us"

# DMs are ignored by default.
dm_policy: disabled
poll_interval_seconds: 5
max_body_chars: 20000
include_raw: false
""".rstrip()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only WhatsApp group ingestor for Hermes digests")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML/JSON config path")
    parser.add_argument("--bridge-url", help="Local WhatsApp bridge URL, default http://127.0.0.1:3000")
    parser.add_argument("--output", help="JSONL output path")
    parser.add_argument("--group", action="append", help="Allowed WhatsApp group JID; repeatable")
    parser.add_argument("--group-policy", choices=["allowlist", "open", "disabled"])
    parser.add_argument("--dm-policy", choices=["disabled", "open"])
    parser.add_argument("--poll-interval", type=float, help="Watch-mode poll interval in seconds")
    parser.add_argument("--max-body-chars", type=int, help="Maximum body chars to persist per message")
    parser.add_argument("--include-raw", action="store_true", help="Persist sanitized raw bridge payload too")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--print-sample-config", action="store_true", help="Print a sample config and exit")
    args = parser.parse_args(argv)

    if args.print_sample_config:
        _print_sample_config()
        return 0

    config = build_config(args)
    stop = False

    def _stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while True:
        try:
            fetched, stored = ingest_once(config)
            print(f"fetched={fetched} stored={stored} output={config.output_path}", flush=True)
        except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
            print(f"ingest error: {exc}", file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once or stop:
            return 0
        time.sleep(config.poll_interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
