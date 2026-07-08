#!/usr/bin/env python3
"""Prepare bounded Discord voice-context packets for Muncho owner review.

The Discord voice context listener writes JSONL transcript events under
``$HERMES_HOME/state/voice_context``.  This script is the bridge between those
raw transcript events and the ordinary Muncho learning loop:

* read only new transcript events since the last checkpoint;
* write a private review artifact for traceability;
* emit a bounded packet on stdout for a cron-spawned agent to summarize;
* never promote knowledge, mutate skills, call Discord, or retain raw audio.

The cron job that uses this script should run WITH an agent.  The script output
is evidence; the model produces the Bulgarian owner-review digest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "muncho.voice_context_digest_packet.v1"
DEFAULT_MAX_EVENTS = 30
DEFAULT_MAX_EVENT_CHARS = 1800
DEFAULT_MAX_TOTAL_CHARS = 18000


SECRETISH_RE = re.compile(
    r"(?i)("
    r"api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|password|secret|token"
    r"|-----BEGIN\s+(?:RSA|OPENSSH|PRIVATE)\s+KEY-----"
    r")"
)


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def hermes_home() -> Path:
    raw = os.environ.get("HERMES_HOME") or "/opt/adventico-ai-platform/hermes-home"
    return Path(raw).expanduser()


def canonical_reports_dir() -> Path:
    raw = (
        os.environ.get("MUNCHO_CANONICAL_REPORTS_DIR")
        or "/opt/adventico-ai-platform/canonical-brain/state/reports"
    )
    return Path(raw).expanduser()


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def atomic_write_json(path: Path, value: Any, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)


def atomic_write_text(path: Path, text: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    if mode is not None:
        os.chmod(path, mode)


def sanitize_text(text: str, max_chars: int) -> tuple[str, int]:
    text = str(text or "").replace("\x00", "").strip()
    redactions = 0
    if SECRETISH_RE.search(text):
        text = SECRETISH_RE.sub("[REDACTED_SECRETISH]", text)
        redactions += 1
    text = re.sub(r"@everyone", "@\u200beveryone", text, flags=re.IGNORECASE)
    text = re.sub(r"@here", "@\u200bhere", text, flags=re.IGNORECASE)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + " ...[truncated]"
    return text, redactions


def parse_jsonl_new_records(
    path: Path,
    *,
    start_offset: int,
    max_event_chars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    records: list[dict[str, Any]] = []
    redactions = 0
    if not path.exists() or not path.is_file():
        return records, start_offset, redactions

    size = path.stat().st_size
    if start_offset < 0 or start_offset > size:
        start_offset = 0

    with path.open("rb") as fh:
        fh.seek(start_offset)
        for raw_line in fh:
            try:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                item = json.loads(line)
            except Exception:
                continue
            if item.get("type") != "discord_voice_context.transcript":
                continue
            transcript, hits = sanitize_text(item.get("transcript", ""), max_event_chars)
            redactions += hits
            if not transcript:
                continue
            records.append(
                {
                    "timestamp": str(item.get("timestamp") or ""),
                    "guild_id": str(item.get("guild_id") or ""),
                    "voice_channel_id": str(item.get("voice_channel_id") or ""),
                    "voice_channel_name": str(item.get("voice_channel_name") or ""),
                    "text_channel_id": str(item.get("text_channel_id") or ""),
                    "speaker_user_id": str(item.get("user_id") or ""),
                    "transcript": transcript,
                    "raw_audio_retained": bool(item.get("raw_audio_retained", False)),
                    "source_file": path.name,
                }
            )
        end_offset = fh.tell()
    return records, end_offset, redactions


def load_new_events(
    voice_dir: Path,
    checkpoint_path: Path,
    *,
    max_events: int,
    max_event_chars: int,
    max_total_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    checkpoint = read_json(checkpoint_path, {"files": {}})
    files_state = checkpoint.get("files") if isinstance(checkpoint, dict) else {}
    if not isinstance(files_state, dict):
        files_state = {}

    selected: list[dict[str, Any]] = []
    next_offsets: dict[str, int] = {}
    redactions_by_file: dict[str, int] = {}
    total_chars = 0

    for path in sorted(voice_dir.glob("discord_g*_vc*_*.jsonl")):
        start = 0
        state = files_state.get(path.name)
        if isinstance(state, dict):
            try:
                start = int(state.get("offset", 0))
            except (TypeError, ValueError):
                start = 0
        records, end_offset, redactions = parse_jsonl_new_records(
            path,
            start_offset=start,
            max_event_chars=max_event_chars,
        )
        next_offsets[path.name] = end_offset
        redactions_by_file[path.name] = redactions

        for record in records:
            if len(selected) >= max_events:
                break
            candidate_chars = len(record["transcript"])
            if selected and total_chars + candidate_chars > max_total_chars:
                break
            selected.append(record)
            total_chars += candidate_chars
        if len(selected) >= max_events or total_chars >= max_total_chars:
            break

    new_checkpoint = {
        "schema_version": "muncho.voice_context_digest_checkpoint.v1",
        "updated_at": iso_now(),
        "files": {
            name: {"offset": offset}
            for name, offset in {**files_state, **next_offsets}.items()
        },
    }
    return selected, new_checkpoint, redactions_by_file


def period_for(events: list[dict[str, Any]]) -> dict[str, str | None]:
    timestamps = sorted(
        str(event.get("timestamp") or "")
        for event in events
        if str(event.get("timestamp") or "")
    )
    return {
        "from": timestamps[0] if timestamps else None,
        "to": timestamps[-1] if timestamps else None,
    }


def packet_id(events: list[dict[str, Any]]) -> str:
    seed = json.dumps(
        [
            {
                "timestamp": event.get("timestamp"),
                "voice_channel_id": event.get("voice_channel_id"),
                "speaker_user_id": event.get("speaker_user_id"),
                "transcript": event.get("transcript", "")[:80],
            }
            for event in events
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def render_markdown_packet(packet: dict[str, Any]) -> str:
    lines = [
        "# Voice Context Owner Review Packet",
        "",
        f"Schema: `{packet['schema_version']}`",
        f"Packet: `{packet['packet_id']}`",
        f"Created: `{packet['created_at']}`",
        f"Period: `{packet['period']['from']}` → `{packet['period']['to']}`",
        f"Events: `{packet['event_count']}`",
        "",
        "## Safety",
        "",
        "- Raw audio retained: no",
        "- Auto-promotion to skills/memory: no",
        "- Purpose: owner review only",
        "",
        "## Transcript Events",
        "",
    ]
    for idx, event in enumerate(packet["events"], start=1):
        lines.extend(
            [
                f"### Event {idx}",
                "",
                f"- Time: `{event['timestamp']}`",
                f"- Voice channel: `{event['voice_channel_name']}` (`{event['voice_channel_id']}`)",
                f"- Speaker user id: `{event['speaker_user_id']}`",
                "",
                event["transcript"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_agent_context(packet: dict[str, Any], private_md_path: Path, public_md_path: Path) -> str:
    """Render bounded evidence and explicit output contract for the cron agent."""
    lines = [
        "VOICE_CONTEXT_OWNER_REVIEW_PACKET",
        f"packet_id: {packet['packet_id']}",
        f"created_at: {packet['created_at']}",
        f"period_from: {packet['period']['from']}",
        f"period_to: {packet['period']['to']}",
        f"event_count: {packet['event_count']}",
        f"private_artifact: {private_md_path}",
        f"public_status_artifact: {public_md_path}",
        "",
        "Output contract:",
        "- Отговори на български.",
        "- Не цитирай суровите transcript-и изцяло; използвай кратки фрагменти само ако са нужни.",
        "- Маркирай STT несигурността явно, когато текстът звучи шумно или непълен.",
        "- Не промотирай знание автоматично; само предложи candidates за owner review.",
        "- Не твърди факти като сигурни, ако са само чути в voice контекст.",
        "- Формат: кратък owner digest с секции: Казуси, Следващи действия, Knowledge candidates, Неясно/рисково.",
        "",
        "Events:",
    ]
    for idx, event in enumerate(packet["events"], start=1):
        lines.extend(
            [
                f"[{idx}] time={event['timestamp']} channel={event['voice_channel_name']} "
                f"speaker_user_id={event['speaker_user_id']}",
                event["transcript"],
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice-dir", type=Path, default=hermes_home() / "state" / "voice_context")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=hermes_home() / "state" / "voice_context_owner_review",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=canonical_reports_dir() / "voice-context-digests",
    )
    parser.add_argument("--max-events", type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument("--max-event-chars", type=int, default=DEFAULT_MAX_EVENT_CHARS)
    parser.add_argument("--max-total-chars", type=int, default=DEFAULT_MAX_TOTAL_CHARS)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint path. Defaults to <state-dir>/checkpoint.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not update checkpoint; still emit a packet when new events exist.",
    )
    args = parser.parse_args(argv)

    if args.max_events <= 0:
        args.max_events = DEFAULT_MAX_EVENTS
    if args.max_event_chars <= 0:
        args.max_event_chars = DEFAULT_MAX_EVENT_CHARS
    if args.max_total_chars <= 0:
        args.max_total_chars = DEFAULT_MAX_TOTAL_CHARS

    checkpoint_path = args.checkpoint or (args.state_dir / "checkpoint.json")
    events, new_checkpoint, redactions_by_file = load_new_events(
        args.voice_dir,
        checkpoint_path,
        max_events=args.max_events,
        max_event_chars=args.max_event_chars,
        max_total_chars=args.max_total_chars,
    )

    if not events:
        return 0

    created_at = iso_now()
    pid = packet_id(events)
    packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": f"voicectx:{pid}",
        "created_at": created_at,
        "period": period_for(events),
        "event_count": len(events),
        "events": events,
        "redaction": {
            "enabled": True,
            "secretish_hits_by_file": redactions_by_file,
        },
        "safety": {
            "raw_audio_retained": False,
            "auto_knowledge_promotion": False,
            "owner_review_required": True,
            "report_only": True,
        },
    }
    new_checkpoint.update(
        {
            "last_packet_id": packet["packet_id"],
            "last_event_timestamp": events[-1].get("timestamp"),
            "last_event_source_file": events[-1].get("source_file"),
            "last_event_count": len(events),
        }
    )

    stamp = created_at.replace(":", "").replace("-", "")
    private_dir = args.state_dir / "private"
    public_dir = args.reports_dir
    private_json = private_dir / f"{packet['packet_id'].replace(':', '-')}-{stamp}.json"
    private_md = private_dir / f"{packet['packet_id'].replace(':', '-')}-{stamp}.md"
    public_md = public_dir / f"{packet['packet_id'].replace(':', '-')}-{stamp}-public-safe.md"

    atomic_write_json(private_json, packet, mode=0o600)
    atomic_write_text(private_md, render_markdown_packet(packet), mode=0o600)
    atomic_write_text(
        public_md,
        "\n".join(
            [
                "# Voice Context Digest Status",
                "",
                f"Packet: `{packet['packet_id']}`",
                f"Created: `{created_at}`",
                f"Period: `{packet['period']['from']}` → `{packet['period']['to']}`",
                f"Events: `{len(events)}`",
                "",
                "Private transcript artifact is stored under Hermes runtime state.",
                "No raw audio retained. No knowledge promotion performed.",
                "",
            ]
        ),
        mode=0o644,
    )

    if not args.dry_run:
        atomic_write_json(checkpoint_path, new_checkpoint, mode=0o600)

    print(render_agent_context(packet, private_md, public_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
