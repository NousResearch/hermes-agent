"""Import a grokbot-export.json into Hermes Bot Mode profiles.

Maps each exported Grok Bot to a Hermes profile (in Bot Mode a Bot IS a
profile), imports its conversations as sessions in the profile's state.db,
writes its instructions to SOUL.md, its memories to profile memory entries,
and pins the canonical chat the way the desktop does when a Bot is born.

Security contract (mirrors ``hermes import-agent``): export files carry no
credentials by design and this module refuses files that violate that
contract's shape. Nothing here ever talks to a network.

Export format (schema 1)::

    {
      "schema": 1,
      "exported_at": "ISO8601",
      "app_version": "0.24.0",
      "bots": [
        {
          "id": "...", "name": "...", "title": "...", "description": "...",
          "instructions": "...", "model": "...", "memories": ["..."],
          "tools": ["..."], "plugins": ["..."]
        }
      ],
      "conversations": [
        {
          "bot_id": "...", "thread_id": "...", "title": "...",
          "messages": [
            {"role": "user|assistant", "text": "...", "ts": 1234.5,
             "attachments": []}
          ]
        }
      ],
      "provenance": {"layers": ["witness"], "warnings": []}
    }
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = 1

# Same delimiter as the Hermes memory store and ``hermes import-agent``.
ENTRY_DELIMITER = "\n§\n"

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9_-]+")
_SLUG_TRIM_RE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")

# Session IDs must be stable across re-runs so a re-import skips cleanly.
_SESSION_ID_PREFIX = "grokbot-import"


class ExportValidationError(ValueError):
    """The export file violates the schema-1 contract."""


class ImportRecord:
    """One bot's import outcome."""

    def __init__(self, name: str, profile_name: str, bot_id: str = "") -> None:
        self.name = name
        self.profile_name = profile_name
        self.bot_id = bot_id
        self.status = "pending"  # imported | skipped | conflict | error
        self.detail = ""
        self.sessions = 0
        self.messages = 0


def slugify(name: str) -> str:
    """Lowercase, hyphenate, and strip a bot name into a profile id."""
    slug = re.sub(r"\s+", "-", (name or "").strip().lower())
    slug = _SLUG_STRIP_RE.sub("-", slug)
    slug = _SLUG_TRIM_RE.sub("", slug)
    slug = re.sub(r"[_-]{2,}", "-", slug)
    return slug or "grok-bot"


def load_export(path) -> Dict[str, Any]:
    """Load and strictly validate a grokbot-export.json (schema 1)."""
    import json

    path = Path(path)
    if not path.is_file():
        raise ExportValidationError(f"export file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ExportValidationError(f"cannot read export file: {exc}") from exc

    if not isinstance(data, dict):
        raise ExportValidationError("export root must be a JSON object")
    if data.get("schema") != EXPORT_SCHEMA_VERSION:
        raise ExportValidationError(
            f"unsupported schema {data.get('schema')!r}; this build reads "
            f"schema {EXPORT_SCHEMA_VERSION}"
        )

    known_top = {
        "schema", "exported_at", "app_version", "account", "bots",
        "conversations", "files", "provenance",
    }
    unknown = sorted(set(data) - known_top)
    if unknown:
        raise ExportValidationError(
            f"unexpected top-level keys (secrets are never stored in export "
            f"files; re-export with a current exporter): {', '.join(unknown)}"
        )

    bots = data.get("bots") or []
    conversations = data.get("conversations") or []
    if not isinstance(bots, list) or not isinstance(conversations, list):
        raise ExportValidationError("'bots' and 'conversations' must be lists")

    bot_ids: set = set()
    for i, bot in enumerate(bots):
        if not isinstance(bot, dict):
            raise ExportValidationError(f"bots[{i}] must be an object")
        bid = str(bot.get("id") or "").strip()
        if not bid:
            raise ExportValidationError(f"bots[{i}].id is required")
        if bid in bot_ids:
            raise ExportValidationError(f"duplicate bot id: {bid}")
        bot_ids.add(bid)
        if not str(bot.get("name") or "").strip():
            raise ExportValidationError(f"bots[{i}] ({bid}).name is required")

    for i, conv in enumerate(conversations):
        if not isinstance(conv, dict):
            raise ExportValidationError(f"conversations[{i}] must be an object")
        if str(conv.get("bot_id") or "") not in bot_ids:
            raise ExportValidationError(
                f"conversations[{i}].bot_id references an unknown bot"
            )
        messages = conv.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ExportValidationError(
                f"conversations[{i}].messages must be a non-empty list"
            )
        for j, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ExportValidationError(
                    f"conversations[{i}].messages[{j}] must be an object"
                )
            role = str(msg.get("role") or "")
            if role not in ("user", "assistant"):
                raise ExportValidationError(
                    f"conversations[{i}].messages[{j}].role must be "
                    f"'user' or 'assistant'"
                )
            if not isinstance(msg.get("text"), str):
                raise ExportValidationError(
                    f"conversations[{i}].messages[{j}].text must be a string"
                )
            ts = msg.get("ts")
            if ts is not None and not isinstance(ts, (int, float)):
                raise ExportValidationError(
                    f"conversations[{i}].messages[{j}].ts must be a number"
                )

    return data


def plan_import(
    export: Dict[str, Any],
    *,
    target_bots: Optional[List[str]] = None,
) -> Tuple[List[ImportRecord], Dict[str, List[Dict[str, Any]]]]:
    """Resolve collision-free profile names and group conversations by bot.

    Returns ``(records, conversations_by_bot)``. Profile-name collisions
    WITHIN the export get a numeric suffix; collisions with pre-existing
    profiles are resolved at import time (merge when the profile carries our
    import marker, otherwise ``--force`` is required).
    """
    records: List[ImportRecord] = []
    by_bot: Dict[str, List[Dict[str, Any]]] = {}
    for conv in export.get("conversations") or []:
        by_bot.setdefault(str(conv["bot_id"]), []).append(conv)

    seen: set = set()
    for bot in export.get("bots") or []:
        bid = str(bot["id"])
        if target_bots and str(bot.get("name")) not in target_bots and bid not in target_bots:
            continue
        base = slugify(str(bot.get("name")))
        name, n = base, 2
        while name in seen:
            name = f"{base}-{n}"
            n += 1
        seen.add(name)
        records.append(ImportRecord(str(bot.get("name")), name, bot_id=str(bid)))
    return records, by_bot


def _first_last_ts(messages: List[Dict[str, Any]]) -> Tuple[float, float]:
    stamps = [float(m["ts"]) for m in messages if m.get("ts") is not None]
    if not stamps:
        now = time.time()
        return now, now
    return min(stamps), max(stamps)


def _map_session(
    conv: Dict[str, Any],
    *,
    profile_name: str,
    bot_name: str,
    bot_model: Optional[str],
    pinned: bool,
    index: int,
) -> Dict[str, Any]:
    """Map one exported conversation onto a SessionDB import dict."""
    messages = conv["messages"]
    first, last = _first_last_ts(messages)
    mapped: List[Dict[str, Any]] = []
    for msg in messages:
        mapped.append(
            {
                "role": msg["role"],
                "content": msg["text"],
                "timestamp": msg.get("ts") or first,
            }
        )
    title = str(conv.get("title") or "").strip() or f"Chat with {bot_name}"
    return {
        "id": f"{_SESSION_ID_PREFIX}-{index}",
        "source": "grokbot-import",
        "title": title,
        "started_at": first,
        "ended_at": last,
        "pinned": 1 if pinned else 0,
        "hidden": 0,
        "profile_name": profile_name,
        "model": bot_model,
        "message_count": len(mapped),
        "messages": mapped,
    }


def _write_soul(profile_dir: Path, bot: Dict[str, Any], exported_at: str) -> None:
    from utils import atomic_write_text

    name = str(bot.get("name") or "").strip()
    title = str(bot.get("title") or "").strip()
    description = str(bot.get("description") or "").strip()
    instructions = str(bot.get("instructions") or "").strip()

    parts = [f"# {name}"]
    if title:
        parts.append(f"**Role:** {title}")
    if description:
        parts.append(f"**About:** {description}")
    if instructions:
        parts.append("\n## Instructions\n")
        parts.append(instructions)
    parts.append(f"\nImported from Grok Bot on {exported_at}.")
    atomic_write_text(profile_dir / "SOUL.md", "\n".join(parts) + "\n")


def _write_memories(profile_dir: Path, bot: Dict[str, Any]) -> None:
    from utils import atomic_write_text

    memories = [str(m).strip() for m in (bot.get("memories") or []) if str(m).strip()]
    if not memories:
        return
    memory_file = profile_dir / "memories" / "MEMORY.md"
    if memory_file.exists():
        existing = memory_file.read_text(encoding="utf-8", errors="replace").strip()
        if existing:
            memories = ([existing] + memories)
    atomic_write_text(memory_file, ENTRY_DELIMITER.join(memories) + "\n")


def _import_bot(
    record: ImportRecord,
    bot: Dict[str, Any],
    conversations: List[Dict[str, Any]],
    exported_at: str,
    *,
    dry_run: bool,
    force: bool,
) -> ImportRecord:
    """Create one profile and import its conversations. Atomic per bot."""
    from hermes_cli.profiles import (
        create_profile,
        profile_exists,
        get_profile_dir,
        write_profile_meta,
    )

    if profile_exists(record.profile_name):
        marker = get_profile_dir(record.profile_name) / "grokbot-import.json"
        if marker.is_file() or force:
            record.status = "conflict"
            record.detail = "existing profile; merging (sessions already present are skipped)"
        else:
            record.status = "conflict"
            record.detail = (
                f"profile '{record.profile_name}' already exists; pass "
                f"--force to import into it anyway"
            )
            return record

    if dry_run:
        record.status = "imported"
        record.sessions = len(conversations)
        record.messages = sum(len(c["messages"]) for c in conversations)
        record.detail = "dry-run: no writes"
        return record

    profile_dir: Optional[Path] = None
    created_here = False
    try:
        profile_dir = get_profile_dir(record.profile_name)
        created_here = not profile_dir.exists()
        if created_here:
            create_profile(
                record.profile_name,
                description=str(bot.get("description") or "").strip(),
            )
        write_profile_meta(
            profile_dir,
            description=str(bot.get("description") or "").strip(),
            description_auto=False,
            display_name=str(bot.get("name") or "").strip(),
        )
        _write_soul(profile_dir, bot, exported_at)
        _write_memories(profile_dir, bot)

        from hermes_state import SessionDB

        db = SessionDB(profile_dir / "state.db")
        sessions = [
            _map_session(
                conv,
                profile_name=record.profile_name,
                bot_name=str(bot.get("name") or ""),
                bot_model=str(bot.get("model")) if bot.get("model") else None,
                pinned=(i == 0),
                index=i,
            )
            for i, conv in enumerate(conversations)
        ]
        result = db.import_sessions(sessions)
        errors = result.get("errors") or []
        if errors:
            raise RuntimeError(
                f"session import rejected {len(errors)} item(s): "
                f"{errors[0].get('error', 'unknown')[:200]}"
            )
        # Pin the canonical chat the way the desktop pins a Bot's chat at
        # creation time.
        if sessions:
            db.set_session_pinned(sessions[0]["id"], True)
        db.close()

        # Import marker: makes re-imports merge idempotently.
        marker = profile_dir / "grokbot-import.json"
        import json as _json

        marker.write_text(
            _json.dumps(
                {
                    "schema": EXPORT_SCHEMA_VERSION,
                    "bot_id": str(bot.get("id") or ""),
                    "imported_at": exported_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        skipped = int(result.get("skipped") or 0)
        record.sessions = len(sessions) - skipped
        record.messages = sum(len(s["messages"]) for s in sessions)
        record.status = "imported"
        if skipped:
            record.detail = f"{skipped} session(s) already present, skipped"
        return record
    except Exception as exc:  # noqa: BLE001 — atomic rollback per bot
        if created_here and profile_dir is not None:
            shutil.rmtree(profile_dir, ignore_errors=True)
        record.status = "error"
        record.detail = f"{type(exc).__name__}: {exc}"
        return record


def run_import(
    export_path,
    *,
    dry_run: bool = False,
    force: bool = False,
    target_bots: Optional[List[str]] = None,
) -> int:
    """Import an export file. Returns a process exit code."""
    from hermes_cli.colors import Colors, color

    export_path = Path(export_path)
    try:
        export = load_export(export_path)
    except ExportValidationError as exc:
        print(color(f"✗ {exc}", Colors.RED), file=sys.stderr)
        return 1

    bots = export.get("bots") or []
    conversations = export.get("conversations") or []
    exported_at = str(export.get("exported_at") or "").strip()

    if not bots:
        print(color("✗ Export contains no bots — nothing to import.", Colors.RED))
        return 1

    records, by_bot = plan_import(export, target_bots=target_bots)
    if not records:
        print(color("✗ No bots matched the requested filter.", Colors.RED))
        return 1

    print()
    print(color(f"◆ Grok Bot import ({len(records)} bot(s))", Colors.CYAN, Colors.BOLD))
    if dry_run:
        print(color("Dry-run mode — no changes written.", Colors.DIM))
    print()

    exit_code = 0
    for record in records:
        bot = next(
            (b for b in bots if str(b.get("id")) == record.bot_id),
            next((b for b in bots if str(b.get("name")) == record.name), bots[0]),
        )
        convs = [c for c in by_bot.get(str(bot["id"]), []) if c.get("bot_id") == bot["id"]]
        record = _import_bot(
            record, bot, convs, exported_at, dry_run=dry_run, force=force
        )
        if record.status == "error":
            exit_code = 1
        mark = {
            "imported": color("✓", Colors.GREEN),
            "conflict": color("⚠", Colors.YELLOW),
            "error": color("✗", Colors.RED),
            "skipped": color("·", Colors.DIM),
        }.get(record.status, color("·", Colors.DIM))
        line = f"  {mark} {record.name} → profile '{record.profile_name}'"
        if record.status == "imported" and not dry_run:
            line += f" ({record.sessions} sessions, {record.messages} messages)"
        print(line)
        if record.detail:
            print(f"      {color(record.detail, Colors.DIM)}")

    print()
    if dry_run:
        print(color("Re-run without --dry-run to import.", Colors.DIM))
    else:
        print(
            color(
                "Done. Open the desktop app to see the imported Bots in the "
                "Bots pane.",
                Colors.DIM,
            )
        )
    return exit_code
