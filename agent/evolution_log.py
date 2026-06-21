from __future__ import annotations

import difflib
import json
import os
import secrets
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA_VERSION = 1
EVENTS_REL_PATH = Path("evolution") / "events.jsonl"
MEMORY_EVENT_TYPES = {"memory.add", "memory.replace", "memory.remove"}
SKILL_EVENT_TYPES = {
    "skill.create",
    "skill.patch",
    "skill.edit",
    "skill.delete",
    "skill.write_file",
    "skill.remove_file",
}


def _coerce_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def generate_event_id(now: datetime | None = None) -> str:
    stamp = _coerce_utc(now).strftime("%Y%m%d_%H%M%S")
    return f"evt_{stamp}_{secrets.token_hex(3)}"


def utc_timestamp(now: datetime | None = None) -> str:
    return _coerce_utc(now).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_unified_diff(before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    return "\n".join(lines)


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars < 0:
        max_chars = 0
    if len(text) <= max_chars:
        return text, False
    return f"{text[:max_chars]}\n[truncated to {max_chars} chars]", True


def redact_text_if_enabled(text: str | None, enabled: bool) -> tuple[str | None, bool]:
    if not enabled or text is None:
        return text, False
    from agent.redact import redact_sensitive_text

    redacted = redact_sensitive_text(text)
    return redacted, redacted != text


def get_evolution_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "evolution"


def get_events_path() -> Path:
    return get_evolution_dir() / "events.jsonl"


def ensure_evolution_dir() -> Path:
    path = get_evolution_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _file_lock(file_obj):
    try:
        if os.name == "nt":
            import msvcrt

            file_obj.seek(0, os.SEEK_END)
            size = max(file_obj.tell(), 1)
            msvcrt.locking(file_obj.fileno(), msvcrt.LK_LOCK, size)
            try:
                yield
            finally:
                file_obj.seek(0, os.SEEK_SET)
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, size)
        else:
            import fcntl

            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
    except Exception:
        yield


def append_event(event: dict) -> None:
    ensure_evolution_dir()
    path = get_events_path()
    with path.open("a", encoding="utf-8") as f:
        with _file_lock(f):
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_events() -> tuple[list[dict], list[str]]:
    path = get_events_path()
    if not path.exists():
        return [], []
    events: list[dict] = []
    warnings: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                warnings.append(f"Skipping malformed event line {line_number}: {exc}")
                continue
            events.append(event)
    return events, warnings


def _config_value(config: dict | None, key: str, default):
    if not isinstance(config, dict):
        return default
    section = config.get("evolution")
    if not isinstance(section, dict):
        return default
    return section.get(key, default)


def build_event(
    *,
    event_type: str,
    source_tool: str,
    target: str,
    target_kind: str,
    target_name: str,
    before_text: str = "",
    after_text: str = "",
    summary: str | None = None,
    reason: str | None = None,
    session_id: str | None = None,
    platform: str | None = None,
    profile: str | None = None,
    now: datetime | None = None,
    config: dict | None = None,
    diff_override: str | None = None,
) -> dict:
    redact_enabled = bool(_config_value(config, "redact", True))
    record_diff = bool(_config_value(config, "record_diff", True))
    max_diff_chars = int(_config_value(config, "max_diff_chars", 20000) or 20000)

    if diff_override is not None:
        diff = diff_override
        diff_format = "unified"
    elif record_diff:
        diff = make_unified_diff(before_text or "", after_text or "")
        diff_format = "unified"
    else:
        diff = "[diff recording disabled]"
        diff_format = None

    diff, diff_truncated = truncate_text(diff, max_diff_chars)
    summary_text = summary or fallback_summary(event_type, target_name, target)
    reason_text = reason

    summary_text, summary_redacted = redact_text_if_enabled(
        summary_text, redact_enabled
    )
    reason_text, reason_redacted = redact_text_if_enabled(reason_text, redact_enabled)
    diff, diff_redacted = redact_text_if_enabled(diff, redact_enabled)

    return {
        "schema_version": SCHEMA_VERSION,
        "id": generate_event_id(now),
        "timestamp": utc_timestamp(now),
        "profile": profile,
        "platform": platform,
        "session_id": session_id,
        "actor": "agent",
        "source_tool": source_tool,
        "type": event_type,
        "target": target,
        "target_kind": target_kind,
        "target_name": target_name,
        "summary": summary_text,
        "reason": reason_text,
        "diff_format": diff_format,
        "diff": diff,
        "redaction_enabled": redact_enabled,
        "redaction_applied": bool(summary_redacted or reason_redacted or diff_redacted),
        "diff_truncated": diff_truncated,
        "max_diff_chars": max_diff_chars,
    }


def fallback_summary(event_type: str, target_name: str, target: str) -> str:
    action = event_type.split(".", 1)[1] if "." in event_type else event_type
    category = event_type.split(".", 1)[0] if "." in event_type else "asset"
    verbs = {
        "add": "Added",
        "replace": "Updated",
        "remove": "Removed",
        "create": "Created",
        "patch": "Patched",
        "edit": "Edited",
        "delete": "Deleted",
        "write_file": "Wrote",
        "remove_file": "Removed file from",
    }
    verb = verbs.get(action, "Updated")
    name = target_name or target
    return f"{verb} {category} {name}"


def resolve_event_id(events: list[dict], query: str) -> tuple[dict | None, list[dict]]:
    q = (query or "").strip()
    if not q:
        return None, []
    matches = [event for event in events if str(event.get("id", "")) == q]
    if not matches:
        matches = [event for event in events if str(event.get("id", "")).endswith(q)]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


def _parse_event_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def filter_events(
    events: list[dict],
    days: int | None = None,
    event_type: str | None = None,
    target_query: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    filtered = list(events)

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filtered = [
            event
            for event in filtered
            if (ts := _parse_event_timestamp(str(event.get("timestamp", ""))))
            is not None
            and ts >= cutoff
        ]

    if event_type:
        requested = event_type.strip().lower()
        if "." in requested:
            filtered = [
                event
                for event in filtered
                if str(event.get("type", "")).lower() == requested
            ]
        else:
            prefix = requested + "."
            filtered = [
                event
                for event in filtered
                if str(event.get("type", "")).lower().startswith(prefix)
            ]

    if target_query:
        needle = target_query.strip().lower()
        filtered = [
            event
            for event in filtered
            if needle in str(event.get("target", "")).lower()
            or needle in str(event.get("target_name", "")).lower()
            or needle in str(event.get("target_kind", "")).lower()
        ]

    if limit is not None:
        filtered = filtered[: max(limit, 0)]

    return filtered


def clear_older_than(days: int, apply: bool = False) -> tuple[int, int]:
    if apply:
        path = get_events_path()
        if not path.exists():
            return 0, 0
        with path.open("r+", encoding="utf-8") as f:
            with _file_lock(f):
                f.seek(0)
                events: list[dict] = []
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        events.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue
                if not events:
                    return 0, 0
                deleted, retained = _partition_clear_events(events, days)
                f.seek(0)
                f.truncate()
                for event in retained:
                    f.write(
                        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                f.flush()
                os.fsync(f.fileno())
                return deleted, len(retained)

    events, _warnings = read_events()
    if not events:
        return 0, 0
    deleted, retained = _partition_clear_events(events, days)
    return deleted, len(retained)


def _partition_clear_events(events: list[dict], days: int) -> tuple[int, list[dict]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    retained: list[dict] = []
    deleted = 0
    for event in events:
        ts = _parse_event_timestamp(str(event.get("timestamp", "")))
        if ts is not None and ts < cutoff:
            deleted += 1
        else:
            retained.append(event)
    return deleted, retained


def _load_config_safely() -> dict:
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def is_enabled(config: dict | None = None) -> bool:
    cfg = config if config is not None else _load_config_safely()
    return bool(_config_value(cfg, "enabled", False))


def record_memory_event(
    action: str,
    target: str,
    before_text: str,
    after_text: str,
    summary: str | None = None,
    reason: str | None = None,
    session_id: str | None = None,
    platform: str | None = None,
) -> None:
    try:
        config = _load_config_safely()
        if not is_enabled(config):
            return
        if before_text == after_text:
            return
        target_rel = "memories/USER.md" if target == "user" else "memories/MEMORY.md"
        event = build_event(
            event_type=f"memory.{action}",
            source_tool="memory",
            target=target_rel,
            target_kind="memory",
            target_name=target,
            before_text=before_text or "",
            after_text=after_text or "",
            summary=summary,
            reason=reason,
            session_id=session_id,
            platform=platform,
            config=config,
        )
        append_event(event)
    except Exception:
        return


def record_skill_event(
    action: str,
    name: str,
    target_rel_path: str,
    before_text: str,
    after_text: str,
    summary: str | None = None,
    reason: str | None = None,
    session_id: str | None = None,
    platform: str | None = None,
    delete_omits_diff: bool = False,
) -> None:
    try:
        config = _load_config_safely()
        if not is_enabled(config):
            return
        if before_text == after_text and not delete_omits_diff:
            return
        event = build_event(
            event_type=f"skill.{action}",
            source_tool="skill_manage",
            target=target_rel_path,
            target_kind="skill",
            target_name=name,
            before_text=before_text or "",
            after_text=after_text or "",
            summary=summary,
            reason=reason,
            session_id=session_id,
            platform=platform,
            config=config,
            diff_override="[skill deleted: content omitted]"
            if delete_omits_diff
            else None,
        )
        append_event(event)
    except Exception:
        return
