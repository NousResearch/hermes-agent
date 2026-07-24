"""Headroom Phase 1 structured tool-result compression experiment.

The plugin is bundled but inert unless explicitly enabled. It uses the
``transform_tool_result`` hook so the core dispatch path stays untouched.

Phase 1 intentionally does not persist raw tool output. That avoids creating
retrieval handles until the storage and access story is proven. Compressed
payloads still include an explicit retrieval-unavailable marker so downstream
tests and callers do not infer a missing raw handle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional


SCHEMA_VERSION = "hermes.headroom.phase1.v1"

_ALLOWED_TOOLS = frozenset({"search_files", "browser_snapshot"})
_EXCLUDED_TOOLS = frozenset({
    "terminal",
    "read_file",
    "delegate_task",
    "patch",
    "write_file",
    "memory",
    "send_message",
    "clarify",
    "cronjob",
})
_DEFAULT_ALLOWLIST = tuple(sorted(_ALLOWED_TOOLS))
_UNTRUSTED_CLOSE = "</untrusted_tool_result>"


@dataclass(frozen=True)
class _Settings:
    enabled: bool
    kill_switch: bool
    allowlist: frozenset[str]
    excluded_tools: frozenset[str]
    max_items: int
    max_field_chars: int
    max_snapshot_chars: int


@dataclass(frozen=True)
class _ParsedJsonResult:
    value: Any
    suffix: str


@dataclass(frozen=True)
class _WrappedResult:
    prefix: str
    body: str
    suffix: str


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def _coerce_int(value: Any, default: int, *, floor: int, ceiling: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(floor, min(ceiling, parsed))


def _load_headroom_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        block = cfg.get("headroom", {})
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _settings() -> _Settings:
    cfg = _load_headroom_config()

    env_enabled = _env_bool("HERMES_HEADROOM_ENABLED")
    enabled = env_enabled if env_enabled is not None else bool(cfg.get("enabled", False))

    kill_switch = bool(cfg.get("kill_switch", False))
    for name in ("HERMES_HEADROOM_DISABLE", "HERMES_HEADROOM_KILL_SWITCH"):
        env_kill = _env_bool(name)
        if env_kill is True:
            kill_switch = True

    raw_allowlist: Any = cfg.get("allowlist", _DEFAULT_ALLOWLIST)
    env_allowlist = os.environ.get("HERMES_HEADROOM_ALLOWLIST")
    if env_allowlist:
        raw_allowlist = [
            part.strip()
            for part in env_allowlist.split(",")
            if part.strip()
        ]
    if not isinstance(raw_allowlist, (list, tuple, set)):
        raw_allowlist = _DEFAULT_ALLOWLIST

    raw_excluded: Any = cfg.get("excluded_tools", _EXCLUDED_TOOLS)
    if not isinstance(raw_excluded, (list, tuple, set)):
        raw_excluded = _EXCLUDED_TOOLS
    # Phase 1 has mandatory exclusions, and config may add more exclusions.
    excluded_tools = frozenset(str(name) for name in raw_excluded) | _EXCLUDED_TOOLS

    allowlist = frozenset(str(name) for name in raw_allowlist)
    # Phase 1 can only narrow the hardcoded allowlist, never widen it.
    allowlist = (allowlist & _ALLOWED_TOOLS) - excluded_tools

    return _Settings(
        enabled=bool(enabled),
        kill_switch=bool(kill_switch),
        allowlist=allowlist,
        excluded_tools=excluded_tools,
        max_items=_coerce_int(cfg.get("max_items"), 8, floor=1, ceiling=50),
        max_field_chars=_coerce_int(cfg.get("max_field_chars"), 240, floor=40, ceiling=2000),
        max_snapshot_chars=_coerce_int(
            cfg.get("max_snapshot_chars"), 2400, floor=200, ceiling=20000
        ),
    )


def _split_untrusted_wrapper(result: str) -> Optional[_WrappedResult]:
    leading_len = len(result) - len(result.lstrip())
    leading = result[:leading_len]
    rest = result[leading_len:]
    if not rest.startswith("<untrusted_tool_result"):
        return None

    close_idx = rest.rfind(_UNTRUSTED_CLOSE)
    if close_idx < 0:
        return None

    suffix_start = close_idx
    if suffix_start > 0 and rest[suffix_start - 1] == "\n":
        suffix_start -= 1

    header_end = rest.find("\n\n")
    if header_end < 0 or header_end + 2 > suffix_start:
        return None

    body_start = header_end + 2
    return _WrappedResult(
        prefix=leading + rest[:body_start],
        body=rest[body_start:suffix_start],
        suffix=rest[suffix_start:],
    )


def _redact_text(text: str) -> tuple[str, bool]:
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(text, force=True)
    except Exception:
        redacted = text
    return redacted, redacted != text


def _redact_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        changed = False
        out = []
        for item in value:
            redacted, item_changed = _redact_value(item)
            changed = changed or item_changed
            out.append(redacted)
        return out, changed
    if isinstance(value, dict):
        changed = False
        out: dict[Any, Any] = {}
        for key, item in value.items():
            out_key = key
            if isinstance(key, str):
                out_key, key_changed = _redact_text(key)
                changed = changed or key_changed
            redacted, item_changed = _redact_value(item)
            changed = changed or item_changed
            out[out_key] = redacted
        return out, changed
    return value, False


def _parse_json_result(result: str) -> Optional[_ParsedJsonResult]:
    """Parse JSON tool output, allowing Hermes' search_files hint suffix.

    ``search_files`` appends a plaintext pagination hint after the JSON when
    results are truncated. That is still an allowlisted structured result, so
    Phase 1 parses the JSON prefix and records that a suffix was present
    instead of silently skipping compression.
    """
    try:
        decoder = json.JSONDecoder()
        value, end = decoder.raw_decode(result)
    except (TypeError, ValueError):
        return None

    suffix = result[end:]
    if suffix.strip() and not suffix.lstrip().startswith("[Hint:"):
        return None
    return _ParsedJsonResult(value=value, suffix=suffix)


def _clip_text(value: Any, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    last_nl = clipped.rfind("\n")
    if last_nl > max_chars // 2:
        clipped = clipped[:last_nl]
    omitted = len(text) - len(clipped)
    return f"{clipped}\n[headroom: truncated {omitted} chars]"


def _ordered_unique(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _compress_search_files(data: Any, settings: _Settings) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if data.get("error"):
        return None

    compressed: dict[str, Any] = {
        "total_count": data.get("total_count", 0),
        "truncated": bool(data.get("truncated", False)),
    }

    matches = data.get("matches")
    if isinstance(matches, list):
        sample = []
        paths: list[str] = []
        for item in matches[: settings.max_items]:
            if isinstance(item, dict):
                path = str(item.get("path", ""))
                if path:
                    paths.append(path)
                sample.append({
                    "path": path,
                    "line": item.get("line"),
                    "content": _clip_text(item.get("content", ""), settings.max_field_chars),
                })
            else:
                sample.append({"content": _clip_text(item, settings.max_field_chars)})
        compressed.update({
            "kind": "matches",
            "returned_count": len(matches),
            "omitted_count": max(0, len(matches) - len(sample)),
            "paths": _ordered_unique(paths, settings.max_items),
            "matches": sample,
        })
        return compressed

    files = data.get("files")
    if isinstance(files, list):
        sample_files = [str(item) for item in files[: settings.max_items]]
        compressed.update({
            "kind": "files",
            "returned_count": len(files),
            "omitted_count": max(0, len(files) - len(sample_files)),
            "files": sample_files,
        })
        return compressed

    counts = data.get("counts")
    if isinstance(counts, dict):
        sorted_counts = sorted(
            counts.items(),
            key=lambda item: item[1] if isinstance(item[1], int) else 0,
            reverse=True,
        )
        compressed.update({
            "kind": "counts",
            "returned_count": len(counts),
            "omitted_count": max(0, len(counts) - settings.max_items),
            "counts": [
                {"path": str(path), "count": count}
                for path, count in sorted_counts[: settings.max_items]
            ],
        })
        return compressed

    return None


def _compact_metadata(value: Any, settings: _Settings) -> Any:
    if isinstance(value, str):
        return _clip_text(value, settings.max_field_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_compact_metadata(item, settings) for item in value[: settings.max_items]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys(), key=str)[: settings.max_items]:
            out[str(key)] = _compact_metadata(value[key], settings)
        if len(value) > settings.max_items:
            out["_headroom_omitted_keys"] = len(value) - settings.max_items
        return out
    return _clip_text(value, settings.max_field_chars)


def _compress_browser_snapshot(data: Any, settings: _Settings) -> Optional[dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if data.get("error") or data.get("success") is False:
        return None

    snapshot = data.get("snapshot")
    if not isinstance(snapshot, str):
        return None

    clipped_snapshot = _clip_text(snapshot, settings.max_snapshot_chars)
    metadata = {
        key: _compact_metadata(value, settings)
        for key, value in data.items()
        if key not in {"snapshot", "success"}
    }

    return {
        "success": data.get("success", True),
        "kind": "browser_snapshot",
        "snapshot": clipped_snapshot,
        "snapshot_stats": {
            "original_chars": len(snapshot),
            "returned_chars": len(clipped_snapshot),
            "original_lines": len(snapshot.splitlines()),
            "omitted_chars": max(0, len(snapshot) - len(clipped_snapshot)),
        },
        "metadata": metadata,
    }


def _source_scope(**kwargs: Any) -> dict[str, str]:
    scope: dict[str, str] = {}
    for key in ("session_id", "task_id", "tool_call_id", "turn_id"):
        value = kwargs.get(key)
        if value:
            scope[key] = str(value)
    return scope


def _compress_result(
    *,
    tool_name: str,
    result: str,
    settings: _Settings,
    hook_kwargs: dict[str, Any],
) -> Optional[str]:
    parsed_result = _parse_json_result(result)
    if parsed_result is None:
        return None

    redacted, had_redaction = _redact_value(parsed_result.value)
    if tool_name == "search_files":
        payload = _compress_search_files(redacted, settings)
    elif tool_name == "browser_snapshot":
        payload = _compress_browser_snapshot(redacted, settings)
    else:
        payload = None

    if payload is None:
        return None

    payload["_headroom"] = {
        "schema_version": SCHEMA_VERSION,
        "compressed": True,
        "tool": tool_name,
        "original_chars": len(result),
        "redacted": had_redaction,
        "scope": _source_scope(**hook_kwargs),
        "retrieval": {
            "available": False,
            "reason": "raw_storage_not_enabled_phase1",
        },
    }
    if parsed_result.suffix:
        payload["_headroom"]["source_suffix"] = {
            "present": True,
            "chars": len(parsed_result.suffix),
            "kind": (
                "search_files_hint"
                if parsed_result.suffix.lstrip().startswith("[Hint:")
                else "unknown"
            ),
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _on_transform_tool_result(
    tool_name: str = "",
    result: Any = None,
    **kwargs: Any,
) -> Optional[str]:
    settings = _settings()
    if not settings.enabled or settings.kill_switch:
        return None
    if tool_name in settings.excluded_tools:
        return None
    if tool_name not in settings.allowlist:
        return None
    if not isinstance(result, str):
        return None

    wrapped = _split_untrusted_wrapper(result)
    if wrapped is not None:
        compressed = _compress_result(
            tool_name=tool_name,
            result=wrapped.body,
            settings=settings,
            hook_kwargs=kwargs,
        )
        if compressed is None:
            return None
        return wrapped.prefix + compressed + wrapped.suffix

    return _compress_result(
        tool_name=tool_name,
        result=result,
        settings=settings,
        hook_kwargs=kwargs,
    )


def register(ctx) -> None:
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
