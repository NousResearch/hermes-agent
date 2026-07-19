"""Assembly-time tool-output hygiene for API request copies.

This module is intentionally non-destructive: callers pass the per-request
``api_messages`` copy, never the canonical session history.  Tool messages stay
present with their original ``tool_call_id`` so provider replay invariants are
preserved; only old/superseded/error tool *content* is replaced with stable
one-line stubs when the feature flag is enabled.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_PATH_RE = re.compile(
    r"(?:(?:~|/Users|/Volumes|/tmp|/var|/private/var)/[^\s'\"`<>),;]+)"
)


@dataclass(frozen=True)
class ToolCallInfo:
    ordinal: int
    name: str
    args_text: str
    args_sha: str


@dataclass(frozen=True)
class HygieneConfig:
    enabled: bool = False
    stale_context_ratio: float = 0.50
    stale_protect_tail_tokens: int = 40_000
    protect_last_n: int = 20
    failed_after_user_turns: int = 3
    max_stub_paths: int = 5


@dataclass(frozen=True)
class HygieneStats:
    dedup_pruned: int = 0
    failed_pruned: int = 0
    stale_pruned: int = 0
    saved_chars: int = 0

    @property
    def total_pruned(self) -> int:
        return self.dedup_pruned + self.failed_pruned + self.stale_pruned

    @property
    def saved_tokens_rough(self) -> int:
        return max(0, self.saved_chars // 4)


def _is_truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def config_from_mapping(mapping: Mapping[str, Any] | None, *, compression_protect_last_n: int = 20) -> HygieneConfig:
    root = mapping if isinstance(mapping, Mapping) else {}
    raw_hygiene = root.get("hygiene")
    hygiene: Mapping[str, Any] = raw_hygiene if isinstance(raw_hygiene, Mapping) else {}

    def _int(name: str, default: int, *, floor: int = 0) -> int:
        try:
            value = int(hygiene.get(name, default))
        except (TypeError, ValueError):
            return default
        return max(floor, value)

    def _float(name: str, default: float, *, floor: float = 0.0, ceiling: float = 1.0) -> float:
        try:
            value = float(hygiene.get(name, default))
        except (TypeError, ValueError):
            return default
        return min(ceiling, max(floor, value))

    return HygieneConfig(
        enabled=_is_truthy(hygiene.get("enabled"), default=False),
        stale_context_ratio=_float("stale_context_ratio", 0.50, floor=0.01, ceiling=0.99),
        stale_protect_tail_tokens=_int("stale_protect_tail_tokens", 40_000, floor=1_000),
        protect_last_n=_int("protect_last_n", compression_protect_last_n, floor=0),
        failed_after_user_turns=_int("failed_after_user_turns", 3, floor=0),
        max_stub_paths=_int("max_stub_paths", 5, floor=0),
    )


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _content_chars(content: Any) -> int:
    return len(_text_content(content))


def _message_tokens_rough(msg: Mapping[str, Any]) -> int:
    # Keep deterministic and cheap.  The main loop does the authoritative rough
    # token estimate separately; this only defines the protected tail window.
    return max(1, len(str(msg)) // 4)


def _normalize_args(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except Exception:
            return args.strip()
        try:
            return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return args.strip()
    try:
        return json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(args)


def _sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]


def _iter_tool_calls(msg: Mapping[str, Any]) -> Iterable[Tuple[str, str, str]]:
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    rows: list[Tuple[str, str, str]] = []
    for tc in tool_calls:
        if not isinstance(tc, Mapping):
            continue
        call_id = str(tc.get("id") or tc.get("call_id") or "").strip()
        fn = tc.get("function")
        if not isinstance(fn, Mapping):
            continue
        name = str(fn.get("name") or "").strip()
        args = _normalize_args(fn.get("arguments", ""))
        if call_id and name:
            rows.append((call_id, name, args))
    return rows


def _tool_call_index(messages: Sequence[Mapping[str, Any]]) -> Dict[str, ToolCallInfo]:
    index: Dict[str, ToolCallInfo] = {}
    ordinal = 0
    for msg in messages:
        if not isinstance(msg, Mapping) or msg.get("role") != "assistant":
            continue
        for call_id, name, args_text in _iter_tool_calls(msg):
            ordinal += 1
            index[call_id] = ToolCallInfo(
                ordinal=ordinal,
                name=name,
                args_text=args_text,
                args_sha=_sha12(args_text),
            )
    return index


def _tool_name(msg: Mapping[str, Any], info: ToolCallInfo | None) -> str:
    return str(
        msg.get("name")
        or msg.get("tool_name")
        or (info.name if info else "")
        or "unknown"
    )


def _extract_paths(text: str, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    seen: set[str] = set()
    paths: list[str] = []
    for match in _PATH_RE.finditer(text):
        path = match.group(0).rstrip(".]")
        if path not in seen:
            seen.add(path)
            paths.append(path)
            if len(paths) >= limit:
                break
    return paths


def _looks_failed(text: str) -> tuple[bool, str]:
    lowered = text.lower()
    if "error executing tool" in lowered:
        return True, "execution_error"
    if "traceback (most recent call last)" in lowered:
        return True, "traceback"
    if "permission denied" in lowered:
        return True, "permission_denied"
    if "timed out" in lowered or "timeout" in lowered:
        return True, "timeout"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, Mapping):
        err = parsed.get("error")
        if err:
            return True, str(parsed.get("error_type") or parsed.get("type") or "error")[:80]
        exit_code = parsed.get("exit_code", parsed.get("returncode"))
        if isinstance(exit_code, int) and exit_code != 0:
            return True, f"exit_code_{exit_code}"
    if re.search(r"\b(error|failed|exception)\b", lowered):
        return True, "error"
    return False, ""


def _user_turns_after(messages: Sequence[Mapping[str, Any]], idx: int) -> int:
    return sum(1 for msg in messages[idx + 1 :] if isinstance(msg, Mapping) and msg.get("role") == "user")


def _protected_tail_indices(messages: Sequence[Mapping[str, Any]], cfg: HygieneConfig) -> set[int]:
    protected: set[int] = set()
    if cfg.protect_last_n > 0:
        start = max(0, len(messages) - cfg.protect_last_n)
        protected.update(range(start, len(messages)))
    remaining = cfg.stale_protect_tail_tokens
    for idx in range(len(messages) - 1, -1, -1):
        if remaining <= 0:
            break
        protected.add(idx)
        remaining -= _message_tokens_rough(messages[idx])
    return protected


def _replace_content(msg: MutableMapping[str, Any], stub: str) -> int:
    old_chars = _content_chars(msg.get("content"))
    msg["content"] = stub
    return max(0, old_chars - len(stub))


def _dedup_stub(tool: str, info: ToolCallInfo | None, latest: ToolCallInfo | None, original_chars: int) -> str:
    latest_ord = latest.ordinal if latest else "?"
    args_sha = info.args_sha if info else "unknown"
    return (
        f"[pruned: superseded by call #{latest_ord}; tool={tool}; "
        f"args_sha={args_sha}; original_chars={original_chars}]"
    )


def _failed_stub(tool: str, info: ToolCallInfo | None, error_type: str, original_chars: int) -> str:
    ordinal = info.ordinal if info else "?"
    return (
        f"[pruned: old failed tool output; call=#{ordinal}; tool={tool}; "
        f"error_type={error_type}; original_chars={original_chars}]"
    )


def _stale_stub(tool: str, info: ToolCallInfo | None, original_chars: int, paths: list[str]) -> str:
    ordinal = info.ordinal if info else "?"
    args_sha = info.args_sha if info else "unknown"
    if paths:
        pointer = " paths=" + ",".join(paths)
    else:
        pointer = " paths=none"
    return (
        f"[pruned: stale tool output; call=#{ordinal}; tool={tool}; args_sha={args_sha}; "
        f"original_chars={original_chars}; requery=use the tool_call_id or rerun the tool if needed;{pointer}]"
    )


def apply_api_tool_output_hygiene(
    messages: List[Dict[str, Any]],
    *,
    config: HygieneConfig,
    request_tokens: int,
    context_length: int,
    session_id: str = "",
) -> tuple[List[Dict[str, Any]], HygieneStats]:
    """Return an API-message copy with old tool outputs replaced by stubs.

    ``messages`` is expected to already be an API-call copy.  With
    ``config.enabled`` false this returns the exact same list object and an empty
    stats object, preserving flag-off equality for callers/tests.
    """
    if not config.enabled:
        return messages, HygieneStats()

    call_index = _tool_call_index(messages)
    latest_by_key: Dict[Tuple[str, str], ToolCallInfo] = {}
    for info in call_index.values():
        latest_by_key[(info.name, info.args_text)] = info

    context_ratio = (request_tokens / context_length) if context_length > 0 else 0.0
    stale_enabled = context_ratio >= config.stale_context_ratio
    protected_tail = _protected_tail_indices(messages, config) if stale_enabled else set()

    # Copy lazily: once enabled, create new dicts so even tests cannot observe
    # mutation of the API list passed by the caller.  Content lists are replaced
    # wholesale only when pruned; otherwise shallow-copying message dicts is
    # enough because this module never mutates nested structures.
    out: List[Dict[str, Any]] = [dict(msg) if isinstance(msg, Mapping) else msg for msg in messages]

    dedup = failed = stale = saved = 0
    for idx, msg in enumerate(out):
        if not isinstance(msg, MutableMapping) or msg.get("role") != "tool":
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("id") or "").strip()
        info = call_index.get(call_id)
        tool = _tool_name(msg, info)
        text = _text_content(msg.get("content"))
        original_chars = len(text)

        if info is not None:
            latest = latest_by_key.get((info.name, info.args_text))
            if latest is not None and latest.ordinal != info.ordinal:
                saved += _replace_content(msg, _dedup_stub(tool, info, latest, original_chars))
                dedup += 1
                continue

        is_failed, error_type = _looks_failed(text)
        if is_failed and _user_turns_after(out, idx) >= config.failed_after_user_turns:
            saved += _replace_content(msg, _failed_stub(tool, info, error_type, original_chars))
            failed += 1
            continue

        if stale_enabled and idx not in protected_tail:
            paths = _extract_paths(text, limit=config.max_stub_paths)
            saved += _replace_content(msg, _stale_stub(tool, info, original_chars, paths))
            stale += 1

    stats = HygieneStats(
        dedup_pruned=dedup,
        failed_pruned=failed,
        stale_pruned=stale,
        saved_chars=saved,
    )
    if stats.total_pruned:
        logger.info(
            "context_hygiene session=%s pruned=%d dedup=%d failed=%d stale=%d saved_tokens=%d saved_chars=%d request_tokens=%d context_length=%d context_ratio=%.3f",
            session_id or "-",
            stats.total_pruned,
            stats.dedup_pruned,
            stats.failed_pruned,
            stats.stale_pruned,
            stats.saved_tokens_rough,
            stats.saved_chars,
            request_tokens,
            context_length,
            context_ratio,
        )
    return out, stats


__all__ = [
    "HygieneConfig",
    "HygieneStats",
    "apply_api_tool_output_hygiene",
    "config_from_mapping",
]
