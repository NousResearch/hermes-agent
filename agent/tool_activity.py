"""Session-static tool activity metadata and deterministic display summaries."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_ACTIVITY_REASON_KEY = "activity_reason"
_LEGACY_REASONING_KEY = "reasoning"
_REASONING_DESCRIPTION = (
    "Short display-safe goal or intent for this call, in present tense; "
    "12 words or fewer; do not repeat paths or commands; no period"
)
_MAX_REASON_LENGTH = 160
_MAX_SUMMARY_LENGTH = 240
_URL_RE = re.compile(
    r"(?<![a-z0-9_])(?:[a-z][a-z0-9+.-]*://|//)[^\s<>\"']+",
    re.IGNORECASE,
)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:token|secret|pass(?:word|wd)?|credential|signature|cookie|session|"
    r"auth(?:orization)?|api[-_]?key|access[-_]?key|(?:^|[-_.])sig(?:$|[-_.]))",
    re.IGNORECASE,
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"(?:[a-z][a-z0-9_.-]*(?:token|secret|pass(?:word|wd)?|credential|signature|cookie|session|"
    r"auth(?:orization)?|api[-_]?key|access[-_]?key)[a-z0-9_.-]*)|sig"
    r")\s*[:=]\s*([^\s&,;]+)"
)
_AUTH_HEADER_RE = re.compile(r"(?i)\b(authorization|proxy-authorization|cookie|set-cookie)\s*:\s*[^\r\n]+")
_AUTH_ASSIGNMENT_RE = re.compile(
    r"(?i)\b((?:proxy-)?authorization)\s*=\s*"
    r"(?:(?:bearer|basic|token)\s+[^\s,;]+|digest\s+[^\r\n]+)"
)
_DIGEST_AUTH_RE = re.compile(r"(?i)\bdigest\s+(?=[a-z][a-z0-9_-]*\s*=)[^\r\n]+")
_AUTH_SCHEME_RE = re.compile(r"(?i)\b(bearer|basic|token)\s+(?!\[REDACTED\])[^\s,;]+")


def _redact_query_pair(key: str, value: str) -> tuple[str, str]:
    # ``parse_qsl`` decodes percent escapes before splitting.  A payload such
    # as ``token%3Dsecret`` therefore arrives as a key containing the embedded
    # delimiter and value.  Never re-emit that embedded value as part of the
    # key when its prefix is sensitive.
    for delimiter in ("=", ":"):
        prefix, separator, _embedded_value = key.partition(delimiter)
        if separator and _SENSITIVE_KEY_RE.search(prefix):
            return prefix, "[REDACTED]"
    if _SENSITIVE_KEY_RE.search(key):
        return key, "[REDACTED]"
    return key, value


def _redact_url(match: re.Match[str]) -> str:
    raw = match.group(0)
    try:
        parsed = urlsplit(raw)
        host = parsed.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = f":{parsed.port}" if parsed.port is not None else ""
        except ValueError:
            port = ""
        netloc = f"{host}{port}"
        query = urlencode(
            [
                _redact_query_pair(key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        fragment = "[REDACTED]" if _SENSITIVE_KEY_RE.search(parsed.fragment) else parsed.fragment
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, fragment))
    except Exception:
        return "[REDACTED_URL]"


def redact_activity_text(value: Any) -> str:
    """Apply the strict redaction required for user-visible activity metadata."""
    text = str(value or "")
    # Structural forms must be removed before the generic scanner rewrites only
    # the assignment value (for example ``authorization=Bearer``) and leaves the
    # actual credential after the following space.
    text = _URL_RE.sub(_redact_url, text)
    text = _AUTH_HEADER_RE.sub(lambda match: f"{match.group(1)}: [REDACTED]", text)
    text = _AUTH_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    text = _DIGEST_AUTH_RE.sub("Digest [REDACTED]", text)
    text = _AUTH_SCHEME_RE.sub(lambda match: f"{match.group(1)} [REDACTED]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    try:
        return redact_sensitive_text(text, force=True)
    except Exception:
        # Presentation redaction is fail-closed: an unavailable or faulty
        # redactor must omit display metadata rather than expose raw text.
        logger.warning("Activity redaction failed; omitting presentation text")
        return ""


def redact_activity_args(value: Any, *, _key: str | None = None) -> Any:
    """Return a recursively sanitized copy of presentation-facing tool args."""
    if _key is not None and _SENSITIVE_KEY_RE.search(_key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            key: redact_activity_args(item, _key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_activity_args(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_activity_args(item) for item in value)
    if isinstance(value, str):
        return redact_activity_text(value)
    return value


def sanitize_persisted_tool_activity(metadata: Any) -> dict[str, Any] | None:
    """Whitelist and re-sanitize replayable activity metadata at the storage boundary."""
    if not isinstance(metadata, dict):
        return None

    safe: dict[str, Any] = {}
    for key, limit in (("reason", _MAX_REASON_LENGTH), ("summary", _MAX_SUMMARY_LENGTH)):
        raw = metadata.get(key)
        if not isinstance(raw, str):
            continue
        value = " ".join(redact_activity_text(raw).split())[:limit].strip()
        if value:
            safe[key] = value

    # Activity persistence is opt-in: status/duration alone must not create
    # replay metadata when neither display field was enabled.
    if not safe:
        return None

    status = metadata.get("status")
    if status in {"running", "completed", "failed", "blocked", "cancelled", "timeout"}:
        safe["status"] = status
    if isinstance(metadata.get("is_error"), bool):
        safe["is_error"] = metadata["is_error"]
    duration = metadata.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        duration_value = float(duration)
        if math.isfinite(duration_value) and duration_value >= 0:
            safe["duration_seconds"] = duration_value
    return safe


def augment_tool_schemas(
    tool_schemas: list[dict],
    *,
    enabled: bool,
    activity_tool_names: set[str] | None = None,
) -> list[dict]:
    """Return a cloned schema list with the optional display-reason field.

    This is intentionally a session-construction transform: callers snapshot the
    returned list and never re-run it during a conversation, preserving cache
    stability. Malformed schemas and incompatible collisions are left unchanged.
    """
    if not enabled:
        return tool_schemas
    augmented = copy.deepcopy(tool_schemas)
    for tool in augmented:
        function = tool.get("function") if isinstance(tool, dict) else None
        params = function.get("parameters") if isinstance(function, dict) else None
        if not isinstance(params, dict) or params.get("type") not in (None, "object"):
            continue
        tool_name = function.get("name", "unknown") if isinstance(function, dict) else "unknown"
        properties = params.get("properties")
        if not isinstance(properties, dict):
            continue
        required = params.get("required")
        if required is None:
            required = []
        if not isinstance(required, list):
            logger.warning(
                "Skipping required tool-reason augmentation for %s: malformed required list",
                tool_name,
            )
            continue
        existing = properties.get(_ACTIVITY_REASON_KEY)
        if existing is not None:
            if activity_tool_names is not None and tool_name in activity_tool_names:
                continue
            logger.warning(
                "Skipping tool-reason augmentation for %s: pre-existing activity_reason property",
                tool_name,
            )
            continue
        # ``reasoning`` was Hermes' legacy display field, but it is also a valid
        # native tool argument. A newly constructed session cannot distinguish
        # those meanings safely, so leave native schemas and dispatch untouched.
        if _LEGACY_REASONING_KEY in properties:
            continue
        reason_schema = {"type": "string", "description": _REASONING_DESCRIPTION}
        # Dict insertion order controls JSON-schema/UI presentation order.
        params["properties"] = {_ACTIVITY_REASON_KEY: reason_schema, **{
            key: value for key, value in properties.items() if key != _ACTIVITY_REASON_KEY
        }}
        params["required"] = [
            _ACTIVITY_REASON_KEY,
            *[key for key in required if key != _ACTIVITY_REASON_KEY],
        ]
        if activity_tool_names is not None and isinstance(tool_name, str) and tool_name:
            activity_tool_names.add(tool_name)
    return augmented


def extract_tool_reasoning(args: dict[str, Any], *, enabled: bool = True) -> str | None:
    """Remove the display-only reason from dispatch arguments and sanitize it.

    ``activity_reason`` is the current transport field. ``reasoning`` remains a
    fallback for conversations whose session-static schemas predate the rename.
    Presence, rather than truthiness, selects the new field so a tool's native
    ``reasoning`` argument is never consumed when both names exist.
    """
    if not enabled or not isinstance(args, dict):
        return None
    if _ACTIVITY_REASON_KEY in args:
        raw = args.pop(_ACTIVITY_REASON_KEY)
    else:
        raw = args.pop(_LEGACY_REASONING_KEY, None)
    if not isinstance(raw, str):
        return None
    text = " ".join(raw.split()).rstrip(".")
    if not text:
        return None
    return redact_activity_text(" ".join(text.split()[:12])[:_MAX_REASON_LENGTH])


def _json_result(raw_result: Any) -> Any:
    if not isinstance(raw_result, str):
        return raw_result
    try:
        return json.loads(raw_result)
    except (TypeError, ValueError):
        # Some tools append a display hint after their JSON payload. Parse only
        # the leading JSON value; summaries consume structural counts and never
        # expose either the payload body or trailing text.
        try:
            data, _end = json.JSONDecoder().raw_decode(raw_result.lstrip())
            return data
        except (TypeError, ValueError):
            return None


def _line_count(text: str) -> int:
    return len(text.splitlines())


def _first_error(raw_result: Any, data: Any) -> str | None:
    error = data.get("error") if isinstance(data, dict) else None
    if error is None and isinstance(data, dict) and data.get("success") is False:
        error = data.get("message")
    if error is None and isinstance(raw_result, str) and raw_result.lstrip().lower().startswith("error"):
        error = raw_result
    if error is None:
        return None
    return redact_activity_text(" ".join(str(error).splitlines()[:1]))


def _count_result(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    for key in ("total_count", "count", "total"):
        value = data.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    for key in ("results", "items", "data"):
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _safe_int(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _quantity(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _structured_list_count(data: Any, *keys: str) -> int | None:
    """Return a count only from an explicitly list-shaped result.

    Display summaries must not infer arbitrary units from serialized line
    counts.  Callers provide the domain keys whose list lengths are meaningful
    for that tool (skills, images, entities, and so on).
    """
    if isinstance(data, list):
        return len(data)
    if not isinstance(data, dict):
        return None
    explicit = _safe_int(data.get("count"))
    if explicit is not None:
        return explicit
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _todo_summary(name: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    state = data.get("summary")
    if not isinstance(state, dict):
        return ""
    total = _safe_int(state.get("total"))
    if total is None:
        return ""
    parts = [_quantity(total, "task")]
    for key, label in (
        ("in_progress", "active"),
        ("pending", "pending"),
        ("completed", "done"),
        ("cancelled", "cancelled"),
    ):
        count = _safe_int(state.get(key))
        if count:
            parts.append(f"{count} {label}")
    return f"{name}: " + " · ".join(parts)


def _session_search_summary(name: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    mode = data.get("mode")
    if mode == "read":
        count = _safe_int(data.get("message_count"))
        return f"{name}: {_quantity(count, 'message')}" if count is not None else ""
    if mode == "scroll":
        count = _structured_list_count(data, "messages")
        return f"{name}: {_quantity(count, 'message')}" if count is not None else ""
    if mode in {"browse", "discover"}:
        count = _structured_list_count(data, "results")
        return f"{name}: {_quantity(count, 'session')}" if count is not None else ""
    return ""


def _process_summary(name: str, args: dict[str, Any], data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if args.get("action") == "list":
        count = _structured_list_count(data, "processes")
        return f"{name}: {_quantity(count, 'process', 'processes')}" if count is not None else ""
    exit_code = data.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return f"{name}: exit {exit_code}"
    state = data.get("status")
    if isinstance(state, str) and state in {
        "running", "exited", "killed", "timeout", "interrupted", "already_exited",
    }:
        return f"{name}: {state.replace('_', ' ')}"
    return ""


def _delegate_summary(name: str, data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    count = _safe_int(data.get("count"))
    if data.get("status") == "dispatched" and count is not None:
        return f"{name}: {_quantity(count, 'task')} dispatched"
    results = data.get("results")
    if isinstance(results, list):
        return f"{name}: {_quantity(len(results), 'task')} finished"
    return ""


_LIST_TOOL_SUMMARIES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "browser_get_images": (("images",), "image", None),
    "ha_list_entities": (("entities", "results"), "entity", "entities"),
    "ha_list_services": (("services", "results"), "service", None),
    "mcp__qmd__list_resources": (("resources", "results"), "resource", None),
    "skills_list": (("skills",), "skill", None),
}


def _list_tool_summary(name: str, data: Any) -> str:
    keys, singular, plural = _LIST_TOOL_SUMMARIES[name]
    count = _structured_list_count(data, *keys)
    return f"{name}: {_quantity(count, singular, plural)}" if count is not None else ""


def _web_result_count(data: Any) -> int | None:
    count = _count_result(data)
    if count is not None:
        return count
    if isinstance(data, dict):
        nested = data.get("data")
        if isinstance(nested, dict):
            for key in ("web", "results", "items"):
                value = nested.get(key)
                if isinstance(value, list):
                    return len(value)
    return None


def _read_summary(name: str, args: dict[str, Any], data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    if data.get("status") == "unchanged" and data.get("content_returned") is False:
        return f"{name}: unchanged"

    content = data.get("content")
    shown = _line_count(content) if isinstance(content, str) else None
    total = _safe_int(data.get("total_lines"))
    offset = _safe_int(args.get("offset"), minimum=1) or 1
    next_offset = _safe_int(data.get("next_offset"), minimum=1)

    if shown is None:
        if total is not None:
            return f"{name}: {_quantity(total, 'line')} total"
        return ""

    amount = _quantity(shown, "line")
    partial = total is not None and (
        offset != 1 or shown < total or bool(data.get("truncated"))
    )
    if partial and shown > 0:
        end = offset + shown - 1
        detail = f"{offset}–{end} of {total}"
        if next_offset is not None:
            detail += f"; next {next_offset}"
        return f"{name}: {amount} ({detail})"
    if partial and total is not None:
        return f"{name}: {amount} (of {total})"
    return f"{name}: {amount}"


def _write_summary(name: str, args: dict[str, Any], data: Any) -> str:
    content = args.get("content")
    line_count = _line_count(content) if isinstance(content, str) else None

    byte_count = None
    if isinstance(data, dict):
        byte_count = _safe_int(data.get("bytes_written"))
    if byte_count is None and isinstance(content, str):
        byte_count = len(content.encode("utf-8"))

    parts: list[str] = []
    if line_count is not None:
        parts.append(_quantity(line_count, "line"))
    if byte_count is not None:
        parts.append(_quantity(byte_count, "byte"))
    return f"{name}: {', '.join(parts)}" if parts else ""


def _search_file_count(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    counts = data.get("counts")
    if isinstance(counts, dict):
        return len(counts)
    files = data.get("files")
    if isinstance(files, list):
        return len(files)
    matches = data.get("matches")
    if isinstance(matches, list):
        paths = {
            item.get("path")
            for item in matches
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        return len(paths)
    dense = data.get("matches_text")
    if isinstance(dense, str):
        # SearchResult's dense format emits path headers without indentation
        # and match rows with two leading spaces. Count headers only.
        return sum(1 for line in dense.splitlines() if line and not line[:1].isspace())
    return None


def _search_files_summary(name: str, args: dict[str, Any], data: Any) -> str:
    count = _count_result(data)
    if count is None:
        return ""
    file_count = _search_file_count(data)
    file_semantics = args.get("target") == "files" or args.get("output_mode") == "files_only"
    if file_semantics:
        summary = f"{name}: {_quantity(count, 'file')}"
    else:
        summary = f"{name}: {_quantity(count, 'match', 'matches')}"
        if file_count is not None:
            summary += f" in {_quantity(file_count, 'file')}"
    if isinstance(data, dict) and data.get("truncated"):
        summary += " (truncated)"
    return summary


def _diff_stats(diff_text: str) -> tuple[int, int]:
    added = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    deleted = sum(
        1
        for line in diff_text.splitlines()
        if line.startswith("-") and not line.startswith("---")
    )
    return added, deleted


def _changed_file_count(data: Any, diff_text: str) -> int | None:
    paths: set[str] = set()
    if isinstance(data, dict):
        for key in ("files_modified", "files_created", "files_deleted"):
            values = data.get(key)
            if isinstance(values, list):
                paths.update(value for value in values if isinstance(value, str))
    if paths:
        return len(paths)

    headers: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith(("*** Update File: ", "*** Add File: ", "*** Delete File: ")):
            headers.add(line.split(": ", 1)[1])
        elif line.startswith("diff --git "):
            headers.add(line)
        elif line.startswith("+++ "):
            path = line[4:].strip()
            if path != "/dev/null":
                headers.add(path)
    return len(headers) or None


def _patch_summary(name: str, args: dict[str, Any], text: str, data: Any) -> str:
    diff_text = data.get("diff") if isinstance(data, dict) else None
    if not isinstance(diff_text, str) or not diff_text:
        diff_text = next(
            (args[key] for key in ("patch", "diff") if isinstance(args.get(key), str)),
            text,
        )
    added, deleted = _diff_stats(diff_text)
    file_count = _changed_file_count(data, diff_text)
    if added or deleted:
        summary = f"{name}: +{added}/-{deleted}"
        if file_count == 1:
            summary += " in 1 file"
        elif file_count is not None:
            summary += f" across {file_count} files"
        return summary
    if file_count == 1:
        return f"{name}: applied to 1 file"
    if file_count is not None:
        return f"{name}: applied to {file_count} files"
    return f"{name}: applied"


def summarize_tool_result(
    tool_name: str,
    args: dict[str, Any] | None,
    raw_result: Any,
    *,
    duration_s: float | None = None,
    status: str | None = None,
    error: str | None = None,
) -> str:
    """Build a display-safe one-line summary without changing raw tool output."""
    name = str(tool_name or "tool")
    text = raw_result if isinstance(raw_result, str) else str(raw_result or "")
    data = _json_result(raw_result)
    error_text = error or _first_error(raw_result, data)
    failed = status in {"error", "failed", "blocked"} or bool(error_text)
    if name in {"terminal", "execute_code", "exec_command"} and isinstance(data, dict) and data.get("exit_code") not in (None, 0):
        failed = True
        error_text = error_text or f"exit {data['exit_code']}"
    if failed:
        detail = redact_activity_text(" ".join(str(error_text or status or "error").splitlines()[:1]))
        return _cap(f"{name}: error: {detail}")

    safe_args = args if isinstance(args, dict) else {}
    if name == "read_file":
        summary = _read_summary(name, safe_args, data)
    elif name == "write_file":
        summary = _write_summary(name, safe_args, data)
    elif name in {"patch", "edit", "apply_patch", "diff"}:
        summary = _patch_summary(name, safe_args, text, data)
    elif name in {"terminal", "execute_code", "exec_command"}:
        exit_code = data.get("exit_code") if isinstance(data, dict) else None
        tail = f" exit {exit_code}" if exit_code is not None else " done"
        summary = f"{name}:{tail}"
        if duration_s is not None:
            summary += f" in {duration_s:.1f}s"
    elif name == "search_files":
        summary = _search_files_summary(name, safe_args, data)
    elif name == "web_search":
        count = _web_result_count(data)
        summary = f"{name}: {_quantity(count, 'result')}" if count is not None else ""
    elif name == "web_extract":
        count = _web_result_count(data)
        summary = f"{name}: {_quantity(count, 'page')}" if count is not None else ""
    elif name in _LIST_TOOL_SUMMARIES:
        summary = _list_tool_summary(name, data)
    elif name == "todo":
        summary = _todo_summary(name, data)
    elif name == "session_search":
        summary = _session_search_summary(name, data)
    elif name == "process":
        summary = _process_summary(name, safe_args, data)
    elif name == "delegate_task":
        summary = _delegate_summary(name, data)
    elif name == "cronjob" and safe_args.get("action") == "list":
        count = _structured_list_count(data, "jobs")
        summary = f"{name}: {_quantity(count, 'job')}" if count is not None else ""
    else:
        # The card already carries the tool name, status, duration, and call
        # context.  Unknown and acknowledgement-only tools have no trustworthy
        # domain unit, so a serialized line count is noise rather than a result.
        summary = ""
    return _cap(redact_activity_text(summary))


def _cap(text: str) -> str:
    one_line = " ".join(redact_activity_text(text).splitlines())
    return one_line if len(one_line) <= _MAX_SUMMARY_LENGTH else one_line[: _MAX_SUMMARY_LENGTH - 3] + "..."
