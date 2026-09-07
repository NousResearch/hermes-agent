"""Final provider-boundary enforcement for source-bound LLM egress."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shlex
from hashlib import sha256
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from agent.llm_egress_firewall import (
    AuthorizedEgress,
    CodexReasoningReplaySegment,
    EgressBlocked,
    LLMEgressFirewall,
    LiteralSegment,
    OutboundText,
    SanitizedSegment,
    SourceBoundSegment,
    SourcePresentationSegment,
    SourceGrant,
    TypedOutboundRequest,
    UntrustedProvenanceSegment,
    ValidatedToolSyntaxSegment,
    DestinationClass,
    GeneratedContextKey,
    GeneratedContextSegment,
    classify_destination,
    source_grant_digest,
    static_literal_sha256,
    validate_sanitized_text,
    content_free_violation_locations,
    redact_remote_unsafe_text,
    validate_tool_syntax,
)
from agent.message_sanitization import tool_result_id_variants
from agent.redact import redact_sensitive_text
from agent.source_provenance import DEFAULT_POLICY_DIGEST, SourceProvenanceRegistry


# Timeout is a non-content SDK control. Header/query values remain in the
# authorized JSON body so credentials or other caller-controlled text cannot
# be appended after the firewall receipt is written.
_SDK_CONTROL_KEYS = frozenset({"timeout"})
_INTERNAL_EGRESS_KEYS = frozenset({"_hermes_source_provenance"})
_PROTOCOL_LITERAL_FIELDS = frozenset({"role", "type"})
_TOOL_PROTOCOL_IDENTIFIER_FIELDS = frozenset(
    {"id", "call_id", "tool_call_id", "response_item_id"}
)
_PROTOCOL_LITERAL_VALUES = frozenset({
    "assistant",
    "computer_call_output",
    "developer",
    "function_call",
    "function_call_output",
    "input_image",
    "input_text",
    "output_text",
    "reasoning",
    "system",
    "tool",
    "user",
})
_PROTECTED_REMOTE_PROVIDERS = frozenset({
    "anthropic",
    "openai-codex",
    "nous",
    "nous-portal",
    "nousresearch",
})
logger = logging.getLogger(__name__)

_VALIDATED_SYNTAX_TOOL_NAMES = frozenset({"terminal"})
_REMOTE_KANBAN_PROJECTION_TOOL_NAMES = frozenset({"kanban_show"})
_REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES = frozenset({"terminal"})
_REMOTE_KANBAN_SEARCH_PROJECTION_TOOL_NAMES = frozenset({"search_files"})
_REMOTE_KANBAN_READ_FILE_PROJECTION_TOOL_NAMES = frozenset({"read_file"})
_REMOTE_KANBAN_WEB_REPLAY_TOOL_NAMES = frozenset({"web_extract", "web_search"})
_REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES = frozenset({"patch", "write_file"})
_REMOTE_KANBAN_READONLY_REPLAY_TOOL_NAMES = frozenset(
    {
        "kanban_show",
        "search_files",
        "read_file",
        "web_extract",
    }
)
_GITHUB_LIST_TERMINAL_MAX_ROWS = 100
_GITHUB_LIST_TERMINAL_MAX_ITEM_BYTES = 512
_GITHUB_LIST_TERMINAL_MAX_OUTPUT_BYTES = 10_240
_GIT_GREP_TERMINAL_MAX_MATCHES = 200
_GITHUB_API_EXTRACT_ARGUMENT_REPLAY = (
    '{"urls":["https://api.github.com/repos/<owner>/<repo>/<list>"]}'
)
_GITHUB_API_CURL_ARGUMENT_REPLAY = (
    '{"command":"curl GitHub REST list (details omitted)"}'
)
_GITHUB_PLAIN_LIST_OUTPUT_REPLAY = (
    "GitHub list output omitted; use --json for bounded fields."
)
_REJECTED_TERMINAL_COMMAND_REPLAY = json.dumps(
    {"command": "<rejected terminal command omitted>"}, separators=(",", ":")
)
_GIT_WORKSPACE_DIAGNOSTIC_REPLAY = (
    "git workspace diagnostic completed locally; raw paths and commit subjects "
    "were omitted from remote replay."
)
_READ_FILE_REPLAY_ELISION = (
    "read_file completed locally, but its raw content cannot be replayed on "
    "this protected route. Request only the needed narrow range again."
)
_STRUCTURED_SEARCH_REPLAY_ELISION = (
    "search completed locally; structured output omitted from remote replay."
)
_FILE_MUTATION_REPLAY_ELISION = (
    "local file mutation completed; raw source and diff omitted from remote replay. "
    "Inspect git diff and status for the exact result."
)
_FILE_MUTATION_ARGUMENT_REPLAY = json.dumps(
    {"path": "<local-file>", "content": "omitted from remote replay"},
    separators=(",", ":"),
)
_REMOTE_KANBAN_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key)\s*[:=]\s*[^\s,}\"']+"
)
_REMOTE_KANBAN_PROJECTION_ELISION = (
    "kanban_show completed locally. The bounded task assignment is already "
    "present in your worker context; do not request or repeat the raw board "
    "record remotely. Continue with the assigned work or use a lifecycle tool."
)


def _project_bound_kanban_show(value: str) -> GeneratedContextSegment:
    """Expose only the redacted current assignment needed by a remote worker."""

    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return GeneratedContextSegment(_REMOTE_KANBAN_PROJECTION_ELISION)
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return GeneratedContextSegment(_REMOTE_KANBAN_PROJECTION_ELISION)

    projection = {
        "task": {
            key: task[key]
            for key in ("title", "body", "status", "workspace_access")
            if key in task
        },
        "parents": payload.get("parents", []),
        "children": payload.get("children", []),
        "worker_instruction": (
            "Use the dispatcher-assigned current workspace. Do not invent or search "
            "for alternate worktrees; report an unresolved assignment and stop."
        ),
    }
    safe = redact_remote_unsafe_text(
        redact_sensitive_text(json.dumps(projection, sort_keys=True), force=True)
    )
    safe = _REMOTE_KANBAN_SECRET_ASSIGNMENT.sub(r"\1=<redacted>", safe)
    return GeneratedContextSegment(
        "kanban_show completed locally. Bounded sanitized task projection:\n" + safe
    )


def _project_bound_search_files(value: str) -> GeneratedContextSegment:
    """Retain search locations without replaying matched source bytes.

    ``search_files`` necessarily returns excerpts of local source.  A protected
    worker may use the count and (when compact) the file/line locations to
    choose a narrow ``read_file`` request, whose exact bytes are independently
    source-provenance bound.  Never parse or replay ``matches_text``: it is a
    dense display format containing source content.
    """

    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, Mapping):
        return GeneratedContextSegment(
            "search_files completed locally. Its raw result was omitted from the "
            "remote replay; narrow the search or use read_file for a known path."
        )

    projection: dict[str, Any] = {"search_files_projection": "locations-v1"}
    total_count = payload.get("total_count")
    if isinstance(total_count, int) and not isinstance(total_count, bool):
        projection["total_count"] = max(0, min(total_count, 1_000_000))
    if payload.get("truncated") is True:
        projection["truncated"] = True

    raw_files = payload.get("files")
    if isinstance(raw_files, list):
        files: list[str] = []
        for raw_path in raw_files[:100]:
            if not isinstance(raw_path, str) or not raw_path or len(raw_path) > 512:
                continue
            normalized = raw_path[2:] if raw_path.startswith("./") else raw_path
            path = PurePosixPath(normalized)
            if (
                path.is_absolute()
                or "\\" in normalized
                or any(
                    part in {"", ".", ".."}
                    or (part.startswith(".") and part != ".github")
                    for part in path.parts
                )
            ):
                continue
            safe_path = redact_remote_unsafe_text(
                redact_sensitive_text(path.as_posix(), force=True)
            )
            if safe_path == path.as_posix():
                files.append(safe_path)
        if files:
            projection["files"] = files

    raw_matches = payload.get("matches")
    if isinstance(raw_matches, list):
        matches: list[dict[str, Any]] = []
        for raw_match in raw_matches[:100]:
            if not isinstance(raw_match, Mapping):
                continue
            path = raw_match.get("path")
            line = raw_match.get("line")
            if not isinstance(path, str) or not isinstance(line, int) or isinstance(line, bool):
                continue
            safe_path = redact_remote_unsafe_text(
                redact_sensitive_text(path, force=True)
            )
            matches.append({"path": safe_path, "line": max(1, min(line, 10_000_000))})
        if matches:
            projection["matches"] = matches

    safe = redact_remote_unsafe_text(
        redact_sensitive_text(
            json.dumps(projection, ensure_ascii=False, separators=(",", ":")),
            force=True,
        )
    )
    return GeneratedContextSegment(safe)


def _project_web_search_replay(value: str) -> SanitizedSegment:
    """Keep bounded public result identity, never raw page/search excerpts."""

    start = value.find("{") if isinstance(value, str) else -1
    end = value.rfind("}") if isinstance(value, str) else -1
    try:
        payload = json.loads(value[start : end + 1]) if 0 <= start <= end else None
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = None

    raw_results: Any = None
    if isinstance(payload, Mapping):
        search = payload.get("search")
        if isinstance(search, Mapping):
            raw_results = search.get("web")
        if raw_results is None:
            raw_results = payload.get("results")

    results: list[dict[str, str]] = []
    if isinstance(raw_results, list):
        for raw in raw_results[:20]:
            if not isinstance(raw, Mapping):
                continue
            projected: dict[str, str] = {}
            for key in ("url", "title"):
                item = raw.get(key)
                if not isinstance(item, str):
                    continue
                candidate = redact_remote_unsafe_text(
                    redact_sensitive_text(
                        item,
                        force=True,
                        redact_url_credentials=True,
                    )
                )
                try:
                    projected[key] = validate_sanitized_text(candidate, max_bytes=2_048)
                except (TypeError, ValueError):
                    continue
            if projected:
                results.append(projected)

    projection = {
        "kind": "web results",
        "results": results,
        "raw excerpts omitted": True,
    }
    rendered = json.dumps(projection, ensure_ascii=False, separators=(",", ":"))
    return SanitizedSegment(validate_sanitized_text(rendered))


_APPLICATION_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:t_[0-9a-f]{8}|[0-9a-f]{40}|[0-9a-f]{64}|"
    r"[a-z][a-z0-9]{0,31}(?:[_-][a-z][a-z0-9]{0,31}){1,7}"
    r"(?::v[0-9]{1,3})?)(?![A-Za-z0-9_-])"
)
_VERIFIED_DIAGNOSTIC_ATOM = re.compile(
    r"(?<![A-Za-z0-9_])(?:PASS|WARN|SUMMARY|REQUIREMENTS|AVAILABILITY|"
    r"HANDLING|VERIFICATION|[0-9]{1,10}|0x[0-9a-fA-F]{1,16}|"
    r"_?[A-Za-z][A-Za-z0-9]{0,63}(?:_[A-Za-z0-9]{1,64}){1,7})"
    r"(?![A-Za-z0-9_])"
)
_CREDENTIAL_ENV_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_KEY",
    "_PASSWORD",
    "_CREDENTIAL",
)

_PRIVATE_PATH_IN_TEXT = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"/(?:Users|home|private|var/folders|root|Volumes)/[^\s\"'`)]+"
    r"|~(?:/|\\)[^\s\"'`)]+"
    r"|[A-Za-z]:\\+(?:Users|Documents and Settings)\\+[^\s\"'`)]+"
    r")",
    re.IGNORECASE,
)


def _sanitize_protected_kanban_body(value: Any) -> Any:
    """Remove host paths from protected Kanban tool results before typing.

    This deliberately does not rewrite secrets or arbitrary encoded content;
    those remain visible to the fail-closed firewall scans and are denied.
    """

    if isinstance(value, str):
        text = value
        for name in (
            "HERMES_KANBAN_CLAIM_LOCK",
            "HERMES_KANBAN_RUN_ID",
            "HERMES_SESSION_ID",
            "HERMES_STREAM_STALE_GIVEUP",
            "HERMES_TURN_LEASE_TIMEOUT",
        ):
            raw = os.environ.get(name)
            if raw:
                text = re.sub(
                    rf"(?m)^(?P<label>{re.escape(name)}=){re.escape(raw)}$",
                    rf"\g<label>${name}",
                    text,
                )
        replacements = (
            (os.environ.get("HERMES_KANBAN_WORKSPACE"), "."),
            (os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT"), "$HERMES_KANBAN_WORKSPACES_ROOT"),
            (os.environ.get("HERMES_KANBAN_DB"), "$HERMES_KANBAN_DB"),
            (os.environ.get("HERMES_CONTROL_HOME"), "$HERMES_CONTROL_HOME"),
            (os.environ.get("HERMES_HOME"), "$HERMES_PROFILE_HOME"),
        )
        for raw, token in sorted(
            ((raw, token) for raw, token in replacements if raw),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            text = text.replace(raw, token)
        return _PRIVATE_PATH_IN_TEXT.sub("<private-path>", text)
    if isinstance(value, Mapping):
        return {
            _sanitize_protected_kanban_body(key): _sanitize_protected_kanban_body(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_protected_kanban_body(item) for item in value]
    return value


def provider_uses_egress_firewall(provider: Any) -> bool:
    """Return whether an exact configured provider owns a protected remote lane."""

    return str(provider or "").strip().lower() in _PROTECTED_REMOTE_PROVIDERS


def _exact_provider_secret_values() -> tuple[str, ...]:
    """Snapshot exact profile and credential environment values before send.

    This is the final provider-boundary interlock for the exact applied-secret
    class tracked in #77165; shape-based redaction remains an independent scan.
    """

    try:
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
    except Exception:
        home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    try:
        from hermes_cli.env_loader import get_secret_source_values

        values = list(get_secret_source_values(home).values())
    except Exception:
        values = []
    values.extend(
        value
        for name, value in os.environ.items()
        if value and name.upper().endswith(_CREDENTIAL_ENV_SUFFIXES)
    )
    return tuple(
        dict.fromkeys(value for value in values if isinstance(value, str) and value)
    )


def _read_grant_text(grant: SourceGrant) -> str | None:
    try:
        lines = Path(grant.canonical_path).read_bytes().splitlines(keepends=True)
        return b"".join(lines[grant.line_start - 1 : grant.line_end]).decode("utf-8")
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return None


def _grant_texts(grants: Sequence[SourceGrant]) -> tuple[tuple[str, SourceGrant], ...]:
    unique: dict[str, SourceGrant] = {}
    for grant in grants:
        if not isinstance(grant, SourceGrant):
            continue
        text = _read_grant_text(grant)
        if text:
            unique.setdefault(text, grant)
    return tuple(sorted(unique.items(), key=lambda item: (-len(item[0]), item[0])))


def _approved_sanitized(text: str, *, cap: int) -> SanitizedSegment:
    # Admission is finalized by LLMEgressFirewall so every denial is reported
    # as its content-free EgressBlocked decision. Keep only the local type and
    # byte bound here; the firewall repeats secret/base64/path scans on the
    # rendered request immediately before dispatch.
    if not isinstance(text, str):
        raise TypeError("sanitized segment must be text")
    if cap <= 0 or len(text.encode("utf-8")) > cap:
        raise ValueError("sanitized segment exceeds byte cap")
    return SanitizedSegment(text)


def _split_utf8_chunks(text: str, cap: int) -> list[str]:
    """Split text into UTF-8-safe chunks no larger than ``cap`` bytes."""

    chunks: list[str] = []
    pending: list[str] = []
    pending_bytes = 0
    for character in text:
        character_bytes = len(character.encode("utf-8"))
        if character_bytes > cap:
            raise ValueError("sanitized segment exceeds byte cap")
        if pending and pending_bytes + character_bytes > cap:
            chunks.append("".join(pending))
            pending = []
            pending_bytes = 0
        pending.append(character)
        pending_bytes += character_bytes
    if pending:
        chunks.append("".join(pending))
    return chunks


def _approved_sanitized_segments(
    text: str,
    *,
    cap: int,
    allow_line_split: bool = False,
) -> list[SanitizedSegment]:
    """Admit one independently sourced text segment without cap laundering.

    Normal callers may provide multiple bounded messages or exact-grant-separated
    segments. Protected Kanban context has an additional deterministic source
    boundary: complete lines from the locally projected task payload are packed
    into independently bounded segments without changing provider-visible text.
    Oversized individual lines are split only at UTF-8 character boundaries;
    the firewall re-scans adjacent chunks as one logical span.
    """

    if not allow_line_split or len(text.encode("utf-8")) <= cap:
        return [_approved_sanitized(text, cap=cap)]

    segments: list[SanitizedSegment] = []
    pending = ""
    pending_bytes = 0
    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > cap:
            if pending:
                segments.append(_approved_sanitized(pending, cap=cap))
                pending = ""
                pending_bytes = 0
            segments.extend(
                _approved_sanitized(chunk, cap=cap)
                for chunk in _split_utf8_chunks(line, cap)
            )
            continue
        if pending and pending_bytes + line_bytes > cap:
            segments.append(_approved_sanitized(pending, cap=cap))
            pending = ""
            pending_bytes = 0
        pending += line
        pending_bytes += line_bytes
    if pending or not segments:
        segments.append(_approved_sanitized(pending, cap=cap))
    return segments


def _segment_text(
    text: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
    allow_line_split: bool = False,
) -> SanitizedSegment | SourceBoundSegment | OutboundText:
    matches: list[tuple[int, int, SourceGrant]] = []
    cursor = 0
    while cursor < len(text):
        chosen: tuple[int, int, SourceGrant] | None = None
        for granted_text, grant in grant_texts:
            start = text.find(granted_text, cursor)
            if start < 0:
                continue
            candidate = (start, start + len(granted_text), grant)
            if chosen is None or candidate[:2] < chosen[:2]:
                chosen = candidate
        if chosen is None:
            break
        matches.append(chosen)
        cursor = chosen[1]

    if not matches:
        sanitized = _approved_sanitized_segments(
            text,
            cap=sanitized_cap,
            allow_line_split=allow_line_split,
        )
        return sanitized[0] if len(sanitized) == 1 else OutboundText(tuple(sanitized))

    segments: list[SanitizedSegment | SourceBoundSegment] = []
    cursor = 0
    for start, end, grant in matches:
        if start > cursor:
            segments.extend(
                _approved_sanitized_segments(
                    text[cursor:start],
                    cap=sanitized_cap,
                    allow_line_split=allow_line_split,
                )
            )
        digest = source_grant_digest(grant)
        segments.append(SourceBoundSegment(digest))
        used_grants[digest] = grant
        cursor = end
    if cursor < len(text):
        segments.extend(
            _approved_sanitized_segments(
                text[cursor:],
                cap=sanitized_cap,
                allow_line_split=allow_line_split,
            )
        )
    return segments[0] if len(segments) == 1 else OutboundText(tuple(segments))


def _segment_protected_context(
    text: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
) -> SanitizedSegment | SourceBoundSegment | ValidatedToolSyntaxSegment | OutboundText:
    """Preserve exact text while typing narrow application-owned identifiers."""

    segments: list[SanitizedSegment | SourceBoundSegment | ValidatedToolSyntaxSegment] = []
    cursor = 0
    for match in _APPLICATION_IDENTIFIER_TOKEN.finditer(text):
        if match.start() > cursor:
            prefix = _segment_text(
                text[cursor : match.start()],
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
                allow_line_split=True,
            )
            segments.extend(prefix.segments if isinstance(prefix, OutboundText) else (prefix,))
        token = validate_tool_syntax(match.group(0), "application_identifier")
        segments.append(ValidatedToolSyntaxSegment(token, "application_identifier"))
        cursor = match.end()
    if cursor < len(text):
        suffix = _segment_text(
            text[cursor:],
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
            allow_line_split=True,
        )
        segments.extend(suffix.segments if isinstance(suffix, OutboundText) else (suffix,))
    if not segments:
        return _segment_text(
            text,
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
            allow_line_split=True,
        )
    return segments[0] if len(segments) == 1 else OutboundText(tuple(segments))


def _recognized_tool_call_ids(
    value: Any, tool_names: frozenset[str]
) -> frozenset[str]:
    """Bind a narrow output handling rule to an exact prior tool call."""

    recognized: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name in tool_names
            ):
                call_id = item.get("call_id") or item.get("id")
                if isinstance(call_id, str):
                    recognized.update(tool_result_id_variants(call_id))
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    call_id = call.get("call_id") or call.get("id")
                    if (
                        isinstance(function, Mapping)
                        and function.get("name") in tool_names
                        and isinstance(call_id, str)
                    ):
                        recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _recognized_syntax_tool_call_ids(value: Any) -> frozenset[str]:
    """Return preceding terminal calls eligible for strict syntax parsing."""

    return _recognized_tool_call_ids(value, _VALIDATED_SYNTAX_TOOL_NAMES)


def _scratch_read_file_tool_call_ids(value: Any) -> frozenset[str]:
    """Recognize worker scratch-file reads that have no source authority."""

    recognized: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
            arguments = (
                function.get("arguments") if isinstance(function, Mapping) else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if item.get("type") in {"function", "function_call"} and name == "read_file":
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                    path = parsed.get("path") if isinstance(parsed, Mapping) else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    path = None
                if isinstance(path, str) and path.startswith(("/tmp/", "/private/tmp/")) and isinstance(call_id, str):
                    recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _github_list_terminal_call_limits(value: Any) -> dict[str, int]:
    """Bind a small GitHub list projection to an exact preceding terminal call.

    GitHub list and issue-view JSON contain opaque database identifiers that
    are neither useful to a Kanban worker nor safe to replay remotely. This
    deliberately recognizes only bounded, literal ``gh issue|pr list`` or
    ``gh issue view`` JSON forms used by the White-Knight intake; every other
    terminal result follows the normal fail-closed path.
    """

    limits: dict[str, int] = {}

    def command_limit(arguments: Any) -> int | None:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if "--json" not in tokens:
            return None
        if tokens[:3] in (["gh", "issue", "view"], ["gh", "pr", "view"]):
            return 1
        if len(tokens) < 5 or tokens[:3] not in (
            ["gh", "issue", "list"],
            ["gh", "pr", "list"],
        ):
            return None
        for index, token in enumerate(tokens):
            raw_limit = (
                token.split("=", 1)[1]
                if token.startswith("--limit=")
                else tokens[index + 1]
                if token == "--limit" and index + 1 < len(tokens)
                else None
            )
            if raw_limit is None:
                continue
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError):
                return None
            return limit if 0 < limit <= _GITHUB_LIST_TERMINAL_MAX_ROWS else None
        return None

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            if item.get("type") in {"function", "function_call"} and direct_name == "terminal":
                arguments = (
                    direct_function.get("arguments")
                    if isinstance(direct_function, Mapping)
                    else item.get("arguments")
                )
                limit = command_limit(arguments)
                call_id = item.get("call_id") or item.get("id")
                if limit is not None and isinstance(call_id, str):
                    for variant in tool_result_id_variants(call_id):
                        limits[variant] = limit
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return limits


def _github_api_extract_call_limits(value: Any) -> dict[str, int]:
    """Bind GitHub REST list projections to exact ``web_extract`` calls.

    GitHub's public REST list responses contain database and node identifiers
    which do not help a Kanban worker assess an issue or pull request.  Admit
    only bounded ``issues`` and ``pulls`` list endpoints, and only when every
    URL in the extract call has that exact shape.  Arbitrary web content
    remains fail-closed.
    """

    limits: dict[str, int] = {}

    def extract_limit(arguments: Any) -> int | None:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            urls = parsed.get("urls") if isinstance(parsed, Mapping) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(urls, list) or not 1 <= len(urls) <= 5:
            return None
        row_limits: list[int] = []
        for url in urls:
            if not isinstance(url, str):
                return None
            parts = urlsplit(url)
            if (
                parts.scheme != "https"
                or parts.hostname != "api.github.com"
                or not re.fullmatch(r"/repos/[^/]+/[^/]+/(?:issues|pulls)", parts.path)
            ):
                return None
            query = parse_qs(parts.query, keep_blank_values=True)
            raw_per_page = query.get("per_page", ["30"])
            if len(raw_per_page) != 1:
                return None
            try:
                limit = int(raw_per_page[0])
            except (TypeError, ValueError):
                return None
            if not 0 < limit <= _GITHUB_LIST_TERMINAL_MAX_ROWS:
                return None
            row_limits.append(limit)
        return max(row_limits)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            if item.get("type") in {"function", "function_call"} and direct_name == "web_extract":
                arguments = (
                    direct_function.get("arguments")
                    if isinstance(direct_function, Mapping)
                    else item.get("arguments")
                )
                limit = extract_limit(arguments)
                call_id = item.get("call_id") or item.get("id")
                if limit is not None and isinstance(call_id, str):
                    for variant in tool_result_id_variants(call_id):
                        limits[variant] = limit
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return limits


def _github_api_curl_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize exact, read-only GitHub REST list ``curl`` commands.

    This recognition controls only what is replayed to a remote provider *after*
    the local terminal tool has run.  It does not authorize command execution.
    Headers, redirects to another command, output paths outside the workspace,
    and arbitrary shell fragments are deliberately excluded.
    """

    recognized: set[str] = set()

    def is_bounded_list_url(raw_url: str) -> bool:
        parts = urlsplit(raw_url)
        if (
            parts.scheme != "https"
            or parts.hostname != "api.github.com"
            or not re.fullmatch(r"/repos/[^/]+/[^/]+/(?:issues|pulls)", parts.path)
        ):
            return False
        raw_per_page = parse_qs(parts.query, keep_blank_values=True).get(
            "per_page", ["30"]
        )
        if len(raw_per_page) != 1:
            return False
        try:
            return 0 < int(raw_per_page[0]) <= _GITHUB_LIST_TERMINAL_MAX_ROWS
        except (TypeError, ValueError):
            return False

    def is_bounded_curl(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not tokens:
            return False
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token == "&&":
                if not segments[-1]:
                    return False
                segments.append([])
                continue
            if token in {";", "|", "||", "&"}:
                return False
            segments[-1].append(token)
        if not segments[-1]:
            return False
        for segment in segments:
            if not segment or segment[0] != "curl":
                return False
            urls = 0
            index = 1
            while index < len(segment):
                token = segment[index]
                if token in {"-o", "--output"}:
                    index += 1
                    if (
                        index >= len(segment)
                        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.json", segment[index])
                    ):
                        return False
                elif token.startswith("-"):
                    if token not in {"--fail", "--silent", "--show-error", "--location"} and not (
                        token.startswith("-") and set(token[1:]) <= {"f", "s", "S", "L"}
                    ):
                        return False
                elif is_bounded_list_url(token):
                    urls += 1
                else:
                    return False
                index += 1
            if urls != 1:
                return False
        return True

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name == "terminal"
                and is_bounded_curl(arguments)
                and isinstance(call_id, str)
            ):
                recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _plain_github_list_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize bounded plain ``gh issue|pr list`` fallback output."""

    recognized: set[str] = set()

    def is_plain_list(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if len(tokens) < 3 or tokens[:1] != ["gh"] or tokens[2:3] != ["list"]:
            return False
        if tokens[1] not in {"issue", "pr"} or "--json" in tokens:
            return False
        index = 3
        limit = 30
        while index < len(tokens):
            token = tokens[index]
            if token not in {"--repo", "--state", "--limit"} or index + 1 >= len(tokens):
                return False
            value = tokens[index + 1]
            if token == "--repo" and not re.fullmatch(r"[^/\s]+/[^/\s]+", value):
                return False
            if token == "--limit":
                try:
                    limit = int(value)
                except ValueError:
                    return False
            index += 2
        return 0 < limit <= _GITHUB_LIST_TERMINAL_MAX_ROWS

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name == "terminal"
                and is_plain_list(arguments)
                and isinstance(call_id, str)
            ):
                recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _combined_github_list_terminal_call_limits(value: Any) -> dict[str, int]:
    """Bind the standard workspace-status plus ``gh --json`` intake chain."""

    limits: dict[str, int] = {}

    def command_limit(arguments: Any) -> int | None:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token == "&&":
                if not segments[-1]:
                    return None
                segments.append([])
            else:
                segments[-1].append(token)
        if len(segments) < 3 or segments[:2] != [["pwd"], ["git", "status", "--short"]]:
            return None
        row_limits: list[int] = []
        for segment in segments[2:]:
            if len(segment) < 5 or segment[:3] not in (
                ["gh", "issue", "list"], ["gh", "pr", "list"]
            ) or "--json" not in segment:
                return None
            try:
                index = segment.index("--limit")
                limit = int(segment[index + 1])
            except (ValueError, IndexError):
                return None
            if not 0 < limit <= _GITHUB_LIST_TERMINAL_MAX_ROWS:
                return None
            row_limits.append(limit)
        return max(row_limits) if row_limits else None

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
            arguments = function.get("arguments") if isinstance(function, Mapping) else item.get("arguments")
            call_id = item.get("call_id") or item.get("id")
            limit = command_limit(arguments) if item.get("type") in {"function", "function_call"} and name == "terminal" else None
            if limit is not None and isinstance(call_id, str):
                for variant in tool_result_id_variants(call_id):
                    limits[variant] = limit
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return limits


def _project_combined_github_list_terminal_result(text: str, *, max_rows: int) -> str | None:
    """Project JSON arrays from the exact status-plus-GitHub intake chain."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_output, str):
        return None
    decoder = json.JSONDecoder()
    rows: list[Any] = []
    cursor = 0
    while True:
        start = raw_output.find("[", cursor)
        if start < 0:
            break
        try:
            decoded, cursor = decoder.raw_decode(raw_output, start)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, list):
            return None
        rows.extend(decoded)
    if not rows:
        return None
    return _project_github_list_terminal_result(
        json.dumps({"exit_code": wrapper.get("exit_code"), "output": json.dumps(rows)}),
        max_rows=max_rows,
    )


def _combined_github_view_terminal_call_limits(value: Any) -> dict[str, int]:
    """Bind a small chain of explicit GitHub issue/PR views to projection."""

    limits: dict[str, int] = {}

    def command_limit(arguments: Any) -> int | None:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token == "&&":
                if not segments[-1]:
                    return None
                segments.append([])
            else:
                segments[-1].append(token)
        if not 1 <= len(segments) <= 5:
            return None
        for segment in segments:
            if (
                len(segment) < 7
                or segment[:2] != ["gh", "issue"]
                or segment[2] != "view"
                or not segment[3].isdigit()
                or "--repo" not in segment
                or "--json" not in segment
            ):
                return None
            try:
                repo = segment[segment.index("--repo") + 1]
            except IndexError:
                return None
            if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
                return None
        return len(segments)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
            arguments = function.get("arguments") if isinstance(function, Mapping) else item.get("arguments")
            call_id = item.get("call_id") or item.get("id")
            limit = command_limit(arguments) if item.get("type") in {"function", "function_call"} and name == "terminal" else None
            if limit is not None and isinstance(call_id, str):
                for variant in tool_result_id_variants(call_id):
                    limits[variant] = limit
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return limits


def _project_combined_github_view_terminal_result(text: str, *, max_rows: int) -> str | None:
    """Project concatenated GitHub view objects without bodies/comments."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_output, str):
        return None
    decoder = json.JSONDecoder()
    rows: list[Any] = []
    cursor = 0
    while True:
        start = raw_output.find("{", cursor)
        if start < 0:
            break
        try:
            decoded, cursor = decoder.raw_decode(raw_output, start)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, Mapping):
            return None
        rows.append(decoded)
    if not rows or len(rows) > max_rows:
        return None
    return _project_github_list_terminal_result(
        json.dumps({"exit_code": wrapper.get("exit_code"), "output": json.dumps(rows)}),
        max_rows=max_rows,
    )


def _git_workspace_diagnostic_call_ids(value: Any) -> frozenset[str]:
    """Recognize one read-only workspace summary whose output is nonessential.

    The command's final component prints arbitrary commit subjects, so even a
    benign worktree check must not turn those strings into remote context.
    Keep this exact rather than treating general ``git`` output as safe.
    """

    recognized: set[str] = set()

    def is_workspace_diagnostic(tokens: list[str]) -> bool:
        # Git supports more than one no-argument ``--show-*`` rev-parse
        # selector.  Its value is omitted either way; the rest of the command
        # remains an exact read-only status/branch/log sequence.
        if tokens[0:2] != ["git", "rev-parse"]:
            return False
        try:
            separator = tokens.index("&&", 2)
        except ValueError:
            return False
        return (
            separator > 2
            and all(
                token.startswith("--") or token == "HEAD"
                for token in tokens[2:separator]
            )
            and tokens[separator:] == [
                "&&", "git", "branch", "--show-current", "&&",
                "git", "log", "--oneline", "-5",
            ]
        )

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            if item.get("type") in {"function", "function_call"} and direct_name == "terminal":
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                    command = parsed.get("command") if isinstance(parsed, Mapping) else None
                    tokens = shlex.split(command) if isinstance(command, str) else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    tokens = []
                call_id = item.get("call_id") or item.get("id")
                if is_workspace_diagnostic(tokens) and isinstance(call_id, str):
                    recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _git_grep_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize a line-numbered, read-only ``git grep`` result for projection."""

    recognized: set[str] = set()

    def is_line_numbered_git_grep(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if tokens[:2] != ["git", "grep"] or "--" not in tokens:
            return False
        separator = tokens.index("--")
        return (
            separator > 2
            and separator + 1 < len(tokens)
            and any(token in {"-n", "--line-number"} for token in tokens[2:separator])
        )

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name == "terminal"
                and is_line_numbered_git_grep(arguments)
                and isinstance(call_id, str)
            ):
                recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _rg_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize a line-numbered ``rg`` result for location-only projection."""

    recognized: set[str] = set()

    def is_line_numbered_rg(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if len(tokens) < 4 or tokens[0] != "rg":
            return False
        if not any(token in {"-n", "--line-number"} for token in tokens[1:]):
            return False
        return not any(
            token in {"--json", "--files", "--files-with-matches"}
            for token in tokens[1:]
        )

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name == "terminal"
                and is_line_numbered_rg(arguments)
                and isinstance(call_id, str)
            ):
                recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _kanban_assignees_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize the exact JSON roster command used by protected workers."""

    recognized: set[str] = set()

    def is_assignee_roster(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if len(tokens) >= 2 and tokens[0] == "env" and tokens[1].startswith(
            "HERMES_HOME="
        ):
            tokens = tokens[2:]
        if tokens[:1] == ["hermes"]:
            tokens = tokens[1:]
        elif (
            len(tokens) >= 3
            and Path(tokens[0]).name.startswith("python")
            and tokens[1:3] == ["-m", "hermes_cli.main"]
        ):
            tokens = tokens[3:]
        else:
            return False
        if tokens[:1] != ["kanban"]:
            return False
        args = tokens[1:]
        if args == ["assignees", "--json"]:
            return True
        if len(args) == 4 and args[:1] == ["--board"]:
            board = args[1]
            return (
                bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", board))
                and args[2:] == ["assignees", "--json"]
            )
        if len(args) == 3 and args[0].startswith("--board="):
            board = args[0].split("=", 1)[1]
            return (
                bool(re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", board))
                and args[1:] == ["assignees", "--json"]
            )
        return False

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            arguments = (
                direct_function.get("arguments")
                if isinstance(direct_function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and direct_name == "terminal"
                and is_assignee_roster(arguments)
                and isinstance(call_id, str)
            ):
                recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _project_kanban_assignees_terminal_result(text: str) -> str | None:
    """Project an assignee listing to valid, on-disk profile names only."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
        rows = json.loads(raw_output) if isinstance(raw_output, str) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list):
        return None
    assignees: list[str] = []
    for row in rows[:256]:
        if not isinstance(row, Mapping) or row.get("on_disk") is not True:
            continue
        name = row.get("name")
        if (
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name)
        ):
            assignees.append(name)
    projection: dict[str, Any] = {
        "kanban_assignees_projection": "v1",
        "assignees": assignees,
    }
    omitted = len(rows) - len(assignees)
    if omitted:
        projection["omitted_entries"] = omitted
    exit_code = wrapper.get("exit_code")
    return json.dumps(
        {
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "output": json.dumps(projection, separators=(",", ":")),
        },
        separators=(",", ":"),
    )


def _project_line_numbered_search_terminal_result(text: str) -> str | None:
    """Replace matching source lines from a recognized search with locations."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_output, str):
        return None
    matches: list[dict[str, Any]] = []
    for raw_line in raw_output.splitlines():
        if not raw_line:
            continue
        if raw_line == "--":
            continue
        matched = re.match(r"^(?P<path>[^:\n]+):(?P<line>[1-9]\d*):", raw_line)
        if matched is None:
            # ``rg --context`` emits neighboring source lines as
            # ``path-line-content``. Discard them rather than treating their
            # content as remotely replayable; only actual match locations are
            # useful to the follow-up read/search tools.
            if re.match(r"^[^:\n]+-[1-9]\d*-", raw_line):
                continue
            return None
        if len(matches) >= _GIT_GREP_TERMINAL_MAX_MATCHES:
            continue
        matches.append(
            {"path": matched.group("path"), "line": int(matched.group("line"))}
        )
    projection: dict[str, Any] = {
        "git_grep_locations": "locations-v1",
        "matches": matches,
    }
    line_count = sum(1 for line in raw_output.splitlines() if line)
    if line_count > len(matches):
        projection["omitted_matches"] = line_count - len(matches)
    exit_code = wrapper.get("exit_code") if isinstance(wrapper.get("exit_code"), int) else None
    return json.dumps(
        {"exit_code": exit_code, "output": json.dumps(projection, separators=(",", ":"))},
        separators=(",", ":"),
    )


def _bounded_remote_text(value: str) -> str:
    """Keep a display field UTF-8 bounded before remote GitHub replay."""

    raw = value.encode("utf-8")
    if len(raw) > _GITHUB_LIST_TERMINAL_MAX_ITEM_BYTES:
        value = raw[:_GITHUB_LIST_TERMINAL_MAX_ITEM_BYTES].decode("utf-8", "ignore") + "…"
    return redact_remote_unsafe_text(value)


def _project_github_list_terminal_result(text: str, *, max_rows: int) -> str | None:
    """Retain bounded GitHub review evidence while dropping opaque metadata."""

    try:
        wrapper = json.loads(text)
        raw_rows = wrapper.get("output") if isinstance(wrapper, Mapping) else None
        decoded = json.loads(raw_rows) if isinstance(raw_rows, str) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    rows = (
        decoded
        if isinstance(decoded, list)
        else [decoded]
        if isinstance(decoded, Mapping)
        else None
    )
    if rows is None:
        return None

    items: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        if not isinstance(row, Mapping):
            continue
        projected: dict[str, Any] = {}
        if isinstance(row.get("number"), int):
            projected["number"] = row["number"]
        for key in ("baseRefName", "headRefName"):
            ref_name = row.get(key)
            if (
                isinstance(ref_name, str)
                and 0 < len(ref_name) <= 255
                and re.fullmatch(r"[A-Za-z0-9._/-]+", ref_name)
                and ".." not in ref_name
                and "@{" not in ref_name
                and "//" not in ref_name
                and not ref_name.startswith(("/", "."))
                and not ref_name.endswith(("/", "."))
            ):
                projected[key] = ref_name
        for key in ("baseRefOid", "headRefOid"):
            oid = row.get(key)
            if isinstance(oid, str) and re.fullmatch(r"[0-9a-fA-F]{40}", oid):
                projected[key] = oid
        repository = row.get("headRepository")
        if isinstance(repository, Mapping):
            projected_repository = {
                key: value
                for key, value in repository.items()
                if key in {"name", "nameWithOwner"}
                and isinstance(value, str)
                and 0 < len(value) <= 200
                and re.fullmatch(
                    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?",
                    value,
                )
            }
            if projected_repository:
                projected["headRepository"] = projected_repository
        for key in ("title", "url", "state", "createdAt", "updatedAt", "reviewDecision"):
            if isinstance(row.get(key), str):
                projected[key] = _bounded_remote_text(row[key])
        if isinstance(row.get("isDraft"), bool):
            projected["isDraft"] = row["isDraft"]
        labels = row.get("labels")
        if isinstance(labels, list):
            projected_labels = [
                _bounded_remote_text(label["name"])
                for label in labels
                if isinstance(label, Mapping) and isinstance(label.get("name"), str)
            ]
            if projected_labels:
                projected["labels"] = projected_labels
        for key in ("author", "assignees"):
            raw_people = row.get(key)
            people = raw_people if isinstance(raw_people, list) else [raw_people]
            logins = [
                _bounded_remote_text(person["login"])
                for person in people
                if isinstance(person, Mapping) and isinstance(person.get("login"), str)
            ]
            if logins:
                projected[key] = logins if isinstance(raw_people, list) else logins[0]
        if not projected:
            continue
        candidate = {"github_list_projection": "v1", "items": [*items, projected]}
        if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > _GITHUB_LIST_TERMINAL_MAX_OUTPUT_BYTES:
            break
        items.append(projected)

    projection: dict[str, Any] = {"github_list_projection": "v1", "items": items}
    omitted = max(0, min(len(rows), max_rows) - len(items)) + max(0, len(rows) - max_rows)
    if omitted:
        projection["omitted_items"] = omitted
    exit_code = wrapper.get("exit_code") if isinstance(wrapper.get("exit_code"), int) else None
    return json.dumps(
        {"exit_code": exit_code, "output": json.dumps(projection, ensure_ascii=False, separators=(",", ":"))},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _project_github_api_extract_result(text: str, *, max_rows: int) -> str | None:
    """Keep useful fields from a verified GitHub REST list extract only."""

    try:
        decoded = json.loads(text)
        results = decoded.get("results") if isinstance(decoded, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(results, list):
        return None

    projected_results: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, Mapping):
            return None
        url = result.get("url")
        content = result.get("content")
        if not isinstance(url, str) or not isinstance(content, str):
            return None
        projected = _project_github_list_terminal_result(
            json.dumps({"exit_code": 0, "output": content}), max_rows=max_rows
        )
        if projected is None:
            return None
        wrapped = json.loads(projected)
        projection_text = wrapped.get("output") if isinstance(wrapped, Mapping) else None
        projection = json.loads(projection_text) if isinstance(projection_text, str) else None
        items = projection.get("items") if isinstance(projection, Mapping) else None
        if not isinstance(items, list):
            return None
        candidate = {"url": url, "items": items}
        if (
            len(
                json.dumps(
                    {
                        "github_api_extract_projection": "v1",
                        "results": [*projected_results, candidate],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            > _GITHUB_LIST_TERMINAL_MAX_OUTPUT_BYTES
        ):
            break
        projected_results.append(candidate)

    return json.dumps(
        {
            "github_api_extract_projection": "v1",
            "results": projected_results,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _rejected_terminal_call_ids(value: Any) -> frozenset[str]:
    """Return terminal calls that the local policy rejected before execution."""

    terminal_ids: set[str] = set()
    rejected: set[str] = set()

    def remember_variants(target: set[str], call_id: Any) -> None:
        if isinstance(call_id, str):
            target.update(tool_result_id_variants(call_id))

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            direct_function = item.get("function")
            direct_name = (
                direct_function.get("name")
                if isinstance(direct_function, Mapping)
                else item.get("name")
            )
            call_id = item.get("call_id") or item.get("id")
            if item.get("type") in {"function", "function_call"} and direct_name == "terminal":
                remember_variants(terminal_ids, call_id)
            if (
                item.get("type") == "function_call_output"
                and isinstance(call_id, str)
                and call_id in terminal_ids
            ):
                raw_output = item.get("output")
                try:
                    result = json.loads(raw_output) if isinstance(raw_output, str) else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    result = None
                if (
                    isinstance(result, Mapping)
                    and result.get("exit_code") == -1
                    and isinstance(result.get("error"), str)
                    and result["error"].startswith(
                        "BLOCKED: Command flagged as dangerous"
                    )
                ):
                    remember_variants(rejected, call_id)
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(rejected)


def _segment_protected_tool_result(
    text: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
) -> SanitizedSegment | SourceBoundSegment | OutboundText:
    """Admit ordinary tool output without treating it as trusted source.

    Protected cloud workers need normal terminal results to make progress.
    Provenance is therefore not a standalone deny reason for a matched tool
    result: output takes the same bounded, source-aware path as other
    non-source text. This does not grant source authority or bypass the final
    secret, encoding, path, size, or receipt checks; unsafe output still fails
    closed there.
    """

    segments: list[SanitizedSegment | SourceBoundSegment | ValidatedToolSyntaxSegment] = []
    cursor = 0
    for match in _VERIFIED_DIAGNOSTIC_ATOM.finditer(text):
        if match.start() > cursor:
            prefix = _segment_text(
                text[cursor : match.start()],
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
                allow_line_split=True,
            )
            segments.extend(prefix.segments if isinstance(prefix, OutboundText) else (prefix,))
        atom = validate_tool_syntax(match.group(0), "verified_diagnostic_atom")
        segments.append(ValidatedToolSyntaxSegment(atom, "verified_diagnostic_atom"))
        cursor = match.end()
    if cursor < len(text):
        suffix = _segment_text(
            text[cursor:],
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
            allow_line_split=True,
        )
        segments.extend(suffix.segments if isinstance(suffix, OutboundText) else (suffix,))
    if not segments:
        return _segment_text(
            text,
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
            allow_line_split=True,
        )
    return segments[0] if len(segments) == 1 else OutboundText(tuple(segments))


def _untrusted_content_digest(value: Any) -> str:
    """Hash unbound tool content without retaining or rendering its bytes."""

    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        try:
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError):
            payload = repr(value).encode("utf-8", errors="replace")
    return sha256(payload).hexdigest()


def _segment_read_file_presentation(
    text: str,
    metadata: Any,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    registry: SourceProvenanceRegistry | None = None,
    session_id: str = "",
    turn_id: str = "",
    request_id: str = "",
    policy_digest: str = "",
) -> SourcePresentationSegment | UntrustedProvenanceSegment:
    """Bind the real JSON/line-number presentation to one exact read grant."""

    denied = UntrustedProvenanceSegment(sha256(text.encode("utf-8")).hexdigest())
    if not isinstance(metadata, Mapping):
        return denied
    if metadata.get("presentation_kind") != "read_file_json_v1":
        return denied
    if metadata.get("content_sha256") != sha256(text.encode("utf-8")).hexdigest():
        return denied
    digests = metadata.get("source_grant_digests")
    if not isinstance(digests, (list, tuple)) or not digests:
        return denied
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return denied
    if not isinstance(parsed, dict) or not isinstance(parsed.get("content"), str):
        return denied
    allowed_digests = {value for value in digests if isinstance(value, str)}
    candidates: list[tuple[str, SourceGrant]] = []
    for raw_text, grant in grant_texts:
        digest = source_grant_digest(grant)
        if digest not in allowed_digests or metadata.get("request_id") != grant.request_id:
            continue
        expected = "\n".join(
            f"{line_number}|{line}"
            for line_number, line in enumerate(
                raw_text.split("\n"), start=grant.line_start
            )
        )
        if parsed["content"] == expected:
            candidates.append((digest, grant))
    if not candidates and registry is not None:
        original_request_id = metadata.get("request_id")
        if isinstance(original_request_id, str):
            for original_digest in allowed_digests:
                rebound = registry.rebind_validated_presentation(
                    original_digest,
                    original_request_id=original_request_id,
                    session_id=session_id,
                    turn_id=turn_id,
                    request_id=request_id,
                    policy_digest=policy_digest,
                )
                if rebound is None:
                    continue
                raw_text = _read_grant_text(rebound)
                if raw_text is None:
                    continue
                expected = "\n".join(
                    f"{line_number}|{line}"
                    for line_number, line in enumerate(
                        raw_text.split("\n"), start=rebound.line_start
                    )
                )
                if parsed["content"] == expected:
                    candidates.append((source_grant_digest(rebound), rebound))
    if len(candidates) != 1:
        return denied
    digest, grant = candidates[0]
    used_grants[digest] = grant
    return SourcePresentationSegment(digest, text, "read_file_json_v1")


def _typed_payload(
    value: Any,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
    field_name: str | None = None,
    syntax_tool_call_ids: frozenset[str] = frozenset(),
    elided_kanban_tool_call_ids: frozenset[str] = frozenset(),
    search_projection_tool_call_ids: frozenset[str] = frozenset(),
    read_file_projection_tool_call_ids: frozenset[str] = frozenset(),
    web_replay_tool_call_ids: frozenset[str] = frozenset(),
    file_mutation_replay_tool_call_ids: frozenset[str] = frozenset(),
    scratch_read_file_tool_call_ids: frozenset[str] = frozenset(),
    git_workspace_diagnostic_call_ids: frozenset[str] = frozenset(),
    git_grep_projection_tool_call_ids: frozenset[str] = frozenset(),
    rg_projection_tool_call_ids: frozenset[str] = frozenset(),
    kanban_assignees_terminal_call_ids: frozenset[str] = frozenset(),
    github_list_terminal_call_limits: Mapping[str, int] | None = None,
    github_api_extract_call_limits: Mapping[str, int] | None = None,
    github_api_curl_terminal_call_ids: frozenset[str] = frozenset(),
    plain_github_list_terminal_call_ids: frozenset[str] = frozenset(),
    combined_github_list_terminal_call_limits: Mapping[str, int] | None = None,
    combined_github_view_terminal_call_limits: Mapping[str, int] | None = None,
    rejected_terminal_call_ids: frozenset[str] = frozenset(),
    terminal_replay_tool_call_ids: frozenset[str] = frozenset(),
    redact_terminal_arguments: bool = False,
    redact_readonly_tool_arguments: bool = False,
    protected_tool_content: bool = False,
    elide_kanban_tool_content: bool = False,
    protected_kanban_context: bool = False,
    generated_context: bool = False,
    redact_generated_context: bool = False,
    allow_codex_reasoning_replay: bool = False,
    registry: SourceProvenanceRegistry | None = None,
    request_identity: tuple[str, str, str, str] = ("", "", "", ""),
) -> Any:
    if isinstance(value, str):
        if field_name in _PROTOCOL_LITERAL_FIELDS and value in _PROTOCOL_LITERAL_VALUES:
            return LiteralSegment(value)
        if protected_tool_content:
            return _segment_protected_tool_result(
                value,
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
            )
        if elide_kanban_tool_content:
            # Return only the bounded, redacted current assignment; omit
            # comments, run history, identifiers, and raw host paths.
            return _project_bound_kanban_show(value)
        if generated_context and redact_generated_context:
            return GeneratedContextSegment(redact_remote_unsafe_text(value))
        if protected_kanban_context:
            return _segment_protected_context(
                value,
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
            )
        return _segment_text(
            value,
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
        )
    if isinstance(value, Mapping):
        source_metadata = value.get("_source_provenance")
        is_read_file_result = (
            value.get("role") == "tool"
            and (
                value.get("tool_name") == "read_file"
                or value.get("name") == "read_file"
            )
        )
        output_call_id = value.get("tool_call_id") or value.get("call_id")
        is_recognized_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in syntax_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_elided_kanban_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in elided_kanban_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_search_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in search_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_read_file_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in read_file_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_web_replay_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in web_replay_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_file_mutation_replay_result = (
            isinstance(output_call_id, str)
            and output_call_id in file_mutation_replay_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_scratch_read_file_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in scratch_read_file_tool_call_ids
            and (value.get("role") == "tool" or value.get("type") == "function_call_output")
        )
        is_git_workspace_diagnostic_result = (
            isinstance(output_call_id, str)
            and output_call_id in git_workspace_diagnostic_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_git_grep_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in git_grep_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_rg_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in rg_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_kanban_assignees_result = (
            isinstance(output_call_id, str)
            and output_call_id in kanban_assignees_terminal_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        github_list_limit = (
            github_list_terminal_call_limits.get(output_call_id)
            if isinstance(github_list_terminal_call_limits, Mapping)
            and isinstance(output_call_id, str)
            else None
        )
        github_api_extract_limit = (
            github_api_extract_call_limits.get(output_call_id)
            if isinstance(github_api_extract_call_limits, Mapping)
            and isinstance(output_call_id, str)
            else None
        )
        direct_function = value.get("function")
        direct_name = (
            direct_function.get("name")
            if isinstance(direct_function, Mapping)
            else value.get("name")
        )
        is_github_api_curl_terminal_call = (
            isinstance(output_call_id, str)
            and output_call_id in github_api_curl_terminal_call_ids
            and value.get("type") in {"function", "function_call"}
            and direct_name == "terminal"
        )
        is_plain_github_list_terminal_result = (
            isinstance(output_call_id, str)
            and output_call_id in plain_github_list_terminal_call_ids
            and (value.get("role") == "tool" or value.get("type") == "function_call_output")
        )
        combined_github_list_limit = (
            combined_github_list_terminal_call_limits.get(output_call_id)
            if isinstance(combined_github_list_terminal_call_limits, Mapping)
            and isinstance(output_call_id, str)
            else None
        )
        combined_github_view_limit = (
            combined_github_view_terminal_call_limits.get(output_call_id)
            if isinstance(combined_github_view_terminal_call_limits, Mapping)
            and isinstance(output_call_id, str)
            else None
        )
        is_rejected_terminal_call = (
            value.get("type") in {"function", "function_call"}
            and direct_name == "terminal"
            and isinstance(output_call_id, str)
            and output_call_id in rejected_terminal_call_ids
        )
        is_terminal_replay_result = (
            isinstance(output_call_id, str)
            and output_call_id in terminal_replay_tool_call_ids
            and (value.get("role") == "tool" or value.get("type") == "function_call_output")
        )
        is_terminal_replay_call = (
            value.get("type") in {"function", "function_call"}
            and direct_name == "terminal"
            and isinstance(output_call_id, str)
            and output_call_id in terminal_replay_tool_call_ids
        )
        is_file_mutation_replay_call = (
            value.get("type") in {"function", "function_call"}
            and isinstance(direct_name, str)
            and direct_name in _REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES
            and isinstance(output_call_id, str)
            and output_call_id in file_mutation_replay_tool_call_ids
        )
        typed: dict[Any, Any] = {}
        context_mapping = value.get("role") in {"system", "developer"}
        is_tool_protocol_mapping = (
            value.get("role") in {"assistant", "tool"}
            or value.get("type") in {"function", "function_call", "function_call_output"}
        )
        is_untrusted_tool_result = (
            protected_kanban_context
            and is_tool_protocol_mapping
            and isinstance(output_call_id, str)
            and source_metadata is None
            and not is_read_file_result
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
            and not any(
                (
                    is_recognized_tool_result,
                    is_elided_kanban_tool_result,
                    is_search_projection_tool_result,
                    is_read_file_projection_tool_result,
                    is_web_replay_tool_result,
                    is_file_mutation_replay_result,
                    is_scratch_read_file_tool_result,
                    is_git_workspace_diagnostic_result,
                    is_git_grep_projection_tool_result,
                    is_rg_projection_tool_result,
                    is_kanban_assignees_result,
                    is_plain_github_list_terminal_result,
                    is_terminal_replay_result,
                )
            )
        )
        is_codex_reasoning_replay = (
            allow_codex_reasoning_replay
            and value.get("type") == "reasoning"
            and isinstance(value.get("encrypted_content"), str)
            and isinstance(value.get("summary", []), list)
        )
        for key, item in value.items():
            if key == "_source_provenance":
                continue
            if (
                key in _TOOL_PROTOCOL_IDENTIFIER_FIELDS
                and is_tool_protocol_mapping
                and isinstance(item, str)
            ):
                # Provider-issued call IDs are transport linkage, not model
                # content.  Keep them exact so opaque IDs cannot be mistaken
                # for a base64 payload and sever a function result from its call.
                typed[key] = ValidatedToolSyntaxSegment(
                    item, "tool_protocol_identifier"
                )
                continue
            is_structured_result = (
                key in {"content", "output"}
                and isinstance(item, (list, Mapping))
            )
            if is_untrusted_tool_result and key in {"content", "output"}:
                violation_reasons = {
                    reason
                    for _, reasons in content_free_violation_locations(item)
                    for reason in reasons
                }
                if not violation_reasons:
                    typed[key] = UntrustedProvenanceSegment(
                        _untrusted_content_digest(item)
                    )
                    continue
            structured_text = (
                _structured_tool_output_text(item) if is_structured_result else None
            )
            if is_kanban_assignees_result and structured_text is not None:
                projected = _project_kanban_assignees_terminal_result(structured_text)
                if projected is not None:
                    typed[key] = GeneratedContextSegment(projected)
                    continue
            if isinstance(github_list_limit, int) and structured_text is not None:
                projected = _project_github_list_terminal_result(
                    structured_text, max_rows=github_list_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(projected)
                    continue
            if (
                isinstance(combined_github_list_limit, int)
                and structured_text is not None
            ):
                projected = _project_combined_github_list_terminal_result(
                    structured_text, max_rows=combined_github_list_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if (
                isinstance(combined_github_view_limit, int)
                and structured_text is not None
            ):
                projected = _project_combined_github_view_terminal_result(
                    structured_text, max_rows=combined_github_view_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if is_structured_result and (
                is_read_file_result
                or is_read_file_projection_tool_result
                or is_scratch_read_file_tool_result
            ):
                if (
                    source_metadata is not None
                    and not is_scratch_read_file_tool_result
                    and structured_text is not None
                ):
                    typed[key] = _segment_read_file_presentation(
                        structured_text,
                        source_metadata,
                        grant_texts,
                        used_grants,
                        registry=registry,
                        session_id=request_identity[0],
                        turn_id=request_identity[1],
                        request_id=request_identity[2],
                        policy_digest=request_identity[3],
                    )
                    continue
                typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
                continue
            if is_structured_result and (
                is_search_projection_tool_result
                or is_git_grep_projection_tool_result
                or is_rg_projection_tool_result
            ):
                typed[key] = GeneratedContextSegment(
                    _STRUCTURED_SEARCH_REPLAY_ELISION
                )
                continue
            if (
                is_structured_result
                and is_web_replay_tool_result
                and github_api_extract_limit is None
            ):
                if structured_text is not None:
                    typed[key] = _project_web_search_replay(structured_text)
                    continue
            if is_structured_result and is_file_mutation_replay_result:
                typed[key] = GeneratedContextSegment(_FILE_MUTATION_REPLAY_ELISION)
                continue
            if is_structured_result and is_git_workspace_diagnostic_result:
                typed[key] = GeneratedContextSegment(
                    _GIT_WORKSPACE_DIAGNOSTIC_REPLAY
                )
                continue
            if is_structured_result and is_plain_github_list_terminal_result:
                typed[key] = GeneratedContextSegment(
                    _GITHUB_PLAIN_LIST_OUTPUT_REPLAY
                )
                continue
            if is_structured_result and is_terminal_replay_result:
                # The Responses API represents function-call output as an
                # array of input_text/input_image items.  A recognized local
                # terminal call gets the same outcome-only replay boundary as
                # its scalar counterpart; recursively typing the array would
                # expose raw stdout to the remote firewall.
                typed[key] = GeneratedContextSegment(_terminal_replay_result(""))
                continue
            if (
                is_read_file_projection_tool_result
                and source_metadata is None
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                # An exact call-id proves this is the local read tool's result,
                # but an error/denial has no source grant.  Replay only the
                # bounded outcome instead of treating the error text as source.
                typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
                continue
            if (
                (
                    is_read_file_result
                    or (
                        is_read_file_projection_tool_result
                        and source_metadata is not None
                    )
                )
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                if is_scratch_read_file_tool_result:
                    typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
                    continue
                typed[key] = _segment_read_file_presentation(
                    item,
                    source_metadata,
                    grant_texts,
                    used_grants,
                    registry=registry,
                    session_id=request_identity[0],
                    turn_id=request_identity[1],
                    request_id=request_identity[2],
                    policy_digest=request_identity[3],
                )
                continue
            if (
                is_search_projection_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = _project_bound_search_files(item)
                continue
            if (
                is_web_replay_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
                and github_api_extract_limit is None
            ):
                # Search results originated outside the managed workspace and
                # are useful only as untrusted public evidence. Preserve that
                # evidence after the same path/secret/encoding redaction used
                # for remote-safe generated context, while keeping it charged
                # to the sanitized-text budget. The exact call-id binding is
                # required so arbitrary tool output cannot claim this lane.
                typed[key] = _project_web_search_replay(item)
                continue
            if (
                is_file_mutation_replay_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(_FILE_MUTATION_REPLAY_ELISION)
                continue
            if (
                is_kanban_assignees_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_kanban_assignees_terminal_result(item)
                if projected is not None:
                    typed[key] = GeneratedContextSegment(projected)
                    continue
            if (
                is_git_workspace_diagnostic_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(_GIT_WORKSPACE_DIAGNOSTIC_REPLAY)
                continue
            if (
                is_git_grep_projection_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_line_numbered_search_terminal_result(item)
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if (
                is_rg_projection_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_line_numbered_search_terminal_result(item)
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if (
                isinstance(github_list_limit, int)
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_github_list_terminal_result(
                    item, max_rows=github_list_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if (
                isinstance(github_api_extract_limit, int)
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_github_api_extract_result(
                    item, max_rows=github_api_extract_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
            if (
                isinstance(github_api_extract_limit, int)
                and key == "arguments"
                and isinstance(item, str)
            ):
                # The bounded REST request already ran locally.  Its exact
                # URL is not necessary for the remote reasoning turn, and
                # repository path atoms can resemble an encoded payload.
                typed[key] = GeneratedContextSegment(
                    _GITHUB_API_EXTRACT_ARGUMENT_REPLAY
                )
                continue
            if is_github_api_curl_terminal_call and key == "arguments":
                typed[key] = GeneratedContextSegment(
                    _GITHUB_API_CURL_ARGUMENT_REPLAY
                )
                continue
            if (
                is_plain_github_list_terminal_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(_GITHUB_PLAIN_LIST_OUTPUT_REPLAY)
                continue
            if (
                isinstance(combined_github_list_limit, int)
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_combined_github_list_terminal_result(
                    item, max_rows=combined_github_list_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(redact_remote_unsafe_text(projected))
                    continue
            if (
                isinstance(combined_github_view_limit, int)
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_combined_github_view_terminal_result(
                    item, max_rows=combined_github_view_limit
                )
                if projected is not None:
                    typed[key] = GeneratedContextSegment(redact_remote_unsafe_text(projected))
                    continue
            if is_rejected_terminal_call and key == "arguments":
                typed[key] = GeneratedContextSegment(_REJECTED_TERMINAL_COMMAND_REPLAY)
                continue
            if is_terminal_replay_result and key in {"content", "output"} and isinstance(item, str):
                # Terminal stdout is produced locally and can contain source,
                # credentials, or opaque values.  It must not cause a remote
                # worker to fail closed after the local command already ran.
                # Specialized GitHub/search projections above retain the few
                # bounded facts a worker needs; all other stdout is outcome-only.
                typed[key] = GeneratedContextSegment(_terminal_replay_result(item))
                continue
            if is_terminal_replay_call and key == "arguments" and isinstance(item, str):
                typed[key] = GeneratedContextSegment(_terminal_replay_command(item))
                continue
            if (
                is_file_mutation_replay_call
                and key == "arguments"
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(_FILE_MUTATION_ARGUMENT_REPLAY)
                continue
            if (
                redact_terminal_arguments
                and direct_name == "terminal"
                and key == "arguments"
                and isinstance(item, str)
            ):
                # Chat-completions nests tool arguments under ``function``;
                # by that recursive pass the outer call ID is unavailable.
                # Every protected worker terminal command is local-only, so
                # retain its coarse command class without replaying raw text.
                typed[key] = GeneratedContextSegment(_terminal_replay_command(item))
                continue
            if (
                redact_readonly_tool_arguments
                and key == "arguments"
                and isinstance(direct_name, str)
                and direct_name in _REMOTE_KANBAN_READONLY_REPLAY_TOOL_NAMES
                and isinstance(item, str)
            ):
                # The local call has already run.  Its read-only arguments are
                # replayed only as remote context, where an ordinary search
                # term (for example "DISABLE") can look like base64.  Redact
                # opaque or secret-shaped text here without changing the
                # executed call or relaxing validation for write-capable tools.
                typed[key] = GeneratedContextSegment(redact_remote_unsafe_text(item))
                continue
            typed_key = (
                GeneratedContextKey(key)
                if generated_context and redact_generated_context
                else key
            )
            if is_codex_reasoning_replay and key == "encrypted_content":
                typed[typed_key] = CodexReasoningReplaySegment(item)
                continue
            typed[typed_key] = _typed_payload(
                item,
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
                field_name=key,
                syntax_tool_call_ids=syntax_tool_call_ids,
                elided_kanban_tool_call_ids=elided_kanban_tool_call_ids,
                search_projection_tool_call_ids=search_projection_tool_call_ids,
                read_file_projection_tool_call_ids=read_file_projection_tool_call_ids,
                web_replay_tool_call_ids=web_replay_tool_call_ids,
                file_mutation_replay_tool_call_ids=file_mutation_replay_tool_call_ids,
                scratch_read_file_tool_call_ids=scratch_read_file_tool_call_ids,
                git_workspace_diagnostic_call_ids=git_workspace_diagnostic_call_ids,
                git_grep_projection_tool_call_ids=git_grep_projection_tool_call_ids,
                rg_projection_tool_call_ids=rg_projection_tool_call_ids,
                kanban_assignees_terminal_call_ids=kanban_assignees_terminal_call_ids,
                github_list_terminal_call_limits=github_list_terminal_call_limits,
                github_api_extract_call_limits=github_api_extract_call_limits,
                github_api_curl_terminal_call_ids=github_api_curl_terminal_call_ids,
                plain_github_list_terminal_call_ids=plain_github_list_terminal_call_ids,
                combined_github_list_terminal_call_limits=combined_github_list_terminal_call_limits,
                combined_github_view_terminal_call_limits=combined_github_view_terminal_call_limits,
                rejected_terminal_call_ids=rejected_terminal_call_ids,
                terminal_replay_tool_call_ids=terminal_replay_tool_call_ids,
                redact_terminal_arguments=redact_terminal_arguments,
                redact_readonly_tool_arguments=redact_readonly_tool_arguments,
                protected_tool_content=(
                    is_recognized_tool_result and key in {"content", "output"}
                ),
                elide_kanban_tool_content=(
                    is_elided_kanban_tool_result and key in {"content", "output"}
                ),
                protected_kanban_context=protected_kanban_context,
                generated_context=(
                    redact_generated_context
                    and (
                        generated_context
                        or context_mapping
                        or key in {"instructions", "system_prompt", "tools"}
                    )
                ),
                redact_generated_context=redact_generated_context,
                allow_codex_reasoning_replay=allow_codex_reasoning_replay,
                registry=registry,
                request_identity=request_identity,
            )
        return typed
    if isinstance(value, (list, tuple)):
        return [
            _typed_payload(
                item,
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
                field_name=field_name,
                syntax_tool_call_ids=syntax_tool_call_ids,
                elided_kanban_tool_call_ids=elided_kanban_tool_call_ids,
                search_projection_tool_call_ids=search_projection_tool_call_ids,
                read_file_projection_tool_call_ids=read_file_projection_tool_call_ids,
                web_replay_tool_call_ids=web_replay_tool_call_ids,
                file_mutation_replay_tool_call_ids=file_mutation_replay_tool_call_ids,
                scratch_read_file_tool_call_ids=scratch_read_file_tool_call_ids,
                git_workspace_diagnostic_call_ids=git_workspace_diagnostic_call_ids,
                git_grep_projection_tool_call_ids=git_grep_projection_tool_call_ids,
                rg_projection_tool_call_ids=rg_projection_tool_call_ids,
                kanban_assignees_terminal_call_ids=kanban_assignees_terminal_call_ids,
                github_list_terminal_call_limits=github_list_terminal_call_limits,
                github_api_extract_call_limits=github_api_extract_call_limits,
                github_api_curl_terminal_call_ids=github_api_curl_terminal_call_ids,
                plain_github_list_terminal_call_ids=plain_github_list_terminal_call_ids,
                combined_github_list_terminal_call_limits=combined_github_list_terminal_call_limits,
                combined_github_view_terminal_call_limits=combined_github_view_terminal_call_limits,
                rejected_terminal_call_ids=rejected_terminal_call_ids,
                terminal_replay_tool_call_ids=terminal_replay_tool_call_ids,
                redact_terminal_arguments=redact_terminal_arguments,
                redact_readonly_tool_arguments=redact_readonly_tool_arguments,
                protected_tool_content=protected_tool_content,
                elide_kanban_tool_content=elide_kanban_tool_content,
                protected_kanban_context=protected_kanban_context,
                generated_context=generated_context,
                redact_generated_context=redact_generated_context,
                allow_codex_reasoning_replay=allow_codex_reasoning_replay,
                registry=registry,
                request_identity=request_identity,
            )
            for item in value
        ]
    return value


def _structured_tool_output_text(value: Any) -> str | None:
    """Return the sole text item from a Responses function output array.

    Specialized projectors may inspect this exact transport shape.  Mixed,
    image-bearing, extended, or multi-item outputs stay on the conservative
    whole-result elision path.
    """

    if not isinstance(value, list) or len(value) != 1:
        return None
    item = value[0]
    if not isinstance(item, Mapping) or set(item) != {"type", "text"}:
        return None
    text = item.get("text")
    if item.get("type") != "input_text" or not isinstance(text, str):
        return None
    return text


def _terminal_replay_command(arguments: str) -> str:
    """Replay a local command only after strict sensitive-text redaction."""

    return redact_remote_unsafe_text(redact_sensitive_text(arguments, force=True))


def _terminal_replay_result(output: str) -> str:
    """Preserve a local terminal exit status without replaying raw output."""

    try:
        parsed = json.loads(output)
        exit_code = parsed.get("exit_code") if isinstance(parsed, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        exit_code = None
    return json.dumps(
        {
            "terminal_result": "completed",
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "raw_output": "omitted_from_remote_replay",
        },
        separators=(",", ":"),
    )


def _structural_literal_hashes(value: Any) -> frozenset[str]:
    literals: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if isinstance(key, str):
                    literals.add(key)
                    if (
                        key in _PROTOCOL_LITERAL_FIELDS
                        and isinstance(child, str)
                        and child in _PROTOCOL_LITERAL_VALUES
                    ):
                        literals.add(child)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (bool, int)):
            literals.add(json.dumps(item, ensure_ascii=True, separators=(",", ":")))
        elif isinstance(item, float) and math.isfinite(item):
            literals.add(
                json.dumps(
                    item, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                )
            )

    visit(value)
    return frozenset(static_literal_sha256(literal) for literal in literals)


def _typed_payload_violation_locations(
    value: Any,
) -> tuple[tuple[str, str, int, tuple[str, ...]], ...]:
    """Summarize unsafe typed segments without recording their text.

    This is diagnostic-only evidence for a failed final authorization.  It
    deliberately retains neither raw values nor hashes that could be used to
    correlate secret material across requests.
    """

    locations: list[tuple[str, str, int, tuple[str, ...]]] = []
    text_segments = (
        SanitizedSegment,
        GeneratedContextSegment,
        LiteralSegment,
        ValidatedToolSyntaxSegment,
        CodexReasoningReplaySegment,
        SourcePresentationSegment,
        SourceBoundSegment,
        UntrustedProvenanceSegment,
    )

    def visit(item: Any, path: str) -> None:
        if isinstance(item, OutboundText):
            for index, segment in enumerate(item.segments):
                visit(segment, f"{path}.segments[{index}]")
            return
        if isinstance(item, text_segments):
            text = getattr(item, "text", None)
            if not isinstance(text, str):
                return
            reasons = tuple(
                sorted({reason for _, found in content_free_violation_locations(text) for reason in found})
            )
            if reasons:
                locations.append((path, type(item).__name__, len(text.encode("utf-8")), reasons))
            return
        if isinstance(item, Mapping):
            for index, (_, child) in enumerate(item.items()):
                visit(child, f"{path}.map[{index}].value")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}.sequence[{index}]")

    visit(value, "$")
    return tuple(locations)


def _route_for_agent(agent: Any, route: Any | None) -> Any:
    if route is not None:
        return route
    provider = str(getattr(agent, "provider", "") or "")
    base_url = getattr(agent, "base_url", None)
    api_mode = getattr(agent, "api_mode", None)
    if provider == "openai-codex" and not base_url:
        base_url = "https://chatgpt.com/backend-api/codex"
        api_mode = api_mode or "codex_responses"
    return SimpleNamespace(
        provider=provider,
        model=str(getattr(agent, "model", "") or ""),
        base_url=base_url,
        api_mode=api_mode,
    )


def _route_field(route: Any, name: str, default: Any = None) -> Any:
    """Read route fields from both provider objects and serialized mappings."""

    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _restore_source_provenance_sidecar(
    body: Mapping[str, Any], sidecar: Any
) -> dict[str, Any]:
    """Reattach exact content-bound metadata after either wire conversion.

    Chat Completions retains tool messages, while Codex Responses converts
    them to ``function_call_output`` items.  The latter must recover the same
    internal envelope before typing; otherwise a verified read is mistaken
    for untrusted structured output and silently elided.
    """

    restored = dict(body)
    messages = restored.get("messages")
    if not isinstance(sidecar, list):
        return restored
    if isinstance(messages, list):
        copied_messages = list(messages)
        changed = False
        for entry in sidecar:
            if not isinstance(entry, Mapping):
                continue
            index = entry.get("message_index")
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if index < 0 or index >= len(copied_messages):
                continue
            message = copied_messages[index]
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if (
                message.get("role") != "tool"
                or not isinstance(content, str)
                or message.get("tool_call_id") != entry.get("tool_call_id")
                or entry.get("content_sha256")
                != sha256(content.encode("utf-8")).hexdigest()
            ):
                continue
            copied = dict(message)
            copied["_source_provenance"] = {
                key: entry[key]
                for key in (
                    "request_id",
                    "source_grant_digests",
                    "content_sha256",
                    "presentation_kind",
                )
                if key in entry
            }
            copied_messages[index] = copied
            changed = True
        if changed:
            restored["messages"] = copied_messages

    def _restore_input_items(input_items: Any) -> tuple[Any, bool]:
        if not isinstance(input_items, list):
            return input_items, False
        copied_input = list(input_items)
        changed = False
        for entry in sidecar:
            if not isinstance(entry, Mapping):
                continue
            expected_sha = entry.get("content_sha256")
            original_call_id = entry.get("tool_call_id")
            if not isinstance(expected_sha, str) or not isinstance(original_call_id, str):
                continue
            try:
                from agent.codex_responses_adapter import _clamp_responses_call_id

                expected_call_id = _clamp_responses_call_id(original_call_id)
            except Exception:
                expected_call_id = original_call_id
            candidates: list[int] = []
            for index, item in enumerate(copied_input):
                if not isinstance(item, Mapping):
                    continue
                output = item.get("output")
                output_text = (
                    _structured_tool_output_text(output)
                    if isinstance(output, (list, Mapping))
                    else output
                )
                if (
                    item.get("type") == "function_call_output"
                    and item.get("call_id") == expected_call_id
                    and isinstance(output_text, str)
                    and sha256(output_text.encode("utf-8")).hexdigest() == expected_sha
                ):
                    candidates.append(index)
            if len(candidates) != 1:
                continue
            index = candidates[0]
            copied = dict(copied_input[index])
            copied["_source_provenance"] = {
                key: entry[key]
                for key in (
                    "request_id",
                    "source_grant_digests",
                    "content_sha256",
                    "presentation_kind",
                )
                if key in entry
            }
            copied_input[index] = copied
            changed = True
        return copied_input if changed else input_items, changed

    restored_input, input_changed = _restore_input_items(restored.get("input"))
    if input_changed:
        restored["input"] = restored_input

    # The consumer-Codex SDK transform bypass moves the already-normalized
    # bulk ``input`` under ``extra_body`` immediately before dispatch.  That
    # remains provider wire data, so bind the same exact call-id/content proof
    # there as well; no other nested shape is accepted.
    extra_body = restored.get("extra_body")
    if isinstance(extra_body, Mapping):
        restored_extra_input, extra_input_changed = _restore_input_items(
            extra_body.get("input")
        )
        if extra_input_changed:
            copied_extra_body = dict(extra_body)
            copied_extra_body["input"] = restored_extra_input
            restored["extra_body"] = copied_extra_body
    return restored


def authorize_agent_sdk_kwargs(
    agent: Any,
    kwargs: Mapping[str, Any],
    *,
    route: Any | None = None,
    sdk_control_keys: Sequence[str] = _SDK_CONTROL_KEYS,
) -> tuple[dict[str, Any], AuthorizedEgress]:
    controls = {key: kwargs[key] for key in sdk_control_keys if key in kwargs}
    resolved_route = _route_for_agent(agent, route)
    route_provider = _route_field(resolved_route, "provider", "")
    protected_provider_route = provider_uses_egress_firewall(route_provider)
    protected_remote_marker = (
        os.environ.get("HERMES_KANBAN_PROTECTED_REMOTE") == "1"
    )
    # The marker is deliberately process-local, but a fallback/reconstructed
    # worker still carries its task identity. Re-derive the protected Kanban
    # boundary from that durable identity plus the exact provider route so a
    # fallback cannot turn private task context into a repeated egress block.
    protected_kanban_remote = protected_remote_marker or (
        bool(str(os.environ.get("HERMES_KANBAN_TASK") or "").strip())
        and protected_provider_route
    )
    sidecar = kwargs.get("_hermes_source_provenance")
    body = {
        key: value
        for key, value in kwargs.items()
        if key not in controls and key not in _INTERNAL_EGRESS_KEYS
    }
    if protected_kanban_remote:
        body = _sanitize_protected_kanban_body(body)
    body = _restore_source_provenance_sidecar(body, sidecar)
    session_id = str(getattr(agent, "session_id", "") or "")
    turn_id = str(getattr(agent, "_current_turn_id", "") or "")
    request_id = str(getattr(agent, "_current_api_request_id", "") or "")
    policy_digest = str(
        getattr(agent, "_llm_egress_policy_digest", "")
        or getattr(agent, "llm_egress_policy_digest", "")
        or DEFAULT_POLICY_DIGEST
    )
    registry = getattr(agent, "_source_provenance_registry", None)
    grants = (
        registry.grants_for_request(request_id)
        if isinstance(registry, SourceProvenanceRegistry)
        else ()
    )
    sanitized_segment_cap = int(
        getattr(agent, "_llm_egress_max_sanitized_segment_bytes", 32_768)
    )
    sanitized_aggregate_cap = int(
        getattr(agent, "_llm_egress_max_sanitized_bytes", 32_768)
    )
    used_grants: dict[str, SourceGrant] = {}
    # Protected providers must use the bounded-context path regardless of
    # whether the worker inherited the dispatcher marker.  The marker is
    # still required for path redaction and the reduced Kanban toolset, but it
    # is not a safe prerequisite for transport framing: fallback/provider
    # resolution can rebuild the agent without preserving that process-global
    # flag.  Without this route-derived guard, a large protected request raises
    # ValueError while typing, bypassing the firewall's content-free receipt
    # and triggering a provider fallback loop.
    protected_remote_context = protected_remote_marker or protected_provider_route
    # Generated framing (system/developer messages and tool schema) is
    # application-owned.  It can use the established non-secret path/base64
    # redaction on every protected cloud route, including ordinary chat and
    # goal-judge calls.  User content and unbound tool results do not become
    # generated context and remain fail-closed.
    redact_protected_generated_context = (
        str(route_provider or "").strip().lower() == "openai-codex"
        or protected_provider_route
    )
    typed_body = _typed_payload(
        body,
        _grant_texts(grants),
        used_grants,
        sanitized_cap=sanitized_segment_cap,
        syntax_tool_call_ids=(
            _recognized_syntax_tool_call_ids(body)
            if protected_kanban_remote
            else frozenset()
        ),
        elided_kanban_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_PROJECTION_TOOL_NAMES)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        search_projection_tool_call_ids=(
            _recognized_tool_call_ids(
                body, _REMOTE_KANBAN_SEARCH_PROJECTION_TOOL_NAMES
            )
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        read_file_projection_tool_call_ids=(
            _recognized_tool_call_ids(
                body, _REMOTE_KANBAN_READ_FILE_PROJECTION_TOOL_NAMES
            )
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        web_replay_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_WEB_REPLAY_TOOL_NAMES)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        file_mutation_replay_tool_call_ids=(
            _recognized_tool_call_ids(
                body, _REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES
            )
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        scratch_read_file_tool_call_ids=(
            _scratch_read_file_tool_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        git_workspace_diagnostic_call_ids=(
            _git_workspace_diagnostic_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        git_grep_projection_tool_call_ids=(
            _git_grep_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        rg_projection_tool_call_ids=(
            _rg_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        kanban_assignees_terminal_call_ids=(
            _kanban_assignees_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        github_list_terminal_call_limits=(
            _github_list_terminal_call_limits(body)
            if protected_kanban_remote and protected_provider_route
            else None
        ),
        github_api_extract_call_limits=(
            _github_api_extract_call_limits(body)
            if protected_kanban_remote and protected_provider_route
            else None
        ),
        github_api_curl_terminal_call_ids=(
            _github_api_curl_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        plain_github_list_terminal_call_ids=(
            _plain_github_list_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        combined_github_list_terminal_call_limits=(
            _combined_github_list_terminal_call_limits(body)
            if protected_kanban_remote and protected_provider_route
            else None
        ),
        combined_github_view_terminal_call_limits=(
            _combined_github_view_terminal_call_limits(body)
            if protected_kanban_remote and protected_provider_route
            else None
        ),
        rejected_terminal_call_ids=(
            _rejected_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        terminal_replay_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        redact_readonly_tool_arguments=(
            protected_kanban_remote and protected_provider_route
        ),
        redact_terminal_arguments=(
            protected_kanban_remote and protected_provider_route
        ),
        protected_kanban_context=protected_remote_context,
        redact_generated_context=redact_protected_generated_context,
        allow_codex_reasoning_replay=(
            str(route_provider or "").strip().lower() == "openai-codex"
            and str(getattr(agent, "api_mode", "") or "") == "codex_responses"
        ),
        registry=registry if isinstance(registry, SourceProvenanceRegistry) else None,
        request_identity=(session_id, turn_id, request_id, policy_digest),
    )
    request = TypedOutboundRequest(
        payload=typed_body,
        session_id=session_id,
        turn_id=turn_id,
        request_id=request_id,
        policy_digest=policy_digest,
    )
    state_dir = Path(
        getattr(agent, "_llm_egress_state_dir", "")
        or Path.home() / ".hermes" / "egress"
    )
    max_serialized_bytes = int(
        getattr(agent, "_llm_egress_max_serialized_bytes", 262_144)
    )
    max_conservative_tokens = int(
        getattr(agent, "_llm_egress_max_conservative_tokens", 87_382)
    )
    firewall = LLMEgressFirewall(
        state_dir,
        policy_digest=policy_digest,
        max_serialized_bytes=max_serialized_bytes,
        max_conservative_tokens=max_conservative_tokens,
        max_granted_serialized_bytes=int(
            getattr(
                agent,
                "_llm_egress_max_granted_serialized_bytes",
                max_serialized_bytes,
            )
        ),
        max_granted_conservative_tokens=int(
            getattr(
                agent,
                "_llm_egress_max_granted_conservative_tokens",
                max_conservative_tokens,
            )
        ),
        max_sanitized_bytes=sanitized_aggregate_cap,
        max_sanitized_segment_bytes=sanitized_segment_cap,
        static_literal_hashes_by_policy={
            policy_digest: _structural_literal_hashes(body)
        },
        exact_secret_values=_exact_provider_secret_values(),
    )
    try:
        authorization = firewall.authorize(
            request,
            resolved_route,
            grants=tuple(used_grants.values()),
        )
    except EgressBlocked:
        typed_locations = _typed_payload_violation_locations(typed_body)
        if typed_locations:
            logger.warning(
                "LLM egress blocked typed locations: %s", typed_locations
            )
        locations = content_free_violation_locations(body)
        if locations:
            logger.warning("LLM egress blocked structural locations: %s", locations)
        raise
    if isinstance(registry, SourceProvenanceRegistry):
        registry.remember_validated_presentations(tuple(used_grants.values()))
    rebuilt = json.loads(authorization.payload_bytes)
    if not isinstance(rebuilt, dict):
        raise TypeError("authorized provider payload must be a JSON object")
    rebuilt.update(controls)
    return rebuilt, authorization


def dispatch_authorized_agent_request(
    agent: Any,
    kwargs: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], Any],
    *,
    route: Any | None = None,
    sdk_control_keys: Sequence[str] = _SDK_CONTROL_KEYS,
) -> Any:
    resolved_route = _route_for_agent(agent, route)
    destination = classify_destination(
        str(_route_field(resolved_route, "provider", "") or ""),
        _route_field(resolved_route, "base_url"),
        _route_field(resolved_route, "api_mode"),
    )
    if destination in {DestinationClass.LOCAL_PROCESS, DestinationClass.LOOPBACK}:
        return callback(dict(kwargs))
    authorized, receipt = authorize_agent_sdk_kwargs(
        agent,
        kwargs,
        route=resolved_route,
        sdk_control_keys=sdk_control_keys,
    )
    # Recreate the exact body digest immediately before the provider callback.
    # Only explicit non-content SDK controls are excluded; headers/query are
    # scanned and included in the firewall-authorized JSON body.
    wire_body = {
        key: value for key, value in authorized.items() if key not in sdk_control_keys
    }
    wire_bytes = json.dumps(
        wire_body,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    receipt.verify_payload(wire_bytes)
    return callback(MappingProxyType(authorized))
