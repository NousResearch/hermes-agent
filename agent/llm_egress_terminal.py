"""Terminal-tool output classifiers and projection helpers for LLM egress."""

from __future__ import annotations

import json
import re
import shlex
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from agent.redact import redact_sensitive_text

from agent.llm_egress_firewall import (
    OutboundText,
    SanitizedSegment,
    SourceBoundSegment,
    SourceGrant,
    SourcePresentationSegment,
    UntrustedProvenanceSegment,
    ValidatedToolSyntaxSegment,
    redact_remote_unsafe_text,
    source_grant_digest,
    validate_tool_syntax,
)
from agent.message_sanitization import tool_result_id_variants
from agent.source_provenance import SourceProvenanceRegistry

_VALIDATED_SYNTAX_TOOL_NAMES = frozenset({"terminal"})
_REMOTE_KANBAN_PROJECTION_TOOL_NAMES = frozenset({"kanban_show"})
_REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES = frozenset({"terminal", "browser_exec"})
_REMOTE_KANBAN_SEARCH_PROJECTION_TOOL_NAMES = frozenset({"search_files"})
_REMOTE_KANBAN_READ_FILE_PROJECTION_TOOL_NAMES = frozenset({"read_file"})
_REMOTE_KANBAN_WEB_REPLAY_TOOL_NAMES = frozenset({"web_extract", "web_search"})
_REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES = frozenset({"patch", "write_file"})
_REMOTE_KANBAN_READONLY_REPLAY_TOOL_NAMES = frozenset(
    {
        "kanban_show",
        "kanban_attachments",
        "search_files",
        "read_file",
        "web_extract",
        "web_search",
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
_VERIFIED_DIAGNOSTIC_ATOM = re.compile(
    r"(?<![A-Za-z0-9_])(?:PASS|WARN|SUMMARY|REQUIREMENTS|AVAILABILITY|"
    r"HANDLING|VERIFICATION|[0-9]{1,10}|0x[0-9a-fA-F]{1,16}|"
    r"_?[A-Za-z][A-Za-z0-9]{0,63}(?:_[A-Za-z0-9]{1,64}){1,7})"
    r"(?![A-Za-z0-9_])"
)


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



# PR76-specific constants
_GITHUB_PR_FEEDBACK_TERMINAL_SUBCOMMANDS = frozenset(
    {
        "inspect-pr",
        "complete-feedback",
        "retire-feedback",
        "submit-review",
        "status",
    }
)
_GITHUB_PR_FEEDBACK_TERMINAL_RESULT_KEYS = frozenset(
    {
        "base_branch",
        "base_sha",
        "codex_retrigger_status",
        "event",
        "expected_head_sha",
        "fallback",
        "feedback_body_excerpt",
        "feedback_id",
        "feedback_is_bot",
        "feedback_kind",
        "feedback_reviewer",
        "head_ref_name",
        "head_repository",
        "head_sha",
        "local_ci_status",
        "number",
        "observed_head_sha",
        "pr_number",
        "pr_state",
        "state",
        "task_id",
        "reason",
        "repository",
        "resolved_head_sha",
        "review_thread_resolved",
        "status",
        "error_excerpt",
        "receipt_id",
        "manifest_digest",
        "handoff_reason",
        "handoff_status",
        "repair_status",
        "retryable",
        "command_count",
    }
)
_GIT_DIFF_NAME_ONLY_MAX_FILES = 200
_GIT_REVIEW_SUMMARY_MAX_FILES = 200
_PYTEST_DIAGNOSTIC_MAX_LINES = 32
_PYTEST_DIAGNOSTIC_MAX_BYTES = 4096
_FILE_MUTATION_ERROR_MAX_BYTES = 1024


# Application identifier pattern for typed context segmentation
_APPLICATION_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:t_[0-9a-f]{8}|[0-9a-f]{40}|[0-9a-f]{64}|"
    r"[a-z][a-z0-9]{0,31}(?:[_-][a-z][a-z0-9]{0,31}){1,7}"
    r"(?::v[0-9]{1,3})?)(?![A-Za-z0-9_-])"
)


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


def _pytest_terminal_call_ids(value: Any) -> frozenset[str]:
    """Bind bounded pytest diagnostics to exact local test calls."""

    recognized: set[str] = set()

    def is_pytest_command(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if "pytest" not in tokens:
            return False
        index = tokens.index("pytest")
        return index == 0 or tokens[index - 1] in {"-m", "run"}

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
            arguments = (
                function.get("arguments")
                if isinstance(function, Mapping)
                else item.get("arguments")
            )
            call_id = item.get("call_id") or item.get("id")
            if (
                item.get("type") in {"function", "function_call"}
                and name == "terminal"
                and is_pytest_command(arguments)
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


def _github_pr_feedback_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize governed PR-feedback terminal commands with JSON status output."""

    recognized: set[str] = set()

    def is_hermes_launcher_token(token: str) -> bool:
        return Path(token).name == "hermes" or token == "<private-path>"

    def command_is_pr_feedback(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        for index, token in enumerate(tokens):
            if (
                token == "github-pr-feedback"
                and index > 0
                and (
                    tokens[max(0, index - 2):index] == ["-m", "hermes_cli.main"]
                    or is_hermes_launcher_token(tokens[index - 1])
                )
            ):
                return (
                    index + 1 < len(tokens)
                    and tokens[index + 1] in _GITHUB_PR_FEEDBACK_TERMINAL_SUBCOMMANDS
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
            if item.get("type") in {"function", "function_call"} and direct_name == "terminal":
                arguments = (
                    direct_function.get("arguments")
                    if isinstance(direct_function, Mapping)
                    else item.get("arguments")
                )
                call_id = item.get("call_id") or item.get("id")
                if command_is_pr_feedback(arguments) and isinstance(call_id, str):
                    recognized.update(tool_result_id_variants(call_id))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _github_api_paginate_terminal_call_limits(value: Any) -> dict[str, int]:
    """Bind exact paginated GitHub issue/PR list calls to bounded projection.

    ``gh api --paginate`` is the repository-owned command required by the
    White-Knight intake.  Its output contains opaque ids and other fields that
    are not useful to the remote reasoning turn, so only the two public list
    endpoints are admitted and projected through the same bounded row filter
    as ``gh issue/pr list --json``.  Other ``gh api`` commands remain
    fail-closed.
    """

    limits: dict[str, int] = {}

    def command_limit(arguments: Any) -> int | None:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if len(tokens) != 4 or tokens[:3] != ["gh", "api", "--paginate"]:
            return None
        path = tokens[3]
        if not re.fullmatch(
            r"/repos/[^/\s]+/[^/\s]+/(?:issues|pulls)\?state=open(?:&per_page=[1-9]\d{0,2})?",  # windows-footgun: ok — regex literal
            path,
        ):
            return None
        query = parse_qs(urlsplit(path).query, keep_blank_values=True)
        if query.get("state") != ["open"]:
            return None
        raw_per_page = query.get("per_page", [str(_GITHUB_LIST_TERMINAL_MAX_ROWS)])[0]
        try:
            limit = int(raw_per_page)
        except (TypeError, ValueError):
            return None
        return limit if 0 < limit <= _GITHUB_LIST_TERMINAL_MAX_ROWS else None

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            function = item.get("function")
            name = function.get("name") if isinstance(function, Mapping) else item.get("name")
            arguments = function.get("arguments") if isinstance(function, Mapping) else item.get("arguments")
            call_id = item.get("call_id") or item.get("id")
            limit = (
                command_limit(arguments)
                if item.get("type") in {"function", "function_call"}
                and name == "terminal"
                else None
            )
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


def _project_github_api_paginate_terminal_result(
    text: str, *, max_rows: int
) -> str | None:
    """Project concatenated JSON arrays returned by ``gh api --paginate``."""

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
        while cursor < len(raw_output) and raw_output[cursor].isspace():
            cursor += 1
        if cursor >= len(raw_output):
            break
        try:
            decoded, cursor = decoder.raw_decode(raw_output, cursor)
        except json.JSONDecodeError:
            return None
        if not isinstance(decoded, list):
            return None
        rows.extend(decoded)
    return _project_github_list_terminal_result(
        json.dumps(
            {"exit_code": wrapper.get("exit_code"), "output": json.dumps(rows)}
        ),
        max_rows=max_rows,
    )


def _git_diff_name_only_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize read-only ``git diff --name-only`` calls for path projection."""

    recognized: set[str] = set()

    def is_git_diff_name_only(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if len(tokens) < 3 or tokens[:2] != ["git", "diff"]:
            return False
        allowed_options = {
            "--name-only",
            "--",
            "--cached",
            "--staged",
            "--no-renames",
        }
        if "--name-only" not in tokens[2:]:
            return False
        if any(token in {"--patch", "-p", "--name-status", "--stat"} for token in tokens[2:]):
            return False
        return all(
            token in allowed_options
            or re.fullmatch(r"[0-9a-fA-F]{7,64}(?:\.\.\.?[0-9a-fA-F]{7,64})?", token)
            or re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,256}", token)
            for token in tokens[2:]
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
                and is_git_diff_name_only(arguments)
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


def _git_review_summary_terminal_call_ids(value: Any) -> frozenset[str]:
    """Recognize safe read-only Git review diagnostics for bounded projection."""

    recognized: set[str] = set()

    def is_git_review_summary(arguments: Any) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = parsed.get("command") if isinstance(parsed, Mapping) else None
            tokens = shlex.split(command) if isinstance(command, str) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if "&&" in tokens:
            segments: list[list[str]] = []
            current: list[str] = []
            for token in tokens:
                if token == "&&":
                    if not current:
                        return False
                    segments.append(current)
                    current = []
                    continue
                current.append(token)
            if not current:
                return False
            segments.append(current)
            return all(is_git_review_summary({"command": shlex.join(segment)}) for segment in segments)
        if len(tokens) < 2 or tokens[0] != "git":
            return False
        if tokens[1] == "status":
            allowed = {"--short", "-s", "--branch", "-b", "--porcelain", "--porcelain=v1"}
            return bool(tokens[2:]) and all(token in allowed for token in tokens[2:])
        if len(tokens) >= 3 and tokens[1] == "diff":
            summary_options = {
                "--check",
                "--compact-summary",
                "--name-status",
                "--stat",
                "--summary",
            }
            if not any(token in summary_options for token in tokens[2:]):
                return False
            rejected = {"--patch", "-p", "--name-only", "--raw", "--binary"}
            if any(token in rejected for token in tokens[2:]):
                return False
            return all(
                token in summary_options
                or token in {"--", "--cached", "--staged", "--no-renames", "--exit-code", "--quiet"}
                or re.fullmatch(r"[0-9a-fA-F]{7,64}(?:\.\.\.?[0-9a-fA-F]{7,64})?", token)
                or re.fullmatch(r"[A-Za-z0-9_./:@+-]{1,256}", token)
                for token in tokens[2:]
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
                and is_git_review_summary(arguments)
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


def _project_git_diff_name_only_terminal_result(text: str) -> str | None:
    """Replay only bounded relative paths from ``git diff --name-only``."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_output, str):
        return None
    files: list[str] = []
    omitted = 0
    for raw_line in raw_output.splitlines():
        candidate = raw_line.strip()
        if not candidate:
            continue
        normalized = candidate[2:] if candidate.startswith("./") else candidate
        path = PurePosixPath(normalized)
        if (
            path.is_absolute()
            or "\\" in normalized
            or any(
                part in {"", ".", ".."} or (part.startswith(".") and part != ".github")
                for part in path.parts
            )
        ):
            omitted += 1
            continue
        safe_path = redact_remote_unsafe_text(
            redact_sensitive_text(path.as_posix(), force=True)
        )
        if safe_path != path.as_posix() or len(safe_path.encode("utf-8")) > 512:
            omitted += 1
            continue
        if len(files) < _GIT_DIFF_NAME_ONLY_MAX_FILES:
            files.append(safe_path)
        else:
            omitted += 1
    projection: dict[str, Any] = {
        "git_diff_name_only": "paths-v1",
        "files": files,
    }
    if omitted:
        projection["omitted_files"] = omitted
    exit_code = wrapper.get("exit_code") if isinstance(wrapper.get("exit_code"), int) else None
    return json.dumps(
        {"exit_code": exit_code, "output": json.dumps(projection, separators=(",", ":"))},
        separators=(",", ":"),
    )


def _safe_repo_relative_path(value: str) -> str | None:
    normalized = value[2:] if value.startswith("./") else value
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or "\\" in normalized
        or any(
            part in {"", ".", ".."} or (part.startswith(".") and part != ".github")
            for part in path.parts
        )
    ):
        return None
    safe_path = redact_remote_unsafe_text(
        redact_sensitive_text(path.as_posix(), force=True)
    )
    if safe_path != path.as_posix() or len(safe_path.encode("utf-8")) > 512:
        return None
    return safe_path


def _project_git_review_summary_terminal_result(text: str) -> str | None:
    """Project safe Git review diagnostics without replaying source content."""

    try:
        wrapper = json.loads(text)
        raw_output = wrapper.get("output") if isinstance(wrapper, Mapping) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw_output, str):
        return None
    files: list[str] = []
    omitted = 0
    insertion_count = 0
    deletion_count = 0
    warning_count = 0
    error_count = 0
    status_count = 0
    has_branch_header = False
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            has_branch_header = True
            continue
        check_match = re.match(r"^(?P<path>[^:\n]+):[1-9][0-9]*:", line)
        if check_match is not None:
            safe_path = _safe_repo_relative_path(check_match.group("path").strip())
            if safe_path is not None and len(files) < _GIT_REVIEW_SUMMARY_MAX_FILES:
                files.append(safe_path)
            else:
                omitted += 1
            error_count += 1
            continue
        stat_match = re.match(r"^(?P<path>.+?)\s+\|\s+(?P<count>[0-9]+)(?:\s+(?P<bar>[+\-]+))?$", line)
        if stat_match is not None:
            safe_path = _safe_repo_relative_path(stat_match.group("path").strip())
            if safe_path is not None and len(files) < _GIT_REVIEW_SUMMARY_MAX_FILES:
                files.append(safe_path)
                bar = stat_match.group("bar") or ""
                insertion_count += bar.count("+")
                deletion_count += bar.count("-")
            else:
                omitted += 1
            continue
        changed_match = re.search(r"(?P<count>[0-9]+) files? changed", line)
        insertion_match = re.search(r"(?P<count>[0-9]+) insertions?\(\+\)", line)
        deletion_match = re.search(r"(?P<count>[0-9]+) deletions?\(-\)", line)
        if changed_match is not None:
            status_count += int(changed_match.group("count"))
            if insertion_match is not None:
                insertion_count += int(insertion_match.group("count"))
            if deletion_match is not None:
                deletion_count += int(deletion_match.group("count"))
            continue
        if re.match(r"^(?:[ MADRCU?!]{2}|[MADRCU?!]{1,2})\s+", raw_line):
            status_count += 1
            candidate = raw_line[2:].strip()
            if " -> " in candidate:
                candidate = candidate.rsplit(" -> ", 1)[1].strip()
            safe_path = _safe_repo_relative_path(candidate)
            if safe_path is not None and len(files) < _GIT_REVIEW_SUMMARY_MAX_FILES:
                files.append(safe_path)
            else:
                omitted += 1
            continue
        if re.search(r"\bwarning\b", line, re.IGNORECASE):
            warning_count += 1
            continue
        if re.search(r"\berror\b", line, re.IGNORECASE):
            error_count += 1
            continue
        omitted += 1
    projection: dict[str, Any] = {"git_review_summary": "v1"}
    if files:
        projection["files"] = files
    if omitted:
        projection["omitted_lines"] = omitted
    if status_count:
        projection["status_entries"] = status_count
    if insertion_count:
        projection["insertions"] = insertion_count
    if deletion_count:
        projection["deletions"] = deletion_count
    if warning_count:
        projection["warnings"] = warning_count
    if error_count:
        projection["errors"] = error_count
    if has_branch_header:
        projection["branch_header_present"] = True
    exit_code = wrapper.get("exit_code") if isinstance(wrapper.get("exit_code"), int) else None
    return json.dumps(
        {"exit_code": exit_code, "output": json.dumps(projection, separators=(",", ":"))},
        separators=(",", ":"),
    )


def _github_pr_feedback_terminal_result(output: str) -> str:
    """Replay bounded JSON status from governed PR-feedback commands."""

    def safe_failure_excerpt(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        safe = redact_remote_unsafe_text(
            redact_sensitive_text(value, force=True, redact_url_credentials=True)
        )
        encoded = safe.encode("utf-8")
        if len(encoded) > 1200:
            safe = encoded[:1190].decode("utf-8", errors="ignore") + "\n<truncated>"
        return safe

    exit_code = None
    text = output
    failure_excerpt = None
    try:
        parsed = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, Mapping):
        maybe_exit = parsed.get("exit_code")
        if isinstance(maybe_exit, int):
            exit_code = maybe_exit
        for key in ("stderr", "error"):
            failure_excerpt = safe_failure_excerpt(parsed.get(key))
            if failure_excerpt:
                break
        for key in ("stdout", "output", "content"):
            value = parsed.get(key)
            if isinstance(value, str):
                text = value
                break
    payload: dict[str, object] | None = None
    decoder = json.JSONDecoder()
    for line in reversed(str(text or "").splitlines() or [str(text or "")]):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            candidate, _end = decoder.raw_decode(stripped)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(candidate, Mapping):
            payload = {}
            for key, value in candidate.items():
                if key not in _GITHUB_PR_FEEDBACK_TERMINAL_RESULT_KEYS:
                    continue
                if key in {"receipt_id", "manifest_digest"} and (
                    not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                ):
                    continue
                if value is None or isinstance(value, (bool, int)):
                    payload[str(key)] = value
                    continue
                if not isinstance(value, str):
                    continue
                limit = 1800 if key == "feedback_body_excerpt" else 200
                if len(value) <= limit:
                    payload[str(key)] = value
            break
    if failure_excerpt is None and exit_code not in (None, 0) and payload is None:
        # Terminal backends merge stderr into output. Retain only recognizable
        # launch diagnostics, never arbitrary failed-command stdout/source.
        diagnostics = [line for line in str(text or "").splitlines() if re.search(
            r"(?:ModuleNotFoundError:|ImportError:|command not found|No such file or directory)",
            line,
        )]
        failure_excerpt = safe_failure_excerpt("\n".join(diagnostics[:4]))
    replay: dict[str, object] = {
        "terminal_result": "github_pr_feedback",
        "exit_code": exit_code,
        "raw_output": "omitted_from_remote_replay",
    }
    if payload is not None:
        replay["json"] = payload
    if isinstance(parsed, Mapping):
        session_id = parsed.get("session_id")
        if isinstance(session_id, str) and re.fullmatch(r"proc_[0-9a-f]{12}", session_id):
            replay["session_id"] = session_id
            pid = parsed.get("pid")
            if type(pid) is int and 0 < pid < 2**31:
                replay["pid"] = pid
    if failure_excerpt:
        replay["error_excerpt"] = failure_excerpt
    return json.dumps(replay, sort_keys=True, separators=(",", ":"))
