"""Payload classifiers and bounded projections for LLM egress."""

from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from agent.llm_egress_firewall import (
    CodexReasoningReplaySegment,
    GeneratedContextKey,
    GeneratedContextSegment,
    LiteralSegment,
    OutboundText,
    SanitizedSegment,
    SourceBoundSegment,
    SourceGrant,
    SourcePresentationSegment,
    UntrustedProvenanceSegment,
    ValidatedToolSyntaxSegment,
    content_free_violation_locations,
    redact_remote_unsafe_text,
    source_grant_digest,
    static_literal_sha256,
    validate_sanitized_text,
)
from agent.llm_egress_terminal import (
    _FILE_MUTATION_ARGUMENT_REPLAY,
    _FILE_MUTATION_REPLAY_ELISION,
    _GITHUB_API_CURL_ARGUMENT_REPLAY,
    _GITHUB_API_EXTRACT_ARGUMENT_REPLAY,
    _GITHUB_PLAIN_LIST_OUTPUT_REPLAY,
    _GIT_WORKSPACE_DIAGNOSTIC_REPLAY,
    _READ_FILE_REPLAY_ELISION,
    _REJECTED_TERMINAL_COMMAND_REPLAY,
    _REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_PROJECTION_ELISION,
    _REMOTE_KANBAN_SECRET_ASSIGNMENT,
    _REMOTE_KANBAN_READONLY_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES,
    _STRUCTURED_SEARCH_REPLAY_ELISION,
    _approved_sanitized,
    _github_api_curl_terminal_call_ids,
    _github_api_extract_call_limits,
    _github_api_paginate_terminal_call_limits,
    _github_pr_feedback_terminal_call_ids,
    _github_pr_feedback_terminal_result,
    _github_list_terminal_call_limits,
    _git_diff_name_only_terminal_call_ids,
    _git_grep_terminal_call_ids,
    _git_review_summary_terminal_call_ids,
    _kanban_assignees_terminal_call_ids,
    _plain_github_list_terminal_call_ids,
    _project_combined_github_list_terminal_result,
    _project_combined_github_view_terminal_result,
    _project_github_api_extract_result,
    _project_github_api_paginate_terminal_result,
    _project_github_list_terminal_result,
    _project_git_diff_name_only_terminal_result,
    _project_git_review_summary_terminal_result,
    _project_kanban_assignees_terminal_result,
    _project_line_numbered_search_terminal_result,
    _pytest_terminal_call_ids,
    _PYTEST_DIAGNOSTIC_MAX_BYTES,
    _PYTEST_DIAGNOSTIC_MAX_LINES,
    _rejected_terminal_call_ids,
    _recognized_syntax_tool_call_ids,
    _rg_terminal_call_ids,
    _safe_repo_relative_path,
    _scratch_read_file_tool_call_ids,
    _segment_protected_context,
    _segment_protected_tool_result,
    _segment_read_file_presentation,
    _segment_text,
)
from agent.redact import redact_sensitive_text
from agent.source_provenance import SourceProvenanceRegistry

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
_REMOTE_KANBAN_ATTACHMENT_TOOL_NAMES = frozenset({"kanban_attachments"})
_REMOTE_KANBAN_TOOL_SEARCH_PROJECTION_TOOL_NAMES = frozenset(
    {"tool_search", "tool_describe"}
)
_REMOTE_KANBAN_LIFECYCLE_TOOL_NAMES = frozenset(
    {
        "kanban_attach",
        "kanban_attach_url",
        "kanban_block",
        "kanban_comment",
        "kanban_complete",
        "kanban_heartbeat",
        "kanban_link",
        "kanban_request_changes",
        "kanban_request_review",
    }
)
_REMOTE_KANBAN_TASK_SPEC_VERSION = "v1"
_REMOTE_KANBAN_TASK_TITLE_MAX_BYTES = 1024
_REMOTE_KANBAN_TASK_BODY_MAX_BYTES = 8 * 1024
_REMOTE_KANBAN_ATTACHMENT_ELISION = (
    "kanban_attachments completed locally; attachment metadata and contents "
    "were omitted from remote replay. Continue with the assigned work or use a lifecycle tool."
)
_REMOTE_KANBAN_LIFECYCLE_ELISION = (
    "Kanban lifecycle action completed locally; its raw control-plane result "
    "was omitted from remote replay."
)
_FILE_MUTATION_ERROR_MAX_BYTES = 1024
_GITHUB_API_PAGINATE_ARGUMENT_REPLAY = (
    '{"command":"gh api --paginate GitHub REST list (details omitted)"}'
)
def _project_bound_kanban_lifecycle(value: str) -> GeneratedContextSegment:
    """Replay only a fixed outcome for an exact local lifecycle call."""

    # Lifecycle results can include comment text, paths, opaque ids, or
    # backend errors. The worker only needs the fact that its local action
    # returned; exact call-id binding is enforced by the caller.
    return GeneratedContextSegment(_REMOTE_KANBAN_LIFECYCLE_ELISION)

def _project_bound_kanban_show(value: str) -> GeneratedContextSegment:
    """Expose only the redacted current assignment needed by a remote worker."""

    try:
        payload = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return GeneratedContextSegment(_REMOTE_KANBAN_PROJECTION_ELISION)
    task = payload.get("task") if isinstance(payload, dict) else None
    if not isinstance(task, dict):
        return GeneratedContextSegment(_REMOTE_KANBAN_PROJECTION_ELISION)

    # Only the exact versioned producer contract may carry assignment text.
    # Forged/unbound board-shaped JSON stays on the elision path. The producer
    # has already capped the fields and the redaction/final scans remain
    # mandatory before this generated context can leave the host.
    task_spec = payload.get("protected_task_spec")

    def bounded_text(item: Any, max_bytes: int) -> str:
        text = item if isinstance(item, str) else ""
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        suffix = "\n<truncated>"
        budget = max(0, max_bytes - len(suffix.encode("utf-8")))
        return encoded[:budget].decode("utf-8", errors="ignore") + suffix

    projected_task: dict[str, Any] = {
        key: task[key]
        for key in ("status", "workspace_access")
        if key in task
    }
    if (
        isinstance(task_spec, dict)
        and task_spec.get("version") == _REMOTE_KANBAN_TASK_SPEC_VERSION
    ):
        projected_task.update(
            {
                "title": bounded_text(
                    task_spec.get("title"), _REMOTE_KANBAN_TASK_TITLE_MAX_BYTES
                ),
                "body": bounded_text(
                    task_spec.get("body"), _REMOTE_KANBAN_TASK_BODY_MAX_BYTES
                ),
            }
        )

    projection = {
        "task": projected_task,
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

def _project_bound_kanban_attachments(value: str) -> GeneratedContextSegment:
    """Elide attachment payloads while preserving exact call/result binding."""

    # Attachment records can contain source excerpts, credentials, and opaque
    # blobs.  The worker already has the bounded task assignment; replaying
    # attachment content is unnecessary and would make the protected route
    # pay for a retry when provenance cannot be established.
    return GeneratedContextSegment(_REMOTE_KANBAN_ATTACHMENT_ELISION)

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

def _project_bound_tool_search(value: str) -> GeneratedContextSegment:
    """Replay only the bounded outcome of local tool catalog discovery."""

    return GeneratedContextSegment(
        "tool_search completed locally. Its catalog result was omitted from "
        "remote replay; use the already connected terminal tool."
    )

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

def _typed_payload(value: Any, grant_texts: Sequence[tuple[str, SourceGrant]], used_grants: dict[str, SourceGrant], **kwargs: Any) -> Any:
    """Classify a payload through the scalar, mapping, or sequence path."""

    if isinstance(value, str):
        return _typed_payload_string(value, grant_texts, used_grants, **kwargs)
    if isinstance(value, (list, tuple)):
        return _typed_payload_sequence(value, grant_texts, used_grants, **kwargs)
    return _typed_payload_mapping(value, grant_texts, used_grants, **kwargs)


def _typed_payload_string(
    value: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
    field_name: str | None = None,
    protected_tool_content: bool = False,
    elide_kanban_tool_content: bool = False,
    kanban_attachment_tool_content: bool = False,
    protected_kanban_context: bool = False,
    generated_context: bool = False,
    redact_generated_context: bool = False,
    **_: Any,
) -> Any:
    """Classify a scalar without carrying mapping-only policy state."""

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
        return _project_bound_kanban_show(value)
    if kanban_attachment_tool_content:
        return _project_bound_kanban_attachments(value)
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


def _typed_payload_sequence(
    value: list[Any] | tuple[Any, ...],
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    **kwargs: Any,
) -> list[Any]:
    """Recursively classify sequence items through the shared dispatcher."""

    return [_typed_payload(item, grant_texts, used_grants, **kwargs) for item in value]


def _typed_payload_mapping(
    value: Any,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
    field_name: str | None = None,
    syntax_tool_call_ids: frozenset[str] = frozenset(),
    pytest_terminal_call_ids: frozenset[str] = frozenset(),
    elided_kanban_tool_call_ids: frozenset[str] = frozenset(),
    kanban_attachment_tool_call_ids: frozenset[str] = frozenset(),
    kanban_lifecycle_tool_call_ids: frozenset[str] = frozenset(),
    search_projection_tool_call_ids: frozenset[str] = frozenset(),
    tool_search_projection_tool_call_ids: frozenset[str] = frozenset(),
    read_file_projection_tool_call_ids: frozenset[str] = frozenset(),
    web_replay_tool_call_ids: frozenset[str] = frozenset(),
    file_mutation_replay_tool_call_ids: frozenset[str] = frozenset(),
    scratch_read_file_tool_call_ids: frozenset[str] = frozenset(),
    git_workspace_diagnostic_call_ids: frozenset[str] = frozenset(),
    git_grep_projection_tool_call_ids: frozenset[str] = frozenset(),
    rg_projection_tool_call_ids: frozenset[str] = frozenset(),
    git_diff_name_only_projection_tool_call_ids: frozenset[str] = frozenset(),
    git_review_summary_projection_tool_call_ids: frozenset[str] = frozenset(),
    github_pr_feedback_terminal_call_ids: frozenset[str] = frozenset(),
    kanban_assignees_terminal_call_ids: frozenset[str] = frozenset(),
    github_list_terminal_call_limits: Mapping[str, int] | None = None,
    github_api_extract_call_limits: Mapping[str, int] | None = None,
    github_api_paginate_call_limits: Mapping[str, int] | None = None,
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
    kanban_attachment_tool_content: bool = False,
    protected_kanban_context: bool = False,
    generated_context: bool = False,
    redact_generated_context: bool = False,
    allow_codex_reasoning_replay: bool = False,
    registry: SourceProvenanceRegistry | None = None,
    request_identity: tuple[str, str, str, str] = ("", "", "", ""),
) -> Any:
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
        is_kanban_attachment_result = (
            isinstance(output_call_id, str)
            and output_call_id in kanban_attachment_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_kanban_lifecycle_result = (
            isinstance(output_call_id, str)
            and output_call_id in kanban_lifecycle_tool_call_ids
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
        is_tool_search_projection_result = (
            isinstance(output_call_id, str)
            and output_call_id in tool_search_projection_tool_call_ids
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
        is_git_diff_name_only_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in git_diff_name_only_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_git_review_summary_projection_tool_result = (
            isinstance(output_call_id, str)
            and output_call_id in git_review_summary_projection_tool_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_pytest_terminal_result = (
            isinstance(output_call_id, str)
            and output_call_id in pytest_terminal_call_ids
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        is_github_pr_feedback_terminal_result = (
            isinstance(output_call_id, str)
            and output_call_id in github_pr_feedback_terminal_call_ids
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
        github_api_paginate_limit = (
            github_api_paginate_call_limits.get(output_call_id)
            if isinstance(github_api_paginate_call_limits, Mapping)
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
        is_tool_result_mapping = (
            isinstance(output_call_id, str)
            and (
                value.get("role") == "tool"
                or value.get("type") == "function_call_output"
            )
        )
        # A protected remote worker may only replay a tool result through one
        # of the exact call-id-bound projections above.  If a provider or
        # bridge hands us a tool result without the preceding recognized call,
        # keep it explicitly untrusted so the final firewall fails closed
        # instead of treating its text as ordinary sanitized context.
        handled_tool_result = any(
            (
                is_recognized_tool_result,
                is_elided_kanban_tool_result,
                is_kanban_attachment_result,
                is_kanban_lifecycle_result,
                is_search_projection_tool_result,
                is_tool_search_projection_result,
                is_read_file_projection_tool_result,
                is_web_replay_tool_result,
                is_file_mutation_replay_result,
                is_scratch_read_file_tool_result,
                is_git_workspace_diagnostic_result,
                is_git_grep_projection_tool_result,
                is_rg_projection_tool_result,
                is_git_diff_name_only_projection_tool_result,
                is_git_review_summary_projection_tool_result,
                is_pytest_terminal_result,
                is_github_pr_feedback_terminal_result,
                is_kanban_assignees_result,
                isinstance(github_list_limit, int),
                isinstance(github_api_extract_limit, int),
                isinstance(github_api_paginate_limit, int),
                is_github_api_curl_terminal_call,
                is_plain_github_list_terminal_result,
                isinstance(combined_github_list_limit, int),
                isinstance(combined_github_view_limit, int),
                is_terminal_replay_result,
                is_read_file_result,
            )
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
        # Assistant turns are provider-generated history.  They can echo a
        # locally granted source excerpt after a tool call; replaying that
        # echo as an ordinary sanitized segment would trip the provenance
        # overlap guard on the next cloud request.  Treat only this generated
        # role as application context for the remote-safe redaction path;
        # user/task content remains fail-closed.
        generated_assistant_mapping = value.get("role") == "assistant"
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
                    is_kanban_attachment_result,
                    is_kanban_lifecycle_result,
                    is_search_projection_tool_result,
                    is_tool_search_projection_result,
                    is_read_file_projection_tool_result,
                    is_web_replay_tool_result,
                    is_file_mutation_replay_result,
                    is_scratch_read_file_tool_result,
                    is_git_workspace_diagnostic_result,
                    is_git_grep_projection_tool_result,
                    is_rg_projection_tool_result,
                    is_git_diff_name_only_projection_tool_result,
                    is_git_review_summary_projection_tool_result,
                    is_pytest_terminal_result,
                    is_github_pr_feedback_terminal_result,
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
        mapping_state = locals()
        for key, item in value.items():
            _typed_payload_mapping_item(mapping_state, key, item, typed)
        return typed
    return value


def _typed_payload_mapping_item(
    state: Mapping[str, Any], key: Any, item: Any, typed: dict[Any, Any]
) -> None:
    """Classify one mapping field while keeping projection families isolated."""
    value = state['value']
    grant_texts = state['grant_texts']
    used_grants = state['used_grants']
    sanitized_cap = state['sanitized_cap']
    field_name = state['field_name']
    syntax_tool_call_ids = state['syntax_tool_call_ids']
    pytest_terminal_call_ids = state['pytest_terminal_call_ids']
    elided_kanban_tool_call_ids = state['elided_kanban_tool_call_ids']
    kanban_attachment_tool_call_ids = state['kanban_attachment_tool_call_ids']
    kanban_lifecycle_tool_call_ids = state['kanban_lifecycle_tool_call_ids']
    search_projection_tool_call_ids = state['search_projection_tool_call_ids']
    tool_search_projection_tool_call_ids = state['tool_search_projection_tool_call_ids']
    read_file_projection_tool_call_ids = state['read_file_projection_tool_call_ids']
    web_replay_tool_call_ids = state['web_replay_tool_call_ids']
    file_mutation_replay_tool_call_ids = state['file_mutation_replay_tool_call_ids']
    scratch_read_file_tool_call_ids = state['scratch_read_file_tool_call_ids']
    git_workspace_diagnostic_call_ids = state['git_workspace_diagnostic_call_ids']
    git_grep_projection_tool_call_ids = state['git_grep_projection_tool_call_ids']
    rg_projection_tool_call_ids = state['rg_projection_tool_call_ids']
    git_diff_name_only_projection_tool_call_ids = state['git_diff_name_only_projection_tool_call_ids']
    git_review_summary_projection_tool_call_ids = state['git_review_summary_projection_tool_call_ids']
    github_pr_feedback_terminal_call_ids = state['github_pr_feedback_terminal_call_ids']
    kanban_assignees_terminal_call_ids = state['kanban_assignees_terminal_call_ids']
    github_list_terminal_call_limits = state['github_list_terminal_call_limits']
    github_api_extract_call_limits = state['github_api_extract_call_limits']
    github_api_paginate_call_limits = state['github_api_paginate_call_limits']
    github_api_curl_terminal_call_ids = state['github_api_curl_terminal_call_ids']
    plain_github_list_terminal_call_ids = state['plain_github_list_terminal_call_ids']
    combined_github_list_terminal_call_limits = state['combined_github_list_terminal_call_limits']
    combined_github_view_terminal_call_limits = state['combined_github_view_terminal_call_limits']
    rejected_terminal_call_ids = state['rejected_terminal_call_ids']
    terminal_replay_tool_call_ids = state['terminal_replay_tool_call_ids']
    redact_terminal_arguments = state['redact_terminal_arguments']
    redact_readonly_tool_arguments = state['redact_readonly_tool_arguments']
    protected_tool_content = state['protected_tool_content']
    elide_kanban_tool_content = state['elide_kanban_tool_content']
    kanban_attachment_tool_content = state['kanban_attachment_tool_content']
    protected_kanban_context = state['protected_kanban_context']
    generated_context = state['generated_context']
    redact_generated_context = state['redact_generated_context']
    allow_codex_reasoning_replay = state['allow_codex_reasoning_replay']
    registry = state['registry']
    request_identity = state['request_identity']
    combined_github_list_limit = state['combined_github_list_limit']
    combined_github_view_limit = state['combined_github_view_limit']
    context_mapping = state['context_mapping']
    direct_function = state['direct_function']
    direct_name = state['direct_name']
    generated_assistant_mapping = state['generated_assistant_mapping']
    github_api_extract_limit = state['github_api_extract_limit']
    github_api_paginate_limit = state['github_api_paginate_limit']
    github_list_limit = state['github_list_limit']
    handled_tool_result = state['handled_tool_result']
    is_codex_reasoning_replay = state['is_codex_reasoning_replay']
    is_elided_kanban_tool_result = state['is_elided_kanban_tool_result']
    is_file_mutation_replay_call = state['is_file_mutation_replay_call']
    is_file_mutation_replay_result = state['is_file_mutation_replay_result']
    is_git_diff_name_only_projection_tool_result = state['is_git_diff_name_only_projection_tool_result']
    is_git_grep_projection_tool_result = state['is_git_grep_projection_tool_result']
    is_git_review_summary_projection_tool_result = state['is_git_review_summary_projection_tool_result']
    is_git_workspace_diagnostic_result = state['is_git_workspace_diagnostic_result']
    is_github_api_curl_terminal_call = state['is_github_api_curl_terminal_call']
    is_github_pr_feedback_terminal_result = state['is_github_pr_feedback_terminal_result']
    is_kanban_assignees_result = state['is_kanban_assignees_result']
    is_kanban_attachment_result = state['is_kanban_attachment_result']
    is_kanban_lifecycle_result = state['is_kanban_lifecycle_result']
    is_plain_github_list_terminal_result = state['is_plain_github_list_terminal_result']
    is_pytest_terminal_result = state['is_pytest_terminal_result']
    is_read_file_projection_tool_result = state['is_read_file_projection_tool_result']
    is_read_file_result = state['is_read_file_result']
    is_recognized_tool_result = state['is_recognized_tool_result']
    is_rejected_terminal_call = state['is_rejected_terminal_call']
    is_rg_projection_tool_result = state['is_rg_projection_tool_result']
    is_scratch_read_file_tool_result = state['is_scratch_read_file_tool_result']
    is_search_projection_tool_result = state['is_search_projection_tool_result']
    is_terminal_replay_call = state['is_terminal_replay_call']
    is_terminal_replay_result = state['is_terminal_replay_result']
    is_tool_protocol_mapping = state['is_tool_protocol_mapping']
    is_tool_result_mapping = state['is_tool_result_mapping']
    is_tool_search_projection_result = state['is_tool_search_projection_result']
    is_untrusted_tool_result = state['is_untrusted_tool_result']
    is_web_replay_tool_result = state['is_web_replay_tool_result']
    output_call_id = state['output_call_id']
    source_metadata = state['source_metadata']
    if key == "_source_provenance":
        return True
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
        return True
    if _typed_payload_mapping_structured(state, key, item, typed):
        return
    if _typed_payload_mapping_scalar(state, key, item, typed):
        return
    typed_key = (
        GeneratedContextKey(key)
        if generated_context and redact_generated_context
        else key
    )
    if is_codex_reasoning_replay and key == "encrypted_content":
        typed[typed_key] = CodexReasoningReplaySegment(item)
        return True
    if is_codex_reasoning_replay and key == "summary":
        typed[typed_key] = _typed_payload(
            item,
            grant_texts,
            used_grants,
            sanitized_cap=sanitized_cap,
            field_name=key,
            generated_context=True,
            redact_generated_context=True,
            tool_search_projection_tool_call_ids=tool_search_projection_tool_call_ids,
            registry=registry,
            request_identity=request_identity,
        )
        return True
    typed[typed_key] = _typed_payload(
        item,
        grant_texts,
        used_grants,
        sanitized_cap=sanitized_cap,
        field_name=key,
        syntax_tool_call_ids=syntax_tool_call_ids,
        pytest_terminal_call_ids=pytest_terminal_call_ids,
        elided_kanban_tool_call_ids=elided_kanban_tool_call_ids,
        kanban_attachment_tool_call_ids=kanban_attachment_tool_call_ids,
        kanban_lifecycle_tool_call_ids=kanban_lifecycle_tool_call_ids,
        search_projection_tool_call_ids=search_projection_tool_call_ids,
        tool_search_projection_tool_call_ids=tool_search_projection_tool_call_ids,
        read_file_projection_tool_call_ids=read_file_projection_tool_call_ids,
        web_replay_tool_call_ids=web_replay_tool_call_ids,
        file_mutation_replay_tool_call_ids=file_mutation_replay_tool_call_ids,
        scratch_read_file_tool_call_ids=scratch_read_file_tool_call_ids,
        git_workspace_diagnostic_call_ids=git_workspace_diagnostic_call_ids,
        git_grep_projection_tool_call_ids=git_grep_projection_tool_call_ids,
        rg_projection_tool_call_ids=rg_projection_tool_call_ids,
        git_diff_name_only_projection_tool_call_ids=git_diff_name_only_projection_tool_call_ids,
        git_review_summary_projection_tool_call_ids=git_review_summary_projection_tool_call_ids,
        github_pr_feedback_terminal_call_ids=github_pr_feedback_terminal_call_ids,
        kanban_assignees_terminal_call_ids=kanban_assignees_terminal_call_ids,
        github_list_terminal_call_limits=github_list_terminal_call_limits,
        github_api_extract_call_limits=github_api_extract_call_limits,
        github_api_paginate_call_limits=github_api_paginate_call_limits,
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
        kanban_attachment_tool_content=(
            is_kanban_attachment_result and key in {"content", "output"}
        ),
        protected_kanban_context=protected_kanban_context,
        generated_context=(
            redact_generated_context
            and (
                generated_context
                or context_mapping
                or generated_assistant_mapping
                or key in {"instructions", "system_prompt", "tools"}
            )
        ),
        redact_generated_context=redact_generated_context,
        allow_codex_reasoning_replay=allow_codex_reasoning_replay,
        registry=registry,
        request_identity=request_identity,
    )
    return


def _typed_payload_mapping_structured(
    state: Mapping[str, Any], key: Any, item: Any, typed: dict[Any, Any]
) -> bool:
    """Handle Responses-style structured tool output projections."""
    value = state['value']
    grant_texts = state['grant_texts']
    used_grants = state['used_grants']
    sanitized_cap = state['sanitized_cap']
    field_name = state['field_name']
    syntax_tool_call_ids = state['syntax_tool_call_ids']
    pytest_terminal_call_ids = state['pytest_terminal_call_ids']
    elided_kanban_tool_call_ids = state['elided_kanban_tool_call_ids']
    kanban_attachment_tool_call_ids = state['kanban_attachment_tool_call_ids']
    kanban_lifecycle_tool_call_ids = state['kanban_lifecycle_tool_call_ids']
    search_projection_tool_call_ids = state['search_projection_tool_call_ids']
    tool_search_projection_tool_call_ids = state['tool_search_projection_tool_call_ids']
    read_file_projection_tool_call_ids = state['read_file_projection_tool_call_ids']
    web_replay_tool_call_ids = state['web_replay_tool_call_ids']
    file_mutation_replay_tool_call_ids = state['file_mutation_replay_tool_call_ids']
    scratch_read_file_tool_call_ids = state['scratch_read_file_tool_call_ids']
    git_workspace_diagnostic_call_ids = state['git_workspace_diagnostic_call_ids']
    git_grep_projection_tool_call_ids = state['git_grep_projection_tool_call_ids']
    rg_projection_tool_call_ids = state['rg_projection_tool_call_ids']
    git_diff_name_only_projection_tool_call_ids = state['git_diff_name_only_projection_tool_call_ids']
    git_review_summary_projection_tool_call_ids = state['git_review_summary_projection_tool_call_ids']
    github_pr_feedback_terminal_call_ids = state['github_pr_feedback_terminal_call_ids']
    kanban_assignees_terminal_call_ids = state['kanban_assignees_terminal_call_ids']
    github_list_terminal_call_limits = state['github_list_terminal_call_limits']
    github_api_extract_call_limits = state['github_api_extract_call_limits']
    github_api_paginate_call_limits = state['github_api_paginate_call_limits']
    github_api_curl_terminal_call_ids = state['github_api_curl_terminal_call_ids']
    plain_github_list_terminal_call_ids = state['plain_github_list_terminal_call_ids']
    combined_github_list_terminal_call_limits = state['combined_github_list_terminal_call_limits']
    combined_github_view_terminal_call_limits = state['combined_github_view_terminal_call_limits']
    rejected_terminal_call_ids = state['rejected_terminal_call_ids']
    terminal_replay_tool_call_ids = state['terminal_replay_tool_call_ids']
    redact_terminal_arguments = state['redact_terminal_arguments']
    redact_readonly_tool_arguments = state['redact_readonly_tool_arguments']
    protected_tool_content = state['protected_tool_content']
    elide_kanban_tool_content = state['elide_kanban_tool_content']
    kanban_attachment_tool_content = state['kanban_attachment_tool_content']
    protected_kanban_context = state['protected_kanban_context']
    generated_context = state['generated_context']
    redact_generated_context = state['redact_generated_context']
    allow_codex_reasoning_replay = state['allow_codex_reasoning_replay']
    registry = state['registry']
    request_identity = state['request_identity']
    combined_github_list_limit = state['combined_github_list_limit']
    combined_github_view_limit = state['combined_github_view_limit']
    context_mapping = state['context_mapping']
    direct_function = state['direct_function']
    direct_name = state['direct_name']
    generated_assistant_mapping = state['generated_assistant_mapping']
    github_api_extract_limit = state['github_api_extract_limit']
    github_api_paginate_limit = state['github_api_paginate_limit']
    github_list_limit = state['github_list_limit']
    handled_tool_result = state['handled_tool_result']
    is_codex_reasoning_replay = state['is_codex_reasoning_replay']
    is_elided_kanban_tool_result = state['is_elided_kanban_tool_result']
    is_file_mutation_replay_call = state['is_file_mutation_replay_call']
    is_file_mutation_replay_result = state['is_file_mutation_replay_result']
    is_git_diff_name_only_projection_tool_result = state['is_git_diff_name_only_projection_tool_result']
    is_git_grep_projection_tool_result = state['is_git_grep_projection_tool_result']
    is_git_review_summary_projection_tool_result = state['is_git_review_summary_projection_tool_result']
    is_git_workspace_diagnostic_result = state['is_git_workspace_diagnostic_result']
    is_github_api_curl_terminal_call = state['is_github_api_curl_terminal_call']
    is_github_pr_feedback_terminal_result = state['is_github_pr_feedback_terminal_result']
    is_kanban_assignees_result = state['is_kanban_assignees_result']
    is_kanban_attachment_result = state['is_kanban_attachment_result']
    is_kanban_lifecycle_result = state['is_kanban_lifecycle_result']
    is_plain_github_list_terminal_result = state['is_plain_github_list_terminal_result']
    is_pytest_terminal_result = state['is_pytest_terminal_result']
    is_read_file_projection_tool_result = state['is_read_file_projection_tool_result']
    is_read_file_result = state['is_read_file_result']
    is_recognized_tool_result = state['is_recognized_tool_result']
    is_rejected_terminal_call = state['is_rejected_terminal_call']
    is_rg_projection_tool_result = state['is_rg_projection_tool_result']
    is_scratch_read_file_tool_result = state['is_scratch_read_file_tool_result']
    is_search_projection_tool_result = state['is_search_projection_tool_result']
    is_terminal_replay_call = state['is_terminal_replay_call']
    is_terminal_replay_result = state['is_terminal_replay_result']
    is_tool_protocol_mapping = state['is_tool_protocol_mapping']
    is_tool_result_mapping = state['is_tool_result_mapping']
    is_tool_search_projection_result = state['is_tool_search_projection_result']
    is_untrusted_tool_result = state['is_untrusted_tool_result']
    is_web_replay_tool_result = state['is_web_replay_tool_result']
    output_call_id = state['output_call_id']
    source_metadata = state['source_metadata']
    is_structured_result = (
        key in {"content", "output"}
        and isinstance(item, (list, Mapping))
    )
    structured_text = (
        _structured_tool_output_text(item) if is_structured_result else None
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
            return True
    structured_text = (
        _structured_tool_output_text(item) if is_structured_result else None
    )
    if is_structured_result and is_kanban_lifecycle_result:
        typed[key] = _project_bound_kanban_lifecycle(structured_text or "")
        return True
    if is_kanban_assignees_result and structured_text is not None:
        projected = _project_kanban_assignees_terminal_result(structured_text)
        if projected is not None:
            typed[key] = GeneratedContextSegment(projected)
            return True
    if isinstance(github_list_limit, int) and structured_text is not None:
        projected = _project_github_list_terminal_result(
            structured_text, max_rows=github_list_limit
        )
        if projected is not None:
            typed[key] = GeneratedContextSegment(projected)
            return True
    if (
        isinstance(github_api_paginate_limit, int)
        and structured_text is not None
    ):
        projected = _project_github_api_paginate_terminal_result(
            structured_text, max_rows=github_api_paginate_limit
        )
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
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
            return True
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
            return True
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
            segment = _segment_read_file_presentation(
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
            if isinstance(segment, UntrustedProvenanceSegment) and protected_kanban_context:
                typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
            else:
                typed[key] = _replace_structured_tool_output_text(item, segment)
            return True
        typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
        return True
    if is_structured_result and (
        is_search_projection_tool_result
        or is_tool_search_projection_result
        or is_git_grep_projection_tool_result
        or is_rg_projection_tool_result
    ):
        typed[key] = GeneratedContextSegment(
            _STRUCTURED_SEARCH_REPLAY_ELISION
        )
        return True
    if (
        is_structured_result
        and is_git_diff_name_only_projection_tool_result
        and structured_text is not None
    ):
        projected = _project_git_diff_name_only_terminal_result(structured_text)
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
    if (
        is_structured_result
        and is_git_review_summary_projection_tool_result
        and structured_text is not None
    ):
        projected = _project_git_review_summary_terminal_result(structured_text)
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
    if (
        is_structured_result
        and is_pytest_terminal_result
        and structured_text is not None
    ):
        typed[key] = GeneratedContextSegment(
            _pytest_terminal_result(structured_text)
        )
        return True
    if (
        is_structured_result
        and is_web_replay_tool_result
        and github_api_extract_limit is None
    ):
        if structured_text is not None:
            typed[key] = _project_web_search_replay(structured_text)
            return True
    if is_structured_result and is_file_mutation_replay_result:
        projected = _project_file_mutation_result(structured_text or "")
        typed[key] = GeneratedContextSegment(projected)
        return True
    if is_structured_result and is_git_workspace_diagnostic_result:
        typed[key] = GeneratedContextSegment(
            _GIT_WORKSPACE_DIAGNOSTIC_REPLAY
        )
        return True
    if is_structured_result and is_plain_github_list_terminal_result:
        typed[key] = GeneratedContextSegment(
            _GITHUB_PLAIN_LIST_OUTPUT_REPLAY
        )
        return True
    if is_structured_result and is_github_pr_feedback_terminal_result:
        typed[key] = GeneratedContextSegment(
            _github_pr_feedback_terminal_result(structured_text or "")
        )
        return True
    if is_structured_result and is_terminal_replay_result:
        # The Responses API represents function-call output as an
        # array of input_text/input_image items.  A recognized local
        # terminal call gets the same outcome-only replay boundary as
        # its scalar counterpart; recursively typing the array would
        # expose raw stdout to the remote firewall.
        typed[key] = GeneratedContextSegment(_terminal_replay_result(""))
        return True
    return False


def _typed_payload_mapping_scalar(
    state: Mapping[str, Any], key: Any, item: Any, typed: dict[Any, Any]
) -> bool:
    """Dispatch scalar tool-result and terminal replay projections."""
    if _typed_payload_mapping_scalar_content(state, key, item, typed):
        return True
    return _typed_payload_mapping_scalar_arguments(state, key, item, typed)


def _typed_payload_mapping_scalar_content(
    state: Mapping[str, Any], key: Any, item: Any, typed: dict[Any, Any]
) -> bool:
    """Handle content projections and bounded terminal output."""
    value = state['value']
    grant_texts = state['grant_texts']
    used_grants = state['used_grants']
    sanitized_cap = state['sanitized_cap']
    field_name = state['field_name']
    syntax_tool_call_ids = state['syntax_tool_call_ids']
    pytest_terminal_call_ids = state['pytest_terminal_call_ids']
    elided_kanban_tool_call_ids = state['elided_kanban_tool_call_ids']
    kanban_attachment_tool_call_ids = state['kanban_attachment_tool_call_ids']
    kanban_lifecycle_tool_call_ids = state['kanban_lifecycle_tool_call_ids']
    search_projection_tool_call_ids = state['search_projection_tool_call_ids']
    tool_search_projection_tool_call_ids = state['tool_search_projection_tool_call_ids']
    read_file_projection_tool_call_ids = state['read_file_projection_tool_call_ids']
    web_replay_tool_call_ids = state['web_replay_tool_call_ids']
    file_mutation_replay_tool_call_ids = state['file_mutation_replay_tool_call_ids']
    scratch_read_file_tool_call_ids = state['scratch_read_file_tool_call_ids']
    git_workspace_diagnostic_call_ids = state['git_workspace_diagnostic_call_ids']
    git_grep_projection_tool_call_ids = state['git_grep_projection_tool_call_ids']
    rg_projection_tool_call_ids = state['rg_projection_tool_call_ids']
    git_diff_name_only_projection_tool_call_ids = state['git_diff_name_only_projection_tool_call_ids']
    git_review_summary_projection_tool_call_ids = state['git_review_summary_projection_tool_call_ids']
    github_pr_feedback_terminal_call_ids = state['github_pr_feedback_terminal_call_ids']
    kanban_assignees_terminal_call_ids = state['kanban_assignees_terminal_call_ids']
    github_list_terminal_call_limits = state['github_list_terminal_call_limits']
    github_api_extract_call_limits = state['github_api_extract_call_limits']
    github_api_paginate_call_limits = state['github_api_paginate_call_limits']
    github_api_curl_terminal_call_ids = state['github_api_curl_terminal_call_ids']
    plain_github_list_terminal_call_ids = state['plain_github_list_terminal_call_ids']
    combined_github_list_terminal_call_limits = state['combined_github_list_terminal_call_limits']
    combined_github_view_terminal_call_limits = state['combined_github_view_terminal_call_limits']
    rejected_terminal_call_ids = state['rejected_terminal_call_ids']
    terminal_replay_tool_call_ids = state['terminal_replay_tool_call_ids']
    redact_terminal_arguments = state['redact_terminal_arguments']
    redact_readonly_tool_arguments = state['redact_readonly_tool_arguments']
    protected_tool_content = state['protected_tool_content']
    elide_kanban_tool_content = state['elide_kanban_tool_content']
    kanban_attachment_tool_content = state['kanban_attachment_tool_content']
    protected_kanban_context = state['protected_kanban_context']
    generated_context = state['generated_context']
    redact_generated_context = state['redact_generated_context']
    allow_codex_reasoning_replay = state['allow_codex_reasoning_replay']
    registry = state['registry']
    request_identity = state['request_identity']
    combined_github_list_limit = state['combined_github_list_limit']
    combined_github_view_limit = state['combined_github_view_limit']
    context_mapping = state['context_mapping']
    direct_function = state['direct_function']
    direct_name = state['direct_name']
    generated_assistant_mapping = state['generated_assistant_mapping']
    github_api_extract_limit = state['github_api_extract_limit']
    github_api_paginate_limit = state['github_api_paginate_limit']
    github_list_limit = state['github_list_limit']
    handled_tool_result = state['handled_tool_result']
    is_codex_reasoning_replay = state['is_codex_reasoning_replay']
    is_elided_kanban_tool_result = state['is_elided_kanban_tool_result']
    is_file_mutation_replay_call = state['is_file_mutation_replay_call']
    is_file_mutation_replay_result = state['is_file_mutation_replay_result']
    is_git_diff_name_only_projection_tool_result = state['is_git_diff_name_only_projection_tool_result']
    is_git_grep_projection_tool_result = state['is_git_grep_projection_tool_result']
    is_git_review_summary_projection_tool_result = state['is_git_review_summary_projection_tool_result']
    is_git_workspace_diagnostic_result = state['is_git_workspace_diagnostic_result']
    is_github_api_curl_terminal_call = state['is_github_api_curl_terminal_call']
    is_github_pr_feedback_terminal_result = state['is_github_pr_feedback_terminal_result']
    is_kanban_assignees_result = state['is_kanban_assignees_result']
    is_kanban_attachment_result = state['is_kanban_attachment_result']
    is_kanban_lifecycle_result = state['is_kanban_lifecycle_result']
    is_plain_github_list_terminal_result = state['is_plain_github_list_terminal_result']
    is_pytest_terminal_result = state['is_pytest_terminal_result']
    is_read_file_projection_tool_result = state['is_read_file_projection_tool_result']
    is_read_file_result = state['is_read_file_result']
    is_recognized_tool_result = state['is_recognized_tool_result']
    is_rejected_terminal_call = state['is_rejected_terminal_call']
    is_rg_projection_tool_result = state['is_rg_projection_tool_result']
    is_scratch_read_file_tool_result = state['is_scratch_read_file_tool_result']
    is_search_projection_tool_result = state['is_search_projection_tool_result']
    is_terminal_replay_call = state['is_terminal_replay_call']
    is_terminal_replay_result = state['is_terminal_replay_result']
    is_tool_protocol_mapping = state['is_tool_protocol_mapping']
    is_tool_result_mapping = state['is_tool_result_mapping']
    is_tool_search_projection_result = state['is_tool_search_projection_result']
    is_untrusted_tool_result = state['is_untrusted_tool_result']
    is_web_replay_tool_result = state['is_web_replay_tool_result']
    output_call_id = state['output_call_id']
    source_metadata = state['source_metadata']
    is_structured_result = (
        key in {"content", "output"}
        and isinstance(item, (list, Mapping))
    )
    structured_text = (
        _structured_tool_output_text(item) if is_structured_result else None
    )
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
        return True
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
            return True
        segment = _segment_read_file_presentation(
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
        if (
            isinstance(segment, UntrustedProvenanceSegment)
            and protected_kanban_context
            and value.get("type") == "function_call_output"
        ):
            typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
        else:
            typed[key] = segment
        return True
    if (
        is_search_projection_tool_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = _project_bound_search_files(item)
        return True
    if (
        is_tool_search_projection_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = _project_bound_tool_search(item)
        return True
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
        return True
    if (
        is_file_mutation_replay_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(
            _project_file_mutation_result(item)
        )
        return True
    if (
        is_kanban_assignees_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        projected = _project_kanban_assignees_terminal_result(item)
        if projected is not None:
            typed[key] = GeneratedContextSegment(projected)
            return True
    if (
        is_git_workspace_diagnostic_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(_GIT_WORKSPACE_DIAGNOSTIC_REPLAY)
        return True
    return False


def _typed_payload_mapping_scalar_arguments(
    state: Mapping[str, Any], key: Any, item: Any, typed: dict[Any, Any]
) -> bool:
    """Handle replay arguments and fail-closed tool output."""
    sanitized_cap = state['sanitized_cap']
    redact_terminal_arguments = state['redact_terminal_arguments']
    redact_readonly_tool_arguments = state['redact_readonly_tool_arguments']
    protected_kanban_context = state['protected_kanban_context']
    combined_github_list_limit = state['combined_github_list_limit']
    combined_github_view_limit = state['combined_github_view_limit']
    direct_name = state['direct_name']
    github_api_extract_limit = state['github_api_extract_limit']
    github_api_paginate_limit = state['github_api_paginate_limit']
    github_list_limit = state['github_list_limit']
    handled_tool_result = state['handled_tool_result']
    is_file_mutation_replay_call = state['is_file_mutation_replay_call']
    is_git_diff_name_only_projection_tool_result = state['is_git_diff_name_only_projection_tool_result']
    is_git_grep_projection_tool_result = state['is_git_grep_projection_tool_result']
    is_git_review_summary_projection_tool_result = state['is_git_review_summary_projection_tool_result']
    is_github_api_curl_terminal_call = state['is_github_api_curl_terminal_call']
    is_github_pr_feedback_terminal_result = state['is_github_pr_feedback_terminal_result']
    is_kanban_lifecycle_result = state['is_kanban_lifecycle_result']
    is_plain_github_list_terminal_result = state['is_plain_github_list_terminal_result']
    is_pytest_terminal_result = state['is_pytest_terminal_result']
    is_rejected_terminal_call = state['is_rejected_terminal_call']
    is_rg_projection_tool_result = state['is_rg_projection_tool_result']
    is_terminal_replay_call = state['is_terminal_replay_call']
    is_terminal_replay_result = state['is_terminal_replay_result']
    is_tool_result_mapping = state['is_tool_result_mapping']
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
            return True
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
            return True
    if (
        is_git_diff_name_only_projection_tool_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        projected = _project_git_diff_name_only_terminal_result(item)
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
    if (
        is_git_review_summary_projection_tool_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        projected = _project_git_review_summary_terminal_result(item)
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
    if (
        is_pytest_terminal_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(_pytest_terminal_result(item))
        return True
    if (
        is_github_pr_feedback_terminal_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(
            _github_pr_feedback_terminal_result(item)
        )
        return True
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
            return True
    if (
        isinstance(github_api_paginate_limit, int)
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        projected = _project_github_api_paginate_terminal_result(
            item, max_rows=github_api_paginate_limit
        )
        if projected is not None:
            typed[key] = GeneratedContextSegment(
                redact_remote_unsafe_text(projected)
            )
            return True
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
            return True
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
        return True
    if (
        isinstance(github_api_paginate_limit, int)
        and key == "arguments"
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(
            _GITHUB_API_PAGINATE_ARGUMENT_REPLAY
        )
        return True
    if is_github_api_curl_terminal_call and key == "arguments":
        typed[key] = GeneratedContextSegment(
            _GITHUB_API_CURL_ARGUMENT_REPLAY
        )
        return True
    if (
        is_plain_github_list_terminal_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(_GITHUB_PLAIN_LIST_OUTPUT_REPLAY)
        return True
    if (
        is_kanban_lifecycle_result
        and key in {"content", "output"}
        and isinstance(item, str)
    ):
        typed[key] = _project_bound_kanban_lifecycle(item)
        return True
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
            return True
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
            return True
    if is_rejected_terminal_call and key == "arguments":
        typed[key] = GeneratedContextSegment(_REJECTED_TERMINAL_COMMAND_REPLAY)
        return True
    if is_terminal_replay_result and key in {"content", "output"} and isinstance(item, str):
        # Terminal stdout is produced locally and can contain source,
        # credentials, or opaque values.  It must not cause a remote
        # worker to fail closed after the local command already ran.
        # Specialized GitHub/search projections above retain the few
        # bounded facts a worker needs; all other stdout is outcome-only.
        typed[key] = GeneratedContextSegment(_terminal_replay_result(item))
        return True
    if is_terminal_replay_call and key == "arguments" and isinstance(item, str):
        typed[key] = GeneratedContextSegment(_terminal_replay_command(item))
        return True
    if (
        protected_kanban_context
        and is_tool_result_mapping
        and key in {"content", "output"}
        and not handled_tool_result
    ):
        raw = (
            item
            if isinstance(item, str)
            else json.dumps(item, ensure_ascii=False, sort_keys=True)
        )
        # Retain the original text as a separately scanned segment so
        # diagnostics still report its concrete content class (for
        # example ``base64_payload``) alongside the provenance denial.
        # The untrusted marker guarantees this payload can never be
        # sent, even when it contains no shape-based violation.
        typed[key] = OutboundText(
            (
                UntrustedProvenanceSegment(
                    sha256(raw.encode("utf-8")).hexdigest()
                ),
                _approved_sanitized(raw, cap=sanitized_cap),
            )
        )
        return True
    if (
        is_file_mutation_replay_call
        and key == "arguments"
        and isinstance(item, str)
    ):
        typed[key] = GeneratedContextSegment(_FILE_MUTATION_ARGUMENT_REPLAY)
        return True
    if (
        redact_terminal_arguments
        and isinstance(direct_name, str)
        and direct_name in _REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES
        and key == "arguments"
        and isinstance(item, str)
    ):
        # Chat-completions nests tool arguments under ``function``;
        # by that recursive pass the outer call ID is unavailable.
        # Every protected worker terminal command is local-only, so
        # retain its coarse command class without replaying raw text.
        typed[key] = GeneratedContextSegment(_terminal_replay_command(item))
        return True
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
        return True
    return False

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

def _replace_structured_tool_output_text(value: Any, text: Any) -> Any:
    """Keep the Responses output container around a typed text segment."""

    if _structured_tool_output_text(value) is None:
        return text
    copied = {"type": LiteralSegment(value[0]["type"])}
    copied["text"] = text
    return [copied]

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

def _pytest_terminal_result(output: str) -> str:
    """Replay bounded pytest failure facts without source or raw stdout."""

    try:
        parsed = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    exit_code = parsed.get("exit_code") if isinstance(parsed, Mapping) else None
    raw_output = parsed.get("output") if isinstance(parsed, Mapping) else None
    if not isinstance(raw_output, str):
        raw_output = ""

    diagnostics: list[str] = []
    diagnostic_bytes = 0
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        summary = (
            line.startswith(("FAILED ", "ERROR ", "E   ", "INTERNALERROR>"))
            or line.startswith("collected ")
            or line.startswith("!!!!!!!!!!!!!!!!")
            or (
                line.startswith("=")
                and line.endswith("=")
                and re.search(
                    r"\b(?:passed|failed|error|errors|skipped|warnings)\b",
                    line,
                    re.IGNORECASE,
                )
                is not None
            )
        )
        if not summary:
            continue
        safe = redact_remote_unsafe_text(
            redact_sensitive_text(line, force=True, redact_url_credentials=True)
        )
        encoded = safe.encode("utf-8")
        if not encoded or diagnostic_bytes + len(encoded) + 1 > _PYTEST_DIAGNOSTIC_MAX_BYTES:
            break
        diagnostics.append(safe)
        diagnostic_bytes += len(encoded) + 1
        if len(diagnostics) >= _PYTEST_DIAGNOSTIC_MAX_LINES:
            break

    return json.dumps(
        {
            "terminal_result": "pytest",
            "exit_code": exit_code if isinstance(exit_code, int) else None,
            "diagnostics": diagnostics,
            "raw_output": "omitted_from_remote_replay",
        },
        separators=(",", ":"),
    )

def _project_file_mutation_result(output: str) -> str:
    """Replay bounded mutation outcome metadata without source or diff text.

    A protected worker must be able to distinguish a landed patch from a
    validation failure. The old fixed elision hid that distinction, so a
    worker could re-apply an already-landed edit or report a false blocker.
    Keep only typed outcome/count fields and a short sanitized error; never
    replay the unified diff, source, or absolute paths.
    """

    try:
        parsed = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, Mapping):
        return _FILE_MUTATION_REPLAY_ELISION
    projection: dict[str, Any] = {
        "file_mutation": "completed",
        "success": bool(parsed.get("success")),
    }
    if parsed.get("no_change") is True:
        projection["no_change"] = True
    for field in ("files_modified", "files_created", "files_deleted"):
        values = parsed.get(field)
        if isinstance(values, list) and values:
            projection[field + "_count"] = len(values)
    error = parsed.get("error")
    if isinstance(error, str) and error.strip():
        safe_error = redact_remote_unsafe_text(
            redact_sensitive_text(
                error.strip()[:_FILE_MUTATION_ERROR_MAX_BYTES],
                force=True,
                redact_url_credentials=True,
            )
        )
        if safe_error:
            projection["error"] = safe_error
    note = parsed.get("note")
    if isinstance(note, str) and note.strip():
        safe_note = redact_remote_unsafe_text(
            redact_sensitive_text(
                note.strip()[:_FILE_MUTATION_ERROR_MAX_BYTES],
                force=True,
                redact_url_credentials=True,
            )
        )
        if safe_note:
            projection["note"] = safe_note
    return json.dumps(projection, separators=(",", ":"))

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
