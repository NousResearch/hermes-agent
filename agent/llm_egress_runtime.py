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
from agent.llm_egress_terminal import (
    _APPLICATION_IDENTIFIER_TOKEN,
    _FILE_MUTATION_ARGUMENT_REPLAY,
    _FILE_MUTATION_ERROR_MAX_BYTES,
    _FILE_MUTATION_REPLAY_ELISION,
    _GITHUB_API_CURL_ARGUMENT_REPLAY,
    _GITHUB_API_EXTRACT_ARGUMENT_REPLAY,
    _GITHUB_LIST_TERMINAL_MAX_ITEM_BYTES,
    _GITHUB_LIST_TERMINAL_MAX_OUTPUT_BYTES,
    _GITHUB_LIST_TERMINAL_MAX_ROWS,
    _GITHUB_PLAIN_LIST_OUTPUT_REPLAY,
    _GITHUB_PR_FEEDBACK_TERMINAL_RESULT_KEYS,
    _GITHUB_PR_FEEDBACK_TERMINAL_SUBCOMMANDS,
    _GIT_DIFF_NAME_ONLY_MAX_FILES,
    _GIT_GREP_TERMINAL_MAX_MATCHES,
    _GIT_REVIEW_SUMMARY_MAX_FILES,
    _GIT_WORKSPACE_DIAGNOSTIC_REPLAY,
    _PYTEST_DIAGNOSTIC_MAX_BYTES,
    _PYTEST_DIAGNOSTIC_MAX_LINES,
    _READ_FILE_REPLAY_ELISION,
    _REJECTED_TERMINAL_COMMAND_REPLAY,
    _REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_PROJECTION_ELISION,
    _REMOTE_KANBAN_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_READONLY_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_READ_FILE_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_SEARCH_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_SECRET_ASSIGNMENT,
    _REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_WEB_REPLAY_TOOL_NAMES,
    _STRUCTURED_SEARCH_REPLAY_ELISION,
    _VALIDATED_SYNTAX_TOOL_NAMES,
    _VERIFIED_DIAGNOSTIC_ATOM,
    _approved_sanitized,
    _approved_sanitized_segments,
    _bounded_remote_text,
    _combined_github_list_terminal_call_limits,
    _combined_github_view_terminal_call_limits,
    _git_diff_name_only_terminal_call_ids,
    _git_grep_terminal_call_ids,
    _git_review_summary_terminal_call_ids,
    _git_workspace_diagnostic_call_ids,
    _github_api_curl_terminal_call_ids,
    _github_api_extract_call_limits,
    _github_api_paginate_terminal_call_limits,
    _github_list_terminal_call_limits,
    _github_pr_feedback_terminal_call_ids,
    _github_pr_feedback_terminal_result,
    _kanban_assignees_terminal_call_ids,
    _plain_github_list_terminal_call_ids,
    _project_combined_github_list_terminal_result,
    _project_combined_github_view_terminal_result,
    _project_git_diff_name_only_terminal_result,
    _project_git_review_summary_terminal_result,
    _project_github_api_extract_result,
    _project_github_api_paginate_terminal_result,
    _project_github_list_terminal_result,
    _project_kanban_assignees_terminal_result,
    _project_line_numbered_search_terminal_result,
    _pytest_terminal_call_ids,
    _recognized_syntax_tool_call_ids,
    _recognized_tool_call_ids,
    _rejected_terminal_call_ids,
    _rg_terminal_call_ids,
    _safe_repo_relative_path,
    _scratch_read_file_tool_call_ids,
    _segment_protected_context,
    _segment_protected_tool_result,
    _segment_read_file_presentation,
    _segment_text,
    _split_utf8_chunks,
)

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

_REMOTE_KANBAN_ATTACHMENT_TOOL_NAMES = frozenset({"kanban_attachments"})
# Local action results are safe to replay only as bounded outcomes, and only
# when the result is bound to the exact preceding call.  Browser Use runs
# through ``browser_exec`` rather than ``terminal``; omitting it here makes a
# protected worker treat its own browser result as untrusted provenance and
# fail on otherwise harmless page identifiers or encoded-looking text.
# Both catalog bridge calls return model-readable tool schemas/descriptions.
# Protected workers must replay only the bounded local outcome; otherwise a
# tool_describe result is treated as untrusted provider content and can trip
# the egress firewall on harmless schema words.
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
_GITHUB_API_PAGINATE_ARGUMENT_REPLAY = (
    '{"command":"gh api --paginate GitHub REST list (details omitted)"}'
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

def _typed_payload(
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
            if is_structured_result and is_kanban_lifecycle_result:
                typed[key] = _project_bound_kanban_lifecycle(structured_text or "")
                continue
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
                        typed[key] = segment
                    continue
                typed[key] = GeneratedContextSegment(_READ_FILE_REPLAY_ELISION)
                continue
            if is_structured_result and (
                is_search_projection_tool_result
                or is_tool_search_projection_result
                or is_git_grep_projection_tool_result
                or is_rg_projection_tool_result
            ):
                typed[key] = GeneratedContextSegment(
                    _STRUCTURED_SEARCH_REPLAY_ELISION
                )
                continue
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
                    continue
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
                    continue
            if (
                is_structured_result
                and is_pytest_terminal_result
                and structured_text is not None
            ):
                typed[key] = GeneratedContextSegment(
                    _pytest_terminal_result(structured_text)
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
                projected = _project_file_mutation_result(structured_text or "")
                typed[key] = GeneratedContextSegment(projected)
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
            if is_structured_result and is_github_pr_feedback_terminal_result:
                typed[key] = GeneratedContextSegment(
                    _github_pr_feedback_terminal_result(structured_text or "")
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
                continue
            if (
                is_search_projection_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = _project_bound_search_files(item)
                continue
            if (
                is_tool_search_projection_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = _project_bound_tool_search(item)
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
                typed[key] = GeneratedContextSegment(
                    _project_file_mutation_result(item)
                )
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
                is_git_diff_name_only_projection_tool_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                projected = _project_git_diff_name_only_terminal_result(item)
                if projected is not None:
                    typed[key] = GeneratedContextSegment(
                        redact_remote_unsafe_text(projected)
                    )
                    continue
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
                    continue
            if (
                is_pytest_terminal_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(_pytest_terminal_result(item))
                continue
            if (
                is_github_pr_feedback_terminal_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(
                    _github_pr_feedback_terminal_result(item)
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
            if (
                isinstance(github_api_paginate_limit, int)
                and key == "arguments"
                and isinstance(item, str)
            ):
                typed[key] = GeneratedContextSegment(
                    _GITHUB_API_PAGINATE_ARGUMENT_REPLAY
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
                is_kanban_lifecycle_result
                and key in {"content", "output"}
                and isinstance(item, str)
            ):
                typed[key] = _project_bound_kanban_lifecycle(item)
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
                continue
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
                protected_tool_content=protected_tool_content,
                elide_kanban_tool_content=elide_kanban_tool_content,
                kanban_attachment_tool_content=kanban_attachment_tool_content,
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

def _is_codex_responses_replay_body(body: Any) -> bool:
    """Return whether *body* carries Codex Responses reasoning replay items."""

    messages = body.get("input") if isinstance(body, Mapping) else None
    if not isinstance(messages, list):
        messages = body.get("messages") if isinstance(body, Mapping) else None
    if not isinstance(messages, list):
        return False
    for message in messages:
        items: list[Any]
        if isinstance(message, Mapping) and message.get("type") == "reasoning":
            items = [message]
        elif isinstance(message, Mapping) and isinstance(message.get("content"), list):
            items = list(message["content"])
        elif isinstance(message, list):
            items = message
        else:
            continue
        for item in items:
            if (
                isinstance(item, Mapping)
                and item.get("type") == "reasoning"
                and isinstance(item.get("encrypted_content"), str)
                and isinstance(item.get("summary", []), list)
            ):
                return True
    return False

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
        pytest_terminal_call_ids=(
            _pytest_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        elided_kanban_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_PROJECTION_TOOL_NAMES)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        kanban_attachment_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_ATTACHMENT_TOOL_NAMES)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        kanban_lifecycle_tool_call_ids=(
            _recognized_tool_call_ids(body, _REMOTE_KANBAN_LIFECYCLE_TOOL_NAMES)
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
        tool_search_projection_tool_call_ids=(
            _recognized_tool_call_ids(
                body, _REMOTE_KANBAN_TOOL_SEARCH_PROJECTION_TOOL_NAMES
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
        git_diff_name_only_projection_tool_call_ids=(
            _git_diff_name_only_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        git_review_summary_projection_tool_call_ids=(
            _git_review_summary_terminal_call_ids(body)
            if protected_kanban_remote and protected_provider_route
            else frozenset()
        ),
        github_pr_feedback_terminal_call_ids=(
            _github_pr_feedback_terminal_call_ids(body)
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
        github_api_paginate_call_limits=(
            _github_api_paginate_terminal_call_limits(body)
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
            and (
                str(getattr(agent, "api_mode", "") or "") == "codex_responses"
                or _is_codex_responses_replay_body(body)
            )
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
    from hermes_constants import get_hermes_home

    _configured_state_dir = getattr(agent, "_llm_egress_state_dir", "")
    state_dir = Path(_configured_state_dir) if _configured_state_dir else (
        Path(get_hermes_home()) / "egress"
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
