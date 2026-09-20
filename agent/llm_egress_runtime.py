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
from agent.llm_egress_classifier import (
    _REMOTE_KANBAN_ATTACHMENT_TOOL_NAMES,
    _REMOTE_KANBAN_LIFECYCLE_TOOL_NAMES,
    _REMOTE_KANBAN_TOOL_SEARCH_PROJECTION_TOOL_NAMES,
    _structural_literal_hashes,
    _structured_tool_output_text,
    _typed_payload,
    _typed_payload_violation_locations,
)
from agent.llm_egress_terminal import (
    _REMOTE_KANBAN_FILE_MUTATION_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_READ_FILE_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_SEARCH_PROJECTION_TOOL_NAMES,
    _REMOTE_KANBAN_TERMINAL_REPLAY_TOOL_NAMES,
    _REMOTE_KANBAN_WEB_REPLAY_TOOL_NAMES,
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
    _kanban_assignees_terminal_call_ids,
    _plain_github_list_terminal_call_ids,
    _pytest_terminal_call_ids,
    _read_grant_text,
    _recognized_syntax_tool_call_ids,
    _recognized_tool_call_ids,
    _rejected_terminal_call_ids,
    _rg_terminal_call_ids,
    _scratch_read_file_tool_call_ids,
)
from agent.message_sanitization import tool_result_id_variants
from agent.redact import redact_sensitive_text
from agent.source_provenance import DEFAULT_POLICY_DIGEST, SourceProvenanceRegistry


# Timeout is a non-content SDK control. Header/query values remain in the
# authorized JSON body so credentials or other caller-controlled text cannot
# be appended after the firewall receipt is written.
_SDK_CONTROL_KEYS = frozenset({"timeout"})
_INTERNAL_EGRESS_KEYS = frozenset({"_hermes_source_provenance"})
_PROTECTED_REMOTE_PROVIDERS = frozenset({
    "anthropic",
    "openai-codex",
    "nous",
    "nous-portal",
    "nousresearch",
})
logger = logging.getLogger(__name__)

# Local action results are safe to replay only as bounded outcomes, and only
# when the result is bound to the exact preceding call.  Browser Use runs
# through ``browser_exec`` rather than ``terminal``; omitting it here makes a
# protected worker treat its own browser result as untrusted provenance and
# fail on otherwise harmless page identifiers or encoded-looking text.
# Both catalog bridge calls return model-readable tool schemas/descriptions.
# Protected workers must replay only the bounded local outcome; otherwise a
# tool_describe result is treated as untrusted provider content and can trip
# the egress firewall on harmless schema words.
_GITHUB_API_PAGINATE_ARGUMENT_REPLAY = (
    '{"command":"gh api --paginate GitHub REST list (details omitted)"}'
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


def egress_enforcement_enabled() -> bool:
    """Return the operator-controlled LLM egress enforcement posture.

    Enforcement remains enabled by default.  The explicit temporary disable
    switch preserves the firewall and its diagnostics while allowing operator
    testing to continue until the false-positive cases are repaired.
    """
    try:
        from hermes_cli import managed_scope
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
        if managed_scope.is_key_managed("runtime.llm_egress_enforcement"):
            posture = str((config.get("runtime") or {}).get("llm_egress_enforcement", "enabled"))
            return posture.strip().lower() not in {"0", "false", "off", "disabled", "disable", "monitor"}
    except Exception:
        pass
    try:
        from hermes_cli.config import load_config_readonly

        runtime = load_config_readonly().get("runtime") or {}
        posture = str(runtime.get("llm_egress_enforcement", "enabled"))
        return posture.strip().lower() not in {"0", "false", "off", "disabled", "disable", "monitor"}
    except Exception:
        # A malformed or unavailable config must not silently weaken the boundary.
        return True


# Benchmark-backed per-profile model route table, installed at startup by
# install_performance_route_table() when a route artifact is configured.
_PERFORMANCE_ROUTE_TABLE: "Any | None" = None


def install_performance_route_table(table: Any) -> None:
    """Activate a compiled benchmark-backed route table for _route_for_agent.

    Call once at startup with the result of
    ``agent.model_performance_router.compile_profile_routes`` (or
    ``hermes_cli.profile_route_compiler.compile_profile_configs``).  The table
    is keyed by profile name then surface name; ``_route_for_agent`` consults
    it when the agent exposes a ``performance_surface`` attribute.
    """
    global _PERFORMANCE_ROUTE_TABLE
    _PERFORMANCE_ROUTE_TABLE = table


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




def _grant_texts(grants: Sequence[SourceGrant]) -> tuple[tuple[str, SourceGrant], ...]:
    unique: dict[str, SourceGrant] = {}
    for grant in grants:
        if not isinstance(grant, SourceGrant):
            continue
        text = _read_grant_text(grant)
        if text:
            unique.setdefault(text, grant)
    return tuple(sorted(unique.items(), key=lambda item: (-len(item[0]), item[0])))


































































































def _route_for_agent(agent: Any, route: Any | None) -> Any:
    if route is not None:
        return route
    # Consult the benchmark-backed performance route table when available and
    # the agent declares a surface.  Falls through to agent defaults on miss.
    surface = str(getattr(agent, "performance_surface", "") or "")
    if _PERFORMANCE_ROUTE_TABLE is not None and surface:
        try:
            from agent.model_performance_router import resolve_route
            profile = str(getattr(agent, "profile", "") or "default")
            privacy = str(getattr(agent, "privacy_class", "") or "sanitized")
            return resolve_route(
                _PERFORMANCE_ROUTE_TABLE,
                profile=profile,
                surface=surface,
                privacy=privacy,
                required_context=0,
            )
        except Exception:
            pass
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
    consumed_entries: set[int] = set()
    if isinstance(messages, list):
        copied_messages = list(messages)
        matches: dict[int, list[int]] = {}
        for entry_index, entry in enumerate(sidecar):
            if not isinstance(entry, Mapping):
                continue
            if entry_index in consumed_entries:
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
            matches.setdefault(index, []).append(entry_index)
        changed = False
        for index, entry_indices in matches.items():
            if len(entry_indices) != 1:
                continue
            entry_index = entry_indices[0]
            entry = sidecar[entry_index]
            copied = dict(copied_messages[index])
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
            consumed_entries.add(entry_index)
        if changed:
            restored["messages"] = copied_messages

    def _restore_input_items(input_items: Any) -> tuple[Any, bool]:
        if not isinstance(input_items, list):
            return input_items, False
        copied_input = list(input_items)
        matches: dict[int, list[int]] = {}
        for entry_index, entry in enumerate(sidecar):
            if not isinstance(entry, Mapping) or entry_index in consumed_entries:
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
            if len(candidates) == 1:
                matches.setdefault(candidates[0], []).append(entry_index)
        changed = False
        for index, entry_indices in matches.items():
            if len(entry_indices) != 1:
                continue
            entry_index = entry_indices[0]
            entry = sidecar[entry_index]
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
            consumed_entries.add(entry_index)
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
    body = _restore_source_provenance_sidecar(body, sidecar)
    classification_body = body
    if protected_kanban_remote:
        body = _sanitize_protected_kanban_body(body)
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
            _scratch_read_file_tool_call_ids(classification_body)
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
    if not egress_enforcement_enabled():
        return callback({key: value for key, value in kwargs.items() if key not in _INTERNAL_EGRESS_KEYS})
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
