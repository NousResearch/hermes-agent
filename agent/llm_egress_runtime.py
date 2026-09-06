"""Final provider-boundary enforcement for source-bound LLM egress."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from agent.llm_egress_firewall import (
    AuthorizedEgress,
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
    DestinationClass,
    classify_destination,
    source_grant_digest,
    static_literal_sha256,
    validate_sanitized_text,
    content_free_violation_locations,
)
from agent.source_provenance import DEFAULT_POLICY_DIGEST, SourceProvenanceRegistry


# Timeout is a non-content SDK control. Header/query values remain in the
# authorized JSON body so credentials or other caller-controlled text cannot
# be appended after the firewall receipt is written.
_SDK_CONTROL_KEYS = frozenset({"timeout"})
_INTERNAL_EGRESS_KEYS = frozenset({"_hermes_source_provenance"})
_PROTOCOL_LITERAL_FIELDS = frozenset({"role", "type"})
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


def _approved_sanitized_segments(text: str, *, cap: int) -> list[SanitizedSegment]:
    """Admit one independently sourced text segment without cap laundering.

    A caller may provide multiple bounded messages or exact-grant-separated
    segments. Arbitrarily slicing one oversized raw value would manufacture
    those independent boundaries and bypass the per-segment policy.
    """

    return [_approved_sanitized(text, cap=cap)]


def _segment_text(
    text: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
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
        sanitized = _approved_sanitized_segments(text, cap=sanitized_cap)
        return sanitized[0] if len(sanitized) == 1 else OutboundText(tuple(sanitized))

    segments: list[SanitizedSegment | SourceBoundSegment] = []
    cursor = 0
    for start, end, grant in matches:
        if start > cursor:
            segments.extend(
                _approved_sanitized_segments(text[cursor:start], cap=sanitized_cap)
            )
        digest = source_grant_digest(grant)
        segments.append(SourceBoundSegment(digest))
        used_grants[digest] = grant
        cursor = end
    if cursor < len(text):
        segments.extend(_approved_sanitized_segments(text[cursor:], cap=sanitized_cap))
    return segments[0] if len(segments) == 1 else OutboundText(tuple(segments))


def _recognized_syntax_tool_call_ids(value: Any) -> frozenset[str]:
    """Bind output admission to a preceding call of an exact recognized tool."""

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
                and direct_name in _VALIDATED_SYNTAX_TOOL_NAMES
            ):
                call_id = item.get("call_id") or item.get("id")
                if isinstance(call_id, str):
                    recognized.add(call_id)
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, Mapping):
                        continue
                    function = call.get("function")
                    if (
                        isinstance(function, Mapping)
                        and function.get("name") in _VALIDATED_SYNTAX_TOOL_NAMES
                        and isinstance(call.get("id"), str)
                    ):
                        recognized.add(call["id"])
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return frozenset(recognized)


def _segment_protected_tool_result(
    text: str,
    grant_texts: Sequence[tuple[str, SourceGrant]],
    used_grants: dict[str, SourceGrant],
    *,
    sanitized_cap: int,
) -> UntrustedProvenanceSegment:
    """Never infer non-source authority from generic terminal stdout shape."""

    del grant_texts, used_grants, sanitized_cap
    return UntrustedProvenanceSegment(sha256(text.encode("utf-8")).hexdigest())


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
    protected_tool_content: bool = False,
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
        typed: dict[str, Any] = {}
        for key, item in value.items():
            if key == "_source_provenance":
                continue
            if is_read_file_result and key == "content" and isinstance(item, str):
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
            typed[key] = _typed_payload(
                item,
                grant_texts,
                used_grants,
                sanitized_cap=sanitized_cap,
                field_name=key,
                syntax_tool_call_ids=syntax_tool_call_ids,
                protected_tool_content=(
                    is_recognized_tool_result and key in {"content", "output"}
                ),
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
                protected_tool_content=protected_tool_content,
                registry=registry,
                request_identity=request_identity,
            )
            for item in value
        ]
    return value


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


def _restore_source_provenance_sidecar(
    body: Mapping[str, Any], sidecar: Any
) -> dict[str, Any]:
    """Reattach only exact content-bound metadata to internal message copies."""

    restored = dict(body)
    messages = restored.get("messages")
    if not isinstance(messages, list) or not isinstance(sidecar, list):
        return restored
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
    return restored


def authorize_agent_sdk_kwargs(
    agent: Any,
    kwargs: Mapping[str, Any],
    *,
    route: Any | None = None,
    sdk_control_keys: Sequence[str] = _SDK_CONTROL_KEYS,
) -> tuple[dict[str, Any], AuthorizedEgress]:
    controls = {key: kwargs[key] for key in sdk_control_keys if key in kwargs}
    sidecar = kwargs.get("_hermes_source_provenance")
    body = {
        key: value
        for key, value in kwargs.items()
        if key not in controls and key not in _INTERNAL_EGRESS_KEYS
    }
    if os.environ.get("HERMES_KANBAN_PROTECTED_REMOTE") == "1":
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
    typed_body = _typed_payload(
        body,
        _grant_texts(grants),
        used_grants,
        sanitized_cap=sanitized_segment_cap,
        syntax_tool_call_ids=(
            _recognized_syntax_tool_call_ids(body)
            if os.environ.get("HERMES_KANBAN_PROTECTED_REMOTE") == "1"
            else frozenset()
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
            _route_for_agent(agent, route),
            grants=tuple(used_grants.values()),
        )
    except EgressBlocked:
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
        str(getattr(resolved_route, "provider", "") or ""),
        getattr(resolved_route, "base_url", None),
        getattr(resolved_route, "api_mode", None),
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
