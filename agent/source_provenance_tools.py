"""Trusted tool-boundary adapters for source provenance.

The large tool dispatch and file-tool modules call these narrow adapters but
do not own the security policy or grant-construction responsibilities.
"""

from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from agent.source_provenance import active_source_provenance


def source_provenance_activation(agent: Any, function_name: str):
    """Activate request identity only around the trusted ``read_file`` tool."""

    if function_name != "read_file":
        return nullcontext()
    try:
        from agent.source_provenance import (
            DEFAULT_POLICY_DIGEST,
            SourceProvenanceRegistry,
            activate_source_provenance,
            following_api_request_id,
            source_provenance_registry_for_agent,
        )

        session_id = str(getattr(agent, "session_id", "") or "")
        turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        request_id = following_api_request_id(
            str(getattr(agent, "_current_api_request_id", "") or ""), turn_id
        )
        policy_digest = str(
            getattr(agent, "_llm_egress_policy_digest", "")
            or getattr(agent, "llm_egress_policy_digest", "")
            or DEFAULT_POLICY_DIGEST
        )
        if not all((session_id, turn_id, request_id, policy_digest)):
            return nullcontext()
        registry = source_provenance_registry_for_agent(agent)
        if not isinstance(registry, SourceProvenanceRegistry):
            return nullcontext()
        return activate_source_provenance(
            registry,
            session_id=session_id,
            turn_id=turn_id,
            request_id=request_id,
            policy_digest=policy_digest,
        )
    except Exception:
        return nullcontext()


def attach_trusted_source_provenance_metadata(
    agent: Any,
    function_name: str,
    *,
    content: Any | None = None,
) -> dict[str, Any] | None:
    """Return an opaque, content-bound trusted-read envelope.

    The envelope is carried only in the internal tool message and consumed by
    the egress runtime.  It binds the exact presentation bytes to grants for
    the exact following provider request; it is never serialized to a provider.
    """

    if function_name != "read_file":
        return None
    try:
        from agent.llm_egress_firewall import source_grant_digest
        from agent.source_provenance import (
            SourceProvenanceRegistry,
            following_api_request_id,
        )

        registry = getattr(agent, "_source_provenance_registry", None)
        turn_id = str(getattr(agent, "_current_turn_id", "") or "")
        request_id = following_api_request_id(
            str(getattr(agent, "_current_api_request_id", "") or ""), turn_id
        )
        if not isinstance(registry, SourceProvenanceRegistry) or not request_id:
            return None
        digests = tuple(
            source_grant_digest(grant)
            for grant in registry.grants_for_request(request_id)
        )
        if not digests:
            return None
        metadata = getattr(agent, "_source_provenance_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            agent._source_provenance_metadata = metadata
        envelope: dict[str, Any] = {
            "request_id": request_id,
            "source_grant_digests": digests,
        }
        if isinstance(content, str):
            envelope["content_sha256"] = sha256(content.encode("utf-8")).hexdigest()
            envelope["presentation_kind"] = "read_file_json_v1"
        metadata[request_id] = envelope
        return envelope
    except Exception:
        return None


def build_source_provenance_sidecar(messages: Any) -> list[dict[str, Any]]:
    """Extract bounded content-free provenance before wire conversion strips it."""

    if not isinstance(messages, list):
        return []
    sidecar: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        metadata = message.get("_source_provenance")
        content = message.get("content")
        if (
            message.get("role") != "tool"
            or not isinstance(metadata, dict)
            or not isinstance(content, str)
            or metadata.get("content_sha256")
            != sha256(content.encode("utf-8")).hexdigest()
        ):
            continue
        sidecar.append(
            {
                **metadata,
                "message_index": index,
                "tool_call_id": message.get("tool_call_id"),
            }
        )
    return sidecar


def issue_active_read_provenance(
    *,
    resolved: Path | PurePosixPath,
    source_path: Path | None,
    offset: int,
    limit: int,
    returned_content: str,
    result_dict: dict[str, Any],
    file_ops: Any,
) -> None:
    """Issue provenance only for a verified local ``read_file`` result."""

    context = active_source_provenance()
    if context is None or not isinstance(resolved, Path) or source_path is None:
        return
    if result_dict.get("error") or result_dict.get("truncated_by") or not returned_content:
        return
    try:
        canonical = resolved.resolve(strict=True)
        if resolved.is_symlink() or not canonical.is_file():
            return
        selected: list[bytes] = []
        with canonical.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if line_number < offset:
                    continue
                if line_number >= offset + limit:
                    break
                selected.append(raw_line)
        if not selected:
            return
        raw = b"".join(selected)
        decoded = raw.decode("utf-8")
        from tools.tool_output_limits import get_max_line_length

        if any(len(line) > get_max_line_length() for line in decoded.split("\n")):
            return
        if returned_content != file_ops._add_line_numbers(decoded, offset):
            return
        context.registry.issue_file_slice(
            path=source_path,
            line_start=offset,
            line_end=offset + len(selected) - 1,
            content=raw,
            session_id=context.session_id,
            turn_id=context.turn_id,
            request_id=context.request_id,
            policy_digest=context.policy_digest,
        )
    except Exception:
        return


__all__ = [
    "attach_trusted_source_provenance_metadata",
    "build_source_provenance_sidecar",
    "issue_active_read_provenance",
    "source_provenance_activation",
]
