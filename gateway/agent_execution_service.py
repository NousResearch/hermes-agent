"""Shared gateway agent execution helpers."""

from __future__ import annotations

from contextlib import contextmanager
import logging
import re
from typing import Any, Iterator

from gateway.agent_runtime import GatewayAgentRuntimeSpec


logger = logging.getLogger(__name__)


def create_gateway_agent(
    *,
    runtime_spec: GatewayAgentRuntimeSpec,
    session_id: str,
    source: Any,
    session_db: Any = None,
    prefill_messages: list[dict[str, Any]] | None = None,
    max_iterations: int | None = None,
    enabled_toolsets: list[str] | None = None,
    quiet_mode: bool = True,
    verbose_logging: bool = False,
    skip_memory: bool = False,
    skip_context_files: bool = False,
    persist_session: bool | None = None,
) -> Any:
    """Construct an AIAgent from a shared runtime spec."""
    from run_agent import AIAgent

    agent_kwargs = {
        "model": runtime_spec.turn_route["model"],
        **runtime_spec.turn_route["runtime"],
        "max_iterations": int(max_iterations or runtime_spec.max_iterations),
        "quiet_mode": quiet_mode,
        "verbose_logging": verbose_logging,
        "enabled_toolsets": list(enabled_toolsets or runtime_spec.enabled_toolsets),
        "ephemeral_system_prompt": runtime_spec.combined_ephemeral,
        "reasoning_config": runtime_spec.reasoning_config,
        "providers_allowed": runtime_spec.provider_routing.get("only"),
        "providers_ignored": runtime_spec.provider_routing.get("ignore"),
        "providers_order": runtime_spec.provider_routing.get("order"),
        "provider_sort": runtime_spec.provider_routing.get("sort"),
        "provider_require_parameters": runtime_spec.provider_routing.get("require_parameters", False),
        "provider_data_collection": runtime_spec.provider_routing.get("data_collection"),
        "session_id": session_id,
        "platform": getattr(getattr(source, "platform", None), "value", "unknown") if getattr(source, "platform", None) else "unknown",
        "user_id": getattr(source, "user_id", None),
        "fallback_model": runtime_spec.fallback_model,
    }
    if session_db is not None:
        agent_kwargs["session_db"] = session_db
    if prefill_messages is not None:
        agent_kwargs["prefill_messages"] = prefill_messages
    if skip_memory:
        agent_kwargs["skip_memory"] = True
    if skip_context_files:
        agent_kwargs["skip_context_files"] = True
    if persist_session is not None:
        agent_kwargs["persist_session"] = persist_session
    return AIAgent(**agent_kwargs)


def normalize_conversation_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert stored gateway transcript items into agent-ready messages."""
    agent_history: list[dict[str, Any]] = []
    for msg in list(history or []):
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue

        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"
        if has_tool_calls or has_tool_call_id or is_tool_message:
            clean_msg = {k: v for k, v in msg.items() if k != "timestamp"}
            agent_history.append(clean_msg)
            continue

        content = msg.get("content")
        if not content:
            continue
        if msg.get("mirror"):
            mirror_src = msg.get("mirror_source", "another session")
            content = f"[Delivered from {mirror_src}] {content}"
        entry = {"role": role, "content": content}
        if role == "assistant":
            for reasoning_key in ("reasoning", "reasoning_details", "codex_reasoning_items"):
                reasoning_value = msg.get(reasoning_key)
                if reasoning_value:
                    entry[reasoning_key] = reasoning_value
        agent_history.append(entry)
    return agent_history


def collect_history_media_paths(agent_history: list[dict[str, Any]]) -> set[str]:
    """Collect already-seen MEDIA: tags from prior tool messages."""
    history_media_paths: set[str] = set()
    for msg in agent_history:
        if msg.get("role") not in {"tool", "function"}:
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" not in content:
            continue
        for match in re.finditer(r"MEDIA:(\S+)", content):
            path = match.group(1).strip().rstrip('",}')
            if path:
                history_media_paths.add(path)
    return history_media_paths


@contextmanager
def gateway_approval_context(
    *,
    session_key: str,
    admin_user_ids: list[str] | None,
    is_admin_user: bool | None,
    external_backend: Any = None,
) -> Iterator[None]:
    """Apply approval/session context for one foreground/background execution."""
    from tools.approval import (
        reset_current_admin_policy,
        reset_current_session_key,
        reset_external_approval_backend,
        set_current_admin_policy,
        set_current_session_key,
        set_external_approval_backend,
    )

    session_token = set_current_session_key(str(session_key or ""))
    admin_tokens = set_current_admin_policy(
        list(admin_user_ids or []),
        is_admin_user,
    )
    backend_token = None
    if external_backend is not None:
        backend_token = set_external_approval_backend(external_backend)
    try:
        yield
    finally:
        if backend_token is not None:
            reset_external_approval_backend(backend_token)
        reset_current_admin_policy(admin_tokens)
        reset_current_session_key(session_token)
