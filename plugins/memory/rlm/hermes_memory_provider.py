from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import request

try:
    from agent.memory_provider import (
        MemoryProvider,
        PRE_COMPRESS_CHECKPOINT_API_VERSION,
        RecallStatus,
    )  # type: ignore[import-not-found]
except ImportError:  # RLM tests run without the Hermes source tree on sys.path.
    PRE_COMPRESS_CHECKPOINT_API_VERSION = 2

    class MemoryProvider:  # type: ignore[no-redef]
        pass

    class RecallStatus:  # type: ignore[no-redef]
        def __init__(self, provider_label: str, count: int, glyph: str = "🧠") -> None:
            self.provider_label, self.count, self.glyph = provider_label, count, glyph


_MAX_CONTEXT_CHARS = 8_000


def _post(base_url: str, path: str, payload: dict[str, Any], timeout: float = 8.0) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def rlm_orient(
    objective: str,
    *,
    project: str,
    context_summary: str,
    include_retrieval: bool,
    limit: int,
    audience_profile: str,
    audience_capability: str,
    gateway_url: str = "http://127.0.0.1:8100",
) -> Any:
    return _post(
        gateway_url,
        "/tools/orient",
        {
            "objective": objective,
            "project": project,
            "context_summary": context_summary,
            "include_retrieval": include_retrieval,
            "limit": limit,
            "audience_profile": audience_profile,
            "audience_capability": audience_capability,
        },
    )


def rlm_recall_experience(
    query: str,
    *,
    project: str,
    limit: int,
    include_unverified: bool,
    gateway_url: str = "http://127.0.0.1:8100",
) -> Any:
    return _post(
        gateway_url,
        "/tools/experience_memory.recall",
        {
            "query": query,
            "project": project,
            "limit": limit,
            "include_unverified": include_unverified,
        },
    )


def rlm_record_hermes_turn(
    *,
    session_id: str,
    user_content: str,
    assistant_content: str,
    agent_identity: str,
    agent_workspace: str,
    platform: str,
    messages: list[dict[str, Any]],
    turn_author: dict[str, Any] | None,
    gateway_url: str = "http://127.0.0.1:8100",
) -> Any:
    return _post(
        gateway_url,
        "/memory/hermes/turn",
        {
            "session_id": session_id,
            "user_content": user_content,
            "assistant_content": assistant_content,
            "agent_identity": agent_identity,
            "agent_workspace": agent_workspace,
            "platform": platform,
            "messages": messages,
            "turn_author": turn_author or {},
        },
    )


def _selected_orientation_rows(packet: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    authority = str(packet.get("authority") or "")
    if authority:
        lines.append(f"Authority: {authority}")
    continuity = packet.get("continuity")
    if isinstance(continuity, dict) and continuity.get("verdict"):
        lines.append(
            f"Continuity: {continuity.get('verdict')}"
            + (f" — {continuity.get('reason')}" if continuity.get("reason") else "")
        )
    evidence = packet.get("evidence")
    selected = evidence.get("selected", []) if isinstance(evidence, dict) else []
    for row in selected[:6] if isinstance(selected, list) else []:
        if not isinstance(row, dict):
            continue
        text = row.get("summary") or row.get("claim") or row.get("content")
        if text:
            lines.append(f"- {str(text)[:800]}")
    if packet.get("next_action"):
        lines.append(f"Next action: {packet['next_action']}")
    return lines


def _experience_rows(packet: dict[str, Any]) -> list[str]:
    hits = packet.get("hits", []) if isinstance(packet, dict) else []
    lines: list[str] = []
    for hit in hits[:3] if isinstance(hits, list) else []:
        if not isinstance(hit, dict):
            continue
        summary = hit.get("summary") or ""
        objective = hit.get("objective") or ""
        if summary or objective:
            lines.append(
                f"- [{hit.get('outcome', 'unknown')}; id={hit.get('episode_id', '')}] "
                f"{str(objective)[:240]} | {str(summary)[:700]}"
            )
    return lines


class HermesRLMMemoryProvider(MemoryProvider):
    """Hermes adapter for the existing RLM/GBrain/Librarian memory plane."""

    pre_compress_checkpoint_api_version = PRE_COMPRESS_CHECKPOINT_API_VERSION

    def __init__(self, gateway_url: str | None = None) -> None:
        self.gateway_url = (gateway_url or os.getenv("RLM_GATEWAY_URL") or "http://127.0.0.1:8100").rstrip("/")
        self._call_gateway_url = self.gateway_url != "http://127.0.0.1:8100"
        self._session_id = ""
        self._agent_context = "primary"
        self._agent_identity = "default"
        self._agent_workspace = "hermes"
        self._platform = "cli"
        self._last_recall_count = 0

    @property
    def name(self) -> str:
        return "rlm"

    def is_available(self) -> bool:
        try:
            with request.urlopen(self.gateway_url + "/health", timeout=2.0) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def unavailable_reason(self) -> str:
        return f"RLM gateway is not reachable at {self.gateway_url}"

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "default")
        self._agent_workspace = str(kwargs.get("agent_workspace") or "hermes")
        self._platform = str(kwargs.get("platform") or "cli")

    def system_prompt_block(self) -> str:
        return (
            "# Cohesive Memory\n"
            "RLM is the retrieval and episodic memory plane. GBrain remains curated durable truth; "
            "live files and services remain authoritative over recalled context."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall_count = 0
        if not query.strip():
            return ""
        sections: list[str] = []
        try:
            orient_kwargs: dict[str, Any] = {
                "project": "agent",
                "context_summary": "",
                "include_retrieval": True,
                "limit": 6,
                "audience_profile": "worker",
                "audience_capability": "",
            }
            if self._call_gateway_url:
                orient_kwargs["gateway_url"] = self.gateway_url
            orientation = rlm_orient(query, **orient_kwargs)
            orientation_lines = _selected_orientation_rows(orientation if isinstance(orientation, dict) else {})
            if orientation_lines:
                sections.append("[RLM Fused Orientation]\n" + "\n".join(orientation_lines))
                self._last_recall_count += 1
        except Exception:
            pass
        try:
            recall_kwargs: dict[str, Any] = {
                "project": "agent",
                "limit": 3,
                "include_unverified": False,
            }
            if self._call_gateway_url:
                recall_kwargs["gateway_url"] = self.gateway_url
            experience = rlm_recall_experience(query, **recall_kwargs)
            experience_lines = _experience_rows(experience if isinstance(experience, dict) else {})
            if experience_lines:
                sections.append("[RLM Verified Experience]\n" + "\n".join(experience_lines))
                self._last_recall_count += 1
        except Exception:
            pass
        return "\n\n".join(sections)[:_MAX_CONTEXT_CHARS]

    def recall_status(self) -> RecallStatus | None:
        if not self._last_recall_count:
            return None
        return RecallStatus("RLM memory plane", self._last_recall_count)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
        turn_author: dict[str, Any] | None = None,
    ) -> None:
        if self._agent_context != "primary" or not user_content.strip():
            return
        turn_kwargs: dict[str, Any] = {
            "session_id": session_id or self._session_id,
            "user_content": user_content,
            "assistant_content": assistant_content,
            "agent_identity": self._agent_identity,
            "agent_workspace": self._agent_workspace,
            "platform": self._platform,
            "messages": list(messages or []),
            "turn_author": turn_author,
        }
        if self._call_gateway_url:
            turn_kwargs["gateway_url"] = self.gateway_url
        rlm_record_hermes_turn(**turn_kwargs)

    def on_pre_compress(
        self,
        messages: list[dict[str, Any]],
        *,
        require_checkpoint: bool = False,
    ) -> str:
        if self._agent_context != "primary":
            if require_checkpoint:
                raise RuntimeError("RLM checkpoint refused outside the primary agent context")
            return ""
        user_content = ""
        assistant_content = ""
        for message in reversed(messages):
            role = message.get("role") if isinstance(message, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if role == "assistant" and isinstance(content, str) and content and not assistant_content:
                assistant_content = content
            elif role == "user" and isinstance(content, str) and content and not user_content:
                user_content = content
            if user_content and assistant_content:
                break
        if not user_content:
            if require_checkpoint:
                raise RuntimeError("RLM checkpoint has no user evidence to persist")
            return ""
        turn_kwargs: dict[str, Any] = {
            "session_id": self._session_id,
            "user_content": user_content,
            "assistant_content": assistant_content,
            "agent_identity": self._agent_identity,
            "agent_workspace": self._agent_workspace,
            "platform": self._platform,
            "messages": list(messages),
            "turn_author": None,
        }
        if self._call_gateway_url:
            turn_kwargs["gateway_url"] = self.gateway_url
        receipt = rlm_record_hermes_turn(**turn_kwargs)
        if require_checkpoint and (not isinstance(receipt, dict) or receipt.get("stored") is not True):
            raise RuntimeError("RLM checkpoint did not confirm durable storage")
        return "RLM checkpoint stored: unverified continuity and temporal lineage were preserved."

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    def backup_paths(self) -> list[str]:
        root = Path.home() / "Library/Application Support/Eve/RLM/state"
        return [str(root)] if root.exists() else []


def register(ctx: Any) -> None:
    ctx.register_memory_provider(HermesRLMMemoryProvider())
