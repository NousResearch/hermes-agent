"""Shared runtime helpers for gateway transcript persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(slots=True)
class GatewayTranscriptPersistenceResult:
    """Summary of what the gateway transcript persistence path did."""

    agent_failed_early: bool
    wrote_session_meta: bool
    used_fallback_transcript: bool
    persisted_messages: int


def did_gateway_agent_fail_early(agent_result: dict[str, Any]) -> bool:
    """Return True when the agent failed before producing any visible reply."""

    return bool(agent_result.get("failed") and not agent_result.get("final_response"))


def build_gateway_session_meta_entry(
    *,
    tool_defs: list[dict[str, Any]] | None,
    model: str,
    platform: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build the self-describing transcript entry for fresh sessions."""

    return {
        "role": "session_meta",
        "tools": tool_defs or [],
        "model": model,
        "platform": platform,
        "timestamp": timestamp,
    }


def extract_new_gateway_transcript_messages(
    *,
    agent_result: dict[str, Any],
    agent_messages: list[dict[str, Any]] | None,
    history_len: int,
    visible_final_response: Any,
    sync_visible_final_response: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return only the new transcript messages produced by this turn."""

    resolved_messages = agent_messages or []
    history_offset = agent_result.get("history_offset", history_len)
    new_messages = (
        resolved_messages[history_offset:]
        if len(resolved_messages) > history_offset
        else []
    )
    return sync_visible_final_response(
        new_messages,
        raw_final_response=agent_result.get("final_response"),
        visible_final_response=visible_final_response,
    )


def persist_gateway_agent_transcript(
    *,
    session_store: Any,
    session_id: str,
    session_key: str,
    platform: str,
    history: list[dict[str, Any]],
    agent_result: dict[str, Any],
    agent_messages: list[dict[str, Any]] | None,
    message_text: str,
    visible_final_response: str,
    resolve_gateway_model: Callable[[], str],
    sync_visible_final_response: Callable[..., list[dict[str, Any]]],
    session_db_present: bool,
    logger: Any | None = None,
    timestamp: str | None = None,
) -> GatewayTranscriptPersistenceResult:
    """Persist one gateway turn into the transcript stores."""

    ts = timestamp or datetime.now().isoformat()
    agent_failed_early = did_gateway_agent_fail_early(agent_result)
    wrote_session_meta = False
    used_fallback_transcript = False
    persisted_messages = 0

    if agent_failed_early and logger is not None:
        logger.info(
            "Skipping transcript persistence for failed request in "
            "session %s to prevent session growth loop.",
            session_id,
        )

    if not agent_failed_early and not history:
        session_store.append_to_transcript(
            session_id,
            build_gateway_session_meta_entry(
                tool_defs=agent_result.get("tools", []),
                model=resolve_gateway_model(),
                platform=platform,
                timestamp=ts,
            ),
        )
        wrote_session_meta = True

    if not agent_failed_early:
        new_messages = extract_new_gateway_transcript_messages(
            agent_result=agent_result,
            agent_messages=agent_messages,
            history_len=len(history),
            visible_final_response=visible_final_response,
            sync_visible_final_response=sync_visible_final_response,
        )

        if not new_messages:
            session_store.append_to_transcript(
                session_id,
                {"role": "user", "content": message_text, "timestamp": ts},
            )
            persisted_messages += 1
            used_fallback_transcript = True
            if visible_final_response:
                session_store.append_to_transcript(
                    session_id,
                    {
                        "role": "assistant",
                        "content": visible_final_response,
                        "timestamp": ts,
                    },
                )
                persisted_messages += 1
        else:
            for msg in new_messages:
                if msg.get("role") == "system":
                    continue
                session_store.append_to_transcript(
                    session_id,
                    {**msg, "timestamp": ts},
                    skip_db=session_db_present,
                )
                persisted_messages += 1

    session_store.update_session(
        session_key,
        last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
    )

    return GatewayTranscriptPersistenceResult(
        agent_failed_early=agent_failed_early,
        wrote_session_meta=wrote_session_meta,
        used_fallback_transcript=used_fallback_transcript,
        persisted_messages=persisted_messages,
    )
