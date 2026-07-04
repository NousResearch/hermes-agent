"""Hermes transcript to OpenViking session-message conversion."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.message_content import flatten_message_text

from .schemas import (
    _OPENVIKING_RECALL_TOOL_NAMES,
    _TOOL_STATUS_COMPLETED,
    _TOOL_STATUS_COMPLETED_ALIASES,
    _TOOL_STATUS_ERROR,
    _TOOL_STATUS_ERROR_ALIASES,
    _TOOL_STATUS_PENDING,
)

class OpenVikingTranscriptMixin:
    """Transcript conversion helpers used by OpenVikingMemoryProvider."""

    @staticmethod
    def _message_text(content: Any) -> str:
        """Extract text from OpenAI-style string/list content."""
        return flatten_message_text(content)

    @classmethod
    def _message_matches_text(cls, message: Dict[str, Any], expected: Any) -> bool:
        expected_text = cls._message_text(expected).strip()
        if not expected_text:
            return False
        actual_text = cls._message_text(message.get("content")).strip()
        return actual_text == expected_text

    @classmethod
    def _extract_current_turn_messages(
        cls,
        messages: Optional[List[Dict[str, Any]]],
        user_content: str,
        assistant_content: str,
    ) -> List[Dict[str, Any]]:
        """Slice the completed turn out of Hermes' full canonical transcript."""
        if not messages:
            return []

        end_idx: Optional[int] = None
        if cls._message_text(assistant_content).strip():
            for idx in range(len(messages) - 1, -1, -1):
                message = messages[idx]
                if (
                    isinstance(message, dict)
                    and message.get("role") == "assistant"
                    and cls._message_matches_text(message, assistant_content)
                ):
                    end_idx = idx
                    break
        if end_idx is None:
            for idx in range(len(messages) - 1, -1, -1):
                message = messages[idx]
                if isinstance(message, dict) and message.get("role") == "assistant":
                    end_idx = idx
                    break
        if end_idx is None:
            end_idx = len(messages) - 1

        start_idx: Optional[int] = None
        if cls._message_text(user_content).strip():
            for idx in range(end_idx, -1, -1):
                message = messages[idx]
                if (
                    isinstance(message, dict)
                    and message.get("role") == "user"
                    and cls._message_matches_text(message, user_content)
                ):
                    start_idx = idx
                    break
        if start_idx is None:
            for idx in range(end_idx, -1, -1):
                message = messages[idx]
                if isinstance(message, dict) and message.get("role") == "user":
                    start_idx = idx
                    break
        if start_idx is None:
            return []

        return [message for message in messages[start_idx : end_idx + 1] if isinstance(message, dict)]

    @staticmethod
    def _tool_call_id(tool_call: Dict[str, Any]) -> str:
        return str(tool_call.get("id") or tool_call.get("tool_call_id") or "")

    @staticmethod
    def _tool_call_name(tool_call: Dict[str, Any]) -> str:
        function = tool_call.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return str(tool_call.get("name") or "")

    @staticmethod
    def _is_openviking_recall_tool_name(tool_name: Any) -> bool:
        return str(tool_name or "").strip().lower() in _OPENVIKING_RECALL_TOOL_NAMES

    @staticmethod
    def _tool_call_input(tool_call: Dict[str, Any]) -> Dict[str, Any]:
        function = tool_call.get("function")
        raw_args: Any = None
        if isinstance(function, dict):
            raw_args = function.get("arguments")
        if raw_args is None:
            raw_args = tool_call.get("args")
        if raw_args is None:
            return {}
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            if not raw_args.strip():
                return {}
            try:
                parsed = json.loads(raw_args)
            except Exception:
                return {"value": raw_args}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        return {"value": raw_args}

    @classmethod
    def _tool_result_status(cls, message: Dict[str, Any]) -> str:
        raw_status = str(message.get("status") or message.get("tool_status") or "").lower()
        if raw_status in _TOOL_STATUS_ERROR_ALIASES:
            return _TOOL_STATUS_ERROR
        if raw_status in _TOOL_STATUS_COMPLETED_ALIASES:
            return _TOOL_STATUS_COMPLETED

        text = cls._message_text(message.get("content")).strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                status = str(parsed.get("status") or "").lower()
                exit_code = parsed.get("exit_code")
                if (
                    status in _TOOL_STATUS_ERROR_ALIASES
                    or parsed.get("success") is False
                    or bool(parsed.get("error"))
                    or (isinstance(exit_code, int) and exit_code != 0)
                ):
                    return _TOOL_STATUS_ERROR

        return _TOOL_STATUS_COMPLETED

    @classmethod
    def _messages_to_openviking_batch(
        cls,
        messages: List[Dict[str, Any]],
        *,
        assistant_peer_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Convert Hermes canonical messages into OpenViking batch payloads."""
        assistant_peer_id = str(assistant_peer_id or "").strip()
        tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
        completed_tool_ids: set[str] = set()
        skipped_tool_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "tool":
                tool_id = str(message.get("tool_call_id") or message.get("id") or "")
                if tool_id:
                    completed_tool_ids.add(tool_id)
                    if cls._is_openviking_recall_tool_name(message.get("name")):
                        skipped_tool_ids.add(tool_id)
                continue
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                tool_id = cls._tool_call_id(tool_call)
                tool_name = cls._tool_call_name(tool_call)
                if tool_id:
                    tool_calls_by_id[tool_id] = {
                        "tool_name": tool_name,
                        "tool_input": cls._tool_call_input(tool_call),
                    }
                    if cls._is_openviking_recall_tool_name(tool_name):
                        skipped_tool_ids.add(tool_id)

        payload_messages: List[Dict[str, Any]] = []
        pending_tool_parts: List[Dict[str, Any]] = []

        def payload_message(role: str, parts: List[Dict[str, Any]]) -> Dict[str, Any]:
            payload: Dict[str, Any] = {"role": role, "parts": parts}
            if role == "assistant" and assistant_peer_id:
                payload["peer_id"] = assistant_peer_id
            return payload

        def flush_tool_parts() -> None:
            nonlocal pending_tool_parts
            if pending_tool_parts:
                payload_messages.append(payload_message("assistant", pending_tool_parts))
                pending_tool_parts = []

        for message in messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "")
            if role in {"system", "developer"}:
                continue

            if role == "tool":
                tool_id = str(message.get("tool_call_id") or message.get("id") or "")
                prior_call = tool_calls_by_id.get(tool_id, {})
                tool_name = str(message.get("name") or prior_call.get("tool_name") or "")
                if tool_id in skipped_tool_ids or cls._is_openviking_recall_tool_name(tool_name):
                    continue
                tool_part = {
                    "type": "tool",
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "tool_input": prior_call.get("tool_input", {}),
                    "tool_output": cls._message_text(message.get("content")),
                    "tool_status": cls._tool_result_status(message),
                }
                pending_tool_parts.append(tool_part)
                continue

            if role not in {"user", "assistant"}:
                continue

            flush_tool_parts()
            parts: List[Dict[str, Any]] = []
            text = cls._message_text(message.get("content"))
            if text:
                parts.append({"type": "text", "text": text})

            if role == "assistant":
                for tool_call in message.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    tool_id = cls._tool_call_id(tool_call)
                    tool_name = cls._tool_call_name(tool_call)
                    if tool_id in skipped_tool_ids or cls._is_openviking_recall_tool_name(tool_name):
                        continue
                    if tool_id in completed_tool_ids:
                        continue
                    # Reuse the tool_input parsed in the pre-scan when available
                    # (non-empty ids are cached); fall back to parsing for the
                    # uncached empty-id case so we never drop arguments.
                    prior_call = tool_calls_by_id.get(tool_id) if tool_id else None
                    tool_input = (
                        prior_call["tool_input"]
                        if prior_call is not None
                        else cls._tool_call_input(tool_call)
                    )
                    parts.append({
                        "type": "tool",
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "tool_input": tool_input,
                        "tool_status": _TOOL_STATUS_PENDING,
                    })

            if parts:
                payload_messages.append(payload_message(role, parts))

        flush_tool_parts()
        return payload_messages
