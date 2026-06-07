"""Build Cursor run requests from OpenAI-format messages.

This ports the request-shaping logic from the oh-my-pi Cursor provider into
Python, focused on the Task 3 surface needed to serialize Run requests and
preserve deterministic conversation history/blobs.
"""

from __future__ import annotations

import hashlib
import json
import uuid

from agent.cursor.constants import normalize_cursor_model_id
from agent.cursor.proto import agent_pb2


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
CURSOR_NATIVE_TOOL_NAMES = {"bash", "read", "write", "delete", "ls", "grep", "lsp", "todo"}


def _normalize_system_prompts(system_prompt: list[str] | str | None) -> list[str]:
    if system_prompt is None:
        return []
    if isinstance(system_prompt, str):
        system_prompt = [system_prompt]
    return [item.strip() for item in system_prompt if isinstance(item, str) and item.strip()]


def build_cursor_system_prompt_jsons(system_prompt: list[str] | str | None) -> list[str]:
    prompts = _normalize_system_prompts(system_prompt)
    if not prompts:
        return [json.dumps({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})]
    return [json.dumps({"role": "system", "content": prompt}) for prompt in prompts]


def create_blob_id(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def store_cursor_blob(blob_store: dict[str, bytes], data: bytes) -> bytes:
    blob_id = create_blob_id(data)
    blob_store[blob_id.hex()] = data
    return blob_id


def deterministic_message_id(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (
        f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-"
        f"{digest[16:20]}-{digest[20:32]}"
    )


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _build_root_prompt_content(content) -> list[dict]:
    if isinstance(content, str):
        text = content.strip()
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []

    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text", "").strip()
            if text:
                parts.append({"type": "text", "text": text})
        elif item.get("type") == "image_url":
            image = item.get("image_url")
            if isinstance(image, dict):
                url = image.get("url", "")
                media_type = image.get("mime_type") or "image/png"
            else:
                url = image or ""
                media_type = "image/png"
            if url:
                parts.append({"type": "image", "image": url, "mediaType": media_type})
    return parts


def _tool_result_to_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return ""


def _assistant_text(message: dict) -> str:
    return _extract_text_content(message.get("content", ""))


def _find_last_user_message_index(messages: list[dict]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") in {"user", "developer"}:
            return index
    return -1


def build_root_prompt_messages_json(
    messages: list[dict],
    system_prompt_ids: list[bytes],
    blob_store: dict[str, bytes],
    active_user_message_index: int | None = None,
) -> list[bytes]:
    if active_user_message_index is None:
        active_user_message_index = _find_last_user_message_index(messages)

    entries = list(system_prompt_ids)
    for index, message in enumerate(messages):
        if active_user_message_index >= 0 and index == active_user_message_index:
            break

        role = message.get("role")
        if role in {"user", "developer"}:
            content = _build_root_prompt_content(message.get("content", ""))
            if not content:
                continue
            payload = {"role": "user", "content": content}
        elif role == "assistant":
            text = _assistant_text(message)
            if not text:
                continue
            payload = {"role": "assistant", "content": [{"type": "text", "text": text}]}
        elif role in {"tool", "tool_result"}:
            text = _tool_result_to_text(message)
            if not text:
                continue
            payload = {
                "role": "user",
                "content": [{"type": "text", "text": f"[Tool Result]\n{text}"}],
            }
        else:
            continue

        entries.append(
            store_cursor_blob(blob_store, json.dumps(payload, sort_keys=True).encode("utf-8"))
        )
    return entries


def create_cursor_user_message(content, text: str, message_id: str | None = None):
    return agent_pb2.UserMessage(
        text=text,
        message_id=message_id or str(uuid.uuid4()),
    )


def _cursor_user_content_key(content) -> str:
    if isinstance(content, str):
        return content.strip()
    digest = hashlib.sha256()
    for item in content or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type", "")
        digest.update(str(item_type).encode("utf-8"))
        if item_type == "text":
            digest.update(item.get("text", "").encode("utf-8"))
        elif item_type == "image_url":
            image = item.get("image_url")
            if isinstance(image, dict):
                digest.update(str(image.get("mime_type", "")).encode("utf-8"))
                digest.update(str(image.get("url", "")).encode("utf-8"))
            else:
                digest.update(str(image or "").encode("utf-8"))
    return digest.hexdigest()


def build_conversation_turns(
    messages: list[dict],
    blob_store: dict[str, bytes],
    active_user_message_index: int | None = None,
) -> list[bytes]:
    if active_user_message_index is None:
        active_user_message_index = _find_last_user_message_index(messages)

    turns: list[bytes] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role not in {"user", "developer"}:
            index += 1
            continue
        if active_user_message_index >= 0 and index == active_user_message_index:
            break

        user_text = _extract_text_content(message.get("content", ""))
        if not user_text:
            index += 1
            continue

        turn_number = len(turns)
        user_message = create_cursor_user_message(
            message.get("content", ""),
            user_text,
            deterministic_message_id(
                f"u:{turn_number}:{_cursor_user_content_key(message.get('content', ''))}"
            ),
        )
        user_blob_id = store_cursor_blob(blob_store, user_message.SerializeToString())

        step_blob_ids: list[bytes] = []
        index += 1
        while index < len(messages) and messages[index].get("role") not in {"user", "developer"}:
            step_message = messages[index]
            if step_message.get("role") == "assistant":
                text = _assistant_text(step_message)
            elif step_message.get("role") in {"tool", "tool_result"}:
                result_text = _tool_result_to_text(step_message)
                text = f"[Tool Result]\n{result_text}" if result_text else ""
            else:
                text = ""

            if text:
                step = agent_pb2.ConversationStep(
                    assistant_message=agent_pb2.AssistantMessage(text=text)
                )
                step_blob_ids.append(store_cursor_blob(blob_store, step.SerializeToString()))
            index += 1

        turn = agent_pb2.ConversationTurnStructure(
            agent_conversation_turn=agent_pb2.AgentConversationTurnStructure(
                user_message=user_blob_id,
                steps=step_blob_ids,
            )
        )
        turns.append(store_cursor_blob(blob_store, turn.SerializeToString()))

    return turns


def build_mcp_tool_definitions(tools: list[dict] | None) -> list[dict]:
    if not tools:
        return []

    definitions = []
    for tool in tools:
        name = tool.get("name")
        if not name or name in CURSOR_NATIVE_TOOL_NAMES:
            continue
        schema = tool.get("parameters") or {"type": "object", "properties": {}, "required": []}
        definitions.append(
            {
                "name": name,
                "description": tool.get("description", ""),
                "providerIdentifier": "hermes-agent",
                "toolName": name,
                "inputSchema": schema,
            }
        )
    return definitions


def encode_tool_input_schema(schema: dict | None) -> bytes:
    """Serialize a JSON Schema dict as protobuf ``google.protobuf.Value`` bytes."""
    from google.protobuf import json_format
    from google.protobuf import struct_pb2

    value = struct_pb2.Value()
    json_format.ParseDict(
        schema or {"type": "object", "properties": {}, "required": []},
        value,
    )
    return value.SerializeToString()


def build_grpc_request(
    *,
    messages: list[dict],
    system_prompt: list[str] | str | None,
    tools: list[dict] | None,
    model_id: str,
    conversation_id: str,
    cached_conversation_state: agent_pb2.ConversationStateStructure | None = None,
    custom_system_prompt: str | None = None,
) -> tuple[bytes, agent_pb2.ConversationStateStructure, dict[str, bytes]]:
    model_id = normalize_cursor_model_id(model_id)
    blob_store: dict[str, bytes] = {}

    system_prompt_ids = [
        store_cursor_blob(blob_store, payload.encode("utf-8"))
        for payload in build_cursor_system_prompt_jsons(system_prompt)
    ]

    active_index = len(messages) - 1
    active_message = messages[active_index] if messages else None
    active_is_user = active_message is not None and active_message.get("role") in {"user", "developer"}

    if active_is_user:
        active_user_message = active_message
        user_text = _extract_text_content(active_user_message.get("content", ""))
        action = agent_pb2.ConversationAction(
            user_message_action=agent_pb2.UserMessageAction(
                user_message=create_cursor_user_message(active_user_message.get("content", ""), user_text)
            )
        )
        history_cutoff = active_index
    else:
        action = agent_pb2.ConversationAction(resume_action=agent_pb2.ResumeAction())
        history_cutoff = -1

    turns = build_conversation_turns(messages, blob_store, history_cutoff)
    root_prompt_messages_json = build_root_prompt_messages_json(
        messages, system_prompt_ids, blob_store, history_cutoff
    )

    if cached_conversation_state is not None:
        conversation_state = agent_pb2.ConversationStateStructure()
        conversation_state.CopyFrom(cached_conversation_state)
        del conversation_state.root_prompt_messages_json[:]
        conversation_state.root_prompt_messages_json.extend(root_prompt_messages_json)
        del conversation_state.turns[:]
        conversation_state.turns.extend(turns)
    else:
        conversation_state = agent_pb2.ConversationStateStructure(
            root_prompt_messages_json=root_prompt_messages_json,
            turns=turns,
            todos=[],
            pending_tool_calls=[],
            previous_workspace_uris=[],
            file_states={},
            file_states_v2={},
            summary_archives=[],
            turn_timings=[],
            subagent_states={},
            self_summary_count=0,
            read_paths=[],
        )

    run_request = agent_pb2.AgentRunRequest(
        conversation_state=conversation_state,
        action=action,
        model_details=agent_pb2.ModelDetails(
            model_id=model_id,
            display_model_id=model_id,
            display_name=model_id,
        ),
        conversation_id=conversation_id,
    )
    if custom_system_prompt:
        run_request.custom_system_prompt = custom_system_prompt
    client_message = agent_pb2.AgentClientMessage(run_request=run_request)
    return client_message.SerializeToString(), conversation_state, blob_store


def build_run_request_bytes(
    *,
    messages: list[dict],
    system_prompt: list[str] | str | None,
    tools: list[dict] | None,
    model_id: str,
    conversation_id: str,
    blob_store: dict[str, bytes] | None = None,
    conversation_state: agent_pb2.ConversationStateStructure | None = None,
    custom_system_prompt: str | None = None,
) -> tuple[bytes, agent_pb2.ConversationStateStructure, dict[str, bytes]]:
    """Compatibility wrapper for Cursor stream transport callers."""

    if blob_store is None:
        blob_store = {}

    request_bytes, next_state, built_blob_store = build_grpc_request(
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        model_id=model_id,
        conversation_id=conversation_id,
        cached_conversation_state=conversation_state,
        custom_system_prompt=custom_system_prompt,
    )

    if blob_store is not built_blob_store:
        blob_store.clear()
        blob_store.update(built_blob_store)
        built_blob_store = blob_store

    return request_bytes, next_state, built_blob_store

