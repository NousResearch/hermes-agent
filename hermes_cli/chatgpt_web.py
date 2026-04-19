"""Helpers for ChatGPT.com web-model access and streaming conversation transport."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import random
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Optional

import httpx

DEFAULT_CHATGPT_WEB_BASE_URL = "https://chatgpt.com/backend-api/f"
DEFAULT_CHATGPT_WEB_MODELS = [
    "gpt-5-thinking",
    "gpt-5-instant",
    "gpt-5",
    "gpt-4o",
    "gpt-4.1",
    "o3",
    "o4-mini",
]
DEFAULT_CHATGPT_WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
_TOOL_RESPONSE_CONTINUATION_HINT = (
    "[Hermes continuation hint: if the main task is not complete yet, your next assistant "
    "message should usually be EXACTLY ONE <tool_call>...</tool_call> block for the single "
    "best next tool call. Use the tool schemas plus this tool result to guess the next call. "
    "Do not reply with progress narration like 'I will continue'.]"
)


def _default_user_agent() -> str:
    return os.getenv("CHATGPT_WEB_USER_AGENT", "").strip() or DEFAULT_CHATGPT_WEB_USER_AGENT


def _default_device_id() -> str:
    return os.getenv("CHATGPT_WEB_DEVICE_ID", "").strip() or str(uuid.uuid4())


def _chatgpt_web_debug_base() -> str:
    return os.getenv("CHATGPT_WEB_DEBUG_BASE", "").strip()


def _parse_cookie_header(raw_cookie_header: str) -> "OrderedDict[str, str]":
    cookies: "OrderedDict[str, str]" = OrderedDict()
    for part in str(raw_cookie_header or "").split(";"):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        name = name.strip()
        if name:
            cookies[name] = value.strip()
    return cookies


def _normalize_browser_cookies(browser_cookies: Any) -> list[dict[str, Any]]:
    parsed = browser_cookies
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        parsed = [{"name": str(name), "value": value} for name, value in parsed.items()]
    if not isinstance(parsed, list):
        return []

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if not name or not value:
            continue
        domain = str(item.get("domain") or "").strip()
        path = str(item.get("path") or "").strip()
        key = (name, domain, path)
        if key in seen:
            continue
        seen.add(key)
        normalized_item = dict(item)
        normalized_item["name"] = name
        normalized_item["value"] = value
        normalized.append(normalized_item)
    return normalized


def _extract_cookie_value(raw_cookie_header: str, cookie_name: str) -> str:
    return _parse_cookie_header(raw_cookie_header).get(str(cookie_name or "").strip(), "")


def _build_cookie_header(
    *,
    session_token: str = "",
    device_id: str = "",
    cookie_header: str = "",
    browser_cookies: Any = None,
) -> str:
    cookies = _parse_cookie_header(cookie_header)
    for item in _normalize_browser_cookies(browser_cookies):
        name = str(item.get("name") or "").strip()
        value = str(item.get("value") or "").strip()
        if name and value:
            cookies[name] = value
    if session_token:
        cookies["__Secure-next-auth.session-token"] = session_token
    if device_id:
        cookies["oai-did"] = device_id
    return "; ".join(f"{name}={value}" for name, value in cookies.items())


def _build_chatgpt_web_headers(
    *,
    access_token: str,
    session_token: str = "",
    cookie_header: str = "",
    browser_cookies: Any = None,
    user_agent: str = "",
    device_id: str = "",
    accept: str = "application/json",
) -> dict[str, str]:
    resolved_device_id = (
        device_id
        or _extract_cookie_value(cookie_header, "oai-did")
        or _default_device_id()
    )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": accept,
        "User-Agent": user_agent or _default_user_agent(),
        "Content-Type": "application/json",
        "Oai-Device-Id": resolved_device_id,
        "Referer": "https://chatgpt.com/",
        "Origin": "https://chatgpt.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }
    resolved_cookie_header = _build_cookie_header(
        session_token=session_token,
        device_id=headers["Oai-Device-Id"],
        cookie_header=cookie_header,
        browser_cookies=browser_cookies,
    )
    if resolved_cookie_header:
        headers["Cookie"] = resolved_cookie_header
    return headers


def _fetch_chatgpt_web_access_token_from_session(
    session_token: str,
    *,
    user_agent: str = "",
    device_id: str = "",
    timeout: float = 15.0,
) -> str:
    session_token = (session_token or "").strip()
    if not session_token:
        raise ValueError("ChatGPT web session token is required")

    did = device_id or _default_device_id()
    headers = {
        "Accept": "application/json",
        "User-Agent": user_agent or _default_user_agent(),
        "Oai-Device-Id": did,
        "Cookie": _build_cookie_header(session_token=session_token, device_id=did),
    }
    response = httpx.get(
        "https://chatgpt.com/api/auth/session",
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    payload = response.json()
    access_token = str(payload.get("accessToken") or "").strip()
    if not access_token:
        raise ValueError("ChatGPT session exchange did not return an accessToken")
    return access_token


def resolve_chatgpt_web_runtime_credentials(*, force_refresh: bool = False) -> dict[str, Any]:
    del force_refresh  # reserved for future provider-specific refresh logic

    access_token = os.getenv("CHATGPT_WEB_ACCESS_TOKEN", "").strip()
    session_token = os.getenv("CHATGPT_WEB_SESSION_TOKEN", "").strip()
    if access_token:
        return {
            "provider": "chatgpt-web",
            "api_key": access_token,
            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
            "source": "access-token",
            "session_token": session_token,
        }
    if session_token:
        return {
            "provider": "chatgpt-web",
            "api_key": _fetch_chatgpt_web_access_token_from_session(session_token),
            "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
            "source": "session-token",
            "session_token": session_token,
        }

    try:
        from agent.credential_pool import load_pool
        from hermes_cli.auth import _codex_access_token_is_expiring

        pool = load_pool("chatgpt-web")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                pool_api_key = str(getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "") or "").strip()
                pool_session_token = str(getattr(entry, "session_token", "") or "").strip()
                if pool_session_token and (not pool_api_key or _codex_access_token_is_expiring(pool_api_key, 0)):
                    pool_api_key = _fetch_chatgpt_web_access_token_from_session(pool_session_token)
                if pool_api_key:
                    return {
                        "provider": "chatgpt-web",
                        "api_key": pool_api_key,
                        "base_url": (getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", "") or DEFAULT_CHATGPT_WEB_BASE_URL).rstrip("/"),
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "session_token": pool_session_token,
                    }
    except Exception:
        pass

    from hermes_cli.auth import resolve_codex_runtime_credentials

    creds = resolve_codex_runtime_credentials(force_refresh=False, refresh_if_expiring=True)
    return {
        "provider": "chatgpt-web",
        "api_key": str(creds.get("api_key") or "").strip(),
        "base_url": DEFAULT_CHATGPT_WEB_BASE_URL,
        "source": "codex-oauth",
        "session_token": "",
    }


def fetch_chatgpt_web_model_ids(
    access_token: Optional[str] = None,
    *,
    session_token: str = "",
    user_agent: str = "",
    device_id: str = "",
    timeout: float = 15.0,
) -> list[str]:
    token = (access_token or "").strip()
    resolved_session = (session_token or os.getenv("CHATGPT_WEB_SESSION_TOKEN", "")).strip()
    if not token:
        token = resolve_chatgpt_web_runtime_credentials().get("api_key", "")
    if not token:
        return list(DEFAULT_CHATGPT_WEB_MODELS)

    headers = _build_chatgpt_web_headers(
        access_token=token,
        session_token=resolved_session,
        user_agent=user_agent,
        device_id=device_id,
    )
    try:
        response = httpx.get(
            "https://chatgpt.com/backend-api/models",
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return list(DEFAULT_CHATGPT_WEB_MODELS)

    raw_models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(raw_models, list):
        return list(DEFAULT_CHATGPT_WEB_MODELS)

    model_ids: list[str] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or item.get("id") or item.get("name") or item.get("model") or "").strip()
        if not slug:
            continue
        visibility = str(item.get("visibility") or "").strip().lower()
        if visibility in {"hidden", "hide"}:
            continue
        if slug not in model_ids:
            model_ids.append(slug)
    return model_ids or list(DEFAULT_CHATGPT_WEB_MODELS)


def _generate_proof_token(seed: str, difficulty: str, user_agent: str) -> str:
    prefix = "gAAAAAB"
    now_utc = datetime.now(timezone.utc)
    config = [
        random.randint(1000, 3000),
        now_utc.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        None,
        0,
        user_agent,
    ]
    diff_len = len(difficulty or "")
    for attempt in range(100000):
        config[3] = attempt
        answer = base64.b64encode(json.dumps(config).encode()).decode()
        candidate = hashlib.sha3_512((seed + answer).encode()).hexdigest()
        if candidate[:diff_len] <= difficulty:
            return prefix + answer
    fallback_base = base64.b64encode(seed.encode()).decode()
    return prefix + "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + fallback_base


def _prepare_conversation(
    client: httpx.Client,
    *,
    headers: dict[str, str],
    model: str,
    conversation_id: Optional[str],
    parent_message_id: str,
    history_and_training_disabled: bool,
) -> str:
    payload: dict[str, Any] = {
        "action": "next",
        "parent_message_id": parent_message_id,
        "model": model,
        "conversation_mode": {"kind": "primary_assistant"},
        "supports_buffering": True,
        "supported_encodings": ["v1"],
        "system_hints": [],
    }
    if history_and_training_disabled:
        payload["history_and_training_disabled"] = True
    if conversation_id:
        payload["conversation_id"] = conversation_id

    response = client.post(
        "https://chatgpt.com/backend-api/f/conversation/prepare",
        headers=headers,
        json=payload,
    )
    if response.status_code >= 400:
        return ""
    data = response.json()
    return str(data.get("conduit_token") or "").strip()


def _chat_requirements(
    client: httpx.Client,
    *,
    headers: dict[str, str],
) -> dict[str, Any]:
    response = client.post(
        "https://chatgpt.com/backend-api/sentinel/chat-requirements",
        headers=headers,
        json={},
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _format_initial_message(
    *,
    instructions: str,
    messages: list[dict[str, Any]],
    has_remote_thread: bool,
) -> str:
    latest_user = ""
    transcript_lines: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        raw_content = item.get("content")
        content = str(raw_content or "")
        rendered = ""

        if role == "user":
            latest_user = content
            rendered = content.strip()
        elif role == "assistant":
            rendered_parts: list[str] = []
            if content.strip():
                rendered_parts.append(content.strip())
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                    name = str(function.get("name") or "").strip()
                    if not name:
                        continue
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except Exception:
                            pass
                    rendered_parts.append(
                        "<tool_call>\n"
                        + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
                        + "\n</tool_call>"
                    )
            rendered = "\n".join(part for part in rendered_parts if part).strip()
        elif role == "tool":
            if content.strip():
                rendered = (
                    "<tool_response>\n"
                    f"{content.strip()}\n"
                    f"{_TOOL_RESPONSE_CONTINUATION_HINT}\n"
                    "</tool_response>"
                )

        if rendered:
            transcript_lines.append(f"{role.title()}:\n{rendered}")

    if has_remote_thread:
        prompt_parts: list[str] = []
        if instructions.strip():
            prompt_parts.append(
                f"Developer instructions (higher priority than the conversation below):\n{instructions.strip()}"
            )
        if latest_user.strip():
            prompt_parts.append(f"Latest user request:\n{latest_user.strip()}")
        return "\n\n".join(part for part in prompt_parts if part).strip()

    prompt_parts: list[str] = []
    if instructions.strip():
        prompt_parts.append(f"Developer instructions (higher priority than the conversation below):\n{instructions.strip()}")
    if transcript_lines:
        prompt_parts.append("Conversation so far:\n" + "\n".join(transcript_lines))
    return "\n\n".join(part for part in prompt_parts if part).strip()


def _extract_event_message(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    message = event.get("message")
    if isinstance(message, dict):
        return message
    nested = event.get("v")
    if isinstance(nested, dict):
        message = nested.get("message")
        if isinstance(message, dict):
            return message
    return None


def _extract_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") if isinstance(message.get("content"), dict) else {}
    if content.get("content_type") != "text":
        return ""
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return ""
    return str(parts[0] or "")


def _extract_message_metadata(message: dict[str, Any]) -> dict[str, Any]:
    metadata = message.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _strip_asset_pointer_prefix(asset_pointer: str) -> str:
    pointer = str(asset_pointer or "").strip()
    if pointer.startswith("sediment://"):
        return pointer[len("sediment://"):]
    if pointer.startswith("file-service://"):
        return pointer[len("file-service://"):]
    return pointer


def _extract_message_image_assets(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content") if isinstance(message.get("content"), dict) else {}
    if content.get("content_type") != "multimodal_text":
        return []
    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        return []

    message_id = str(message.get("id") or "").strip()
    message_metadata = _extract_message_metadata(message)
    assets: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if str(part.get("content_type") or "").strip().lower() != "image_asset_pointer":
            continue
        asset_pointer = str(part.get("asset_pointer") or "").strip()
        file_id = _strip_asset_pointer_prefix(asset_pointer)
        if not file_id:
            continue
        part_metadata = part.get("metadata") if isinstance(part.get("metadata"), dict) else {}
        assets.append({
            "message_id": message_id,
            "asset_pointer": asset_pointer,
            "file_id": file_id,
            "width": part.get("width"),
            "height": part.get("height"),
            "size_bytes": part.get("size_bytes"),
            "metadata": part_metadata,
            "async_task_id": message_metadata.get("async_task_id"),
        })
    return assets


def _looks_like_image_generation_spec(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    try:
        payload = json.loads(candidate)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    prompt = payload.get("prompt")
    return isinstance(prompt, str) and bool(prompt.strip()) and any(key in payload for key in ("size", "n", "transparent_background"))


def _message_suggests_image_generation(message: dict[str, Any]) -> bool:
    metadata = _extract_message_metadata(message)
    if any(key in metadata for key in ("image_gen_task_id", "async_task_id", "image_gen_multi_stream", "image_gen_async")):
        return True
    if _extract_message_image_assets(message):
        return True
    author = message.get("author") if isinstance(message.get("author"), dict) else {}
    role = str(author.get("role") or "").strip().lower()
    return role == "tool" and bool(str(author.get("name") or "").strip()) and "processing image" in _extract_message_text(message).lower()


def _fetch_chatgpt_web_conversation(
    client: httpx.Client,
    *,
    headers: dict[str, str],
    conversation_id: str,
) -> dict[str, Any]:
    response = client.get(
        f"https://chatgpt.com/backend-api/conversation/{conversation_id}",
        headers=headers,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _conversation_node_order(
    conversation_payload: dict[str, Any],
    preferred_message_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    mapping = conversation_payload.get("mapping") if isinstance(conversation_payload.get("mapping"), dict) else {}
    if not mapping:
        return []

    ordered_ids: list[str] = []
    for message_id in preferred_message_ids or []:
        message_id = str(message_id or "").strip()
        if message_id and message_id in mapping and message_id not in ordered_ids:
            ordered_ids.append(message_id)

    current_node = str(conversation_payload.get("current_node") or "").strip()
    cursor = current_node
    visited: set[str] = set()
    while cursor and cursor not in visited:
        visited.add(cursor)
        if cursor in mapping and cursor not in ordered_ids:
            ordered_ids.append(cursor)
        node = mapping.get(cursor)
        parent = node.get("parent") if isinstance(node, dict) else None
        cursor = str(parent or "").strip()

    for node_id in mapping:
        if node_id not in ordered_ids:
            ordered_ids.append(node_id)

    return [mapping[node_id] for node_id in ordered_ids if isinstance(mapping.get(node_id), dict)]


def _fetch_chatgpt_web_file_download_link(
    client: httpx.Client,
    *,
    headers: dict[str, str],
    file_id: str,
    conversation_id: str = "",
    post_id: str = "",
    inline: bool = False,
    check_context_scopes_for_conversation_id: str = "",
) -> dict[str, Any]:
    resolved_file_id = str(file_id or "").strip().replace("#", "*")
    if not resolved_file_id:
        return {}

    params: dict[str, Any] = {"inline": str(bool(inline)).lower()}
    if conversation_id:
        params["conversation_id"] = conversation_id
    if post_id:
        params["post_id"] = post_id
    if check_context_scopes_for_conversation_id:
        params["check_context_scopes_for_conversation_id"] = check_context_scopes_for_conversation_id

    response = client.get(
        f"https://chatgpt.com/backend-api/files/download/{resolved_file_id}",
        headers=headers,
        params=params,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _resolve_chatgpt_web_generated_images(
    client: httpx.Client,
    *,
    headers: dict[str, str],
    conversation_id: str,
    preferred_message_ids: Optional[list[str]] = None,
    timeout: float = 240.0,
    poll_interval: float = 2.0,
) -> list[dict[str, Any]]:
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return []

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            conversation_payload = _fetch_chatgpt_web_conversation(
                client,
                headers=headers,
                conversation_id=conversation_id,
            )
        except Exception:
            if time.monotonic() >= deadline:
                return []
            time.sleep(poll_interval)
            continue

        resolved: list[dict[str, Any]] = []
        seen_file_ids: set[str] = set()
        for node in _conversation_node_order(conversation_payload, preferred_message_ids=preferred_message_ids):
            message = node.get("message") if isinstance(node, dict) else None
            if not isinstance(message, dict):
                continue
            for image in _extract_message_image_assets(message):
                file_id = str(image.get("file_id") or "").strip()
                if not file_id or file_id in seen_file_ids:
                    continue
                try:
                    link_payload = _fetch_chatgpt_web_file_download_link(
                        client,
                        headers=headers,
                        file_id=file_id,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    continue

                if str(link_payload.get("status") or "").strip().lower() != "success":
                    continue
                download_url = str(link_payload.get("download_url") or "").strip()
                if not download_url:
                    continue
                resolved.append({
                    **image,
                    "download_url": download_url,
                    "file_name": str(link_payload.get("file_name") or "").strip(),
                    "mime_type": str(link_payload.get("mime_type") or "").strip(),
                    "file_size_bytes": link_payload.get("file_size_bytes"),
                })
                seen_file_ids.add(file_id)

        if resolved:
            return resolved
        if time.monotonic() >= deadline:
            return []
        time.sleep(poll_interval)


def _decode_json_pointer(path: str) -> list[str]:
    if not isinstance(path, str) or not path.startswith("/"):
        return []
    tokens = path.split("/")[1:]
    return [token.replace("~1", "/").replace("~0", "~") for token in tokens]


def _looks_like_message_patch_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    return any(
        isinstance(item, dict)
        and str(item.get("p") or "").startswith("/message/")
        and str(item.get("o") or "").strip().lower() in {"append", "add", "replace"}
        for item in value
    )


def _apply_message_patch(message: dict[str, Any], patch_op: dict[str, Any]) -> bool:
    path = str(patch_op.get("p") or "")
    op = str(patch_op.get("o") or "").strip().lower()
    value = patch_op.get("v")

    tokens = _decode_json_pointer(path)
    if not tokens or tokens[0] != "message":
        return False
    tokens = tokens[1:]
    if not tokens:
        return False

    current: Any = message
    for idx, token in enumerate(tokens[:-1]):
        next_token = tokens[idx + 1]
        next_is_index = next_token.isdigit()
        if isinstance(current, dict):
            child = current.get(token)
            if child is None:
                child = [] if next_is_index else {}
                current[token] = child
            current = child
            continue
        if isinstance(current, list):
            if not token.isdigit():
                return False
            list_index = int(token)
            while len(current) <= list_index:
                current.append([] if next_is_index else {})
            child = current[list_index]
            if child is None:
                child = [] if next_is_index else {}
                current[list_index] = child
            current = child
            continue
        return False

    leaf = tokens[-1]
    if isinstance(current, dict):
        existing = current.get(leaf)
        if op == "append":
            if existing is None:
                current[leaf] = value
            elif isinstance(existing, str) and isinstance(value, str):
                current[leaf] = existing + value
            elif isinstance(existing, list):
                existing.append(value)
            elif isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                current[leaf] = value
            return True
        if op in {"add", "replace"}:
            current[leaf] = value
            return True
        return False

    if isinstance(current, list):
        if not leaf.isdigit():
            return False
        list_index = int(leaf)
        while len(current) <= list_index:
            current.append(None)
        existing = current[list_index]
        if op == "append":
            if existing is None:
                current[list_index] = value
            elif isinstance(existing, str) and isinstance(value, str):
                current[list_index] = existing + value
            elif isinstance(existing, list):
                existing.append(value)
            elif isinstance(existing, dict) and isinstance(value, dict):
                existing.update(value)
            else:
                current[list_index] = value
            return True
        if op in {"add", "replace"}:
            current[list_index] = value
            return True
    return False


def _split_chatgpt_web_message_content(content: Any) -> tuple[str, list[str]]:
    """Return best-effort text plus any attached image sources."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        if content is None:
            return "", []
        return str(content), []

    text_parts: list[str] = []
    image_sources: list[str] = []
    for part in content:
        if isinstance(part, str):
            if part:
                text_parts.append(part)
            continue
        if not isinstance(part, dict):
            rendered = str(part or "")
            if rendered:
                text_parts.append(rendered)
            continue
        ptype = str(part.get("type") or "").strip().lower()
        if ptype in {"text", "input_text"}:
            rendered = str(part.get("text") or "")
            if rendered:
                text_parts.append(rendered)
            continue
        if ptype in {"image_url", "input_image"}:
            image_value = part.get("image_url")
            if isinstance(image_value, dict):
                image_value = image_value.get("url")
            if image_value is None:
                image_value = part.get("url") or part.get("source") or part.get("path")
            image_source = str(image_value or "").strip()
            if image_source:
                image_sources.append(image_source)
            continue
        rendered = str(part.get("content") or part.get("text") or "")
        if rendered:
            text_parts.append(rendered)
    return "\n".join(part for part in text_parts if part).strip(), image_sources


def _messages_include_chatgpt_web_images(messages: list[dict[str, Any]]) -> bool:
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        _, image_sources = _split_chatgpt_web_message_content(item.get("content"))
        if image_sources:
            return True
    return False


def stream_chatgpt_web_completion(
    *,
    access_token: str,
    model: str,
    messages: list[dict[str, Any]],
    instructions: str = "",
    conversation_id: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    session_token: str = "",
    cookie_header: str = "",
    browser_cookies: Any = None,
    on_delta: Optional[Callable[[str], None]] = None,
    timeout: float = 1800.0,
    history_and_training_disabled: bool = False,
    user_agent: str = "",
    device_id: str = "",
    client: Optional[httpx.Client] = None,
) -> dict[str, Any]:
    token = (access_token or "").strip()
    if not token:
        raise ValueError("ChatGPT web access token is required")

    ua = user_agent or _default_user_agent()
    did = device_id or _default_device_id()
    convo_id = (conversation_id or "").strip() or None
    parent_id = (parent_message_id or "").strip() or str(uuid.uuid4())
    prompt_text = _format_initial_message(
        instructions=instructions,
        messages=messages,
        has_remote_thread=bool(convo_id),
    )
    if not prompt_text:
        raise ValueError("ChatGPT web prompt is empty")

    base_headers = _build_chatgpt_web_headers(
        access_token=token,
        session_token=session_token,
        cookie_header=cookie_header,
        browser_cookies=browser_cookies,
        user_agent=ua,
        device_id=did,
    )

    client_ctx = nullcontext(client) if client is not None else httpx.Client(timeout=timeout, follow_redirects=True)
    with client_ctx as client:
        conduit_token = _prepare_conversation(
            client,
            headers=base_headers,
            model=model,
            conversation_id=convo_id,
            parent_message_id=parent_id,
            history_and_training_disabled=history_and_training_disabled,
        )
        chat_requirements = _chat_requirements(client, headers=base_headers)
        requirement_token = str(chat_requirements.get("token") or "").strip()
        proof_token = ""
        proof = chat_requirements.get("proofofwork")
        if isinstance(proof, dict):
            seed = str(proof.get("seed") or "")
            difficulty = str(proof.get("difficulty") or "")
            if seed and difficulty:
                proof_token = _generate_proof_token(seed, difficulty, ua)

        payload: dict[str, Any] = {
            "action": "next",
            "messages": [
                {
                    "id": str(uuid.uuid4()),
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": [prompt_text]},
                    "metadata": {},
                }
            ],
            "parent_message_id": parent_id,
            "model": model,
            "conversation_mode": {"kind": "primary_assistant"},
            "enable_message_followups": True,
            "supports_buffering": True,
            "supported_encodings": ["v1"],
            "system_hints": [],
            "history_and_training_disabled": history_and_training_disabled,
        }
        if convo_id:
            payload["conversation_id"] = convo_id

        headers = {
            **base_headers,
            "Accept": "text/event-stream",
            "openai-sentinel-chat-requirements-token": requirement_token,
        }
        if conduit_token:
            headers["x-conduit-token"] = conduit_token
        if proof_token:
            headers["openai-sentinel-proof-token"] = proof_token

        final_text = ""
        assistant_message_id = parent_id
        final_conversation_id = convo_id
        assistant_message: Optional[dict[str, Any]] = None
        saw_stream_complete = False
        saw_image_generation = False
        image_message_ids: list[str] = []
        resolved_images: list[dict[str, Any]] = []
        api_start = time.monotonic()
        with client.stream(
            "POST",
            "https://chatgpt.com/backend-api/f/conversation",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", "ignore")
                    if not isinstance(line, str) or not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(event, dict):
                        continue

                    event_conversation_id = str(event.get("conversation_id") or "").strip()
                    if not event_conversation_id:
                        nested_v = event.get("v")
                        if isinstance(nested_v, dict):
                            event_conversation_id = str(nested_v.get("conversation_id") or "").strip()
                    if event_conversation_id:
                        final_conversation_id = event_conversation_id

                    event_type = str(event.get("type") or "").strip()
                    if event_type == "message_stream_complete":
                        saw_stream_complete = True
                    elif event_type == "server_ste_metadata":
                        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
                        if (
                            str(metadata.get("tool_name") or "").strip() == "ImageGenToolTemporal"
                            or str(metadata.get("turn_use_case") or "").strip().lower() == "image gen"
                        ):
                            saw_image_generation = True

                    marker_message_id = str(event.get("message_id") or "").strip()
                    if marker_message_id:
                        assistant_message_id = marker_message_id

                    message = _extract_event_message(event)
                    if isinstance(message, dict):
                        if _message_suggests_image_generation(message):
                            saw_image_generation = True
                            message_id = str(message.get("id") or "").strip()
                            if message_id and message_id not in image_message_ids:
                                image_message_ids.append(message_id)

                        author = message.get("author") if isinstance(message.get("author"), dict) else {}
                        if author.get("role") == "assistant":
                            assistant_message = copy.deepcopy(message)
                            message_id = str(message.get("id") or "").strip()
                            if message_id:
                                assistant_message_id = message_id
                            text = _extract_message_text(assistant_message)
                            if text:
                                delta = text[len(final_text):] if text.startswith(final_text) else text
                                final_text = text
                                if delta and on_delta is not None:
                                    on_delta(delta)
                        continue

                    patch_ops = event.get("v") if isinstance(event.get("v"), list) else None
                    is_patch_event = (
                        str(event.get("o") or "").strip().lower() == "patch"
                        or _looks_like_message_patch_list(patch_ops)
                    )
                    if is_patch_event and patch_ops is not None:
                        if assistant_message is None:
                            assistant_message = {
                                "id": assistant_message_id,
                                "author": {"role": "assistant"},
                                "content": {"content_type": "text", "parts": [""]},
                                "metadata": {},
                            }
                        for patch_op in patch_ops:
                            if not isinstance(patch_op, dict):
                                continue
                            if not _apply_message_patch(assistant_message, patch_op):
                                continue
                            if _message_suggests_image_generation(assistant_message):
                                saw_image_generation = True
                            text = _extract_message_text(assistant_message)
                            if not text:
                                continue
                            delta = text[len(final_text):] if text.startswith(final_text) else text
                            final_text = text
                            if delta and on_delta is not None:
                                on_delta(delta)
            except httpx.RemoteProtocolError:
                if not final_text.strip() and not saw_stream_complete:
                    raise

        if final_conversation_id and saw_image_generation:
            default_image_timeout = float(os.getenv("CHATGPT_WEB_IMAGE_POLL_TIMEOUT", "240"))
            remaining_timeout = max(0.0, float(timeout) - (time.monotonic() - api_start))
            image_timeout = min(default_image_timeout, remaining_timeout)
            if image_timeout > 0.0:
                poll_interval = max(0.25, float(os.getenv("CHATGPT_WEB_IMAGE_POLL_INTERVAL", "2")))
                resolved_images = _resolve_chatgpt_web_generated_images(
                    client,
                    headers=base_headers,
                    conversation_id=final_conversation_id,
                    preferred_message_ids=image_message_ids,
                    timeout=image_timeout,
                    poll_interval=poll_interval,
                )

    if resolved_images:
        image_urls = [str(item.get("download_url") or "").strip() for item in resolved_images]
        image_urls = [url for url in image_urls if url]
        if image_urls:
            cleaned_text = final_text.strip()
            if not cleaned_text or _looks_like_image_generation_spec(cleaned_text) or cleaned_text.lower().startswith("processing image"):
                final_text = "\n".join(image_urls)
            else:
                joined_urls = "\n".join(image_urls)
                if joined_urls not in cleaned_text:
                    final_text = f"{cleaned_text}\n\n{joined_urls}"
            preferred_message_id = str(resolved_images[0].get("message_id") or "").strip()
            if preferred_message_id:
                assistant_message_id = preferred_message_id

    if not final_text.strip():
        raise RuntimeError("ChatGPT web transport returned no assistant text")

    return {
        "content": final_text.strip(),
        "conversation_id": final_conversation_id,
        "parent_message_id": assistant_message_id,
        "message_id": assistant_message_id,
        "model": model,
        "finish_reason": "stop",
        "images": resolved_images,
    }
