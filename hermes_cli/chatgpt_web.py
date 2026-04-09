"""Helpers for ChatGPT.com web-model access and streaming conversation transport."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import time
import uuid
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


def _default_user_agent() -> str:
    return os.getenv("CHATGPT_WEB_USER_AGENT", "").strip() or DEFAULT_CHATGPT_WEB_USER_AGENT


def _default_device_id() -> str:
    return os.getenv("CHATGPT_WEB_DEVICE_ID", "").strip() or str(uuid.uuid4())


def _build_cookie_header(*, session_token: str = "", device_id: str = "") -> str:
    parts: list[str] = []
    if session_token:
        parts.append(f"__Secure-next-auth.session-token={session_token}")
    if device_id:
        parts.append(f"oai-did={device_id}")
    return "; ".join(parts)


def _build_chatgpt_web_headers(
    *,
    access_token: str,
    session_token: str = "",
    user_agent: str = "",
    device_id: str = "",
    accept: str = "application/json",
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": accept,
        "User-Agent": user_agent or _default_user_agent(),
        "Content-Type": "application/json",
        "Oai-Device-Id": device_id or _default_device_id(),
        "Referer": "https://chatgpt.com/",
        "Origin": "https://chatgpt.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
    }
    cookie_header = _build_cookie_header(
        session_token=session_token,
        device_id=headers["Oai-Device-Id"],
    )
    if cookie_header:
        headers["Cookie"] = cookie_header
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

        pool = load_pool("chatgpt-web")
        if pool and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                pool_api_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
                if pool_api_key:
                    return {
                        "provider": "chatgpt-web",
                        "api_key": str(pool_api_key).strip(),
                        "base_url": (getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", "") or DEFAULT_CHATGPT_WEB_BASE_URL).rstrip("/"),
                        "source": f"pool:{getattr(entry, 'label', 'unknown')}",
                        "session_token": "",
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
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if role == "user":
            latest_user = content
        if role in {"user", "assistant"} and content.strip():
            transcript_lines.append(f"{role.title()}: {content.strip()}")

    if has_remote_thread:
        return latest_user.strip()

    prompt_parts: list[str] = []
    if instructions.strip():
        prompt_parts.append(f"System instructions:\n{instructions.strip()}")
    if transcript_lines:
        prompt_parts.append("Conversation so far:\n" + "\n".join(transcript_lines))
    return "\n\n".join(part for part in prompt_parts if part).strip()


def stream_chatgpt_web_completion(
    *,
    access_token: str,
    model: str,
    messages: list[dict[str, Any]],
    instructions: str = "",
    conversation_id: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    session_token: str = "",
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
        with client.stream(
            "POST",
            "https://chatgpt.com/backend-api/f/conversation",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", "ignore")
                if not isinstance(line, str) or not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    event = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(event, dict):
                    continue
                event_conversation_id = str(event.get("conversation_id") or "").strip()
                if event_conversation_id:
                    final_conversation_id = event_conversation_id
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                author = message.get("author") if isinstance(message.get("author"), dict) else {}
                if author.get("role") != "assistant":
                    continue
                message_id = str(message.get("id") or "").strip()
                if message_id:
                    assistant_message_id = message_id
                content = message.get("content") if isinstance(message.get("content"), dict) else {}
                if content.get("content_type") != "text":
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list) or not parts:
                    continue
                text = str(parts[0] or "")
                if not text:
                    continue
                delta = text[len(final_text):] if text.startswith(final_text) else text
                final_text = text
                if delta and on_delta is not None:
                    on_delta(delta)

    if not final_text.strip():
        raise RuntimeError("ChatGPT web transport returned no assistant text")

    return {
        "content": final_text.strip(),
        "conversation_id": final_conversation_id,
        "parent_message_id": assistant_message_id,
        "message_id": assistant_message_id,
        "model": model,
        "finish_reason": "stop",
    }
