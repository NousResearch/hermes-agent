"""Persistent Qwen-web session hints for OpenAI-compatible relays.

Some Qwen web relays expose an OpenAI-compatible ``/chat/completions``
endpoint but keep the real upstream conversation as ``chatId``/``parentId``.
Hermes normally treats OpenAI-compatible providers as stateless, so a relay can
create a fresh Qwen chat every turn unless we give it a stable conversation key.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from utils import base_url_host_matches


_STATE_VERSION = 1
_STATE_FILE = "qwen_sessions.json"


def _normalized_base_url(value: Any) -> str:
    return str(value or "").strip().rstrip("/")


def _custom_provider_entry_matches_model(agent: Any, entry: dict[str, Any]) -> bool:
    model = str(getattr(agent, "model", "") or "").strip()
    entry_model = str(entry.get("model", "") or "").strip()
    if entry_model:
        return not model or entry_model == model

    models = entry.get("models")
    if isinstance(models, dict) and models:
        return not model or model in models
    return True


def _qwen_provider_identity(agent: Any) -> str:
    """Return the stable Qwen-web provider identity, or ``""``.

    Runtime resolution can turn a user-defined provider such as ``qwen-local``
    into the generic provider name ``custom`` while preserving its base URL and
    a normalized entry in ``agent._custom_providers``.  The session bridge must
    still recognize that endpoint as Qwen and must use the original provider
    key for the conversation digest, otherwise CLI/runtime-resolved and
    gateway-switched sessions get different conversation IDs.
    """

    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if not provider:
        return ""

    base_url = str(getattr(agent, "base_url", "") or "")
    if base_url_host_matches(base_url, "portal.qwen.ai"):
        return ""

    if "qwen" in provider:
        return provider

    if provider != "custom":
        return ""

    target_url = _normalized_base_url(base_url)
    if not target_url:
        return ""

    for entry in getattr(agent, "_custom_providers", []) or []:
        if not isinstance(entry, dict):
            continue
        if _normalized_base_url(entry.get("base_url")) != target_url:
            continue
        if not _custom_provider_entry_matches_model(agent, entry):
            continue

        provider_key = str(entry.get("provider_key", "") or "").strip()
        name = str(entry.get("name", "") or "").strip()
        haystack = f"{provider_key} {name}".lower()
        if "qwen" in haystack:
            return (provider_key or name).lower()

    return ""


def is_qwen_session_provider(agent: Any) -> bool:
    """Return True for custom/user Qwen-web relays that need session hints.

    The built-in Qwen OAuth/Portal provider already has its own metadata
    contract (``sessionId``/``promptId``) and should not receive chatId/parentId
    fields intended for local web relays.
    """

    return bool(_qwen_provider_identity(agent))


def _state_path() -> Path:
    return get_hermes_home() / "state" / _STATE_FILE


def _read_state() -> dict[str, Any]:
    path = _state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": _STATE_VERSION, "sessions": {}}
    except Exception:
        return {"version": _STATE_VERSION, "sessions": {}}

    if not isinstance(data, dict):
        return {"version": _STATE_VERSION, "sessions": {}}
    sessions = data.get("sessions")
    if not isinstance(sessions, dict):
        data["sessions"] = {}
    data["version"] = _STATE_VERSION
    return data


def _write_state(data: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{_STATE_FILE}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def _stable_session_source(agent: Any) -> str:
    gateway_key = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    if gateway_key:
        return gateway_key

    parts = [
        str(getattr(agent, "platform", "") or ""),
        str(getattr(agent, "_chat_id", "") or ""),
        str(getattr(agent, "_thread_id", "") or ""),
        str(getattr(agent, "session_id", "") or ""),
    ]
    source = ":".join(p for p in parts if p)
    return source or "hermes"


def _conversation_id(agent: Any) -> str:
    provider = _qwen_provider_identity(agent) or str(getattr(agent, "provider", "") or "")
    base_url = _normalized_base_url(getattr(agent, "base_url", ""))
    source = _stable_session_source(agent)
    digest = hashlib.sha256(f"{provider}\n{base_url}\n{source}".encode("utf-8")).hexdigest()
    return f"hermes_{digest[:32]}"


def load_qwen_session(agent: Any) -> dict[str, Any] | None:
    if not is_qwen_session_provider(agent):
        return None
    conversation_id = _conversation_id(agent)
    data = _read_state()
    session = data.get("sessions", {}).get(conversation_id)
    if isinstance(session, dict):
        return dict(session)
    return {"conversation_id": conversation_id}


def build_qwen_session_metadata(agent: Any) -> dict[str, Any] | None:
    """Build top-level OpenAI ``metadata`` for Qwen-web relays.

    ``conversation_id`` gives the relay a stable scoped key. If Hermes has seen
    upstream ``chatId``/``parentId`` before, include them so the relay can resume
    even after its in-memory map was lost.
    """

    session = load_qwen_session(agent)
    if session is None:
        return None

    conversation_id = str(session.get("conversation_id") or _conversation_id(agent))
    metadata: dict[str, Any] = {
        "conversation_id": conversation_id,
        "sessionId": conversation_id,
    }

    chat_id = session.get("chatId") or session.get("chat_id")
    parent_id = session.get("parentId") or session.get("parent_id")
    if chat_id:
        metadata["chatId"] = chat_id
    if parent_id:
        metadata["parentId"] = parent_id
    return metadata


def _get_response_value(response: Any, *names: str) -> Any:
    if response is None:
        return None
    if isinstance(response, dict):
        for name in names:
            if response.get(name):
                return response.get(name)
    for name in names:
        value = getattr(response, name, None)
        if value:
            return value
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, dict):
        for name in names:
            if extra.get(name):
                return extra.get(name)
    return None


def maybe_update_qwen_session_from_response(agent: Any, response: Any) -> None:
    """Persist upstream Qwen chat identifiers from a successful response."""

    if not is_qwen_session_provider(agent):
        return

    chat_id = _get_response_value(response, "chatId", "chat_id")
    parent_id = _get_response_value(response, "parentId", "parent_id", "response_id")
    if not chat_id and not parent_id:
        return

    conversation_id = _conversation_id(agent)
    data = _read_state()
    sessions = data.setdefault("sessions", {})
    previous = sessions.get(conversation_id) if isinstance(sessions.get(conversation_id), dict) else {}
    entry = {
        "conversation_id": conversation_id,
        "provider": _qwen_provider_identity(agent) or str(getattr(agent, "provider", "") or ""),
        "base_url": _normalized_base_url(getattr(agent, "base_url", "")),
        "session_source": _stable_session_source(agent),
        "updated_at": int(time.time()),
    }
    if previous:
        entry.update(previous)
    if chat_id:
        entry["chatId"] = chat_id
    if parent_id:
        entry["parentId"] = parent_id
    entry["updated_at"] = int(time.time())
    sessions[conversation_id] = entry
    _write_state(data)
