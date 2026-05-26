"""Hermes API memory provider.

Provides contact-aware memory by resolving the current gateway user against a
local Hermes API people graph, then injecting that contact's ``contactMd`` and
``memoryMd`` into the agent context. Also exposes lightweight tools for looking
up contacts and updating contact-scoped memory.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:4000"
_TIMEOUT_SECONDS = 5
_MAX_CONTEXT_CHARS = 6000


def _config_path(hermes_home: str) -> Path:
    return Path(hermes_home) / "hermes_api_memory.json"


def _load_config(hermes_home: Optional[str] = None) -> Dict[str, Any]:
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except Exception:
            hermes_home = ""
    if not hermes_home:
        return {}
    path = _config_path(hermes_home)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        logger.debug("Failed to parse Hermes API memory config at %s", path, exc_info=True)
        return {}


def _save_config(values: Dict[str, Any], hermes_home: str) -> None:
    clean = {}
    base_url = str(values.get("base_url") or "").strip().rstrip("/")
    if base_url:
        clean["base_url"] = base_url
    path = _config_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _base_url() -> str:
    env_value = os.environ.get("HERMES_API_BASE_URL", "").strip()
    if env_value:
        return env_value.rstrip("/")
    config_value = str(_load_config().get("base_url") or "").strip()
    return (config_value or _DEFAULT_BASE_URL).rstrip("/")


def _request_json(
    method: str,
    path: str,
    *,
    query: Optional[Dict[str, str]] = None,
    body: Optional[Dict[str, Any]] = None,
    timeout: int = _TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    url = f"{_base_url()}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Hermes API {method} {path} failed with HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hermes API unavailable at {_base_url()}: {exc.reason}") from exc

    if not raw.strip():
        return {}
    return json.loads(raw)


def _data(payload: Dict[str, Any]) -> Any:
    return payload.get("data") if isinstance(payload, dict) else None


def _detect_identity_kind(platform: str, user_id: str) -> str:
    platform = (platform or "").lower().strip()
    if platform in {"telegram", "whatsapp", "discord", "slack", "signal", "matrix"}:
        return platform
    if "@s.whatsapp.net" in user_id or "@lid" in user_id or "@g.us" in user_id:
        return "whatsapp"
    if user_id.isdigit():
        return "telegram"
    return platform or "telegram"


def _contact_label(contact: Dict[str, Any]) -> str:
    return str(contact.get("name") or contact.get("username") or contact.get("id") or "unknown")


def _format_contact_context(contact: Dict[str, Any]) -> str:
    parts = [f"## Current contact: {_contact_label(contact)}"]
    username = contact.get("username")
    email = contact.get("email")
    tags = contact.get("tags") or []
    if username:
        parts.append(f"username: {username}")
    if email:
        parts.append(f"email: {email}")
    if tags:
        parts.append("tags: " + ", ".join(str(t) for t in tags))

    contact_md = (contact.get("contactMd") or "").strip()
    memory_md = (contact.get("memoryMd") or "").strip()
    if contact_md:
        parts.append("### Contact profile\n" + contact_md)
    if memory_md:
        parts.append("### Contact memory\n" + memory_md)

    text = "\n\n".join(parts).strip()
    if len(text) > _MAX_CONTEXT_CHARS:
        text = text[: _MAX_CONTEXT_CHARS - 20].rstrip() + "\n…"
    return text


def _result(success: bool, **kwargs: Any) -> str:
    return json.dumps({"success": success, **kwargs}, ensure_ascii=False)


RESOLVE_IDENTITY_SCHEMA = {
    "name": "hermes_api_resolve_identity",
    "description": "Resolve a platform identity to a contact using Hermes API.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "description": "Identity kind, e.g. whatsapp, telegram, email, github.",
            },
            "value": {"type": "string", "description": "Identity value to resolve."},
        },
        "required": ["kind", "value"],
    },
}

GET_CONTACT_SCHEMA = {
    "name": "hermes_api_get_contact",
    "description": "Get one Hermes API contact, including identities, contactMd, and memoryMd.",
    "parameters": {
        "type": "object",
        "properties": {"contact_id": {"type": "string", "description": "Contact id."}},
        "required": ["contact_id"],
    },
}

LIST_CONTACTS_SCHEMA = {
    "name": "hermes_api_list_contacts",
    "description": "List contacts from Hermes API, optionally filtering client-side by a search string.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional case-insensitive search over name, username, tags, contactMd, memoryMd, and notes.",
            },
            "limit": {"type": "integer", "description": "Maximum contacts to return.", "default": 10},
        },
        "required": [],
    },
}

UPDATE_MEMORY_SCHEMA = {
    "name": "hermes_api_update_contact_memory",
    "description": "Replace a contact's memoryMd in Hermes API.",
    "parameters": {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "Contact id."},
            "memory_md": {"type": "string", "description": "New memoryMd markdown, max 2048 chars."},
        },
        "required": ["contact_id", "memory_md"],
    },
}

APPEND_MEMORY_SCHEMA = {
    "name": "hermes_api_append_contact_memory",
    "description": "Append a short bullet or paragraph to a contact's memoryMd in Hermes API.",
    "parameters": {
        "type": "object",
        "properties": {
            "contact_id": {"type": "string", "description": "Contact id."},
            "content": {"type": "string", "description": "Memory content to append."},
        },
        "required": ["contact_id", "content"],
    },
}


class HermesApiMemoryProvider(MemoryProvider):
    """Contact-aware memory backed by the local Hermes API people graph."""

    def __init__(self) -> None:
        self._session_id = ""
        self._platform = ""
        self._user_id = ""
        self._user_name = ""
        self._contact: Optional[Dict[str, Any]] = None
        self._contact_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "hermes_api"

    def is_available(self) -> bool:
        return bool(_base_url())

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "base_url",
                "description": "Hermes API base URL",
                "secret": False,
                "required": False,
                "default": _DEFAULT_BASE_URL,
                "env_var": "HERMES_API_BASE_URL",
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        _save_config(values, hermes_home)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._platform = str(kwargs.get("platform") or "")
        self._user_id = str(kwargs.get("user_id") or "")
        self._user_name = str(kwargs.get("user_name") or "")

        contact = None
        if self._user_id:
            contact = self._resolve_current_contact(self._platform, self._user_id)
        if not contact and self._user_name:
            contact = self._find_contact_by_text(self._user_name)
        with self._contact_lock:
            self._contact = contact

    def system_prompt_block(self) -> str:
        return (
            "# Hermes API Memory\n"
            f"Active. Base URL: {_base_url()}\n"
            "Provides contact-aware context from the Hermes API people graph. "
            "Use hermes_api_* tools for contact lookup, identity resolution, and contact memory updates."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._contact_lock:
            contact = dict(self._contact) if self._contact else None
        if not contact:
            return ""
        return "<hermes-api-contact-context>\n" + _format_contact_context(contact) + "\n</hermes-api-contact-context>"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        return None

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs: Any,
    ) -> None:
        self._session_id = new_session_id

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Contact memory represents what Hermes knows about the current person.
        # Mirror user-profile writes only; generic ``memory`` entries often hold
        # environment or workflow facts that should not be attached to a contact.
        if action not in {"add", "replace"} or target != "user":
            return
        with self._contact_lock:
            contact = dict(self._contact) if self._contact else None
        if not contact:
            return
        entry = content.strip()
        if not entry:
            return
        try:
            self._append_contact_memory(str(contact["id"]), f"- {entry}")
        except Exception as exc:
            logger.debug("Hermes API memory mirror failed: %s", exc)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            RESOLVE_IDENTITY_SCHEMA,
            GET_CONTACT_SCHEMA,
            LIST_CONTACTS_SCHEMA,
            UPDATE_MEMORY_SCHEMA,
            APPEND_MEMORY_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        try:
            if tool_name == "hermes_api_resolve_identity":
                return self._tool_resolve_identity(args)
            if tool_name == "hermes_api_get_contact":
                return self._tool_get_contact(args)
            if tool_name == "hermes_api_list_contacts":
                return self._tool_list_contacts(args)
            if tool_name == "hermes_api_update_contact_memory":
                return self._tool_update_memory(args)
            if tool_name == "hermes_api_append_contact_memory":
                return self._tool_append_memory(args)
        except Exception as exc:
            return tool_error(str(exc), tool=tool_name)
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def _resolve_current_contact(self, platform: str, user_id: str) -> Optional[Dict[str, Any]]:
        kind = _detect_identity_kind(platform, user_id)
        matches = _data(_request_json("GET", "/api/v1/identities", query={"kind": kind, "value": user_id})) or []
        if not matches:
            return None
        contact_id = matches[0].get("contact", {}).get("id")
        if not contact_id:
            return None
        return self._get_contact(contact_id)

    def _find_contact_by_text(self, text: str) -> Optional[Dict[str, Any]]:
        rows = self._list_contacts(query=text, limit=1)
        return rows[0] if rows else None

    def _get_contact(self, contact_id: str) -> Dict[str, Any]:
        contact = _data(_request_json("GET", f"/api/v1/contacts/{urllib.parse.quote(contact_id, safe='')}"))
        if not isinstance(contact, dict):
            raise RuntimeError(f"Contact not found: {contact_id}")
        return contact

    def _list_contacts(self, *, query: str = "", limit: int = 10) -> List[Dict[str, Any]]:
        rows = _data(_request_json("GET", "/api/v1/contacts")) or []
        if query:
            needle = query.lower()
            filtered = []
            for row in rows:
                haystack = " ".join(
                    str(value)
                    for value in [
                        row.get("name"),
                        row.get("username"),
                        row.get("email"),
                        " ".join(row.get("tags") or []),
                        row.get("contactMd"),
                        row.get("memoryMd"),
                        row.get("notes"),
                    ]
                    if value
                ).lower()
                if needle in haystack:
                    filtered.append(row)
            rows = filtered
        return rows[: max(1, min(int(limit or 10), 50))]

    def _update_contact_memory(self, contact_id: str, memory_md: str) -> Dict[str, Any]:
        if len(memory_md) > 2048:
            raise ValueError("memory_md exceeds Hermes API limit of 2048 chars")
        contact = _data(
            _request_json(
                "PATCH",
                f"/api/v1/contacts/{urllib.parse.quote(contact_id, safe='')}",
                body={"memoryMd": memory_md},
            )
        )
        if not isinstance(contact, dict):
            raise RuntimeError(f"Failed to update contact: {contact_id}")
        with self._contact_lock:
            if self._contact and self._contact.get("id") == contact_id:
                self._contact = contact
        return contact

    def _append_contact_memory(self, contact_id: str, content: str) -> Dict[str, Any]:
        contact = self._get_contact(contact_id)
        existing = str(contact.get("memoryMd") or "").rstrip()
        addition = content.strip()
        if not addition:
            raise ValueError("content is empty")
        next_memory = f"{existing}\n{addition}".strip() if existing else addition
        return self._update_contact_memory(contact_id, next_memory)

    def _tool_resolve_identity(self, args: Dict[str, Any]) -> str:
        kind = str(args.get("kind") or "").strip()
        value = str(args.get("value") or "").strip()
        if not kind or not value:
            return tool_error("kind and value are required", tool="hermes_api_resolve_identity")
        matches = _data(_request_json("GET", "/api/v1/identities", query={"kind": kind, "value": value})) or []
        return _result(True, data=matches)

    def _tool_get_contact(self, args: Dict[str, Any]) -> str:
        contact_id = str(args.get("contact_id") or "").strip()
        if not contact_id:
            return tool_error("contact_id is required", tool="hermes_api_get_contact")
        return _result(True, data=self._get_contact(contact_id))

    def _tool_list_contacts(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        limit = int(args.get("limit") or 10)
        return _result(True, data=self._list_contacts(query=query, limit=limit))

    def _tool_update_memory(self, args: Dict[str, Any]) -> str:
        contact_id = str(args.get("contact_id") or "").strip()
        memory_md = str(args.get("memory_md") or "")
        if not contact_id:
            return tool_error("contact_id is required", tool="hermes_api_update_contact_memory")
        return _result(True, data=self._update_contact_memory(contact_id, memory_md))

    def _tool_append_memory(self, args: Dict[str, Any]) -> str:
        contact_id = str(args.get("contact_id") or "").strip()
        content = str(args.get("content") or "").strip()
        if not contact_id:
            return tool_error("contact_id is required", tool="hermes_api_append_contact_memory")
        return _result(True, data=self._append_contact_memory(contact_id, content))


def register(ctx) -> None:
    """Register Hermes API as a memory provider plugin."""
    ctx.register_memory_provider(HermesApiMemoryProvider())
