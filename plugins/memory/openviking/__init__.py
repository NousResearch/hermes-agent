"""OpenViking memory plugin — full bidirectional MemoryProvider interface.

Context database by Volcengine (ByteDance) that organizes agent knowledge
into a filesystem hierarchy (viking:// URIs) with tiered context loading,
automatic memory extraction, and session management.

Original PR #3369 by Mibayy, rewritten to use the full OpenViking session
lifecycle instead of read-only search endpoints.

Config via environment variables (profile-scoped via each profile's .env)
or a linked OpenViking CLI config:
  OPENVIKING_ENDPOINT  — Server URL (default: http://127.0.0.1:1933)
  OPENVIKING_API_KEY   — API key (required for authenticated servers)
  OPENVIKING_ACCOUNT   — Tenant account for local/trusted mode (default: default)
  OPENVIKING_USER      — Tenant user for local/trusted mode (default: default)
  OPENVIKING_AGENT     — Hermes peer ID in OpenViking (default: hermes)

Capabilities:
  - Automatic memory extraction on session commit (6 categories)
  - Tiered context: L0 (~100 tokens), L1 (~2k), L2 (full)
  - Semantic search with hierarchical directory retrieval
  - Filesystem-style browsing via viking:// URIs
  - Resource ingestion (URLs, docs, code)
"""

from __future__ import annotations

import atexit
import json
import logging
import mimetypes
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse
from urllib.request import url2pathname

from agent.message_content import flatten_message_text
from agent.memory_provider import MemoryProvider
from agent.skill_commands import extract_user_instruction_from_skill_message
from tools.registry import tool_error
from utils import atomic_json_write, env_var_enabled

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_OPENVIKING_SERVICE_ENDPOINT = "https://api.vikingdb.cn-beijing.volces.com/openviking"
_DEFAULT_AGENT = "hermes"
_AGENT_PROMPT_LABEL = "Hermes peer ID in OpenViking"
_OVCLI_CONFIG_ENV = "OPENVIKING_CLI_CONFIG_FILE"
_OVCLI_DEFAULT_RELATIVE_PATH = ".openviking/ovcli.conf"
_OVCLI_SAVED_PREFIX = "ovcli.conf."
_OPENVIKING_ENV_KEYS = (
    "OPENVIKING_ENDPOINT",
    "OPENVIKING_API_KEY",
    "OPENVIKING_ACCOUNT",
    "OPENVIKING_USER",
    "OPENVIKING_AGENT",
)
_TIMEOUT = 30.0
_SESSION_DRAIN_TIMEOUT = 10.0
_DEFERRED_COMMIT_TIMEOUT = (_TIMEOUT * 2) + 5.0
_REMOTE_RESOURCE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")
_SYNC_TRACE_ENV = "HERMES_OPENVIKING_SYNC_TRACE"

# Maps the viking_remember `category` enum to a viking:// subdirectory.
# Keep in sync with REMEMBER_SCHEMA.parameters.properties.category.enum.
_CATEGORY_SUBDIR_MAP = {
    "preference": "preferences",
    "entity": "entities",
    "event": "events",
    "case": "cases",
    "pattern": "patterns",
}
_DEFAULT_MEMORY_SUBDIR = "preferences"

# Maps the built-in memory tool's `target` ("user" vs "memory") to a subdir
# for on_memory_write mirroring. User profile facts → preferences; agent
# notes / observations → patterns. Anything unknown falls back to the default.
_MEMORY_WRITE_TARGET_SUBDIR_MAP = {
    "user": "preferences",
    "memory": "patterns",
}
_LOCAL_OPENVIKING_HOSTS = {"localhost", "127.0.0.1", "::1"}
_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT = 60.0
_OPENVIKING_SERVER_LOG_RELATIVE_PATH = Path("logs") / "openviking-server.log"
_OPENVIKING_RESPONDED_FAILURE_PREFIX = "OpenViking server responded"
_SETUP_CANCELLED = object()


@dataclass(frozen=True)
class _OvcliProfile:
    source: str
    name: str
    path: Path
    data: dict
    values: dict
    is_active: bool = False


class _OpenVikingHTTPError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _sanitize_openviking_error_message(message: str, status_code: Optional[int] = None) -> str:
    text = (message or "").strip()
    status = f"HTTP {status_code}" if status_code else "HTTP error"
    looks_like_html = bool(re.search(r"^\s*<(!doctype|html|head|body)\b", text, flags=re.IGNORECASE))
    if looks_like_html:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"\s+", " ", title_match.group(1)).strip()
            if "|" in title:
                title = title.split("|", 1)[1].strip()
            if status_code and title.startswith(f"{status_code}:"):
                title = title.split(":", 1)[1].strip()
            if title:
                return f"{status}: {title}"
        return f"{status}: OpenViking endpoint returned an HTML error page."

    if len(text) > 300:
        return text[:297].rstrip() + "..."
    return text or status


def _format_openviking_exception(error: Exception) -> str:
    status_code = None
    if isinstance(error, _OpenVikingHTTPError):
        status_code = error.status_code
    else:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    return _sanitize_openviking_error_message(str(error), status_code)


def _derive_openviking_user_text(content: Any) -> str:
    """Strip Hermes slash-skill scaffolding before sending content to OpenViking.

    Defense-in-depth: MemoryManager already strips skill scaffolding for the
    whole provider fan-out (see ``MemoryManager._strip_skill_scaffolding``), so
    in normal operation this receives already-clean text and passes it through
    unchanged. It stays here so OpenViking is correct if its hooks are ever
    invoked outside the manager. Delegates to the canonical extractor in
    ``agent.skill_commands`` — no duplicated marker literals, no drift risk.
    """
    return extract_user_instruction_from_skill_message(content) or ""


def _sync_trace_enabled() -> bool:
    return env_var_enabled(_SYNC_TRACE_ENV)


def _preview(value: Any, limit: int = 160) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", "\\n")
    if len(text) > limit:
        return text[:limit] + "..."
    return text


# ---------------------------------------------------------------------------
# Process-level atexit safety net — ensures pending sessions are committed
# even if shutdown_memory_provider is never called (e.g. gateway crash,
# SIGKILL, or exception in the session expiry watcher preventing shutdown).
# ---------------------------------------------------------------------------
_last_active_provider: Optional["OpenVikingMemoryProvider"] = None


def _atexit_commit_sessions():
    """Fire on_session_end for the last active provider on process exit."""
    global _last_active_provider
    provider = _last_active_provider
    if provider is None:
        return
    _last_active_provider = None
    try:
        provider.on_session_end([])
    except Exception:
        pass  # best-effort at shutdown time


atexit.register(_atexit_commit_sessions)


# ---------------------------------------------------------------------------
# HTTP helper — uses httpx to avoid requiring the openviking SDK
# ---------------------------------------------------------------------------

def _get_httpx():
    """Lazy import httpx."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


class _VikingClient:
    """Thin HTTP client for the OpenViking REST API."""

    def __init__(self, endpoint: str, api_key: str = "",
                 account: Optional[str] = None, user: Optional[str] = None,
                 agent: Optional[str] = None):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        # Account/user are local/trusted-mode tenant identity. API-key requests
        # omit these headers by default; trusted-mode retry may send them only
        # after OpenViking explicitly asks for asserted tenant identity.
        self._account = account or os.environ.get("OPENVIKING_ACCOUNT", "default")
        self._user = user or os.environ.get("OPENVIKING_USER", "default")
        self._agent = agent if agent is not None else os.environ.get("OPENVIKING_AGENT", _DEFAULT_AGENT)
        self._httpx = _get_httpx()
        if self._httpx is None:
            raise ImportError("httpx is required for OpenViking: pip install httpx")

    def _headers(self, *, include_tenant: bool | None = None) -> dict:
        if include_tenant is None:
            include_tenant = not bool(self._api_key)

        h = {"Content-Type": "application/json"}
        if self._agent:
            h["X-OpenViking-Actor-Peer"] = self._agent
        if include_tenant:
            if self._account:
                h["X-OpenViking-Account"] = self._account
            if self._user:
                h["X-OpenViking-User"] = self._user
        if self._api_key:
            h["X-API-Key"] = self._api_key
            h["Authorization"] = "Bearer " + self._api_key
        return h

    def _url(self, path: str) -> str:
        return f"{self._endpoint}{path}"

    def _multipart_headers(self, *, include_tenant: bool | None = None) -> dict:
        headers = self._headers(include_tenant=include_tenant)
        headers.pop("Content-Type", None)
        return headers

    @staticmethod
    def _needs_trusted_identity_retry(exc: Exception) -> bool:
        message = str(exc)
        return (
            "Trusted mode requests must include X-OpenViking-Account" in message
            or "Trusted mode requests must include X-OpenViking-User" in message
            or "Trusted mode requests must include X-OpenViking-Account or explicit account_id" in message
        )

    def _send_with_trusted_identity_retry(self, send, *, multipart: bool = False) -> dict:
        try:
            headers = self._multipart_headers() if multipart else self._headers()
            return self._parse_response(send(headers))
        except Exception as exc:
            if not self._api_key or not self._needs_trusted_identity_retry(exc):
                raise
            headers = (
                self._multipart_headers(include_tenant=True)
                if multipart else self._headers(include_tenant=True)
            )
            return self._parse_response(send(headers))

    def _parse_response(self, resp) -> dict:
        try:
            data = resp.json()
        except Exception:
            data = None

        if resp.status_code >= 400:
            message = _sanitize_openviking_error_message(
                getattr(resp, "text", ""),
                resp.status_code,
            )
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    code = error.get("code", "HTTP_ERROR")
                    message = f"{code}: {error.get('message', message)}"
                    raise _OpenVikingHTTPError(message, resp.status_code)
                if data.get("status") == "error":
                    raise _OpenVikingHTTPError(str(data), resp.status_code)
            raise _OpenVikingHTTPError(message or f"HTTP {resp.status_code}", resp.status_code)

        if isinstance(data, dict) and data.get("status") == "error":
            error = data.get("error")
            if isinstance(error, dict):
                code = error.get("code", "OPENVIKING_ERROR")
                message = error.get("message", "")
                raise RuntimeError(f"{code}: {message}")
            raise RuntimeError(str(data))

        if data is None:
            return {}
        return data

    def get(self, path: str, **kwargs) -> dict:
        return self._send_with_trusted_identity_retry(
            lambda headers: self._httpx.get(
                self._url(path), headers=headers, timeout=_TIMEOUT, **kwargs
            )
        )

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        return self._send_with_trusted_identity_retry(
            lambda headers: self._httpx.post(
                self._url(path), json=payload or {}, headers=headers,
                timeout=_TIMEOUT, **kwargs
            )
        )

    def upload_temp_file(self, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

        def _send(headers):
            with file_path.open("rb") as f:
                return self._httpx.post(
                    self._url("/api/v1/resources/temp_upload"),
                    files={"file": (file_path.name, f, mime_type)},
                    headers=headers,
                    timeout=_TIMEOUT,
                )

        data = self._send_with_trusted_identity_retry(_send, multipart=True)
        result = data.get("result", {})
        temp_file_id = result.get("temp_file_id", "")
        if not temp_file_id:
            raise RuntimeError("OpenViking temp upload did not return temp_file_id")
        return temp_file_id

    def health(self) -> bool:
        try:
            resp = self._httpx.get(
                self._url("/health"), headers=self._headers(), timeout=3.0
            )
            return resp.status_code == 200
        except Exception:
            return False

    def health_payload(self) -> dict:
        resp = self._httpx.get(
            self._url("/health"), headers=self._headers(), timeout=3.0
        )
        return self._parse_response(resp)

    def validate_auth(self) -> dict:
        """Validate authenticated OpenViking access without mutating state."""
        return self.get("/api/v1/system/status")

    def validate_root_access(self) -> dict:
        """Validate ROOT access against a read-only admin endpoint."""
        return self.get("/api/v1/admin/accounts")


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SEARCH_SCHEMA = {
    "name": "viking_search",
    "description": (
        "Semantic search over the OpenViking knowledge base. "
        "Returns ranked results with viking:// URIs for deeper reading. "
        "Use mode='deep' for complex queries that need reasoning across "
        "multiple sources, 'fast' for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {
                "type": "string", "enum": ["auto", "fast", "deep"],
                "description": "Search depth (default: auto).",
            },
            "scope": {
                "type": "string",
                "description": "Viking URI prefix to scope search (e.g. 'viking://resources/docs/').",
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

READ_SCHEMA = {
    "name": "viking_read",
    "description": (
        "Read content at a viking:// URI. Three detail levels:\n"
        "  abstract — ~100 token summary (L0)\n"
        "  overview — ~2k token key points (L1)\n"
        "  full — complete content (L2)\n"
        "Start with abstract/overview, only use full when you need details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "viking:// URI to read."},
            "level": {
                "type": "string", "enum": ["abstract", "overview", "full"],
                "description": "Detail level (default: overview).",
            },
        },
        "required": ["uri"],
    },
}

BROWSE_SCHEMA = {
    "name": "viking_browse",
    "description": (
        "Browse the OpenViking knowledge store like a filesystem.\n"
        "  list — show directory contents\n"
        "  tree — show hierarchy\n"
        "  stat — show metadata for a URI"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["tree", "list", "stat"],
                "description": "Browse action.",
            },
            "path": {
                "type": "string",
                "description": "Viking URI path (default: viking://). Examples: 'viking://resources/', 'viking://user/memories/'.",
            },
        },
        "required": ["action"],
    },
}

REMEMBER_SCHEMA = {
    "name": "viking_remember",
    "description": (
        "Explicitly store a fact or memory in the OpenViking knowledge base. "
        "Use for important information the agent should remember long-term. "
        "The system automatically categorizes and indexes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "entity", "event", "case", "pattern"],
                "description": "Memory category (default: auto-detected).",
            },
        },
        "required": ["content"],
    },
}

ADD_RESOURCE_SCHEMA = {
    "name": "viking_add_resource",
    "description": (
        "Add a remote URL or local file/directory to the OpenViking knowledge base. "
        "Remote resources must be public http(s), git, or ssh URLs. "
        "Local files are uploaded first using OpenViking temp_upload. "
        "The system automatically parses, indexes, and generates summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Remote URL or local file/directory path to add."},
            "reason": {
                "type": "string",
                "description": "Why this resource is relevant (improves search).",
            },
            "to": {
                "type": "string",
                "description": "Optional target viking:// URI for the resource.",
            },
            "parent": {
                "type": "string",
                "description": "Optional parent viking:// URI. Cannot be used with to.",
            },
            "instruction": {
                "type": "string",
                "description": "Optional processing instruction for semantic extraction.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for processing to complete.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds when wait is true.",
            },
        },
        "required": ["url"],
    },
}

WRITE_SCHEMA = {
    "name": "viking_write",
    "description": (
        "Write structured content directly to a viking:// URI. "
        "Use this when you already know where the entry should live "
        "and want to store exact text without waiting for session extraction."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {
                "type": "string",
                "description": "Target viking:// URI to write the content to.",
            },
            "content": {
                "type": "string",
                "description": "The text to persist under the URI.",
            },
            "mode": {
                "type": "string",
                "enum": ["replace", "append"],
                "description": "Write mode: 'replace' (default) or 'append'.",
            },
        },
        "required": ["uri", "content"],
    },
}

LINK_SCHEMA = {
    "name": "viking_link",
    "description": (
        "Create a semantic relation between two OpenViking entries. "
        "This lets the knowledge graph understand how entities, events, and resources connect."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "URI of the source entry (from_uri).",
            },
            "target": {
                "type": "string",
                "description": "URI of the related entry (to_uri).",
            },
            "reason": {
                "type": "string",
                "description": "Why these entries are related (improves graph quality).",
            },
        },
        "required": ["source", "target"],
    },
}

GREP_SCHEMA = {
    "name": "viking_grep",
    "description": (
        "Run OpenViking's exact-match grep search. Use for filenames, error codes, "
        "or phrases the semantic search might miss."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Term or regex pattern to match."},
            "uri": {
                "type": "string",
                "description": "URI prefix to limit the search (default: viking://).",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive match (default: false).",
            },
            "limit": {"type": "integer", "description": "Max entries to return (node_limit)."},
        },
        "required": ["pattern"],
    },
}

GLOB_SCHEMA = {
    "name": "viking_glob",
    "description": (
        "Find OpenViking entries by path/glob patterns. Useful when you know part of "
        "a filename or directory structure."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern like viking://docs/*.md."},
            "uri": {
                "type": "string",
                "description": "Root URI to search under (default: viking://).",
            },
            "limit": {"type": "integer", "description": "Max results (node_limit)."},
        },
        "required": ["pattern"],
    },
}

EXTRACT_SCHEMA = {
    "name": "viking_extract",
    "description": (
        "Force memory extraction on an ongoing OpenViking session so memories become "
        "searchable before the session commits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID to extract (defaults to current session).",
            },
        },
    },
}


# Recall tools (read-only) whose results we never re-ingest into OpenViking —
# echoing recalled memory back into the session transcript would re-store it.
# Write tools (viking_remember / viking_add_resource) are intentionally NOT
# here. Derived from the canonical schema names so renames can't desync.
_OPENVIKING_RECALL_TOOL_NAMES = {
    SEARCH_SCHEMA["name"],
    READ_SCHEMA["name"],
    BROWSE_SCHEMA["name"],
}

# Canonical tool_status values emitted in OpenViking batch tool parts.
_TOOL_STATUS_COMPLETED = "completed"
_TOOL_STATUS_ERROR = "error"
_TOOL_STATUS_PENDING = "pending"
# Inbound status aliases (from varied tool-result shapes) -> canonical above.
_TOOL_STATUS_ERROR_ALIASES = {"error", "failed", "failure"}
_TOOL_STATUS_COMPLETED_ALIASES = {"completed", "complete", "success", "succeeded"}


def _zip_directory(dir_path: Path) -> Path:
    """Create a temporary zip file containing a directory tree."""
    root = dir_path.resolve()
    zip_path = Path(tempfile.gettempdir()) / f"openviking_upload_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_symlink():
                continue
            if file_path.is_file():
                try:
                    file_path.resolve().relative_to(root)
                except ValueError:
                    continue
                arcname = str(file_path.relative_to(dir_path)).replace("\\", "/")
                zipf.write(file_path, arcname=arcname)
    return zip_path


def _is_windows_absolute_path(value: str) -> bool:
    return (
        len(value) >= 3
        and value[0].isalpha()
        and value[1] == ":"
        and value[2] in {"/", "\\"}
    )


def _is_remote_resource_source(value: str) -> bool:
    return value.startswith(_REMOTE_RESOURCE_PREFIXES)


def _is_local_path_reference(value: str) -> bool:
    if not value or "\n" in value or "\r" in value:
        return False
    if _is_remote_resource_source(value):
        return False
    if _is_windows_absolute_path(value):
        return True
    return (
        value.startswith(("/", "./", "../", "~/", ".\\", "..\\", "~\\"))
        or "/" in value
        or "\\" in value
    )


def _path_from_file_uri(uri: str) -> Path | str:
    parsed = urlparse(uri)
    if parsed.netloc not in {"", "localhost"}:
        return f"Unsupported non-local file URI: {uri}"
    return Path(url2pathname(parsed.path)).expanduser()


def _clean_config_value(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _default_ovcli_config_path() -> Path:
    return Path.home() / _OVCLI_DEFAULT_RELATIVE_PATH


def _resolve_ovcli_config_path(config_path: str = "") -> Path:
    env_path = os.environ.get(_OVCLI_CONFIG_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser()
    if config_path:
        return Path(config_path).expanduser()
    return _default_ovcli_config_path()


def _ovcli_config_dir() -> Path:
    return _default_ovcli_config_path().parent


def _load_ovcli_config(path: Optional[Path] = None) -> dict:
    config_path = path or _resolve_ovcli_config_path()
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"OpenViking CLI config must be a JSON object: {config_path}")
    return data


def _connection_values_from_ovcli(data: dict) -> dict:
    api_key = _clean_config_value(data.get("api_key")) or _clean_config_value(data.get("root_api_key"))
    root_api_key = _clean_config_value(data.get("root_api_key"))
    send_identity = not api_key or api_key == root_api_key
    account = _clean_config_value(data.get("account") or data.get("account_id"))
    user = _clean_config_value(data.get("user") or data.get("user_id"))
    return {
        "endpoint": _normalize_openviking_url(data.get("url")),
        "api_key": api_key,
        "root_api_key": root_api_key,
        "account": account if send_identity else "",
        "user": user if send_identity else "",
        "agent": _clean_config_value(data.get("actor_peer_id") or data.get("agent_id")),
    }


def _is_valid_ovcli_profile_name(name: str) -> bool:
    if not name or name.strip() != name or name.startswith("."):
        return False
    if "/" in name or "\\" in name:
        return False
    return all(ch.isascii() and (ch.isalnum() or ch in {"-", "_"}) for ch in name)


def _validate_openviking_identity_value(value: str, *, field: str) -> tuple[bool, str, str]:
    label = "Account ID" if field == "account" else "User ID"
    identifier = "account_id" if field == "account" else "user_id"
    trimmed = value.strip()
    if not trimmed:
        return False, f"{label} cannot be empty.", ""
    if trimmed != value:
        return False, f"{label} cannot start or end with whitespace.", ""
    if field == "account" and trimmed.startswith("_"):
        return False, "Account ID cannot start with '_'.", ""
    if not all(ch.isascii() and (ch.isalnum() or ch in {"_", "-", ".", "@"}) for ch in trimmed):
        return False, f"{label} can only contain letters, numbers, '_', '-', '.', and '@'.", ""
    if trimmed.count("@") > 1:
        return False, f"{identifier} must have at most one '@'.", ""
    return True, "", trimmed


def _normalize_openviking_url(url: str) -> str:
    trimmed = _clean_config_value(url).rstrip("/")
    if not trimmed:
        return _DEFAULT_ENDPOINT
    lower = trimmed.lower()
    if lower in {"::1", "[::1]"}:
        return "http://[::1]:1933"
    if lower.startswith("[::1]:"):
        return f"http://[::1]:{trimmed.rsplit(':', 1)[1]}"
    if lower.startswith("::1:"):
        return f"http://[::1]:{trimmed.rsplit(':', 1)[1]}"
    if "://" in trimmed:
        return trimmed
    host, _sep, port = trimmed.partition(":")
    if host.lower() in {"localhost", "127.0.0.1"}:
        return f"http://{host}:{port or '1933'}"
    return trimmed


def _load_profile(path: Path, *, source: str, name: str) -> Optional[_OvcliProfile]:
    try:
        data = _load_ovcli_config(path)
    except Exception as e:
        logger.debug("Skipping invalid OpenViking CLI config %s: %s", path, e)
        return None
    return _OvcliProfile(
        source=source,
        name=name,
        path=path,
        data=data,
        values=_connection_values_from_ovcli(data),
    )


def _profile_identity(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except OSError:
        return str(path.expanduser())


def _profiles_equivalent(left: _OvcliProfile, right: _OvcliProfile) -> bool:
    return left.values == right.values


def _discover_ovcli_profiles() -> list[_OvcliProfile]:
    profiles: list[_OvcliProfile] = []
    seen_paths: set[str] = set()

    def add(path: Path, *, source: str, name: str) -> None:
        if not path.exists() or not path.is_file():
            return
        identity = _profile_identity(path)
        if identity in seen_paths:
            return
        profile = _load_profile(path, source=source, name=name)
        if profile is None:
            return
        seen_paths.add(identity)
        profiles.append(profile)

    env_path = os.environ.get(_OVCLI_CONFIG_ENV, "").strip()
    if env_path:
        add(Path(env_path).expanduser(), source="env", name=_OVCLI_CONFIG_ENV)

    active_path = _default_ovcli_config_path()
    active_profile = _load_profile(active_path, source="active", name="active") if active_path.exists() else None

    config_dir = _ovcli_config_dir()
    saved_start = len(profiles)
    if config_dir.exists():
        for path in sorted(config_dir.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            name = path.name.removeprefix(_OVCLI_SAVED_PREFIX)
            if name == path.name or name == "bak" or not _is_valid_ovcli_profile_name(name):
                continue
            add(path, source="saved", name=name)

    if active_profile is not None:
        marked_active = False
        for idx in range(saved_start, len(profiles)):
            if profiles[idx].source == "saved" and _profiles_equivalent(profiles[idx], active_profile):
                profiles[idx] = replace(profiles[idx], is_active=True)
                marked_active = True
                break
        has_env_profile = any(profile.source == "env" for profile in profiles)
        has_saved_profile = any(profile.source == "saved" for profile in profiles)
        active_identity = _profile_identity(active_profile.path)
        if not marked_active and not has_env_profile and not has_saved_profile and active_identity not in seen_paths:
            profiles.append(active_profile)

    return profiles


def _is_local_openviking_url(value: str) -> bool:
    candidate = _normalize_openviking_url(value)
    if not candidate:
        return False
    if "://" not in candidate:
        candidate = f"//{candidate}"
    parsed = urlparse(candidate)
    scheme = (parsed.scheme or "http").lower()
    return scheme == "http" and (parsed.hostname or "").lower() in _LOCAL_OPENVIKING_HOSTS


def _load_hermes_openviking_config() -> dict:
    try:
        from hermes_cli.config import load_config

        config = load_config()
        memory_config = config.get("memory", {}) if isinstance(config, dict) else {}
        provider_config = memory_config.get("openviking", {}) if isinstance(memory_config, dict) else {}
        return dict(provider_config) if isinstance(provider_config, dict) else {}
    except Exception:
        return {}


def _env_value(name: str) -> Optional[str]:
    return os.environ[name].strip() if name in os.environ else None


def _first_nonempty(*values: Optional[str], default: str = "") -> str:
    for value in values:
        if value:
            return value
    return default


def _resolve_connection_settings(provider_config: Optional[dict] = None) -> dict:
    provider_config = dict(provider_config or {})
    ovcli_values: dict = {}
    if provider_config.get("use_ovcli_config"):
        ovcli_path = _resolve_ovcli_config_path(str(provider_config.get("ovcli_config_path") or ""))
        ovcli_values = _connection_values_from_ovcli(_load_ovcli_config(ovcli_path))

    endpoint_env = _env_value("OPENVIKING_ENDPOINT")
    api_key_env = _env_value("OPENVIKING_API_KEY")
    account_env = _env_value("OPENVIKING_ACCOUNT")
    user_env = _env_value("OPENVIKING_USER")
    agent_env = _env_value("OPENVIKING_AGENT")

    return {
        "endpoint": _first_nonempty(endpoint_env, ovcli_values.get("endpoint"), default=_DEFAULT_ENDPOINT),
        "api_key": api_key_env if api_key_env is not None else ovcli_values.get("api_key", ""),
        "account": account_env if account_env is not None else ovcli_values.get("account", ""),
        "user": user_env if user_env is not None else ovcli_values.get("user", ""),
        "agent": _first_nonempty(agent_env, ovcli_values.get("agent"), default=_DEFAULT_AGENT),
    }


def _env_writes_from_connection_values(values: dict) -> dict:
    writes = {}
    mapping = {
        "OPENVIKING_ENDPOINT": "endpoint",
        "OPENVIKING_API_KEY": "api_key",
        "OPENVIKING_ACCOUNT": "account",
        "OPENVIKING_USER": "user",
        "OPENVIKING_AGENT": "agent",
    }
    for env_key, value_key in mapping.items():
        value = _clean_config_value(values.get(value_key))
        if value:
            writes[env_key] = value
    return writes


def _restrict_secret_file_permissions(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as e:
        logger.debug("Could not restrict permissions on %s: %s", path, e)


def _precreate_secret_file(path: Path) -> None:
    """Create (or tighten) a secret-bearing file with 0600 BEFORE writing.

    Writing the file first and chmod-ing afterwards leaves a window where a
    freshly-created file is world-readable under the default umask (e.g. 0644),
    briefly exposing the api_key/root_api_key. Pre-creating with 0600 closes
    that window; an existing file is tightened to 0600 here too.
    """
    try:
        if not path.exists():
            os.close(os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600))
        _restrict_secret_file_permissions(path)
    except OSError as e:
        logger.debug("Could not pre-create secret file %s: %s", path, e)


def _write_env_vars(env_path: Path, env_writes: dict, remove_keys: tuple[str, ...] = ()) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    remove_set = set(remove_keys) - set(env_writes)
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_keys = set()
    new_lines = []
    for line in existing_lines:
        key_match = line.split("=", 1)[0].strip() if "=" in line else ""
        if key_match in remove_set:
            continue
        if key_match in env_writes:
            new_lines.append(f"{key_match}={env_writes[key_match]}")
            updated_keys.add(key_match)
        else:
            new_lines.append(line)
    for key, val in env_writes.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")
    # Pre-create with 0600 so secrets are never briefly world-readable.
    _precreate_secret_file(env_path)
    env_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    _restrict_secret_file_permissions(env_path)


def _remember_ovcli_path(provider_config: dict, ovcli_path: Path) -> None:
    default_path = _default_ovcli_config_path().expanduser()
    if os.environ.get(_OVCLI_CONFIG_ENV, "").strip() or ovcli_path.expanduser() != default_path:
        provider_config["ovcli_config_path"] = str(ovcli_path)
    else:
        provider_config.pop("ovcli_config_path", None)


def _ovcli_data_from_connection_values(values: dict) -> dict:
    data = {"url": _normalize_openviking_url(_clean_config_value(values.get("endpoint")) or _DEFAULT_ENDPOINT)}
    api_key = _clean_config_value(values.get("api_key"))
    root_api_key = _clean_config_value(values.get("root_api_key"))
    account = _clean_config_value(values.get("account"))
    user = _clean_config_value(values.get("user"))
    agent = _clean_config_value(values.get("agent")) or _DEFAULT_AGENT
    if api_key:
        data["api_key"] = api_key
    if root_api_key:
        data["root_api_key"] = root_api_key
    if account:
        data["account"] = account
    if user:
        data["user"] = user
    if agent:
        data["actor_peer_id"] = agent
    return data


def _write_ovcli_config(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # atomic_json_write creates the temp file with mode 0o600 and os.replace()s
    # it into place — no half-written config on crash and no chmod-after-write
    # TOCTOU window for the api_key/root_api_key it carries.
    atomic_json_write(path, _ovcli_data_from_connection_values(values), mode=0o600)


def _validate_openviking_reachability(endpoint: str) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(endpoint)
    try:
        client = _VikingClient(endpoint)
        if hasattr(client, "health_payload"):
            payload = client.health_payload()
            if payload.get("healthy") is False:
                return False, "OpenViking server responded but reported unhealthy status."
            if payload:
                return True, ""
        elif client.health():
            return True, ""
    except Exception as e:
        if _status_code_from_error(e) is not None:
            return False, f"OpenViking server responded with {_format_openviking_exception(e)}."
        return False, f"OpenViking server is not reachable at {endpoint}: {_format_openviking_exception(e)}"
    return False, f"OpenViking server is not reachable at {endpoint}."


def _validate_openviking_auth(values: dict) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    try:
        client = _VikingClient(
            endpoint,
            _clean_config_value(values.get("api_key")),
            account=_clean_config_value(values.get("account")),
            user=_clean_config_value(values.get("user")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        client.validate_auth()
    except Exception as e:
        return False, f"OpenViking authentication validation failed: {_format_openviking_exception(e)}"
    return True, ""


def _validate_openviking_root_access(values: dict) -> tuple[bool, str]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    try:
        client = _VikingClient(
            endpoint,
            _clean_config_value(values.get("api_key")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        client.validate_root_access()
    except Exception as e:
        return False, f"OpenViking root API key validation failed: {_format_openviking_exception(e)}"
    return True, ""


def _validate_openviking_user_key_scope(values: dict) -> tuple[bool, str]:
    root_ok, _message = _validate_openviking_root_access(values)
    if not root_ok:
        return True, ""
    return (
        False,
        "That key has ROOT access. Choose Root API key and provide account/user, "
        "or enter a user API key.",
    )


def _status_code_from_error(error: Exception) -> Optional[int]:
    if isinstance(error, _OpenVikingHTTPError):
        return error.status_code
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _admin_probe_means_regular_key(error: Exception) -> bool:
    return _status_code_from_error(error) in {401, 403, 404}


def _should_probe_openviking_auth(health: dict, *, require_api_key: bool, has_api_key: bool) -> bool:
    if require_api_key or has_api_key:
        return True
    auth_mode = health.get("auth_mode")
    if auth_mode == "dev":
        return False
    if auth_mode in {"api_key", "trusted", None}:
        return True
    return False


def _validate_openviking_setup_values(
    values: dict,
    *,
    require_api_key: bool = False,
) -> tuple[bool, str, Optional[str]]:
    endpoint = _normalize_openviking_url(values.get("endpoint"))
    api_key = _clean_config_value(values.get("api_key"))
    if require_api_key and not api_key:
        return False, "Remote OpenViking configs require an API key.", None

    try:
        client = _VikingClient(
            endpoint,
            api_key,
            account=_clean_config_value(values.get("account")),
            user=_clean_config_value(values.get("user")),
            agent=_clean_config_value(values.get("agent")) or _DEFAULT_AGENT,
        )
        health = client.health_payload()
        if health.get("healthy") is False:
            return False, "OpenViking server responded but reported unhealthy status.", None
        if _should_probe_openviking_auth(
            health,
            require_api_key=require_api_key,
            has_api_key=bool(api_key),
        ):
            client.validate_auth()
        if not api_key:
            return True, "", None
        try:
            client.validate_root_access()
            return True, "", "root"
        except Exception as e:
            if _admin_probe_means_regular_key(e):
                return True, "", "user"
            raise
    except Exception as e:
        return False, f"OpenViking validation failed: {_format_openviking_exception(e)}", None


def _retry_or_cancel_manual_setup(select, title: str, message: str, cancelled):
    print(f"  {message}")
    choice = select(
        title,
        [
            ("Retry", "try this step again"),
            ("Cancel setup", "no changes saved"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if choice == 0:
        return True
    return _SETUP_CANCELLED


def _print_validation_progress(message: str) -> None:
    print(f"  {message}", flush=True)


def _local_openviking_bind(endpoint: str) -> tuple[str, int]:
    normalized = _normalize_openviking_url(endpoint)
    parsed = urlparse(normalized)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 1933
    return host, port


def _openviking_server_log_path() -> Path:
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    except Exception:
        home = Path(os.environ.get("HERMES_HOME", "")).expanduser() if os.environ.get("HERMES_HOME") else Path.home() / ".hermes"
    return home / _OPENVIKING_SERVER_LOG_RELATIVE_PATH


def _start_local_openviking_server(endpoint: str) -> tuple[bool, str]:
    server_cmd = shutil.which("openviking-server")
    if not server_cmd:
        return False, "openviking-server was not found on PATH. Start it manually, then retry."
    try:
        host, port = _local_openviking_bind(endpoint)
    except ValueError as e:
        return False, f"Could not parse local OpenViking URL: {e}"
    log_path = _openviking_server_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                [server_cmd, "--host", host, "--port", str(port)],
                stdout=log_file,
                stderr=log_file,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except Exception as e:
        return False, f"Could not start openviking-server: {e}"
    return True, f"Started openviking-server on {host}:{port} in the background. Logs: {log_path}"


def _wait_for_openviking_health(endpoint: str, *, timeout_seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ok, _message = _validate_openviking_reachability(endpoint)
        if ok:
            return True
        time.sleep(0.5)
    return False


def _reachability_failure_allows_local_autostart(message: str) -> bool:
    return not (message or "").startswith(_OPENVIKING_RESPONDED_FAILURE_PREFIX)


def _handle_unreachable_endpoint(
    endpoint: str,
    message: str,
    select,
    cancelled,
    *,
    allow_local_autostart: bool = True,
):
    if _is_local_openviking_url(endpoint) and allow_local_autostart:
        print(f"  {message}")
        choice = select(
            "  Local OpenViking server is down",
            [
                ("Start local OpenViking", "run openviking-server and retry"),
                ("Retry URL", "enter the server URL again"),
                ("Cancel setup", "no changes saved"),
            ],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == 0:
            started, start_message = _start_local_openviking_server(endpoint)
            print(f"  {start_message}")
            if not started:
                return False
            print("  Waiting for OpenViking server to become reachable...", flush=True)
            if _wait_for_openviking_health(
                endpoint,
                timeout_seconds=_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT,
            ):
                print("  OpenViking server is reachable.")
                return True
            print("  OpenViking server did not become reachable.")
            return False
        if choice == 1:
            return False
        return _SETUP_CANCELLED

    return _retry_or_cancel_manual_setup(
        select,
        "  OpenViking server unhealthy" if _is_local_openviking_url(endpoint) else "  OpenViking server unreachable",
        message,
        cancelled,
    )


def _emit_runtime_warning(message: str, warning_callback=None) -> None:
    logger.warning("%s", message)
    if warning_callback:
        try:
            warning_callback(message)
        except Exception:
            logger.debug("OpenViking runtime warning callback failed", exc_info=True)


def _emit_runtime_status(message: str, status_callback=None) -> None:
    logger.info("%s", message)
    if status_callback:
        try:
            status_callback(message)
        except Exception:
            logger.debug("OpenViking runtime status callback failed", exc_info=True)


def _runtime_openviking_timeout_message(endpoint: str) -> str:
    return (
        f"Local OpenViking server at {endpoint} is not reachable. "
        "Tried to start openviking-server, but it did not become reachable "
        f"within {_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT:.0f} seconds. "
        "OpenViking memory disabled for this Hermes run."
    )


def _classify_runtime_openviking_health(client: _VikingClient, endpoint: str) -> tuple[str, str]:
    """Classify runtime health without treating every false result as server absence."""
    try:
        if hasattr(client, "health_payload"):
            payload = client.health_payload()
            if payload.get("healthy") is False:
                return (
                    "responded",
                    f"OpenViking server at {endpoint} responded but reported unhealthy status.",
                )
            return "healthy", ""
        if client.health():
            return "healthy", ""
    except _OpenVikingHTTPError as e:
        return (
            "responded",
            f"OpenViking server at {endpoint} responded with {_format_openviking_exception(e)}.",
        )
    except Exception:
        return "unreachable", ""
    return "unreachable", ""


def _prompt_profile_name(prompt, select, cancelled) -> str | object:
    while True:
        name = _clean_config_value(prompt("OpenViking profile name"))
        if _is_valid_ovcli_profile_name(name):
            return name
        retry = _retry_or_cancel_manual_setup(
            select,
            "  Invalid OpenViking profile name",
            "Profile names can only contain letters, numbers, '-' and '_'.",
            cancelled,
        )
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _confirm_replace_existing_profile(path: Path, values: dict, select, cancelled):
    if not path.exists():
        return True
    try:
        existing_data = _load_ovcli_config(path)
    except Exception:
        existing_data = {}
    if existing_data == _ovcli_data_from_connection_values(values):
        return True
    choice = select(
        "  OpenViking profile already exists",
        [
            ("Choose another name", "leave the existing profile unchanged"),
            ("Replace profile", "overwrite this saved OpenViking profile"),
            ("Cancel setup", "no changes saved"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if choice == 1:
        return True
    if choice == 0:
        return False
    return _SETUP_CANCELLED


def _prompt_manual_connection_values(prompt, select, cancelled, *, service: bool = False):
    if service:
        endpoint = _OPENVIKING_SERVICE_ENDPOINT
        print(f"  OpenViking Service endpoint: {endpoint}")
    else:
        while True:
            endpoint = _normalize_openviking_url(prompt("OpenViking server URL", default=_DEFAULT_ENDPOINT))
            _print_validation_progress("Checking OpenViking server...")
            reachable, message = _validate_openviking_reachability(endpoint)
            if reachable:
                print("  OpenViking server is reachable.")
                break
            retry = _handle_unreachable_endpoint(
                endpoint,
                message,
                select,
                cancelled,
                allow_local_autostart=_reachability_failure_allows_local_autostart(message),
            )
            if retry is True:
                break
            if retry is _SETUP_CANCELLED:
                return _SETUP_CANCELLED

    is_local = _is_local_openviking_url(endpoint)
    api_key_type = "user" if service else ""
    prefilled_api_key = ""
    prefilled_agent = ""
    while True:
        values = {
            "endpoint": endpoint,
            "api_key": "",
            "root_api_key": "",
            "account": "",
            "user": "",
            "agent": "",
        }
        if not api_key_type and is_local:
            credential_choice = select(
                "  OpenViking credential",
                [
                    ("No API key", "local dev mode"),
                    ("User API key", "server derives account/user automatically"),
                    ("Root API key", "requires account and user IDs"),
                ],
                default=0,
                cancel_returns=cancelled,
            )
            if credential_choice == cancelled:
                return _SETUP_CANCELLED
            if credential_choice == 0:
                values["agent"] = _clean_config_value(
                    prompt(_AGENT_PROMPT_LABEL, default=_DEFAULT_AGENT)
                ) or _DEFAULT_AGENT
                _print_validation_progress("Validating OpenViking local dev access...")
                valid, message, _role = _validate_openviking_setup_values(values)
                if valid:
                    print("  OpenViking local dev access validated.")
                    return values
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking credential failed",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            api_key_type = "root" if credential_choice == 2 else "user"
        elif not api_key_type:
            credential_choice = select(
                "  OpenViking API key type",
                [
                    ("User API key", "server derives account/user automatically"),
                    ("Root API key", "requires account and user IDs"),
                ],
                default=0,
                cancel_returns=cancelled,
            )
            if credential_choice == cancelled:
                return _SETUP_CANCELLED
            api_key_type = "root" if credential_choice == 1 else "user"

        values["api_key_type"] = api_key_type
        if service:
            api_key_label = "OpenViking API key"
        else:
            api_key_label = (
                "OpenViking root API key"
                if api_key_type == "root"
                else "OpenViking user API key"
            )
        if prefilled_api_key:
            values["api_key"] = prefilled_api_key
            prefilled_api_key = ""
        else:
            values["api_key"] = _clean_config_value(prompt(api_key_label, secret=True))
        if not values["api_key"]:
            retry = _retry_or_cancel_manual_setup(
                select,
                "  OpenViking API key required",
                f"{api_key_label} is required.",
                cancelled,
            )
            if retry is _SETUP_CANCELLED:
                return _SETUP_CANCELLED
            continue

        if api_key_type == "root":
            _print_validation_progress("Validating OpenViking root API key...")
            valid, message, role = _validate_openviking_setup_values(values, require_api_key=True)
            root_ok = valid and role == "root"
            if not root_ok:
                if valid and role == "user":
                    print("  That key is valid, but it is a user API key.")
                    route_choice = select(
                        "  OpenViking key is a user key",
                        [
                            ("Use as User API key", "server derives account/user automatically"),
                            ("Re-enter Root API key", "try another root key"),
                            ("Cancel setup", "no changes saved"),
                        ],
                        default=0,
                        cancel_returns=cancelled,
                    )
                    if route_choice == 0:
                        prefilled_api_key = values["api_key"]
                        api_key_type = "user"
                        continue
                    if route_choice == 1:
                        api_key_type = "root"
                        continue
                    return _SETUP_CANCELLED
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking root API key failed",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            print("  OpenViking root API key validated.")
            values["root_api_key"] = values["api_key"]
            account_ok, account_message, account = _validate_openviking_identity_value(
                prompt("OpenViking account"),
                field="account",
            )
            user_ok, user_message, user = _validate_openviking_identity_value(
                prompt("OpenViking user"),
                field="user",
            )
            values["account"] = account
            values["user"] = user
            if not account_ok or not user_ok:
                message = account_message if not account_ok else user_message
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking tenant identity required",
                    message,
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                prefilled_api_key = values["api_key"]
                continue

        if prefilled_agent:
            values["agent"] = prefilled_agent
            prefilled_agent = ""
        else:
            values["agent"] = _clean_config_value(
                prompt(_AGENT_PROMPT_LABEL, default=_DEFAULT_AGENT)
            ) or _DEFAULT_AGENT
        _print_validation_progress("Validating OpenViking API access...")
        valid, message, role = _validate_openviking_setup_values(
            values,
            require_api_key=service or not is_local,
        )
        if valid:
            if api_key_type == "user":
                if role == "root":
                    print("  That key is valid, but it has root access.")
                    route_choice = select(
                        "  OpenViking user API key is root key",
                        [
                            ("Configure as Root API key", "provide account and user IDs"),
                            ("Re-enter User API key", "try another user key"),
                            ("Cancel setup", "no changes saved"),
                        ],
                        default=0,
                        cancel_returns=cancelled,
                    )
                    if route_choice == 0:
                        prefilled_api_key = values["api_key"]
                        prefilled_agent = values["agent"]
                        api_key_type = "root"
                        continue
                    if route_choice == 1:
                        api_key_type = "user"
                        continue
                    return _SETUP_CANCELLED
            if api_key_type == "root" and role != "root":
                retry = _retry_or_cancel_manual_setup(
                    select,
                    "  OpenViking root API key failed",
                    "The supplied key was not accepted as a root API key.",
                    cancelled,
                )
                if retry is _SETUP_CANCELLED:
                    return _SETUP_CANCELLED
                continue
            print("  OpenViking API access validated.")
            return values
        retry = _retry_or_cancel_manual_setup(
            select,
            "  OpenViking API access failed",
            message,
            cancelled,
        )
        if retry is _SETUP_CANCELLED:
            return _SETUP_CANCELLED


def _set_openviking_provider(config: dict, provider_config: dict) -> None:
    config["memory"]["provider"] = "openviking"
    config["memory"]["openviking"] = provider_config


def _link_ovcli_profile(
    *,
    config: dict,
    provider_config: dict,
    env_path: Path,
    ovcli_path: Path,
) -> None:
    for key in ("endpoint", "api_key", "root_api_key", "account", "user", "agent", "api_key_type"):
        provider_config.pop(key, None)
    provider_config["use_ovcli_config"] = True
    _remember_ovcli_path(provider_config, ovcli_path)
    _set_openviking_provider(config, provider_config)
    _write_env_vars(env_path, {}, remove_keys=_OPENVIKING_ENV_KEYS)
    for key in _OPENVIKING_ENV_KEYS:
        os.environ.pop(key, None)


def _save_hermes_only_config(
    *,
    config: dict,
    provider_config: dict,
    env_path: Path,
    values: dict,
) -> None:
    provider_config["use_ovcli_config"] = False
    provider_config.pop("ovcli_config_path", None)
    _set_openviking_provider(config, provider_config)
    _write_env_vars(
        env_path,
        _env_writes_from_connection_values(values),
        remove_keys=_OPENVIKING_ENV_KEYS,
    )


def _profile_display_name(profile: _OvcliProfile) -> str:
    if profile.source == "env":
        return _OVCLI_CONFIG_ENV
    if profile.source == "active":
        return "ovcli.conf"
    return profile.name


def _profile_description(profile: _OvcliProfile) -> str:
    endpoint = _clean_config_value(profile.values.get("endpoint")) or _DEFAULT_ENDPOINT
    return f"{endpoint} ({profile.path})"


def _validate_profile_for_setup(profile: _OvcliProfile) -> tuple[bool, str, Optional[str]]:
    require_api_key = not _is_local_openviking_url(profile.values.get("endpoint", ""))
    return _validate_openviking_setup_values(profile.values, require_api_key=require_api_key)


def _print_openviking_ready(message: str, path: Optional[Path] = None) -> None:
    print("\n  OpenViking memory is ready")
    print(f"  {message}")
    if path is not None:
        print(f"  Config file: {path}")
    print("  Start a new Hermes session to activate.\n")


def _run_existing_profile_setup(
    *,
    profiles: list[_OvcliProfile],
    select,
    cancelled,
    config: dict,
    provider_config: dict,
    env_path: Path,
) -> bool | object:
    while True:
        choice = select(
            "  OpenViking profile",
            [(_profile_display_name(profile), _profile_description(profile)) for profile in profiles],
            default=0,
            cancel_returns=cancelled,
        )
        if choice == cancelled:
            return _SETUP_CANCELLED
        if choice < 0 or choice >= len(profiles):
            return _SETUP_CANCELLED

        profile = profiles[choice]
        _print_validation_progress("Validating OpenViking profile...")
        ok, message, _role = _validate_profile_for_setup(profile)
        if ok:
            _link_ovcli_profile(
                config=config,
                provider_config=provider_config,
                env_path=env_path,
                ovcli_path=profile.path,
            )
            _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
            return True

        print(f"  {message}")
        retry = select(
            "  OpenViking profile validation failed",
            [
                ("Choose another profile", "select a different OpenViking profile"),
                ("Retry validation", "try this profile again"),
                ("Cancel setup", "no changes saved"),
            ],
            default=0,
            cancel_returns=cancelled,
        )
        if retry == 0:
            continue
        if retry == 1:
            _print_validation_progress("Validating OpenViking profile...")
            ok, message, _role = _validate_profile_for_setup(profile)
            if ok:
                _link_ovcli_profile(
                    config=config,
                    provider_config=provider_config,
                    env_path=env_path,
                    ovcli_path=profile.path,
                )
                _print_openviking_ready(f"Linked profile: {_profile_display_name(profile)}", profile.path)
                return True
            print(f"  {message}")
            continue
        return _SETUP_CANCELLED


def _mirror_manual_config_to_openviking_store(
    *,
    prompt,
    select,
    cancelled,
    values: dict,
) -> Path | object:
    while True:
        name = _prompt_profile_name(prompt, select, cancelled)
        if name is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        path = _ovcli_config_dir() / f"{_OVCLI_SAVED_PREFIX}{name}"
        replace = _confirm_replace_existing_profile(path, values, select, cancelled)
        if replace is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        if replace is False:
            continue
        _write_ovcli_config(path, values)
        return path


def _run_create_profile_setup(
    *,
    prompt,
    select,
    cancelled,
    config: dict,
    provider_config: dict,
    env_path: Path,
) -> bool | object:
    source_choice = select(
        "  OpenViking connection",
        [
            ("OpenViking Service (VolcEngine Cloud)", "use the managed OpenViking endpoint"),
            ("Custom", "use a local, VPS, or self-hosted OpenViking server"),
        ],
        default=0,
        cancel_returns=cancelled,
    )
    if source_choice == cancelled:
        return _SETUP_CANCELLED

    values = _prompt_manual_connection_values(prompt, select, cancelled, service=(source_choice == 0))
    if values is _SETUP_CANCELLED:
        return _SETUP_CANCELLED
    if values is None:
        return False

    save_choice = select(
        "  Save OpenViking config",
        [
            ("Keep in Hermes only", "write values only to Hermes .env"),
            ("Mirror to OpenViking store", "write ~/.openviking/ovcli.conf.<name> and link it"),
        ],
        default=1,
        cancel_returns=cancelled,
    )
    if save_choice == cancelled:
        return _SETUP_CANCELLED

    if save_choice == 1:
        ovcli_path = _mirror_manual_config_to_openviking_store(
            prompt=prompt,
            select=select,
            cancelled=cancelled,
            values=values,
        )
        if ovcli_path is _SETUP_CANCELLED:
            return _SETUP_CANCELLED
        _link_ovcli_profile(
            config=config,
            provider_config=provider_config,
            env_path=env_path,
            ovcli_path=ovcli_path,
        )
        _print_openviking_ready("Created and linked OpenViking profile.", ovcli_path)
        return True

    _save_hermes_only_config(
        config=config,
        provider_config=provider_config,
        env_path=env_path,
        values=values,
    )
    _print_openviking_ready("Connection saved to Hermes .env.")
    return True


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class OpenVikingMemoryProvider(MemoryProvider):
    """Full bidirectional memory via OpenViking context database."""

    def backup_paths(self) -> List[str]:
        """OpenViking's ovcli config lives at ~/.openviking/ovcli.conf by
        default (or OPENVIKING_CLI_CONFIG_FILE). Capture the resolved file so
        endpoint/api-key survive a backup/import cycle."""
        try:
            cfg = _resolve_ovcli_config_path()
            # The home-scoped guard in the backup walk drops anything outside
            # the user's home; an env override pointing elsewhere is skipped
            # there rather than here.
            return [str(cfg)]
        except Exception:
            return []

    def __init__(self):
        self._client: Optional[_VikingClient] = None
        self._endpoint = ""
        self._api_key = ""
        self._session_id = ""
        self._turn_count = 0
        # Guards the (_session_id, _turn_count) pair. sync_turn runs on the
        # MemoryManager's background sync executor while on_session_end /
        # on_session_switch run on the caller's thread, so the snapshot+reset
        # of the turn counter and the session-id rotation must be atomic
        # against a concurrent increment. See hermes-agent#28296 review.
        self._session_state_lock = threading.Lock()
        # Commit only after session writes drain. The set is keyed by the sid
        # the writer is POSTing under (snapshotted at spawn), so on_session_end
        # / on_session_switch see every still-alive writer for that sid even
        # if later writes have replaced the latest-tracked thread.
        self._inflight_writers: Dict[str, Set[threading.Thread]] = {}
        self._inflight_lock = threading.Lock()
        self._deferred_commit_sids: Set[str] = set()
        self._deferred_commit_threads: Set[threading.Thread] = set()
        self._deferred_commit_lock = threading.Lock()
        self._committed_session_ids: Set[str] = set()
        self._committed_session_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._runtime_start_lock = threading.Lock()
        self._runtime_start_thread: Optional[threading.Thread] = None
        # All prefetch threads ever spawned (daemon, short-lived). Tracked so
        # shutdown() can drain them and rapid re-queues don't orphan a still-
        # running thread by overwriting the single _prefetch_thread slot.
        self._prefetch_threads: Set[threading.Thread] = set()
        # Set on shutdown so deferred-commit / writer finalizers stop issuing
        # network writes against a torn-down provider.
        self._shutting_down = False
        # Drop prefetch results from older switch generations.
        self._prefetch_generation = 0

    @property
    def name(self) -> str:
        return "openviking"

    def is_available(self) -> bool:
        """Check if OpenViking endpoint is configured. No network calls."""
        if os.environ.get("OPENVIKING_ENDPOINT"):
            return True
        provider_config = _load_hermes_openviking_config()
        if not provider_config.get("use_ovcli_config"):
            return False
        try:
            ovcli_path = _resolve_ovcli_config_path(str(provider_config.get("ovcli_config_path") or ""))
            return bool(_connection_values_from_ovcli(_load_ovcli_config(ovcli_path)).get("endpoint"))
        except Exception:
            return False

    def get_config_schema(self):
        return [
            {
                "key": "endpoint",
                "description": "OpenViking server URL",
                "required": True,
                "default": _DEFAULT_ENDPOINT,
                "env_var": "OPENVIKING_ENDPOINT",
            },
            {
                "key": "api_key",
                "description": "OpenViking API key (leave blank for local dev mode)",
                "secret": True,
                "env_var": "OPENVIKING_API_KEY",
            },
            {
                "key": "account",
                "description": "OpenViking tenant account ID (blank for user API keys)",
                "env_var": "OPENVIKING_ACCOUNT",
            },
            {
                "key": "user",
                "description": "OpenViking user ID within the account (blank for user API keys)",
                "env_var": "OPENVIKING_USER",
            },
            {
                "key": "agent",
                "description": (
                    "Hermes peer ID in OpenViking, sent as the actor peer and "
                    "used for peer-scoped memories"
                ),
                "default": "hermes",
                "env_var": "OPENVIKING_AGENT",
            },
        ]

    def get_status_config(self, provider_config: dict) -> dict:
        provider_config = dict(provider_config or {})
        if provider_config.get("use_ovcli_config"):
            ovcli_path = _resolve_ovcli_config_path(str(provider_config.get("ovcli_config_path") or ""))
            try:
                settings = _resolve_connection_settings(provider_config)
            except Exception as e:
                return {
                    "use_ovcli_config": True,
                    "ovcli_config_path": str(ovcli_path),
                    "error": _format_openviking_exception(e),
                }

            display = {
                "use_ovcli_config": True,
                "ovcli_config_path": str(ovcli_path),
                "endpoint": settings.get("endpoint") or _DEFAULT_ENDPOINT,
                "agent": settings.get("agent") or _DEFAULT_AGENT,
            }
            if settings.get("account"):
                display["account"] = settings["account"]
            if settings.get("user"):
                display["user"] = settings["user"]
            env_overrides = [key for key in _OPENVIKING_ENV_KEYS if _env_value(key) is not None]
            if env_overrides:
                display["env_overrides"] = ", ".join(env_overrides)
            return display

        display = dict(provider_config)
        for key in ("api_key", "root_api_key"):
            if key in display:
                display[key] = "(set)"
        return display

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Custom setup that can reuse OpenViking's shared CLI config."""
        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _CANCELLED, _curses_select, _print_cancelled_setup, _prompt

        hermes_home_path = Path(hermes_home)
        env_path = hermes_home_path / ".env"
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        provider_config = config["memory"].get("openviking", {})
        if not isinstance(provider_config, dict):
            provider_config = {}

        print("\n  OpenViking memory setup\n")

        profiles = _discover_ovcli_profiles()
        if profiles:
            setup_options = [
                ("Use existing OpenViking profile", "choose from detected ovcli.conf profiles"),
                ("Create new OpenViking profile", "enter a new URL/API key"),
            ]
            choice = _curses_select(
                "  OpenViking config source",
                setup_options,
                default=0,
                cancel_returns=_CANCELLED,
            )
            if choice == _CANCELLED:
                _print_cancelled_setup()
                return

            if choice == 0:
                result = _run_existing_profile_setup(
                    profiles=profiles,
                    select=_curses_select,
                    cancelled=_CANCELLED,
                    config=config,
                    provider_config=provider_config,
                    env_path=env_path,
                )
                if result is _SETUP_CANCELLED:
                    _print_cancelled_setup()
                    return
                if result:
                    save_config(config)
                return

        else:
            print("  No existing OpenViking CLI profiles found. Creating a new config.")

        result = _run_create_profile_setup(
            prompt=_prompt,
            select=_curses_select,
            cancelled=_CANCELLED,
            config=config,
            provider_config=provider_config,
            env_path=env_path,
        )
        if result is _SETUP_CANCELLED:
            _print_cancelled_setup()
            return
        if result:
            save_config(config)

    def _start_runtime_openviking_waiter(
        self,
        *,
        status_callback=None,
        warning_callback=None,
    ) -> None:
        with self._runtime_start_lock:
            if self._runtime_start_thread and self._runtime_start_thread.is_alive():
                return
            self._runtime_start_thread = threading.Thread(
                target=self._finish_runtime_openviking_start,
                kwargs={
                    "status_callback": status_callback,
                    "warning_callback": warning_callback,
                },
                daemon=True,
                name="openviking-runtime-start",
            )
            self._runtime_start_thread.start()

    def _finish_runtime_openviking_start(
        self,
        *,
        status_callback=None,
        warning_callback=None,
    ) -> None:
        endpoint = self._endpoint
        if not _wait_for_openviking_health(
            endpoint,
            timeout_seconds=_LOCAL_OPENVIKING_AUTOSTART_TIMEOUT,
        ):
            _emit_runtime_warning(
                _runtime_openviking_timeout_message(endpoint),
                warning_callback,
            )
            return

        try:
            client = _VikingClient(
                endpoint,
                self._api_key,
                account=self._account,
                user=self._user,
                agent=self._agent,
            )
            if not client.health():
                _emit_runtime_warning(
                    f"OpenViking server at {endpoint} is still not reachable after auto-start; "
                    "OpenViking memory disabled for this Hermes run.",
                    warning_callback,
                )
                return
        except ImportError:
            logger.warning("httpx not installed — OpenViking plugin disabled")
            return
        except Exception as e:
            _emit_runtime_warning(
                f"OpenViking server at {endpoint} could not be attached after auto-start: {e}. "
                "OpenViking memory disabled for this Hermes run.",
                warning_callback,
            )
            return

        self._client = client
        _emit_runtime_status(
            f"Local OpenViking server at {endpoint} is reachable; OpenViking memory is active for later turns.",
            status_callback,
        )

    def _handle_runtime_openviking_unreachable(
        self,
        *,
        status_callback=None,
        warning_callback=None,
    ) -> None:
        endpoint = self._endpoint
        if not _is_local_openviking_url(endpoint):
            _emit_runtime_warning(
                f"Remote OpenViking server at {endpoint} is not reachable; "
                "OpenViking memory disabled for this Hermes run. "
                "Check the configured endpoint and network connectivity.",
                warning_callback,
            )
            self._client = None
            return

        started, start_message = _start_local_openviking_server(endpoint)
        if not started:
            _emit_runtime_warning(
                f"Local OpenViking server at {endpoint} is not reachable. {start_message} "
                "OpenViking memory disabled for this Hermes run.",
                warning_callback,
            )
            self._client = None
            return

        self._client = None
        _emit_runtime_status(
            f"{start_message} OpenViking memory is starting in the background and will attach when ready.",
            status_callback,
        )
        self._start_runtime_openviking_waiter(
            status_callback=status_callback,
            warning_callback=warning_callback,
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        settings = _resolve_connection_settings(_load_hermes_openviking_config())
        self._endpoint = settings["endpoint"]
        self._api_key = settings["api_key"]
        self._account = settings["account"]
        self._user = settings["user"]
        self._agent = settings["agent"]
        self._session_id = session_id
        self._turn_count = 0
        warning_callback = (
            kwargs.get("warning_callback")
            if kwargs.get("platform") == "cli"
            else None
        )
        status_callback = (
            kwargs.get("status_callback")
            if kwargs.get("platform") == "cli"
            else None
        )

        try:
            self._client = _VikingClient(
                self._endpoint, self._api_key,
                account=self._account, user=self._user, agent=self._agent,
            )
            health_state, health_message = _classify_runtime_openviking_health(self._client, self._endpoint)
            if health_state == "unreachable":
                self._handle_runtime_openviking_unreachable(
                    status_callback=status_callback,
                    warning_callback=warning_callback,
                )
            elif health_state != "healthy":
                _emit_runtime_warning(
                    f"{health_message} OpenViking memory disabled for this Hermes run.",
                    warning_callback,
                )
                self._client = None
        except ImportError:
            logger.warning("httpx not installed — OpenViking plugin disabled")
            self._client = None

        # Register as the last active provider for atexit safety net
        global _last_active_provider
        _last_active_provider = self

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        # Provide brief info about the knowledge base
        try:
            # Check what's in the knowledge base via a root listing
            resp = self._client.get("/api/v1/fs/ls", params={"uri": "viking://"})
            result = resp.get("result", [])
            children = len(result) if isinstance(result, list) else 0
            if children == 0:
                return ""
            return (
                "# OpenViking Knowledge Base\n"
                f"Active. Endpoint: {self._endpoint}\n"
                "Use viking_search to find information, viking_read for details "
                "(abstract/overview/full), viking_browse to explore.\n"
                "Use viking_remember to store facts, viking_add_resource to index URLs/docs."
            )
        except Exception as e:
            logger.warning("OpenViking system_prompt_block failed: %s", e)
            return (
                "# OpenViking Knowledge Base\n"
                f"Active. Endpoint: {self._endpoint}\n"
                "Use viking_search, viking_read, viking_browse, "
                "viking_remember, viking_add_resource."
            )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return prefetched results from the background thread."""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## OpenViking Context\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Fire a background search to pre-load relevant context."""
        query = _derive_openviking_user_text(query)
        if not self._client or not query:
            return

        # Drop prefetch results from older switch generations.
        with self._prefetch_lock:
            gen = self._prefetch_generation

        holder: List[threading.Thread] = []

        def _run():
            try:
                client = _VikingClient(
                    self._endpoint, self._api_key,
                    account=self._account, user=self._user, agent=self._agent,
                )
                resp = client.post("/api/v1/search/find", {
                    "query": query,
                    "limit": 5,
                })
                result = resp.get("result", {})
                parts = []
                for ctx_type in ("memories", "resources"):
                    items = result.get(ctx_type, [])
                    for item in items[:3]:
                        uri = item.get("uri", "")
                        abstract = item.get("abstract", "")
                        score = item.get("score", 0)
                        if abstract:
                            parts.append(f"- [{score:.2f}] {abstract} ({uri})")
                if parts:
                    with self._prefetch_lock:
                        if gen != self._prefetch_generation:
                            return
                        self._prefetch_result = "\n".join(parts)
            except Exception as e:
                logger.debug("OpenViking prefetch failed: %s", e)
            finally:
                with self._prefetch_lock:
                    if holder:
                        self._prefetch_threads.discard(holder[0])

        thread = threading.Thread(
            target=_run, daemon=True, name="openviking-prefetch"
        )
        holder.append(thread)
        with self._prefetch_lock:
            self._prefetch_thread = thread
            self._prefetch_threads.add(thread)
        thread.start()

    def _spawn_writer(self, sid: str, target: Callable[[], None], name: str) -> None:
        """Spawn a daemon writer tracked in _inflight_writers[sid].

        Tracking is keyed by sid (not by a single latest-thread slot) so that
        on_session_end / on_session_switch can drain every still-alive writer
        for the session being committed.
        """
        holder: List[threading.Thread] = []

        def _wrapped():
            try:
                target()
            finally:
                with self._inflight_lock:
                    workers = self._inflight_writers.get(sid)
                    if workers is not None:
                        workers.discard(holder[0])
                        if not workers:
                            self._inflight_writers.pop(sid, None)

        thread = threading.Thread(target=_wrapped, daemon=True, name=name)
        holder.append(thread)
        with self._inflight_lock:
            self._inflight_writers.setdefault(sid, set()).add(thread)
        thread.start()

    def _drain_finalizers(self, timeout: float) -> bool:
        """Join every in-flight async session finalizer within a timeout.

        The switch-path commit runs on a daemon finalizer thread so it never
        blocks the caller's command thread; this lets shutdown and tests wait
        for those commits deterministically. Returns True if all drained.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._deferred_commit_lock:
                workers = [t for t in self._deferred_commit_threads if t.is_alive()]
            if not workers:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            for t in workers:
                slice_left = deadline - time.monotonic()
                if slice_left <= 0:
                    break
                # Floor the per-join wait so a thread whose join() returns
                # instantly while still reporting alive can't hot-spin this loop.
                t.join(timeout=min(slice_left, 0.05))

    def _drain_writers(self, sid: str, timeout: float) -> bool:
        """Join every in-flight writer for sid within a shared timeout budget.

        Returns True if all writers drained, False if any are still alive when
        the budget runs out. Callers use the False return to skip the commit.
        """
        if not sid:
            return True
        deadline = time.monotonic() + timeout
        while True:
            with self._inflight_lock:
                workers = [t for t in self._inflight_writers.get(sid, ()) if t.is_alive()]
            if not workers:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            for t in workers:
                slice_left = deadline - time.monotonic()
                if slice_left <= 0:
                    break
                t.join(timeout=slice_left)

    def _new_client(self) -> _VikingClient:
        return _VikingClient(
            self._endpoint,
            self._api_key,
            account=self._account,
            user=self._user,
            agent=self._agent,
        )

    @staticmethod
    def _text_part(content: str) -> Dict[str, str]:
        return {"type": "text", "text": content}

    def _turn_batch_payload(self, user_content: str, assistant_content: str) -> Dict[str, Any]:
        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "parts": [self._text_part(assistant_content)],
        }
        if self._agent:
            assistant_message["peer_id"] = self._agent
        return {
            "messages": [
                {"role": "user", "parts": [self._text_part(user_content)]},
                assistant_message,
            ]
        }

    def _post_session_turn(
        self,
        client: _VikingClient,
        sid: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        client.post(
            f"/api/v1/sessions/{sid}/messages/batch",
            self._turn_batch_payload(user_content, assistant_content),
        )

    def _session_has_pending_tokens(self, sid: str) -> bool:
        try:
            response = self._client.get(f"/api/v1/sessions/{sid}")
        except Exception:
            return False
        session = self._unwrap_result(response)
        if not isinstance(session, dict):
            return False
        try:
            return int(session.get("pending_tokens") or 0) > 0
        except (TypeError, ValueError):
            return False

    def _has_committed_session(self, sid: str) -> bool:
        with self._committed_session_lock:
            return sid in self._committed_session_ids

    def _mark_session_committed(self, sid: str) -> None:
        with self._committed_session_lock:
            self._committed_session_ids.add(sid)

    def _session_needs_commit(self, sid: str, turn_count: int) -> bool:
        # Already-committed sessions never need a second commit, regardless of
        # the turn counter — a racing sync_turn can re-increment _turn_count
        # after a commit+reset, so the committed-guard must win over turn_count.
        if self._has_committed_session(sid):
            return False
        if turn_count > 0:
            return True
        return self._session_has_pending_tokens(sid)

    def _commit_session(self, sid: str, turn_count: int, *, context: str) -> bool:
        try:
            self._client.post(
                f"/api/v1/sessions/{sid}/commit",
                {"keep_recent_count": 0},
            )
            self._mark_session_committed(sid)
            logger.info("OpenViking session %s committed %s (%d turns)", sid, context, turn_count)
            return True
        except Exception as e:
            logger.warning("OpenViking session commit failed for %s: %s", sid, e)
            return False

    def _finalize_session_async(self, sid: str, turn_count: int, *, context: str) -> None:
        """Drain the old session's writers and commit it on a daemon thread.

        Used by on_session_switch (and the deferred-commit fallback) so the
        potentially-multi-second drain + pending-token GET + commit POST never
        runs on the caller's command thread. Deduped by sid so a rapid second
        switch can't stack two finalizers for the same session, and a no-op
        once shutdown has begun so we don't POST against a torn-down client.
        """
        if not sid:
            return
        with self._deferred_commit_lock:
            if self._shutting_down or sid in self._deferred_commit_sids:
                return
            self._deferred_commit_sids.add(sid)

        holder: List[threading.Thread] = []

        def _finalize() -> None:
            try:
                if self._shutting_down:
                    return
                if not self._drain_writers(sid, timeout=_DEFERRED_COMMIT_TIMEOUT):
                    logger.warning(
                        "OpenViking writer for %s still alive after drain — "
                        "leaving session uncommitted",
                        sid,
                    )
                    return
                if self._shutting_down:
                    return
                if self._session_needs_commit(sid, turn_count):
                    self._commit_session(sid, turn_count, context=context)
            finally:
                with self._deferred_commit_lock:
                    self._deferred_commit_sids.discard(sid)
                    if holder:
                        self._deferred_commit_threads.discard(holder[0])

        thread = threading.Thread(
            target=_finalize,
            daemon=True,
            name=f"openviking-finalize-{sid}",
        )
        holder.append(thread)
        with self._deferred_commit_lock:
            self._deferred_commit_threads.add(thread)
        thread.start()

    def _invalidate_prefetch_state(self) -> None:
        # Bump the generation under the same lock used by prefetch workers so
        # late results from an older session are discarded deterministically.
        with self._prefetch_lock:
            self._prefetch_generation += 1
            self._prefetch_result = ""
            # Join EVERY tracked prefetch thread, not just the latest slot — a
            # rapid re-queue can leave an older thread for the abandoned session
            # still running (consistent with shutdown()).
            workers = [t for t in self._prefetch_threads if t.is_alive()]
        for t in workers:
            t.join(timeout=3.0)
        with self._prefetch_lock:
            self._prefetch_result = ""

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

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Record the conversation turn in OpenViking's session (non-blocking)."""
        if not self._client:
            return

        user_content = _derive_openviking_user_text(user_content)
        if not user_content:
            return

        turn_messages = (
            self._extract_current_turn_messages(messages, user_content, assistant_content)
            if messages is not None
            else []
        )
        if turn_messages:
            turn_messages = [dict(message) for message in turn_messages]
            for message in turn_messages:
                if message.get("role") == "user":
                    message["content"] = user_content
                    break
        batch_messages = self._messages_to_openviking_batch(
            turn_messages,
            assistant_peer_id=getattr(self, "_agent", _DEFAULT_AGENT),
        )

        if _sync_trace_enabled():
            logger.info(
                "OpenViking sync_turn trace: session_arg=%r cached_session=%r "
                "messages_param_supported=true messages_present=%s message_count=%s "
                "turn_message_count=%d batch_message_count=%d user_len=%d assistant_len=%d "
                "user_preview=%r assistant_preview=%r",
                session_id,
                self._session_id,
                messages is not None,
                len(messages) if messages is not None else None,
                len(turn_messages),
                len(batch_messages),
                len(str(user_content or "")),
                len(str(assistant_content or "")),
                _preview(user_content),
                _preview(assistant_content),
            )

        # Snapshot the sid and bump the turn counter atomically so a
        # concurrent on_session_switch/on_session_end can't interleave its
        # snapshot+reset between the read and the increment (lost turn) and so
        # the turn is unambiguously attributed to the session it targets.
        with self._session_state_lock:
            sid = str(session_id or self._session_id).strip()
            if not sid:
                return
            self._turn_count += 1

        def _sync():
            def _post_turn(client: _VikingClient) -> None:
                if batch_messages:
                    payload = {"messages": batch_messages}
                    if _sync_trace_enabled():
                        logger.info(
                            "OpenViking sync_turn trace: POST /api/v1/sessions/%s/messages/batch payload=%s",
                            sid,
                            json.dumps(payload, ensure_ascii=False),
                        )
                    try:
                        client.post(f"/api/v1/sessions/{sid}/messages/batch", payload)
                        return
                    except Exception as batch_error:
                        logger.warning(
                            "OpenViking structured sync failed; falling back to text sync: %s",
                            batch_error,
                        )

                self._post_session_turn(
                    client,
                    sid,
                    user_content[:4000],
                    self._message_text(assistant_content)[:4000],
                )

            try:
                client = self._new_client()
                _post_turn(client)
            except Exception as e:
                logger.debug("OpenViking sync_turn failed, reconnecting: %s", e)
                try:
                    client = self._new_client()
                    _post_turn(client)
                except Exception as retry_error:
                    logger.warning("OpenViking sync_turn failed: %s", retry_error)

        self._spawn_writer(sid, _sync, name="openviking-sync")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Commit the session to trigger memory extraction.

        OpenViking automatically extracts 6 categories of memories:
        profile, preferences, entities, events, cases, and patterns.
        """
        if not self._client:
            return

        # Snapshot sid + turn count atomically against a concurrent sync_turn
        # increment. on_session_end runs at teardown so the drain+commit stays
        # synchronous here (we want it to land before the process exits), but
        # the counter read must still be consistent.
        with self._session_state_lock:
            sid = self._session_id
            turn_count = self._turn_count

        # Commit only after session writes drain.
        if not self._drain_writers(sid, timeout=_SESSION_DRAIN_TIMEOUT):
            logger.warning(
                "OpenViking writer for %s still alive after drain — skipping commit",
                sid,
            )
            return

        if not self._session_needs_commit(sid, turn_count):
            return

        if self._commit_session(sid, turn_count, context="on session end"):
            # Mark clean so a follow-up on_session_switch skips its own commit.
            with self._session_state_lock:
                if self._session_id == sid:
                    self._turn_count = 0

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Commit the old session and rotate cached state to the new session_id.

        Fires on /resume, /branch, /reset, /new, and context compression.
        Without this hook, ``_session_id`` stays stuck at the value
        ``initialize()`` cached, so subsequent ``sync_turn()`` writes land in
        the already-closed old session and ``on_session_end()`` tries to
        commit it a second time. The new session never accumulates messages,
        and memory extraction never fires for it. See hermes-agent#28296.

        Flushes any in-flight sync under the old session_id, commits the old
        session if it has pending turns (same extraction semantics as
        ``on_session_end``), drains and clears any stale prefetch result,
        then rotates ``_session_id`` and resets ``_turn_count``.
        """
        new_id = str(new_session_id or "").strip()
        if not new_id or not self._client:
            return

        rewound = bool(kwargs.get("rewound"))

        # Rotate cached session state synchronously (cheap, in-memory) and
        # snapshot the old session under the lock so a concurrent sync_turn
        # either lands fully before the rotation (counted under old) or fully
        # after (counted under new) — never split. The OLD session's commit
        # (drain + pending-token GET + commit POST, potentially many seconds)
        # is then offloaded so /new, /branch, /resume, /undo never block the
        # caller's command thread (cf. the end-of-turn-sync offload in #41945).
        with self._session_state_lock:
            old_session_id = self._session_id
            old_turn_count = self._turn_count
            rotate = not (rewound or new_id == old_session_id)
            if rotate:
                self._session_id = new_id
                self._turn_count = 0

        # Invalidate stale prefetch OUTSIDE the session lock — it takes its own
        # _prefetch_lock and may join a prefetch thread for up to 3s, which we
        # must not do while holding the session lock (would block sync_turn and
        # risk lock-ordering coupling).
        self._invalidate_prefetch_state()

        if not rotate:
            # Same-session rewind (/undo) or no-op rotation: no commit, no
            # counter reset — just the prefetch invalidation above.
            logger.debug(
                "OpenViking on_session_switch invalidated state without rotation: "
                "session=%s rewound=%s",
                old_session_id, rewound,
            )
            return

        # Drain + commit the OLD session off the command thread.
        if old_session_id:
            self._finalize_session_async(old_session_id, old_turn_count, context="on switch")

        logger.debug(
            "OpenViking on_session_switch: old=%s new=%s parent=%s reset=%s",
            old_session_id, new_id, parent_session_id, reset,
        )

    def _build_memory_uri(self, subdir: str) -> str:
        """Build a viking:// memory URI under the configured peer namespace."""
        slug = uuid.uuid4().hex[:12]
        return f"viking://user/peers/{self._agent}/memories/{subdir}/mem_{slug}.md"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to OpenViking via content/write."""
        if not self._client or action != "add" or not content:
            return

        subdir = _MEMORY_WRITE_TARGET_SUBDIR_MAP.get(target, _DEFAULT_MEMORY_SUBDIR)
        uri = self._build_memory_uri(subdir)

        def _write():
            try:
                client = _VikingClient(
                    self._endpoint, self._api_key,
                    account=self._account, user=self._user, agent=self._agent,
                )
                client.post("/api/v1/content/write", {
                    "uri": uri,
                    "content": content,
                    "mode": "create",
                })
            except Exception as e:
                logger.debug("OpenViking memory mirror failed: %s", e)

        t = threading.Thread(target=_write, daemon=True, name="openviking-memwrite")
        t.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            SEARCH_SCHEMA,
            READ_SCHEMA,
            BROWSE_SCHEMA,
            REMEMBER_SCHEMA,
            ADD_RESOURCE_SCHEMA,
            WRITE_SCHEMA,
            LINK_SCHEMA,
            GREP_SCHEMA,
            GLOB_SCHEMA,
            EXTRACT_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if not self._client:
            return tool_error("OpenViking server not connected")

        try:
            if tool_name == "viking_search":
                return self._tool_search(args)
            elif tool_name == "viking_read":
                return self._tool_read(args)
            elif tool_name == "viking_browse":
                return self._tool_browse(args)
            elif tool_name == "viking_remember":
                return self._tool_remember(args)
            elif tool_name == "viking_add_resource":
                return self._tool_add_resource(args)
            elif tool_name == "viking_write":
                return self._tool_write(args)
            elif tool_name == "viking_link":
                return self._tool_link(args)
            elif tool_name == "viking_grep":
                return self._tool_grep(args)
            elif tool_name == "viking_glob":
                return self._tool_glob(args)
            elif tool_name == "viking_extract":
                return self._tool_extract(args)
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def shutdown(self) -> None:
        # Stop deferred finalizers from issuing new commits against a
        # torn-down client, then drain everything still in flight.
        self._shutting_down = True
        # Wait for every in-flight writer across all tracked sessions.
        with self._inflight_lock:
            all_workers = [
                t for workers in self._inflight_writers.values() for t in workers
            ]
        with self._deferred_commit_lock:
            deferred_workers = list(self._deferred_commit_threads)
        with self._prefetch_lock:
            prefetch_workers = list(self._prefetch_threads)
        for t in all_workers:
            if t.is_alive():
                t.join(timeout=5.0)
        for t in deferred_workers:
            if t.is_alive():
                t.join(timeout=5.0)
        for t in prefetch_workers:
            if t.is_alive():
                t.join(timeout=5.0)
        # Clear atexit reference so it doesn't double-commit.
        global _last_active_provider
        if _last_active_provider is self:
            _last_active_provider = None

    # -- Tool implementations ------------------------------------------------

    @staticmethod
    def _unwrap_result(resp: Any) -> Any:
        """Return OpenViking payload body regardless of wrapped/unwrapped shape."""
        if isinstance(resp, dict) and "result" in resp:
            return resp.get("result")
        return resp

    @staticmethod
    def _normalize_summary_uri(uri: str) -> str:
        """Map pseudo summary files to their parent directory URI for L0/L1 reads."""
        if not uri:
            return uri
        for suffix in ("/.abstract.md", "/.overview.md", "/.read.md", "/.full.md"):
            if uri.endswith(suffix):
                return uri[: -len(suffix)] or "viking://"
        return uri

    def _is_directory_uri(self, uri: str) -> bool | None:
        """Probe fs/stat to decide if a URI is a directory.

        Returns True/False when the server answers cleanly, and None when the
        probe itself fails (network error, unexpected shape). Callers should
        treat None as "unknown" and fall back to the exception-based path.
        """
        try:
            resp = self._client.get("/api/v1/fs/stat", params={"uri": uri})
        except Exception:
            return None
        result = self._unwrap_result(resp)
        if isinstance(result, dict):
            if "isDir" in result:
                return bool(result.get("isDir"))
            if "is_dir" in result:
                return bool(result.get("is_dir"))
            if result.get("type") == "dir":
                return True
            if result.get("type") == "file":
                return False
        return None

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        payload: Dict[str, Any] = {"query": query}
        mode = args.get("mode", "auto")
        if args.get("scope"):
            payload["target_uri"] = args["scope"]
        if args.get("limit"):
            payload["limit"] = args["limit"]

        endpoint = "/api/v1/search/search" if mode == "deep" else "/api/v1/search/find"
        if endpoint == "/api/v1/search/search" and self._session_id:
            payload["session_id"] = self._session_id

        resp = self._client.post(endpoint, payload)
        result = resp.get("result", {})

        # Format results for the model — keep it concise
        scored_entries = []
        for ctx_type in ("memories", "resources", "skills"):
            items = result.get(ctx_type, [])
            for item in items:
                raw_score = item.get("score")
                sort_score = raw_score if raw_score is not None else 0.0
                entry = {
                    "uri": item.get("uri", ""),
                    "type": ctx_type.rstrip("s"),
                    "score": round(raw_score, 3) if raw_score is not None else 0.0,
                    "abstract": item.get("abstract", ""),
                }
                if item.get("relations"):
                    entry["related"] = [r.get("uri") for r in item["relations"][:3]]
                scored_entries.append((sort_score, entry))

        scored_entries.sort(key=lambda x: x[0], reverse=True)
        formatted = [entry for _, entry in scored_entries]

        return json.dumps({
            "results": formatted,
            "total": result.get("total", len(formatted)),
        }, ensure_ascii=False)

    def _tool_read(self, args: dict) -> str:
        uri = args.get("uri", "")
        if not uri:
            return tool_error("uri is required")

        level = args.get("level", "overview")

        summary_level = level in {"abstract", "overview"}
        # OpenViking expects directory URIs for pseudo summary files
        # (e.g. viking://user/hermes/.overview.md).
        resolved_uri = self._normalize_summary_uri(uri) if summary_level else uri
        used_fallback = False

        # abstract/overview endpoints are directory-only on OpenViking
        # (v0.3.x returns 500/412 for file URIs). When the caller asks for a
        # summary level on a non-pseudo URI, probe fs/stat first and route
        # file URIs straight to /content/read instead of eating a failing
        # round-trip. The pseudo-URI path already points at a directory, so
        # skip the probe there.
        if summary_level and resolved_uri == uri:
            is_dir = self._is_directory_uri(uri)
            if is_dir is False:
                resolved_uri = uri
                used_fallback = True

        # Map our level names to OpenViking GET endpoints.
        endpoint = "/api/v1/content/read"
        if not used_fallback:
            if level == "abstract":
                endpoint = "/api/v1/content/abstract"
            elif level == "overview":
                endpoint = "/api/v1/content/overview"

        try:
            resp = self._client.get(endpoint, params={"uri": resolved_uri})
        except Exception:
            # OpenViking may return HTTP 500 for abstract/overview reads on normal
            # file URIs (mem_*.md). For those, gracefully fallback to full read.
            if not summary_level or resolved_uri != uri or used_fallback:
                raise
            resp = self._client.get("/api/v1/content/read", params={"uri": uri})
            used_fallback = True

        result = self._unwrap_result(resp)
        # Content endpoints may return either plain strings or objects.
        if isinstance(result, str):
            content = result
        elif isinstance(result, dict):
            content = result.get("content", "") or result.get("text", "")
        else:
            content = ""

        # Truncate long content to avoid flooding context.
        max_len = 8000
        if level == "overview":
            max_len = 4000
        elif level == "abstract":
            max_len = 1200

        if len(content) > max_len:
            content = content[:max_len] + "\n\n[... truncated, use a more specific URI or full level]"

        payload = {
            "uri": uri,
            "resolved_uri": resolved_uri,
            "level": level,
            "content": content,
        }
        if used_fallback:
            payload["fallback"] = "content/read"

        return json.dumps(payload, ensure_ascii=False)

    def _tool_browse(self, args: dict) -> str:
        action = args.get("action", "list")
        path = args.get("path", "viking://")

        # Map action to the correct fs endpoint (all GET with uri= param)
        endpoint_map = {"tree": "/api/v1/fs/tree", "list": "/api/v1/fs/ls", "stat": "/api/v1/fs/stat"}
        endpoint = endpoint_map.get(action, "/api/v1/fs/ls")
        resp = self._client.get(endpoint, params={"uri": path})
        result = self._unwrap_result(resp)

        # Format list/tree results for readability
        if action in {"list", "tree"}:
            raw_entries = result
            if isinstance(result, dict):
                raw_entries = result.get("entries") or result.get("items") or result.get("children") or []

            if isinstance(raw_entries, list):
                entries = []
                for e in raw_entries[:50]:  # cap at 50 entries
                    uri = e.get("uri", "")
                    name = e.get("rel_path") or e.get("name") or (uri.rsplit("/", 1)[-1] if uri else "")
                    is_dir = bool(e.get("isDir") or e.get("is_dir") or e.get("type") == "dir")
                    entries.append({
                        "name": name,
                        "uri": uri,
                        "type": "dir" if is_dir else "file",
                        "abstract": e.get("abstract", ""),
                    })
                return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False)

    def _tool_remember(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        category = args.get("category", "")
        subdir = _CATEGORY_SUBDIR_MAP.get(category, _DEFAULT_MEMORY_SUBDIR)
        uri = self._build_memory_uri(subdir)

        # Write directly via content/write API.
        # This creates the file, stores the content, and queues vector indexing
        # in a single call — no dependency on session commit / VLM extraction.
        try:
            result = self._client.post("/api/v1/content/write", {
                "uri": uri,
                "content": content,
                "mode": "create",
            })
            written = result.get("result", {}).get("written_bytes", 0)
            return json.dumps({
                "status": "stored",
                "message": f"Memory stored ({written}b) and queued for vector indexing.",
            })
        except Exception as e:
            logger.error("OpenViking content/write failed: %s", e)
            return tool_error(f"Failed to store memory: {e}")

    def _tool_add_resource(self, args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return tool_error("url is required")

        if args.get("to") and args.get("parent"):
            return tool_error("Cannot specify both 'to' and 'parent'")

        payload: Dict[str, Any] = {}
        for key in ("reason", "to", "parent", "instruction", "wait", "timeout"):
            if key in args and args[key] not in {None, ""}:
                payload[key] = args[key]

        parsed_url = urlparse(url)
        if _is_remote_resource_source(url):
            source_path = None
        elif parsed_url.scheme == "file":
            source_path = _path_from_file_uri(url)
            if isinstance(source_path, str):
                return tool_error(source_path)
        elif parsed_url.scheme and not _is_windows_absolute_path(url):
            source_path = None
        else:
            source_path = Path(url).expanduser()

        cleanup_path: Optional[Path] = None
        try:
            if source_path is not None:
                if source_path.exists():
                    if source_path.is_dir():
                        payload["source_name"] = source_path.name
                        cleanup_path = _zip_directory(source_path)
                        upload_path = cleanup_path
                    elif source_path.is_file():
                        payload["source_name"] = source_path.name
                        upload_path = source_path
                    else:
                        return tool_error(f"Unsupported local resource path: {url}")
                    payload["temp_file_id"] = self._client.upload_temp_file(upload_path)
                elif _is_local_path_reference(url):
                    return tool_error(f"Local resource path does not exist: {url}")
                else:
                    payload["path"] = url
            else:
                payload["path"] = url

            resp = self._client.post("/api/v1/resources", payload)
            result = resp.get("result", {})
        finally:
            if cleanup_path:
                cleanup_path.unlink(missing_ok=True)

        return json.dumps({
            "status": "added",
            "root_uri": result.get("root_uri", ""),
            "message": "Resource queued for processing. Use viking_search after a moment to find it.",
        }, ensure_ascii=False)

    def _tool_write(self, args: dict) -> str:
        # POST /api/v1/content/write — WriteContentRequest (extra="forbid")
        # Accepted fields: uri, content, mode, wait, timeout, telemetry
        uri = args.get("uri", "")
        content = args.get("content", "")
        if not uri or not content:
            return tool_error("uri and content are required")

        payload: Dict[str, Any] = {"uri": uri, "content": content}
        if args.get("mode"):
            payload["mode"] = args["mode"]

        resp = self._client.post("/api/v1/content/write", payload)
        result = resp.get("result", {})

        return json.dumps({
            "status": "written",
            "uri": uri,
            "result": result,
        }, ensure_ascii=False)

    def _tool_link(self, args: dict) -> str:
        # POST /api/v1/relations/link — LinkRequest: from_uri, to_uris, reason
        source = args.get("source", "")
        target = args.get("target", "")
        if not source or not target:
            return tool_error("source and target are required")

        payload: Dict[str, Any] = {
            "from_uri": source,
            "to_uris": target,
        }
        if args.get("reason"):
            payload["reason"] = args["reason"]

        resp = self._client.post("/api/v1/relations/link", payload)
        data = resp.get("result", {})

        return json.dumps({
            "status": "linked",
            "source": source,
            "target": target,
            "details": data,
        }, ensure_ascii=False)

    def _tool_grep(self, args: dict) -> str:
        # POST /api/v1/search/grep — GrepRequest: uri (required), pattern (required),
        # case_insensitive, node_limit, level_limit
        pattern = args.get("pattern", "")
        if not pattern:
            return tool_error("pattern is required")

        payload: Dict[str, Any] = {
            "uri": args.get("uri", "viking://"),
            "pattern": pattern,
        }
        if args.get("case_insensitive"):
            payload["case_insensitive"] = args["case_insensitive"]
        if args.get("limit"):
            payload["node_limit"] = args["limit"]

        resp = self._client.post("/api/v1/search/grep", payload)
        result = resp.get("result", [])

        formatted = []
        for item in result:
            formatted.append({
                "uri": item.get("uri"),
                "snippet": item.get("snippet"),
                "line": item.get("line"),
            })

        return json.dumps({"results": formatted}, ensure_ascii=False)

    def _tool_glob(self, args: dict) -> str:
        # POST /api/v1/search/glob — GlobRequest: pattern (required), uri, node_limit
        pattern = args.get("pattern", "")
        if not pattern:
            return tool_error("pattern is required")

        payload: Dict[str, Any] = {"pattern": pattern}
        if args.get("uri"):
            payload["uri"] = args["uri"]
        if args.get("limit"):
            payload["node_limit"] = args["limit"]

        resp = self._client.post("/api/v1/search/glob", payload)
        result = resp.get("result", [])

        entries = [{"uri": e.get("uri"), "name": e.get("name")} for e in result]
        return json.dumps({"entries": entries}, ensure_ascii=False)

    def _tool_extract(self, args: dict) -> str:
        # POST /api/v1/sessions/{session_id}/extract — no request body
        session_id = args.get("session_id", self._session_id)
        if not session_id:
            return tool_error("session_id is required")

        resp = self._client.post(f"/api/v1/sessions/{session_id}/extract")
        result = resp.get("result", {})

        return json.dumps({
            "status": "extracted",
            "session_id": session_id,
            "details": result,
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register OpenViking as a memory provider plugin."""
    ctx.register_memory_provider(OpenVikingMemoryProvider())
