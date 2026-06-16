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
  OPENVIKING_ACCOUNT   — Optional tenant account override
  OPENVIKING_USER      — Optional tenant user override
  OPENVIKING_AGENT     — Tenant agent (default: hermes)

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
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_OPENVIKING_SERVICE_ENDPOINT = "https://api.vikingdb.cn-beijing.volces.com/openviking"
_DEFAULT_ACCOUNT = ""
_DEFAULT_USER = ""
_DEFAULT_AGENT = "hermes"
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
_REMOTE_RESOURCE_PREFIXES = ("http://", "https://", "git@", "ssh://", "git://")

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
        self._account = account if account is not None else os.environ.get("OPENVIKING_ACCOUNT", _DEFAULT_ACCOUNT)
        self._user = user if user is not None else os.environ.get("OPENVIKING_USER", _DEFAULT_USER)
        self._agent = agent if agent is not None else os.environ.get("OPENVIKING_AGENT", _DEFAULT_AGENT)
        self._httpx = _get_httpx()
        if self._httpx is None:
            raise ImportError("httpx is required for OpenViking: pip install httpx")

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._agent:
            h["X-OpenViking-Actor-Peer"] = self._agent
            h["X-OpenViking-Agent"] = self._agent
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

    def _multipart_headers(self) -> dict:
        headers = self._headers()
        headers.pop("Content-Type", None)
        return headers

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
        resp = self._httpx.get(
            self._url(path), headers=self._headers(), timeout=_TIMEOUT, **kwargs
        )
        return self._parse_response(resp)

    def post(self, path: str, payload: dict = None, **kwargs) -> dict:
        resp = self._httpx.post(
            self._url(path), json=payload or {}, headers=self._headers(),
            timeout=_TIMEOUT, **kwargs
        )
        return self._parse_response(resp)

    def upload_temp_file(self, file_path: Path) -> str:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with file_path.open("rb") as f:
            resp = self._httpx.post(
                self._url("/api/v1/resources/temp_upload"),
                files={"file": (file_path.name, f, mime_type)},
                headers=self._multipart_headers(),
                timeout=_TIMEOUT,
            )
        data = self._parse_response(resp)
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
    return (parsed.hostname or "").lower() in _LOCAL_OPENVIKING_HOSTS


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
    path.write_text(json.dumps(_ovcli_data_from_connection_values(values), indent=2) + "\n", encoding="utf-8")
    _restrict_secret_file_permissions(path)


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


def _start_local_openviking_server(endpoint: str) -> tuple[bool, str]:
    server_cmd = shutil.which("openviking-server")
    if not server_cmd:
        return False, "openviking-server was not found on PATH. Start it manually, then retry."
    try:
        host, port = _local_openviking_bind(endpoint)
    except ValueError as e:
        return False, f"Could not parse local OpenViking URL: {e}"
    try:
        subprocess.Popen(
            [server_cmd, "--host", host, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as e:
        return False, f"Could not start openviking-server: {e}"
    return True, f"Started openviking-server on {host}:{port} in the background."


def _wait_for_openviking_health(endpoint: str, *, timeout_seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ok, _message = _validate_openviking_reachability(endpoint)
        if ok:
            return True
        time.sleep(0.5)
    return False


def _handle_unreachable_endpoint(endpoint: str, message: str, select, cancelled):
    if _is_local_openviking_url(endpoint):
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
        "  OpenViking server unreachable",
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
            retry = _handle_unreachable_endpoint(endpoint, message, select, cancelled)
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
                    prompt("OpenViking agent", default=_DEFAULT_AGENT)
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
                prompt("OpenViking agent", default=_DEFAULT_AGENT)
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

    def __init__(self):
        self._client: Optional[_VikingClient] = None
        self._endpoint = ""
        self._api_key = ""
        self._session_id = ""
        self._turn_count = 0
        self._sync_thread: Optional[threading.Thread] = None
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._runtime_start_lock = threading.Lock()
        self._runtime_start_thread: Optional[threading.Thread] = None

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
                "description": "OpenViking agent ID within the account ([hermes], useful in multi-agent mode)",
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
            if not self._client.health():
                self._handle_runtime_openviking_unreachable(
                    status_callback=status_callback,
                    warning_callback=warning_callback,
                )
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
        if not self._client or not query:
            return

        def _run():
            try:
                client = _VikingClient(
                    self._endpoint, self._api_key,
                    account=self._account, user=self._user, agent=self._agent,
                )
                resp = client.post("/api/v1/search/find", {
                    "query": query,
                    "top_k": 5,
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
                        self._prefetch_result = "\n".join(parts)
            except Exception as e:
                logger.debug("OpenViking prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(
            target=_run, daemon=True, name="openviking-prefetch"
        )
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Record the conversation turn in OpenViking's session (non-blocking)."""
        if not self._client:
            return

        self._turn_count += 1

        def _sync():
            try:
                client = _VikingClient(
                    self._endpoint, self._api_key,
                    account=self._account, user=self._user, agent=self._agent,
                )
                sid = self._session_id

                # Add user message
                client.post(f"/api/v1/sessions/{sid}/messages", {
                    "role": "user",
                    "content": user_content[:4000],  # trim very long messages
                })
                # Add assistant message
                client.post(f"/api/v1/sessions/{sid}/messages", {
                    "role": "assistant",
                    "content": assistant_content[:4000],
                })
            except Exception as e:
                logger.debug("OpenViking sync_turn failed: %s", e)

        # Wait for any previous sync to finish before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(
            target=_sync, daemon=True, name="openviking-sync"
        )
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Commit the session to trigger memory extraction.

        OpenViking automatically extracts 6 categories of memories:
        profile, preferences, entities, events, cases, and patterns.
        """
        if not self._client:
            return

        # Wait for any pending sync to finish first — do this before the
        # turn_count check so the last turn's messages are flushed even if
        # the count hasn't been incremented yet.
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)

        if self._turn_count == 0:
            return

        try:
            self._client.post(f"/api/v1/sessions/{self._session_id}/commit")
            logger.info("OpenViking session %s committed (%d turns)", self._session_id, self._turn_count)
        except Exception as e:
            logger.warning("OpenViking session commit failed: %s", e)

    def _build_memory_uri(self, subdir: str) -> str:
        """Build a viking:// memory URI under the configured user/agent/subdir."""
        slug = uuid.uuid4().hex[:12]
        return f"viking://user/{self._user}/agent/{self._agent}/memories/{subdir}/mem_{slug}.md"

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
        return [SEARCH_SCHEMA, READ_SCHEMA, BROWSE_SCHEMA, REMEMBER_SCHEMA, ADD_RESOURCE_SCHEMA]

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
            return tool_error(f"Unknown tool: {tool_name}")
        except Exception as e:
            return tool_error(str(e))

    def shutdown(self) -> None:
        # Wait for background threads to finish
        for t in (self._sync_thread, self._prefetch_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        # Clear atexit reference so it doesn't double-commit
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
        if mode != "auto":
            payload["mode"] = mode
        if args.get("scope"):
            payload["target_uri"] = args["scope"]
        if args.get("limit"):
            payload["top_k"] = args["limit"]

        resp = self._client.post("/api/v1/search/find", payload)
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


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register OpenViking as a memory provider plugin."""
    ctx.register_memory_provider(OpenVikingMemoryProvider())
