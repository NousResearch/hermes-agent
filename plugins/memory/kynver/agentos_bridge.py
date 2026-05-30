"""Runtime-agnostic Kynver AgentOS REST bridge.

Hermes is only an adapter here: Kynver owns durable operating state, while
Hermes keeps local machine-control tools. This bridge intentionally exposes a
small JSON transport that other runtimes can reuse.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

_DEFAULT_API_URL = "https://www.kynver.com"
_DEFAULT_AGENT_OS_SLUG = "ghost"
_DEFAULT_TIMEOUT_SECONDS = 3.0
_DEFAULT_SIDE_EFFECT_TIMEOUT_SECONDS = 3.0
_VALID_MODES = {"enabled", "observe", "disabled"}


class KynverAgentOSError(RuntimeError):
    """Raised for redacted, user-safe Kynver AgentOS failures."""


@dataclass(frozen=True)
class KynverAgentOSConfig:
    api_url: str = _DEFAULT_API_URL
    api_key: str = ""
    slug: str = _DEFAULT_AGENT_OS_SLUG
    timeout: float = _DEFAULT_TIMEOUT_SECONDS
    side_effect_timeout: float = _DEFAULT_SIDE_EFFECT_TIMEOUT_SECONDS
    mode: str = "enabled"
    tasks_disabled: bool = False
    skills_disabled: bool = False
    session_sync_disabled: bool = False
    todo_mirror_disabled: bool = False
    memory_disabled: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.api_url and self.api_key and self.slug)

    @property
    def enabled(self) -> bool:
        return self.configured and self.mode != "disabled"

    @property
    def observe_only(self) -> bool:
        return self.mode == "observe"


def _active_env_path() -> Path:
    return get_hermes_home() / ".env"


def _load_profile_env(path: Path | None = None) -> dict[str, str]:
    """Load active profile ``.env`` without mutating ``os.environ``."""
    env_path = path or _active_env_path()
    out: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return out
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            out[key] = value.strip().strip("\"'")
    return out


def _env_bool_disabled(env: Mapping[str, str], name: str) -> bool:
    raw = str(env.get(name) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_timeout_seconds(
    env: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0.25, float(raw) / 1000.0)
    except ValueError:
        return default


def load_kynver_agentos_config(
    env: Mapping[str, str] | None = None,
) -> KynverAgentOSConfig:
    """Load Kynver config from profile env, overridden by process env."""
    merged: dict[str, str] = dict(_load_profile_env())
    merged.update(dict(env or os.environ))

    mode = (merged.get("KYNVER_AGENTOS_MODE") or "enabled").strip().lower()
    if mode not in _VALID_MODES:
        mode = "enabled"

    timeout = _env_timeout_seconds(merged, "KYNVER_FETCH_TIMEOUT_MS", _DEFAULT_TIMEOUT_SECONDS)
    side_effect_timeout = _env_timeout_seconds(
        merged,
        "KYNVER_OBSERVER_TIMEOUT_MS",
        _DEFAULT_SIDE_EFFECT_TIMEOUT_SECONDS,
    )

    return KynverAgentOSConfig(
        api_url=(merged.get("KYNVER_API_URL") or _DEFAULT_API_URL).strip().rstrip("/"),
        api_key=(merged.get("KYNVER_API_KEY") or "").strip(),
        slug=(merged.get("KYNVER_AGENT_OS_SLUG") or _DEFAULT_AGENT_OS_SLUG).strip(),
        timeout=timeout,
        side_effect_timeout=side_effect_timeout,
        mode=mode,
        tasks_disabled=_env_bool_disabled(merged, "KYNVER_TASKS_DISABLED"),
        skills_disabled=_env_bool_disabled(merged, "KYNVER_SKILLS_DISABLED"),
        session_sync_disabled=_env_bool_disabled(merged, "KYNVER_SESSION_SYNC_DISABLED"),
        todo_mirror_disabled=_env_bool_disabled(merged, "KYNVER_TODO_MIRROR_DISABLED"),
        memory_disabled=_env_bool_disabled(merged, "KYNVER_MEMORY_DISABLED"),
    )


def agentos_configured(env: Mapping[str, str] | None = None) -> bool:
    return load_kynver_agentos_config(env).configured


def agentos_enabled(env: Mapping[str, str] | None = None) -> bool:
    return load_kynver_agentos_config(env).enabled


def redact(text: str) -> str:
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", str(text))
    redacted = re.sub(
        r"(?im)^(\s*(?:authorization|x-api-key)\s*:\s*).+$",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)([\"'](?:api[-_]?key|apikey|token|secret|password|authorization|x-api-key)[\"']\s*:\s*[\"'])([^\"']+)([\"'])",
        r"\1[REDACTED]\3",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(api[_-]?key|apikey|token|secret|password)=([^\s&\"']+)",
        r"\1=[REDACTED]",
        redacted,
    )
    return redacted[:2000]


def quote_path_segment(value: str) -> str:
    return urllib.parse.quote(str(value or "").strip(), safe="")


class KynverAgentOSClient:
    """Small stdlib-only REST client for AgentOS endpoints."""

    def __init__(self, config: KynverAgentOSConfig | None = None):
        self.config = config or load_kynver_agentos_config()

    def api_path(self, path: str, *, slug: str | None = None) -> str:
        if "\x00" in path or ".." in path.split("?")[0].split("/"):
            raise KynverAgentOSError("Invalid AgentOS API path")
        clean = path if path.startswith("/") else f"/{path}"
        if clean.startswith("/api/"):
            return clean
        if clean.startswith("/agent-os/"):
            return f"/api{clean}"
        effective_slug = quote_path_segment(slug or self.config.slug or _DEFAULT_AGENT_OS_SLUG)
        return f"/api/agent-os/{effective_slug}{clean}"

    def request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
        *,
        slug: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        if not self.config.enabled:
            raise KynverAgentOSError(
                "Kynver AgentOS is not configured or is disabled. Set "
                "KYNVER_API_KEY, KYNVER_AGENT_OS_SLUG, and optionally "
                "KYNVER_AGENTOS_MODE=enabled."
            )
        url = f"{self.config.api_url}{self.api_path(path, slug=slug)}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "hermes-kynver-agentos/1.0",
            },
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.config.timeout) as res:  # nosec B310
                payload = res.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise KynverAgentOSError(redact(f"Kynver AgentOS HTTP {exc.code}: {detail}")) from exc
        except Exception as exc:
            raise KynverAgentOSError(redact(f"Kynver AgentOS request failed: {exc}")) from exc
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return payload

    def get(self, path: str, *, slug: str | None = None, timeout: float | None = None) -> Any:
        return self.request("GET", path, slug=slug, timeout=timeout)

    def post(self, path: str, body: Any, *, slug: str | None = None, timeout: float | None = None) -> Any:
        return self.request("POST", path, body, slug=slug, timeout=timeout)

    def patch(self, path: str, body: Any, *, slug: str | None = None, timeout: float | None = None) -> Any:
        return self.request("PATCH", path, body, slug=slug, timeout=timeout)

    def delete(self, path: str, *, slug: str | None = None, timeout: float | None = None) -> Any:
        return self.request("DELETE", path, slug=slug, timeout=timeout)


def agentos_available(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the active profile has enough Kynver config for calls."""

    return agentos_enabled(env)


def probe_agentos_health(client: KynverAgentOSClient | None = None) -> bool:
    """Lightweight health check via GET ``/stats`` (same route as MCP context tool)."""

    try:
        probe = client or KynverAgentOSClient()
        if not probe.config.enabled:
            return False
        probe.get("/stats")
        return True
    except Exception:
        return False
