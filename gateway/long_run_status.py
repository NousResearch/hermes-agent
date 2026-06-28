"""Optional enrichment for long-running gateway status messages.

This module is deliberately fail-soft: callers already have a deterministic
heartbeat. Enrichment may make that heartbeat more readable, but it must never
be required for chat liveness.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_hermes_home


_STATUS_LINE_RE = re.compile(r"^[\w\s,./:;()+#\-]+$")
_SECRET_WORD_RE = re.compile(r"\b(token|secret|password|api[_ -]?key|bearer|sk-[A-Za-z0-9])\b", re.I)
_ALLOWED_STATUS_LABELS = (
    "running tests",
    "waiting on network",
    "reading files",
    "editing code",
    "using terminal",
    "using browser",
    "checking logs",
    "verifying results",
    "working",
)


@dataclass(frozen=True)
class LongRunStatusEnrichmentConfig:
    enabled: bool = False
    local_enabled: bool = True
    local_url: str = "http://127.0.0.1:11434"
    local_model: str = "qwen2.5:0.5b"
    local_timeout_seconds: float = 1.5
    local_keep_alive: str = "0s"
    cloud_enabled: bool = True
    cloud_provider: str = "openrouter"
    cloud_model: str = "deepseek/deepseek-v4-flash"
    cloud_timeout_seconds: float = 3.0
    cloud_max_calls_per_day: int = 20


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def config_from_gateway_config(user_config: Mapping[str, Any] | None) -> LongRunStatusEnrichmentConfig:
    """Resolve enrichment config from ``agent.gateway_status_enrichment``.

    Unknown/malformed values fall back to safe defaults. Environment variables
    are intentionally not used here; this is a per-gateway UX knob, not a secret.
    """
    agent_cfg = (user_config or {}).get("agent") if isinstance(user_config, Mapping) else None
    raw = agent_cfg.get("gateway_status_enrichment") if isinstance(agent_cfg, Mapping) else None
    if not isinstance(raw, Mapping):
        raw = {}
    raw_local = raw.get("local")
    raw_cloud = raw.get("cloud")
    local: Mapping[str, Any] = raw_local if isinstance(raw_local, Mapping) else {}
    cloud: Mapping[str, Any] = raw_cloud if isinstance(raw_cloud, Mapping) else {}
    return LongRunStatusEnrichmentConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        local_enabled=_as_bool(local.get("enabled"), True),
        local_url=str(local.get("url") or "http://127.0.0.1:11434").rstrip("/"),
        local_model=str(local.get("model") or "qwen2.5:0.5b"),
        local_timeout_seconds=max(0.1, _as_float(local.get("timeout_seconds"), 1.5)),
        local_keep_alive=str(local.get("keep_alive") or "0s"),
        cloud_enabled=_as_bool(cloud.get("enabled"), True),
        cloud_provider=str(cloud.get("provider") or "openrouter"),
        cloud_model=str(cloud.get("model") or "deepseek/deepseek-v4-flash"),
        cloud_timeout_seconds=max(0.1, _as_float(cloud.get("timeout_seconds"), 3.0)),
        cloud_max_calls_per_day=max(0, _as_int(cloud.get("max_calls_per_day"), 20)),
    )


def sanitize_activity_snapshot(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only non-sensitive telemetry for model-based status wording."""
    src = snapshot or {}
    out: dict[str, Any] = {}
    for key in (
        "elapsed_seconds",
        "api_call_count",
        "max_iterations",
        "current_tool",
        "last_activity_desc",
        "platform",
    ):
        value = src.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            out[key] = value
        else:
            text = str(value).strip()
            # Keep status generic. Do not ship arbitrary tool args/output.
            if len(text) > 80:
                text = text[:77] + "..."
            if _SECRET_WORD_RE.search(text):
                continue
            out[key] = text
    return out


def normalize_status_line(text: Any) -> Optional[str]:
    """Validate and normalize one model-produced status phrase."""
    raw = str(text or "")
    if "\n" in raw or "\r" in raw:
        return None
    line = raw.strip().strip('"').strip("'")
    if not line:
        return None
    line = re.sub(r"\s+", " ", line)
    if len(line) > 96:
        line = line[:93].rstrip() + "..."
    if _SECRET_WORD_RE.search(line):
        return None
    if not _STATUS_LINE_RE.match(line):
        return None
    return line.rstrip(".")


def normalize_status_label(text: Any) -> Optional[str]:
    """Reduce model output to one approved status label.

    Small local models may embellish even when instructed not to. Treat their
    output as a fuzzy classifier, not prose: accept only a known label, or a
    sentence containing exactly one known label.
    """
    line = normalize_status_line(text)
    if not line:
        return None
    lowered = line.lower()
    if "test" in lowered or "pytest" in lowered:
        return "running tests"
    if "network" in lowered or "http" in lowered or "api" in lowered:
        return "waiting on network"
    if "log" in lowered or "journal" in lowered:
        return "checking logs"
    if "verify" in lowered or "checking" in lowered:
        return "verifying results"
    if "read" in lowered or "file" in lowered:
        return "reading files"
    if "edit" in lowered or "patch" in lowered or "code" in lowered:
        return "editing code"
    if "terminal" in lowered or "shell" in lowered:
        return "using terminal"
    if "browser" in lowered:
        return "using browser"
    matches = [label for label in _ALLOWED_STATUS_LABELS if label in lowered]
    if len(matches) == 1:
        return matches[0]
    if lowered in _ALLOWED_STATUS_LABELS:
        return lowered
    return None


def _prompt_for(snapshot: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"))
    return [
        {
            "role": "system",
            "content": (
                "Choose exactly one status label from this list: "
                "running tests, waiting on network, reading files, editing code, "
                "using terminal, using browser, checking logs, verifying results, working. "
                "Return only the label. Do not add details."
            ),
        },
        {"role": "user", "content": payload},
    ]


def _ollama_chat_sync(config: LongRunStatusEnrichmentConfig, snapshot: Mapping[str, Any]) -> Optional[str]:
    body = {
        "model": config.local_model,
        "messages": _prompt_for(snapshot),
        "stream": False,
        "keep_alive": config.local_keep_alive,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 24,
            "num_ctx": 512,
        },
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{config.local_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.local_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None
    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return normalize_status_label(content)


def _openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    # Gateway normally loads ~/.hermes/.env, but standalone tests/smokes may not.
    env_path = get_hermes_home() / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


def _state_path() -> Path:
    return get_hermes_home() / "run-state" / "long-run-status-enrichment.json"


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.localtime())


def _cloud_budget_available(config: LongRunStatusEnrichmentConfig) -> bool:
    if config.cloud_max_calls_per_day <= 0:
        return False
    path = _state_path()
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {}
    today = _today()
    if state.get("date") != today:
        return True
    return int(state.get("cloud_calls", 0) or 0) < config.cloud_max_calls_per_day


def _record_cloud_call() -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    today = _today()
    try:
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {}
    if state.get("date") != today:
        state = {"date": today, "cloud_calls": 0}
    state["cloud_calls"] = int(state.get("cloud_calls", 0) or 0) + 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _openrouter_chat_sync(config: LongRunStatusEnrichmentConfig, snapshot: Mapping[str, Any]) -> Optional[str]:
    if config.cloud_provider != "openrouter":
        return None
    if not _cloud_budget_available(config):
        return None
    key = _openrouter_key()
    if not key:
        return None
    body = {
        "model": config.cloud_model,
        "messages": _prompt_for(snapshot),
        "temperature": 0,
        "max_tokens": 32,
        "reasoning": {"enabled": False},
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://hermes-agent.local/long-run-status",
            "X-Title": "Hermes long-run status enrichment",
        },
        method="POST",
    )
    _record_cloud_call()
    try:
        with urllib.request.urlopen(req, timeout=config.cloud_timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
        return None
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        return None
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return normalize_status_label(content)


async def enrich_long_run_status(
    snapshot: Mapping[str, Any] | None,
    config: LongRunStatusEnrichmentConfig,
) -> Optional[str]:
    """Return one optional enriched status phrase.

    Local Ollama is tried first. Cheap cloud fallback is only tried if local is
    unavailable/fails and the daily call budget allows it. All inputs are
    sanitized run metadata, never user transcript or tool output.
    """
    if not config.enabled:
        return None
    clean = sanitize_activity_snapshot(snapshot)
    if not clean:
        return None
    if config.local_enabled:
        try:
            local = await asyncio.wait_for(
                asyncio.to_thread(_ollama_chat_sync, config, clean),
                timeout=config.local_timeout_seconds + 0.2,
            )
        except (asyncio.TimeoutError, OSError):
            local = None
        if local:
            return local
    if config.cloud_enabled:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_openrouter_chat_sync, config, clean),
                timeout=config.cloud_timeout_seconds + 0.2,
            )
        except (asyncio.TimeoutError, OSError):
            return None
    return None
