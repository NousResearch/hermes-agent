"""Helpers for long-tail gateway command surfaces.

This module keeps config/env editing, session export, and value formatting
out of ``gateway.run`` so the command handlers stay readable.
"""

from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from hermes_cli.config import (
    get_config_path,
    get_env_path,
    get_env_value,
    load_config,
    save_config,
    save_env_value,
)


_ENV_LIKE_EXPLICIT_KEYS = {
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "VOICE_TOOLS_OPENAI_KEY",
    "PARALLEL_API_KEY",
    "FIRECRAWL_API_KEY",
    "FIRECRAWL_API_URL",
    "TAVILY_API_KEY",
    "BROWSERBASE_API_KEY",
    "BROWSERBASE_PROJECT_ID",
    "BROWSER_USE_API_KEY",
    "FAL_KEY",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "TERMINAL_SSH_HOST",
    "TERMINAL_SSH_USER",
    "TERMINAL_SSH_KEY",
    "SUDO_PASSWORD",
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "GITHUB_TOKEN",
    "HONCHO_API_KEY",
    "WANDB_API_KEY",
    "TINKER_API_KEY",
}
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def is_env_like_key(key: str) -> bool:
    """Return True when ``key`` should be treated as an env var path."""
    normalized = str(key or "").strip().upper()
    if not normalized:
        return False
    if normalized in _ENV_LIKE_EXPLICIT_KEYS:
        return True
    if normalized.endswith("_API_KEY") or normalized.endswith("_TOKEN"):
        return True
    if normalized.startswith("TERMINAL_SSH"):
        return True
    return bool(_ENV_NAME_RE.match(normalized))


def parse_value_text(raw: str) -> Any:
    """Parse a user value using YAML semantics, falling back to a string."""
    text = str(raw or "").strip()
    if not text:
        return ""
    try:
        value = yaml.safe_load(text)
    except Exception:
        return text
    return text if value is None else value


def load_raw_user_config() -> dict[str, Any]:
    """Load the on-disk user config without default merging."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def save_raw_user_config(config: dict[str, Any]) -> Path:
    """Persist the raw user config to the canonical config path."""
    save_config(config)
    return get_config_path()


def get_nested_value(mapping: dict[str, Any], path: str) -> Any:
    """Resolve a dotted path from a nested dict."""
    current: Any = mapping
    for part in str(path or "").split("."):
        if not part:
            continue
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def set_nested_value(mapping: dict[str, Any], path: str, value: Any) -> Any:
    """Set a dotted path in a nested dict, creating parent dicts as needed."""
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        raise KeyError(path)

    current = mapping
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value
    return value


def unset_nested_value(mapping: dict[str, Any], path: str) -> bool:
    """Delete a dotted path from a nested dict. Returns True if removed."""
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return False
    current = mapping
    parents: list[tuple[dict[str, Any], str]] = []
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current or not isinstance(current[part], dict):
            return False
        parents.append((current, part))
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        return False
    current.pop(parts[-1], None)
    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break
    return True


def read_config_or_env_value(key: str) -> tuple[str, Any]:
    """Return (kind, value) where kind is ``env`` or ``config``."""
    if is_env_like_key(key):
        return "env", get_env_value(key.upper())
    return "config", get_nested_value(load_raw_user_config(), key)


def write_config_or_env_value(key: str, raw_value: str) -> tuple[str, Any, Path]:
    """Write a config or env value and return (kind, stored_value, path)."""
    if is_env_like_key(key):
        normalized = key.upper()
        save_env_value(normalized, str(raw_value))
        return "env", str(raw_value), get_env_path()

    config = load_raw_user_config()
    value = parse_value_text(raw_value)
    set_nested_value(config, key, value)
    path = save_raw_user_config(config)
    return "config", value, path


def unset_config_or_env_value(key: str) -> tuple[str, bool, Path]:
    """Unset a config or env value and return (kind, removed, path)."""
    if is_env_like_key(key):
        return "env", unset_env_key(key.upper()), get_env_path()

    config = load_raw_user_config()
    removed = unset_nested_value(config, key)
    if removed:
        save_raw_user_config(config)
    return "config", removed, get_config_path()


def unset_env_key(key: str) -> bool:
    """Remove an environment variable from ``~/.hermes/.env`` and ``os.environ``."""
    env_path = get_env_path()
    if not env_path.exists():
        os.environ.pop(key, None)
        return False

    with open(env_path, encoding="utf-8") as handle:
        lines = handle.readlines()
    kept = [
        line
        for line in lines
        if not line.strip().startswith(f"{key}=")
    ]
    if kept == lines:
        os.environ.pop(key, None)
        return False

    with open(env_path, "w", encoding="utf-8") as handle:
        handle.writelines(kept)
    os.environ.pop(key, None)
    return True


def format_yaml_block(value: Any) -> str:
    """Render ``value`` as a YAML code block."""
    rendered = yaml.safe_dump(value, sort_keys=False, allow_unicode=True).strip()
    return f"```yaml\n{rendered or 'null'}\n```"


def default_export_path(session_id: str, cwd: str | Path | None, suffix: str = ".html") -> Path:
    """Build the default export path inside ``cwd``."""
    base_dir = Path(cwd or os.getcwd())
    export_dir = base_dir / ".hermes-exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_session_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id or "session").strip("-") or "session"
    return export_dir / f"{safe_session_id}{suffix}"


def resolve_export_path(raw_path: str, cwd: str | Path | None, *, session_id: str) -> Path:
    """Resolve a user path relative to ``cwd`` or fall back to a default export path."""
    candidate = str(raw_path or "").strip()
    if not candidate:
        return default_export_path(session_id, cwd, ".html")
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path(cwd or os.getcwd()) / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_session_export_html(snapshot: dict[str, Any]) -> str:
    """Render an HTML session export with metadata, prompt, and transcript."""
    meta = snapshot.get("session") or {}
    prompt = snapshot.get("context_prompt") or ""
    transcript = snapshot.get("messages") or []

    message_rows: list[str] = []
    for message in transcript:
        role = html.escape(str(message.get("role") or "unknown"))
        content = html.escape(str(message.get("content") or ""))
        message_rows.append(
            "<article class='message'>"
            f"<h3>{role}</h3>"
            f"<pre>{content}</pre>"
            "</article>"
        )

    meta_json = html.escape(json.dumps(meta, ensure_ascii=False, indent=2))
    prompt_html = html.escape(prompt)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hermes Session Export</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 2rem; background: #f5f4ef; color: #1d1d1b; }}
    h1, h2 {{ margin-bottom: 0.5rem; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fff; padding: 1rem; border-radius: 12px; border: 1px solid #ddd5c6; }}
    .message {{ margin: 1rem 0; }}
    .message h3 {{ margin-bottom: 0.35rem; text-transform: uppercase; font-size: 0.85rem; letter-spacing: 0.06em; color: #6d5d3d; }}
  </style>
</head>
<body>
  <h1>Hermes Session Export</h1>
  <h2>Session Metadata</h2>
  <pre>{meta_json}</pre>
  <h2>Current Session Context Prompt</h2>
  <pre>{prompt_html}</pre>
  <h2>Transcript</h2>
  {''.join(message_rows) or '<p>No transcript messages available.</p>'}
</body>
</html>
"""

