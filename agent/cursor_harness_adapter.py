"""Hermes provider adapter for the external Hermes Cursor Harness plugin.

The harness remains an independently installable plugin/package.  This module
is the thin Hermes-core route that lets normal provider selection delegate a
turn to Cursor while Hermes keeps session ownership, callbacks, and transcript
persistence.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


CURSOR_HARNESS_PROVIDER = "cursor-harness"
CURSOR_HARNESS_API_MODE = "cursor_harness"
CURSOR_MODEL_PREFIX = "cursor/"


class CursorHarnessUnavailable(RuntimeError):
    """Raised when the provider route is selected but the harness is missing."""


def is_cursor_harness_route(
    *,
    provider: str | None = None,
    api_mode: str | None = None,
    model: str | None = None,
) -> bool:
    provider_norm = (provider or "").strip().lower()
    mode_norm = (api_mode or "").strip().lower()
    model_norm = (model or "").strip().lower()
    return (
        provider_norm in {CURSOR_HARNESS_PROVIDER, "cursor"}
        or mode_norm == CURSOR_HARNESS_API_MODE
        or model_norm.startswith(CURSOR_MODEL_PREFIX)
    )


def cursor_model_from_hermes(model: str | None) -> str | None:
    """Map Hermes model ids such as ``cursor/composer-2`` to Cursor ids."""

    raw = (model or "").strip()
    if not raw:
        return None
    if raw.lower().startswith(CURSOR_MODEL_PREFIX):
        raw = raw.split("/", 1)[1].strip()
    if raw.lower() in {"", "default"}:
        return None
    return raw


def render_cursor_prompt(
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    current_turn_user_idx: int | None = None,
    user_injections: Iterable[str] = (),
    hermes_session_id: str | None = None,
    model: str | None = None,
    platform: str | None = None,
) -> str:
    """Render Hermes chat state into a single Cursor-agent prompt."""

    sections: list[str] = []
    header: dict[str, str] = {}
    if hermes_session_id:
        header["hermes_session_id"] = hermes_session_id
    if model:
        header["hermes_model"] = model
    if platform:
        header["hermes_platform"] = platform
    if header:
        sections.append("<hermes_turn_metadata>\n" + json.dumps(header, indent=2, sort_keys=True) + "\n</hermes_turn_metadata>")

    if system_prompt and str(system_prompt).strip():
        sections.append("<hermes_system_context>\n" + str(system_prompt).strip() + "\n</hermes_system_context>")

    rendered_messages: list[str] = []
    injection_text = "\n\n".join(str(item).strip() for item in user_injections if str(item).strip())
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown").strip() or "unknown"
        content = _message_text(msg)
        if idx == current_turn_user_idx and role == "user" and injection_text:
            content = (content + "\n\n" + injection_text).strip()
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            content = (content + "\n\nTool calls:\n" + _stable_json(tool_calls)).strip()
        if content:
            rendered_messages.append(f"{role}:\n{content}")

    if rendered_messages:
        sections.append("<hermes_conversation>\n" + "\n\n---\n\n".join(rendered_messages) + "\n</hermes_conversation>")

    sections.append(
        "You are Cursor Agent running inside a Hermes-owned session. "
        "Use the repository as the source of truth, follow the current user request, "
        "and return the final answer for Hermes to deliver."
    )
    return "\n\n".join(sections).strip()


def run_cursor_harness_provider_turn(
    *,
    prompt: str,
    hermes_session_id: str,
    model: str | None = None,
    project: str | None = None,
    mode: str | None = None,
    transport: str | None = None,
    timeout_sec: float | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run one provider-routed Cursor turn through the installed harness."""

    _ensure_harness_importable()
    from hermes_cursor_harness.config import load_config
    from hermes_cursor_harness.harness import run_turn
    from hermes_cursor_harness.store import HarnessStore

    cfg = load_config()
    store = HarnessStore(cfg.state_dir)
    return run_turn(
        cfg=cfg,
        store=store,
        project=project or _default_project(),
        prompt=prompt,
        mode=mode or _env("HERMES_CURSOR_HARNESS_MODE"),
        model=cursor_model_from_hermes(model),
        hermes_session_id=hermes_session_id,
        new_session=False,
        transport=transport or _env("HERMES_CURSOR_HARNESS_TRANSPORT"),
        timeout_sec=timeout_sec,
        event_callback=event_callback,
    )


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    text = _content_text(content).strip()
    if msg.get("tool_name"):
        text = f"{msg.get('tool_name')}: {text}".strip()
    return text


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "result"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return _stable_json(content)
    return str(content)


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _default_project() -> str:
    return _env("HERMES_CURSOR_PROJECT") or _env("HERMES_CURSOR_HARNESS_PROJECT") or os.getcwd()


def _ensure_harness_importable() -> None:
    try:
        import hermes_cursor_harness  # noqa: F401
        return
    except Exception:
        pass

    for candidate in _candidate_harness_paths():
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            try:
                import hermes_cursor_harness  # noqa: F401
                return
            except Exception:
                try:
                    sys.path.remove(str(candidate))
                except ValueError:
                    pass

    raise CursorHarnessUnavailable(
        "The cursor-harness provider requires the hermes-cursor-harness plugin. "
        "Install it into ~/.hermes/plugins/hermes-cursor-harness or install the "
        "package on PYTHONPATH."
    )


def _candidate_harness_paths() -> list[Path]:
    paths: list[Path] = []
    explicit = _env("HERMES_CURSOR_HARNESS_PYTHONPATH")
    if explicit:
        paths.append(Path(explicit).expanduser())
    hermes_home = Path(os.getenv("HERMES_HOME", "~/.hermes")).expanduser()
    paths.append(hermes_home / "plugins" / "hermes-cursor-harness")

    here = Path(__file__).resolve()
    for parent in here.parents:
        paths.append(parent / "hermes-cursor-harness")
        paths.append(parent.parent / "hermes-cursor-harness")
    return paths
