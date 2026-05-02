"""Kaze Claude mode plugin.

Chat-level Telegram "Claude lane" for Hermes/Kaze.

When enabled for a Telegram chat, plain (non-slash) messages in that chat are
routed through Hermes' normal agent/tool loop, but with a per-session runtime
override to an Anthropic/Claude model. This preserves Hermes as the runtime
owner (sessions, tools, approvals, safety) while providing a Claude-powered
execution lane that can use Hermes tools to edit files and run commands.

Slash commands remain escape hatches and are never captured as Claude-mode plain
messages.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

PLUGIN_NAME = "kaze-claude-mode"
DEFAULT_CLAUDE_MODEL = os.environ.get("KAZE_CLAUDE_MODE_MODEL", "anthropic/claude-sonnet-4.6").strip()

MODE_COMMANDS = {"claude-mode", "claude_mode"}
INTERNAL_MODE = "kaze-claude-mode-dispatch"
ESCAPE_COMMANDS = {
    "approve",
    "background",
    "commands",
    "debug",
    "deny",
    "help",
    "new",
    "queue",
    "q",
    "reset",
    "restart",
    "status",
    "stop",
}
MAX_REPLY_CHARS = 3900

_CTX = None  # set by register()
_BOUNCE_PREFIX = (
    "Claude mode is enabled for this chat, but the toolful Claude backend is not configured.\n"
    "Routing this message to normal Hermes/Kaze instead.\n"
    "Run `/claude-mode status` for setup info."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", "")
    return getattr(platform, "value", str(platform or ""))


def state_key_from_source(source: Any) -> str:
    """Stable per-chat key; includes thread/topic when Hermes exposes it."""
    platform = _platform_value(source) or "unknown"
    chat_id = str(getattr(source, "chat_id", "") or "unknown")
    thread_id = str(getattr(source, "thread_id", "") or "")
    if thread_id:
        return f"{platform}:{chat_id}:{thread_id}"
    return f"{platform}:{chat_id}"


def _state_path() -> Path:
    raw = os.environ.get("KAZE_CLAUDE_MODE_STATE", "").strip()
    if raw:
        return Path(raw)
    return get_hermes_home() / "state" / "kaze_claude_mode.json"


def _load_state(path: Path | None = None) -> dict[str, Any]:
    path = _state_path() if path is None else path
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def is_enabled(key: str, *, path: Path | None = None) -> bool:
    path = _state_path() if path is None else path
    entry = (_load_state(path).get("chats") or {}).get(key) or {}
    return bool(entry.get("enabled"))


def set_enabled(
    key: str,
    enabled: bool,
    *,
    source: Any = None,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = _state_path() if path is None else path
    state = _load_state(path)
    chats = state.setdefault("chats", {})
    if not isinstance(chats, dict):
        chats = {}
        state["chats"] = chats
    entry = chats.setdefault(key, {})
    if not isinstance(entry, dict):
        entry = {}
        chats[key] = entry
    entry.update({
        "enabled": bool(enabled),
        "updated_at": _now_iso(),
    })
    if source is not None:
        entry["platform"] = _platform_value(source)
        entry["chat_id"] = str(getattr(source, "chat_id", "") or "")
        thread_id = getattr(source, "thread_id", None)
        if thread_id:
            entry["thread_id"] = str(thread_id)
    if extra:
        entry.update({k: v for k, v in extra.items() if v is not None})
    state["updated_at"] = _now_iso()
    _atomic_write_json(path, state)
    return entry


def _encode_packet(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_packet(raw: str) -> dict[str, Any]:
    try:
        data = base64.urlsafe_b64decode(raw.strip().encode("ascii"))
        parsed = json.loads(data.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _command_name(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return ""
    token = stripped.split(maxsplit=1)[0].lstrip("/")
    # Telegram group commands may arrive as /cmd@bot.
    return token.split("@", 1)[0].lower()


def _command_args(text: str) -> str:
    stripped = (text or "").strip()
    parts = stripped.split(maxsplit=1)
    return parts[1] if len(parts) > 1 else ""


def _resolve_toolful_claude_backend(*, model: str | None = None) -> tuple[bool, str, dict[str, Any]]:
    """Return (ok, model, runtime) for the Claude toolful lane."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    target_model = (model or DEFAULT_CLAUDE_MODEL or "anthropic/claude-sonnet-4.6").strip()
    runtime = resolve_runtime_provider(requested="anthropic", target_model=target_model)
    ok = bool(runtime.get("api_key")) and str(runtime.get("provider") or "").strip().lower() == "anthropic"
    return ok, target_model, runtime


def _apply_gateway_override(gateway: Any, session_key: str, *, model: str, runtime: dict[str, Any]) -> None:
    """Apply a per-session Claude lane override to the gateway runner."""
    if not gateway or not session_key:
        return
    overrides = getattr(gateway, "_session_model_overrides", None)
    if not isinstance(overrides, dict):
        return
    overrides[session_key] = {
        "model": model,
        "provider": runtime.get("provider") or "anthropic",
        "api_key": runtime.get("api_key") or "",
        "base_url": runtime.get("base_url") or "",
        "api_mode": runtime.get("api_mode") or "",
    }
    try:
        gateway._evict_cached_agent(session_key)
    except Exception:
        pass


def _clear_gateway_override(gateway: Any, session_key: str) -> None:
    if not gateway or not session_key:
        return
    try:
        overrides = getattr(gateway, "_session_model_overrides", None)
        if isinstance(overrides, dict):
            overrides.pop(session_key, None)
        gateway._evict_cached_agent(session_key)
    except Exception:
        pass


def build_pre_dispatch_decision(event: Any, gateway: Any = None) -> dict[str, Any] | None:
    """Return a pre_gateway_dispatch rewrite/allow decision for an event."""
    text = getattr(event, "text", None) or ""
    source = getattr(event, "source", None)
    if not source or not text.strip():
        return None
    platform = _platform_value(source)
    if platform != "telegram":
        return None

    key = state_key_from_source(source)
    cmd = _command_name(text)
    if cmd in MODE_COMMANDS:
        args = _command_args(text).strip().lower()
        session_key = ""
        if gateway is not None:
            try:
                session_key = gateway._session_key_for_source(source)
            except Exception:
                session_key = ""

        if args in {"off", "disable", "disabled"} and session_key:
            _clear_gateway_override(gateway, session_key)

        if args in {"on", "enable", "enabled"}:
            ok, model, _runtime = _resolve_toolful_claude_backend()
            if ok:
                set_enabled(
                    key,
                    True,
                    source=source,
                    extra={
                        "backend": "hermes_toolful",
                        "provider": "anthropic",
                        "model": model,
                        "tool_edit_active": True,
                        "last_enable_error": "",
                    },
                )
            else:
                set_enabled(
                    key,
                    False,
                    source=source,
                    extra={
                        "backend": "unavailable",
                        "provider": "anthropic",
                        "model": DEFAULT_CLAUDE_MODEL,
                        "tool_edit_active": False,
                        "last_enable_error": "missing Anthropic credentials",
                    },
                )

        packet = {
            "key": key,
            "args": _command_args(text),
            "platform": platform,
            "chat_id": str(getattr(source, "chat_id", "") or ""),
            "thread_id": str(getattr(source, "thread_id", "") or ""),
            "session_key": session_key,
        }
        return {"action": "rewrite", "text": f"/{INTERNAL_MODE} {_encode_packet(packet)}"}

    if cmd:
        return None

    if is_enabled(key):
        if gateway is None:
            return {"action": "rewrite", "text": f"{_BOUNCE_PREFIX}\n\n{text}"}

        try:
            session_key = gateway._session_key_for_source(source)
        except Exception:
            session_key = ""

        ok, model, runtime = _resolve_toolful_claude_backend()
        if ok and session_key:
            _apply_gateway_override(gateway, session_key, model=model, runtime=runtime)
            return None

        if session_key:
            _clear_gateway_override(gateway, session_key)
        return {"action": "rewrite", "text": f"{_BOUNCE_PREFIX}\n\n{text}"}

    return None


def pre_gateway_dispatch(event: Any = None, gateway: Any = None, session_store: Any = None, **_: Any) -> dict[str, Any] | None:
    return build_pre_dispatch_decision(event, gateway)


def _format_backend_status() -> tuple[str, bool, str]:
    ok, model, runtime = _resolve_toolful_claude_backend()
    if ok:
        api_mode = str(runtime.get("api_mode") or "").strip() or "auto"
        base_url = str(runtime.get("base_url") or "").strip() or "default"
        backend = (
            f"Hermes toolful Claude lane (provider=`anthropic`, model=`{model}`, "
            f"api_mode=`{api_mode}`, base_url=`{base_url}`)"
        )
        return backend, True, model
    backend = (
        "Claude lane unavailable (missing Anthropic credentials).\n"
        f"Configure in {display_hermes_home()}/.env (ANTHROPIC_API_KEY) or run `hermes auth add anthropic`."
    )
    return backend, False, (DEFAULT_CLAUDE_MODEL or "anthropic/claude-sonnet-4.6")


class contextlib_suppress:
    def __init__(self, *exceptions: type[BaseException]) -> None:
        self.exceptions = exceptions or (Exception,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, self.exceptions)


def _status_message(key: str) -> str:
    state = _load_state()
    entry = ((state.get("chats") or {}).get(key) or {}) if isinstance(state, dict) else {}
    mode = "on" if entry.get("enabled") else "off"
    updated = entry.get("updated_at") or "unknown"
    backend, toolful, _model = _format_backend_status()
    tool_state = "**active**" if (entry.get("enabled") and toolful) else "**inactive**"
    return (
        f"Claude mode: **{mode}**\n"
        f"chat: `{key}`\n"
        f"updated: `{updated}`\n"
        f"backend: {backend}\n"
        f"tool/edit: {tool_state}"
    )


async def _run_smoke(session_key: str) -> str:
    """Prove tool/edit capability without leaking outputs or source bodies."""
    global _CTX
    if _CTX is None:
        return "Smoke failed: plugin context unavailable (tools not wired)."

    backend, toolful, model = _format_backend_status()
    if not toolful:
        return f"Smoke: **unavailable**\nbackend: {backend}"

    try:
        from tools.approval import reset_current_session_key, set_current_session_key
        token = set_current_session_key(session_key or "")
    except Exception:
        token = None
        reset_current_session_key = None

    tmp_dir = get_hermes_home() / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"claude_mode_smoke_{os.urandom(3).hex()}.txt"
    marker_cmd = "KAZE_CLAUDE_MODE_TOOL_OK"
    marker_file = "KAZE_CLAUDE_MODE_FILE_OK"
    try:
        term_raw = _CTX.dispatch_tool(
            "terminal",
            {"command": f"echo {marker_cmd}", "timeout": 10},
            task_id="claude_mode_smoke",
        )
        term = json.loads(term_raw) if isinstance(term_raw, str) else {}
        term_out = str(term.get("stdout") or term.get("output") or term.get("result") or "")
        if marker_cmd not in term_out:
            return "Smoke failed: terminal tool did not return expected marker."

        write_raw = _CTX.dispatch_tool(
            "write_file",
            {"path": str(tmp_path), "content": marker_file + "\n"},
            task_id="claude_mode_smoke",
        )
        write_payload = json.loads(write_raw) if isinstance(write_raw, str) else {}
        if write_payload.get("error"):
            return "Smoke failed: write_file tool returned an error."

        try:
            tmp_path.unlink(missing_ok=True)
        except TypeError:
            if tmp_path.exists():
                tmp_path.unlink()

        return (
            "Smoke: **OK**\n"
            f"- model lane: `{model}`\n"
            f"- tool call: `{marker_cmd}`\n"
            f"- file write+cleanup: `{tmp_path.name}`"
        )
    finally:
        if token is not None and reset_current_session_key is not None:
            with contextlib_suppress(Exception):
                reset_current_session_key(token)
        with contextlib_suppress(Exception):
            if tmp_path.exists():
                tmp_path.unlink()


async def handle_internal_mode(raw_args: str) -> str:
    packet = _decode_packet(raw_args)
    key = str(packet.get("key") or "").strip()
    args = str(packet.get("args") or "").strip().lower()
    session_key = str(packet.get("session_key") or "").strip()
    if not key:
        return "Claude mode could not identify this chat."
    if args in {"on", "enable", "enabled"}:
        if not is_enabled(key):
            backend, _toolful, _model = _format_backend_status()
            return (
                "Claude mode: **off**\n"
                f"backend: {backend}\n\n"
                "Cannot enable tool/edit Claude mode without Anthropic credentials."
            )
        backend, _toolful, _model = _format_backend_status()
        return (
            "Claude mode: **on**\n"
            f"backend: {backend}\n"
            "Plain (non-slash) messages in this chat now route through Claude + Hermes tools.\n"
            "Use `/claude-mode off` to return to normal Hermes/Kaze."
        )
    if args in {"off", "disable", "disabled"}:
        set_enabled(key, False, extra={"tool_edit_active": False})
        return "Claude mode: **off**\nPlain messages now use normal Kaze/Hermes again."
    if args in {"status", ""}:
        return _status_message(key) + "\n\nUsage: `/claude-mode on|off|status|smoke`"
    if args == "smoke":
        return await _run_smoke(session_key)
    return "Usage: `/claude-mode on|off|status|smoke`"


async def handle_public_mode(raw_args: str) -> str:
    return "Use `/claude-mode on|off|status|smoke` from Telegram so the plugin can identify the chat."


def register(ctx: Any) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_command(
        "claude-mode",
        handle_public_mode,
        "Toggle chat-level Claude Code fallback mode",
        "on|off|status|smoke",
    )
    ctx.register_command(INTERNAL_MODE, handle_internal_mode, "Internal Kaze Claude mode control", "<packet>")
