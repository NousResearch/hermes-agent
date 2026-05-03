"""Kaze Claude mode plugin.

Chat-level Telegram "Claude lane" for Hermes/Kaze.

Spec-aligned behavior:
- Hermes remains the Telegram/runtime/session owner (routing, auth, approvals).
- When enabled for a Telegram chat, *plain (non-slash)* messages are routed to a
  Claude Code CLI execution lane (headless `claude -p`), not a hidden model
  provider switch.
- Slash commands remain escape hatches and are never captured as Claude-mode
  plain messages.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shlex import quote as shell_quote
from shutil import which
from string import ascii_lowercase
from typing import Any

from hermes_constants import display_hermes_home, get_hermes_home

PLUGIN_NAME = "kaze-claude-mode"

MODE_COMMANDS = {"claude-mode", "claude_mode"}
INTERNAL_MODE = "kaze-claude-mode-dispatch"
INTERNAL_RUN = "kaze-claude-mode-run"

MAX_REPLY_CHARS = 3900
_PENDING_TTL_SECS = 120
_APPROVAL_ID_LEN = 5
_APPROVAL_ID_ALPHABET = "".join([c for c in ascii_lowercase if c != "l"])
_TRANSCRIPT_MAX_MESSAGES = 40
_TRANSCRIPT_MAX_CHARS = 12000

_CTX = None  # set by register()

_BOUNCE_PREFIX = (
    "Claude mode is enabled for this chat, but the Claude Code CLI backend is not available.\n"
    "Run `/claude-mode status` for setup info."
)


@dataclass(frozen=True)
class _PendingPrompt:
    chat_key: str
    session_key: str
    prompt: str
    created_at: float
    platform: str = ""
    chat_id: str = ""
    thread_id: str = ""


_PENDING: dict[str, _PendingPrompt] = {}


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


def _claude_code_cli_candidates() -> tuple[str, ...]:
    return ("claude", "claude-code")


def _resolve_claude_code_cli_backend() -> tuple[bool, dict[str, Any]]:
    """Return (ok, info) for the Claude Code CLI backend.

    This checks binary presence only; auth is exercised by smoke and live runs.
    """
    for cmd in _claude_code_cli_candidates():
        path = which(cmd)
        if path:
            return True, {"backend": "claude-code-cli", "cmd": cmd, "path": path}
    return False, {
        "backend": "claude-code-cli",
        "cmd": "",
        "path": "",
        "error": "Claude Code CLI not found on PATH",
    }


def _claude_code_allowed_tools() -> str:
    # Safe baseline: file ops + search only (no Bash by default).
    raw = os.environ.get("KAZE_CLAUDE_MODE_ALLOWED_TOOLS", "").strip()
    return raw or "Read,Write,Edit,Grep,Glob"


def _claude_code_permission_mode() -> str:
    raw = os.environ.get("KAZE_CLAUDE_MODE_PERMISSION_MODE", "acceptEdits").strip()
    allowed = {"acceptEdits", "auto", "default", "dontAsk", "plan", "bypassPermissions"}
    return raw if raw in allowed else "acceptEdits"


def _update_chat(
    key: str,
    *,
    source: Any = None,
    path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update a chat entry in the profile-safe state file without toggling enablement."""
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
    entry["updated_at"] = _now_iso()
    if source is not None:
        entry["platform"] = _platform_value(source)
        entry["chat_id"] = str(getattr(source, "chat_id", "") or "")
        thread_id = getattr(source, "thread_id", None)
        if thread_id:
            entry["thread_id"] = str(thread_id)
    if extra:
        for k, v in extra.items():
            if v is None:
                entry.pop(k, None)
            else:
                entry[k] = v
    state["updated_at"] = _now_iso()
    _atomic_write_json(path, state)
    return entry


def _chat_entry(key: str) -> dict[str, Any]:
    state = _load_state()
    entry = ((state.get("chats") or {}).get(key) or {}) if isinstance(state, dict) else {}
    return entry if isinstance(entry, dict) else {}


def _chat_allowed_tools_extra(key: str) -> list[str]:
    raw = _chat_entry(key).get("allowed_tools_extra") or []
    if not isinstance(raw, list):
        return []
    items: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return items


def _effective_allowed_tools(key: str) -> str:
    base = _claude_code_allowed_tools()
    extras = _chat_allowed_tools_extra(key)
    parts = [p.strip() for p in base.split(",") if p.strip()]
    parts.extend(extras)
    seen: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return ",".join(deduped)


def _is_yolo_enabled(key: str) -> bool:
    return bool(_chat_entry(key).get("yolo"))


def _effective_permission_mode(key: str) -> str:
    if _is_yolo_enabled(key):
        return "bypassPermissions"
    return _claude_code_permission_mode()


def _tools_from_allowed_tools(allowed_tools: str) -> str:
    """Derive a built-in tool allowlist for `--tools` from allowedTools rules.

    Claude Code's `--allowedTools` controls auto-approval, not tool availability.
    `--tools` limits which built-in tools are present at all.
    """
    builtins: list[str] = []
    for raw in (allowed_tools or "").split(","):
        token = raw.strip()
        if not token or token.startswith("mcp__"):
            continue
        name = token.split("(", 1)[0].strip()
        if not name:
            continue
        builtins.append(name)
    if not builtins:
        builtins = ["Read", "Write", "Edit", "Grep", "Glob"]
    seen: set[str] = set()
    out: list[str] = []
    for name in builtins:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return ",".join(out)


def _pending_approval(key: str) -> dict[str, Any] | None:
    raw = _chat_entry(key).get("pending_approval")
    return raw if isinstance(raw, dict) and raw else None


def _has_pending_approval(key: str) -> bool:
    return bool(_pending_approval(key))


def _new_approval_id() -> str:
    # Five lowercase letters drawn from a-z without l (avoids 1/I ambiguity on phones).
    return "".join(_APPROVAL_ID_ALPHABET[b % len(_APPROVAL_ID_ALPHABET)] for b in os.urandom(_APPROVAL_ID_LEN))


def _approval_prompt_path(approval_id: str) -> Path:
    base = get_hermes_home() / "state" / "kaze_claude_mode_prompts"
    return base / f"{approval_id}.txt"


def _write_private_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _read_private_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _clear_pending_approval(key: str) -> None:
    pending = _pending_approval(key) or {}
    prompt_path = pending.get("prompt_path")
    _update_chat(key, extra={"pending_approval": None})
    if isinstance(prompt_path, str) and prompt_path:
        with contextlib_suppress(Exception):
            Path(prompt_path).unlink(missing_ok=True)


def _looks_like_permission_block(text: str) -> bool:
    lowered = (text or "").lower()
    if "permission blocked" in lowered:
        return True
    if "permission" in lowered and ("approve" in lowered or "prompt" in lowered or "denied" in lowered):
        return True
    return False


_TOOL_RULE_RE = re.compile(r"(mcp__[-a-zA-Z0-9_]+__[-a-zA-Z0-9_*]+|[A-Za-z][A-Za-z0-9_]*(?:\([^\)]{1,200}\))?)")


def _extract_tool_rule(text: str) -> str:
    """Best-effort extraction of a tool rule from Claude Code errors."""
    for match in _TOOL_RULE_RE.finditer(text or ""):
        candidate = (match.group(1) or "").strip()
        if not candidate:
            continue
        # Skip obvious non-tool words.
        if candidate.lower() in {"permission", "permissions", "error", "blocked", "headless"}:
            continue
        # Prefer explicit specifiers (Tool(...)) or mcp__ patterns.
        if candidate.startswith("mcp__") or "(" in candidate:
            return candidate
    # Fallback to a bare tool name if present.
    for match in _TOOL_RULE_RE.finditer(text or ""):
        candidate = (match.group(1) or "").strip()
        if candidate and "(" not in candidate and candidate.startswith(("Read", "Write", "Edit", "Bash", "WebFetch", "WebSearch", "Grep", "Glob")):
            return candidate
    return ""


def _claude_code_max_turns() -> int:
    raw = os.environ.get("KAZE_CLAUDE_MODE_MAX_TURNS", "90").strip()
    try:
        value = int(raw)
    except Exception:
        value = 90
    return max(1, min(180, value))


def _claude_code_timeout_secs() -> int:
    raw = os.environ.get("KAZE_CLAUDE_MODE_TIMEOUT", "900").strip()
    try:
        value = int(raw)
    except Exception:
        value = 900
    return max(30, min(3600, value))


def _claude_code_max_budget_usd() -> str:
    raw = os.environ.get("KAZE_CLAUDE_MODE_MAX_BUDGET_USD", "").strip()
    if not raw:
        return ""
    try:
        value = float(raw)
    except Exception:
        return ""
    if value <= 0:
        return ""
    return str(value)


def _gc_pending(now: float | None = None) -> None:
    now = time.time() if now is None else now
    expired = [k for k, v in _PENDING.items() if (now - v.created_at) > _PENDING_TTL_SECS]
    for k in expired:
        _PENDING.pop(k, None)


def _stash_pending_prompt(
    *,
    chat_key: str,
    session_key: str,
    prompt: str,
    platform: str = "",
    chat_id: str = "",
    thread_id: str = "",
) -> str:
    _gc_pending()
    token = os.urandom(9).hex()
    _PENDING[token] = _PendingPrompt(
        chat_key=chat_key,
        session_key=session_key,
        prompt=prompt,
        created_at=time.time(),
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
    )
    return token


def _pop_pending_prompt(token: str) -> _PendingPrompt | None:
    _gc_pending()
    return _PENDING.pop((token or "").strip(), None)


def _message_text_for_transcript(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return " ".join(parts)
    return "" if content is None else str(content)


def _format_recent_transcript(session_store: Any, session_key: str) -> str:
    if session_store is None or not session_key:
        return ""
    try:
        session_store._ensure_loaded()
        entry = getattr(session_store, "_entries", {}).get(session_key)
        session_id = getattr(entry, "session_id", "") if entry else ""
        if not session_id:
            return ""
        messages = session_store.load_transcript(session_id)
    except Exception:
        return ""
    lines: list[str] = []
    for message in messages[-_TRANSCRIPT_MAX_MESSAGES:]:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        if role == "tool":
            continue
        text = _message_text_for_transcript(message).strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        lines.append(f"{role}: {text[:1000]}")
    excerpt = "\n".join(lines)
    if len(excerpt) > _TRANSCRIPT_MAX_CHARS:
        excerpt = excerpt[-_TRANSCRIPT_MAX_CHARS:]
        first_newline = excerpt.find("\n")
        if first_newline >= 0:
            excerpt = excerpt[first_newline + 1 :]
    return excerpt


def _build_claude_mode_prompt(*, user_text: str, transcript: str) -> str:
    if not transcript:
        return user_text
    return (
        "You are the Claude Code CLI lane invoked from Hermes/Kaze for this Telegram chat.\n"
        "Use the recent Hermes session transcript below as the available Telegram chat history. "
        "Do not claim access to messages outside this transcript. Answer the user's latest message directly and concisely.\n\n"
        "<recent_telegram_session_transcript>\n"
        f"{transcript}\n"
        "</recent_telegram_session_transcript>\n\n"
        "<latest_user_message>\n"
        f"{user_text}\n"
        "</latest_user_message>"
    )


def build_pre_dispatch_decision(event: Any, gateway: Any = None, session_store: Any = None) -> dict[str, Any] | None:
    """Return a pre_gateway_dispatch rewrite/allow decision for an event."""
    text = getattr(event, "text", None) or ""
    source = getattr(event, "source", None)
    if not source or not text.strip():
        return None
    platform = _platform_value(source)
    if platform != "telegram":
        return None

    chat_key = state_key_from_source(source)
    cmd = _command_name(text)

    # /claude-mode itself is rewritten to our internal dispatcher so we can
    # identify the chat + (when possible) the gateway session key.
    if cmd in MODE_COMMANDS:
        session_key = ""
        if gateway is not None:
            try:
                session_key = gateway._session_key_for_source(source)
            except Exception:
                session_key = ""
        packet = {
            "key": chat_key,
            "args": _command_args(text),
            "platform": platform,
            "chat_id": str(getattr(source, "chat_id", "") or ""),
            "thread_id": str(getattr(source, "thread_id", "") or ""),
            "session_key": session_key,
        }
        return {"action": "rewrite", "text": f"/{INTERNAL_MODE} {_encode_packet(packet)}"}

    # Slash commands are escape hatches.
    if cmd:
        return None

    # Plain messages in an enabled chat route to Claude Code.
    if is_enabled(chat_key):
        ok, _info = _resolve_claude_code_cli_backend()
        if not ok:
            # Backend missing: fail closed with a clear message. Avoid embedding
            # user text in the rewritten command to reduce logging leakage.
            return {
                "action": "rewrite",
                "text": f"/{INTERNAL_MODE} {_encode_packet({'key': chat_key, 'args': 'unavailable'})}",
            }
        if gateway is None:
            return {
                "action": "rewrite",
                "text": f"/{INTERNAL_MODE} {_encode_packet({'key': chat_key, 'args': 'unavailable'})}",
            }
        try:
            session_key = gateway._session_key_for_source(source)
        except Exception:
            session_key = ""
        transcript = _format_recent_transcript(session_store, session_key)
        prompt = _build_claude_mode_prompt(user_text=text, transcript=transcript)
        token = _stash_pending_prompt(
            chat_key=chat_key,
            session_key=session_key,
            prompt=prompt,
            platform=platform,
            chat_id=str(getattr(source, "chat_id", "") or ""),
            thread_id=str(getattr(source, "thread_id", "") or ""),
        )
        return {"action": "rewrite", "text": f"/{INTERNAL_RUN} {token}"}

    return None


def pre_gateway_dispatch(event: Any = None, gateway: Any = None, session_store: Any = None, **_: Any) -> dict[str, Any] | None:
    return build_pre_dispatch_decision(event, gateway, session_store=session_store)


def _format_backend_status() -> tuple[str, bool]:
    ok, info = _resolve_claude_code_cli_backend()
    if ok:
        return f"`claude-code-cli` (cmd=`{info.get('cmd')}`, path=`{info.get('path')}`)", True
    return (
        "Claude Code CLI unavailable.\n"
        "Install: `npm install -g @anthropic-ai/claude-code`\n"
        "Auth check: `claude auth status --text`\n"
        "PATH hint: ensure your npm global bin is on PATH for the gateway process.\n"
        f"(Hermes home: {display_hermes_home()})",
        False,
    )


class contextlib_suppress:
    def __init__(self, *exceptions: type[BaseException]) -> None:
        self.exceptions = exceptions or (Exception,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, self.exceptions)


def _status_message(key: str) -> str:
    entry = _chat_entry(key)
    mode = "on" if entry.get("enabled") else "off"
    updated = entry.get("updated_at") or "unknown"
    backend, toolful = _format_backend_status()
    tool_state = "**active**" if (entry.get("enabled") and toolful) else "**inactive**"
    yolo = "**on**" if _is_yolo_enabled(key) else "**off**"
    pending = _pending_approval(key)
    pending_line = "pending: **none**"
    if pending:
        pending_id = pending.get("id") or "unknown"
        pending_tool = pending.get("tool_rule") or "unknown"
        pending_line = f"pending: **yes** (`{pending_id}` / `{pending_tool}`)"
    return (
        f"Claude mode: **{mode}**\n"
        f"chat: `{key}`\n"
        f"updated: `{updated}`\n"
        f"backend: {backend}\n"
        f"allowedTools: `{_effective_allowed_tools(key)}`\n"
        f"permission-mode: `{_effective_permission_mode(key)}`\n"
        f"max-turns: `{_claude_code_max_turns()}`\n"
        f"yolo: {yolo}\n"
        f"{pending_line}\n"
        f"tool/edit: {tool_state}"
    )


def _truncate_reply(text: str, limit: int = MAX_REPLY_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n\n…(truncated)…"


def _extract_terminal_text(payload: dict[str, Any]) -> str:
    stdout = str(payload.get("stdout") or payload.get("output") or payload.get("result") or "")
    stderr = str(payload.get("stderr") or "")
    combined = stdout.strip() or stderr.strip()
    return combined


class _TelegramProgress:
    """Best-effort single-message Telegram progress updater for Claude Code."""

    def __init__(self, *, chat_id: str, thread_id: str = "") -> None:
        self.chat_id = str(chat_id or "")
        self.thread_id = str(thread_id or "")
        self._bot = None
        self._message_id = ""
        self._last_edit = 0.0
        self._last_text = ""
        self.active = False

    async def start(self, text: str) -> bool:
        if not self.chat_id:
            return False
        try:
            from hermes_cli.config import load_config
            from telegram import Bot

            cfg = load_config()
            platforms = cfg.get("platforms", {}) if isinstance(cfg, dict) else {}
            telegram_cfg = platforms.get("telegram", {}) if isinstance(platforms, dict) else {}
            token = str(telegram_cfg.get("token") or os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
            if not token:
                return False
            self._bot = Bot(token=token)
            kwargs: dict[str, Any] = {"chat_id": int(self.chat_id), "text": text}
            if self.thread_id:
                kwargs["message_thread_id"] = int(self.thread_id)
            msg = await self._bot.send_message(**kwargs)
            self._message_id = str(getattr(msg, "message_id", "") or "")
            self._last_text = text
            self._last_edit = time.time()
            self.active = bool(self._message_id)
            return self.active
        except Exception:
            self.active = False
            return False

    async def edit(self, text: str, *, force: bool = False) -> None:
        if not self.active or self._bot is None or not self._message_id:
            return
        text = _truncate_reply(text, 3900)
        if not force and text == self._last_text:
            return
        now = time.time()
        if not force and (now - self._last_edit) < 1.0:
            return
        try:
            await self._bot.edit_message_text(
                chat_id=int(self.chat_id),
                message_id=int(self._message_id),
                text=text,
            )
            self._last_text = text
            self._last_edit = now
        except Exception:
            # Progress is auxiliary; never fail the Claude run because Telegram
            # rejected an edit or rate-limited us.
            return

    async def typing(self) -> None:
        if not self.active or self._bot is None:
            return
        try:
            await self._bot.send_chat_action(chat_id=int(self.chat_id), action="typing")
        except Exception:
            return


def _stream_status_text(*, status: str, assistant_text: str = "", detail: str = "") -> str:
    parts = ["Claude mode: **running**", f"status: {status}"]
    if detail:
        parts.append(detail)
    if assistant_text.strip():
        parts.append("\n" + assistant_text.strip())
    return _truncate_reply("\n".join(parts), 3900)


def _summarize_stream_event(obj: dict[str, Any]) -> tuple[str, str]:
    event = obj.get("event") if isinstance(obj.get("event"), dict) else {}
    etype = str(event.get("type") or "")
    if etype == "content_block_start":
        block = event.get("content_block") if isinstance(event.get("content_block"), dict) else {}
        if block.get("type") == "tool_use":
            return "tool", str(block.get("name") or "tool")
    if etype == "content_block_delta":
        delta = event.get("delta") if isinstance(event.get("delta"), dict) else {}
        if delta.get("type") == "text_delta":
            return "text", str(delta.get("text") or "")
    if etype == "message_stop":
        return "status", "finalizing"
    subtype = str(obj.get("subtype") or "")
    if obj.get("type") == "system" and subtype == "status":
        return "status", str(obj.get("status") or "working")
    return "", ""


def _result_text_from_stream_obj(obj: dict[str, Any]) -> str:
    result = obj.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
    content = message.get("content") if isinstance(message.get("content"), list) else []
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts).strip()


def _dispatch_tool(tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
    """Dispatch built-in Hermes tools from plugin command context.

    Plugin discovery can happen before built-in tool modules are imported in
    standalone command/plugin probes, so force idempotent built-in discovery
    before using PluginContext.dispatch_tool.
    """
    from tools.registry import discover_builtin_tools

    discover_builtin_tools()
    return _CTX.dispatch_tool(tool_name, args, **kwargs)


async def _run_claude_code_streaming(
    *,
    prompt: str,
    chat_key: str,
    session_key: str,
    task_id: str,
    chat_id: str,
    thread_id: str = "",
    workdir: str | None = None,
) -> tuple[bool, str, bool]:
    """Run Claude Code with stream-json and edit one Telegram progress message.

    Returns (ok, text, delivered). When delivered=True, the final result/error was
    already sent via the progress message and command dispatch should return None.
    """
    ok, info = _resolve_claude_code_cli_backend()
    if not ok:
        backend, _toolful = _format_backend_status()
        return False, backend, False

    progress = _TelegramProgress(chat_id=chat_id, thread_id=thread_id)
    started = await progress.start(
        _stream_status_text(status="starting Claude Code…", detail=f"max-turns: `{_claude_code_max_turns()}`")
    )
    if not started:
        return False, "", False

    allowed = _effective_allowed_tools(chat_key) if chat_key else _claude_code_allowed_tools()
    tools_list = _tools_from_allowed_tools(allowed)
    permission_mode = _effective_permission_mode(chat_key) if chat_key else _claude_code_permission_mode()
    max_turns = _claude_code_max_turns()
    budget = _claude_code_max_budget_usd()
    cmd = [
        str(info.get("cmd") or "claude"),
        "-p",
        "--verbose",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--include-hook-events",
        "--allowedTools",
        allowed,
        "--tools",
        tools_list,
        "--permission-mode",
        permission_mode,
        "--max-turns",
        str(max_turns),
    ]
    if budget:
        cmd.extend(["--max-budget-usd", budget])

    assistant_text = ""
    final_text = ""
    stderr_text = ""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir or None,
        )
        assert proc.stdin is not None
        proc.stdin.write((prompt or "").encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()

        async def _read_stderr() -> bytes:
            assert proc.stderr is not None
            return await proc.stderr.read()

        stderr_task = asyncio.create_task(_read_stderr())

        async def _read_stdout() -> None:
            nonlocal assistant_text, final_text
            assert proc.stdout is not None
            while True:
                raw_line = await proc.stdout.readline()
                if not raw_line:
                    break
                try:
                    obj = json.loads(raw_line.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                kind, value = _summarize_stream_event(obj)
                if kind == "text" and value:
                    assistant_text += value
                    await progress.edit(_stream_status_text(status="answering…", assistant_text=assistant_text))
                elif kind == "tool" and value:
                    await progress.edit(
                        _stream_status_text(status="working…", assistant_text=assistant_text, detail=f"tool: `{value}`"),
                        force=False,
                    )
                elif kind == "status" and value:
                    await progress.typing()
                    await progress.edit(_stream_status_text(status=value, assistant_text=assistant_text), force=False)
                if obj.get("type") == "result":
                    final_text = _result_text_from_stream_obj(obj) or assistant_text

        await asyncio.wait_for(_read_stdout(), timeout=_claude_code_timeout_secs())
        exit_code = await asyncio.wait_for(proc.wait(), timeout=5)
        stderr_bytes = await stderr_task
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
    except asyncio.TimeoutError:
        with contextlib_suppress(Exception):
            proc.kill()  # type: ignore[name-defined]
        msg = f"Claude Code error:\nTimed out after `{_claude_code_timeout_secs()}` seconds."
        await progress.edit(msg, force=True)
        return False, msg, True
    except Exception as e:
        msg = f"Claude Code error:\n{e}"
        await progress.edit(_truncate_reply(msg), force=True)
        return False, msg, True

    out = (final_text or assistant_text or stderr_text or "").strip()
    if exit_code not in (0, None):
        err = out or "Claude Code returned a non-zero exit code."
        if chat_key and _looks_like_permission_block(err):
            _clear_pending_approval(chat_key)
            approval_id = _new_approval_id()
            prompt_sha256 = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
            prompt_disk_path = _approval_prompt_path(approval_id)
            _write_private_text(prompt_disk_path, prompt or "")
            tool_rule = _extract_tool_rule(err)
            _update_chat(
                chat_key,
                extra={
                    "pending_approval": {
                        "id": approval_id,
                        "tool_rule": tool_rule,
                        "created_at": _now_iso(),
                        "prompt_path": str(prompt_disk_path),
                        "prompt_sha256": prompt_sha256,
                        "session_key": session_key or "",
                    }
                },
            )
            safe_tool = tool_rule or "unknown"
            msg = _truncate_reply(
                "Claude mode: **approval pending**\n"
                f"- id: `{approval_id}`\n"
                f"- tool: `{safe_tool}`\n"
                "Reply `/claude-mode approve` to allow and retry, or `/claude-mode deny` to cancel."
            )
            await progress.edit(msg, force=True)
            return False, msg, True
        msg = _truncate_reply(f"Claude Code error:\n{err}")
        await progress.edit(msg, force=True)
        return False, msg, True

    final = _truncate_reply(out or "(Claude Code returned no output.)")
    await progress.edit(final, force=True)
    return True, final, True


async def _run_claude_code_print(
    *,
    prompt: str,
    chat_key: str,
    session_key: str,
    task_id: str,
    workdir: str | None = None,
    platform: str = "",
    chat_id: str = "",
    thread_id: str = "",
) -> tuple[bool, str]:
    """Run Claude Code CLI in headless print mode.

    Critical: do NOT put the raw prompt on the shell command line (it would be
    logged by approval/safety layers). Feed it via a temp file.
    """
    global _CTX
    if _CTX is None:
        return False, "Claude mode failed: plugin context unavailable (tools not wired)."

    ok, info = _resolve_claude_code_cli_backend()
    if not ok:
        backend, _toolful = _format_backend_status()
        return False, backend

    if platform == "telegram" and chat_id and not workdir:
        ok_stream, out_stream, delivered = await _run_claude_code_streaming(
            prompt=prompt,
            chat_key=chat_key,
            session_key=session_key,
            task_id=task_id,
            chat_id=chat_id,
            thread_id=thread_id,
            workdir=workdir,
        )
        if delivered:
            return ok_stream, ""

    try:
        from tools.approval import reset_current_session_key, set_current_session_key
        token = set_current_session_key(session_key or "")
    except Exception:
        token = None
        reset_current_session_key = None

    tmp_dir = get_hermes_home() / "tmp" / "kaze_claude_mode"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = tmp_dir / f"prompt_{os.urandom(4).hex()}.txt"
    try:
        write_raw = _dispatch_tool(
            "write_file",
            {"path": str(prompt_path), "content": prompt},
            task_id=task_id,
        )
        write_payload = json.loads(write_raw) if isinstance(write_raw, str) else {}
        if write_payload.get("error"):
            return False, "Claude mode failed: could not stage prompt for Claude Code."

        allowed = _effective_allowed_tools(chat_key) if chat_key else _claude_code_allowed_tools()
        tools_list = _tools_from_allowed_tools(allowed)
        max_turns = _claude_code_max_turns()

        permission_mode = _effective_permission_mode(chat_key) if chat_key else _claude_code_permission_mode()

        # Use $(cat <file>) so the prompt is NOT present in the command string
        # Hermes logs (inbound previews, approval queues, etc.).
        cmd = (
            f"{shell_quote(str(info.get('cmd') or 'claude'))} -p "
            f"\"$(cat {shell_quote(str(prompt_path))})\" "
            f"--allowedTools {shell_quote(allowed)} "
            f"--tools {shell_quote(tools_list)} "
            f"--permission-mode {shell_quote(permission_mode)} "
            f"--max-turns {max_turns}"
        )
        term_raw = _dispatch_tool(
            "terminal",
            {"command": cmd, "timeout": _claude_code_timeout_secs(), "workdir": workdir},
            task_id=task_id,
        )
        term = json.loads(term_raw) if isinstance(term_raw, str) else {}
        if term.get("status") == "approval_required":
            return False, "Claude mode: waiting for approval to run Claude Code CLI."
        out = _extract_terminal_text(term)
        exit_code = term.get("exit_code")
        if exit_code not in (None, 0, "0"):
            err = out or "Claude Code returned a non-zero exit code."
            if chat_key and _looks_like_permission_block(err):
                # Replace any previous pending approval for this chat.
                _clear_pending_approval(chat_key)
                approval_id = _new_approval_id()
                prompt_sha256 = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
                prompt_disk_path = _approval_prompt_path(approval_id)
                _write_private_text(prompt_disk_path, prompt or "")
                tool_rule = _extract_tool_rule(err)
                _update_chat(
                    chat_key,
                    extra={
                        "pending_approval": {
                            "id": approval_id,
                            "tool_rule": tool_rule,
                            "created_at": _now_iso(),
                            "prompt_path": str(prompt_disk_path),
                            "prompt_sha256": prompt_sha256,
                            "session_key": session_key or "",
                        }
                    },
                )
                safe_tool = tool_rule or "unknown"
                return (
                    False,
                    _truncate_reply(
                        "Claude mode: **approval pending**\n"
                        f"- id: `{approval_id}`\n"
                        f"- tool: `{safe_tool}`\n"
                        "Reply `/claude-mode approve` to allow and retry, or `/claude-mode deny` to cancel."
                    ),
                )
            return False, _truncate_reply(f"Claude Code error:\n{err}")
        return True, _truncate_reply(out or "(Claude Code returned no output.)")
    finally:
        if token is not None and reset_current_session_key is not None:
            with contextlib_suppress(Exception):
                reset_current_session_key(token)
        with contextlib_suppress(Exception):
            if prompt_path.exists():
                prompt_path.unlink()


async def handle_internal_mode(raw_args: str) -> str:
    packet = _decode_packet(raw_args)
    key = str(packet.get("key") or "").strip()
    raw = str(packet.get("args") or "").strip()
    args = raw.lower()
    session_key = str(packet.get("session_key") or "").strip()
    if not key:
        return "Claude mode could not identify this chat."

    head, *_rest = raw.split(maxsplit=1) if raw else [""]
    head_l = head.lower()
    tail = _rest[0] if _rest else ""

    if args in {"on", "enable", "enabled"}:
        ok, _info = _resolve_claude_code_cli_backend()
        backend, toolful = _format_backend_status()
        if not ok:
            set_enabled(
                key,
                False,
                extra={
                    "backend": "claude-code-cli",
                    "tool_edit_active": False,
                    "last_enable_error": "Claude Code CLI not found",
                },
            )
            return (
                "Claude mode: **off**\n"
                f"backend: {backend}\n\n"
                "Cannot enable Claude Code mode because the CLI backend is unavailable."
            )
        set_enabled(
            key,
            True,
            extra={
                "backend": "claude-code-cli",
                "tool_edit_active": True,
                "yolo": bool(_chat_entry(key).get("yolo")),
                "last_enable_error": "",
            },
        )
        return (
            "Claude mode: **on**\n"
            f"backend: {backend}\n"
            "Plain (non-slash) messages in this chat now route through Claude Code CLI (headless `claude -p`).\n"
            "Use `/claude-mode off` to return to normal Hermes/Kaze."
        )

    if args in {"off", "disable", "disabled"}:
        set_enabled(key, False, extra={"tool_edit_active": False})
        return "Claude mode: **off**\nPlain messages now use normal Kaze/Hermes again."

    if args in {"status", ""}:
        return _status_message(key) + "\n\nUsage: `/claude-mode on|off|status|smoke|approve|deny|yolo`"

    if args == "unavailable":
        backend, _toolful = _format_backend_status()
        return f"{_BOUNCE_PREFIX}\n\nbackend: {backend}\n\nUse `/hermes <message>` to talk to Hermes."

    if args == "smoke":
        backend, toolful = _format_backend_status()
        ok, info = _resolve_claude_code_cli_backend()
        if not ok or not toolful:
            return f"Smoke: **unavailable**\nbackend: {backend}"

        # Prove the real backend: run `claude -p` and verify it can edit a temp file.
        tmp_dir = get_hermes_home() / "tmp" / "kaze_claude_mode_smoke"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tmp_dir / f"smoke_{os.urandom(3).hex()}.txt"
        marker = "KAZE_CLAUDE_MODE_FILE_OK"
        _dispatch_tool("write_file", {"path": str(tmp_file), "content": "before\n"}, task_id="claude_mode_smoke")

        prompt = (
            "In the current directory, open the file named "
            f"{tmp_file.name} and replace its entire content with exactly this single line:\n"
            f"{marker}\n"
            "Do not add any other text."
        )
        ok_run, out = await _run_claude_code_print(
            prompt=prompt,
            chat_key=key,
            session_key=session_key,
            task_id="claude_mode_smoke",
            workdir=str(tmp_dir),
        )
        if not ok_run:
            with contextlib_suppress(Exception):
                tmp_file.unlink(missing_ok=True)
            return f"Smoke: **FAILED**\n{out}"

        # Verify edit happened (single-line marker) and clean up.
        body = ""
        try:
            raw = _dispatch_tool(
                "terminal",
                {
                    "command": (
                        f"python -c {shell_quote('import pathlib;print(pathlib.Path(' + repr(str(tmp_file)) + ').read_text())')}"
                    ),
                    "timeout": 10,
                },
                task_id="claude_mode_smoke_verify",
            )
            payload = json.loads(raw) if isinstance(raw, str) else {}
            body = _extract_terminal_text(payload)
        except Exception:
            body = ""
        with contextlib_suppress(Exception):
            tmp_file.unlink(missing_ok=True)
        if marker not in body:
            return "Smoke: **FAILED**\nClaude Code ran but did not apply the expected file edit in the temp dir."

        return (
            "Smoke: **OK**\n"
            "- backend: `claude-code-cli`\n"
            f"- allowedTools: `{_effective_allowed_tools(key)}`\n"
            f"- tools: `{_tools_from_allowed_tools(_effective_allowed_tools(key))}`\n"
            f"- permission-mode: `{_effective_permission_mode(key)}`\n"
            f"- yolo: `{'on' if _is_yolo_enabled(key) else 'off'}`\n"
            f"- temp edit+cleanup: `{tmp_file.name}`"
        )

    if head_l == "yolo":
        sub = tail.strip().lower()
        if sub in {"", "status"}:
            return _status_message(key) + "\n\nUsage: `/claude-mode yolo on|off|status`"
        if sub in {"on", "enable", "enabled", "true", "1"}:
            _update_chat(key, extra={"yolo": True})
            return _status_message(key)
        if sub in {"off", "disable", "disabled", "false", "0"}:
            _update_chat(key, extra={"yolo": False})
            return _status_message(key)
        return "Usage: `/claude-mode yolo on|off|status`"

    if head_l in {"approve", "deny"}:
        pending = _pending_approval(key)
        if not pending:
            return "Claude mode: no pending approval for this chat."

        pending_id = str(pending.get("id") or "").strip()
        token_or_rule = tail.strip()
        if token_or_rule and re.fullmatch(rf"[{_APPROVAL_ID_ALPHABET}]{{{_APPROVAL_ID_LEN}}}", token_or_rule):
            if token_or_rule != pending_id:
                return f"Claude mode: unknown approval id `{token_or_rule}` for this chat."
            token_or_rule = ""

        if head_l == "deny":
            _clear_pending_approval(key)
            return "Claude mode: denied and cleared the pending approval."

        tool_rule = token_or_rule or str(pending.get("tool_rule") or "").strip()
        if not tool_rule:
            return (
                "Claude mode: pending approval has no parsed tool rule.\n"
                "Retry with an explicit rule, e.g. `/claude-mode approve Bash(git status *)`."
            )

        # Persist chat-scoped allow rule (escape hatch; doesn't flip yolo).
        extras = _chat_allowed_tools_extra(key)
        if tool_rule not in extras:
            extras.append(tool_rule)
            _update_chat(key, extra={"allowed_tools_extra": extras})

        # Re-run the original prompt and clear pending on success (or refresh on re-block).
        prompt_path = pending.get("prompt_path")
        prompt_text = ""
        if isinstance(prompt_path, str) and prompt_path:
            with contextlib_suppress(Exception):
                prompt_text = _read_private_text(Path(prompt_path))
        if not prompt_text:
            _clear_pending_approval(key)
            return "Claude mode: pending approval expired (missing prompt). Please resend your message."

        ok_run, out = await _run_claude_code_print(
            prompt=prompt_text,
            chat_key=key,
            session_key=str(pending.get("session_key") or session_key or ""),
            task_id="claude_mode_approve",
        )
        if ok_run:
            _clear_pending_approval(key)
        return out

    return "Usage: `/claude-mode on|off|status|smoke|approve|deny|yolo`"


async def handle_internal_run(raw_args: str) -> str:
    pending = _pop_pending_prompt(raw_args)
    if pending is None:
        return "Claude mode: expired request. Please resend the message."

    ok, _info = _resolve_claude_code_cli_backend()
    if not ok:
        set_enabled(
            pending.chat_key,
            False,
            extra={
                "backend": "claude-code-cli",
                "tool_edit_active": False,
                "last_enable_error": "Claude Code CLI not found",
            },
        )
        backend, _toolful = _format_backend_status()
        return (
            "Claude mode: **off** (auto-disabled)\n"
            f"backend: {backend}\n\n"
            "Claude Code CLI is unavailable. Your message was not routed.\n"
            "Use `/hermes <message>` to talk to Hermes, or re-enable after setup."
        )

    ok_run, out = await _run_claude_code_print(
        prompt=pending.prompt,
        chat_key=pending.chat_key,
        session_key=pending.session_key,
        task_id="claude_mode_run",
        platform=pending.platform,
        chat_id=pending.chat_id,
        thread_id=pending.thread_id,
    )
    return out or None


async def handle_public_mode(raw_args: str) -> str:
    return "Use `/claude-mode on|off|status|smoke|approve|deny|yolo` from Telegram so the plugin can identify the chat."


def register(ctx: Any) -> None:
    global _CTX
    _CTX = ctx
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_command(
        "claude-mode",
        handle_public_mode,
        "Toggle chat-level Claude Code CLI mode",
        "on|off|status|smoke",
    )
    ctx.register_command(INTERNAL_MODE, handle_internal_mode, "Internal Kaze Claude mode control", "<packet>")
    ctx.register_command(INTERNAL_RUN, handle_internal_run, "Internal Kaze Claude Code runner", "<token>")
