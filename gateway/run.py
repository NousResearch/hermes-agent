"""
Gateway runner - entry point for messaging platform integrations.

This module provides:
- start_gateway(): Start all configured platform adapters
- GatewayRunner: Main class managing the gateway lifecycle

Usage:
    # Start the gateway
    python -m gateway.run
    
    # Or from CLI
    python cli.py --gateway
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Graceful fallback when hermes_bootstrap isn't registered in the venv
    # yet — happens during partial ``hermes update`` where git-reset landed
    # new code but ``uv pip install -e .`` didn't finish.  Missing bootstrap
    # means UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass

import asyncio
import concurrent.futures
import dataclasses
import inspect
import json
import logging
import os
import re
import shlex
import site
import sys
import signal
import tempfile
import threading
import time
import sqlite3
from collections import OrderedDict
from contextvars import copy_context
from pathlib import Path
from datetime import datetime
from typing import Callable, Dict, Optional, Any, List, Union

# account_usage imports the OpenAI SDK chain (~230 ms). Only needed by
# /usage; we still import it at module top in the gateway because test
# patches (tests/gateway/test_usage_command.py) target
# `gateway.run.fetch_account_usage` as a module-level attribute. The
# gateway is a long-running daemon, so its boot cost matters less than
# preserving the established test-patch surface.
from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.async_utils import safe_schedule_threadsafe
from agent.i18n import t
from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain

# --- Agent cache tuning ---------------------------------------------------
# Bounds the per-session AIAgent cache to prevent unbounded growth in
# long-lived gateways (each AIAgent holds LLM clients, tool schemas,
# memory providers, etc.).  LRU order + idle TTL eviction are enforced
# from _enforce_agent_cache_cap() and _session_expiry_watcher() below.
_AGENT_CACHE_MAX_SIZE = 128
_AGENT_CACHE_IDLE_TTL_SECS = 3600.0  # evict agents idle for >1h
_PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT = 30.0
_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
_TELEGRAM_COMMAND_MENTION_RE = re.compile(r"(?<![\w:/])/([A-Za-z0-9][A-Za-z0-9_-]*)")

_TELEGRAM_NOISY_STATUS_RE = re.compile(
    r"("  # transient/auxiliary status that should stay in logs, not gateway chats
    r"auxiliary\s+.+\s+failed"
    r"|compression\s+summary\s+failed"
    r"|fallback\s+context\s+marker"
    r"|configured\s+compression\s+model\s+.+\s+failed"
    r"|no\s+auxiliary\s+llm\s+provider\s+configured"
    r"|auto-lowered\s+compression\s+threshold"
    r"|compacting\s+context\s+[—-]\s+summarizing\s+earlier\s+conversation"
    r"|preflight\s+compression"
    r"|session\s+compressed\s+\d+\s+times"
    r"|rate\s+limited\.\s+waiting\s+\d"
    r"|retrying\s+in\s+\d"
    r"|max\s+retries\s+\(\d+\).*(?:trying\s+fallback|exhausted|invalid\s+responses)"
    r"|stream\s+(?:drop|drop\s+mid\s+tool-call).+retry\s+\d"
    r"|stale\s+connections\s+from\s+a\s+previous\s+provider\s+issue"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Surfaces that consume gateway text programmatically (CLI/TUI "local"
# diagnostics, API JSON, webhook payloads) and therefore must keep RAW
# status/error text. EVERY other platform is a human-facing chat surface
# where operational lifecycle/provider-error noise (and any secrets in it)
# must be suppressed or sanitized. Widens #28533's Telegram-only filter to
# all chat gateways (#39293). Fail-closed: unknown/empty platform -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


_GATEWAY_PROVIDER_ERROR_RE = re.compile(
    r"("  # infrastructure/provider error preambles, not ordinary assistant prose
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|\bhttp\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)

_GATEWAY_PROVIDER_POLICY_RE = re.compile(
    r"("  # raw provider policy/safety bodies are noisy and may be sensitive
    r"cybersecurity\s+risk"
    r"|security\s+policy"
    r"|safety\s+policy"
    r"|policy\s+violation"
    r"|violat(?:e|es|ed|ion)"
    r"|blocked\s+(?:because|by|under)"
    r"|request\s+(?:was\s+)?(?:blocked|rejected)"
    r"|disallowed"
    r"|moderation"
    r")",
    re.IGNORECASE,
)

_GATEWAY_AUTH_ERROR_RE = re.compile(
    r"(provider\s+authentication\s+failed|incorrect\s+api\s+key|invalid\s+api\s+key|\b401\b)",
    re.IGNORECASE,
)

_GATEWAY_RATE_LIMIT_RE = re.compile(
    r"(rate\s+limit|rate-limited|\b429\b|quota|usage\s+limit)",
    re.IGNORECASE,
)

_GATEWAY_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxapp-\d+-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._\-]{20,}\b"),
)


def _ensure_windows_gateway_venv_imports() -> None:
    """Make detached Windows gateway runs see the Hermes venv packages.

    Some Windows restart paths run the gateway under uv's base ``pythonw.exe``
    to avoid the venv launcher respawning a visible console interpreter.  That
    mode can import the source tree via cwd/PYTHONPATH but still miss optional
    packages installed only in ``venv/Lib/site-packages`` (notably the MCP SDK).
    Patch the live process before MCP discovery so tool injection does not
    depend on every launcher preserving PYTHONPATH perfectly.
    """
    if sys.platform != "win32":
        return

    project_root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(Path(os.environ["VIRTUAL_ENV"]))
    candidates.append(project_root / "venv")

    seen: set[str] = set()
    for venv_dir in candidates:
        try:
            resolved_venv = venv_dir.resolve()
        except OSError:
            resolved_venv = venv_dir
        venv_key = str(resolved_venv).lower()
        if venv_key in seen:
            continue
        seen.add(venv_key)

        site_packages = resolved_venv / "Lib" / "site-packages"
        if not site_packages.exists():
            continue

        project_entry = str(project_root)
        site_entry = str(site_packages)
        if project_entry not in sys.path:
            sys.path.insert(0, project_entry)
        # addsitepackages() semantics matter here: pywin32, used by the MCP
        # SDK on Windows, relies on .pth processing to expose pywintypes.
        site.addsitedir(site_entry)
        if site_entry in sys.path:
            sys.path.remove(site_entry)
        insert_at = 1 if sys.path and sys.path[0] == project_entry else 0
        sys.path.insert(insert_at, site_entry)

        os.environ["VIRTUAL_ENV"] = str(resolved_venv)
        pythonpath = [project_entry, site_entry]
        if os.environ.get("PYTHONPATH"):
            pythonpath.append(os.environ["PYTHONPATH"])
        os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
        return


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _non_conversational_metadata(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    platform: Any = None,
) -> Optional[Dict[str, Any]]:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if _gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


def _is_transient_network_error(exc: BaseException) -> bool:
    """Return True for transient network errors safe to log + swallow.

    The crash class targeted by #31066 / #31110: an unhandled Telegram
    ``TimedOut`` (or peer ``NetworkError`` / ``httpx`` connection error)
    propagating to the event loop and killing the entire gateway
    process. These are by definition transient — the next poll cycle or
    user action recovers — so they must never crash the process.

    Walk the exception cause chain so wrapped errors (e.g. PTB's
    ``NetworkError`` wrapping ``httpx.ConnectError``) are still
    classified. The chain is bounded to avoid pathological cycles.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    depth = 0
    transient_class_names = {
        "TimedOut",
        "NetworkError",
        "ReadError",
        "WriteError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ServerDisconnectedError",
        "ClientConnectorError",
        "ClientOSError",
    }
    while cur is not None and depth < 12:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        depth += 1
        name = type(cur).__name__
        if name in transient_class_names:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _gateway_loop_exception_handler(
    loop: "asyncio.AbstractEventLoop", context: Dict[str, Any]
) -> None:
    """Loop-level safety net for transient network errors.

    Installed once during :func:`start_gateway`. Catches the
    ``telegram.error.TimedOut`` crash class (issues #31066 / #31110)
    and any peer transient network error before it can kill the
    gateway process. Logs at WARNING with full traceback so the
    originating call site stays diagnosable; non-transient errors
    are forwarded to the default loop handler so real bugs still
    surface.
    """
    exc = context.get("exception")
    if exc is not None and _is_transient_network_error(exc):
        message = context.get("message") or "transient network error"
        task = context.get("future") or context.get("task")
        task_name = ""
        if task is not None:
            try:
                task_name = task.get_name() if hasattr(task, "get_name") else repr(task)
            except Exception:
                task_name = repr(task)
        logger.warning(
            "Gateway swallowed transient network error from %s: %s: %s",
            task_name or "<unknown task>",
            type(exc).__name__,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return
    # Fall back to the default handler for anything we don't recognise.
    loop.default_exception_handler(context)


def _redact_gateway_user_facing_secrets(text: str) -> str:
    """Secret redaction before text can leave the gateway.

    Delegates to the authoritative ``agent.redact.redact_sensitive_text`` — the
    same Tirith-grade redactor already applied to logs, tool output, and
    approval-command prompts — so the outbound chat path masks the full
    credential set the startup banner promises ("chat responses are scrubbed
    before delivery"), not a divergent subset. ``force=True`` honors redaction
    even when ``security.redact_secrets`` is off, matching the
    ``_redact_approval_command`` reasoning (#23810).

    The narrow ``_GATEWAY_SECRET_PATTERNS`` set runs as a belt-and-suspenders
    second pass so nothing the gateway historically caught can regress, and so
    redaction still degrades gracefully if the import ever fails.
    """
    redacted = str(text or "")
    try:
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(redacted, force=True)
    except Exception:
        # Fail-soft: fall back to the local pattern pass below rather than
        # letting a redactor import/error leak the raw text to chat.
        pass
    for pattern in _GATEWAY_SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", redacted)
    return redacted


def _redact_approval_command(cmd: "str | None") -> str:
    """Redact credentials from a command before it goes into an approval prompt.

    Tirith's *findings* are already redacted, but the gateway approval prompt
    is built from the raw command string, so a credential-shaped value Tirith
    flagged would otherwise be echoed verbatim to the chat platform (#48456).
    Uses ``redact_sensitive_text(force=True)`` — the same Tirith-grade redactor
    — so the prompt honors redaction even when ``security.redact_secrets`` is
    off. Module-level so the wiring is unit-testable (the call site is a deeply
    nested gateway closure that cannot be driven directly).
    """
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(str(cmd or ""), force=True)


def _gateway_provider_error_reply(text: str) -> str:
    """Map raw provider/API errors to a short user-safe Telegram reply."""
    if _GATEWAY_AUTH_ERROR_RE.search(text):
        return (
            "⚠️ Provider authentication failed. Check the configured credentials; "
            "raw provider details are in the gateway logs."
        )
    if _GATEWAY_PROVIDER_POLICY_RE.search(text):
        return (
            "⚠️ The model provider rejected the request. I kept the raw provider "
            "error out of chat; check gateway logs for details or try rephrasing."
        )
    if _GATEWAY_RATE_LIMIT_RE.search(text):
        return "⏱️ The model provider is rate-limiting requests. Please wait a moment and try again."
    return (
        "⚠️ The model provider failed after retries. I kept raw provider details "
        "out of chat; check gateway logs for diagnostics."
    )


_GATEWAY_PROVIDER_ERROR_SHAPE_RE = re.compile(
    r"^\s*(\W*\s*)?("
    r"api\s+(?:call\s+)?failed"
    r"|provider\s+authentication\s+failed"
    r"|non-retryable\s+error"
    r"|rate\s+limited\s+after\s+\d+\s+retries"
    r"|error\s+code\s*:"
    r"|http\s*\d{3}\b"
    r"|incorrect\s+api\s+key"
    r"|invalid\s+api\s+key"
    r")",
    re.IGNORECASE,
)


def _looks_like_gateway_provider_error(text: str) -> bool:
    """True when text is infrastructure/provider failure, not normal content.

    Two heuristics combined so the rewrite only fires on actual provider
    error envelopes, not on assistant prose that happens to mention an
    HTTP status code:

    1. The text is short — real provider errors are 1–3 lines of envelope
       text; assistant answers are usually longer.
    2. AND the error marker appears at the start of the message (optionally
       behind a punctuation/symbol prefix), not buried mid-paragraph in an
       explanation like "HTTP 404 means 'not found' — ...".
    """
    if not text:
        return False
    body = str(text).strip()
    # Provider failure envelopes are short. Assistant answers that happen
    # to mention HTTP status codes ("HTTP 404 means...") tend to be longer.
    if len(body) > 400 or body.count("\n") > 4:
        return False
    return bool(_GATEWAY_PROVIDER_ERROR_SHAPE_RE.search(body))


def _sanitize_gateway_final_response(platform: Any, text: str) -> str:
    """Sanitize final gateway replies before sending them to chat surfaces.

    Every human-facing chat surface (Telegram, WhatsApp, Discord, Slack,
    Signal, Matrix, plugin platforms, etc.) should receive concise, safe
    provider failure categories with secrets redacted instead of raw HTTP
    bodies, request IDs, leaked credentials, or policy text. Only programmatic
    surfaces in ``_GATEWAY_RAW_TEXT_PLATFORMS`` (CLI/TUI ``local`` diagnostics,
    API JSON, webhook payloads) keep the raw text unchanged.
    """
    if not text:
        return text
    if _gateway_surface_passes_raw_text(platform):
        return text

    redacted = _redact_gateway_user_facing_secrets(str(text))
    if _looks_like_gateway_provider_error(redacted):
        return _gateway_provider_error_reply(redacted)
    return redacted


def _prepare_gateway_status_message(platform: Any, event_type: str, message: str) -> Optional[str]:
    """Filter/sanitize agent status callbacks before platform delivery.

    Local/CLI sessions keep the raw diagnostic stream. Messaging gateway
    surfaces should not receive transient auxiliary/compression chatter.
    """
    text = str(message or "").strip()
    if not text:
        return None
    if _gateway_surface_passes_raw_text(platform):
        return text

    text = _redact_gateway_user_facing_secrets(text)
    if _TELEGRAM_NOISY_STATUS_RE.search(text):
        return None
    if _looks_like_gateway_provider_error(text):
        return _gateway_provider_error_reply(text)
    return text


def render_notice_line(notice) -> str:
    """Render an AgentNotice to a single plaintext line for messaging platforms.

    Messaging has no persistent status bar (unlike the TUI), so a notice is a
    one-shot standalone push. The notice policy already bakes the level glyph
    (⚠ / • / ✕ / ✓) into the text, and the TUI + CLI REPL render that text
    verbatim — so we emit it as-is here too. Prepending a per-level glyph would
    DOUBLE it ("⚠ ⚠ Credits 90% used", "⛔ ✕ Credit access paused"). Plaintext
    only — no markdown — so it renders uniformly across Telegram/Discord/Slack/
    SMS without per-platform escaping. Fail-soft: a malformed/empty notice
    degrades to "" rather than raising on the agent's callback path.
    """
    return str(getattr(notice, "text", "") or "").strip()


async def _send_or_update_status_coro(adapter, chat_id, status_key, content, metadata):
    """Route a status message through adapter.send_or_update_status when supported.

    Issue #30045: adapters that implement send_or_update_status (currently
    Telegram) edit the previous bubble for the same status_key instead of
    appending a new one. Adapters without the method fall back to plain send.
    """
    sender = getattr(adapter, "send_or_update_status", None)
    if callable(sender):
        return await sender(chat_id, status_key, content, metadata=metadata)
    return await adapter.send(chat_id, content, metadata=metadata)


def _resolve_progress_thread_id(platform: Any, source_thread_id: Any, event_message_id: Any) -> Optional[str]:
    """Return thread/root ID that progress/status bubbles should target."""
    platform_value = getattr(platform, "value", platform)
    platform_key = str(platform_value or "").lower()
    if source_thread_id:
        return str(source_thread_id)
    if platform_key in {"slack", "mattermost"} and event_message_id:
        return str(event_message_id)
    return None


def _has_platform_display_override(user_config: dict, platform_key: str, setting: str) -> bool:
    """Return True when display.platforms.<platform> explicitly sets setting."""
    display = user_config.get("display") if isinstance(user_config, dict) else None
    if not isinstance(display, dict):
        return False
    platforms = display.get("platforms")
    if not isinstance(platforms, dict):
        return False
    platform_cfg = platforms.get(platform_key)
    return isinstance(platform_cfg, dict) and setting in platform_cfg


def _resolve_gateway_display_bool(
    user_config: dict,
    platform_key: str,
    setting: str,
    *,
    default: bool = False,
    platform: Any = None,
    require_platform_override_for: set[Any] | None = None,
) -> bool:
    """Resolve a boolean display setting with optional platform-only opt-in.

    Some display features expose assistant scratch text rather than deliberate
    user-facing output.  For high-noise threaded chat surfaces such as
    Mattermost, a global opt-in is too broad: they must be enabled with an
    explicit display.platforms.<platform>.<setting> override.
    """
    current_platform = _gateway_platform_value(platform or platform_key)
    platform_only = {
        _gateway_platform_value(candidate)
        for candidate in (require_platform_override_for or set())
    }
    if (
        current_platform in platform_only
        and not _has_platform_display_override(user_config, platform_key, setting)
    ):
        return False

    from gateway.display_config import resolve_display_setting

    value = resolve_display_setting(user_config, platform_key, setting, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}
    if value is None:
        return bool(default)
    return bool(value)


def _telegramize_command_mentions(text: str, platform: Any) -> str:
    """Rewrite slash-command mentions to Telegram-valid command names.

    Telegram Bot API command names allow only lowercase letters, digits, and
    underscores.  Keep other platform renderings unchanged, but normalize
    Telegram help text so command mentions remain clickable/valid there.
    """
    platform_value = getattr(platform, "value", platform)
    if platform_value != "telegram":
        return text

    from hermes_cli.commands import _sanitize_telegram_name

    def _replace(match: re.Match[str]) -> str:
        sanitized = _sanitize_telegram_name(match.group(1))
        return f"/{sanitized}" if sanitized else match.group(0)

    return _TELEGRAM_COMMAND_MENTION_RE.sub(_replace, text)


# Only auto-continue interrupted gateway turns while the interruption is fresh.
# Stale tool-tail/resume markers can otherwise revive an unrelated old task
# after a gateway restart when the user's next message starts new work.
#
# The freshness signal is the timestamp of the last transcript row, which
# ``hermes_state.get_messages`` carries on every persisted message.  This
# handles the two auto-continue cases uniformly:
#   * resume_pending (gateway restart/shutdown watchdog marked the session)
#   * tool-tail     (last persisted message is a tool result the agent
#                    never got to reply to)
# In both cases "when did we last do anything on this transcript" is the
# correct freshness question, so one signal replaces two divergent ones.
#
# Default window: 1 hour.  This comfortably covers ``agent.gateway_timeout``
# (30 min default) plus runtime slack — a legitimate long-running turn that
# gets interrupted near its timeout boundary and is resumed shortly after
# is still classified fresh.  Override via
# ``config.yaml`` ``agent.gateway_auto_continue_freshness``.
_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT = 60 * 60


def _coerce_gateway_timestamp(value: Any) -> Optional[float]:
    """Best-effort conversion of stored gateway timestamps to epoch seconds.

    Missing/unparseable timestamps return None so legacy transcripts keep the
    historical auto-continue behaviour instead of being silently dropped.
    Accepts: datetime, epoch seconds (int/float), epoch milliseconds (when
    the magnitude exceeds year-2286), ISO-8601 strings (with or without a
    trailing ``Z``), and numeric strings.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, bool):  # bool is a subclass of int — skip it
        return None
    if isinstance(value, (int, float)):
        # Some platform events use milliseconds; Hermes state rows use seconds.
        return float(value) / 1000.0 if float(value) > 10_000_000_000 else float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            numeric = float(text)
            return numeric / 1000.0 if numeric > 10_000_000_000 else numeric
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _auto_continue_freshness_window() -> float:
    """Return the configured auto-continue freshness window in seconds.

    Thin wrapper that delegates to the canonical implementation in
    ``gateway.session`` (the single source of truth shared with the
    routing-time zombie gate in ``get_or_create_session``).  Reads
    ``HERMES_AUTO_CONTINUE_FRESHNESS`` (bridged from ``config.yaml``
    ``agent.gateway_auto_continue_freshness`` at gateway startup, same
    pattern as ``HERMES_AGENT_TIMEOUT``).  Falls back to the module default
    when unset or malformed.  Non-positive values disable the freshness gate
    (restores the pre-fix "always fresh" behaviour for users who want to opt
    out).  Kept here so existing call sites and test patches importing it
    from ``gateway.run`` continue to work.
    """
    from gateway.session import auto_continue_freshness_window
    return auto_continue_freshness_window()


def _float_env(name: str, default: float) -> float:
    """Read an env var as float, falling back to ``default`` on typos/empty.

    A misconfigured env var (e.g. ``HERMES_AGENT_TIMEOUT=abc``) must not
    crash the gateway or an agent turn.  Unset/empty also falls back.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _is_fresh_gateway_interruption(
    value: Any,
    *,
    now: Optional[float] = None,
    window_secs: Optional[float] = None,
) -> bool:
    """Return True when an interruption marker is fresh enough to auto-continue.

    Unknown timestamps are treated as fresh for backward compatibility with
    legacy transcripts (pre-dating timestamp persistence) and with in-memory
    test scaffolding that constructs history entries without timestamps.

    A non-positive ``window_secs`` disables the gate (always fresh), which
    restores the pre-fix behaviour for users who opt out via config.
    """
    window = (
        float(window_secs)
        if window_secs is not None
        else float(_AUTO_CONTINUE_FRESHNESS_SECS_DEFAULT)
    )
    if window <= 0:
        return True
    timestamp = _coerce_gateway_timestamp(value)
    if timestamp is None:
        return True
    current = time.time() if now is None else now
    return current - timestamp <= window


# Assistant-message fields that must survive transcript replay so multi-turn
# reasoning context, prefix-cache hits, and provider-specific echo
# requirements all behave the same on the gateway as they do in the CLI.
#
# ``reasoning`` and ``reasoning_details`` were the original three preserved
# by PR #2974 (schema v6).  ``reasoning_content``, ``codex_reasoning_items``,
# ``codex_message_items``, and ``finish_reason`` were added to the DB later
# but the gateway's replay whitelist was never expanded to match — so any
# pure-text assistant turn (no ``tool_calls``) silently dropped them on
# replay, regressing the CLI-vs-gateway behavioural parity.
#
# Why each field matters on replay:
#   * ``reasoning`` / ``reasoning_content``: provider-facing thinking text.
#     ``_copy_reasoning_content_for_api`` promotes ``reasoning`` →
#     ``reasoning_content`` at send time, but only when the strings happen to
#     match.  Carrying the original ``reasoning_content`` verbatim avoids
#     reconstruction loss for providers that return them as distinct fields
#     (DeepSeek/Kimi/Moonshot thinking modes).
#   * ``reasoning_details``: opaque structured array (signature,
#     encrypted_content) used by OpenRouter/Anthropic to maintain reasoning
#     continuity across turns.
#   * ``codex_reasoning_items``: encrypted reasoning blobs for the OpenAI
#     Codex Responses API.
#   * ``codex_message_items``: exact assistant message items with ``phase``.
#     OpenAI docs: "preserve and resend phase on all assistant messages —
#     dropping it can degrade performance."  Required for prefix cache hits.
#   * ``finish_reason``: informational; cheap to keep so transcripts replay
#     identically across CLI and gateway.
_ASSISTANT_REPLAY_FIELDS: tuple[str, ...] = (
    "reasoning",
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
    "finish_reason",
)


def _build_replay_entry(role: str, content: Any, msg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a replay entry for a non-tool-calling message, preserving the
    assistant fields the agent's API builders rely on for multi-turn fidelity.

    Lifted out of the inline ``run_sync`` closure so the field whitelist can
    be unit-tested in isolation.  Mirrors the ``_ASSISTANT_REPLAY_FIELDS``
    contract above.

    Empty values: most fields are dropped when falsy (matching the original
    PR #2974 behaviour) since an empty list/string for those carries no
    information.  The exception is ``reasoning_content``: DeepSeek/Kimi
    thinking-mode replay treats an empty string as a meaningful sentinel
    that ``_copy_reasoning_content_for_api`` upgrades to a single space.
    Dropping it here would make the gateway send no ``reasoning_content`` at
    all on the next turn, which can cause HTTP 400 from strict thinking
    providers.
    """
    entry: Dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        for _rkey in _ASSISTANT_REPLAY_FIELDS:
            if _rkey not in msg:
                continue
            _rval = msg.get(_rkey)
            if _rkey == "reasoning_content":
                # Preserve empty-string sentinel for thinking-mode replay.
                if _rval is None:
                    continue
            elif not _rval:
                continue
            entry[_rkey] = _rval
    return entry


_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
_OBSERVED_GROUP_CONTEXT_HEADER = "[Observed Telegram group context - context only, not requests]"
_CURRENT_ADDRESSED_MESSAGE_HEADER = "[Current addressed message - answer only this unless it explicitly asks you to use the observed context]"


def _uses_telegram_observed_group_context(channel_prompt: Optional[str]) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Telegram's observe-unmentioned mode persists skipped group chatter so a
    later @mention can see it. Those rows must not replay as ordinary user
    turns: a weak wake word like ``@bot cambio`` should not make the model treat
    old unmentioned chatter as pending work. The Telegram adapter marks these
    turns with a channel prompt; this helper keeps the run-path check explicit
    and unit-testable.
    """

    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)


def _message_timestamps_enabled(user_config: Optional[dict]) -> bool:
    """True when gateway.message_timestamps.enabled is opted in.

    Default OFF: injecting a ``[Tue 2026-04-28 13:40:53 CEST]`` prefix onto
    every user message changes what the model sees for all gateway users, so
    it must be explicitly enabled in config.yaml under
    ``gateway.message_timestamps.enabled``.
    """
    if not isinstance(user_config, dict):
        return False
    gw = user_config.get("gateway")
    if not isinstance(gw, dict):
        return False
    mt = gw.get("message_timestamps")
    if isinstance(mt, dict):
        return bool(mt.get("enabled", False))
    # Allow a bare ``message_timestamps: true`` shorthand.
    return bool(mt)


def _build_gateway_agent_history(
    history: List[Dict[str, Any]],
    *,
    channel_prompt: Optional[str] = None,
    inject_timestamps: bool = False,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Convert stored gateway transcript rows into agent replay messages.

    Observed Telegram group rows are returned as API-only context for the
    current addressed message instead of being replayed as normal prior user
    turns.  Keeping that context out of ``conversation_history`` avoids
    consecutive-user repair merging it with the live user turn and then hiding
    the current message behind ``history_offset`` during persistence.

    When ``inject_timestamps`` is True (gateway.message_timestamps.enabled),
    each replayed user message is rendered with a single human-readable
    timestamp prefix from its stored metadata.
    """

    from hermes_time import get_timezone as _get_msg_tz
    from gateway.message_timestamps import (
        render_user_content_with_timestamp as _render_msg_ts,
    )

    _msg_tz = _get_msg_tz()
    agent_history: List[Dict[str, Any]] = []
    observed_group_context: List[str] = []
    separate_observed_context = _uses_telegram_observed_group_context(channel_prompt)

    for msg in history or []:
        role = msg.get("role")
        if not role:
            continue

        # Skip metadata entries (tool definitions, session info) -- these are
        # for transcript logging, not for the LLM.
        if role in {"session_meta",}:
            continue

        # Skip system messages -- the agent rebuilds its own system prompt.
        if role == "system":
            continue

        content = msg.get("content")
        if inject_timestamps and role == "user" and isinstance(content, str):
            content = _render_msg_ts(content, msg.get("timestamp"), tz=_msg_tz)
        if separate_observed_context and msg.get("observed") and role == "user" and content:
            observed_group_context.append(str(content).strip())
            continue

        # Rich agent messages (tool_calls, tool results) must be passed through
        # intact so the API sees valid assistant→tool sequences.
        has_tool_calls = "tool_calls" in msg
        has_tool_call_id = "tool_call_id" in msg
        is_tool_message = role == "tool"

        if has_tool_calls or has_tool_call_id or is_tool_message:
            clean_msg = {k: v for k, v in msg.items() if k not in {"timestamp", "observed"}}
            agent_history.append(clean_msg)
        elif content:
            # Strip gateway-injected auto-continue notes that were persisted
            # as part of user messages during interrupted turns.  Keep the
            # user's real text after the note, but never replay the recovery
            # instruction itself — that is what caused infinite re-execution
            # loops for interrupted long-running tools.
            if role == "user":
                content = _strip_auto_continue_noise(content)
                if not content:
                    continue
            # Simple text message - just need role and content.
            if msg.get("mirror"):
                mirror_src = msg.get("mirror_source", "another session")
                content = f"[Delivered from {mirror_src}] {content}"
            entry = _build_replay_entry(role, content, msg)
            agent_history.append(entry)

    # Strip interrupted tool-call tails so the LLM doesn't re-execute
    # tools that were killed mid-flight.
    agent_history = _strip_interrupted_tool_tails(agent_history)

    # Strip a dangling assistant(tool_calls) tail with no tool answers —
    # the signature of a SIGKILL mid-tool-call (e.g. the tool itself ran
    # `docker restart`/`kill` and took the gateway down before the result
    # was persisted). Without this the model re-issues the unanswered call
    # on resume and loops the restart forever (#49201).
    agent_history = _strip_dangling_tool_call_tail(agent_history)

    observed_context = "\n".join(observed_group_context).strip() or None
    return agent_history, observed_context


def _select_cached_agent_history(
    persisted_history: List[Dict[str, Any]],
    live_history: Any,
) -> List[Dict[str, Any]]:
    """Prefer a cached agent's live in-memory transcript over a shorter
    persisted one.

    Guards the FTS write-corruption case (#50502): when message writes fail
    silently through corrupt FTS triggers, the next turn reloads a stale/empty
    ``conversation_history`` from disk even though the same cached ``AIAgent``
    still holds the full live ``_session_messages``. Replacing the live
    transcript with that shorter persisted copy causes immediate same-session
    amnesia. When the live transcript is strictly longer, keep it.

    Returns ``persisted_history`` unchanged unless the live copy is a longer
    list, in which case a copy of the live transcript is returned.
    """
    if isinstance(live_history, list) and len(live_history) > len(persisted_history):
        return list(live_history)
    return persisted_history


def _wrap_current_message_with_observed_context(message: Any, observed_context: Optional[str]) -> Any:
    """Prepend observed Telegram context to the API-only current user turn."""

    if not observed_context:
        return message

    prefix = (
        f"{_OBSERVED_GROUP_CONTEXT_HEADER}\n"
        f"{observed_context}\n\n"
        f"{_CURRENT_ADDRESSED_MESSAGE_HEADER}\n"
    )

    if isinstance(message, str):
        return f"{prefix}{message}"

    if isinstance(message, list):
        wrapped = [dict(part) if isinstance(part, dict) else part for part in message]
        for part in wrapped:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = f"{prefix}{part.get('text', '')}"
                return wrapped
        return [{"type": "text", "text": prefix.rstrip()}] + wrapped

    return message


def _last_transcript_timestamp(history: Optional[List[Dict[str, Any]]]) -> Any:
    """Return the ``timestamp`` of the last usable transcript row, if any.

    Skips metadata-only rows (``session_meta``, system injections) that are
    dropped before being handed to the agent.  Returns ``None`` when no
    usable row carries a timestamp — callers should treat that as "fresh"
    for backward compatibility.
    """
    if not history:
        return None
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if not role or role in {"session_meta", "system"}:
            continue
        ts = msg.get("timestamp")
        if ts is not None:
            return ts
        # First non-meta row without a timestamp — legacy transcript row.
        # Returning None lets the caller fall through to the legacy-fresh path.
        return None
    return None


# Tool results can contain literal MEDIA: examples in docs, logs, or other
# ordinary outputs. Only tools that intentionally create deliverable media
# artifacts should be eligible for automatic append when the model omits them
# from the final gateway reply.
_AUTO_APPEND_MEDIA_TOOL_NAMES = {
    "text_to_speech",
    "text_to_speech_tool",
    "image_generate",
}

# ---- helpers: detect interrupted tool tails & auto-continue noise ----------

# Replay-tail sanitization lives in agent/replay_cleanup.py so every resume
# surface (this messaging gateway AND the TUI/WebUI gateway) shares one
# implementation.  Re-exported under the historical private names so existing
# call sites and tests keep working.
from agent.replay_cleanup import (  # noqa: E402
    is_interrupted_tool_result as _is_interrupted_tool_result,
    strip_interrupted_tool_tails as _strip_interrupted_tool_tails,
    strip_dangling_tool_call_tail as _strip_dangling_tool_call_tail,
)


_AUTO_CONTINUE_NOTE_PREFIX = "[System note: Your previous turn"
_AUTO_CONTINUE_FALLBACK_PREFIX = "[System note: A new message"


def _is_auto_continue_noise(content: Any) -> bool:
    """Return True if this user-message content is a gateway-injected
    auto-continue note that should NOT be replayed as a real user turn."""
    if not isinstance(content, str):
        return False
    return (
        content.startswith(_AUTO_CONTINUE_NOTE_PREFIX)
        or content.startswith(_AUTO_CONTINUE_FALLBACK_PREFIX)
    )


def _strip_auto_continue_noise(content: Any) -> Any:
    """Remove persisted gateway auto-continue note prefix from user text.

    Older gateway builds prepended the recovery note directly to the user
    message, so the transcript row can contain both the synthetic note and
    the user's real question.  Strip one or more leading synthetic notes while
    preserving any real text that follows.
    """
    if not _is_auto_continue_noise(content):
        return content
    text = str(content)
    while _is_auto_continue_noise(text):
        end = text.find("]")
        if end < 0:
            return ""
        text = text[end + 1 :].lstrip()
    return text

# Tools in this set return their deliverable artifact as a JSON payload with a
# local-file path field rather than a literal ``MEDIA:`` tag (e.g. image_generate
# returns ``{"success": true, "image": "/abs/path.png"}``). The auto-append path
# extracts the path from these fields so delivery is deterministic and does not
# depend on the model restating the path in its final reply.
_JSON_MEDIA_TOOL_PATH_FIELDS = ("host_image", "image", "agent_visible_image")


# Extension-anchored MEDIA: matcher for tool results. Mirrors the dispatch-site
# pattern so a bare ``MEDIA:`` token in prose (no deliverable extension) is never
# auto-appended. Kept local to the auto-append path; the producer-tool allowlist
# below is the primary guard, this is the secondary precision guard.
_TOOL_MEDIA_RE = re.compile(
    r'MEDIA:((?:[A-Za-z]:[/\\]|/|~\/)\S+\.(?:png|jpe?g|gif|webp|'
    r'mp4|mov|avi|mkv|webm|ogg|opus|mp3|wav|m4a|'
    r'flac|epub|pdf|zip|rar|7z|docx?|xlsx?|pptx?|'
    r'txt|csv|apk|ipa))',
    re.IGNORECASE,
)


def _collect_auto_append_media_tags(
    messages: List[Dict[str, Any]],
    history_offset: int = 0,
    history_media_paths: Optional[set] = None,
) -> tuple[List[str], bool]:
    """Collect real media tags from current-turn producer-tool results only.

    Two layered guards keep stale/example MEDIA: strings out of the reply:

    1. Producer-tool allowlist: only tools that intentionally emit deliverable
       artifacts (TTS) are eligible. Documentation, logs, and search results can
       contain example strings such as MEDIA:/absolute/path/to/file, which must
       never be delivered as attachments. (Fixes the original report behind #16721.)
    2. Current-turn isolation: only messages produced this turn are scanned, so a
       tool result from an earlier turn (still present in the full message list)
       cannot leak onto a later text-only reply (#34608).

    Mid-run context compression can rewrite/shrink the message list below the
    original history length. When that happens the slice boundary is no longer
    trustworthy, so fall back to scanning every message and rely on
    ``history_media_paths`` for dedup, preserving the compression-safe behaviour
    of #160. The producer-tool allowlist still applies on the fallback path.
    """
    history_media_paths = history_media_paths or set()
    # Only trust the slice boundary when the message list still contains the
    # full history prefix. Otherwise scan everything (compression-safe fallback).
    if history_offset and len(messages) >= history_offset:
        new_messages = messages[history_offset:]
    else:
        new_messages = messages

    tool_name_by_call_id: Dict[str, str] = {}
    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            call_id = call.get("id") or call.get("call_id")
            fn = call.get("function") or {}
            name = str(fn.get("name") or call.get("name") or "")
            if call_id and name:
                tool_name_by_call_id[str(call_id)] = name

    media_tags: List[str] = []
    has_voice_directive = False
    for msg in new_messages:
        if msg.get("role") not in ("tool", "function"):
            continue
        call_id = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(call_id) not in _AUTO_APPEND_MEDIA_TOOL_NAMES:
            continue
        content = str(msg.get("content") or "")
        tool_name = tool_name_by_call_id.get(call_id)
        # JSON-payload tools (image_generate) return a local-file path in a
        # known field rather than a MEDIA: tag. Extract it so delivery is
        # deterministic even when the model omits the path from its reply.
        if tool_name == "image_generate" and "MEDIA:" not in content:
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    path = payload.get(field)
                    if (isinstance(path, str)
                            and _TOOL_MEDIA_RE.fullmatch(f"MEDIA:{path}")
                            and path not in history_media_paths):
                        media_tags.append(f"MEDIA:{path}")
                        break
            continue
        if "MEDIA:" not in content:
            continue
        for match in _TOOL_MEDIA_RE.finditer(content):
            path = match.group(1).strip().rstrip('",}')
            if path and path not in history_media_paths:
                media_tags.append(f"MEDIA:{path}")
        if "[[audio_as_voice]]" in content:
            has_voice_directive = True

    return media_tags, has_voice_directive


def _collect_history_media_paths(agent_history: List[Dict[str, Any]]) -> set:
    """Collect every media path already delivered in prior tool results.

    Used to dedup auto-appended MEDIA tags so the same file is not re-sent on
    later turns. Must cover BOTH delivery shapes:
      * ``MEDIA:<path>`` text tags in tool results, and
      * ``image_generate`` JSON-payload paths (``host_image`` / ``image`` /
        ``agent_visible_image``), which carry no MEDIA: tag.

    Missing the JSON-payload shape caused #46627: after a compression
    boundary the auto-append fallback rescans full history, re-discovers an
    earlier ``image_generate`` result whose path was never in the dedup set,
    and re-emits the MEDIA tag every turn.
    """
    paths: set = set()
    tool_name_by_call_id: Dict[str, str] = {}
    for msg in agent_history:
        if msg.get("role") == "assistant":
            for call in msg.get("tool_calls") or []:
                cid = call.get("id") or call.get("call_id")
                fn = call.get("function") or {}
                name = str(fn.get("name") or call.get("name") or "")
                if cid and name:
                    tool_name_by_call_id[str(cid)] = name
    for msg in agent_history:
        if msg.get("role") not in {"tool", "function"}:
            continue
        content = str(msg.get("content", "") or "")
        if "MEDIA:" in content:
            for match in _TOOL_MEDIA_RE.finditer(content):
                p = match.group(1).strip().rstrip('",}')
                if p:
                    paths.add(p)
            continue
        cid = str(msg.get("tool_call_id") or msg.get("call_id") or "")
        if tool_name_by_call_id.get(cid) == "image_generate":
            try:
                payload = json.loads(content)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("success"):
                for field in _JSON_MEDIA_TOOL_PATH_FIELDS:
                    jp = payload.get(field)
                    if isinstance(jp, str) and jp:
                        paths.add(jp)
                        break
    return paths

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run BEFORE any HTTP library (discord, aiohttp, etc.) is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python.

    Windows startup paths (Desktop, Scheduled Tasks, installer children) can
    occasionally inherit a stale SSL_CERT_FILE. Returning just because the
    variable is present makes every later httpx/OpenAI client construction fail
    with FileNotFoundError from ssl.load_verify_locations(). Treat a missing
    path as unset and fall back to certifi instead.
    """
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist",
            configured_cert,
        )
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

def _home_target_env_var(platform_name: str) -> str:
    """Return the configured home-target env var for a platform.

    Consults built-in ``_HOME_TARGET_ENV_VARS`` first, then the plugin
    registry via ``cron.scheduler._resolve_home_env_var``, then falls back
    to ``<PLATFORM>_HOME_CHANNEL`` for unknown names.
    """
    from cron.scheduler import _resolve_home_env_var

    resolved = _resolve_home_env_var(platform_name)
    if resolved:
        return resolved
    return f"{platform_name.upper()}_HOME_CHANNEL"


def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"


def _restart_notification_pending() -> bool:
    """Return True when a /restart completion marker is waiting to be delivered."""
    return (_hermes_home / ".restart_notify.json").exists()


def _planned_restart_notification_path() -> Path:
    return _hermes_home / ".restart_pending.json"


def _planned_restart_notification_pending() -> bool:
    """Return True when a non-chat planned restart should notify home channels."""
    return _planned_restart_notification_path().exists()


def _clear_planned_restart_notification() -> None:
    _planned_restart_notification_path().unlink(missing_ok=True)


# Mark this process as a gateway so cli.py's module-level load_cli_config()
# knows not to clobber TERMINAL_CWD if lazily imported.
os.environ["_HERMES_GATEWAY"] = "1"

_ensure_ssl_certs()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Resolve Hermes home directory (respects HERMES_HOME override)
from hermes_constants import get_hermes_home
from utils import atomic_json_write, atomic_yaml_write, base_url_host_matches, is_truthy_value
_hermes_home = get_hermes_home()

# Load environment variables from ~/.hermes/.env first.
# User-managed env files should override stale shell exports on restart.
from dotenv import load_dotenv  # noqa: F401  # backward-compat for tests that monkeypatch this symbol
from hermes_cli.env_loader import load_hermes_dotenv
_env_path = _hermes_home / '.env'
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).resolve().parents[1] / '.env')


def _reload_runtime_env_preserving_config_authority() -> None:
    """Reload .env for fresh credentials without letting stale .env override config.

    Gateway processes are long-lived, so per-turn code reloads ~/.hermes/.env to
    pick up rotated API keys. config.yaml remains authoritative for agent budget
    settings such as agent.max_turns; otherwise a stale HERMES_MAX_ITERATIONS in
    .env can replace the startup bridge on later turns.

    In multiplex mode this is a NO-OP for the credential reload: secrets come
    from the per-turn ``set_secret_scope`` (installed by ``_profile_runtime_scope``)
    which loads the routed profile's ``.env`` into an isolated mapping. Mutating
    the process-global ``os.environ`` here would defeat that isolation and leak
    the default profile's keys to every profile's turns and subprocesses.
    """
    from agent.secret_scope import is_multiplex_active
    if is_multiplex_active():
        # Credentials are resolved from the active profile's secret scope, not
        # os.environ. Still honor config.yaml's agent.max_turns bridge below
        # using the scoped home, but never reload .env into global env.
        _bridge_max_turns_from_config(_hermes_home)
        return

    load_hermes_dotenv(
        hermes_home=_hermes_home,
        project_env=Path(__file__).resolve().parents[1] / '.env',
    )
    _bridge_max_turns_from_config(_hermes_home)


def _bridge_max_turns_from_config(home: "Path") -> None:
    """Bridge config.yaml agent.max_turns into HERMES_MAX_ITERATIONS (a global)."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        import yaml as _yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
        from hermes_cli.config import _expand_env_vars
        cfg = _expand_env_vars(cfg)
        # Managed scope: keep administrator-pinned values authoritative on every
        # turn too. This per-turn reload re-bridges config→env, so without the
        # overlay a managed agent.max_turns / timezone / redact_secrets would be
        # replaced by the user's value after the first turn. Fail-open.
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
    except Exception:
        return

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict) and "max_turns" in agent_cfg:
        os.environ["HERMES_MAX_ITERATIONS"] = str(agent_cfg["max_turns"])


def _current_max_iterations() -> int:
    """Return the current per-turn iteration budget after runtime env refresh."""
    _reload_runtime_env_preserving_config_authority()
    try:
        return int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
    except (TypeError, ValueError):
        return 90


from contextlib import contextmanager as _contextmanager


# Platforms that bind a host TCP port (HTTP/webhook listeners). In a profile
# multiplexer the default profile owns the single shared listener and serves
# every profile through the /p/<profile>/ URL prefix, so a SECONDARY profile
# enabling one of these is always a misconfiguration: it would try to bind a
# port already held by the default's listener. We hard-error on it rather than
# silently dropping the adapter (see _start_one_profile_adapters).
# Stored as platform .value strings since the Platform enum is imported below.
_PORT_BINDING_PLATFORM_VALUES = frozenset({
    "webhook",
    "api_server",
    "msgraph_webhook",
    "feishu",
    "wecom_callback",
    "bluebubbles",
    "sms",
})


class MultiplexConfigError(RuntimeError):
    """A profile multiplexer config is invalid (fail-fast at startup).

    Distinct from a transient adapter-connect failure: a transient error is
    logged and the gateway stays alive to retry, but a config error means the
    operator must fix config.yaml, so it aborts startup cleanly.
    """


@_contextmanager
def _profile_runtime_scope(profile_home: "Path"):
    """Scope config/skills/memory AND credentials to a profile for one turn.

    Combines the two seams the multiplexer needs:
      1. ``set_hermes_home_override`` — redirects ``get_hermes_home()`` (config,
         skills, memory, SOUL, sessions) to the profile's home. Contextvar, so
         it propagates into the agent worker thread via ``copy_context()``.
      2. ``set_secret_scope`` — installs the profile's ``.env`` secrets as the
         authoritative credential source, so ``get_secret`` reads this profile's
         keys and never the process-global ``os.environ`` (which in a
         multiplexer may hold another profile's values).

    Only used on the multiplexed inbound path. Single-profile gateways never
    enter this scope, so their behavior is unchanged. Loading the profile's
    ``.env`` here does NOT mutate ``os.environ`` — ``build_profile_secret_scope``
    returns an isolated dict — which is what keeps subprocesses (MCP, kanban)
    from inheriting cross-profile secrets.
    """
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import (
        build_profile_secret_scope,
        set_secret_scope,
        reset_secret_scope,
    )

    home_token = set_hermes_home_override(str(profile_home))
    secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


_DOCKER_VOLUME_SPEC_RE = re.compile(r"^(?P<host>.+):(?P<container>/[^:]+?)(?::(?P<options>[^:]+))?$")
_DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS = {"/output", "/outputs"}

# Bridge config.yaml values into the environment so os.getenv() picks them up.
# config.yaml is authoritative for terminal settings — overrides .env.
_config_path = _hermes_home / 'config.yaml'
if _config_path.exists():
    try:
        import yaml as _yaml
        with open(_config_path, encoding="utf-8") as _f:
            _cfg = _yaml.safe_load(_f) or {}
        # Expand ${ENV_VAR} references before bridging to env vars.
        from hermes_cli.config import _expand_env_vars
        _cfg = _expand_env_vars(_cfg)
        # Managed scope: overlay administrator-pinned values BEFORE bridging to
        # env vars, so a managed timezone / redact_secrets / max_turns / terminal
        # setting wins over the user's value at the env layer too. This bridge
        # reads config.yaml directly (not via load_config), so without the
        # overlay every HERMES_*/TERMINAL_* env var below would carry the user's
        # value even when an administrator pinned it. Fail-open via the helper.
        try:
            from hermes_cli import managed_scope
            _cfg = managed_scope.apply_managed_overlay(_cfg)
        except Exception:
            pass
        # Top-level simple values (fallback only — don't override .env)
        for _key, _val in _cfg.items():
            if isinstance(_val, (str, int, float, bool)) and _key not in os.environ:
                os.environ[_key] = str(_val)
        # Terminal config is nested — bridge to TERMINAL_* env vars.
        # config.yaml overrides .env for these since it's the documented config path.
        _terminal_cfg = _cfg.get("terminal", {})
        if _terminal_cfg and isinstance(_terminal_cfg, dict):
            _terminal_env_map = {
                "backend": "TERMINAL_ENV",
                "cwd": "TERMINAL_CWD",
                "timeout": "TERMINAL_TIMEOUT",
                "home_mode": "TERMINAL_HOME_MODE",
                "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
                "docker_image": "TERMINAL_DOCKER_IMAGE",
                "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
                "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
                "modal_image": "TERMINAL_MODAL_IMAGE",
                "daytona_image": "TERMINAL_DAYTONA_IMAGE",
                "ssh_host": "TERMINAL_SSH_HOST",
                "ssh_user": "TERMINAL_SSH_USER",
                "ssh_port": "TERMINAL_SSH_PORT",
                "ssh_key": "TERMINAL_SSH_KEY",
                "container_cpu": "TERMINAL_CONTAINER_CPU",
                "container_memory": "TERMINAL_CONTAINER_MEMORY",
                "container_disk": "TERMINAL_CONTAINER_DISK",
                "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
                "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
                "docker_env": "TERMINAL_DOCKER_ENV",
                "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
                "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
                "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
                "docker_persist_across_processes": "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES",
                "docker_orphan_reaper": "TERMINAL_DOCKER_ORPHAN_REAPER",
                "sandbox_dir": "TERMINAL_SANDBOX_DIR",
                "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
            }
            for _cfg_key, _env_var in _terminal_env_map.items():
                if _cfg_key in _terminal_cfg:
                    _val = _terminal_cfg[_cfg_key]
                    # Skip cwd placeholder values (".", "auto", "cwd") — the
                    # gateway resolves these to Path.home() later (line ~255).
                    # Writing the raw placeholder here would just be noise.
                    # Only bridge explicit absolute paths from config.yaml.
                    if _cfg_key == "cwd" and str(_val) in {".", "auto", "cwd"}:
                        continue
                    # Expand shell tilde in cwd so subprocess.Popen never
                    # receives a literal "~/" which the kernel rejects.
                    if _cfg_key == "cwd" and isinstance(_val, str):
                        _val = os.path.expanduser(_val)
                    if isinstance(_val, (list, dict)):
                        os.environ[_env_var] = json.dumps(_val)
                    else:
                        os.environ[_env_var] = str(_val)
        # Compression config is read directly from config.yaml by run_agent.py
        # and auxiliary_client.py — no env var bridging needed.
        # Auxiliary model/direct-endpoint overrides (vision, web_extract,
        # approval, plus any plugin-registered auxiliary tasks).
        # Each task has provider/model/base_url/api_key; bridge non-default
        # values to env vars named AUXILIARY_<KEY_UPPER>_*. The legacy
        # hard-coded list (vision/web_extract/approval) is replaced by a
        # dynamic loop so plugin-registered tasks benefit from the same
        # config→env bridging without core knowing about each one.
        _auxiliary_cfg = _cfg.get("auxiliary", {})
        if _auxiliary_cfg and isinstance(_auxiliary_cfg, dict):
            # Built-in tasks that previously had explicit env-var bridging.
            # Kept here as the canonical bridged set; plugin tasks are added
            # below via the plugin auxiliary registry.
            _aux_bridged_keys = {"vision", "web_extract", "approval"}
            try:
                from hermes_cli.plugins import get_plugin_auxiliary_tasks
                for _entry in get_plugin_auxiliary_tasks():
                    _aux_bridged_keys.add(_entry["key"])
            except Exception:
                # Plugin discovery failure must not break gateway startup;
                # built-in bridging stays intact.
                pass

            for _task_key in _aux_bridged_keys:
                _task_cfg = _auxiliary_cfg.get(_task_key, {})
                if not isinstance(_task_cfg, dict):
                    continue
                _prov = str(_task_cfg.get("provider", "")).strip()
                _model = str(_task_cfg.get("model", "")).strip()
                _base_url = str(_task_cfg.get("base_url", "")).strip()
                _api_key = str(_task_cfg.get("api_key", "")).strip()
                _upper = _task_key.upper()
                if _prov and _prov != "auto":
                    os.environ[f"AUXILIARY_{_upper}_PROVIDER"] = _prov
                if _model:
                    os.environ[f"AUXILIARY_{_upper}_MODEL"] = _model
                if _base_url:
                    os.environ[f"AUXILIARY_{_upper}_BASE_URL"] = _base_url
                if _api_key:
                    os.environ[f"AUXILIARY_{_upper}_API_KEY"] = _api_key
        # config.yaml is the documented, authoritative source for these
        # settings — it unconditionally wins over .env values. Previously
        # the guards below read `if X not in os.environ` and let stale
        # .env entries (e.g. HERMES_MAX_ITERATIONS=60 written by an old
        # `hermes setup` run) silently shadow the user's current config.
        # See PR #18413 / the 60-vs-500 max_turns incident.
        _agent_cfg = _cfg.get("agent", {})
        if _agent_cfg and isinstance(_agent_cfg, dict):
            if "max_turns" in _agent_cfg:
                os.environ["HERMES_MAX_ITERATIONS"] = str(_agent_cfg["max_turns"])
            if "gateway_timeout" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT"] = str(_agent_cfg["gateway_timeout"])
            if "gateway_timeout_warning" in _agent_cfg:
                os.environ["HERMES_AGENT_TIMEOUT_WARNING"] = str(_agent_cfg["gateway_timeout_warning"])
            if "gateway_notify_interval" in _agent_cfg:
                os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] = str(_agent_cfg["gateway_notify_interval"])
            if "restart_drain_timeout" in _agent_cfg:
                os.environ["HERMES_RESTART_DRAIN_TIMEOUT"] = str(_agent_cfg["restart_drain_timeout"])
            if "gateway_auto_continue_freshness" in _agent_cfg:
                os.environ["HERMES_AUTO_CONTINUE_FRESHNESS"] = str(
                    _agent_cfg["gateway_auto_continue_freshness"]
                )
        _display_cfg = _cfg.get("display", {})
        if _display_cfg and isinstance(_display_cfg, dict):
            if "busy_input_mode" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = str(_display_cfg["busy_input_mode"])
            if "busy_text_mode" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_TEXT_MODE"] = str(_display_cfg["busy_text_mode"])
            if "busy_ack_enabled" in _display_cfg:
                os.environ["HERMES_GATEWAY_BUSY_ACK_ENABLED"] = str(_display_cfg["busy_ack_enabled"])
        # Timezone: bridge config.yaml → HERMES_TIMEZONE env var.
        _tz_cfg = _cfg.get("timezone", "")
        if _tz_cfg and isinstance(_tz_cfg, str):
            os.environ["HERMES_TIMEZONE"] = _tz_cfg.strip()
        # Security settings
        _security_cfg = _cfg.get("security", {})
        if isinstance(_security_cfg, dict):
            _redact = _security_cfg.get("redact_secrets")
            if _redact is not None:
                os.environ["HERMES_REDACT_SECRETS"] = str(_redact).lower()
        # Gateway settings (media delivery allowlist + recency trust + strict mode)
        _gateway_cfg = _cfg.get("gateway", {})
        if isinstance(_gateway_cfg, dict):
            _strict = _gateway_cfg.get("strict")
            if _strict is not None:
                os.environ["HERMES_MEDIA_DELIVERY_STRICT"] = (
                    "1" if _strict else "0"
                )
            _allow_dirs = _gateway_cfg.get("media_delivery_allow_dirs")
            if _allow_dirs:
                if isinstance(_allow_dirs, str):
                    _allow_dirs_str = _allow_dirs
                elif isinstance(_allow_dirs, (list, tuple)):
                    _allow_dirs_str = os.pathsep.join(str(p) for p in _allow_dirs if p)
                else:
                    _allow_dirs_str = ""
                if _allow_dirs_str:
                    os.environ["HERMES_MEDIA_ALLOW_DIRS"] = _allow_dirs_str
            _trust_recent = _gateway_cfg.get("trust_recent_files")
            if _trust_recent is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_FILES"] = (
                    "1" if _trust_recent else "0"
                )
            _trust_recent_seconds = _gateway_cfg.get("trust_recent_files_seconds")
            if _trust_recent_seconds is not None:
                os.environ["HERMES_MEDIA_TRUST_RECENT_SECONDS"] = str(_trust_recent_seconds)
            # Bridge gateway.platform_connect_timeout → the internal env var the
            # connect path + Discord adapter ready-wait both read (#19776).
            # Unlike the agent.*/display.* bridges above (config-authoritative),
            # this env var is the manual-override escape hatch, so it WINS if
            # already set explicitly; otherwise config.yaml supplies the value.
            if (
                "platform_connect_timeout" in _gateway_cfg
                and not os.environ.get("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
            ):
                os.environ["HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT"] = str(
                    _gateway_cfg["platform_connect_timeout"]
                )
    except Exception as _bridge_err:
        # Previously this was silent (`except Exception: pass`), which
        # hid partial bridge failures and let .env defaults shadow
        # config.yaml values — users observed max_turns=500 in config
        # but a 60-iteration cap in practice. Surface the failure to
        # stderr so operators see it even though `logger` is not yet
        # initialized at module-import time (logger is defined further
        # down this module).
        print(
            f"  Warning: config.yaml → env bridge failed: "
            f"{type(_bridge_err).__name__}: {_bridge_err}",
            file=sys.stderr,
        )
        print(
            "  Gateway will fall back to .env values, which may not match "
            "your current config.yaml. Run `hermes doctor` to investigate.",
            file=sys.stderr,
        )

# Apply IPv4 preference if configured (before any HTTP clients are created).
try:
    from hermes_constants import apply_ipv4_preference
    _network_cfg = (_cfg if '_cfg' in dir() else {}).get("network", {})
    if isinstance(_network_cfg, dict) and _network_cfg.get("force_ipv4"):
        apply_ipv4_preference(force=True)
except Exception as _bootstrap_exc:
    print(f"  Warning: IPv4 preference application failed: {_bootstrap_exc}", file=sys.stderr)

# Validate config structure early — log warnings so gateway operators see problems
try:
    from hermes_cli.config import print_config_warnings
    print_config_warnings()
except Exception as _bootstrap_exc:
    print(f"  Warning: config validation failed: {_bootstrap_exc}", file=sys.stderr)

# Warn if user has deprecated MESSAGING_CWD / TERMINAL_CWD in .env
try:
    from hermes_cli.config import warn_deprecated_cwd_env_vars
    warn_deprecated_cwd_env_vars()
except Exception as _bootstrap_exc:
    print(f"  Warning: deprecation check failed: {_bootstrap_exc}", file=sys.stderr)

# Gateway runs in quiet mode - suppress debug output and use cwd directly (no temp dirs)
os.environ["HERMES_QUIET"] = "1"

# Enable interactive exec approval for dangerous commands on messaging platforms
os.environ["HERMES_EXEC_ASK"] = "1"

# Set terminal working directory for messaging platforms.
# config.yaml terminal.cwd is the canonical source (bridged to TERMINAL_CWD
# by the config bridge above).  When it's unset or a placeholder, default
# to home directory.  MESSAGING_CWD is accepted as a backward-compat
# fallback (deprecated — the warning above tells users to migrate).
_configured_cwd = os.environ.get("TERMINAL_CWD", "")
if not _configured_cwd or _configured_cwd in {".", "auto", "cwd"}:
    _fallback = os.getenv("MESSAGING_CWD") or str(Path.home())
    os.environ["TERMINAL_CWD"] = _fallback

from gateway.config import (
    Platform,
    _BUILTIN_PLATFORM_VALUES,
    GatewayConfig,
    HomeChannel,
    PlatformConfig,
    load_gateway_config,
)
from gateway.session import (
    SessionStore,
    SessionSource,
    SessionContext,
    build_session_context,
    build_session_context_prompt,
    build_session_key,
    is_shared_multi_user_session,
)
from gateway.delivery import DeliveryRouter, looks_like_telegram_private_chat_id
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.kanban_watchers import GatewayKanbanWatchersMixin
from gateway.slash_commands import GatewaySlashCommandsMixin
from gateway.platforms.base import (
    BasePlatformAdapter,
    EphemeralReply,
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
    merge_pending_message_event,
)
from gateway.restart import (
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
    parse_restart_drain_timeout,
)


from gateway.whatsapp_identity import (
    canonical_whatsapp_identifier as _canonical_whatsapp_identifier,  # noqa: F401
    expand_whatsapp_aliases as _expand_whatsapp_auth_aliases,
    normalize_whatsapp_identifier as _normalize_whatsapp_identifier,
)


logger = logging.getLogger(__name__)


_OWN_POLICY_OPEN_ENV = {
    Platform.WECOM: ("WECOM_DM_POLICY", "WECOM_GROUP_POLICY", "WECOM_ALLOW_ALL_USERS"),
    Platform.WEIXIN: ("WEIXIN_DM_POLICY", "WEIXIN_GROUP_POLICY", "WEIXIN_ALLOW_ALL_USERS"),
    Platform.YUANBAO: ("YUANBAO_DM_POLICY", "YUANBAO_GROUP_POLICY", "YUANBAO_ALLOW_ALL_USERS"),
    Platform.QQBOT: (None, None, "QQ_ALLOW_ALL_USERS"),
    Platform.WHATSAPP: ("WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY", "WHATSAPP_ALLOW_ALL_USERS"),
}


def _own_policy_open_startup_violation(config) -> Optional[str]:
    """Return a startup-abort reason when open policy lacks allow-all opt-in."""
    for platform, platform_config in getattr(config, "platforms", {}).items():
        if not getattr(platform_config, "enabled", False):
            continue
        open_env = _OWN_POLICY_OPEN_ENV.get(platform)
        if not open_env:
            continue
        dm_env, group_env, allow_all_env = open_env
        extra = getattr(platform_config, "extra", None) or {}
        dm_policy = str(
            extra.get("dm_policy")
            or (os.getenv(dm_env, "pairing") if dm_env else "pairing")
        ).strip().lower()
        group_policy = str(
            extra.get("group_policy")
            or (os.getenv(group_env, "pairing") if group_env else "pairing")
        ).strip().lower()
        if dm_policy != "open" and group_policy != "open":
            continue
        gateway_allow_all = os.getenv(
            "GATEWAY_ALLOW_ALL_USERS", ""
        ).lower() in {"true", "1", "yes"}
        platform_opted_in = gateway_allow_all or (
            allow_all_env
            and os.getenv(allow_all_env, "").lower() in {"true", "1", "yes"}
        )
        if platform_opted_in:
            continue
        return f"{platform.value}: open policy without allow-all opt-in"
    return None


# Sentinel placed into _running_agents immediately when a session starts
# processing, *before* any await.  Prevents a second message for the same
# session from bypassing the "already running" guard during the async gap
# between the guard check and actual agent creation.
_AGENT_PENDING_SENTINEL = object()


def _resolve_runtime_agent_kwargs() -> dict:
    """Resolve provider credentials for gateway-created AIAgent instances.

    Provider is read from ``config.yaml`` ``model.provider`` (the single
    source of truth). ``resolve_runtime_provider()`` falls through to env
    var lookups internally for legacy compatibility, but the gateway does
    not consult environment variables for behavioral config — config.yaml
    is authoritative.

    If the primary provider fails with an authentication error, attempt to
    resolve credentials using the fallback provider chain from config.yaml
    before giving up.
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
        _get_model_config,
    )
    from hermes_cli.auth import AuthError, is_rate_limited_auth_error

    try:
        runtime = resolve_runtime_provider()
    except AuthError as auth_exc:
        # Distinguish a transient rate-limit/quota cap (credentials are fine,
        # re-auth cannot help) from a genuine auth failure (expired/revoked
        # token). Both fall through to the fallback chain, but the log message
        # must not mislabel a quota exhaustion as an auth failure (#32790).
        if is_rate_limited_auth_error(auth_exc):
            logger.warning("Primary provider rate-limited (429): %s — trying fallback", auth_exc)
        else:
            logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    model_cfg = _get_model_config()
    max_tokens = None
    _env_mt = os.environ.get("HERMES_MAX_TOKENS")
    if _env_mt:
        try:
            max_tokens = int(_env_mt)
        except (ValueError, TypeError):
            max_tokens = None
    elif isinstance(model_cfg, dict):
        mt = model_cfg.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt
    # Fall back to a per-provider output cap (custom_providers max_output_tokens)
    # only when the documented global model.max_tokens isn't set, so the global
    # key always wins.
    if max_tokens is None:
        _runtime_mot = runtime.get("max_output_tokens")
        if isinstance(_runtime_mot, int) and _runtime_mot > 0:
            max_tokens = _runtime_mot

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "max_tokens": max_tokens,
    }


def _try_resolve_fallback_provider() -> dict | None:
    """Attempt to resolve credentials from the fallback_model/fallback_providers config."""
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        import yaml as _y
        cfg_path = _hermes_home / "config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path, encoding="utf-8") as _f:
            cfg = _y.safe_load(_f) or {}
        fb_list = get_fallback_chain(cfg)
        if not fb_list:
            return None
        for entry in fb_list:
            try:
                explicit_api_key = entry.get("api_key")
                if not explicit_api_key:
                    key_env = str(
                        entry.get("key_env") or entry.get("api_key_env") or ""
                    ).strip()
                    if key_env:
                        explicit_api_key = os.getenv(key_env, "").strip() or None
                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"),
                    explicit_base_url=entry.get("base_url"),
                    explicit_api_key=explicit_api_key,
                )
                # Log the literal `provider` key from config, not the resolved
                # runtime category — an Ollama fallback resolves through the
                # OpenAI-compatible path and would otherwise be logged as
                # "openrouter", contradicting the operator's config (#32790).
                logger.info(
                    "Fallback provider resolved: %s model=%s",
                    entry.get("provider") or runtime.get("provider"),
                    entry.get("model"),
                )
                return {
                    "api_key": runtime.get("api_key"),
                    "base_url": runtime.get("base_url"),
                    "provider": runtime.get("provider"),
                    "api_mode": runtime.get("api_mode"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                    "credential_pool": runtime.get("credential_pool"),
                    "model": entry.get("model"),
                }
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
    except Exception:
        pass
    return None


def _event_media_type_at(event, index: int) -> str:
    """Return the per-attachment MIME for the attachment at *index*.

    Empty string when the platform didn't populate a per-file MIME for
    that slot (some adapters only set a message-level type).
    """
    media_types = getattr(event, "media_types", None) or []
    return media_types[index] if index < len(media_types) else ""


def _event_media_is_image(event, index: int) -> bool:
    """True if the attachment at *index* is an image.

    Trust the per-attachment MIME when present. Only fall back to the
    message-level ``PHOTO`` type when this attachment's MIME is unknown --
    otherwise a document (or any non-image) uploaded alongside an image in
    the same message gets mis-routed as an image, base64'd into a vision
    content part, and the provider 400s ("Could not process image").
    """
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith("image/")
    return getattr(event, "message_type", None) == MessageType.PHOTO


def _event_media_is_audio(event, index: int) -> bool:
    """True if the attachment at *index* is audio (per-attachment MIME first)."""
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith("audio/")
    return getattr(event, "message_type", None) in {MessageType.VOICE, MessageType.AUDIO}


def _event_media_is_video(event, index: int) -> bool:
    """True if the attachment at *index* is video (per-attachment MIME first)."""
    mtype = _event_media_type_at(event, index)
    if mtype:
        return mtype.startswith("video/")
    return getattr(event, "message_type", None) == MessageType.VIDEO


def _build_media_placeholder(event) -> str:
    """Build a text placeholder for media-only events so they aren't dropped.

    When a photo/document is queued during active processing and later
    dequeued, only .text is extracted.  If the event has no caption,
    the media would be silently lost.  This builds a placeholder that
    the vision enrichment pipeline will replace with a real description.
    """
    parts = []
    media_urls = getattr(event, "media_urls", None) or []
    for i, url in enumerate(media_urls):
        if _event_media_is_image(event, i):
            parts.append(f"[User sent an image: {url}]")
        elif _event_media_is_audio(event, i):
            parts.append(f"[User sent audio: {url}]")
        elif _event_media_is_video(event, i):
            parts.append(f"[User sent a video: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)


def _build_document_context_note(display_name: str, agent_path: str, mtype: str) -> str:
    """Context note prepended to a user turn when they attach a document.

    Text documents (``text/*``) have their content inlined upstream by the
    platform adapter, so the note just confirms that and records the path.

    Binary documents (PDF, DOCX, XLSX, …) cannot be inlined as text. The note
    must tell the agent to *extract* the text itself before answering — earlier
    wording ("Ask the user what they'd like you to do with it") steered the
    model into punting back to the user, which is why attached PDFs/DOCX looked
    "unreadable" to the agent even though it has the tools to read them.
    """
    if mtype.startswith("text/"):
        return (
            f"[The user sent a text document: '{display_name}'. "
            f"Its content has been included below. "
            f"The file is also saved at: {agent_path}]"
        )
    return (
        f"[The user sent a document: '{display_name}'. It is saved at: {agent_path}. "
        f"Its text is not inlined here (it's a binary format such as PDF or DOCX). "
        f"To read it, extract the document's text yourself — for example with the "
        f"terminal tool or the ocr-and-documents skill — before answering, instead "
        f"of asking the user to paste the contents.]"
    )


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 0:
        total = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


async def _probe_audio_duration(path: str) -> Optional[str]:
    """Best-effort duration probe. Returns formatted MM:SS / HH:MM:SS, or None on failure."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            def _wav_duration() -> float:
                import wave
                with wave.open(path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate() or 1
                    return frames / float(rate)
            secs = await asyncio.to_thread(_wav_duration)
            return _format_duration(secs)
        except Exception:
            pass

    if ext in (".ogg", ".opus", ".oga"):
        try:
            def _ogg_duration() -> float:
                from mutagen.oggopus import OggOpus
                return float(OggOpus(path).info.length)
            secs = await asyncio.to_thread(_ogg_duration)
            return _format_duration(secs)
        except Exception:
            pass

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0:
            return _format_duration(float(stdout.decode().strip()))
    except Exception:
        pass

    return None


def _dequeue_pending_event(adapter, session_key: str) -> MessageEvent | None:
    """Consume and return the full pending event for a session.

    Queued follow-ups must preserve their media metadata so they can re-enter
    the normal image/STT/document preprocessing path instead of being reduced
    to a placeholder string.
    """
    return adapter.get_pending_message(session_key)


_INTERRUPT_REASON_STOP = "Stop requested"
_INTERRUPT_REASON_RESET = "Session reset requested"
_INTERRUPT_REASON_TIMEOUT = "Execution timed out (inactivity)"
_INTERRUPT_REASON_SSE_DISCONNECT = "SSE client disconnected"
_INTERRUPT_REASON_GATEWAY_SHUTDOWN = "Gateway shutting down"
_INTERRUPT_REASON_GATEWAY_RESTART = "Gateway restarting"

_CONTROL_INTERRUPT_MESSAGES = frozenset(
    {
        _INTERRUPT_REASON_STOP.lower(),
        _INTERRUPT_REASON_RESET.lower(),
        _INTERRUPT_REASON_TIMEOUT.lower(),
        _INTERRUPT_REASON_SSE_DISCONNECT.lower(),
        _INTERRUPT_REASON_GATEWAY_SHUTDOWN.lower(),
        _INTERRUPT_REASON_GATEWAY_RESTART.lower(),
    }
)


def _is_control_interrupt_message(message: Optional[str]) -> bool:
    """Return True when an interrupt message is internal control flow."""
    if not message:
        return False
    normalized = " ".join(str(message).strip().split()).lower()
    return normalized in _CONTROL_INTERRUPT_MESSAGES


def _skill_slug_from_frontmatter(skill_md: Path) -> tuple[str | None, str | None]:
    """Derive the /command slug and declared frontmatter name from a SKILL.md.

    Matches the exact normalization used by
    :func:`agent.skill_commands.scan_skill_commands` so the slug here is the
    same string a user types after the leading ``/`` (e.g. a skill with
    frontmatter ``name: Stable Diffusion Image Generation`` resolves to
    ``stable-diffusion-image-generation`` — NOT the parent directory name,
    which is commonly shorter/different, e.g. ``stable-diffusion``).

    Using the directory name silently broke :func:`_check_unavailable_skill`
    for every skill whose directory name drifted from its frontmatter name
    (19 such skills on a standard install as of 2026-05), causing a generic
    "unknown command" response where a "disabled — enable with …" or
    "not installed — install with …" hint was expected.

    Returns ``(slug, declared_name)`` or ``(None, None)`` when the file
    can't be read or lacks a ``name:`` in its frontmatter.
    """
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None, None
    if not content.startswith("---"):
        return None, None
    end = content.find("\n---", 3)
    if end < 0:
        return None, None
    declared_name: str | None = None
    for line in content[3:end].splitlines():
        line = line.strip()
        if line.startswith("name:"):
            raw = line.split(":", 1)[1].strip()
            # Strip YAML quote wrappers if present
            if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
                raw = raw[1:-1]
            declared_name = raw.strip()
            break
    if not declared_name:
        return None, None
    slug = declared_name.lower().replace(" ", "-").replace("_", "-")
    # Mirror _SKILL_INVALID_CHARS and _SKILL_MULTI_HYPHEN from skill_commands
    import re as _re
    slug = _re.sub(r"[^a-z0-9-]", "", slug)
    slug = _re.sub(r"-{2,}", "-", slug).strip("-")
    if not slug:
        return None, declared_name
    return slug, declared_name


def _check_unavailable_skill(command_name: str) -> str | None:
    """Check if a command matches a known-but-inactive skill.

    Returns a helpful message if the skill exists but is disabled or only
    available as an optional install. Returns None if no match found.

    The slug for each on-disk skill is derived from its frontmatter ``name:``
    (via :func:`_skill_slug_from_frontmatter`), NOT from its containing
    directory name — because the two can differ (e.g. directory
    ``stable-diffusion`` + frontmatter ``Stable Diffusion Image Generation``
    yields slug ``stable-diffusion-image-generation``). Matching on
    directory name would miss that slug entirely and fall through to the
    generic "unknown command" path.
    """
    # Normalize: command uses hyphens, skill names may use hyphens or underscores
    normalized = command_name.lower().replace("_", "-")
    try:
        from tools.skills_tool import _get_disabled_skill_names
        from agent.skill_utils import get_all_skills_dirs, is_excluded_skill_path
        disabled = _get_disabled_skill_names()

        # Check disabled skills across all dirs (local + external)
        for skills_dir in get_all_skills_dirs():
            if not skills_dir.exists():
                continue
            for skill_md in skills_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, declared_name = _skill_slug_from_frontmatter(skill_md)
                if not slug or not declared_name:
                    continue
                # disabled is keyed by the declared frontmatter name (what
                # skills.disabled / skills.platform_disabled store).
                if slug == normalized and declared_name in disabled:
                    return (
                        f"The **{command_name}** skill is installed but disabled.\n"
                        f"Enable it with: `hermes skills config`"
                    )

        # Check optional skills (shipped with repo but not installed)
        from hermes_constants import get_optional_skills_dir
        repo_root = Path(__file__).resolve().parent.parent
        optional_dir = get_optional_skills_dir(repo_root / "optional-skills")
        if optional_dir.exists():
            for skill_md in optional_dir.rglob("SKILL.md"):
                if is_excluded_skill_path(skill_md):
                    continue
                slug, _declared = _skill_slug_from_frontmatter(skill_md)
                if not slug:
                    continue
                if slug == normalized:
                    # Build install path: official/<category>/<name>
                    rel = skill_md.parent.relative_to(optional_dir)
                    parts = list(rel.parts)
                    install_path = f"official/{'/'.join(parts)}"
                    return (
                        f"The **{command_name}** skill is available but not installed.\n"
                        f"Install it with: `hermes skills install {install_path}`"
                    )
    except Exception:
        pass
    return None


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value


def _teams_pipeline_plugin_enabled() -> bool:
    """Return True when the standalone Teams pipeline plugin is enabled."""
    config = _load_gateway_config()
    enabled = cfg_get(config, "plugins", "enabled", default=[])
    if not isinstance(enabled, list):
        return False
    return "teams_pipeline" in enabled or "teams-pipeline" in enabled


def _load_gateway_config() -> dict:
    """Load and parse ~/.hermes/config.yaml, returning {} on any error.

    Uses the module-level ``_hermes_home`` (so tests that monkeypatch it
    still see their fixture) and shares the mtime-keyed raw-yaml cache
    from ``hermes_cli.config.read_raw_config`` when the paths match.

    Managed scope is overlaid on the result (via the shared helper) so the
    gateway honors administrator-pinned values — neither read_raw_config nor a
    direct yaml.safe_load carries the managed merge on its own. Fail-open.
    """
    config_path = _hermes_home / 'config.yaml'
    raw: dict = {}
    used_canonical = False
    try:
        from hermes_cli.config import get_config_path, read_raw_config
        # Fast path: if _hermes_home agrees with the canonical config
        # location, reuse the shared cache. Otherwise fall through to a
        # direct read (keeps test fixtures with a monkeypatched
        # _hermes_home working).
        if config_path == get_config_path():
            raw = read_raw_config()
            used_canonical = True
    except Exception:
        pass

    if not used_canonical:
        try:
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    raw = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("Could not load gateway config from %s", config_path)
            raw = {}

    # Overlay managed scope. read_raw_config() returns the user's raw YAML
    # WITHOUT the managed merge (that lives in load_config/_load_config_impl),
    # so the overlay is required on both paths for the gateway to honor pinned
    # values. Helper is fail-open and a no-op when no managed scope exists.
    try:
        from hermes_cli import managed_scope
        raw = managed_scope.apply_managed_overlay(raw if isinstance(raw, dict) else {})
    except Exception:
        pass
    if not isinstance(raw, dict):
        return {}
    # Canonicalize model-id aliases (model.name / model.model → model.default)
    # and migrate stale root-level provider/base_url into the model section.
    # The gateway bypasses load_config() (it reads raw YAML for speed), so the
    # normalization that load_config() applies must be replayed here or the
    # gateway would resolve an empty model for ``model: {name: <id>}`` configs
    # while the CLI resolves it correctly. See issue #34500. Fail-open.
    try:
        from hermes_cli.config import _normalize_root_model_keys
        raw = _normalize_root_model_keys(raw)
    except Exception:
        pass
    return raw


def _load_gateway_runtime_config() -> dict:
    """Load gateway config for runtime reads, expanding supported ``${VAR}`` refs.

    Runtime helpers should honor the same env-template expansion documented for
    ``config.yaml`` while still respecting tests that monkeypatch
    ``gateway.run._hermes_home``. Build on ``_load_gateway_config()`` rather
    than calling the canonical loader directly so both behaviors stay aligned.

    Expansion failures are intentionally NOT swallowed — silently returning
    the unexpanded dict would mask the very bug this helper exists to fix.
    """
    cfg = _load_gateway_config()
    if not isinstance(cfg, dict) or not cfg:
        return {}
    from hermes_cli.config import _expand_env_vars

    expanded = _expand_env_vars(cfg)
    return expanded if isinstance(expanded, dict) else {}


def _resolve_gateway_model(config: dict | None = None) -> str:
    """Read model from config.yaml — single source of truth.

    Without this, temporary AIAgent instances (e.g. /compress) fall
    back to the hardcoded default which fails when the active provider is
    openai-codex.
    """
    cfg = config if config is not None else _load_gateway_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _resolve_hermes_bin() -> Optional[list[str]]:
    """Resolve the Hermes update command as argv parts.

    Tries in order:
    1. ``shutil.which("hermes")`` — standard PATH lookup
    2. ``sys.executable -m hermes_cli.main`` — fallback when Hermes is running
       from a venv/module invocation and the ``hermes`` shim is not on PATH

    Returns argv parts ready for quoting/joining, or ``None`` if neither works.
    """
    import shutil

    hermes_bin = shutil.which("hermes")
    if hermes_bin:
        return [hermes_bin]

    try:
        import importlib.util

        if importlib.util.find_spec("hermes_cli") is not None:
            return [sys.executable, "-m", "hermes_cli.main"]
    except Exception:
        pass

    return None


def _parse_session_key(session_key: str) -> "dict | None":
    """Parse a session key into its component parts.

    Session keys follow the format
    ``agent:main:{platform}:{chat_type}:{chat_id}[:{extra}...]``.
    Returns a dict with ``platform``, ``chat_type``, ``chat_id``, and
    optionally ``thread_id`` keys, or None if the key doesn't match.

    The 6th element is only returned as ``thread_id`` for chat types where
    it is unambiguous (``dm`` and ``thread``).  For group/channel sessions
    the suffix may be a user_id (per-user isolation) rather than a
    thread_id, so we leave ``thread_id`` out to avoid mis-routing.
    """
    parts = session_key.split(":")
    if len(parts) >= 5 and parts[0] == "agent" and parts[1] == "main":
        result = {
            "platform": parts[2],
            "chat_type": parts[3],
            "chat_id": parts[4],
        }
        if len(parts) > 5 and parts[3] in {"dm", "thread"}:
            result["thread_id"] = parts[5]
        return result
    return None


def _format_gateway_process_notification(evt: dict) -> "str | None":
    """Format a watch pattern event from completion_queue into a [IMPORTANT:] message."""
    evt_type = evt.get("type", "completion")
    _sid = evt.get("session_id", "unknown")
    _cmd = evt.get("command", "unknown")

    if evt_type == "watch_disabled":
        return f"[IMPORTANT: {evt.get('message', '')}]"

    if evt_type == "watch_match":
        _pat = evt.get("pattern", "?")
        _out = evt.get("output", "")
        _sup = evt.get("suppressed", 0)
        text = (
            f"[IMPORTANT: Background process {_sid} matched "
            f"watch pattern \"{_pat}\".\n"
            f"Command: {_cmd}\n"
            f"Matched output:\n{_out}"
        )
        if _sup:
            text += f"\n({_sup} earlier matches were suppressed by rate limit)"
        text += "]"
        return text

    if evt_type == "async_delegation":
        # Reuse the shared rich formatter (self-contained task-source block).
        from tools.process_registry import format_process_notification
        return format_process_notification(evt)

    return None


def _drain_gateway_watch_events(completion_queue) -> "list[dict]":
    """Drain gateway-owned watch events without spinning on requeued events.

    Watch events are handled by the post-turn gateway drain. Process
    completions are owned by their per-process watcher task, and async
    delegation completions are owned by ``_async_delegation_watcher``.
    Requeueing async events inside ``while not queue.empty()`` would make the
    loop non-terminating, so detach the current batch first, then requeue any
    events this drain does not own after the queue is empty.
    """
    watch_events: list[dict] = []
    requeue: list[dict] = []
    while not completion_queue.empty():
        try:
            evt = completion_queue.get_nowait()
        except Exception:
            break
        evt_type = evt.get("type", "completion")
        if evt_type in {"watch_match", "watch_disabled"}:
            watch_events.append(evt)
        elif evt_type == "async_delegation":
            requeue.append(evt)
        # else: process completion events are handled by the watcher task
    for evt in requeue:
        completion_queue.put(evt)
    return watch_events


# Module-level weak reference to the active GatewayRunner instance.
# Used by tools (e.g. send_message) that need to route through a live
# adapter for plugin platforms.  Set in GatewayRunner.__init__().
import weakref as _weakref
_gateway_runner_ref: _weakref.ref = lambda: None


def _normalize_empty_agent_response(
    agent_result: dict,
    response: str,
    *,
    history_len: int = 0,
) -> str:
    """Normalize empty/None agent responses into user-facing messages.

    Consolidates the existing ``failed`` handler and adds a catch-all for
    the case where the agent did work (api_calls > 0) but returned no text.
    Fix for #18765.

    Also surfaces a retry hint when the agent never ran at all
    (api_calls == 0) for a non-interrupted, non-failed turn -- this is the
    silent-drop pattern observed after ``/stop`` where the next user
    message hits a stale generation token and returns an empty result,
    leaving the platform with nothing to send. (#31884)
    """
    if response:
        return response

    if agent_result.get("failed"):
        error_detail = agent_result.get("error", "unknown error")
        error_str = str(error_detail).lower()
        is_context_failure = any(
            p in error_str
            for p in ("context", "token", "too large", "too long", "exceed", "payload")
        ) or ("400" in error_str and history_len > 50)
        if is_context_failure:
            return (
                "⚠️ Session too large for the model's context window.\n"
                "Use /compact to compress the conversation, or "
                "/reset to start fresh."
            )
        return (
            f"The request failed: {str(error_detail)[:300]}\n"
            "Try again or use /reset to start a fresh session."
        )

    api_calls = int(agent_result.get("api_calls", 0) or 0)
    if api_calls > 0 and not agent_result.get("interrupted"):
        if agent_result.get("partial"):
            err = agent_result.get("error", "processing incomplete")
            return f"⚠️ Processing stopped: {str(err)[:200]}. Try again."
        return (
            "⚠️ Processing completed but no response was generated. "
            "This may be a transient error — try sending your message again."
        )

    # api_calls == 0, not failed, not interrupted: the agent never ran for
    # this turn. This is the post-/stop generation-race pattern where the
    # gateway would otherwise silently drop the turn (response=0 chars) and
    # the user sees no reply at all. Surface a short retry hint so the
    # message isn't lost in silence. (#31884)
    if (
        api_calls == 0
        and not agent_result.get("interrupted")
        and not agent_result.get("failed")
        and not agent_result.get("partial")
    ):
        return (
            "⚠️ Your message wasn't processed (the previous turn was still "
            "being cleaned up). Please send it again."
        )

    return response


def _should_clear_resume_pending_after_turn(agent_result: dict) -> bool:
    """Return True only when a gateway turn really completed successfully.

    Restart recovery uses ``resume_pending`` as a durable marker for sessions
    interrupted during gateway drain.  A soft interrupt can still bubble out as
    a syntactically normal agent result with an empty final response; clearing
    the marker in that case loses the recovery signal and startup auto-resume
    has nothing to schedule.
    """
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("interrupted"):
        return False
    if agent_result.get("failed") or agent_result.get("partial") or agent_result.get("error"):
        return False
    if agent_result.get("completed") is False:
        return False
    return True


def _preserve_queued_followup_history_offset(
    current_result: dict,
    followup_result: dict,
) -> dict:
    """Carry the outer history offset through queued follow-up drains.

    ``_process_message_background()`` persists transcript rows only once, after the
    entire in-band queued-follow-up chain returns.  Each recursive ``_run_agent()``
    call advances ``history_offset`` to the history it received, so without
    correction the outermost persistence step sees only the *last* queued turn as
    "new" and silently drops earlier turns from the same drain chain.

    Preserve the earliest (outermost) history offset so the final transcript slice
    still includes every queued turn that ran during the chain.
    """
    if not isinstance(followup_result, dict):
        return followup_result
    if not isinstance(current_result, dict):
        return followup_result

    current_offset = current_result.get("history_offset")
    followup_offset = followup_result.get("history_offset")
    if not isinstance(current_offset, int):
        return followup_result
    if isinstance(followup_offset, int) and followup_offset <= current_offset:
        return followup_result

    merged = dict(followup_result)
    merged["history_offset"] = current_offset
    return merged


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters``.

    The reconnect watcher in ``GatewayRunner._platform_reconnect_watcher``
    constructs a fresh adapter on every retry attempt. When the connect
    call fails — for any of the three reasons (non-retryable error,
    retryable error, exception during connect) — the adapter is dropped
    without ever being installed, so nothing else will call its
    ``disconnect()``. Any resources the adapter opened in ``__init__``
    (e.g. ``APIServerAdapter`` opens a SQLite ``ResponseStore`` that
    holds 2 fds — the db file and its WAL sidecar) stay open until
    garbage collection sweeps the unreachable object, which Python's
    cyclic GC does not do promptly for asyncio-bound objects with
    native handles. The cumulative leak is 2 fds × every retry at the
    300s backoff cap ≈ 12 fds/hour, and the default 2560-fd ulimit
    is exhausted in ~12h of continuous failure, after which every
    open() call on the gateway raises ``OSError: [Errno 24] Too many
    open files`` and the gateway becomes a zombie (#37011).

    This helper centralises the dispose-with-suppression so the three
    failure paths in the reconnect watcher can all call it without
    each one having to know that ``disconnect()`` may itself raise
    on a half-constructed adapter.

    ``adapter`` may be ``None``: the reconnect watcher initialises
    ``adapter = None`` before the ``try`` so the ``except Exception``
    arm can dispose a half-constructed object, and also early-returns
    here when ``_create_adapter()`` returned ``None``.
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters (e.g. APIServerAdapter that
        # crashed during aiohttp app setup) can raise from
        # disconnect() on objects that never finished initializing.
        # We must not let that escape and abort the watcher loop.
        #
        # On Python 3.8+, ``asyncio.CancelledError`` inherits from
        # ``BaseException`` (not ``Exception``), so this ``except
        # Exception`` does not swallow task cancellation. We don't
        # re-raise explicitly because the watcher loop intentionally
        # treats dispose failures as best-effort: a failed ``disconnect``
        # call should not take down the reconnect watcher that
        # itself is what's keeping the gateway alive during a partial
        # outage.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__),
            exc_info=True,
        )


class GatewayRunner(GatewayAuthorizationMixin, GatewayKanbanWatchersMixin, GatewaySlashCommandsMixin):
    """
    Main gateway controller.

    Manages the lifecycle of all platform adapters and routes
    messages to/from the agent.
    """

    # Class-level defaults so partial construction in tests doesn't
    # blow up on attribute access.
    _running_agents_ts: Dict[str, float] = {}
    _busy_input_mode: str = "interrupt"
    _busy_text_mode: str = "interrupt"
    _restart_drain_timeout: float = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    _exit_code: Optional[int] = None
    _draining: bool = False
    _external_drain_active: bool = False
    _restart_requested: bool = False
    _restart_task_started: bool = False
    _restart_detached: bool = False
    _restart_via_service: bool = False
    _detached_restart_helper_started: bool = False
    _restart_command_source: Optional[SessionSource] = None
    _stop_task: Optional[asyncio.Task] = None
    _restart_task: Optional[asyncio.Task] = None
    _session_model_overrides: Dict[str, Dict[str, str]] = {}
    _session_reasoning_overrides: Dict[str, Dict[str, Any]] = {}
    _startup_restore_in_progress: bool = False

    def __init__(self, config: Optional[GatewayConfig] = None):
        global _gateway_runner_ref
        self.config = config or load_gateway_config()
        # Mark the process as a profile multiplexer when configured. This flips
        # agent.secret_scope.get_secret() to fail-closed on any unscoped
        # credential read, so a missed migration crashes loudly instead of
        # leaking a cross-profile value (Workstream A). Inert when off.
        try:
            from agent.secret_scope import set_multiplex_active
            set_multiplex_active(bool(getattr(self.config, "multiplex_profiles", False)))
        except Exception:
            logger.debug("could not set multiplex-active flag", exc_info=True)
        self.adapters: Dict[Platform, BasePlatformAdapter] = {}
        # Multi-profile multiplexing: adapters for NON-default profiles live
        # here, keyed by profile name then Platform. self.adapters stays the
        # default/active profile's map so the ~93 existing self.adapters[...]
        # sites are untouched when multiplexing is off (this dict is empty).
        # Populated by _start_secondary_profile_adapters().
        self._profile_adapters: Dict[str, Dict[Platform, BasePlatformAdapter]] = {}
        self._warn_if_docker_media_delivery_is_risky()
        _gateway_runner_ref = _weakref.ref(self)

        # Load ephemeral config from config.yaml / env vars.
        # Both are injected at API-call time only and never persisted.
        self._prefill_messages = self._load_prefill_messages()
        self._ephemeral_system_prompt = self._load_ephemeral_system_prompt()
        self._reasoning_config = self._load_reasoning_config()
        self._service_tier = self._load_service_tier()
        self._show_reasoning = self._load_show_reasoning()
        self._busy_input_mode = self._load_busy_input_mode()
        self._busy_text_mode = self._load_busy_text_mode()
        self._restart_drain_timeout = self._load_restart_drain_timeout()
        self._provider_routing = self._load_provider_routing()
        self._fallback_model = self._load_fallback_model()

        # Wire process registry into session store for reset protection.
        # A background process older than the configured threshold (default 24h,
        # session_reset.bg_process_max_age_hours) is treated as stale and no
        # longer blocks session idle / daily reset — see #29177. The process is
        # NOT killed, only ignored by the reset guard.
        from tools.process_registry import process_registry
        _bg_max_age_hours = getattr(
            self.config.default_reset_policy, "bg_process_max_age_hours", 24
        )
        _bg_max_age_seconds = (
            _bg_max_age_hours * 3600 if _bg_max_age_hours and _bg_max_age_hours > 0 else None
        )
        self.session_store = SessionStore(
            self.config.sessions_dir, self.config,
            has_active_processes_fn=lambda key: process_registry.has_active_for_session(
                key, max_active_age=_bg_max_age_seconds,
            ),
        )
        self.delivery_router = DeliveryRouter(self.config)
        self._running = False
        self._gateway_loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event = asyncio.Event()
        self._exit_cleanly = False
        self._exit_with_failure = False
        self._exit_reason: Optional[str] = None
        self._exit_code: Optional[int] = None
        self._draining = False
        # External (NAS-driven) drain state — distinct from the shutdown
        # ``_draining`` flag above. Set by ``_drain_control_watcher`` when the
        # ``.drain_request.json`` marker is present: the gateway flips
        # ``gateway_state -> draining`` and refuses NEW turns, but the process
        # does NOT exit (the whole point — quiesce-without-restart, D4a). It is
        # fully reversible: removing the marker reverts to ``running`` and
        # re-accepts turns. ``_draining`` (shutdown) is one-way and ends in
        # process exit; this one is a steady state NAS polls during its
        # request -> poll -> proceed loop.
        self._external_drain_active = False
        self._restart_requested = False
        # Set by shutdown_signal_handler when a SIGTERM/SIGINT arrived
        # WITHOUT a planned-stop / takeover marker — i.e. an unexpected
        # external signal (container/s6 SIGTERM on `docker restart` or
        # image upgrade, OOM-killer, bare `kill`). Distinct from an
        # operator-requested stop, which writes a marker first. Used by
        # _stop_impl to decide whether to persist gateway_state=stopped
        # (see issue #42675): an unexpected signal must NOT persist
        # "stopped", or container_boot refuses to auto-start the gateway
        # on the next boot.
        self._signal_initiated_shutdown = False
        self._restart_task_started = False
        self._restart_detached = False
        self._restart_via_service = False
        self._detached_restart_helper_started = False
        self._restart_command_source: Optional[SessionSource] = None
        # Monotonic-ish wall clock of when this GatewayRunner was constructed.
        # Used by the /restart redelivery guard to bound the window in which a
        # missing dedup marker is treated as a stale redelivery.
        self._startup_time: float = time.time()
        # Set True at startup when this process booted as the result of a
        # chat-originated /restart (i.e. .restart_notify.json existed on boot).
        # A one-shot signal consumed by _is_stale_restart_redelivery so the
        # marker-missing fallback only suppresses a /restart when we KNOW we
        # just came out of a restart cycle — never on a genuine fresh boot.
        self._booted_from_restart: bool = False
        self._stop_task: Optional[asyncio.Task] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._executor_lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        # Set on gateway stop so the recreate-on-shutdown path can't resurrect
        # the pool during a real shutdown.
        self._executor_closing = False
        # Track running agents per session for interrupt support
        # Key: session_key, Value: AIAgent instance
        self._running_agents: Dict[str, Any] = {}
        self._running_agents_ts: Dict[str, float] = {}  # start timestamp per session
        self._active_session_leases: Dict[str, Any] = {}
        self._pending_messages: Dict[str, str] = {}  # Queued messages during interrupt
        # Last successfully-resolved (non-empty) model, keyed by session. Used
        # as a fallback when a fresh config read transiently returns an empty
        # model (e.g. an mtime-keyed config-cache miss during a post-interrupt
        # recovery turn). Without this, the agent is built with model="" and
        # every API call fails HTTP 400 "No models provided" — the session goes
        # silent until the user manually re-sends. See #35314. ``"*"`` holds a
        # process-wide last-known-good for sessions seen for the first time.
        self._last_resolved_model: Dict[str, str] = {}
        # Overflow buffer for explicit /queue commands.  The adapter-level
        # _pending_messages dict is a single slot per session (designed for
        # "next-turn" follow-ups where repeated sends collapse into one
        # event).  /queue has different semantics: each invocation must
        # produce its own full agent turn, in FIFO order, with no merging.
        # When the slot is occupied, additional /queue items land here and
        # are promoted one-at-a-time after each run's drain.  Cleared on
        # /new and /reset.  /model and other mid-session operations
        # preserve the queue.
        self._queued_events: Dict[str, List[MessageEvent]] = {}
        self._pending_native_image_paths_by_session: Dict[str, List[str]] = {}
        self._busy_ack_ts: Dict[str, float] = {}  # last busy-ack timestamp per session (debounce)
        self._session_run_generation: Dict[str, int] = {}
        # Startup restore gate: while restart-interrupted sessions are being
        # auto-resumed, real inbound messages are queued instead of competing
        # with the synthetic resume turns for the same session.  The queued
        # events drain only after all startup resume tasks have finished.
        self._startup_restore_in_progress = False
        self._startup_restore_queue: List[MessageEvent] = []
        self._startup_restore_tasks: List[asyncio.Task] = []
        # LRU cache of live SessionSources keyed by session_key. Used by
        # fallback routing paths (shutdown notifications, synthetic
        # background-process events) when the persisted origin is missing
        # and _parse_session_key can't recover thread_id. Capped so it
        # cannot grow unbounded over a long-running gateway lifetime.
        self._session_sources: "OrderedDict[str, SessionSource]" = OrderedDict()
        self._session_sources_max = 512

        # Cache AIAgent instances per session to preserve prompt caching.
        # Without this, a new AIAgent is created per message, rebuilding the
        # system prompt (including memory) every turn — breaking prefix cache
        # and costing ~10x more on providers with prompt caching (Anthropic).
        # Key: session_key, Value: (AIAgent, config_signature_str)
        #
        # OrderedDict so _enforce_agent_cache_cap() can pop the least-recently-
        # used entry (move_to_end() on cache hits, popitem(last=False) for
        # eviction).  Hard cap via _AGENT_CACHE_MAX_SIZE, idle TTL enforced
        # from _session_expiry_watcher().
        import threading as _threading
        self._agent_cache: "OrderedDict[str, tuple]" = OrderedDict()
        self._agent_cache_lock = _threading.Lock()

        # Per-session model overrides from /model command.
        # Key: session_key, Value: dict with model/provider/api_key/base_url/api_mode
        self._session_model_overrides: Dict[str, Dict[str, str]] = {}
        # Per-session reasoning effort overrides from /reasoning.
        # Key: session_key, Value: parsed reasoning config dict.
        self._session_reasoning_overrides: Dict[str, Dict[str, Any]] = {}
        self._kanban_notifier_profile = self._active_profile_name()
        # Teams meeting pipeline runtime (bound later when msgraph_webhook adapter exists).
        self._teams_pipeline_runtime = None
        self._teams_pipeline_runtime_error: Optional[str] = None
        # Track pending exec approvals per session
        # Key: session_key, Value: {"command": str, "pattern_key": str, ...}
        self._pending_approvals: Dict[str, Dict[str, Any]] = {}

        # Track platforms that failed to connect for background reconnection.
        # Key: Platform enum, Value: {"config": platform_config, "attempts": int, "next_retry": float}
        self._failed_platforms: Dict[Platform, Dict[str, Any]] = {}

        # Track pending /update prompt responses per session.
        # Key: session_key, Value: True when a prompt is waiting for user input.
        self._update_prompt_pending: Dict[str, bool] = {}

        # Slash-confirm state lives in tools.slash_confirm (module-level),
        # so platform adapters can resolve callbacks without a backref to
        # this runner.  Keep a local counter for confirm_id generation so
        # IDs stay compact (button callback_data has a 64-byte cap on
        # some platforms).
        import itertools as _itertools
        self._slash_confirm_counter = _itertools.count(1)

        # Persistent Honcho managers keyed by gateway session key.
        # This preserves write_frequency="session" semantics across short-lived
        # per-message AIAgent instances.



        # Ensure tirith security scanner is available (downloads if needed)
        try:
            from tools.tirith_security import ensure_installed
            ensure_installed(log_failures=False)
        except Exception:
            pass  # Non-fatal — fail-open at scan time if unavailable

        # Startup heads-up (#30882): a gateway in manual approval mode with no
        # automated risk assessor (tirith disabled AND no auxiliary.approval
        # model) can only gate dangerous commands / execute_code scripts via
        # live in-chat approval. With approval routing fixed, those actions now
        # fail closed (block) rather than silently auto-running — surface that
        # so operators knowingly enable tirith or configure auxiliary.approval
        # for unattended gateways.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _appr_cfg = _load_full_config()
            _appr_mode = str(
                cfg_get(_appr_cfg, "approvals", "mode", default="manual") or "manual"
            ).strip().lower()
            _tirith_on = bool(cfg_get(_appr_cfg, "security", "tirith_enabled", default=True))
            _aux_approval = cfg_get(_appr_cfg, "auxiliary", "approval", default=None)
            if _appr_mode == "manual" and not _tirith_on and not _aux_approval:
                logger.warning(
                    "Gateway approvals.mode=manual with no automated risk "
                    "assessor (security.tirith_enabled is false and "
                    "auxiliary.approval is unset): dangerous commands and "
                    "execute_code scripts will BLOCK until a human approves "
                    "them in chat. Enable security.tirith_enabled or configure "
                    "auxiliary.approval for unattended operation."
                )
        except Exception:
            logger.debug("approvals.mode startup check skipped", exc_info=True)

        # Initialize session database for session_search tool support
        self._session_db = None
        try:
            from hermes_state import AsyncSessionDB, SessionDB
            self._session_db = AsyncSessionDB(SessionDB())
        except Exception as e:
            # WARNING (not DEBUG) so the failure appears in errors.log — matches
            # cli.py's handling of the same init path.  Users hitting NFS-mounted
            # HERMES_HOME silently lost /resume, /title, /history, /branch, and
            # session search without this.  The underlying cause (usually
            # "locking protocol" from NFS) is now also captured by
            # hermes_state.get_last_init_error() for slash-command error strings.
            logger.warning("SQLite session store not available: %s", e)

        # Opportunistic state.db maintenance: prune ended sessions older
        # than sessions.retention_days + optional VACUUM. Tracks last-run
        # in state_meta so it only actually executes once per
        # sessions.min_interval_hours.  Gateway is long-lived so blocking
        # a few seconds once per day is acceptable; failures are logged
        # but never raised.
        if self._session_db is not None:
            try:
                from hermes_cli.config import load_config as _load_full_config
                _sess_cfg = (_load_full_config().get("sessions") or {})
                if _sess_cfg.get("auto_prune", False):
                    # Construction-time, before the loop serves traffic; sync DB is fine.
                    self._session_db._db.maybe_auto_prune_and_vacuum(
                        retention_days=int(_sess_cfg.get("retention_days", 90)),
                        min_interval_hours=int(_sess_cfg.get("min_interval_hours", 24)),
                        vacuum=bool(_sess_cfg.get("vacuum_after_prune", True)),
                        sessions_dir=self.config.sessions_dir,
                    )
            except Exception as exc:
                logger.debug("state.db auto-maintenance skipped: %s", exc)

        # Opportunistic shadow-repo cleanup — deletes orphan/stale
        # checkpoint repos under ~/.hermes/checkpoints/.  Opt-in via
        # checkpoints.auto_prune, idempotent via .last_prune marker.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _ckpt_cfg = (_load_full_config().get("checkpoints") or {})
            if _ckpt_cfg.get("auto_prune", False):
                from tools.checkpoint_manager import maybe_auto_prune_checkpoints
                maybe_auto_prune_checkpoints(
                    retention_days=int(_ckpt_cfg.get("retention_days", 7)),
                    min_interval_hours=int(_ckpt_cfg.get("min_interval_hours", 24)),
                    delete_orphans=bool(_ckpt_cfg.get("delete_orphans", True)),
                    max_total_size_mb=int(_ckpt_cfg.get("max_total_size_mb", 500)),
                )
        except Exception as exc:
            logger.debug("checkpoint auto-maintenance skipped: %s", exc)

        # DM pairing store for code-based user authorization
        from gateway.pairing import PairingStore
        self.pairing_store = PairingStore()
        
        # Event hook system
        from gateway.hooks import HookRegistry
        self.hooks = HookRegistry()

        # Per-chat voice reply mode: "off" | "voice_only" | "all"
        self._voice_mode: Dict[str, str] = self._load_voice_modes()
        # Recent voice transcripts per (guild,user) for duplicate suppression.
        # Protects against the same utterance being emitted twice by the voice
        # capture / STT pipeline, which otherwise produces a second delayed reply.
        self._recent_voice_transcripts: Dict[tuple[int, int], List[tuple[float, str]]] = {}

        # Track background tasks to prevent garbage collection mid-execution
        self._background_tasks: set = set()

        # scale-to-zero (Phase 0, F13): gateway-scoped "last inbound seen" clock.
        # There is no such clock today (only a per-agent _last_activity_ts), so the
        # idle predicate needs this. Stamped in _handle_message (the single inbound
        # chokepoint all adapters call); seeded to "now" so a fresh gateway isn't
        # considered idle from epoch. The scale-to-zero watcher (started only when
        # the instance is opted in + relay-only + has a wakeUrl) reads it.
        self._last_inbound_at: float = time.time()
        # Set after a wake (re-arm cooldown, 0.F) so we don't immediately re-go
        # dormant before the drained backlog has a chance to update the clock.
        self._scale_to_zero_cooldown_until: float = 0.0


    def _wire_teams_pipeline_runtime(self) -> None:
        """Bind the Teams meeting pipeline runtime to Graph webhook ingress.

        No-op when the msgraph_webhook adapter isn't running or the
        teams_pipeline plugin isn't enabled — lets the gateway start cleanly
        whether or not the user has opted into the pipeline.
        """
        if Platform.MSGRAPH_WEBHOOK not in self.adapters:
            return
        if not _teams_pipeline_plugin_enabled():
            logger.debug("Teams pipeline plugin is disabled; skipping runtime wiring")
            return
        try:
            from plugins.teams_pipeline.runtime import bind_gateway_runtime
        except Exception as exc:
            logger.warning("Teams pipeline runtime import failed: %s", exc)
            return
        try:
            bound = bind_gateway_runtime(self)
        except Exception as exc:
            logger.warning("Teams pipeline runtime wiring failed: %s", exc)
            return
        if bound:
            logger.info("Teams pipeline runtime bound to msgraph webhook ingress")
        elif self._teams_pipeline_runtime_error:
            logger.warning(
                "Teams pipeline runtime unavailable: %s",
                self._teams_pipeline_runtime_error,
            )


    def _warn_if_docker_media_delivery_is_risky(self) -> None:
        """Warn when Docker-backed gateways lack an explicit export mount.

        MEDIA delivery happens in the gateway process, so paths emitted by the model
        must be readable from the host. A plain container-local path like
        `/workspace/report.txt` or `/output/report.txt` often exists only inside
        Docker, so users commonly need a dedicated export mount such as
        `host-dir:/output`.
        """
        if os.getenv("TERMINAL_ENV", "").strip().lower() != "docker":
            return

        connected = self.config.get_connected_platforms()
        messaging_platforms = [p for p in connected if p not in {Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}]
        if not messaging_platforms:
            return

        raw_volumes = os.getenv("TERMINAL_DOCKER_VOLUMES", "").strip()
        volumes: List[str] = []
        if raw_volumes:
            try:
                parsed = json.loads(raw_volumes)
                if isinstance(parsed, list):
                    volumes = [str(v) for v in parsed if isinstance(v, str)]
            except Exception:
                logger.debug("Could not parse TERMINAL_DOCKER_VOLUMES for gateway media warning", exc_info=True)

        has_explicit_output_mount = False
        for spec in volumes:
            match = _DOCKER_VOLUME_SPEC_RE.match(spec)
            if not match:
                continue
            container_path = match.group("container")
            if container_path in _DOCKER_MEDIA_OUTPUT_CONTAINER_PATHS:
                has_explicit_output_mount = True
                break

        if has_explicit_output_mount:
            return

        logger.warning(
            "Docker backend is enabled for the messaging gateway but no explicit host-visible "
            "output mount (for example '/home/user/.hermes/cache/documents:/output') is configured. "
            "This is fine if the model already emits host-visible paths, but MEDIA file delivery can fail "
            "for container-local paths like '/workspace/...' or '/output/...'."
        )



    # -- Setup skill availability ----------------------------------------

    def _has_setup_skill(self) -> bool:
        """Check if the hermes-agent-setup skill is installed."""
        try:
            from tools.skill_manager_tool import _find_skill
            return _find_skill("hermes-agent-setup") is not None
        except Exception:
            return False

    # -- Voice mode persistence ------------------------------------------

    _VOICE_MODE_PATH = _hermes_home / "gateway_voice_mode.json"

    def _voice_key(self, platform: Platform, chat_id: str) -> str:
        """Return a platform-namespaced key for voice mode state."""
        return f"{platform.value}:{chat_id}"

    def _load_voice_modes(self) -> Dict[str, str]:
        try:
            data = json.loads(self._VOICE_MODE_PATH.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

        if not isinstance(data, dict):
            return {}

        valid_modes = {"off", "voice_only", "all"}
        result = {}
        for chat_id, mode in data.items():
            if mode not in valid_modes:
                continue
            key = str(chat_id)
            # Skip legacy unprefixed keys (warn and skip)
            if ":" not in key:
                logger.warning(
                    "Skipping legacy unprefixed voice mode key %r during migration. "
                    "Re-enable voice mode on that chat to rebuild the prefixed key.",
                    key,
                )
                continue
            result[key] = mode
        return result

    def _save_voice_modes(self) -> None:
        try:
            self._VOICE_MODE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._VOICE_MODE_PATH.write_text(
                json.dumps(self._voice_mode, indent=2)
            )
        except OSError as e:
            logger.warning("Failed to save voice modes: %s", e)

    def _set_adapter_auto_tts_disabled(self, adapter, chat_id: str, disabled: bool) -> None:
        """Update an adapter's in-memory auto-TTS suppression set if present."""
        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        if not isinstance(disabled_chats, set):
            return
        if disabled:
            disabled_chats.add(chat_id)
            # ``/voice off`` also clears any explicit enable — it's a hard override.
            enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
            if isinstance(enabled_chats, set):
                enabled_chats.discard(chat_id)
        else:
            disabled_chats.discard(chat_id)

    def _set_adapter_auto_tts_enabled(self, adapter, chat_id: str, enabled: bool) -> None:
        """Update an adapter's per-chat auto-TTS opt-in set if present.

        Used for ``/voice on``/``/voice tts`` where the user explicitly wants
        auto-TTS even when ``voice.auto_tts`` is False globally.
        """
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if not isinstance(enabled_chats, set):
            return
        if enabled:
            enabled_chats.add(chat_id)
            # An explicit opt-in clears any stale /voice off for this chat.
            disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
            if isinstance(disabled_chats, set):
                disabled_chats.discard(chat_id)
        else:
            enabled_chats.discard(chat_id)

    def _sync_voice_mode_state_to_adapter(self, adapter) -> None:
        """Restore persisted /voice state into a live platform adapter.

        Populates three fields from config + ``self._voice_mode``:
          - ``_auto_tts_default``: global default from ``voice.auto_tts``
          - ``_auto_tts_enabled_chats``: chats with mode ``voice_only``/``all``
          - ``_auto_tts_disabled_chats``: chats with mode ``off``
        """
        platform = getattr(adapter, "platform", None)
        if not isinstance(platform, Platform):
            return

        disabled_chats = getattr(adapter, "_auto_tts_disabled_chats", None)
        enabled_chats = getattr(adapter, "_auto_tts_enabled_chats", None)
        if not isinstance(disabled_chats, set) and not isinstance(enabled_chats, set):
            return

        # Push the global voice.auto_tts default (config.yaml) onto the adapter.
        # Lazy import to avoid adding a module-level dep from gateway → hermes_cli.
        try:
            from hermes_cli.config import load_config as _load_full_config
            _full_cfg = _load_full_config()
            _auto_tts_default = bool(
                (_full_cfg.get("voice") or {}).get("auto_tts", False)
            )
        except Exception:
            _auto_tts_default = False
        if hasattr(adapter, "_auto_tts_default"):
            adapter._auto_tts_default = _auto_tts_default

        prefix = f"{platform.value}:"
        if isinstance(disabled_chats, set):
            disabled_chats.clear()
            disabled_chats.update(
                key[len(prefix):] for key, mode in self._voice_mode.items()
                if mode == "off" and key.startswith(prefix)
            )
        if isinstance(enabled_chats, set):
            enabled_chats.clear()
            enabled_chats.update(
                key[len(prefix):] for key, mode in self._voice_mode.items()
                if mode in {"voice_only", "all"} and key.startswith(prefix)
            )

    async def _safe_adapter_disconnect(self, adapter, platform) -> None:
        """Call adapter.disconnect() defensively, swallowing any error.

        Used when adapter.connect() failed or raised — the adapter may
        have allocated partial resources (aiohttp.ClientSession, poll
        tasks, child subprocesses) that would otherwise leak and surface
        as "Unclosed client session" warnings at process exit.

        Must tolerate partial-init state and never raise, since callers
        use it inside error-handling blocks.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        try:
            if timeout <= 0:
                await adapter.disconnect()
            else:
                await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out after %.1fs while disconnecting %s adapter; continuing shutdown",
                timeout,
                platform.value if platform is not None else "adapter",
            )
        except Exception as e:
            logger.debug(
                "Defensive %s disconnect after failed connect raised: %s",
                platform.value if platform is not None else "adapter",
                e,
            )

    async def _bounded_adapter_teardown(
        self, adapter, platform, *, profile: Optional[str] = None
    ) -> None:
        """Tear down one adapter on the shutdown path with bounded awaits.

        Both ``cancel_background_tasks()`` and ``disconnect()`` can block
        indefinitely when a platform's network state is half-dead (e.g. a
        wedged Feishu/Lark WebSocket thread waiting on I/O). An unbounded
        await here stalls the entire shutdown sequence past systemd's
        ``TimeoutStopSec``; the resulting SIGKILL skips ``atexit`` PID-file
        cleanup, so the next start dies with "PID file race lost" (#14128).

        Each await is wrapped in the existing per-adapter timeout budget
        (``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT``). On timeout we log
        and force forward progress; the loop never hangs regardless of any
        adapter's internal behavior. Never raises.
        """
        timeout = self._adapter_disconnect_timeout_secs()
        suffix = f" (profile: {profile})" if profile else ""
        started_at = time.monotonic()
        try:
            if timeout <= 0:
                await adapter.cancel_background_tasks()
            else:
                await asyncio.wait_for(
                    adapter.cancel_background_tasks(), timeout=timeout
                )
        except asyncio.TimeoutError:
            logger.warning(
                "✗ %s background-task cancel timed out after %.1fs - forcing continue%s",
                platform.value, timeout, suffix,
            )
        except Exception as e:
            logger.debug("✗ %s background-task cancel error%s: %s", platform.value, suffix, e)
        try:
            if timeout <= 0:
                await adapter.disconnect()
            else:
                await asyncio.wait_for(adapter.disconnect(), timeout=timeout)
            logger.info(
                "✓ %s disconnected (%.2fs)%s",
                platform.value, time.monotonic() - started_at, suffix,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "✗ %s disconnect timed out after %.1fs - forcing continue%s",
                platform.value, timeout, suffix,
            )
        except Exception as e:
            logger.error(
                "✗ %s disconnect error after %.2fs%s: %s",
                platform.value, time.monotonic() - started_at, suffix, e,
            )

    def _adapter_disconnect_timeout_secs(self) -> float:
        """Return the per-adapter disconnect timeout used during shutdown."""
        raw = os.getenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        return _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT

    def _platform_connect_timeout_secs(self) -> float:
        """Return the per-platform connect timeout used during startup/retry."""
        raw = os.getenv("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", "").strip()
        if raw:
            try:
                timeout = float(raw)
            except ValueError:
                logger.warning(
                    "Ignoring invalid HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT=%r",
                    raw,
                )
            else:
                return max(0.0, timeout)
        return _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT

    async def _connect_adapter_with_timeout(
        self, adapter, platform, *, is_reconnect: bool = False
    ) -> bool:
        """Connect an adapter without allowing one platform to block others.

        ``is_reconnect`` is forwarded to ``adapter.connect()`` so platform
        adapters can distinguish a cold first boot (drop any stale
        server-side queue) from a watcher reconnect after a prolonged outage
        (preserve the queue so messages sent during the outage are delivered
        rather than silently dropped — #46621).
        """
        timeout = self._platform_connect_timeout_secs()
        if timeout <= 0:
            return await adapter.connect(is_reconnect=is_reconnect)
        try:
            return await asyncio.wait_for(
                adapter.connect(is_reconnect=is_reconnect), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"{platform.value} connect timed out after {timeout:g}s"
            ) from exc

    @property
    def should_exit_cleanly(self) -> bool:
        return self._exit_cleanly

    @property
    def should_exit_with_failure(self) -> bool:
        return self._exit_with_failure

    @property
    def exit_reason(self) -> Optional[str]:
        return self._exit_reason

    @property
    def exit_code(self) -> Optional[int]:
        return self._exit_code

    def _session_key_for_source(self, source: SessionSource) -> str:
        """Resolve the current session key for a source, honoring gateway config when available."""
        if hasattr(self, "session_store") and self.session_store is not None:
            try:
                session_key = self.session_store._generate_session_key(source)
                if isinstance(session_key, str) and session_key:
                    return session_key
            except Exception:
                pass
        config = getattr(self, "config", None)
        # Mirror SessionStore._resolve_profile_for_key so this fallback path
        # produces the same namespace as the primary path: None (legacy
        # agent:main) unless multiplexing is on, then the active profile.
        _profile = None
        if getattr(config, "multiplex_profiles", False):
            if source.profile:
                _profile = source.profile
            else:
                try:
                    from hermes_cli.profiles import get_active_profile_name
                    _profile = get_active_profile_name() or "default"
                except Exception:
                    _profile = None
        return build_session_key(
            source,
            group_sessions_per_user=getattr(config, "group_sessions_per_user", True),
            thread_sessions_per_user=getattr(config, "thread_sessions_per_user", False),
            profile=_profile,
        )

    def _telegram_topic_mode_enabled(self, source: SessionSource) -> bool:
        """Return whether Telegram DM topic mode is active for this chat."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return False
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            raw = session_db.is_telegram_topic_mode_enabled(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
            )
        except Exception:
            logger.debug("Failed to read Telegram topic mode state", exc_info=True)
            return False
        # Only honor a real True from the SessionDB. Any other value
        # (including MagicMock instances from test fixtures that didn't
        # opt into topic mode) means topic mode is off for this chat.
        return raw is True

    # Telegram's General (pinned top) topic in forum-enabled private chats.
    # Bot API behavior varies: some clients omit message_thread_id for
    # General, others send "1". Treat both as "root" for lobby/lane purposes.
    _TELEGRAM_GENERAL_TOPIC_IDS = frozenset({"", "1"})

    def _is_telegram_topic_root_lobby(self, source: SessionSource) -> bool:
        """True for the main Telegram DM (or General topic) when topic mode has made it a lobby."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        return tid in self._TELEGRAM_GENERAL_TOPIC_IDS

    def _is_telegram_topic_lane(self, source: SessionSource) -> bool:
        """True for a user-created Telegram private-chat topic lane."""
        if source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return False
        if not self._telegram_topic_mode_enabled(source):
            return False
        tid = str(source.thread_id or "")
        if not tid or tid in self._TELEGRAM_GENERAL_TOPIC_IDS:
            return False
        return True

    _TELEGRAM_LOBBY_REMINDER_COOLDOWN_S = 30.0

    def _should_send_telegram_lobby_reminder(self, source: SessionSource) -> bool:
        """Rate-limit root-DM lobby reminders to one message per cooldown window.

        A user who forgets multi-session mode is enabled and types several
        prompts in the root DM would otherwise get a reminder for every
        message. Cap it so the first one lands and the rest stay quiet.
        """
        if not hasattr(self, "_telegram_lobby_reminder_ts"):
            self._telegram_lobby_reminder_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_lobby_reminder_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_LOBBY_REMINDER_COOLDOWN_S:
            return False
        self._telegram_lobby_reminder_ts[chat_id] = now
        return True

    def _telegram_topic_root_lobby_message(self) -> str:
        return (
            "This main chat is reserved for system commands.\n\n"
            "To start a new Hermes chat, open the All Messages topic at the top "
            "of this bot interface and send any message there. Telegram will "
            "create a new topic for that message; each topic works as an "
            "independent Hermes session."
        )

    def _telegram_topic_root_new_message(self) -> str:
        return (
            "To start a new parallel Hermes chat, open the All Messages topic "
            "at the top of this bot interface and send any message there. "
            "Telegram will create a new topic for it.\n\n"
            "Each topic is an independent Hermes session. Use /new inside an "
            "existing topic only if you want to replace that topic's current session."
        )

    def _telegram_topic_new_header(self, source: SessionSource) -> Optional[str]:
        if not self._is_telegram_topic_lane(source):
            return None
        return (
            "Started a new Hermes session in this topic.\n\n"
            "Tip: for parallel work, open All Messages and send a message there "
            "to create a separate topic instead of using /new here. /new replaces "
            "the session attached to the current topic."
        )

    def _record_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
    ) -> None:
        """Persist the Telegram topic -> Hermes session binding for topic lanes."""
        session_db = getattr(self, "_session_db", None)
        if session_db is None or not source.chat_id or not source.thread_id:
            return
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        session_db.bind_telegram_topic(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
            user_id=str(source.user_id or ""),
            session_key=session_entry.session_key,
            session_id=session_entry.session_id,
        )

    def _sync_telegram_topic_binding(
        self,
        source: SessionSource,
        session_entry,
        *,
        reason: str,
    ) -> None:
        """Update the topic binding to point at ``session_entry.session_id``.

        Telegram topic lanes persist a (chat_id, thread_id) -> session_id row
        so reopening a topic in a fresh process resumes the right Hermes
        session. When compression rotates ``session_entry.session_id`` mid-turn,
        the binding goes stale and the next inbound message in that topic
        reloads the oversized parent transcript instead of the compressed
        child, retriggering preflight compression — sometimes in a loop
        (#20470, #29712, #33414).
        """
        if not self._is_telegram_topic_lane(source):
            return
        try:
            self._record_telegram_topic_binding(source, session_entry)
        except Exception:
            logger.debug(
                "telegram topic binding refresh failed (%s)", reason, exc_info=True,
            )

    def _recover_telegram_topic_thread_id(
        self,
        source: SessionSource,
    ) -> Optional[str]:
        """Pin DM-topic routing to the user's last-active topic.

        Telegram can omit ``message_thread_id`` or surface General (``1``)
        for some topic-mode DM replies. In those lobby-shaped cases, keep the
        conversation attached to the user's most-recent bound topic.

        Do not rewrite a non-lobby, previously-unbound thread id: a newly
        created Telegram DM topic is also "unknown" until the first inbound
        message is recorded, and rewriting it would send that brand-new topic's
        answer into an older lane. Returns None to leave the source alone.
        """
        if (
            source.platform != Platform.TELEGRAM
            or source.chat_type != "dm"
            or not source.chat_id
            or not source.user_id
            or not self._telegram_topic_mode_enabled(source)
        ):
            return None
        inbound = str(source.thread_id or "")
        is_lobby = not inbound or inbound in self._TELEGRAM_GENERAL_TOPIC_IDS
        if not is_lobby:
            # A non-lobby, unknown thread_id is most likely the first message in
            # a brand-new Telegram DM topic. Preserve it so it can be recorded
            # as a new independent lane below instead of hijacking the latest
            # existing topic binding.
            return None
        session_db = getattr(self, "_session_db", None)
        if session_db is None:
            return None
        # Runs off-loop (always via asyncio.to_thread); use the sync handle.
        session_db = getattr(session_db, "_db", session_db)
        try:
            bindings = session_db.list_telegram_topic_bindings_for_chat(
                chat_id=str(source.chat_id),
            )
        except Exception:
            logger.debug("topic-recover: read failed", exc_info=True)
            return None
        if not bindings:
            return None
        user_id = str(source.user_id)
        for b in bindings:  # newest-first
            if str(b.get("user_id") or "") == user_id:
                recovered = str(b.get("thread_id") or "")
                if recovered and recovered != inbound:
                    return recovered
                return None
        return None

    def _normalize_source_for_session_key(
        self,
        source: SessionSource,
    ) -> SessionSource:
        """Apply Telegram DM topic recovery to a source for session-key purposes.

        ``_handle_message_with_agent`` rewrites ``source.thread_id`` via
        ``_recover_telegram_topic_thread_id`` *before* deriving the session
        key for a normal message turn (a lobby/stripped reply gets pinned to
        the user's last-active topic).  Session-scoped command handlers like
        ``/model`` and ``/reasoning`` derive their override key from the raw
        inbound ``event.source``, which skips that recovery — so the override
        is stored under a different key than the next message turn reads,
        and the override is silently dropped on Telegram forum topics and
        after compression session splits (#30479).

        Returns a recovery-normalized copy when a rewrite applies, otherwise
        the original source unchanged.  Always derive the override storage key
        from the result so storage and read use an identical key.
        """
        try:
            recovered = self._recover_telegram_topic_thread_id(source)
        except Exception:
            return source
        if recovered is None:
            return source
        return dataclasses.replace(source, thread_id=recovered)

    def _resolve_session_agent_runtime(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session, honoring session-scoped /model overrides.

        If the session override already contains a complete provider bundle
        (provider/api_key/base_url/api_mode), prefer it directly instead of
        resolving fresh global runtime state first.
        """
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        model = _resolve_gateway_model(user_config)
        override = self._session_model_overrides.get(resolved_session_key) if resolved_session_key else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                "provider": override.get("provider"),
                "api_key": override.get("api_key"),
                "base_url": override.get("base_url"),
                "api_mode": override.get("api_mode"),
                "max_tokens": override.get("max_tokens"),
            }
            if override_runtime.get("api_key"):
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    resolved_session_key or "", model, override_model,
                    override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # Override exists but has no api_key — fall through to env-based
            # resolution and apply model/provider from the override on top.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                resolved_session_key or "", model, override_model,
            )
        else:
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                resolved_session_key or "", model,
                list(self._session_model_overrides.keys())[:5] if self._session_model_overrides else "[]",
            )

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info(
                "Runtime provider supplied explicit model override: %s -> %s",
                model,
                runtime_model,
            )
            model = runtime_model
        if override and resolved_session_key:
            model, runtime_kwargs = self._apply_session_model_override(
                resolved_session_key, model, runtime_kwargs
            )

        # When the config has no model.default but a provider was resolved
        # (e.g. user ran `hermes auth add openai-codex` without `hermes model`),
        # fall back to the provider's first catalog model so the API call
        # doesn't fail with "model must be a non-empty string".
        if not model and runtime_kwargs.get("provider"):
            try:
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s",
                        model, runtime_kwargs["provider"],
                    )
            except Exception:
                pass

        # Final safety net (#35314): if resolution still produced an empty
        # model — e.g. a transient config-cache miss during a post-interrupt
        # recovery turn returned an empty user_config — reuse the last model we
        # successfully resolved for this session (or, failing that, the most
        # recent one resolved process-wide). Building an agent with model=""
        # makes every API call fail HTTP 400 "No models provided" and the
        # session goes silent until the user manually re-sends. ``getattr``
        # guards against bare test runners built via ``object.__new__``.
        _last_good = getattr(self, "_last_resolved_model", None)
        if _last_good is not None:
            if not model:
                _recovered = _last_good.get(resolved_session_key or "") or _last_good.get("*")
                if _recovered:
                    logger.warning(
                        "Empty model resolved for session=%s — recovering "
                        "last-known-good model %s (config read likely returned "
                        "empty; see #35314)",
                        resolved_session_key or "", _recovered,
                    )
                    model = _recovered
            elif model:
                # Cache the good resolution for future recovery turns.
                if resolved_session_key:
                    _last_good[resolved_session_key] = model
                _last_good["*"] = model

        return model, runtime_kwargs

    def _resolve_turn_agent_config(self, user_message: str, model: str, runtime_kwargs: dict) -> dict:
        """Build the effective model/runtime config for a single turn.

        Always uses the session's primary model/provider.  If `/fast` is
        enabled and the model supports Priority Processing / Anthropic fast
        mode, attach `request_overrides` so the API call is marked
        accordingly.
        """
        from hermes_cli.models import resolve_fast_mode_overrides

        runtime = {
            "api_key": runtime_kwargs.get("api_key"),
            "base_url": runtime_kwargs.get("base_url"),
            "provider": runtime_kwargs.get("provider"),
            "api_mode": runtime_kwargs.get("api_mode"),
            "command": runtime_kwargs.get("command"),
            "args": list(runtime_kwargs.get("args") or []),
            "credential_pool": runtime_kwargs.get("credential_pool"),
            "max_tokens": runtime_kwargs.get("max_tokens"),
        }
        route = {
            "model": model,
            "runtime": runtime,
            "signature": (
                model,
                runtime["provider"],
                runtime["base_url"],
                runtime["api_mode"],
                runtime["command"],
                tuple(runtime["args"]),
            ),
        }

        service_tier = getattr(self, "_service_tier", None)
        if not service_tier:
            route["request_overrides"] = {}
            return route

        try:
            overrides = resolve_fast_mode_overrides(route["model"])
        except Exception:
            overrides = None
        route["request_overrides"] = overrides or {}
        return route

    def _sync_session_model_from_agent(self, session_id: str, agent: Any) -> None:
        """Persist the runtime model/provider actually used by a gateway turn.

        Provider fallback can switch ``agent.model``/``agent.provider`` after the
        session row was created. Keep the session DB metadata in sync so session
        lists, desktop/dashboard details, and follow-up session tooling report the
        backend that actually answered the latest turn.

        Called from the ``run_sync`` closure, which executes off the event loop
        in the executor thread — so the synchronous ``SessionDB`` (``_db``) is
        used directly rather than awaiting the AsyncSessionDB forwarder.
        """
        if not session_id or agent is None or self._session_db is None:
            return
        model = getattr(agent, "model", None)
        if not model:
            return
        runtime = {
            "provider": getattr(agent, "provider", None),
            "base_url": getattr(agent, "base_url", None),
            "api_mode": getattr(agent, "api_mode", None),
            "fallback_active": bool(getattr(agent, "_fallback_activated", False)),
        }
        runtime = {k: v for k, v in runtime.items() if v not in (None, "")}

        try:
            db = self._session_db._db
            row = db.get_session(session_id)
            if not row:
                return
            current_model = row.get("model")
            raw_config = row.get("model_config")
            try:
                config = json.loads(raw_config) if raw_config else {}
            except Exception:
                config = {}
            if not isinstance(config, dict):
                config = {}
            gateway_runtime = dict(config.get("gateway_runtime") or {})
            if current_model == model and all(
                gateway_runtime.get(k) == v for k, v in runtime.items()
            ):
                return
            config["gateway_runtime"] = runtime
            db.update_session_meta(session_id, json.dumps(config), model=model)
        except Exception:
            logger.debug("Failed to sync gateway session model metadata", exc_info=True)

    async def _handle_adapter_fatal_error(self, adapter: BasePlatformAdapter) -> None:
        """React to an adapter failure after startup.

        If the error is retryable (e.g. network blip, DNS failure), queue the
        platform for background reconnection instead of giving up permanently.
        """
        # Snapshot the current owner of this platform slot before doing
        # anything else. If it's neither this adapter nor empty, a different
        # adapter has already taken over (e.g. this is a delayed notification
        # from a background retry chain that raced with, and lost to, a
        # reconnect that already succeeded). Acting on a stale notification
        # would overwrite an already-healthy platform's runtime status and
        # incorrectly re-queue it for reconnection, so bail out before any of
        # that happens.
        existing = self.adapters.get(adapter.platform)
        if existing is not None and existing is not adapter:
            logger.debug(
                "Ignoring stale fatal error from a superseded %s adapter instance: %s",
                adapter.platform.value,
                adapter.fatal_error_code or "unknown",
            )
            return

        logger.error(
            "Fatal %s adapter error (%s): %s",
            adapter.platform.value,
            adapter.fatal_error_code or "unknown",
            adapter.fatal_error_message or "unknown error",
        )
        # Phase 7 Unit 7d-B: a relay credential revoked by opt-out is not an
        # error to retry — render it as a clean "disabled" state, not red
        # "fatal"/"retrying". (The code is set non-retryable, so it also drops
        # out of the reconnect queue below.)
        if adapter.fatal_error_code == "relay_disabled":
            platform_state = "disabled"
        elif adapter.fatal_error_retryable:
            platform_state = "retrying"
        else:
            platform_state = "fatal"
        self._update_platform_runtime_status(
            adapter.platform.value,
            platform_state=platform_state,
            error_code=adapter.fatal_error_code,
            error_message=adapter.fatal_error_message,
        )

        if existing is adapter:
            # Claim this adapter for teardown before awaiting disconnect() —
            # a second fatal-error notification for the same adapter (e.g.
            # from a concurrent recovery path) would otherwise still see
            # itself as "existing" during the await below and disconnect()
            # the same object twice.
            self.adapters.pop(adapter.platform, None)
            self.delivery_router.adapters = self.adapters
            await adapter.disconnect()

        # Queue retryable failures for background reconnection
        if adapter.fatal_error_retryable:
            platform_config = self.config.platforms.get(adapter.platform)
            if platform_config and adapter.platform not in self._failed_platforms:
                self._failed_platforms[adapter.platform] = {
                    "config": platform_config,
                    "attempts": 0,
                    "next_retry": time.monotonic(),
                }
                logger.info(
                    "%s queued for background reconnection",
                    adapter.platform.value,
                )

        if not self.adapters and not self._failed_platforms:
            self._exit_reason = adapter.fatal_error_message or "All messaging adapters disconnected"
            if adapter.fatal_error_retryable:
                self._exit_with_failure = True
                logger.error("No connected messaging platforms remain. Shutting down gateway for service restart.")
            else:
                logger.error("No connected messaging platforms remain. Shutting down gateway cleanly.")
            await self.stop()
        elif not self.adapters and self._failed_platforms:
            # All platforms are down and queued for background reconnection.
            # Keep the gateway alive so:
            #   • cron jobs still run
            #   • the reconnect watcher can recover platforms when the
            #     underlying problem clears (proxy comes back, user runs
            #     `hermes whatsapp`, etc.)
            # We used to exit-with-failure here to trigger systemd restart,
            # but that converted a transient outage into a restart loop and
            # killed in-process state every time. The reconnect watcher
            # already handles long-running recovery — let it do its job.
            logger.warning(
                "No connected messaging platforms remain, but %d platform(s) "
                "queued for reconnection — gateway staying alive, watcher will "
                "retry in background.",
                len(self._failed_platforms),
            )

    def _request_clean_exit(self, reason: str) -> None:
        self._exit_cleanly = True
        self._exit_reason = reason
        self._shutdown_event.set()

    def _running_agent_count(self) -> int:
        return len(self._running_agents)

    # ── scale-to-zero idle detection / dormant-quiesce (Phase 0) ──────────────
    # The gateway-side BEHAVIOUR that consumes the relay scale-to-zero primitives
    # (gateway-gateway Phase 5). Pure logic lives in gateway/scale_to_zero.py; the
    # methods here bind it to the live runner/transport. See ~/nous/specs/
    # scale-to-zero (decisions.md) for the design + the F12/F14 distinctions.

    def _scale_to_zero_has_live_background_work(self) -> bool:
        """Live background work that must block a suspend (D3/F7).

        Backgrounded delegate_task / kanban / terminal(background=true) are NOT
        counted by _running_agent_count(), but suspending mid-flight loses them.
        Checks the runner's own tracked tasks + the process registry's running
        processes + any pending process-completion watchers.
        """
        if any(not t.done() for t in self._background_tasks):
            return True
        try:
            from tools.async_delegation import active_count

            if active_count() > 0:
                return True
        except Exception:  # noqa: BLE001 - never let the idle check raise
            logger.debug("scale-to-zero async-delegation check failed", exc_info=True)
        try:
            from tools.process_registry import process_registry

            if process_registry.has_any_active():
                return True
            if process_registry.pending_watchers:
                return True
        except Exception:  # noqa: BLE001 - never let the idle check raise
            logger.debug("scale-to-zero bg-work check failed", exc_info=True)
        return False

    def _scale_to_zero_idle_timeout_seconds(self) -> float:
        from gateway.scale_to_zero import parse_idle_timeout_seconds

        raw = None
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            stz = gw.get("scale_to_zero") if isinstance(gw, dict) else None
            if isinstance(stz, dict):
                raw = stz.get("idle_timeout_minutes")
        except Exception:  # noqa: BLE001
            raw = None
        return parse_idle_timeout_seconds(raw)

    def _restart_loop_guard_config(self) -> tuple:
        """Return ``(max_restarts, window_seconds)`` for the auto-resume
        restart-loop breaker (#30719, defense-3), read from
        ``gateway.restart_loop_guard`` in config.yaml with the module defaults
        as fallback. ``max_restarts <= 0`` disables the breaker.
        """
        from gateway import restart_loop_guard as _rlg

        max_restarts = _rlg.DEFAULT_MAX_RESTARTS
        window_seconds = _rlg.DEFAULT_WINDOW_SECONDS
        try:
            user_cfg = _load_gateway_config()
            gw = user_cfg.get("gateway") if isinstance(user_cfg, dict) else None
            rlg = gw.get("restart_loop_guard") if isinstance(gw, dict) else None
            if isinstance(rlg, dict):
                if isinstance(rlg.get("max_restarts"), int):
                    max_restarts = rlg["max_restarts"]
                if isinstance(rlg.get("window_seconds"), int) and rlg["window_seconds"] > 0:
                    window_seconds = rlg["window_seconds"]
        except Exception:  # noqa: BLE001
            pass
        return max_restarts, window_seconds

    def _scale_to_zero_should_arm(self) -> bool:
        """Whether to start the idle watcher (D1/D11/§3.4(1))."""
        from gateway.relay import relay_wake_url
        from gateway.scale_to_zero import (
            messaging_is_relay_only_or_absent,
            scale_to_zero_enabled,
            should_arm,
        )

        try:
            # Only ENABLED platforms count. `config.platforms` is pre-seeded with a
            # disabled placeholder PlatformConfig for every KNOWN platform (telegram,
            # discord, slack, …), so `.keys()` is the full ~20-entry catalog regardless
            # of what this instance actually runs. Passing the bare keys made
            # `messaging_is_relay_only_or_absent` see those placeholders as live
            # direct-socket platforms and return False, so scale-to-zero NEVER armed on
            # a real relay-only instance. Mirror the connect loop, which already gates on
            # `platform_config.enabled` (see the `if not platform_config.enabled: continue`
            # in the adapter-connect loop) — arm off the same notion of "active platform."
            platforms = (
                [p for p, pc in self.config.platforms.items() if getattr(pc, "enabled", False)]
                if self.config
                else []
            )
        except Exception:  # noqa: BLE001
            platforms = []
        try:
            wake_url = relay_wake_url()
        except Exception:  # noqa: BLE001
            wake_url = None
        return should_arm(
            enabled=scale_to_zero_enabled(),
            relay_only_or_absent=messaging_is_relay_only_or_absent(platforms),
            wake_url=wake_url,
        )

    def _log_scale_to_zero_not_armed_reason(self) -> None:
        """Log why the idle watcher did NOT arm — but only for an OPTED-IN instance.

        A non-opted instance (no HERMES_SCALE_TO_ZERO stamp) not arming is the normal
        case and must stay silent. When the Labs stamp IS set but the watcher still
        didn't arm, that's the surprising case worth one INFO line so "why won't it
        suspend/wake?" is a log grep, not a box-dive.
        """
        from gateway.relay import relay_wake_url
        from gateway.scale_to_zero import (
            messaging_is_relay_only_or_absent,
            scale_to_zero_enabled,
        )

        try:
            enabled = scale_to_zero_enabled()
            if not enabled:
                return  # not opted in — normal, stay quiet
            try:
                active = (
                    [
                        getattr(p, "value", p)
                        for p, pc in self.config.platforms.items()
                        if getattr(pc, "enabled", False)
                    ]
                    if self.config
                    else []
                )
            except Exception:  # noqa: BLE001
                active = []
            relay_only = messaging_is_relay_only_or_absent(active)
            try:
                wake_url = relay_wake_url()
            except Exception:  # noqa: BLE001
                wake_url = None
            logger.info(
                "scale-to-zero: NOT armed despite opt-in — "
                "relay_only_or_absent=%s (enabled platforms=%s), wake_url=%s. "
                "Need relay-only messaging + a registered wake URL.",
                relay_only,
                active or "none",
                "set" if wake_url else "MISSING",
            )
        except Exception:  # noqa: BLE001 - diagnostics must never block startup
            logger.debug("scale-to-zero: not-armed reason logging failed", exc_info=True)

    def _scale_to_zero_is_idle(self) -> bool:
        from gateway.scale_to_zero import is_idle

        return is_idle(
            running_agent_count=self._running_agent_count(),
            seconds_since_last_inbound=time.time() - self._last_inbound_at,
            idle_timeout_seconds=self._scale_to_zero_idle_timeout_seconds(),
            has_live_background_work=self._scale_to_zero_has_live_background_work(),
        )

    def _scale_to_zero_note_real_inbound(self) -> None:
        """Stamp real inbound and restore lifecycle after a dormant wake.

        The watcher marks runtime status `draining` as it quiesces the relay, but
        dormancy is not the stop/restart drain path: the process remains alive and
        should present as running once real traffic wakes it and re-enters the
        gateway. Internal completion/replay events intentionally do not call this
        helper, so they do not keep an otherwise idle gateway awake.
        """
        self._last_inbound_at = time.time()
        if getattr(self, "_scale_to_zero_cooldown_until", 0.0) > 0:
            try:
                self._update_runtime_status("running")
            except Exception:  # noqa: BLE001 - status restoration is best-effort
                logger.debug("scale-to-zero: status restore failed", exc_info=True)
            self._scale_to_zero_cooldown_until = 0.0

    def _relay_adapter_for_dormancy(self):
        """Return the connected RELAY adapter, if any (the one go_dormant targets)."""
        try:
            from gateway.platforms.base import Platform
        except Exception:  # noqa: BLE001
            return None
        return self.adapters.get(Platform.RELAY)

    async def _scale_to_zero_watcher(self, interval: float = 30.0) -> None:
        """Watch for idle and drive the relay dormant so the platform can suspend.

        Started ONLY when _scale_to_zero_should_arm() (opted in via the Labs
        HERMES_SCALE_TO_ZERO stamp + relay-only/absent messaging + a wakeUrl).
        On a sustained idle window it runs the DORMANT sequence (D12/F12/F14):
          - mark runtime status `draining` (composes with the existing state
            machine, §3.4(6); does NOT set _running=False),
          - relay adapter.go_dormant() — going_idle->ack + supervisor-preserving
            socket close (NOT disconnect(), NOT the run.py stop path),
          - deliberately NO mark_resume_pending (D13 — suspend preserves RAM).
        The process stays alive; the platform (Fly autostop:"suspend") suspends
        the now-traffic-idle machine and autostart wakes it on the wakeUrl poke,
        at which point the preserved reconnect supervisor re-dials and the
        connector drains the buffered backlog. After driving dormant we set a
        re-arm cooldown so a wake's drained backlog isn't immediately re-quiesced.
        """
        await asyncio.sleep(min(interval, 30.0))  # let startup settle
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    return
                if time.time() < self._scale_to_zero_cooldown_until:
                    continue
                if not self._scale_to_zero_is_idle():
                    continue
                adapter = self._relay_adapter_for_dormancy()
                if adapter is None:
                    continue
                go_dormant = getattr(adapter, "go_dormant", None)
                if not callable(go_dormant):
                    continue
                logger.info(
                    "scale-to-zero: gateway idle for >= %.0fs — going dormant "
                    "(relay buffered, socket closed, awaiting platform suspend)",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                try:
                    self._update_runtime_status("draining")
                except Exception:  # noqa: BLE001 - status is best-effort
                    logger.debug("scale-to-zero: status mark failed", exc_info=True)
                try:
                    result = go_dormant()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:  # noqa: BLE001 - dormancy is best-effort
                    logger.debug("scale-to-zero: go_dormant failed", exc_info=True)
                # 0.F: after a wake the drained inbound updates _last_inbound_at,
                # but give it a window so we don't immediately re-go-dormant on the
                # same idle reading before traffic lands.
                self._scale_to_zero_cooldown_until = time.time() + max(interval, 60.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the watcher must never crash the gateway
                logger.debug("scale-to-zero watcher iteration error", exc_info=True)

    def _status_action_label(self) -> str:
        return "restart" if self._restart_requested else "shutdown"

    def _status_action_gerund(self) -> str:
        return "restarting" if self._restart_requested else "shutting down"

    def _queue_during_drain_enabled(self) -> bool:
        # Both "queue" and "steer" modes imply the user doesn't want messages
        # to be lost during restart — queue them for the newly-spawned gateway
        # process to pick up.  "interrupt" mode drops them (current behaviour).
        return self._restart_requested and self._busy_input_mode in {"queue", "steer"}

    # -------- /queue FIFO helpers --------------------------------------
    # /queue must produce one full agent turn per invocation, in FIFO
    # order, with no merging.  The adapter's _pending_messages dict is a
    # single "next-up" slot (shared with photo-burst follow-ups), so we
    # use it for the head of the queue and an overflow list for the
    # tail.  Enqueue puts new items in the slot when free, otherwise in
    # the overflow.  Promotion (called after each run's drain) moves the
    # next overflow item into the slot so the following recursion picks
    # it up.  Clearing happens on /new and /reset via
    # _handle_reset_command.

    def _enqueue_fifo(self, session_key: str, queued_event: "MessageEvent", adapter: Any) -> None:
        """Append a /queue event to the FIFO chain for a session."""
        if adapter is None:
            return
        pending_slot = getattr(adapter, "_pending_messages", None)
        if pending_slot is None:
            return
        queued_events = getattr(self, "_queued_events", None)
        if queued_events is None:
            queued_events = {}
            self._queued_events = queued_events
        if session_key in pending_slot:
            queued_events.setdefault(session_key, []).append(queued_event)
        else:
            pending_slot[session_key] = queued_event

    def _promote_queued_event(
        self,
        session_key: str,
        adapter: Any,
        pending_event: Optional["MessageEvent"],
    ) -> Optional["MessageEvent"]:
        """Promote the next overflow item after the slot was drained.

        Called at the drain site after _dequeue_pending_event consumed
        (or failed to consume) the slot.  If there's an overflow item:
          - When pending_event is None (slot was empty), return the
            overflow head as the new pending_event.
          - When pending_event already exists (slot was populated by an
            interrupt follow-up or similar), stage the overflow head in
            the slot so the NEXT recursion picks it up.
        Returns the (possibly updated) pending_event for drain to use.
        """
        queued_events = getattr(self, "_queued_events", None)
        if not queued_events:
            return pending_event
        overflow = queued_events.get(session_key)
        if not overflow:
            return pending_event
        next_queued = overflow.pop(0)
        if not overflow:
            queued_events.pop(session_key, None)
        if pending_event is None:
            return next_queued
        if adapter is not None and hasattr(adapter, "_pending_messages"):
            adapter._pending_messages[session_key] = next_queued
        else:
            # No adapter — push back so we don't silently drop the item.
            queued_events.setdefault(session_key, []).insert(0, next_queued)
        return pending_event

    def _queue_depth(self, session_key: str, *, adapter: Any = None) -> int:
        """Total pending /queue items for a session — slot + overflow."""
        queued_events = getattr(self, "_queued_events", None) or {}
        depth = len(queued_events.get(session_key, []))
        if adapter is not None and session_key in getattr(adapter, "_pending_messages", {}):
            depth += 1
        return depth

    @staticmethod
    def _is_goal_continuation_event(event_or_text: Any) -> bool:
        """Return True for synthetic /goal continuation turns.

        Goal continuations are normal queued user-role events, so pause/clear
        must distinguish them from real user /queue messages before removing or
        suppressing them.
        """
        text = getattr(event_or_text, "text", event_or_text) or ""
        return str(text).startswith("[Continuing toward your standing goal]\nGoal:")

    def _clear_goal_pending_continuations(self, session_key: str, adapter: Any) -> int:
        """Remove queued synthetic /goal continuations for one session.

        User-issued /goal pause/clear can race with a continuation already
        queued by the judge.  Remove only synthetic goal continuations while
        preserving normal /queue and user follow-up events.
        """
        removed = 0
        pending_slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        if isinstance(pending_slot, dict):
            pending_event = pending_slot.get(session_key)
            if self._is_goal_continuation_event(pending_event):
                pending_slot.pop(session_key, None)
                removed += 1

        queued_events = getattr(self, "_queued_events", None)
        if isinstance(queued_events, dict):
            overflow = queued_events.get(session_key) or []
            if overflow:
                kept = []
                for queued_event in overflow:
                    if self._is_goal_continuation_event(queued_event):
                        removed += 1
                    else:
                        kept.append(queued_event)
                if kept:
                    queued_events[session_key] = kept
                else:
                    queued_events.pop(session_key, None)
        return removed

    def _goal_still_active_for_session(self, session_id: str) -> bool:
        """Best-effort fresh DB check before running a queued continuation."""
        if not session_id:
            return False
        try:
            from hermes_cli.goals import GoalManager
            return GoalManager(session_id=session_id).is_active()
        except Exception as exc:
            logger.debug("goal continuation: active-state recheck failed: %s", exc)
            return False

    def _update_runtime_status(self, gateway_state: Optional[str] = None, exit_reason: Optional[str] = None) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                gateway_state=gateway_state,
                exit_reason=exit_reason,
                restart_requested=self._restart_requested,
                active_agents=self._running_agent_count(),
            )
        except Exception:
            pass

    def _persist_active_agents(self) -> None:
        """Persist the live in-flight agent count to ``gateway_state.json``.

        Called at every turn boundary (a running-agent slot is claimed or
        released) so the dashboard ``/api/status`` readout reflects in-flight
        gateway turns in near-real-time.  Without this the file is only
        rewritten on lifecycle transitions, so any ``active_agents`` read
        between transitions is stale (a turn could start and finish without the
        file ever moving).

        Deliberately passes ONLY ``active_agents`` — ``gateway_state`` and the
        other fields stay ``_UNSET`` so ``write_runtime_status``'s
        read-merge-write preserves the current lifecycle state (``running`` /
        ``draining`` / …).  Passing ``gateway_state=None`` here would clobber it.
        Best-effort: a failed status write must never disrupt a turn.
        """
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(active_agents=self._running_agent_count())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # External drain control (NAS-driven quiesce-without-restart, Phase 2).
    # The dashboard's begin/cancel-drain endpoint writes/removes the
    # ``.drain_request.json`` marker (gateway/drain_control.py); this watcher
    # observes the marker and flips the gateway between accepting and refusing
    # NEW turns, WITHOUT exiting the process. Reversible by design (D4a): NAS
    # POSTs begin-drain, polls /api/status until active_agents hits 0, proceeds
    # with its lifecycle action, then (on cancel/abort) the marker is removed
    # and the gateway re-accepts turns.
    # ------------------------------------------------------------------
    def _enter_external_drain(self) -> None:
        """Begin external drain: stop accepting new turns, flip state.

        Idempotent — re-entering while already draining is a no-op beyond a
        best-effort status re-write. In-flight turns are NOT interrupted (the
        whole point is to let them finish); only NEW turns are refused.
        """
        if self._external_drain_active:
            return
        self._external_drain_active = True
        logger.info(
            "External drain ENGAGED (.drain_request.json present) — refusing "
            "new turns; %d in-flight turn(s) will finish. Process stays up.",
            self._running_agent_count(),
        )
        # Flip the persisted lifecycle state so /api/status.gateway_busy /
        # gateway_drainable track the drain. Preserve active_agents (the
        # read-merge keeps the live count); only the state changes.
        self._update_runtime_status("draining")

    def _exit_external_drain(self) -> None:
        """Cancel external drain: revert state, re-accept new turns.

        Idempotent. Only reverts to ``running`` when we are actually mid-drain
        AND not also shutting down (a real shutdown ``_draining`` must win —
        never resurrect a stopping gateway to ``running``).
        """
        if not self._external_drain_active:
            return
        self._external_drain_active = False
        if self._draining or not self._running:
            # A shutdown drain is in progress / the loop has stopped — do not
            # clobber the terminal state back to running.
            logger.info(
                "External drain marker cleared during shutdown — not reverting "
                "to running (shutdown takes precedence)."
            )
            return
        logger.info(
            "External drain RELEASED (.drain_request.json removed) — "
            "re-accepting new turns; gateway_state -> running."
        )
        self._update_runtime_status("running")

    async def _drain_control_watcher(self, interval: float = 1.0) -> None:
        """Background task: reconcile gateway accept-state with the drain marker.

        Polls ``.drain_request.json`` (presence-based contract,
        gateway/drain_control.py). Marker present -> ``_enter_external_drain``;
        marker absent -> ``_exit_external_drain``. The 1s cadence bounds the
        observe-the-marker latency the live-validation gate checks (point a).
        Reconciles once at startup. A marker stamped with a PRIOR
        instantiation epoch (one that survived a machine restart on the durable
        HERMES_HOME volume — NS-570) is treated as absent by ``drain_requested``
        and is NOT honoured; only a marker from the current instantiation flips
        the gateway into drain. Best-effort: any tick error is logged and the
        loop continues (a transient stat() failure must not wedge the gateway).
        """
        from gateway.drain_control import drain_requested

        while self._running:
            try:
                if drain_requested():
                    self._enter_external_drain()
                else:
                    self._exit_external_drain()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Drain-control watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    def _update_platform_runtime_status(
        self,
        platform: str,
        *,
        platform_state: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(
                platform=platform,
                platform_state=platform_state,
                error_code=error_code,
                error_message=error_message,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-platform circuit breaker (pause/resume) — used by the reconnect
    # watcher when a retryable failure recurs past a threshold, and by the
    # /platform pause|resume slash command for manual control.
    # ------------------------------------------------------------------
    def _pause_failed_platform(self, platform, *, reason: str = "") -> None:
        """Mark a queued platform as paused — keep it in ``_failed_platforms``
        but stop the reconnect watcher from hammering it.

        Used by ``/platform pause <name>`` for manual operator intervention.
        Paused platforms are surfaced in ``/platform list`` and resumed with
        ``/platform resume <name>``.  Note: the reconnect watcher does NOT
        auto-pause — retryable (network/DNS) failures keep retrying at the
        backoff cap indefinitely so a transient outage self-heals without
        manual intervention.
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return
        if info.get("paused"):
            return
        info["paused"] = True
        info["pause_reason"] = reason or "auto-paused after repeated failures"
        # Push next_retry far enough out that even if "paused" is missed
        # by a stale code path, the watcher won't fire on it.
        info["next_retry"] = float("inf")
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="paused",
                error_code=None,
                error_message=info["pause_reason"],
            )
        except Exception:
            pass
        logger.warning(
            "%s paused after %d consecutive failures (%s) — "
            "fix the underlying issue then run `/platform resume %s` "
            "to retry, or `hermes gateway restart` to restart the gateway.",
            platform.value, info.get("attempts", 0),
            info["pause_reason"], platform.value,
        )

    def _resume_paused_platform(self, platform) -> bool:
        """Unpause a platform — reset its attempt counter and schedule an
        immediate retry.  Returns True if the platform was paused and is
        now queued; False if it wasn't paused (or wasn't in the queue).
        """
        info = getattr(self, "_failed_platforms", {}).get(platform)
        if info is None:
            return False
        if not info.get("paused"):
            return False
        info["paused"] = False
        info.pop("pause_reason", None)
        info["attempts"] = 0
        info["next_retry"] = time.monotonic()  # retry on next watcher tick
        try:
            self._update_platform_runtime_status(
                platform.value,
                platform_state="retrying",
                error_code=None,
                error_message=None,
            )
        except Exception:
            pass
        logger.info("%s resumed — retrying on next watcher tick", platform.value)
        return True

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.
        
        Checks HERMES_PREFILL_MESSAGES_FILE env var first, then falls back to
        the top-level prefill_messages_file key in ~/.hermes/config.yaml.
        agent.prefill_messages_file is accepted as a legacy fallback.
        Relative paths are resolved from ~/.hermes/.
        """
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            cfg = _load_gateway_runtime_config()
            file_path = str(cfg.get("prefill_messages_file", "") or "")
            if not file_path:
                file_path = str(cfg_get(cfg, "agent", "prefill_messages_file", default="") or "")
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt from config or env var.
        
        Checks HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then falls back to
        agent.system_prompt in ~/.hermes/config.yaml.
        """
        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        cfg = _load_gateway_runtime_config()
        return str(cfg_get(cfg, "agent", "system_prompt", default="") or "").strip()

    @staticmethod
    def _load_reasoning_config() -> dict | None:
        """Load reasoning effort from config.yaml.

        Reads agent.reasoning_effort from config.yaml. Valid: "none",
        "minimal", "low", "medium", "high", "xhigh". Returns None to use
        default (medium).
        """
        from hermes_constants import parse_reasoning_effort
        cfg = _load_gateway_runtime_config()
        effort = str(cfg_get(cfg, "agent", "reasoning_effort", default="") or "").strip()
        result = parse_reasoning_effort(effort)
        if effort and effort.strip() and result is None:
            logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
        return result

    @staticmethod
    def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
        """Parse `/reasoning` args into `(value, persist_global)`.

        `/reasoning <level>` is session-scoped by default. `--global` may be
        supplied in any position to persist the change to config.yaml.
        """
        import shlex

        text = str(raw_args or "").strip().replace("—", "--")
        if not text:
            return "", False
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        persist_global = False
        value_tokens = []
        for token in tokens:
            if token == "--global":
                persist_global = True
            else:
                value_tokens.append(token)
        return " ".join(value_tokens).strip().lower(), persist_global

    def _resolve_session_reasoning_config(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
    ) -> dict | None:
        """Resolve reasoning effort for a session, honoring session overrides."""
        resolved_session_key = session_key
        if not resolved_session_key and source is not None:
            try:
                resolved_session_key = self._session_key_for_source(source)
            except Exception:
                resolved_session_key = None

        overrides = getattr(self, "_session_reasoning_overrides", {}) or {}
        if resolved_session_key and resolved_session_key in overrides:
            return overrides[resolved_session_key]
        return self._load_reasoning_config()

    def _set_session_reasoning_override(
        self,
        session_key: str,
        reasoning_config: Optional[dict],
    ) -> None:
        """Set or clear the session-scoped reasoning override."""
        if not session_key:
            return
        if not hasattr(self, "_session_reasoning_overrides"):
            self._session_reasoning_overrides = {}
        if reasoning_config is None:
            self._session_reasoning_overrides.pop(session_key, None)
        else:
            self._session_reasoning_overrides[session_key] = dict(reasoning_config)

    @staticmethod
    def _load_service_tier() -> str | None:
        """Load Priority Processing setting from config.yaml.

        Reads agent.service_tier from config.yaml. Accepted values mirror the CLI:
        "fast"/"priority"/"on" => "priority", while "normal"/"off" disables it.
        Returns None when unset or unsupported.
        """
        cfg = _load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()

        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        cfg = _load_gateway_runtime_config()
        return is_truthy_value(
            cfg_get(cfg, "display", "show_reasoning"),
            default=False,
        )

    @staticmethod
    def _load_busy_input_mode() -> str:
        """Load gateway drain-time busy-input behavior from config/env."""
        mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
        if not mode:
            cfg = _load_gateway_runtime_config()
            mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
        if mode == "queue":
            return "queue"
        if mode == "steer":
            return "steer"
        return "interrupt"

    @staticmethod
    def _load_busy_text_mode() -> str:
        """Resolve normal busy TEXT follow-up behavior.

        ``busy_input_mode`` is the single source of truth (default
        ``interrupt``). The legacy ``busy_text_mode`` knob is honored only
        when a user explicitly set it, so existing queue setups keep
        working; new installs follow ``busy_input_mode``. Returns one of
        ``interrupt`` | ``queue`` (``steer`` is handled upstream by
        ``busy_input_mode`` and maps to non-queue text handling here).
        """
        # Legacy explicit override wins for backward compat.
        legacy = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
        if not legacy:
            cfg = _load_gateway_runtime_config()
            legacy = str(cfg_get(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
        if legacy == "interrupt":
            return "interrupt"
        if legacy == "queue":
            return "queue"
        # No explicit legacy knob → follow busy_input_mode.
        input_mode = GatewayRunner._load_busy_input_mode()
        return "queue" if input_mode == "queue" else "interrupt"

    @staticmethod
    def _load_restart_drain_timeout() -> float:
        """Load graceful gateway restart/stop drain timeout in seconds."""
        raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if not raw:
            cfg = _load_gateway_runtime_config()
            raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var.

        Modes:
          - ``all``    — push running-output updates *and* the final message (default)
          - ``result`` — only the final completion message (regardless of exit code)
          - ``error``  — only the final message when exit code is non-zero
          - ``off``    — no watcher messages at all
        """
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "display", "background_process_notifications")
            if raw is False:
                mode = "off"
            elif raw not in {None, ""}:
                mode = str(raw)
        mode = (mode or "all").strip().lower()
        valid = {"all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'all'",
                mode,
            )
            return "all"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> list | None:
        """Load fallback provider chain from config.yaml.

        Returns the merged effective chain from ``fallback_providers`` plus any
        legacy ``fallback_model`` entries. ``fallback_providers`` stays first
        when both keys are present.
        """
        try:
            import yaml as _y
            cfg_path = _hermes_home / "config.yaml"
            if cfg_path.exists():
                with open(cfg_path, encoding="utf-8") as _f:
                    cfg = _y.safe_load(_f) or {}
                fb = get_fallback_chain(cfg)
                if fb:
                    return fb
        except Exception:
            pass
        return None

    def _snapshot_running_agents(self) -> Dict[str, Any]:
        return {
            session_key: agent
            for session_key, agent in self._running_agents.items()
            if agent is not _AGENT_PENDING_SENTINEL
        }

    def _get_max_concurrent_sessions(self) -> Optional[int]:
        """Return the configured active chat session cap, if enabled."""
        try:
            from hermes_cli.active_sessions import resolve_max_concurrent_sessions

            return resolve_max_concurrent_sessions(getattr(self, "config", None))
        except Exception:
            return None

    def _active_session_limit_message(self, session_key: str) -> Optional[str]:
        """Return a user-facing rejection when starting a new session exceeds the cap."""
        max_sessions = self._get_max_concurrent_sessions()
        if max_sessions is None:
            return None
        if session_key in getattr(self, "_running_agents", {}):
            return None
        active_count = len(getattr(self, "_running_agents", {}))
        if active_count < max_sessions:
            return None
        return (
            f"Hermes is at the active session limit ({active_count}/{max_sessions}). "
            "Try again when another session finishes."
        )

    def _claim_active_session_slot(
        self,
        session_key: str,
        source: SessionSource,
    ) -> tuple[Any, Optional[str]]:
        """Claim a cross-process active-session slot for a new gateway turn."""
        if session_key in getattr(self, "_running_agents", {}):
            return None, None
        local_limit_message = self._active_session_limit_message(session_key)
        if local_limit_message is not None:
            return None, local_limit_message
        try:
            from hermes_cli.active_sessions import try_acquire_active_session

            platform = source.platform.value if source and source.platform else "gateway"
            return try_acquire_active_session(
                session_id=session_key,
                surface=f"gateway:{platform}",
                config=getattr(self, "config", None),
                metadata={
                    "platform": platform,
                    "chat_id": getattr(source, "chat_id", "") or "",
                    "user_id": getattr(source, "user_id", "") or "",
                },
            )
        except Exception as exc:
            logger.warning("Failed to claim active session slot: %s", exc)
            return None, None

    @staticmethod
    def _agent_has_active_subagents(running_agent: Any) -> bool:
        """Return True when *running_agent* is currently driving subagents
        via the ``delegate_task`` tool.

        Background (#30170): ``AIAgent.interrupt()`` cascades through the
        parent's ``_active_children`` list and calls ``interrupt()`` on
        every child synchronously, which aborts in-flight subagent work
        and produces a fallback cascade with no actionable signal.
        Demoting ``busy_input_mode='interrupt'`` to ``queue`` semantics
        whenever this helper returns True protects subagent work from
        conversational follow-ups while leaving the explicit ``/stop``
        path (which goes through ``_interrupt_and_clear_session``)
        untouched. Safe-by-default: returns False on any attribute or
        lock error so a missing/broken parent never blocks the existing
        interrupt path.
        """
        if running_agent is None or running_agent is _AGENT_PENDING_SENTINEL:
            return False
        children = getattr(running_agent, "_active_children", None)
        # AIAgent always initialises this as a concrete list (see
        # agent/agent_init.py). Reject anything that isn't a real
        # collection — this guards against ``MagicMock()._active_children``
        # auto-creating a truthy stub in tests and triggering the demotion
        # against an agent that doesn't actually have subagents.
        if not isinstance(children, (list, tuple, set)):
            return False
        if not children:
            return False
        lock = getattr(running_agent, "_active_children_lock", None)
        try:
            if lock is not None:
                with lock:
                    return bool(children)
            return bool(children)
        except Exception:
            return False

    # Hard cap on per-session pending follow-ups for busy_input_mode=queue
    # (and the draining/steer-fallback/subagent-demotion paths that share
    # this entry point).  Without a cap, a stuck agent + a rapid-fire user
    # could grow the overflow list unboundedly.  32 turns of queued
    # follow-ups is far beyond any realistic conversational backlog while
    # still small enough to never threaten memory.
    _BUSY_QUEUE_MAX_PENDING = 32

    def _queue_or_replace_pending_event(self, session_key: str, event: MessageEvent) -> None:
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return
        # #28503 — Previously this called ``merge_pending_message_event``
        # with the default ``merge_text=False``, which silently OVERWROTE
        # the single pending slot when consecutive text messages arrived
        # in ``busy_input_mode: queue``. Route through the FIFO
        # infrastructure shared with ``/queue`` so each follow-up gets
        # its own turn in arrival order. Photo bursts still merge into
        # the head slot via ``merge_pending_message_event`` (album
        # semantics); everything else appends to the overflow tail.
        pending_slot = getattr(adapter, "_pending_messages", None)
        existing = pending_slot.get(session_key) if isinstance(pending_slot, dict) else None
        if existing is not None and (
            getattr(existing, "message_type", None) == MessageType.PHOTO
            or event.message_type == MessageType.PHOTO
            or bool(getattr(existing, "media_urls", None))
            or bool(getattr(event, "media_urls", None))
        ):
            # Preserve photo-burst / media-merge semantics for the head slot.
            merge_pending_message_event(
                adapter._pending_messages,
                session_key,
                event,
                merge_text=event.message_type == MessageType.TEXT,
            )
            return

        if self._queue_depth(session_key, adapter=adapter) >= self._BUSY_QUEUE_MAX_PENDING:
            logger.warning(
                "Dropping busy-mode follow-up for session %s — pending queue at cap (%d).",
                session_key,
                self._BUSY_QUEUE_MAX_PENDING,
            )
            return

        self._enqueue_fifo(session_key, event, adapter)

    async def _handle_active_session_busy_message(self, event: MessageEvent, session_key: str) -> bool:
        # --- Authorization gate (#17775) ---
        # The cold path (_handle_message) checks _is_user_authorized before
        # creating a session.  The busy path must enforce the same check;
        # otherwise unauthorized users in shared threads (Slack/Telegram/Discord)
        # can inject messages into an active session they don't own.
        if not self._is_user_authorized(event.source):
            logger.warning(
                "Dropping message from unauthorized user in active session: "
                "user=%s (%s), platform=%s, session=%s",
                event.source.user_id,
                event.source.user_name,
                event.source.platform.value if event.source.platform else "unknown",
                session_key,
            )
            return True  # handled (silently dropped); do not fall through

        # --- Draining case (gateway restarting/stopping) ---
        if self._draining:
            adapter = self._adapter_for_source(event.source)
            if not adapter:
                return True

            reply_anchor = self._reply_anchor_for_event(event)
            thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
            if self._queue_during_drain_enabled():
                self._queue_or_replace_pending_event(session_key, event)
                message = f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
            else:
                message = f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."

            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
            return True

        # --- Approval response routing (#46866) ---
        # When the agent is blocked waiting for a dangerous-command approval,
        # plain-text responses like "yes" or "approve" must be routed to the
        # approval handler instead of being steered/queued/interrupted.
        # Otherwise approval via messaging platforms never succeeds — the
        # reply is queued behind a turn that can't start until the approval
        # resolves, so the approval times out and auto-denies (a deadlock).
        #
        # Slash forms (/approve, /deny) already bypass to the runner at the
        # base-adapter guard.  This handles the bare-word forms (Signal/SMS
        # users naturally type "yes" rather than "/approve").  Gating on
        # has_blocking_approval(session_key) is the disambiguator that keeps
        # a conversational "yes" from triggering a dangerous command when no
        # approval is actually pending (design intent — see run.py "Pending
        # exec approvals are handled by /approve and /deny" note).
        #
        # We reuse the canonical /approve and /deny handlers rather than
        # re-deriving the resolution + i18n messaging: they resolve the
        # waiting thread, resume typing, AND return a localized confirmation
        # string.  The busy-handler path does not auto-send that return, so
        # we deliver it ourselves (mirroring the draining-case send above).
        try:
            from tools.approval import has_blocking_approval
            if has_blocking_approval(session_key):
                _raw_text = (event.text or "").strip().lower()
                _approve_words = {"approve", "yes", "ok", "okay", "confirm", "y", "👍"}
                _deny_words = {"deny", "no", "reject", "cancel", "n", "👎"}
                _approval_handler = None
                _normalized_args = ""
                if _raw_text in _approve_words:
                    _approval_handler = self._handle_approve_command
                elif _raw_text in _deny_words:
                    _approval_handler = self._handle_deny_command
                elif _raw_text in {"always", "approve always", "always approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "always"
                elif _raw_text in {"session", "approve session", "session approve"}:
                    _approval_handler = self._handle_approve_command
                    _normalized_args = "session"
                if _approval_handler is not None:
                    # Synthesize the canonical "/approve [args]" / "/deny"
                    # command text so the slash handlers parse modifiers via
                    # event.get_command_args().  Always use a literal "/" —
                    # MessageEvent.is_command()/get_command_args() only
                    # recognize the "/" prefix, not the per-platform display
                    # prefix ("!" on Slack/Matrix).
                    _verb = "approve" if _approval_handler is self._handle_approve_command else "deny"
                    _synth = f"/{_verb}"
                    if _normalized_args:
                        _synth = f"{_synth} {_normalized_args}"
                    event.text = _synth
                    _reply = await _approval_handler(event)
                    logger.info(
                        "Approval response via plain text: session=%s verb=%s args=%r",
                        session_key, _verb, _normalized_args,
                    )
                    _adapter = self.adapters.get(event.source.platform)
                    if _adapter and _reply:
                        _text, _eph_ttl = _adapter._unwrap_ephemeral(_reply)
                        if _text:
                            _anchor = self._reply_anchor_for_event(event)
                            await _adapter._send_with_retry(
                                chat_id=event.source.chat_id,
                                content=_text,
                                reply_to=_anchor,
                                metadata=self._thread_metadata_for_source(event.source, _anchor),
                            )
                    return True
        except Exception:
            logger.warning(
                "Plain-text approval routing failed for session %s; "
                "falling through to busy handling",
                session_key, exc_info=True,
            )

        # Normal busy case (agent actively running a task)
        adapter = self._adapter_for_source(event.source)
        if not adapter:
            return False  # let default path handle it

        # --- Internal synthetic events must never interrupt/steer ---
        # Async-delegation completions (delegate_task(background=true)) and
        # background-process completions (terminal notify_on_complete) re-enter
        # the originating session as internal MessageEvents. When the session
        # is busy, treating them like a user TEXT message means interrupt-mode
        # (the default busy_text_mode) aborts the active turn AND sends a "⚡
        # Interrupting current task" ack — exactly the opposite of the design
        # invariant that a completion surfaces as a NEW turn only when idle and
        # never splices into a running turn. Fall through to the base adapter,
        # which queues internal events silently (no interrupt, no ack) so they
        # cascade after the current turn finishes.
        if getattr(event, "internal", False):
            return False

        running_agent = self._running_agents.get(session_key)

        effective_mode = self._busy_input_mode
        busy_text_mode = getattr(self, "_busy_text_mode", "interrupt")
        if (
            event.message_type == MessageType.TEXT
            and busy_text_mode == "queue"
            and effective_mode != "steer"
        ):
            return False

        # Steer mode: inject mid-run via running_agent.steer() instead of
        # queueing + interrupting.  If the agent isn't running yet
        # (sentinel) or lacks steer(), or the payload is empty, fall back
        # to queue semantics so nothing is lost.
        # #30170 — Subagent protection. ``AIAgent.interrupt()`` cascades
        # to every entry in the parent's ``_active_children`` list and
        # aborts in-flight ``delegate_task`` work. Demote ``interrupt``
        # to ``queue`` when the parent is currently driving subagents so
        # a conversational follow-up doesn't destroy minutes of subagent
        # work. Explicit ``/stop`` and ``/new`` slash commands go through
        # ``_interrupt_and_clear_session`` and are unaffected — the
        # operator still has a way to force-cancel everything.
        demoted_for_subagents = (
            effective_mode == "interrupt"
            and self._agent_has_active_subagents(running_agent)
        )
        if demoted_for_subagents:
            logger.info(
                "Demoting busy_input_mode 'interrupt' to 'queue' for session %s "
                "because the running agent has active subagents (#30170)",
                session_key,
            )
            effective_mode = "queue"
        steered = False
        if effective_mode == "steer":
            steer_text = (event.text or "").strip()
            can_steer = (
                steer_text
                and running_agent is not None
                and running_agent is not _AGENT_PENDING_SENTINEL
                and hasattr(running_agent, "steer")
            )
            if can_steer:
                try:
                    steered = bool(running_agent.steer(steer_text))
                except Exception as exc:
                    logger.warning("Gateway steer failed for session %s: %s", session_key, exc)
                    steered = False
            if not steered:
                # Fall back to queue (merge into pending messages, no interrupt)
                effective_mode = "queue"

        # Store the message so it's processed as the next turn after the
        # current run finishes (or is interrupted).  Skip this for a
        # successful steer — the text already landed inside the run and
        # must NOT also be replayed as a next-turn user message.
        #
        # Route through _queue_or_replace_pending_event (the same FIFO
        # infrastructure used by busy queue-mode and /queue) rather than a
        # raw merge_pending_message_event(merge_text=True). The raw merge
        # newline-joins consecutive TEXT follow-ups into a SINGLE pending
        # turn, destroying message boundaries — so two separate user
        # messages sent while the agent was busy (interrupt mode, or a
        # steer that fell back to queue) arrived as one mashed-together
        # turn (#43066 sub-bug 2). The FIFO path gives each text its own
        # turn in arrival order while still preserving photo-burst / album
        # merge semantics for media.
        if not steered:
            self._queue_or_replace_pending_event(session_key, event)

        is_queue_mode = effective_mode == "queue"
        is_steer_mode = effective_mode == "steer"

        # If not in queue/steer mode, interrupt the running agent immediately.
        # This aborts in-flight tool calls and causes the agent loop to exit
        # at the next check point.
        if effective_mode == "interrupt" and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                running_agent.interrupt(event.text)
            except Exception:
                pass  # don't let interrupt failure block the ack

        # Check if busy ack is disabled — skip sending but still process the input.
        # Placed before debounce so we don't stamp a "last ack" timestamp that was
        # never actually delivered.
        busy_ack_enabled = os.environ.get("HERMES_GATEWAY_BUSY_ACK_ENABLED", "true").lower() == "true"
        if not busy_ack_enabled:
            logger.debug("Busy ack suppressed for session %s", session_key)
            return True  # input still processed, just no ack sent

        # Debounce: only send an acknowledgment once every 30 seconds per session
        # to avoid spamming the user when they send multiple messages quickly
        _BUSY_ACK_COOLDOWN = 30
        now = time.time()
        last_ack = self._busy_ack_ts.get(session_key, 0)
        if now - last_ack < _BUSY_ACK_COOLDOWN:
            return True  # interrupt sent (if not queue), ack already delivered recently

        self._busy_ack_ts[session_key] = now

        # Build a status-rich acknowledgment. Mobile chat defaults keep this
        # terse; detailed iteration/tool state is still available in logs and
        # can be opted in per platform via display.platforms.<platform>.busy_ack_detail.
        from gateway.display_config import resolve_display_setting
        status_parts = []
        busy_ack_detail_enabled = bool(
            resolve_display_setting(
                _load_gateway_config(),
                _platform_config_key(event.source.platform),
                "busy_ack_detail",
                True,
            )
        )

        if busy_ack_detail_enabled and running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            try:
                summary = running_agent.get_activity_summary()
                iteration = summary.get("api_call_count", 0)
                max_iter = summary.get("max_iterations", 0)
                current_tool = summary.get("current_tool")
                start_ts = self._running_agents_ts.get(session_key, 0)
                if start_ts:
                    elapsed_min = int((now - start_ts) / 60)
                    if elapsed_min > 0:
                        status_parts.append(f"{elapsed_min} min elapsed")
                if max_iter:
                    status_parts.append(f"iteration {iteration}/{max_iter}")
                if current_tool:
                    status_parts.append(f"running: {current_tool}")
            except Exception:
                pass

        status_detail = f" ({', '.join(status_parts)})" if status_parts else ""
        if is_steer_mode:
            message = (
                f"⏩ Steered into current run{status_detail}. "
                f"Your message arrives after the next tool call."
            )
        elif is_queue_mode and demoted_for_subagents:
            # #30170 — explain the demotion so the user knows their
            # follow-up didn't accidentally kill the subagent and
            # discovers `/stop` as the explicit escape hatch.
            message = (
                f"⏳ Subagent working{status_detail} — your message is queued for "
                f"when it finishes (use /stop to cancel everything)."
            )
        elif is_queue_mode:
            message = (
                f"⏳ Queued for the next turn{status_detail}. "
                f"I'll respond once the current task finishes."
            )
        else:
            message = (
                f"⚡ Interrupting current task{status_detail}. "
                f"I'll respond to your message shortly."
            )

        # First-touch onboarding: the very first time a user sends a message
        # while the agent is busy, append a one-time hint explaining the
        # queue/interrupt knob.  Flag is persisted to config.yaml so it never
        # fires again on this install.
        try:
            from agent.onboarding import (
                BUSY_INPUT_FLAG,
                busy_input_hint_gateway,
                is_seen,
                mark_seen,
            )
            _user_cfg = _load_gateway_config()
            if not is_seen(_user_cfg, BUSY_INPUT_FLAG):
                if is_steer_mode:
                    _hint_mode = "steer"
                elif is_queue_mode:
                    _hint_mode = "queue"
                else:
                    _hint_mode = "interrupt"
                message = (
                    f"{message}\n\n"
                    f"{busy_input_hint_gateway(_hint_mode)}"
                )
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
        except Exception as _onb_err:
            logger.debug("Failed to apply busy-input onboarding hint: %s", _onb_err)

        reply_anchor = self._reply_anchor_for_event(event)
        thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
        try:
            await adapter._send_with_retry(
                chat_id=event.source.chat_id,
                content=message,
                reply_to=(
                    reply_anchor
                    if event.source.platform == Platform.TELEGRAM
                    and event.source.chat_type == "dm"
                    and event.source.thread_id
                    else (None if event.source.platform == Platform.TELEGRAM and event.source.thread_id else event.message_id)
                ),
                metadata=thread_meta,
            )
        except Exception as e:
            logger.debug("Failed to send busy-ack: %s", e)

        return True

    async def _drain_active_agents(self, timeout: float) -> tuple[Dict[str, Any], bool]:
        snapshot = self._snapshot_running_agents()
        last_active_count = self._running_agent_count()
        last_status_at = 0.0

        def _maybe_update_status(force: bool = False) -> None:
            nonlocal last_active_count, last_status_at
            now = asyncio.get_running_loop().time()
            active_count = self._running_agent_count()
            if force or active_count != last_active_count or (now - last_status_at) >= 1.0:
                self._update_runtime_status("draining")
                last_active_count = active_count
                last_status_at = now

        if not self._running_agents:
            _maybe_update_status(force=True)
            return snapshot, False

        _maybe_update_status(force=True)
        if timeout <= 0:
            return snapshot, True

        deadline = asyncio.get_running_loop().time() + timeout
        while self._running_agents and asyncio.get_running_loop().time() < deadline:
            _maybe_update_status()
            await asyncio.sleep(0.1)
        timed_out = bool(self._running_agents)
        _maybe_update_status(force=True)
        return snapshot, timed_out

    def _interrupt_running_agents(self, reason: str) -> None:
        for session_key, agent in list(self._running_agents.items()):
            if agent is _AGENT_PENDING_SENTINEL:
                continue
            try:
                agent.interrupt(reason)
                logger.debug("Interrupted running agent for session %s during shutdown", session_key)
            except Exception as e:
                logger.debug("Failed interrupting agent during shutdown: %s", e)

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the very start of stop() — adapters are still connected so
        messages can be delivered. Best-effort: individual send failures are
        logged and swallowed so they never block the shutdown sequence.
        """
        active = self._snapshot_running_agents()
        restart_source = self._restart_command_source if self._restart_requested else None

        action = "restarting" if self._restart_requested else "shutting down"
        hint = (
            "Your current task will be interrupted. "
            "Send any message after restart and I'll try to resume where you left off."
            if self._restart_requested
            else "Your current task will be interrupted."
        )
        msg = f"⚠️ Gateway {action} — {hint}"

        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in active:
            source = None
            try:
                if getattr(self, "session_store", None) is not None:
                    self.session_store._ensure_loaded()
                    entry = self.session_store._entries.get(session_key)
                    source = getattr(entry, "origin", None) if entry else None
            except Exception as e:
                logger.debug(
                    "Failed to load session origin for shutdown notification %s: %s",
                    session_key,
                    e,
                )

            if source is None:
                source = self._get_cached_session_source(session_key)

            if source is not None:
                platform_str = source.platform.value
                chat_id = str(source.chat_id)
                thread_id = source.thread_id
            else:
                # Fall back to parsing the session key when no persisted
                # origin is available (legacy sessions/tests).
                _parsed = _parse_session_key(session_key)
                if not _parsed:
                    continue
                platform_str = _parsed["platform"]
                chat_id = _parsed["chat_id"]
                thread_id = _parsed.get("thread_id")

            # Deduplicate only identical delivery targets. Thread/topic-aware
            # platforms can share a parent chat while still routing to distinct
            # destinations via metadata.
            dedup_key = (platform_str, chat_id, str(thread_id) if thread_id else None)
            if dedup_key in notified:
                continue

            try:
                platform = Platform(platform_str)
                adapter = self.adapters.get(platform)
                if not adapter:
                    continue

                platform_cfg = self.config.platforms.get(platform)
                if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                    logger.info(
                        "Shutdown notification suppressed for active session: %s has gateway_restart_notification=false",
                        platform_str,
                    )
                    continue

                reply_to_message_id = getattr(source, "message_id", None) if source is not None else None
                if reply_to_message_id is None and restart_source is not None:
                    try:
                        restart_platform = restart_source.platform.value
                        restart_chat_id = str(restart_source.chat_id)
                        restart_thread_id = str(restart_source.thread_id) if restart_source.thread_id else None
                        if (restart_platform, restart_chat_id, restart_thread_id) == dedup_key:
                            reply_to_message_id = getattr(restart_source, "message_id", None)
                    except Exception:
                        pass

                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=getattr(source, "chat_type", None) if source is not None else None,
                    reply_to_message_id=reply_to_message_id,
                    adapter=adapter,
                )

                result = await adapter.send(chat_id, msg, metadata=metadata)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to %s:%s: %s",
                        platform_str,
                        chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to active chat %s:%s",
                    platform_str, chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to %s:%s: %s",
                    platform_str, chat_id, e,
                )

        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return

        # Suppress ONLY the home-channel broadcast when the drain that is ending
        # in this shutdown asked us to be quiet (e.g. a NAS auto-update image
        # migration — drain-gated, then the machine is recreated). On the
        # always-on Hermes Cloud fleet that broadcast would otherwise fire on
        # every routine auto-update, spamming home channels with operator-
        # flavoured "gateway shutting down" pings the user doesn't care about.
        # The per-active-session interrupt pings above are deliberately NOT
        # gated: on a drained shutdown they're empty by construction, and in the
        # force-interrupt (deadline-exceeded) case they carry the genuinely
        # useful "your task was cut off, message me to resume" hint. The flag is
        # only honoured for a CURRENT-epoch marker (drain_notification_suppressed
        # reuses the NS-570 staleness check), so an orphaned marker can never
        # silence a fresh gateway's legitimate broadcast.
        try:
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                logger.info(
                    "Home-channel shutdown broadcast suppressed by drain marker "
                    "(suppress_notification=true)"
                )
                return
        except Exception as e:
            # Never let the suppression check block the shutdown broadcast —
            # fail toward the louder, more-visible behaviour.
            logger.debug("drain_notification_suppressed check failed: %s", e)

        # Snapshot adapters up front: adapter.send() can hit a fatal error
        # path that pops the adapter from self.adapters (see _handle_fatal
        # elsewhere), which would otherwise trigger
        # ``RuntimeError: dictionary changed size during iteration`` —
        # observed in a user report during gateway shutdown.
        for platform, adapter in list(self.adapters.items()):
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Shutdown notification suppressed for home channel: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            dedup_key = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if dedup_key in notified:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(str(home.chat_id), msg, metadata=metadata)
                else:
                    result = await adapter.send(str(home.chat_id), msg)
                if result is not None and getattr(result, "success", True) is False:
                    logger.debug(
                        "Failed to send shutdown notification to home channel %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                notified.add(dedup_key)
                logger.info(
                    "Sent shutdown notification to home channel %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    e,
                )

    async def _finalize_shutdown_agents(self, active_agents: Dict[str, Any]) -> None:
        for agent in active_agents.values():
            # Persist any in-flight transcript to the SQLite session store
            # before teardown (#13121).  An agent forcibly interrupted by the
            # drain-timeout escalation may never reach
            # ``turn_finalizer.finalize_turn`` (the only place that flushes the
            # turn to state.db) — e.g. it was blocked in a tool call that did
            # not abort within the post-interrupt grace window.  Its in-flight
            # tool rounds live only in the in-memory ``_session_messages``
            # (refreshed per tool round in ``conversation_loop`` but never
            # written to SQLite mid-turn), so the immediate pre-restart turn is
            # silently dropped from ``load_transcript()`` on resume.  Flushing
            # here closes that gap; the resume_pending / fresh-tool-tail
            # branches in ``_handle_message_with_agent`` already expect a
            # transcript whose tail may be a pending tool result.  The flush is
            # idempotent (identity-tracked in ``_flush_messages_to_session_db``),
            # so agents that DID finish gracefully re-flush nothing.
            try:
                _flush = getattr(agent, "_flush_messages_to_session_db", None)
                _session_messages = getattr(agent, "_session_messages", None)
                if callable(_flush) and isinstance(_session_messages, list) and _session_messages:
                    # Strip private empty-response retry scaffolding from the
                    # tail first, mirroring the graceful ``_persist_session``
                    # path, so a resumed turn doesn't replay synthetic recovery
                    # nudges.
                    _strip = getattr(
                        agent, "_drop_trailing_empty_response_scaffolding", None
                    )
                    if callable(_strip):
                        try:
                            _strip(_session_messages)
                        except Exception:
                            pass
                    _flush(_session_messages)
            except Exception as _e:
                logger.debug("Shutdown transcript flush failed: %s", _e)
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "on_session_finalize",
                    session_id=getattr(agent, "session_id", None),
                    platform="gateway",
                    reason="shutdown",
                )
            except Exception:
                pass
            # Off-loop + bounded: a wedged memory provider here used to hang
            # the whole shutdown so SIGTERM never completed (#53175).
            await self._cleanup_agent_resources_off_loop(
                agent, context="shutdown finalize"
            )

    def _should_emit_long_running_notification(
        self,
        session_key: Optional[str],
        agent: Any,
        executor_task: Optional[Any],
    ) -> bool:
        """Only emit the heartbeat while this task still owns the live run.

        Guards against a stale ``running: delegate_task`` heartbeat outliving the
        run that started it: stop once the executor finishes, the agent is gone,
        or the session key has been rebound to a different live agent (e.g. the
        user sent ``/new`` and a fresh agent took the slot mid-run, #12029).
        """
        if agent is None:
            return False
        if executor_task is not None and executor_task.done():
            return False
        if session_key and self._running_agents.get(session_key) is not agent:
            return False
        return True

    # Upper bound on off-loop agent-resource cleanup invoked from coroutines
    # running on the gateway's event loop (session-expiry sweep, in-turn
    # cache-hygiene re-eviction). _cleanup_agent_resources is synchronous and
    # can block for a long time (agent.close() does subprocess teardown;
    # shutdown_memory_provider() may do network/SQLite IO via a memory plugin).
    # Calling it inline wedges the whole loop — the bot goes silent, the
    # runtime-status updated_at heartbeat freezes, and SIGTERM cannot be
    # serviced (#53175). Offload to a worker thread under this timeout so the
    # loop is never blocked; mirrors the /new reset path's fix (#35994).
    _CLEANUP_TIMEOUT_S = 30.0

    async def _cleanup_agent_resources_off_loop(
        self, agent: Any, *, context: str = ""
    ) -> None:
        """Run _cleanup_agent_resources in a worker thread with a bounded wait.

        Safe to await from coroutines on the gateway event loop: a slow or
        wedged teardown (memory provider IO, subprocess close) can no longer
        block message processing. On timeout the await is cancelled and the
        worker thread is left to finish (or leak) on its own — the caller
        proceeds regardless, exactly as the /new reset path does (#35994).
        """
        if agent is None:
            return
        if context.startswith("shutdown"):
            try:
                agent._end_session_on_close = False
            except Exception:
                pass
        try:
            await asyncio.wait_for(
                self._run_in_executor_with_context(
                    self._cleanup_agent_resources, agent
                ),
                timeout=self._CLEANUP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent resource cleanup%s exceeded %ss; proceeding without "
                "blocking the event loop (the worker thread is left to finish "
                "on its own). (#53175)",
                f" ({context})" if context else "",
                self._CLEANUP_TIMEOUT_S,
            )
        except Exception as cleanup_exc:
            logger.warning(
                "Agent resource cleanup%s failed: %s (#53175)",
                f" ({context})" if context else "",
                cleanup_exc,
            )

    def _cleanup_agent_resources(self, agent: Any) -> None:
        """Best-effort cleanup for temporary or cached agent instances."""
        if agent is None:
            return
        try:
            if hasattr(agent, "shutdown_memory_provider"):
                # Pass the agent's own conversation transcript so memory
                # providers' ``on_session_end`` hooks see the real messages
                # instead of the empty default (#15165). ``_session_messages``
                # is set on ``AIAgent`` (run_agent.py:1518) and refreshed at
                # the end of every ``run_conversation`` turn via
                # ``_persist_session``; on an agent built through
                # ``object.__new__`` (test stubs) the attribute may be
                # absent, so ``getattr`` with a ``None`` default keeps the
                # call signature-compatible with the pre-fix behaviour
                # (``shutdown_memory_provider(messages=None)``).
                session_messages = getattr(agent, "_session_messages", None)
                if isinstance(session_messages, list):
                    agent.shutdown_memory_provider(session_messages)
                else:
                    agent.shutdown_memory_provider()
        except Exception:
            pass
        # Close tool resources (terminal sandboxes, browser daemons,
        # background processes, httpx clients) to prevent zombie
        # process accumulation.
        try:
            if hasattr(agent, "close"):
                agent.close()
        except Exception:
            pass
        # Auxiliary async clients (session_search/web/vision/etc.) live in a
        # process-global cache and are created inside worker threads. Clean up
        # any entries whose event loop is now dead so their httpx transports do
        # not accumulate across gateway turns.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception:
            pass

    _STUCK_LOOP_THRESHOLD = 3  # restarts while active before auto-suspend
    _STUCK_LOOP_FILE = ".restart_failure_counts"

    def _increment_restart_failure_counts(self, active_session_keys: set) -> None:
        """Increment restart-failure counters for sessions active at shutdown.

        Persists to a JSON file so counters survive across restarts.
        Sessions NOT in active_session_keys are removed (they completed
        successfully, so the loop is broken).
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        try:
            counts = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            counts = {}

        # Increment active sessions, remove inactive ones (loop broken)
        new_counts = {}
        for key in active_session_keys:
            new_counts[key] = counts.get(key, 0) + 1
        # Keep any entries that are still above 0 even if not active now
        # (they might become active again next restart)

        try:
            atomic_json_write(path, new_counts, indent=None)
        except Exception:
            pass

    def _suspend_stuck_loop_sessions(self) -> int:
        """Suspend sessions that have been active across too many restarts.

        Returns the number of sessions suspended.  Called on gateway startup
        AFTER suspend_recently_active() to catch the stuck-loop pattern:
        session loads → agent gets stuck → gateway restarts → repeat.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return 0

        try:
            counts = json.loads(path.read_text())
        except Exception:
            return 0

        suspended = 0
        stuck_keys = [k for k, v in counts.items() if v >= self._STUCK_LOOP_THRESHOLD]

        for session_key in stuck_keys:
            try:
                entry = self.session_store._entries.get(session_key)
                if entry and not entry.suspended:
                    entry.suspended = True
                    suspended += 1
                    logger.warning(
                        "Auto-suspended stuck session %s (active across %d "
                        "consecutive restarts — likely a stuck loop)",
                        session_key, counts[session_key],
                    )
            except Exception:
                pass

        if suspended:
            try:
                self.session_store._save()
            except Exception:
                pass

        # Clear the file — counters start fresh after suspension
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

        return suspended

    def _clear_restart_failure_count(self, session_key: str) -> None:
        """Clear the restart-failure counter for a session that completed OK.

        Called after a successful agent turn to signal the loop is broken.
        """
        import json

        path = _hermes_home / self._STUCK_LOOP_FILE
        if not path.exists():
            return
        try:
            counts = json.loads(path.read_text())
            if session_key in counts:
                del counts[session_key]
                if counts:
                    atomic_json_write(path, counts, indent=None)
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass

    async def _launch_detached_restart_command(self) -> None:
        import shutil
        import subprocess

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            logger.error("Could not locate hermes binary for detached /restart")
            return
        if self._detached_restart_helper_started:
            return
        self._detached_restart_helper_started = True

        current_pid = os.getpid()
        restart_after_s = max(float(getattr(self, "_restart_drain_timeout", 0.0) or 0.0) + 5.0, 5.0)

        # On Windows there's no bash/setsid chain — spawn a tiny Python
        # watcher directly via sys.executable instead.  The watcher polls
        # current_pid, waits for our exit, then runs `hermes gateway
        # restart` with detach flags so the respawn survives the CLI
        # that triggered the /restart command closing its console.
        if sys.platform == "win32":
            import textwrap
            from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

            cmd_argv = [*hermes_cmd, "gateway", "restart"]
            watcher = textwrap.dedent(
                """
                import os, subprocess, sys, time
                from hermes_cli._subprocess_compat import windows_detach_flags_without_breakaway
                pid = int(sys.argv[1])
                restart_after_s = float(sys.argv[2])
                cmd = sys.argv[3:]
                deadline = time.monotonic() + restart_after_s

                def _alive(p):
                    # On Windows, os.kill(pid, 0) is NOT a no-op — it maps to
                    # GenerateConsoleCtrlEvent(0, pid) (bpo-14484). Use the
                    # Win32 handle-based existence check instead.
                    if os.name == 'nt':
                        import ctypes
                        k32 = ctypes.windll.kernel32
                        k32.OpenProcess.restype = ctypes.c_void_p
                        k32.WaitForSingleObject.restype = ctypes.c_uint
                        k32.GetLastError.restype = ctypes.c_uint
                        h = k32.OpenProcess(0x1000 | 0x100000, False, int(p))
                        if not h:
                            return k32.GetLastError() != 87
                        try:
                            return k32.WaitForSingleObject(h, 0) == 0x102
                        finally:
                            k32.CloseHandle(h)
                    try:
                        os.kill(int(p), 0)
                        return True
                    except ProcessLookupError:
                        return False
                    except PermissionError:
                        return True
                    except OSError:
                        return False

                while time.monotonic() < deadline:
                    if not _alive(pid):
                        break
                    time.sleep(0.2)
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=windows_detach_flags_without_breakaway(),
                )
                """
            ).strip()
            watcher_env = os.environ.copy()
            # This watcher is intentionally outside the running gateway. If it
            # inherits the gateway marker, `hermes gateway restart` refuses to
            # run as a self-restart loop guard and the gateway stays stopped.
            watcher_env.pop("_HERMES_GATEWAY", None)
            project_root = Path(__file__).resolve().parent.parent
            watcher_python = sys.executable
            try:
                # Prefer a real GUI-subsystem interpreter for the watcher
                # itself.  With uv venvs, ``python.exe`` can re-exec the base
                # console interpreter and flash even when the Popen carries
                # CREATE_NO_WINDOW; pythonw.exe avoids console allocation.
                from hermes_cli.gateway_windows import _resolve_detached_python

                watcher_python, _watcher_venv_dir, _watcher_site_packages = (
                    _resolve_detached_python(sys.executable)
                )
            except Exception:
                watcher_python = sys.executable
            venv_dir = Path(watcher_env.get("VIRTUAL_ENV") or project_root / "venv")
            site_packages = venv_dir / "Lib" / "site-packages"
            if site_packages.exists():
                watcher_env["VIRTUAL_ENV"] = str(venv_dir)
                pythonpath = [str(project_root), str(site_packages)]
                if watcher_env.get("PYTHONPATH"):
                    pythonpath.append(watcher_env["PYTHONPATH"])
                watcher_env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
            subprocess.Popen(
                [watcher_python, "-c", watcher, str(current_pid), str(restart_after_s), *cmd_argv],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                **windows_detach_popen_kwargs(),
            )
            return

        cmd = " ".join(shlex.quote(part) for part in hermes_cmd)
        shell_cmd = (
            f"deadline=$(( $(date +%s) + {int(restart_after_s)} )); "
            f"while kill -0 {current_pid} 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 0.2; done; "
            f"{cmd} gateway restart"
        )
        # Same marker scrub as the Windows watcher above: this watcher runs
        # `hermes gateway restart` from outside the gateway, but it inherits
        # _HERMES_GATEWAY=1 from us, and the CLI's self-restart loop guard
        # refuses to run when that marker is set — silently (DEVNULL), so the
        # gateway stops and never comes back.
        watcher_env = os.environ.copy()
        watcher_env.pop("_HERMES_GATEWAY", None)
        setsid_bin = shutil.which("setsid")
        if setsid_bin:
            subprocess.Popen(
                [setsid_bin, "bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )
        else:
            subprocess.Popen(
                ["bash", "-lc", shell_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=watcher_env,
                start_new_session=True,
            )

    def _launch_systemd_restart_shortcut(self) -> None:
        """Best-effort helper to bypass systemd's automatic restart delay.

        For planned in-chat restarts, the gateway exits cleanly so systemd does
        not record a failure.  However, units with RestartSteps still count
        automatic restarts and can delay repeated /restart tests.  A transient
        user service survives our cgroup teardown and explicitly starts the
        gateway as soon as this PID exits, while the unit keeps its normal
        backoff for real crash loops.
        """
        if sys.platform != "linux" or not os.environ.get("INVOCATION_ID"):
            return

        try:
            import shutil
            import subprocess

            systemd_run = shutil.which("systemd-run")
            systemctl = shutil.which("systemctl")
            if not systemd_run or not systemctl:
                return

            try:
                from hermes_cli.gateway import get_service_name

                service_name = get_service_name()
            except Exception:
                service_name = "hermes-gateway"

            current_pid = os.getpid()

            # Detect whether the gateway unit is registered as a system or
            # user service.  Daemon-style deployments are typically system
            # units (e.g. /etc/systemd/system/hermes-gateway.service), while
            # `hermes setup` under a non-root account may register a user
            # unit.  Hard-coding ``--user`` broke system-unit deployments:
            # systemctl returned an empty MainPID, the PID-equality check
            # below failed, and the planned-restart helper was never
            # launched — leaving the gateway dead until a manual reboot.
            def _query_pid(scope_flags):
                try:
                    out = subprocess.run(
                        [systemctl, *scope_flags, "show", service_name,
                         "--property=MainPID", "--value"],
                        capture_output=True, text=True, timeout=2,
                    )
                    return (out.stdout or "").strip()
                except Exception:
                    return ""

            system_pid = _query_pid([])
            user_pid = _query_pid(["--user"])
            if str(current_pid) == system_pid:
                scope_flags = []
                systemctl_scope = "systemctl"
            elif str(current_pid) == user_pid:
                scope_flags = ["--user"]
                systemctl_scope = "systemctl --user"
            else:
                # MainPID does not match in either scope — likely invoked
                # outside of systemd or the unit was renamed.  Bail out
                # rather than restart the wrong unit.
                return

            service_arg = shlex.quote(service_name)
            shell_cmd = (
                f"while kill -0 {current_pid} 2>/dev/null; do sleep 0.2; done; "
                f"{systemctl_scope} reset-failed {service_arg}; "
                f"{systemctl_scope} restart {service_arg}"
            )
            unit_name = f"{service_name}-planned-restart-{current_pid}".replace(".", "-")
            subprocess.Popen(
                [
                    systemd_run,
                    *scope_flags,
                    "--collect",
                    "--unit",
                    unit_name,
                    "/bin/sh",
                    "-lc",
                    shell_cmd,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info(
                "Launched systemd planned-restart helper for %s (pid=%s, scope=%s)",
                service_name,
                current_pid,
                "user" if scope_flags else "system",
            )
        except Exception as e:
            logger.debug("Failed to launch systemd planned-restart helper: %s", e)

    def request_restart(self, *, detached: bool = False, via_service: bool = False) -> bool:
        if self._restart_task_started:
            return False
        self._restart_requested = True
        self._restart_detached = detached
        self._restart_via_service = via_service
        self._restart_task_started = True

        async def _run_restart() -> None:
            if detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart helper: %s", e)
            await asyncio.sleep(0.05)
            await self.stop(restart=True, detached_restart=detached, service_restart=via_service)

        # _run_restart is a short-lived self-terminating task (calls stop()
        # then returns).  Don't add it to _background_tasks — _stop_impl
        # cancels all entries in that set, which would cancel _run_restart
        # while it's awaiting _stop_task, propagating CancelledError into
        # _stop_impl and preventing _shutdown_event.set() / _exit_code = 75.
        # See #12875.
        #
        # We still hold a strong reference in self._restart_task: a bare
        # asyncio.create_task() keeps only a weak reference, so the event
        # loop may garbage-collect a still-pending task mid-flight.  The
        # cancel loop in _stop_impl explicitly skips _restart_task for the
        # same reason it skips _stop_task.
        self._restart_task = asyncio.create_task(_run_restart())
        return True

    # Drain-timeout reasons set by _stop_impl() when a still-running turn is
    # force-interrupted; "restart_interrupted" is set by
    # SessionStore.suspend_recently_active() on crash recovery (no
    # .clean_shutdown marker).  All three mean "the agent was mid-turn and
    # we killed it" — eligible for startup auto-resume.
    _AUTO_RESUME_REASONS = frozenset(
        {"restart_timeout", "shutdown_timeout", "restart_interrupted"}
    )

    async def _run_startup_resume_event(
        self,
        adapter: BasePlatformAdapter,
        event: MessageEvent,
        session_key: str,
    ) -> None:
        """Dispatch one synthetic startup resume and wait for its agent turn.

        ``BasePlatformAdapter.handle_message()`` returns after it installs the
        adapter-level guard and spawns the background processing task.  Startup
        restore needs a stronger boundary: inbound messages must stay queued
        until the resumed agent turn itself has finished, otherwise a user
        message can race the restore turn immediately after ``handle_message``
        returns.
        """
        try:
            await adapter.handle_message(event)
            session_tasks = getattr(adapter, "_session_tasks", {})
            task = session_tasks.get(session_key) if isinstance(session_tasks, dict) else None
            if task is not None:
                await asyncio.shield(task)
        finally:
            # _schedule_resume_pending_sessions pre-claims the runner slot
            # before spawning this task.  If adapter.handle_message raises
            # before _handle_message takes ownership, release that pre-claim;
            # otherwise the real run's normal cleanup owns the slot.
            if self._running_agents.get(session_key) is _AGENT_PENDING_SENTINEL:
                self._release_running_agent_state(session_key)

    def _queue_startup_restore_event(self, event: MessageEvent) -> None:
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            queue = []
            self._startup_restore_queue = queue
        queue.append(event)
        try:
            source = event.source
            logger.info(
                "Queued inbound message during gateway startup restore: platform=%s chat=%s",
                source.platform.value if source and source.platform else "unknown",
                source.chat_id if source else "unknown",
            )
        except Exception:
            pass

    async def _drain_startup_restore_queue(self) -> int:
        """Replay inbound messages queued while startup auto-resume ran."""
        drained = 0
        queue = getattr(self, "_startup_restore_queue", None)
        if queue is None:
            return 0
        while queue:
            event = queue.pop(0)
            source = getattr(event, "source", None)
            adapter = self.adapters.get(source.platform) if source is not None else None
            if adapter is None:
                logger.debug(
                    "Dropping startup-restore queued message: adapter unavailable for %s",
                    getattr(getattr(source, "platform", None), "value", None),
                )
                continue
            # Mark this replay so _handle_message does not queue it again while
            # the restore gate remains closed for any fresh inbound arrivals.
            try:
                setattr(event, "_hermes_startup_restore_replay", True)
            except Exception:
                pass
            await adapter.handle_message(event)
            drained += 1
        return drained

    async def _finish_startup_restore(self) -> None:
        """Wait for startup auto-resume, then release and drain inbound queue."""
        tasks = list(getattr(self, "_startup_restore_tasks", []) or [])
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.debug(
                        "startup auto-resume task failed",
                        exc_info=(type(result), result, result.__traceback__),
                    )
        self._startup_restore_tasks = []
        drained = await self._drain_startup_restore_queue()
        self._startup_restore_in_progress = False
        if drained:
            logger.info("Drained %d inbound message(s) queued during startup restore", drained)

    def _schedule_resume_pending_sessions(self, platform=None) -> int:
        """Auto-continue fresh restart-interrupted sessions after startup.

        ``resume_pending`` already preserves the transcript AND the existing
        ``_is_resume_pending`` branch in ``_handle_message_with_agent``
        injects a reason-aware recovery system note on the next turn.  This
        method closes the UX gap by synthesizing that next turn once
        adapters are back online — the event text is empty so the existing
        injection path owns the wording and we never double up.

        Adapters that are not yet ready (adapter missing from
        ``self.adapters``) are skipped silently; their sessions stay
        ``resume_pending`` and will auto-resume on the next real user
        message, or when the platform reconnects — the reconnect watcher
        calls this again scoped to that ``platform``.

        ``platform`` (a ``Platform``) restricts the pass to sessions that
        originated on that platform.  The reconnect path passes it so a
        platform coming back online retries only its own sessions and never
        re-touches another platform's in-flight recoveries.  Sessions whose
        agent is already running are skipped regardless, so a session
        scheduled at startup is never resumed a second time.
        """
        window = _auto_continue_freshness_window()
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                    and (platform is None or entry.origin.platform == platform)
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return 0

        # Defense-3 (#30719): break the SIGTERM-respawn loop. Only count this
        # boot when there are restart-interrupted sessions to resume — a clean
        # boot must not accrue toward the breaker. If too many such boots have
        # happened in the configured window, skip auto-resume for THIS boot:
        # the gateway still comes up and serves real inbound messages, it just
        # stops replaying the session that keeps killing it. The session stays
        # resume_pending, so a real user message can still continue it (a human
        # is now in the loop). Defenses 1-2 cover the cron/CLI/terminal paths;
        # this catches every other SIGTERM source (e.g. a raw `terminal(
        # "launchctl kickstart ai.hermes.gateway")`).
        if candidates:
            try:
                from gateway import restart_loop_guard as _rlg

                _max_restarts, _window = self._restart_loop_guard_config()
                if _rlg.check_and_record(_max_restarts, _window):
                    return 0
            except Exception as exc:  # noqa: BLE001 — breaker must fail OPEN
                logger.debug("Restart-loop guard check skipped: %s", exc)

        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue

            # Already being resumed (e.g. scheduled at startup and still
            # in-flight) — don't synthesize a second continuation turn.
            if entry.session_key in self._running_agents:
                continue

            source = entry.origin
            adapter = self.adapters.get(source.platform)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s",
                    entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue

            # Validate the session owner against the current allowlist
            # before auto-resuming. A session created before
            # TELEGRAM_ALLOWED_USERS (or equivalent) was configured, or
            # before the owner was removed from it, must not silently
            # receive a full agent response on gateway restart just
            # because it has a resume-pending marker (issue #23778).
            try:
                if not self._is_user_authorized(source):
                    logger.warning(
                        "Skipping auto-resume for %s: session owner is no "
                        "longer authorized under the current allowlist",
                        entry.session_key,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "Skipping auto-resume for %s: authorization check failed: %s",
                    entry.session_key, exc,
                )
                continue

            # Claim the session slot *before* spawning the task so that an
            # inbound message arriving between task creation and the task's
            # first await (where _process_message_background sets the real
            # sentinel) sees the slot as occupied and queues behind it
            # instead of spinning up a duplicate AIAgent (#45456).
            self._running_agents[entry.session_key] = _AGENT_PENDING_SENTINEL
            self._running_agents_ts[entry.session_key] = time.time()
            self._persist_active_agents()

            # Empty-text internal event — the _is_resume_pending branch in
            # _handle_message_with_agent prepends the proper reason-aware
            # system note before the turn runs.
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            task = asyncio.create_task(
                self._run_startup_resume_event(adapter, event, entry.session_key)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if getattr(self, "_startup_restore_in_progress", False):
                tasks = getattr(self, "_startup_restore_tasks", None)
                if tasks is None:
                    tasks = []
                    self._startup_restore_tasks = tasks
                tasks.append(task)
            scheduled += 1
        if scheduled:
            logger.info(
                "Scheduled auto-resume for %d restart-interrupted session(s)",
                scheduled,
            )
        return scheduled

    def _startup_should_abort(self) -> bool:
        return (
            self._restart_requested
            or self._draining
            or self._shutdown_event.is_set()
        )

    async def _abort_startup_if_shutdown_requested(
        self,
        adapter: Optional[BasePlatformAdapter] = None,
        platform: Optional[Platform] = None,
    ) -> bool:
        """Clean up and exit startup when restart/shutdown begins mid-startup."""
        if not self._startup_should_abort():
            return False
        if adapter is not None and platform is not None:
            try:
                await adapter.cancel_background_tasks()
            except Exception as e:
                logger.debug("✗ %s background-task cancel error: %s", platform.value, e)
            await self._safe_adapter_disconnect(adapter, platform)
        stop_task = self._stop_task
        current_task = asyncio.current_task()
        if stop_task is not None and stop_task is not current_task:
            await stop_task
        elif not self._shutdown_event.is_set():
            await self.stop(
                restart=self._restart_requested,
                detached_restart=self._restart_detached,
                service_restart=self._restart_via_service,
            )
        return True

    async def start(self) -> bool:
        """
        Start the gateway and all configured platform adapters.
        
        Returns True if at least one adapter connected successfully.
        """
        logger.info("Starting Hermes Gateway...")
        try:
            self._gateway_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._gateway_loop = None
        logger.info("Session storage: %s", self.config.sessions_dir)

        # Sanity-check that systemd's TimeoutStopSec covers our drain
        # window.  When the user upgraded hermes-agent without re-running
        # ``hermes setup``, their unit file may still encode the old
        # default — in which case SIGKILL hits mid-drain and looks like
        # a phantom kill in the journal.  Best-effort, never raises.
        try:
            from gateway.shutdown_forensics import check_systemd_timing_alignment
            _alignment = check_systemd_timing_alignment(self._restart_drain_timeout)
            if _alignment is not None and _alignment.get("mismatch"):
                logger.warning(
                    "Stale systemd unit detected: %s has TimeoutStopSec=%.0fs but "
                    "drain_timeout=%.0fs (expected >=%.0fs). systemd may SIGKILL the "
                    "gateway mid-drain. Run `hermes gateway install --force` "
                    "to regenerate the unit, or shorten agent.restart_drain_timeout.",
                    _alignment.get("unit", "(unknown)"),
                    _alignment["timeout_stop_sec"],
                    _alignment["drain_timeout"],
                    _alignment["expected_min"],
                )
        except Exception as _e:
            logger.debug("check_systemd_timing_alignment failed: %s", _e)
        # Log the resolved max_iterations budget so operators can verify the
        # config.yaml → env bridge did the right thing at a glance (instead
        # of silently running at a stale .env value for weeks).
        try:
            _effective_max_iter = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))
            logger.info(
                "Agent budget: max_iterations=%d (agent.max_turns from config.yaml, "
                "or HERMES_MAX_ITERATIONS from .env, or default 90)",
                _effective_max_iter,
            )
        except Exception:
            pass
        # Redaction status: ON by default (#17691). Surface a prominent
        # warning if an operator has explicitly opted out so they don't
        # forget the downgrade is active — the redactor snapshots its
        # state at import time, so this log line is the source of truth
        # for this process's lifetime.
        try:
            _redact_raw = os.getenv("HERMES_REDACT_SECRETS", "true")
            _redact_on = _redact_raw.lower() in {"1", "true", "yes", "on"}
            if _redact_on:
                logger.info(
                    "Secret redaction: ENABLED (tool output, logs, and chat "
                    "responses are scrubbed before delivery)"
                )
            else:
                logger.warning(
                    "Secret redaction: DISABLED (HERMES_REDACT_SECRETS=%s). "
                    "API keys and tokens may appear verbatim in chat output, "
                    "session JSONs, and logs. Set security.redact_secrets: true "
                    "in config.yaml to re-enable.",
                    _redact_raw,
                )
        except Exception:
            pass
        try:
            from hermes_cli.profiles import get_active_profile_name
            _profile = get_active_profile_name()
            if _profile and _profile != "default":
                logger.info("Active profile: %s", _profile)
        except Exception:
            pass
        try:
            from gateway.status import write_runtime_status
            write_runtime_status(gateway_state="starting", exit_reason=None)
        except Exception:
            pass

        # Log any active supply-chain security advisories. Operators see this
        # in gateway.log and `hermes status` surfaces it; we do NOT block
        # startup or surface it inline to user messages, since the gateway
        # operator is the one who can act on it (uninstall the package,
        # rotate credentials).  See hermes_cli/security_advisories.py.
        try:
            from hermes_cli.security_advisories import (
                detect_compromised,
                gateway_log_message,
            )
            _adv_hits = detect_compromised()
            _adv_msg = gateway_log_message(_adv_hits)
            if _adv_msg:
                logger.warning("%s", _adv_msg)
                logger.warning(
                    "Run `hermes doctor` on the gateway host for full "
                    "remediation steps."
                )
        except Exception:
            logger.debug(
                "security advisory check failed at gateway startup",
                exc_info=True,
            )
        if await self._abort_startup_if_shutdown_requested():
            return True
        
        # Warn if no user allowlists are configured and open access is not opted in
        _builtin_allowed_vars = (
            "TELEGRAM_ALLOWED_USERS", "DISCORD_ALLOWED_USERS",
            "WHATSAPP_ALLOWED_USERS", "WHATSAPP_CLOUD_ALLOWED_USERS",
            "SLACK_ALLOWED_USERS",
            "SIGNAL_ALLOWED_USERS", "SIGNAL_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_USERS",
            "TELEGRAM_GROUP_ALLOWED_CHATS",
            "EMAIL_ALLOWED_USERS",
            "SMS_ALLOWED_USERS", "MATTERMOST_ALLOWED_USERS",
            "MATRIX_ALLOWED_USERS", "DINGTALK_ALLOWED_USERS",
            "FEISHU_ALLOWED_USERS",
            "WECOM_ALLOWED_USERS",
            "WECOM_CALLBACK_ALLOWED_USERS",
            "WEIXIN_ALLOWED_USERS",
            "BLUEBUBBLES_ALLOWED_USERS",
            "QQ_ALLOWED_USERS",
            "YUANBAO_ALLOWED_USERS",
            "GATEWAY_ALLOWED_USERS",
        )
        _builtin_allow_all_vars = (
            "TELEGRAM_ALLOW_ALL_USERS", "DISCORD_ALLOW_ALL_USERS",
            "WHATSAPP_ALLOW_ALL_USERS", "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
            "SLACK_ALLOW_ALL_USERS",
            "SIGNAL_ALLOW_ALL_USERS", "EMAIL_ALLOW_ALL_USERS",
            "SMS_ALLOW_ALL_USERS", "MATTERMOST_ALLOW_ALL_USERS",
            "MATRIX_ALLOW_ALL_USERS", "DINGTALK_ALLOW_ALL_USERS",
            "FEISHU_ALLOW_ALL_USERS",
            "WECOM_ALLOW_ALL_USERS",
            "WECOM_CALLBACK_ALLOW_ALL_USERS",
            "WEIXIN_ALLOW_ALL_USERS",
            "BLUEBUBBLES_ALLOW_ALL_USERS",
            "QQ_ALLOW_ALL_USERS",
            "YUANBAO_ALLOW_ALL_USERS",
        )
        # Also pick up plugin-registered platforms — each entry can declare
        # its own allowed_users_env / allow_all_env, so the warning stays
        # accurate as plugins like IRC come online.
        _plugin_allowed_vars: tuple = ()
        _plugin_allow_all_vars: tuple = ()
        try:
            from gateway.platform_registry import platform_registry
            _plugin_allowed_vars = tuple(
                e.allowed_users_env for e in platform_registry.plugin_entries()
                if e.allowed_users_env
            )
            _plugin_allow_all_vars = tuple(
                e.allow_all_env for e in platform_registry.plugin_entries()
                if e.allow_all_env
            )
        except Exception:
            pass
        _any_allowlist = any(
            os.getenv(v) for v in _builtin_allowed_vars + _plugin_allowed_vars
        )
        _allow_all = os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"} or any(
            os.getenv(v, "").lower() in {"true", "1", "yes"}
            for v in _builtin_allow_all_vars + _plugin_allow_all_vars
        )
        if not _any_allowlist and not _allow_all:
            logger.warning(
                "No env user allowlists configured. Messaging platforms default to "
                "pairing/allowlist policies and will deny unknown senders unless you "
                "configure platform allowlists (e.g., TELEGRAM_ALLOWED_USERS=your_id) "
                "or explicitly opt in with GATEWAY_ALLOW_ALL_USERS=true plus "
                "dm_policy/group_policy: open on the platform."
            )

        reason = _own_policy_open_startup_violation(self.config)
        if reason:
            platform_value = reason.split(":", 1)[0]
            allow_all_env = None
            for platform, open_env in _OWN_POLICY_OPEN_ENV.items():
                if platform.value == platform_value:
                    allow_all_env = open_env[2]
                    break
            logger.error(
                "Refusing to start: %s has dm_policy/group_policy set to 'open' "
                "but neither GATEWAY_ALLOW_ALL_USERS nor %s is enabled.",
                platform_value,
                allow_all_env or "a platform allow-all flag",
            )
            try:
                from gateway.status import write_runtime_status
                write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
            except Exception:
                pass
            self._request_clean_exit(reason)
            return True
        
        # Discover Python plugins before shell hooks so plugin block
        # decisions take precedence in tie cases.  The CLI startup path
        # does this via an explicit call in hermes_cli/main.py; the
        # gateway lazily imports run_agent inside per-request handlers,
        # so the discover_plugins() side-effect in model_tools.py is NOT
        # guaranteed to have run by the time we reach this point.
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
        except Exception:
            logger.warning(
                "plugin discovery failed at gateway startup", exc_info=True,
            )

        # Register the generic relay adapter when a connector relay URL is
        # configured (GATEWAY_RELAY_URL / gateway.relay_url). No URL -> no-op, so
        # direct/single-tenant deployments are unaffected. When configured, the
        # adapter dials the connector over a WebSocket, negotiates its capability
        # descriptor at handshake, and bridges inbound/outbound like any platform.
        try:
            from gateway.relay import (
                register_relay_adapter,
                relay_url,
                self_provision_relay,
                send_relay_policy,
            )

            # Boot-time relay self-provision: resolve the agent's NAS token ->
            # POST /relay/provision -> set GATEWAY_RELAY_* in os.environ BEFORE
            # registration reads them. No-op when relay is unconfigured, a secret
            # is already pinned, or no NAS token resolves (self-hosted, unenrolled).
            # Never raises.
            self_provision_relay()

            if register_relay_adapter():
                logger.info("relay adapter registered (connector at %s)", relay_url())
                # Declare this gateway's relevance policy (mention-gating /
                # free-response / allow-bots) to the connector so the SAME
                # behavior governs relay delivery (Phase 6 Unit ζ). Runs after
                # the secret is resolved; never raises, never blocks boot.
                send_relay_policy()
        except Exception:
            logger.warning(
                "relay adapter registration failed at gateway startup", exc_info=True,
            )

        # Register declarative shell hooks from cli-config.yaml.  Gateway
        # has no TTY, so consent has to come from one of the three opt-in
        # channels (--accept-hooks on launch, HERMES_ACCEPT_HOOKS env var,
        # or hooks_auto_accept: true in config.yaml).  We pass
        # accept_hooks=False here and let register_from_config resolve
        # the effective value from env + config itself — the CLI-side
        # registration already honored --accept-hooks, and re-reading
        # hooks_auto_accept here would just duplicate that lookup.
        # Failures are logged but must never block gateway startup.
        try:
            from hermes_cli.config import load_config
            from agent.shell_hooks import register_from_config
            register_from_config(load_config(), accept_hooks=False)
        except Exception:
            logger.debug(
                "shell-hook registration failed at gateway startup",
                exc_info=True,
            )

        # Discover and load event hooks
        self.hooks.discover_and_load()

        
        # Recover background processes from checkpoint (crash recovery)
        try:
            from tools.process_registry import process_registry
            recovered = process_registry.recover_from_checkpoint()
            if recovered:
                logger.info("Recovered %s background process(es) from previous run", recovered)
        except Exception as e:
            logger.warning("Process checkpoint recovery: %s", e)

        # Suspend sessions that were active when the gateway last exited.
        # This prevents stuck sessions from being blindly resumed on restart,
        # which can create an unrecoverable loop (#7536).  Suspended sessions
        # auto-reset on the next incoming message, giving the user a clean start.
        #
        # SKIP suspension after a clean (graceful) shutdown — the previous
        # process already drained active agents, so sessions aren't stuck.
        # This prevents unwanted auto-resets after `hermes update`,
        # `hermes gateway restart`, or `/restart`.
        _clean_marker = _hermes_home / ".clean_shutdown"
        if _clean_marker.exists():
            logger.info("Previous gateway exited cleanly — skipping session suspension")
            try:
                _clean_marker.unlink()
            except Exception:
                pass
        else:
            try:
                suspended = self.session_store.suspend_recently_active()
                if suspended:
                    logger.info("Marked %d in-flight session(s) as resumable from previous run", suspended)
            except Exception as e:
                logger.warning("Session suspension on startup failed: %s", e)

        # Stuck-loop detection (#7536): if a session has been active across
        # 3+ consecutive restarts, it's probably stuck in a loop (the same
        # history keeps causing the agent to hang).  Auto-suspend it so the
        # user gets a clean slate on the next message.
        try:
            stuck = self._suspend_stuck_loop_sessions()
            if stuck:
                logger.warning("Auto-suspended %d stuck-loop session(s)", stuck)
        except Exception as e:
            logger.debug("Stuck-loop detection failed: %s", e)

        # Serialize startup restore against inbound dispatch.  Platform
        # adapters can begin receiving messages as soon as they connect, but
        # restart-interrupted sessions are not auto-resumed until all startup
        # wiring below completes.  Queue inbound messages until the resume
        # pass runs and every synthetic resume turn has finished.
        self._startup_restore_in_progress = True
        self._startup_restore_queue = []
        self._startup_restore_tasks = []

        connected_count = 0
        enabled_platform_count = 0
        startup_nonretryable_errors: list[str] = []
        startup_retryable_errors: list[str] = []
        
        # Initialize and connect each configured platform
        for platform, platform_config in self.config.platforms.items():
            if await self._abort_startup_if_shutdown_requested():
                return True
            if not platform_config.enabled:
                continue
            enabled_platform_count += 1
            
            adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                # Distinguish between missing builtin deps and missing plugin
                _pval = platform.value
                _builtin_names = {m.value for m in Platform.__members__.values()}
                if _pval not in _builtin_names:
                    logger.warning(
                        "No adapter for '%s' — is the plugin installed? "
                        "(platform is enabled in config.yaml but no plugin registered it)",
                        _pval,
                    )
                else:
                    logger.warning("No adapter available for %s", _pval)
                continue
            
            # Set up message + fatal error handlers
            adapter.set_message_handler(self._handle_message)
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
            adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
            adapter._busy_text_mode = self._busy_text_mode
            
            # Try to connect
            logger.info("Connecting to %s...", platform.value)
            self._update_platform_runtime_status(
                platform.value,
                platform_state="connecting",
                error_code=None,
                error_message=None,
            )
            try:
                success = await self._connect_adapter_with_timeout(adapter, platform)
                if await self._abort_startup_if_shutdown_requested(adapter, platform):
                    return True
                if success:
                    self.adapters[platform] = adapter
                    self._sync_voice_mode_state_to_adapter(adapter)
                    connected_count += 1
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="connected",
                        error_code=None,
                        error_message=None,
                    )
                    logger.info("✓ %s connected", platform.value)
                else:
                    logger.warning("✗ %s failed to connect", platform.value)
                    # Defensive cleanup: a failed connect() may have
                    # allocated resources (aiohttp.ClientSession, poll
                    # tasks, bridge subprocesses) before giving up.
                    # Without this call, those resources are orphaned
                    # and Python logs "Unclosed client session" at
                    # process exit. Adapter disconnect() implementations
                    # are expected to be idempotent and tolerate
                    # partial-init state.
                    await self._safe_adapter_disconnect(adapter, platform)
                    if adapter.has_fatal_error:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying" if adapter.fatal_error_retryable else "fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        target = (
                            startup_retryable_errors
                            if adapter.fatal_error_retryable
                            else startup_nonretryable_errors
                        )
                        target.append(
                            f"{platform.value}: {adapter.fatal_error_message}"
                        )
                        # Queue for reconnection if the error is retryable
                        if adapter.fatal_error_retryable:
                            self._failed_platforms[platform] = {
                                "config": platform_config,
                                "attempts": 1,
                                "next_retry": time.monotonic() + 30,
                            }
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=None,
                            error_message="failed to connect",
                        )
                        startup_retryable_errors.append(
                            f"{platform.value}: failed to connect"
                        )
                        # No fatal error info means likely a transient issue — queue for retry
                        self._failed_platforms[platform] = {
                            "config": platform_config,
                            "attempts": 1,
                            "next_retry": time.monotonic() + 30,
                        }
            except Exception as e:
                logger.error("✗ %s error: %s", platform.value, e)
                # Same defensive cleanup path for exceptions — an adapter
                # that raised mid-connect may still have a live
                # aiohttp.ClientSession or child subprocess.
                await self._safe_adapter_disconnect(adapter, platform)
                self._update_platform_runtime_status(
                    platform.value,
                    platform_state="retrying",
                    error_code=None,
                    error_message=str(e),
                )
                startup_retryable_errors.append(f"{platform.value}: {e}")
                # Unexpected exceptions are typically transient — queue for retry
                self._failed_platforms[platform] = {
                    "config": platform_config,
                    "attempts": 1,
                    "next_retry": time.monotonic() + 30,
                }
            if await self._abort_startup_if_shutdown_requested():
                return True

        # Multi-profile multiplexing: bring up adapters for every OTHER profile
        # this gateway serves. Each profile's adapters connect under that
        # profile's home + credential scope and stamp their inbound events with
        # the profile so the agent turn resolves correctly. No-op when off.
        try:
            _secondary_connected = await self._start_secondary_profile_adapters()
            connected_count += _secondary_connected
        except MultiplexConfigError as e:
            # Invalid multiplexer config — abort startup cleanly so the operator
            # fixes config.yaml rather than running a half-wired gateway.
            reason = str(e)
            logger.error("Gateway multiplexer config error: %s", reason)
            try:
                from gateway.status import write_runtime_status
                write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
            except Exception:
                pass
            self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
            self._request_clean_exit(reason)
            self._startup_restore_in_progress = False
            return True
        except Exception as e:
            logger.error("Secondary-profile adapter startup failed: %s", e, exc_info=True)

        if connected_count == 0:
            if startup_nonretryable_errors:
                reason = "; ".join(startup_nonretryable_errors)
                logger.error("Gateway hit a non-retryable startup conflict: %s", reason)
                try:
                    from gateway.status import write_runtime_status
                    write_runtime_status(gateway_state="startup_failed", exit_reason=reason)
                except Exception:
                    pass
                self._exit_code = GATEWAY_FATAL_CONFIG_EXIT_CODE
                self._request_clean_exit(reason)
                self._startup_restore_in_progress = False
                return True
            if enabled_platform_count > 0:
                if startup_retryable_errors:
                    # All enabled platforms hit retryable failures (network
                    # blip, bridge not paired, npm install timeout, etc.).
                    # Keep the gateway alive so:
                    #   • cron jobs still run
                    #   • the reconnect watcher gets a chance to recover the
                    #     failing platforms once the underlying problem is
                    #     fixed (e.g. user runs `hermes whatsapp`, fixes
                    #     proxy, etc.)
                    # Exiting here used to convert a single misconfigured
                    # platform into an infinite systemd restart loop.
                    reason = "; ".join(startup_retryable_errors)
                    logger.warning(
                        "Gateway started with no connected platforms — "
                        "%d platform(s) queued for retry: %s",
                        len(self._failed_platforms), reason,
                    )
                    try:
                        from gateway.status import write_runtime_status
                        write_runtime_status(
                            gateway_state="degraded",
                            exit_reason=None,
                        )
                    except Exception:
                        pass
                    # Fall through to the normal "running" state — reconnect
                    # watcher takes it from here.
                # All enabled platforms had no adapter (missing library or credentials).
                # In fleet deployments the same config.yaml is shared across nodes that
                # may only have credentials for a subset of platforms.  Rather than
                # failing hard, degrade gracefully and allow cron jobs to run (#5196).
                logger.warning(
                    "No adapter could be created for any of the %d configured platform(s). "
                    "Check that required dependencies are installed and credentials are set. "
                    "Gateway will continue for cron job execution.",
                    enabled_platform_count,
                )
            else:
                logger.warning("No messaging platforms enabled.")
                logger.info("Gateway will continue running for cron job execution.")
        
        # Update delivery router with adapters
        if await self._abort_startup_if_shutdown_requested():
            return True
        self.delivery_router.adapters = self.adapters
        self._wire_teams_pipeline_runtime()

        self._running = True
        self._update_runtime_status("running")
        
        # Emit gateway:startup hook
        hook_count = len(self.hooks.loaded_hooks)
        if hook_count:
            logger.info("%s hook(s) loaded", hook_count)
        await self.hooks.emit("gateway:startup", {
            "platforms": [p.value for p in self.adapters.keys()],
        })
        
        if connected_count > 0:
            logger.info("Gateway running with %s platform(s)", connected_count)
        
        # Build initial channel directory for send_message name resolution
        try:
            from gateway.channel_directory import build_channel_directory
            directory = await build_channel_directory(self.adapters)
            ch_count = sum(len(chs) for chs in directory.get("platforms", {}).values())
            logger.info("Channel directory built: %d target(s)", ch_count)
        except Exception as e:
            logger.warning("Channel directory build failed: %s", e)
        
        # Check if we're restarting after a /update command. If the update is
        # still running, keep watching so we notify once it actually finishes.
        notified = await self._send_update_notification()
        if not notified and any(
            path.exists()
            for path in (
                _hermes_home / ".update_pending.json",
                _hermes_home / ".update_pending.claimed.json",
            )
        ):
            self._schedule_update_notification_watch()

        # Give freshly connected platform adapters a brief moment to settle
        # before sending restart/startup lifecycle messages. In practice this
        # helps Discord thread deliveries right after reconnect.
        if connected_count > 0:
            await asyncio.sleep(1.0)

        # Notify the chat that initiated /restart that the gateway is back.
        planned_restart_notification_pending = _planned_restart_notification_pending()
        # Capture, before _send_restart_notification() unlinks the marker,
        # whether this process booted from a chat-originated /restart. Used as
        # a one-shot signal by the /restart redelivery guard so a missing
        # dedup marker only suppresses a /restart when we KNOW we just came out
        # of a restart cycle (see _is_stale_restart_redelivery).
        if _restart_notification_pending() or planned_restart_notification_pending:
            self._booted_from_restart = True
        await self._send_restart_notification()

        # Broadcast a lightweight "gateway is back" message to configured home
        # channels only for non-chat planned restarts (terminal/SIGUSR1/service
        # paths). Chat-originated /restart already has a precise reply target
        # in .restart_notify.json, so keep that lifecycle in the originating
        # chat/topic instead of also leaking it to the configured home channel.
        if planned_restart_notification_pending:
            try:
                await self._send_home_channel_startup_notifications(
                    skip_targets=None,
                )
            finally:
                _clear_planned_restart_notification()

        # Automatically continue fresh sessions that were interrupted by the
        # previous gateway restart/shutdown.  The resume_pending flag is cleared
        # by the normal successful-turn path, so a failed auto-resume remains
        # visible for manual recovery on the next user message.
        self._schedule_resume_pending_sessions()
        await self._finish_startup_restore()

        # Drain any recovered process watchers (from crash recovery checkpoint)
        try:
            from tools.process_registry import process_registry
            # Detach the current batch atomically: reassigning to a fresh list
            # takes ownership of exactly the watchers present now, so any watcher
            # appended concurrently during the yield below isn't silently dropped
            # by a clear() on the shared list.
            watchers = process_registry.pending_watchers
            process_registry.pending_watchers = []
            # Process in batches of 100 with event-loop yield points to avoid
            # O(n^2) event-loop blocking when recovering thousands of watchers.
            for i, watcher in enumerate(watchers):
                asyncio.create_task(self._run_process_watcher(watcher))
                logger.info("Resumed watcher for recovered process %s", watcher.get("session_id"))
                if i % 100 == 99:
                    await asyncio.sleep(0)
        except Exception as e:
            logger.error("Recovered watcher setup error: %s", e)

        # Start background session expiry watcher to finalize expired sessions
        asyncio.create_task(self._session_expiry_watcher())

        # Start background kanban notifier — delivers `completed`, `blocked`,
        # `spawn_auto_blocked`, and `crashed` events to gateway subscribers
        # so human-in-the-loop workflows hear back without polling.
        asyncio.create_task(self._kanban_notifier_watcher())

        # Start background kanban dispatcher — spawns workers for ready
        # tasks. Gated by `kanban.dispatch_in_gateway` (default True).
        # When false, users run `hermes kanban daemon` externally or
        # simply don't use kanban; this loop becomes a no-op.
        asyncio.create_task(self._kanban_dispatcher_watcher())

        # Start background reconnection watcher for platforms that failed at startup
        if self._failed_platforms:
            logger.info(
                "Starting reconnection watcher for %d failed platform(s): %s",
                len(self._failed_platforms),
                ", ".join(p.value for p in self._failed_platforms),
            )
        asyncio.create_task(self._platform_reconnect_watcher())

        # Start background handoff watcher — picks up CLI sessions marked
        # handoff_state='pending' in state.db and re-binds them to the
        # destination platform's home channel, then forges a synthetic user
        # turn so the agent kicks off the new chat.
        asyncio.create_task(self._handoff_watcher())

        # Start background async-delegation watcher — drains completion events
        # from delegate_task(background=true) subagents and injects each
        # result back into its originating session as a new turn, covering the
        # idle case where the subagent finishes with no agent turn running.
        asyncio.create_task(self._async_delegation_watcher())

        # Start the scale-to-zero idle watcher ONLY when this instance is opted
        # in (the NAS "Labs" HERMES_SCALE_TO_ZERO stamp), messaging is
        # relay-only/absent, and a wakeUrl is registered (decisions.md D1/D11/
        # §3.4(1)). A non-opted instance never starts it, so behaviour is exactly
        # as today. When armed, the watcher drives the relay dormant on sustained
        # idle so the platform (Fly autostop:"suspend") can suspend the machine.
        try:
            if self._scale_to_zero_should_arm():
                logger.info(
                    "scale-to-zero: armed (idle timeout %.0fs) — watching for idle",
                    self._scale_to_zero_idle_timeout_seconds(),
                )
                asyncio.create_task(self._scale_to_zero_watcher())
            else:
                # Surface WHY an OPTED-IN instance didn't arm (a non-opted instance
                # not arming is normal — stay silent there). Without this, a failed
                # arm is invisible and "why won't it suspend/wake?" needs a box-dive.
                self._log_scale_to_zero_not_armed_reason()
        except Exception:  # noqa: BLE001 - arming must never block startup
            logger.debug("scale-to-zero: arm check failed at startup", exc_info=True)

        # Start background drain-control watcher — reconciles the gateway's
        # new-turn accept-state with the external ``.drain_request.json`` marker
        # the dashboard begin/cancel-drain endpoint writes (Phase 2). A marker
        # left behind by a prior instantiation (durable-volume restart, NS-570)
        # is ignored via its instantiation epoch; only a current-epoch marker
        # engages drain on the first tick.
        asyncio.create_task(self._drain_control_watcher())

        logger.info("Press Ctrl+C to stop")
        
        return True

    async def _handoff_watcher(self, interval: float = 2.0) -> None:
        """Background task that processes pending CLI→gateway session handoffs.

        Polls ``state.db`` for sessions in ``handoff_state='pending'`` and,
        for each one:

        1. Atomically claims it (pending → running).
        2. Resolves the destination platform's configured home channel.
        3. Re-binds the gateway's session_key for that home channel to the
           CLI's existing session_id via ``session_store.switch_session`` so
           the full role-aware transcript replays on the next agent turn.
        4. Forges a synthetic ``MessageEvent`` (``internal=True``) with a
           handoff-notice text and dispatches through the normal gateway
           message pipeline so the agent runs and replies on the platform.
        5. Marks the row ``completed`` (or ``failed`` with ``handoff_error``).

        The CLI process is poll-blocked on the row's terminal state and
        prints the result to the user.
        """
        # Initial delay so the gateway is fully connected to its platforms
        # before we try to dispatch handoffs through them.
        await asyncio.sleep(5)
        while self._running:
            try:
                if self._session_db is None:
                    await asyncio.sleep(interval)
                    continue
                pending = await self._session_db.list_pending_handoffs()
                for row in pending:
                    session_id = row.get("id")
                    if not session_id:
                        continue
                    if not await self._session_db.claim_handoff(session_id):
                        # Another tick or another gateway already claimed it.
                        continue
                    try:
                        await self._process_handoff(row)
                        await self._session_db.complete_handoff(session_id)
                    except Exception as exc:
                        logger.warning(
                            "Handoff for session %s failed: %s",
                            session_id, exc, exc_info=True,
                        )
                        await self._session_db.fail_handoff(session_id, str(exc))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Handoff watcher tick error: %s", exc, exc_info=True)
            await asyncio.sleep(interval)

    async def _process_handoff(self, row: Dict[str, Any]) -> None:
        """Execute one handoff row. Raises on failure (caller marks failed)."""
        from gateway.config import Platform
        from gateway.session import SessionSource, build_session_key
        from gateway.platforms.base import MessageEvent

        cli_session_id = row["id"]
        platform_name = (row.get("handoff_platform") or "").strip().lower()
        if not platform_name:
            raise RuntimeError("handoff_platform is empty")

        # Resolve platform enum
        try:
            platform = Platform(platform_name)
        except (ValueError, KeyError):
            raise RuntimeError(f"unknown platform '{platform_name}'")

        # Adapter must be live
        adapter = self.adapters.get(platform)
        if not adapter:
            raise RuntimeError(
                f"platform '{platform_name}' is not active in this gateway"
            )

        # Home channel must be configured
        home = self.config.get_home_channel(platform)
        if not home or not home.chat_id:
            raise RuntimeError(
                f"no home channel configured for {platform_name}; "
                f"run /sethome on the desired chat first"
            )

        cli_title = row.get("title") or cli_session_id[:8]

        # Try to create a fresh thread on the destination so the handoff
        # has its own scrollback. Adapter returns None if threading isn't
        # supported (Matrix/WhatsApp/Signal/SMS) or if creation failed
        # (no permission, topics-mode off, parent is a DM, etc.). When
        # None we fall through to using the home channel directly — the
        # synthetic turn still lands; just without thread isolation.
        thread_name = f"Hermes — {cli_title}"
        try:
            new_thread_id = await adapter.create_handoff_thread(
                str(home.chat_id), thread_name,
            )
        except Exception as exc:
            logger.debug(
                "Handoff: create_handoff_thread raised on %s: %s",
                platform_name, exc, exc_info=True,
            )
            new_thread_id = None

        # Use the new thread if the adapter created one; otherwise fall
        # back to whatever thread (if any) the home channel was configured
        # with.
        effective_thread_id = new_thread_id or (
            str(home.thread_id) if home.thread_id else None
        )

        # Determine chat_type/user_id for the destination source.
        #
        # Telegram private-chat DM topics are represented differently from
        # group/forum threads by the inbound adapter. A handoff-created topic
        # in a positive Telegram chat_id must therefore use the same DM-topic
        # source shape as the user's next real message; otherwise the synthetic
        # handoff turn binds a generic `thread` session key while real replies
        # arrive on a `dm` session key.
        home_chat_id = str(home.chat_id)
        is_telegram_private_chat = (
            platform == Platform.TELEGRAM
            and looks_like_telegram_private_chat_id(home_chat_id)
        )

        if new_thread_id and not is_telegram_private_chat:
            dest_chat_type = "thread"
            dest_user_id = "system:handoff"
        else:
            # No thread — assume DM-style for the home channel. For Telegram
            # private-chat topics, use the real user id (same as chat_id) so
            # topic-mode checks and binding persistence see the same identity as
            # subsequent inbound user messages.
            dest_chat_type = "dm"
            dest_user_id = home_chat_id if is_telegram_private_chat else "system:handoff"

        dest_source = SessionSource(
            platform=platform,
            chat_id=home_chat_id,
            chat_name=home.name,
            chat_type=dest_chat_type,
            user_id=dest_user_id,
            user_name="Handoff",
            thread_id=effective_thread_id,
        )

        # Compute the gateway's session_key for that destination using the
        # same rules its adapters use, so switch_session targets the right
        # entry. For thread destinations build_session_key keys without
        # user_id (thread_sessions_per_user defaults to False) — so the
        # next real user message in the thread shares this same session.
        platform_cfg = self.config.platforms.get(platform)
        extra = platform_cfg.extra if platform_cfg else {}
        session_key = build_session_key(
            dest_source,
            group_sessions_per_user=extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=extra.get("thread_sessions_per_user", False),
        )

        # Make sure there's an entry in the session_store for this key. If
        # the home channel has never been used, get_or_create_session
        # creates one; switch_session then re-points it.
        self.session_store.get_or_create_session(dest_source)

        # Re-bind the destination key to the CLI session_id. switch_session
        # ends the prior session in SQLite and reopens the CLI session under
        # the new key. The CLI's transcript becomes the active one for the
        # gateway from this moment on.
        switched = self.session_store.switch_session(session_key, cli_session_id)
        if switched is None:
            raise RuntimeError(
                f"could not switch session key {session_key} → {cli_session_id}"
            )

        # Evict any cached AIAgent for this session_key so the next dispatch
        # rebuilds it against the CLI session_id (mirrors /resume / /branch).
        self._evict_cached_agent(session_key)

        # Cancel any in-flight running-agent state for the destination key
        # so the synthetic turn isn't queued behind a stale running flag.
        self._release_running_agent_state(session_key)

        synthetic_text = (
            f"[Session was just handed off from CLI (\"{cli_title}\") to this "
            f"channel. The full prior conversation history is loaded above. "
            f"Briefly confirm you're working here and summarize what we were "
            f"working on, so the user can continue from this device.]"
        )

        synthetic_event = MessageEvent(
            text=synthetic_text,
            source=dest_source,
            internal=True,
        )

        logger.info(
            "Handoff: dispatching synthetic turn for CLI session %s → %s "
            "(home=%s, thread=%s, session_key=%s)",
            cli_session_id, platform_name, home.chat_id, effective_thread_id,
            session_key,
        )

        # Dispatch through the runner directly. Going through
        # adapter.handle_message would spawn a background task and we'd
        # lose synchronous error visibility; calling _handle_message inline
        # keeps the success/failure path observable for the watcher.
        response_text = await self._handle_message(synthetic_event)
        if not response_text:
            # Streaming may have already delivered the response inline.
            # Either way, agent ran without raising — count as success.
            return

        # Send the agent's reply to the destination. Route to the new
        # thread if we created one; otherwise the configured home channel
        # (which may itself carry a thread_id).
        send_metadata: Dict[str, Any] = {}
        if effective_thread_id:
            send_metadata["thread_id"] = effective_thread_id
        try:
            result = await adapter.send(
                chat_id=str(home.chat_id),
                content=response_text,
                metadata=send_metadata or None,
            )
        except Exception as exc:
            raise RuntimeError(f"adapter.send failed: {exc}") from exc

        if not getattr(result, "success", True):
            err = getattr(result, "error", "send returned success=False")
            raise RuntimeError(f"adapter.send failed: {err}")

    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that finalizes expired sessions.

        Runs every ``interval`` seconds (default 5 min).  For each session
        whose reset policy has expired, invokes ``on_session_finalize``
        hooks, cleans up the cached AIAgent's tool resources, evicts the
        cache entry so it can be garbage-collected, and marks the session
        so it won't be finalized again.
        """
        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _finalize_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FINALIZE_RETRIES = 3
        while self._running:
            try:
                self.session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.session_store._entries.items()):
                    if entry.expiry_finalized:
                        continue
                    if not self.session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to finalize (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        try:
                            from hermes_cli.plugins import invoke_hook as _invoke_hook
                            _parts = key.split(":")
                            _platform = _parts[2] if len(_parts) > 2 else ""
                            _invoke_hook(
                                "on_session_finalize",
                                session_id=entry.session_id,
                                platform=_platform,
                                reason="session_expired",
                            )
                        except Exception:
                            pass
                        # Shut down memory provider and close tool resources
                        # on the cached agent.  Idle agents live in
                        # _agent_cache (not _running_agents), so look there.
                        _cached_agent = None
                        _cache_lock = getattr(self, "_agent_cache_lock", None)
                        if _cache_lock is not None:
                            with _cache_lock:
                                _cached = self._agent_cache.get(key)
                                _cached_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
                        # Fall back to _running_agents in case the agent is
                        # still mid-turn when the expiry fires.
                        if _cached_agent is None:
                            _cached_agent = self._running_agents.get(key)
                        if _cached_agent and _cached_agent is not _AGENT_PENDING_SENTINEL:
                            await self._cleanup_agent_resources_off_loop(
                                _cached_agent, context="session expiry"
                            )
                        # Drop the cache entry so the AIAgent (and its LLM
                        # clients, tool schemas, memory provider refs) can
                        # be garbage-collected.  Otherwise the cache grows
                        # unbounded across the gateway's lifetime.
                        self._evict_cached_agent(key)
                        # Permanently finalizing this session — drop its
                        # per-session control state so the dicts don't grow
                        # unbounded across the gateway's lifetime. (Idle
                        # agent-cache eviction must NOT prune these: the
                        # session is still alive and a resumed turn rebuilds
                        # its agent from these overrides. Only true session
                        # finalization, /new, and /reset clear them.)
                        self._session_model_overrides.pop(key, None)
                        self._set_session_reasoning_override(key, None)
                        if hasattr(self, "_pending_model_notes"):
                            self._pending_model_notes.pop(key, None)
                        _pending_approvals = getattr(self, "_pending_approvals", None)
                        if isinstance(_pending_approvals, dict):
                            _pending_approvals.pop(key, None)
                        _update_prompt_pending = getattr(self, "_update_prompt_pending", None)
                        if isinstance(_update_prompt_pending, dict):
                            _update_prompt_pending.pop(key, None)
                        with self.session_store._lock:
                            entry.expiry_finalized = True
                            self.session_store._save()
                        logger.debug(
                            "Session expiry finalized for %s",
                            entry.session_id,
                        )
                        _finalize_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _finalize_failures.get(entry.session_id, 0) + 1
                        _finalize_failures[entry.session_id] = failures
                        if failures >= _MAX_FINALIZE_RETRIES:
                            logger.warning(
                                "Session finalize gave up after %d attempts for %s: %s. "
                                "Marking as finalized to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            with self.session_store._lock:
                                entry.expiry_finalized = True
                                self.session_store._save()
                            _finalize_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Session finalize failed (%d/%d) for %s: %s",
                                failures, _MAX_FINALIZE_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _done = sum(
                        1 for _, e in _expired_entries if e.expiry_finalized
                    )
                    _failed = len(_expired_entries) - _done
                    if _failed:
                        logger.info(
                            "Session expiry done: %d finalized, %d pending retry",
                            _done, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d finalized", _done,
                        )

                # Sweep agents that have been idle beyond the TTL regardless
                # of session reset policy.  This catches sessions with very
                # long / "never" reset windows, whose cached AIAgents would
                # otherwise pin memory for the gateway's entire lifetime.
                try:
                    _idle_evicted = self._sweep_idle_cached_agents()
                    if _idle_evicted:
                        logger.info(
                            "Agent cache idle sweep: evicted %d agent(s)",
                            _idle_evicted,
                        )
                except Exception as _e:
                    logger.debug("Idle agent sweep failed: %s", _e)

                # Periodically prune stale SessionStore entries.  The
                # in-memory dict (and sessions.json) would otherwise grow
                # unbounded in gateways serving many rotating chats /
                # threads / users over long time windows.  Pruning is
                # invisible to users — a resumed session just gets a
                # fresh session_id, exactly as if the reset policy fired.
                _last_prune_ts = getattr(self, "_last_session_store_prune_ts", 0.0)
                _prune_interval = 3600.0  # once per hour
                if time.time() - _last_prune_ts > _prune_interval:
                    try:
                        _max_age = int(
                            getattr(self.config, "session_store_max_age_days", 0) or 0
                        )
                        if _max_age > 0:
                            _pruned = self.session_store.prune_old_entries(_max_age)
                            if _pruned:
                                logger.info(
                                    "SessionStore prune: dropped %d stale entries",
                                    _pruned,
                                )
                    except Exception as _e:
                        logger.debug("SessionStore prune failed: %s", _e)
                    self._last_session_store_prune_ts = time.time()
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    def _active_profile_name(self) -> str:
        """Return the profile name this gateway represents."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name() or "default"
        except Exception:
            return "default"

    # ── Kanban board watchers ───────────────────────────────────────────
    # The kanban notifier/dispatcher watcher loops + their helpers live in
    # GatewayKanbanWatchersMixin (gateway/kanban_watchers.py). They use only
    # self state, so inheriting the mixin keeps every self._kanban_* call site
    # working unchanged while lifting ~1,000 LOC out of this file.

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
        Retryable failures (network/DNS blips) keep retrying at the backoff
        cap indefinitely — they self-heal once connectivity returns, so a
        transient outage never requires manual intervention. Non-retryable
        failures (bad auth, etc.) drop out of the queue immediately. The
        circuit breaker (``_pause_failed_platform`` / ``/platform pause``)
        remains available for manual operator control via ``/platform list``
        and ``/platform resume <name>``, but is no longer triggered
        automatically — auto-pausing a recovered platform was the cause of
        bots silently staying dead after a transient DNS failure.
        """
        _BACKOFF_CAP = 300  # 5 minutes max between retries

        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    if self._failed_platforms:
                        break
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms[platform]
                # Skip paused platforms entirely — they need explicit
                # /platform resume to come back.
                if info.get("paused"):
                    continue
                if now < info["next_retry"]:
                    continue  # not time yet

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                logger.info(
                    "Reconnecting %s (attempt %d)...",
                    platform.value, attempt,
                )

                adapter = None
                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._handle_message)
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    adapter.set_busy_session_handler(self._handle_active_session_busy_message)
                    adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
                    adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
                    adapter._busy_text_mode = self._busy_text_mode

                    # Reconnect after an outage: preserve the platform's
                    # server-side update queue so messages sent while the bot
                    # was offline are delivered rather than dropped (#46621).
                    success = await self._connect_adapter_with_timeout(
                        adapter, platform, is_reconnect=True
                    )
                    if success:
                        self.adapters[platform] = adapter
                        self._sync_voice_mode_state_to_adapter(adapter)
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            await build_channel_directory(self.adapters)
                        except Exception:
                            pass

                        # A platform that was offline at gateway startup never
                        # got its restart-interrupted sessions auto-resumed —
                        # the startup pass skips sessions whose adapter isn't
                        # connected yet. Now that it's back, retry the
                        # auto-resume scoped to this platform so recovery
                        # doesn't silently wait for a manual user message.
                        try:
                            self._schedule_resume_pending_sessions(platform=platform)
                        except Exception:
                            logger.debug(
                                "resume-pending reschedule after %s reconnect failed",
                                platform.value,
                                exc_info=True,
                            )
                    # Check if the failure is non-retryable
                    elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        logger.warning(
                            "Reconnect %s: non-retryable error (%s), removing from retry queue",
                            platform.value, adapter.fatal_error_message,
                        )
                        # The adapter is about to be dropped from the queue
                        # without ever being installed on self.adapters, so
                        # nothing else will call disconnect() on it. We must
                        # dispose it here, otherwise the resource owners it
                        # constructed in __init__ (ResponseStore for
                        # APIServerAdapter, etc.) leak 2 fds each. The
                        # gateway hits the 2560-fd limit after ~12h of
                        # failed reconnects at the 300s backoff cap (#37011).
                        await _dispose_unused_adapter(adapter)
                        del self._failed_platforms[platform]
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message or "failed to reconnect",
                        )
                        backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                        info["attempts"] = attempt
                        info["next_retry"] = time.monotonic() + backoff
                        logger.info(
                            "Reconnect %s failed, next retry in %ds",
                            platform.value, backoff,
                        )
                        # Same fd-leak concern as the non-retryable branch
                        # above: the adapter failed to connect and is being
                        # thrown away. Without an explicit dispose call, the
                        # resources it opened in __init__ stay open until
                        # the next GC pass — and aiohttp/SQLite handles
                        # don't get GC'd promptly, so 2 fds/retry leak at
                        # 300s backoff cap = ~12 fds/hour (#37011).
                        await _dispose_unused_adapter(adapter)
                        # Retryable failures (network/DNS blips) keep retrying
                        # at the backoff cap indefinitely — they self-heal once
                        # connectivity returns. We do NOT auto-pause them: a
                        # transient outage must never require manual `/platform
                        # resume` to recover. Non-retryable failures (bad auth,
                        # etc.) already drop out of the queue via the
                        # `not fatal_error_retryable` branch above, so anything
                        # reaching here is by definition retryable.
                except Exception as e:
                    if adapter is not None:
                        # An exception escaping the connect call path
                        # (DNS timeout, aiohttp server.start() crash, etc.)
                        # leaves the adapter in the same unowned state as
                        # the two branches above. Dispose so __init__
                        # resources don't accumulate while the watcher
                        # keeps retrying.
                        await _dispose_unused_adapter(adapter)
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )
                    # A raised exception during reconnect (connect timeout, DNS
                    # resolution failure, etc.) is inherently transient — keep
                    # retrying at the backoff cap rather than auto-pausing.

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def stop(
        self,
        *,
        restart: bool = False,
        detached_restart: bool = False,
        service_restart: bool = False,
    ) -> None:
        """Stop the gateway and disconnect all adapters."""
        if restart:
            self._restart_requested = True
            self._restart_detached = detached_restart
            self._restart_via_service = service_restart
        if self._stop_task is not None:
            await self._stop_task
            return

        async def _stop_impl() -> None:
            def _kill_tool_subprocesses(phase: str) -> None:
                """Kill tool subprocesses + tear down terminal envs + browsers.

                Called twice in the shutdown path: once eagerly after a
                drain timeout forces agent interrupt (so we reclaim bash/
                sleep children before systemd TimeoutStopSec escalates to
                SIGKILL on the cgroup — #8202), and once as a final
                catch-all at the end of _stop_impl() for the graceful
                path or anything respawned mid-teardown.

                All steps are best-effort; exceptions are swallowed so
                one subsystem's failure doesn't block the rest.
                """
                try:
                    from tools.process_registry import process_registry
                    _killed = process_registry.kill_all()
                    if _killed:
                        logger.info(
                            "Shutdown (%s): killed %d tool subprocess(es)",
                            phase, _killed,
                        )
                except Exception as _e:
                    logger.debug("process_registry.kill_all (%s) error: %s", phase, _e)
                try:
                    from tools.async_delegation import interrupt_all as _interrupt_async
                    _async_n = _interrupt_async(reason=f"gateway shutdown ({phase})")
                    if _async_n:
                        logger.info(
                            "Shutdown (%s): interrupted %d background delegation(s)",
                            phase, _async_n,
                        )
                except Exception as _e:
                    logger.debug("async interrupt_all (%s) error: %s", phase, _e)
                try:
                    from tools.terminal_tool import cleanup_all_environments
                    cleanup_all_environments()
                except Exception as _e:
                    logger.debug("cleanup_all_environments (%s) error: %s", phase, _e)
                try:
                    from tools.browser_tool import cleanup_all_browsers
                    cleanup_all_browsers()
                except Exception as _e:
                    logger.debug("cleanup_all_browsers (%s) error: %s", phase, _e)

            logger.info(
                "Stopping gateway%s...",
                " for restart" if self._restart_requested else "",
            )
            _stop_started_at = time.monotonic()

            def _phase_elapsed() -> float:
                return time.monotonic() - _stop_started_at

            self._running = False
            self._draining = True

            # Notify all chats with active agents BEFORE draining.
            # Adapters are still connected here, so messages can be sent.
            await self._notify_active_sessions_of_shutdown()
            logger.info(
                "Shutdown phase: notify_active_sessions done at +%.2fs",
                _phase_elapsed(),
            )

            timeout = self._restart_drain_timeout

            # Pre-mark sessions as resume_pending BEFORE the drain wait.
            # If the process is killed by the service manager during the
            # drain, the durable marker is already written so the next
            # gateway boot can recover in-flight sessions (#27856).
            _pre_drain_keys: list[str] = []
            for _sk, _agent in list(self._running_agents.items()):
                if _agent is _AGENT_PENDING_SENTINEL:
                    continue
                try:
                    self.session_store.mark_resume_pending(
                        _sk,
                        "restart_timeout" if self._restart_requested else "shutdown_timeout",
                    )
                    _pre_drain_keys.append(_sk)
                except Exception as _e:
                    logger.debug("pre-drain mark_resume_pending failed for %s: %s", _sk, _e)

            _drain_started_at = time.monotonic()
            active_agents, timed_out = await self._drain_active_agents(timeout)
            logger.info(
                "Shutdown phase: drain done at +%.2fs (drain took %.2fs, "
                "timed_out=%s, active_at_start=%d, active_now=%d)",
                _phase_elapsed(),
                time.monotonic() - _drain_started_at,
                timed_out,
                len(active_agents),
                self._running_agent_count(),
            )

            if not timed_out:
                # Drain completed gracefully — all running sessions finished.
                # Clear the pre-drain resume_pending markers so sessions that
                # completed during the drain window don't carry a stale flag.
                for _sk in _pre_drain_keys:
                    if _sk not in self._running_agents:
                        try:
                            self.session_store.clear_resume_pending(_sk)
                        except Exception as _e:
                            logger.debug(
                                "clear_resume_pending after drain failed for %s: %s",
                                _sk, _e,
                            )

            if timed_out:
                logger.warning(
                    "Gateway drain timed out after %.1fs with %d active agent(s); interrupting remaining work.",
                    timeout,
                    self._running_agent_count(),
                )
                # Mark forcibly-interrupted sessions as resume_pending BEFORE
                # interrupting the agents.  This preserves each session's
                # session_id + transcript so the next message on the same
                # session_key auto-resumes from the existing conversation
                # instead of getting routed through suspend_recently_active()
                # and converted into a fresh session.  Terminal escalation
                # for genuinely stuck sessions still flows through the
                # existing ``.restart_failure_counts`` stuck-loop counter
                # (incremented below, threshold 3), which sets
                # ``suspended=True`` and overrides resume_pending.
                #
                # Iterate self._running_agents (current) rather than the
                # drain-start ``active_agents`` snapshot — the snapshot
                # may include sessions that finished gracefully during
                # the drain window, and marking those falsely would give
                # them a stray restart-interruption system note on their
                # next turn even though their previous turn completed
                # cleanly.  Skip pending sentinels for the same reason
                # _interrupt_running_agents() does: their agent hasn't
                # started yet, there's nothing to interrupt, and the
                # session shouldn't carry a misleading resume flag.
                _resume_reason = (
                    "restart_timeout" if self._restart_requested else "shutdown_timeout"
                )
                for _sk, _agent in list(self._running_agents.items()):
                    if _agent is _AGENT_PENDING_SENTINEL:
                        continue
                    try:
                        self.session_store.mark_resume_pending(_sk, _resume_reason)
                    except Exception as _e:
                        logger.debug(
                            "mark_resume_pending failed for %s: %s",
                            _sk, _e,
                        )
                self._interrupt_running_agents(
                    _INTERRUPT_REASON_GATEWAY_RESTART if self._restart_requested else _INTERRUPT_REASON_GATEWAY_SHUTDOWN
                )
                interrupt_deadline = asyncio.get_running_loop().time() + 5.0
                while self._running_agents and asyncio.get_running_loop().time() < interrupt_deadline:
                    self._update_runtime_status("draining")
                    await asyncio.sleep(0.1)

                # Kill lingering tool subprocesses NOW, before we spend more
                # budget on adapter disconnect / session DB close.  Under
                # systemd (TimeoutStopSec bounded by drain_timeout+headroom),
                # deferring this to the end of stop() risks systemd escalating
                # to SIGKILL on the cgroup first — at which point bash/sleep
                # children left behind by an interrupted terminal tool get
                # killed by systemd instead of us (issue #8202).  The final
                # catch-all cleanup below still runs for the graceful path.
                _kill_tool_subprocesses("post-interrupt")
                logger.info(
                    "Shutdown phase: post-interrupt tool kill done at +%.2fs",
                    _phase_elapsed(),
                )

            if self._restart_requested and self._restart_detached:
                try:
                    await self._launch_detached_restart_command()
                except Exception as e:
                    logger.error("Failed to launch detached gateway restart: %s", e)

            await self._finalize_shutdown_agents(active_agents)

            # Also shut down memory providers on idle cached agents.
            # _finalize_shutdown_agents only handles agents that were
            # mid-turn at drain time; the _agent_cache may still hold
            # idle agents whose MemoryProviders never received
            # on_session_end().
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)
            if _cache_lock is not None and _cache is not None:
                with _cache_lock:
                    _idle_agents = list(_cache.values())
                    _cache.clear()
                for _entry in _idle_agents:
                    _agent = (
                        _entry[0] if isinstance(_entry, tuple) else _entry
                    )
                    # Bounded + off-loop so a wedged memory provider on one
                    # idle agent can't hang shutdown indefinitely — that path
                    # is why SIGTERM failed to kill the process (#53175).
                    await self._cleanup_agent_resources_off_loop(
                        _agent, context="shutdown idle-cache"
                    )

            for platform, adapter in list(self.adapters.items()):
                await self._bounded_adapter_teardown(adapter, platform)

            # Disconnect secondary-profile adapters (multiplex mode).
            for _prof, _amap in list(getattr(self, "_profile_adapters", {}).items()):
                for platform, adapter in list(_amap.items()):
                    await self._bounded_adapter_teardown(
                        adapter, platform, profile=_prof
                    )
                _amap.clear()
            if hasattr(self, "_profile_adapters"):
                self._profile_adapters.clear()
            logger.info(
                "Shutdown phase: all adapters disconnected at +%.2fs",
                _phase_elapsed(),
            )

            for _task in list(self._background_tasks):
                if _task is self._stop_task:
                    continue
                if _task is self._restart_task:
                    # The restart orchestration task is awaiting _stop_task
                    # right now; cancelling it would propagate CancelledError
                    # into this _stop_impl and skip _shutdown_event.set() /
                    # _exit_code = 75 (#12875).  It self-terminates anyway.
                    continue
                _task.cancel()
            self._background_tasks.clear()

            self.adapters.clear()
            for _session_key in list(self._running_agents):
                self._release_running_agent_state(_session_key)
            self._running_agents.clear()
            self._running_agents_ts.clear()
            if hasattr(self, "_active_session_leases"):
                self._active_session_leases.clear()
            self._pending_messages.clear()
            self._pending_approvals.clear()
            if hasattr(self, '_busy_ack_ts'):
                self._busy_ack_ts.clear()
            self._shutdown_event.set()

            # Global cleanup: kill any remaining tool subprocesses not tied
            # to a specific agent (catch-all for zombie prevention). On the
            # drain-timeout path we already did this earlier after agent
            # interrupt — this second call catches (a) the graceful path
            # where drain succeeded without interrupt, and (b) anything
            # that got respawned between the earlier call and adapter
            # disconnect (defense in depth; safe to call repeatedly).
            _kill_tool_subprocesses("final-cleanup")
            logger.info(
                "Shutdown phase: final-cleanup tool kill done at +%.2fs",
                _phase_elapsed(),
            )

            # Reap the process-global auxiliary-client cache once at the very
            # end of teardown.  Per-turn cleanup runs in _cleanup_agent_resources
            # for each active agent, but clients bound to worker-thread loops
            # that died with their ThreadPoolExecutor (notably cron ticks) only
            # get swept here.  Without this, long-running gateways accumulate
            # async httpx transports until they hit EMFILE on macOS's default
            # RLIMIT_NOFILE=256.  See #14210.
            try:
                from agent.auxiliary_client import shutdown_cached_clients
                shutdown_cached_clients()
            except Exception as _e:
                logger.debug("shutdown_cached_clients error: %s", _e)

            # Close SQLite session DBs so the WAL write lock is released.
            # Without this, --replace and similar restart flows leave the
            # old gateway's connection holding the WAL lock until Python
            # actually exits — causing 'database is locked' errors when
            # the new gateway tries to open the same file.
            # ``self`` holds the DB at ``_session_db`` (an AsyncSessionDB facade);
            # unwrap to the sync handle. ``session_store`` holds it at ``_db``.
            _self_db = getattr(self, "_session_db", None)
            _self_db = getattr(_self_db, "_db", _self_db)
            for _db in (_self_db, getattr(getattr(self, "session_store", None), "_db", None)):
                if _db is None or not hasattr(_db, "close"):
                    continue
                try:
                    _db.close()
                except Exception as _e:
                    logger.debug("SessionDB close error: %s", _e)
            GatewayRunner._shutdown_executor(self)
            logger.info(
                "Shutdown phase: SessionDB close done at +%.2fs",
                _phase_elapsed(),
            )

            from gateway.status import remove_pid_file, release_gateway_runtime_lock
            remove_pid_file()
            release_gateway_runtime_lock()

            # Write a clean-shutdown marker so the next startup knows this
            # wasn't a crash.  suspend_recently_active() only needs to run
            # after unexpected exits.  However, if the drain timed out and
            # agents were force-interrupted, their sessions may be in an
            # incomplete state (trailing tool response, no final assistant
            # message).  Skip the marker in that case so the next startup
            # suspends those sessions — giving users a clean slate instead
            # of resuming a half-finished tool loop.
            if not timed_out:
                try:
                    (_hermes_home / ".clean_shutdown").touch()
                except Exception:
                    pass
            else:
                logger.info(
                    "Skipping .clean_shutdown marker — drain timed out with "
                    "interrupted agents; next startup will suspend recently "
                    "active sessions."
                )

            # Track sessions that were active at shutdown for stuck-loop
            # detection (#7536).  On each restart, the counter increments
            # for sessions that were running.  If a session hits the
            # threshold (3 consecutive restarts while active), the next
            # startup auto-suspends it — breaking the loop.
            if active_agents:
                self._increment_restart_failure_counts(set(active_agents.keys()))

            if self._restart_requested and self._restart_command_source is None:
                try:
                    atomic_json_write(
                        _planned_restart_notification_path(),
                        {
                            "requested_at": time.time(),
                            "via_service": bool(self._restart_via_service),
                            "detached": bool(self._restart_detached),
                        },
                        indent=None,
                    )
                except Exception as e:
                    logger.debug("Failed to write planned restart notification marker: %s", e)

            if self._restart_requested and self._restart_via_service:
                self._launch_systemd_restart_shortcut()
                # Always exit with TEMPFAIL (75) on service-managed
                # restarts.  The shortcut helper above is best-effort and
                # commonly fails on real deployments: non-root gateway
                # units hit Polkit denials when invoking ``systemd-run
                # --system``, headless boxes have no user bus for
                # ``--user``, and operator-managed unit files may use
                # ``Restart=on-failure`` rather than ``Restart=always``.
                # Exit 75 paired with ``RestartForceExitStatus=75`` makes
                # systemd treat the planned restart as a controlled
                # failure and revive the unit via ``Restart=on-failure``,
                # regardless of whether the helper survived.  Without
                # this, a clean exit (0) on Linux left the gateway dead
                # until someone rebooted the host.  ``StartLimitBurst``
                # in the unit file still bounds accidental loops.
                self._exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE
                self._exit_reason = self._exit_reason or "Gateway restart requested"

            self._draining = False
            # Persist the terminal gateway_state. The default is "stopped",
            # but when this teardown was triggered by an UNEXPECTED external
            # signal (container/s6 SIGTERM on `docker restart` or image
            # upgrade, OOM-killer, bare `kill`) we instead persist "running"
            # to preserve the operator's run-intent across the restart.
            #
            # On Docker (s6-overlay), container_boot.py reads gateway_state
            # on the next boot and only auto-starts gateways whose last
            # state was "running" (_AUTOSTART_STATES). Persisting "stopped"
            # — or leaving the mid-shutdown "draining" marker in place — for
            # a routine `docker compose up --force-recreate` permanently
            # suppresses auto-start, so the messaging channels silently stay
            # dark until the operator manually restarts (issue #42675).
            #
            # An operator-initiated stop (`hermes gateway stop`,
            # systemd/launchd ExecStop, the s6 stop path, Ctrl+C) writes a
            # planned-stop marker BEFORE signalling, so it is classified as
            # a planned stop (not signal-initiated) and correctly persists
            # "stopped" — respecting the explicit intent. A restart also
            # persists "stopped" here; the restarting process brings the
            # gateway back up itself.
            if getattr(self, "_signal_initiated_shutdown", False) and not self._restart_requested:
                logger.info(
                    "Gateway stopped by an unexpected signal — persisting "
                    "gateway_state=running so container_boot auto-starts on "
                    "the next boot (issue #42675)"
                )
                self._update_runtime_status("running", self._exit_reason)
            else:
                self._update_runtime_status("stopped", self._exit_reason)
            logger.info("Gateway stopped (total teardown %.2fs)", _phase_elapsed())

        self._stop_task = asyncio.create_task(_stop_impl())
        await self._stop_task

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def _start_secondary_profile_adapters(self) -> int:
        """Bring up adapters for every non-active profile this gateway serves.

        Returns the number of secondary adapters that connected. No-op (returns
        0) unless ``gateway.multiplex_profiles`` is on.

        Each profile's adapters are created and connected under that profile's
        HERMES_HOME + secret scope (``_profile_runtime_scope``), stored in
        ``self._profile_adapters[profile]``, and given a message handler that
        stamps ``source.profile`` before delegating to the shared
        ``_handle_message`` — so the agent turn resolves that profile's config,
        skills, and credentials. Same-platform credential collisions (two
        profiles polling the same bot token) are detected and refused here, the
        only point that sees every profile's resolved credentials together.
        """
        if not getattr(self.config, "multiplex_profiles", False):
            return 0

        try:
            from hermes_cli.profiles import profiles_to_serve, get_active_profile_name
        except Exception:
            return 0

        active = get_active_profile_name() or "default"
        connected = 0
        # (platform, token-fingerprint) -> profile that claimed it. Detects two
        # profiles trying to poll the same bot credential (impossible to do
        # concurrently). Seed with the active profile's adapters.
        claimed: Dict[tuple, str] = {}
        for _plat, _ad in self.adapters.items():
            fp = self._adapter_credential_fingerprint(_ad)
            if fp is not None:
                claimed[(_plat, fp)] = active

        for profile_name, profile_home in profiles_to_serve(multiplex=True):
            if profile_name == active:
                continue  # handled by the primary startup loop
            try:
                connected += await self._start_one_profile_adapters(
                    profile_name, profile_home, claimed
                )
            except MultiplexConfigError:
                # Config error (e.g. a secondary profile binding a port) is not
                # transient — propagate so startup aborts cleanly instead of
                # limping along with a half-configured multiplexer.
                raise
            except Exception as e:
                logger.error(
                    "Failed to start adapters for profile '%s': %s",
                    profile_name, e, exc_info=True,
                )

        # Record served profiles in runtime status for `hermes status`.
        try:
            from gateway.status import write_runtime_status
            served = [active] + sorted(self._profile_adapters.keys())
            write_runtime_status(served_profiles=served)
        except Exception:
            logger.debug("could not record served_profiles", exc_info=True)

        return connected

    async def _start_one_profile_adapters(
        self, profile_name: str, profile_home: "Path", claimed: Dict[tuple, str]
    ) -> int:
        """Create+connect one profile's adapters under its runtime scope."""
        from gateway.config import load_gateway_config

        with _profile_runtime_scope(profile_home):
            profile_cfg = load_gateway_config()
            violation = _own_policy_open_startup_violation(profile_cfg)
        if violation:
            raise MultiplexConfigError(
                f"Profile '{profile_name}' enables {violation}. "
                "Enable GATEWAY_ALLOW_ALL_USERS or the platform allow-all flag "
                "for that profile, or change dm_policy/group_policy away from "
                "'open'."
            )

        profile_map = self._profile_adapters.setdefault(profile_name, {})
        connected = 0
        for platform, platform_config in profile_cfg.platforms.items():
            if not platform_config.enabled:
                continue
            # A secondary profile must NOT enable a port-binding platform: the
            # default profile's listener already serves every profile via the
            # /p/<profile>/ prefix, so a second bind can only collide. This is a
            # config error, not a transient failure — fail fast and loud.
            if platform.value in _PORT_BINDING_PLATFORM_VALUES:
                raise MultiplexConfigError(
                    f"Profile '{profile_name}' enables the port-binding platform "
                    f"'{platform.value}', but gateway.multiplex_profiles is on. The "
                    f"default profile owns the single shared HTTP listener and "
                    f"serves every profile through the /p/{profile_name}/ URL "
                    f"prefix — a secondary profile cannot bind its own port. "
                    f"Remove platforms.{platform.value} from profile "
                    f"'{profile_name}'s config.yaml (configure it only on the "
                    f"default profile)."
                )
            with _profile_runtime_scope(profile_home):
                adapter = self._create_adapter(platform, platform_config)
            if not adapter:
                continue

            # Same-token conflict detection — refuse a duplicate poll.
            fp = self._adapter_credential_fingerprint(adapter)
            if fp is not None:
                owner = claimed.get((platform, fp))
                if owner is not None:
                    logger.error(
                        "Profile '%s' and '%s' both configure %s with the same "
                        "credential — refusing to start the duplicate (a single "
                        "bot token cannot be polled twice). Give each profile its "
                        "own %s credential.",
                        owner, profile_name, platform.value, platform.value,
                    )
                    await self._safe_adapter_disconnect(adapter, platform)
                    continue
                claimed[(platform, fp)] = profile_name

            # Stamp every inbound event from this adapter with its profile so
            # the agent turn (and session key) resolve to the right home.
            adapter.set_message_handler(
                self._make_profile_message_handler(profile_name)
            )
            adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
            adapter.set_session_store(self.session_store)
            adapter.set_busy_session_handler(self._handle_active_session_busy_message)
            adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
            adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
            adapter._busy_text_mode = self._busy_text_mode

            try:
                with _profile_runtime_scope(profile_home):
                    success = await self._connect_adapter_with_timeout(adapter, platform)
                if success:
                    profile_map[platform] = adapter
                    connected += 1
                    logger.info("✓ %s connected (profile: %s)", platform.value, profile_name)
                else:
                    logger.warning("✗ %s failed to connect (profile: %s)", platform.value, profile_name)
                    await self._safe_adapter_disconnect(adapter, platform)
            except Exception as e:
                logger.error("✗ %s error (profile: %s): %s", platform.value, profile_name, e)
                await self._safe_adapter_disconnect(adapter, platform)
        return connected

    def _make_profile_message_handler(self, profile_name: str):
        """Return a message handler that stamps source.profile then delegates."""
        async def _handler(event):
            try:
                if getattr(event, "source", None) is not None and not event.source.profile:
                    event.source.profile = profile_name
            except Exception:
                pass
            return await self._handle_message(event)
        return _handler

    @staticmethod
    def _adapter_credential_fingerprint(adapter: Any) -> Optional[str]:
        """Return a stable, log-safe fingerprint of an adapter's credential.

        Used only to detect two profiles claiming the same bot token. Returns a
        salted hash (never the token itself) of the adapter's primary
        credential, or None when no credential is discoverable (in which case
        we don't attempt conflict detection for it).
        """
        token = None
        for attr in ("token", "bot_token", "_token", "api_token", "_bot_token"):
            val = getattr(adapter, attr, None)
            if isinstance(val, str) and val.strip():
                token = val.strip()
                break
        if not token:
            return None
        import hashlib
        return hashlib.sha256(("hermes-mux:" + token).encode("utf-8")).hexdigest()[:16]

    def _create_adapter(
        self, 
        platform: Platform, 
        config: Any
    ) -> Optional[BasePlatformAdapter]:
        """Create the appropriate adapter for a platform.

        Checks the platform_registry first (plugin adapters), then falls
        through to the built-in if/elif chain for core platforms.
        """
        if hasattr(config, "extra") and isinstance(config.extra, dict):
            config.extra.setdefault(
                "group_sessions_per_user",
                self.config.group_sessions_per_user,
            )
            config.extra.setdefault(
                "thread_sessions_per_user",
                getattr(self.config, "thread_sessions_per_user", False),
            )

        # ── Plugin-registered platforms (checked first) ───────────────────
        try:
            from gateway.platform_registry import platform_registry
            if platform_registry.is_registered(platform.value):
                adapter = platform_registry.create_adapter(platform.value, config)
                if adapter is not None:
                    # Adapters that need a back-reference to the gateway runner
                    # (e.g. for cross-platform admin alerts) declare a
                    # ``gateway_runner`` attribute. Inject it after creation so
                    # plugin adapters don't need a custom factory signature.
                    if hasattr(adapter, "gateway_runner"):
                        adapter.gateway_runner = self
                    return adapter
                # Registered but failed to instantiate — don't silently fall
                # through to built-ins (there are none for plugin platforms).
                logger.error(
                    "Platform '%s' is registered but adapter creation failed "
                    "(check dependencies and config)",
                    platform.value,
                )
                return None
        except Exception as e:
            logger.debug("Platform registry lookup for '%s' failed: %s", platform.value, e)
        # Fall through to built-in adapters below

        if platform == Platform.WHATSAPP_CLOUD:
            from gateway.platforms.whatsapp_cloud import (
                WhatsAppCloudAdapter,
                check_whatsapp_cloud_requirements,
            )
            if not check_whatsapp_cloud_requirements():
                logger.warning(
                    "WhatsApp Cloud: aiohttp/httpx missing — reinstall hermes-agent"
                )
                return None
            return WhatsAppCloudAdapter(config)
        
        elif platform == Platform.SIGNAL:
            from gateway.platforms.signal import SignalAdapter, check_signal_requirements
            if not check_signal_requirements():
                logger.warning("Signal: SIGNAL_HTTP_URL or SIGNAL_ACCOUNT not configured")
                return None
            return SignalAdapter(config)

        elif platform == Platform.WEIXIN:
            from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
            if not check_weixin_requirements():
                logger.warning("Weixin: aiohttp/cryptography not installed")
                return None
            return WeixinAdapter(config)

        elif platform == Platform.API_SERVER:
            from gateway.platforms.api_server import APIServerAdapter, check_api_server_requirements
            if not check_api_server_requirements():
                logger.warning("API Server: aiohttp not installed")
                return None
            return APIServerAdapter(config)

        elif platform == Platform.WEBHOOK:
            from gateway.platforms.webhook import WebhookAdapter, check_webhook_requirements
            if not check_webhook_requirements():
                logger.warning("Webhook: aiohttp not installed")
                return None
            adapter = WebhookAdapter(config)
            adapter.gateway_runner = self  # For cross-platform delivery
            return adapter

        elif platform == Platform.MSGRAPH_WEBHOOK:
            from gateway.platforms.msgraph_webhook import (
                MSGraphWebhookAdapter,
                check_msgraph_webhook_requirements,
            )
            if not check_msgraph_webhook_requirements():
                logger.warning("MSGraph webhook: aiohttp not installed")
                return None
            return MSGraphWebhookAdapter(config)

        elif platform == Platform.BLUEBUBBLES:
            from gateway.platforms.bluebubbles import BlueBubblesAdapter, check_bluebubbles_requirements
            if not check_bluebubbles_requirements():
                logger.warning("BlueBubbles: aiohttp/httpx missing or BLUEBUBBLES_SERVER_URL/BLUEBUBBLES_PASSWORD not configured")
                return None
            return BlueBubblesAdapter(config)

        elif platform == Platform.QQBOT:
            from gateway.platforms.qqbot import QQAdapter, check_qq_requirements
            if not check_qq_requirements():
                logger.warning("QQBot: aiohttp/httpx missing or QQ_APP_ID/QQ_CLIENT_SECRET not configured")
                return None
            return QQAdapter(config)

        elif platform == Platform.YUANBAO:
            from gateway.platforms.yuanbao import YuanbaoAdapter, WEBSOCKETS_AVAILABLE
            if not WEBSOCKETS_AVAILABLE:
                logger.warning("Yuanbao: websockets not installed. Run: pip install websockets")
                return None
            return YuanbaoAdapter(config)

        return None

    def _make_adapter_auth_check(
        self,
        platform: Platform,
    ) -> Callable[[str, Optional[str], Optional[str]], bool]:
        """Build a platform-bound auth callback for adapter use.

        Adapters that fetch external context (e.g. Slack
        ``conversations.replies``) call this through
        ``BasePlatformAdapter._is_sender_authorized`` to mark non-allowlisted
        senders as unverified in LLM context, mitigating indirect prompt
        injection from third parties in shared threads/channels.

        The returned callback delegates to :meth:`_is_user_authorized` so the
        full auth chain — platform allowlists, group allowlists, pairing
        store, allow-all flags — stays the single source of truth.
        """
        def check(
            user_id: str,
            chat_type: Optional[str] = None,
            chat_id: Optional[str] = None,
        ) -> bool:
            if not user_id:
                return False
            source = SessionSource(
                platform=platform,
                chat_id=chat_id or "",
                chat_type=chat_type or "group",
                user_id=user_id,
            )
            return self._is_user_authorized(source)
        return check






    async def _deliver_platform_notice(self, source, content: str) -> None:
        """Deliver a setup/operational notice using platform-specific privacy rules."""
        adapter = self.adapters.get(source.platform)
        if not adapter:
            return

        config = getattr(self, "config", None)
        notice_delivery = "public"
        if config and hasattr(config, "get_notice_delivery"):
            notice_delivery = config.get_notice_delivery(source.platform)

        metadata = self._thread_metadata_for_source(source)
        if notice_delivery == "private" and getattr(source, "user_id", None):
            try:
                result = await adapter.send_private_notice(
                    source.chat_id,
                    source.user_id,
                    content,
                    metadata=metadata,
                )
                if getattr(result, "success", False):
                    return
            except Exception:
                logger.debug(
                    "[%s] send_private_notice failed, falling back to public",
                    getattr(source, "platform", "?"),
                    exc_info=True,
                )

        await adapter.send(source.chat_id, content, metadata=metadata)

    async def _handle_message(self, event: MessageEvent) -> Optional[str]:
        """
        Handle an incoming message from any platform.
        
        This is the core message processing pipeline:
        1. Check user authorization
        2. Check for commands (/new, /reset, etc.)
        3. Check for running agent and interrupt if needed
        4. Get or create session
        5. Build context for agent
        6. Run agent conversation
        7. Return response
        """
        source = event.source

        # 🔴 Cross-session leak guard. This handler runs inside a per-message
        # asyncio task created via create_task(), which snapshots the spawning
        # context with copy_context(). If a *concurrent* message had already
        # bound its session via set_session_vars() when this task was created,
        # we inherited ITS HERMES_SESSION_* ContextVars. Until we bind our own
        # (a few steps down, in _set_session_env), any subprocess spawned here
        # would read the foreign session's identity via the subprocess-env
        # bridge — the _UNSET-strip guard there can't help because the vars are
        # set-to-foreign, not _UNSET. Reset to _UNSET now so that window strips
        # safe (no session) instead of leaking the sibling's. See
        # gateway/session_context.reset_session_vars + the inheritance test.
        try:
            from gateway.session_context import reset_session_vars
            reset_session_vars()
        except Exception:
            logger.debug("reset_session_vars failed at handler entry", exc_info=True)

        if (
            getattr(self, "_startup_restore_in_progress", False)
            and not getattr(event, "internal", False)
            and not getattr(event, "_hermes_startup_restore_replay", False)
        ):
            self._queue_startup_restore_event(event)
            return None

        # Internal events (e.g. background-process completion notifications)
        # are system-generated and must skip user authorization.
        is_internal = bool(getattr(event, "internal", False))

        # scale-to-zero (Phase 0, 0.B/F13): stamp the gateway-scoped last-inbound
        # clock for real (user-originated) inbound only. Internal/system events
        # (background-process completions, startup-restore replays) are NOT
        # traffic — counting them would keep a genuinely idle gateway awake. This
        # clock is what the idle predicate (gateway/scale_to_zero.is_idle) reads.
        if not is_internal:
            self._scale_to_zero_note_real_inbound()

        # Fire pre_gateway_dispatch plugin hook for user-originated messages.
        # Plugins receive the MessageEvent and may return a dict influencing flow:
        #   {"action": "skip",    "reason": ...}    -> drop (no reply, plugin handled)
        #   {"action": "rewrite", "text":  ...}     -> replace event.text, continue
        #   {"action": "allow"}   /   None          -> normal dispatch
        # Hook runs BEFORE auth so plugins can handle unauthorized senders
        # (e.g. customer handover ingest) without triggering the pairing flow.
        if not is_internal:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _hook_results = _invoke_hook(
                    "pre_gateway_dispatch",
                    event=event,
                    gateway=self,
                    session_store=self.session_store,
                )
            except Exception as _hook_exc:
                logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
                _hook_results = []

            for _result in _hook_results:
                if not isinstance(_result, dict):
                    continue
                _action = _result.get("action")
                if _action == "skip":
                    logger.info(
                        "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                        _result.get("reason"),
                        source.platform.value if source.platform else "unknown",
                        source.chat_id or "unknown",
                    )
                    return None
                if _action == "rewrite":
                    _new_text = _result.get("text")
                    if isinstance(_new_text, str):
                        event = dataclasses.replace(event, text=_new_text)
                        source = event.source
                    break
                if _action == "allow":
                    break

        if is_internal:
            pass
        elif source.user_id is None:
            # Messages with no user identity (Telegram service messages,
            # channel forwards, anonymous admin posts, sender_chat) can't
            # be paired, but they can still be authorized via a
            # chat-scoped allowlist (e.g. TELEGRAM_GROUP_ALLOWED_CHATS
            # authorizes every member of the listed chat regardless of
            # sender). Defer to _is_user_authorized so that path runs.
            if not self._is_user_authorized(source):
                logger.debug("Ignoring message with no user_id from %s", source.platform.value)
                return None
        elif not self._is_user_authorized(source):
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # In DMs: offer pairing code. In groups: silently ignore.
            if (
                source.chat_type == "dm"
                and self._get_unauthorized_dm_behavior(
                    source.platform,
                    profile=source.profile,
                )
                == "pair"
            ):
                platform_name = source.platform.value if source.platform else "unknown"
                # Rate-limit ALL pairing responses (code or rejection) to
                # prevent spamming the user with repeated messages when
                # multiple DMs arrive in quick succession.
                if self.pairing_store._is_rate_limited(platform_name, source.user_id):
                    return None
                code = self.pairing_store.generate_code(
                    platform_name, source.user_id, source.user_name or ""
                )
                if code:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            f"Hi~ I don't recognize you yet!\n\n"
                            f"Here's your pairing code: `{code}`\n\n"
                            f"Ask the bot owner to run:\n"
                            f"`hermes pairing approve {platform_name} {code}`"
                        )
                else:
                    adapter = self._adapter_for_source(source)
                    if adapter:
                        await adapter.send(
                            source.chat_id,
                            "Too many pairing requests right now~ "
                            "Please try again later!"
                        )
                    # Record rate limit so subsequent messages are silently ignored
                    self.pairing_store._record_rate_limit(platform_name, source.user_id)
            return None
        
        # Intercept messages that are responses to a pending /update prompt.
        # The update process (detached) wrote .update_prompt.json; the watcher
        # forwarded it to the user; now the user's reply goes back via
        # .update_response so the update process can continue.
        #
        # IMPORTANT: recognized slash commands must bypass this interception.
        # Otherwise control/session commands like /new or /help get silently
        # consumed as update answers instead of being dispatched normally.
        _quick_key = self._session_key_for_source(source)
        _update_prompts = getattr(self, "_update_prompt_pending", {})
        if _update_prompts.get(_quick_key):
            raw = (event.text or "").strip()
            # Accept /approve and /deny as shorthand for yes/no
            cmd = event.get_command()
            if cmd in {"approve", "yes"}:
                response_text = "y"
            elif cmd in {"deny", "no"}:
                response_text = "n"
            else:
                _recognized_cmd = None
                if cmd:
                    try:
                        from hermes_cli.commands import resolve_command as _resolve_update_cmd
                    except Exception:
                        _resolve_update_cmd = None
                    if _resolve_update_cmd is not None:
                        try:
                            _cmd_def = _resolve_update_cmd(cmd)
                            _recognized_cmd = _cmd_def.name if _cmd_def else None
                        except Exception:
                            _recognized_cmd = None
                if _recognized_cmd:
                    response_text = ""
                else:
                    response_text = raw
            if response_text:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text(response_text)
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("Failed to write update response: %s", e)
                    return f"✗ Failed to send response to update process: {e}"
                _update_prompts.pop(_quick_key, None)
                label = response_text if len(response_text) <= 20 else response_text[:20] + "…"
                return f"✓ Sent `{label}` to the update process."
            # Recognized slash command during a pending update prompt:
            # unblock the detached update subprocess by writing a blank
            # response so ``_gateway_prompt`` returns the prompt's default
            # (typically a safe "n" / skip) and exits cleanly instead of
            # blocking on stdin until the 30-minute watcher timeout.
            # The slash command then falls through to normal dispatch.
            if _recognized_cmd:
                response_path = _hermes_home / ".update_response"
                prompt_path = _hermes_home / ".update_prompt.json"
                try:
                    tmp = response_path.with_suffix(".tmp")
                    tmp.write_text("")
                    tmp.replace(response_path)
                    prompt_path.unlink(missing_ok=True)
                    logger.info(
                        "Recognized /%s during pending update prompt for %s; "
                        "cancelled prompt with default and dispatching command",
                        _recognized_cmd,
                        _quick_key,
                    )
                except OSError as e:
                    logger.warning(
                        "Failed to write cancel response for pending update prompt: %s",
                        e,
                    )
                _update_prompts.pop(_quick_key, None)

        # Intercept messages that are responses to a pending clarify.
        # Open-ended prompts and "Other" responses are captured as free text;
        # direct replies to multi-choice prompts are accepted too ("2" maps
        # to the second option, arbitrary text becomes a custom answer). Slash
        # commands still bypass this path so /stop and friends keep working.
        _clarify_mod = None
        try:
            from tools import clarify_gateway as _clarify_mod
            _pending_clarify = _clarify_mod.get_pending_for_session(
                _quick_key, include_choice_prompts=True,
            )
        except Exception:
            _pending_clarify = None
        if _pending_clarify is not None and _clarify_mod is not None:
            _raw_clarify_reply = (event.text or "").strip()
            # Skip slash commands — the user clearly wanted to issue a
            # command, not answer the clarify.  Leave the clarify pending
            # so the user can retry; if it times out, the agent unblocks
            # with an empty response.
            if _raw_clarify_reply and not _raw_clarify_reply.startswith("/"):
                _resolved = _clarify_mod.resolve_text_response_for_session(
                    _quick_key, _raw_clarify_reply,
                )
                if _resolved:
                    logger.info(
                        "Gateway intercepted clarify text response (session=%s, id=%s)",
                        _quick_key, _pending_clarify.clarify_id,
                    )
                    # Acknowledge with empty string so adapters that emit
                    # the agent's response don't double-post.  The agent
                    # itself will produce the next user-facing message.
                    return ""

        # Intercept messages that are responses to a pending /reload-mcp
        # (or future) slash-confirm prompt.  Recognized confirm replies are
        # /approve, /always, /cancel (plus short aliases).  Anything else
        # falls through to normal dispatch — a stale pending confirm does
        # NOT block other commands.
        #
        # Important: if a dangerous-command approval is ALSO pending (agent
        # blocked inside tools/approval.py), the tool approval takes
        # precedence — /approve there unblocks the waiting tool thread.
        # Slash-confirm only catches /approve when no tool approval is live.
        from tools import slash_confirm as _slash_confirm_mod
        _pending_confirm = _slash_confirm_mod.get_pending(_quick_key)
        _tool_approval_live = False
        try:
            from tools.approval import has_blocking_approval
            _tool_approval_live = has_blocking_approval(_quick_key)
        except Exception:
            _tool_approval_live = False
        if _pending_confirm and not _tool_approval_live:
            _raw_reply = (event.text or "").strip()
            # Accept bang-prefixed replies (`!always`, `!cancel`) verbatim.
            # Slack/Matrix instruction text shows the `!` prefix (typed `/`
            # is blocked in Slack threads), but the adapters only rewrite
            # `!<known-command>` — `always`/`cancel` are confirm keywords,
            # not registered commands, so the `!` survives to here.
            _norm_reply = _raw_reply.lstrip("!/").lower()
            _cmd_reply = event.get_command()
            _confirm_choice = None
            if _cmd_reply in {"approve", "yes", "ok", "confirm"}:
                _confirm_choice = "once"
            elif _cmd_reply in {"always", "remember"}:
                _confirm_choice = "always"
            elif _cmd_reply in {"cancel", "no", "deny", "nevermind"}:
                _confirm_choice = "cancel"
            elif _norm_reply in {"approve", "approve once", "once"}:
                _confirm_choice = "once"
            elif _norm_reply in {"always", "always approve"}:
                _confirm_choice = "always"
            elif _norm_reply in {"cancel", "nevermind", "no"}:
                _confirm_choice = "cancel"
            if _confirm_choice is not None:
                _resolved = await _slash_confirm_mod.resolve(
                    _quick_key, _pending_confirm.get("confirm_id"), _confirm_choice,
                )
                return _resolved or ""
            # Stale pending + unrelated command: drop the pending state so
            # the confirm doesn't block normal usage indefinitely.  The user
            # clearly moved on.
            _slash_confirm_mod.clear_if_stale(_quick_key)

        # PRIORITY handling when an agent is already running for this session.
        # Default behavior is to interrupt immediately so user text/stop messages
        # are handled with minimal latency.
        #
        # Special case: Telegram/photo bursts often arrive as multiple near-
        # simultaneous updates. Do NOT interrupt for photo-only follow-ups here;
        # let the adapter-level batching/queueing logic absorb them.

        # Staleness eviction: detect leaked locks from hung/crashed handlers.
        # With inactivity-based timeout, active tasks can run for hours, so
        # wall-clock age alone isn't sufficient.  Evict only when the agent
        # has been *idle* beyond the inactivity threshold (or when the agent
        # object has no activity tracker and wall-clock age is extreme).
        _raw_stale_timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _stale_ts = self._running_agents_ts.get(_quick_key, 0)
        if _quick_key in self._running_agents and _stale_ts:
            _stale_age = time.time() - _stale_ts
            _stale_agent = self._running_agents.get(_quick_key)
            # Never evict the pending sentinel — it was just placed moments
            # ago during the async setup phase before the real agent is
            # created.  Sentinels have no get_activity_summary(), so the
            # idle check below would always evaluate to inf >= timeout and
            # immediately evict them, racing with the setup path.
            _stale_idle = float("inf")  # assume idle if we can't check
            _stale_detail = ""
            if _stale_agent and hasattr(_stale_agent, "get_activity_summary"):
                try:
                    _sa = _stale_agent.get_activity_summary()
                    _stale_idle = _sa.get("seconds_since_activity", float("inf"))
                    _stale_detail = (
                        f" | last_activity={_sa.get('last_activity_desc', 'unknown')} "
                        f"({_stale_idle:.0f}s ago) "
                        f"| iteration={_sa.get('api_call_count', 0)}/{_sa.get('max_iterations', 0)}"
                    )
                except Exception:
                    pass
            # Evict if: agent is idle beyond timeout, OR wall-clock age is
            # extreme (10x timeout or 2h, whichever is larger — catches
            # cases where the agent object was garbage-collected).
            _wall_ttl = max(_raw_stale_timeout * 10, 7200) if _raw_stale_timeout > 0 else float("inf")
            _should_evict = (
                _stale_agent is not _AGENT_PENDING_SENTINEL
                and (
                    (_raw_stale_timeout > 0 and _stale_idle >= _raw_stale_timeout)
                    or _stale_age > _wall_ttl
                )
            )
            if _should_evict:
                logger.warning(
                    "Evicting stale _running_agents entry for %s "
                    "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
                    _quick_key, _stale_age, _stale_idle,
                    _raw_stale_timeout, _stale_detail,
                )
                self._invalidate_session_run_generation(
                    _quick_key,
                    reason="stale_running_agent_eviction",
                )
                self._release_running_agent_state(_quick_key)

        if _quick_key in self._running_agents:
            if event.get_command() == "status":
                return await self._handle_status_command(event)

            # Resolve the command once for all early-intercept checks below.
            from hermes_cli.commands import (
                ACTIVE_SESSION_BYPASS_COMMANDS as _DEDICATED_HANDLERS,
                resolve_command as _resolve_cmd_inner,
            )
            _evt_cmd = event.get_command()
            _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

            # Slash command access control on the running-agent fast-path.
            # Mirrors the cold-path gate further below so non-admin users
            # can't bypass gating just because an agent happens to be busy.
            # /status above is intentionally pre-gate so users always see
            # session state. /help and /whoami fall under the always-allowed
            # floor inside _check_slash_access.
            if _evt_cmd and _cmd_def_inner is not None:
                _denied = self._check_slash_access(source, _cmd_def_inner.name)
                if _denied is not None:
                    return _denied

            # Telegram sends /start for bot launches/deep-links. Treat it as a
            # platform ping, not a user command: no help dump, no agent
            # interrupt, no queued text.
            if _cmd_def_inner and _cmd_def_inner.name == "start":
                logger.info("Ignoring /start platform ping for active session %s", _quick_key)
                return ""

            if _cmd_def_inner and _cmd_def_inner.name == "restart":
                return await self._handle_restart_command(event)

            # /stop must hard-kill the session when an agent is running.
            # A soft interrupt (agent.interrupt()) doesn't help when the agent
            # is truly hung — the executor thread is blocked and never checks
            # _interrupt_requested.  Force-clean _running_agents so the session
            # is unlocked and subsequent messages are processed normally.
            if _cmd_def_inner and _cmd_def_inner.name == "stop":
                await self._interrupt_and_clear_session(
                    _quick_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_STOP,
                    invalidation_reason="stop_command",
                )
                logger.info("STOP for session %s — agent interrupted, session lock released", _quick_key)
                return EphemeralReply(t("gateway.stop.stopped"))

            # /reset and /new must bypass the running-agent guard so they
            # actually dispatch as commands instead of being queued as user
            # text (which would be fed back to the agent with the same
            # broken history — #2170).  Interrupt the agent first, then
            # clear the adapter's pending queue so the stale "/reset" text
            # doesn't get re-processed as a user message after the
            # interrupt completes.
            if _cmd_def_inner and _cmd_def_inner.name == "new":
                # Clear any pending messages so the old text doesn't replay
                await self._interrupt_and_clear_session(
                    _quick_key,
                    source,
                    interrupt_reason=_INTERRUPT_REASON_RESET,
                    invalidation_reason="new_command",
                )
                # Clean up the running agent entry so the reset handler
                # doesn't think an agent is still active.
                return await self._handle_reset_command(event)

            # /queue <prompt> — queue without interrupting.
            # Semantics: each /queue invocation produces its own full agent
            # turn, processed in FIFO order after the current run (and any
            # earlier /queue items) finishes.  Messages are NOT merged.
            if event.get_command() in {"queue", "q"}:
                queued_text = event.get_command_args().strip()
                # Preserve media/reply payloads: a /queue carrying a photo,
                # document, or reply context is valid even with no prompt text
                # (e.g. "/queue" as the caption of an image). Dropping these
                # fields silently lost the attachment when the queued turn ran.
                has_media = bool(getattr(event, "media_urls", None))
                if not queued_text and not has_media:
                    return "Usage: /queue <prompt>"
                adapter = self.adapters.get(source.platform)
                if adapter:
                    queued_event = MessageEvent(
                        text=queued_text,
                        message_type=event.message_type if has_media else MessageType.TEXT,
                        source=event.source,
                        raw_message=event.raw_message,
                        message_id=event.message_id,
                        media_urls=list(getattr(event, "media_urls", []) or []),
                        media_types=list(getattr(event, "media_types", []) or []),
                        reply_to_message_id=event.reply_to_message_id,
                        reply_to_text=event.reply_to_text,
                        reply_to_author_id=event.reply_to_author_id,
                        reply_to_author_name=event.reply_to_author_name,
                        reply_to_is_own_message=event.reply_to_is_own_message,
                        auto_skill=event.auto_skill,
                        channel_prompt=event.channel_prompt,
                        internal=event.internal,
                        timestamp=event.timestamp,
                    )
                    self._enqueue_fifo(_quick_key, queued_event, adapter)
                depth = self._queue_depth(_quick_key, adapter=self.adapters.get(source.platform))
                if depth <= 1:
                    return "Queued for the next turn."
                return f"Queued for the next turn. ({depth} queued)"

            # /steer <prompt> — inject mid-run after the next tool call.
            # Unlike /queue (turn boundary), /steer lands BETWEEN tool-call
            # iterations inside the same agent run, by appending to the
            # last tool result's content. No interrupt, no new user turn,
            # no role-alternation violation.
            if _cmd_def_inner and _cmd_def_inner.name == "steer":
                steer_text = event.get_command_args().strip()
                if not steer_text:
                    return "Usage: /steer <prompt>"
                running_agent = self._running_agents.get(_quick_key)
                if running_agent is _AGENT_PENDING_SENTINEL:
                    # Agent hasn't started yet — queue as turn-boundary fallback.
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        queued_event = MessageEvent(
                            text=steer_text,
                            message_type=MessageType.TEXT,
                            source=event.source,
                            message_id=event.message_id,
                            channel_prompt=event.channel_prompt,
                        )
                        adapter._pending_messages[_quick_key] = queued_event
                    return "Agent still starting — /steer queued for the next turn."
                if running_agent and hasattr(running_agent, "steer"):
                    try:
                        accepted = running_agent.steer(steer_text)
                    except Exception as exc:
                        logger.warning("Steer failed for session %s: %s", _quick_key, exc)
                        return f"⚠️ Steer failed: {exc}"
                    if accepted:
                        preview = steer_text[:60] + ("..." if len(steer_text) > 60 else "")
                        return f"⏩ Steer queued — arrives after the next tool call: '{preview}'"
                    return "Steer rejected (empty payload)."
                # Running agent is missing or lacks steer() — fall back to queue.
                adapter = self.adapters.get(source.platform)
                if adapter:
                    queued_event = MessageEvent(
                        text=steer_text,
                        message_type=MessageType.TEXT,
                        source=event.source,
                        message_id=event.message_id,
                        channel_prompt=event.channel_prompt,
                    )
                    adapter._pending_messages[_quick_key] = queued_event
                return "No active agent — /steer queued for the next turn."

            # /model must not be used while the agent is running.
            if _cmd_def_inner and _cmd_def_inner.name == "model":
                return "Agent is running — wait or /stop first, then switch models."

            # /codex-runtime must not be used while the agent is running.
            # Switching mid-turn would split a turn across two transports.
            if _cmd_def_inner and _cmd_def_inner.name == "codex-runtime":
                return ("Agent is running — wait or /stop first, then "
                        "change runtime.")

            # /approve and /deny must bypass the running-agent interrupt path.
            # The agent thread is blocked on a threading.Event inside
            # tools/approval.py — sending an interrupt won't unblock it.
            # Route directly to the approval handler so the event is signalled.
            if _cmd_def_inner and _cmd_def_inner.name in {"approve", "deny"}:
                if _cmd_def_inner.name == "approve":
                    return await self._handle_approve_command(event)
                return await self._handle_deny_command(event)

            # /agents (/tasks alias) should be query-only and never interrupt.
            if _cmd_def_inner and _cmd_def_inner.name == "agents":
                return await self._handle_agents_command(event)

            # /background must bypass the running-agent guard — it starts a
            # parallel task and must never interrupt the active conversation.
            # /btw is an alias of /background and resolves to the same canonical
            # name, so this branch handles both commands.
            if _cmd_def_inner and _cmd_def_inner.name == "background":
                return await self._handle_background_command(event)

            # /kanban must bypass the guard. It writes to a profile-agnostic
            # DB (kanban.db), not to the running agent's state. In fact
            # /kanban unblock is often the only way to free a worker that
            # has blocked waiting for a peer — letting that be dispatched
            # mid-run is the whole point of the board.
            if _cmd_def_inner and _cmd_def_inner.name == "kanban":
                return await self._handle_kanban_command(event)

            # /goal is safe mid-run for status/pause/clear/wait (inspection
            # and control-plane only — doesn't interrupt the running turn).
            # Setting a new goal text mid-run is rejected with the same
            # "wait or /stop" message as /model so we don't race a second
            # continuation prompt against the current turn.
            if _cmd_def_inner and _cmd_def_inner.name == "goal":
                _goal_arg = (event.get_command_args() or "").strip().lower()
                _goal_verb = _goal_arg.split(None, 1)[0] if _goal_arg else ""
                # Exact-match control verbs (unchanged semantics), plus the
                # wait/unwait barrier verbs which take a pid argument.
                _is_control = (
                    not _goal_arg
                    or _goal_arg in {"status", "pause", "resume", "clear", "stop", "done", "unwait"}
                    or _goal_verb == "wait"
                )
                if _is_control:
                    return await self._handle_goal_command(event)
                return "Agent is running — use /goal status / pause / clear / wait mid-run, or /stop before setting a new goal."

            if _cmd_def_inner and _cmd_def_inner.name == "moa":
                return "Agent is running — wait or /stop first, then run /moa."

            # /subgoal is safe mid-run — it only modifies the goal's
            # subgoals list, which the judge reads at the next turn
            # boundary. No race with the running turn.
            if _cmd_def_inner and _cmd_def_inner.name == "subgoal":
                return await self._handle_subgoal_command(event)

            # Session-level toggles that are safe to run mid-agent —
            # /yolo can unblock a pending approval prompt, /verbose cycles
            # the tool-progress display mode for the ongoing stream.
            # Both modify session state without needing agent interaction
            # and must not be queued (the safety net would discard them).
            # /fast and /reasoning are config-only and take effect next
            # message, so they fall through to the catch-all busy response
            # below — users should wait and set them between turns.
            if _cmd_def_inner and _cmd_def_inner.name in {"yolo", "verbose"}:
                if _cmd_def_inner.name == "yolo":
                    return await self._handle_yolo_command(event)
                if _cmd_def_inner.name == "verbose":
                    return await self._handle_verbose_command(event)
                if _cmd_def_inner.name == "footer":
                    return await self._handle_footer_command(event)

            # Gateway-handled info/control commands with dedicated
            # running-agent handlers.
            if _cmd_def_inner and _cmd_def_inner.name in _DEDICATED_HANDLERS:
                if _cmd_def_inner.name == "help":
                    return await self._handle_help_command(event)
                if _cmd_def_inner.name == "commands":
                    return await self._handle_commands_command(event)
                if _cmd_def_inner.name == "profile":
                    return await self._handle_profile_command(event)
                if _cmd_def_inner.name == "update":
                    return await self._handle_update_command(event)
                if _cmd_def_inner.name == "version":
                    return await self._handle_version_command(event)

            # Catch-all: any other recognized slash command reached the
            # running-agent guard. Reject gracefully rather than falling
            # through to interrupt + discard. Without this, commands
            # like /model, /reasoning, /voice, /insights, /title,
            # /resume, /retry, /undo, /compress, /usage,
            # /reload-mcp, /sethome, /reset (all registered as Discord
            # slash commands) would interrupt the agent AND get
            # silently discarded by the slash-command safety net,
            # producing a zero-char response. See #5057, #6252, #10370.
            if _cmd_def_inner:
                return (
                    f"⏳ Agent is running — `/{_cmd_def_inner.name}` can't run "
                    f"mid-turn. Wait for the current response or `/stop` first."
                )

            if event.message_type == MessageType.PHOTO:
                logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key)
                adapter = self.adapters.get(source.platform)
                if adapter:
                    merge_pending_message_event(adapter._pending_messages, _quick_key, event)
                return None

            _telegram_followup_grace = float(
                os.getenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0")
            )
            _started_at = self._running_agents_ts.get(_quick_key, 0)
            if (
                source.platform == Platform.TELEGRAM
                and event.message_type == MessageType.TEXT
                and _telegram_followup_grace > 0
                and _started_at
                and (time.time() - _started_at) <= _telegram_followup_grace
            ):
                logger.debug(
                    "Telegram follow-up arrived %.2fs after run start for %s — queueing without interrupt",
                    time.time() - _started_at,
                    _quick_key,
                )
                adapter = self.adapters.get(source.platform)
                if adapter:
                    if self._busy_input_mode == "queue":
                        self._enqueue_fifo(_quick_key, event, adapter)
                    else:
                        merge_pending_message_event(
                            adapter._pending_messages,
                            _quick_key,
                            event,
                            merge_text=True,
                        )
                return None

            running_agent = self._running_agents.get(_quick_key)
            if running_agent is _AGENT_PENDING_SENTINEL:
                # Agent is being set up but not ready yet.
                if event.get_command() == "stop":
                    # Force-clean the sentinel so the session is unlocked.
                    self._release_running_agent_state(_quick_key)
                    logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key)
                    return EphemeralReply("⚡ Force-stopped. The agent was still starting — session unlocked.")
                # Queue the message so it will be picked up after the
                # agent starts.
                adapter = self.adapters.get(source.platform)
                if adapter:
                    merge_pending_message_event(
                        adapter._pending_messages,
                        _quick_key,
                        event,
                        merge_text=True,
                    )
                return None
            if self._draining:
                if self._queue_during_drain_enabled():
                    self._queue_or_replace_pending_event(_quick_key, event)
                return (
                    f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
                    if self._queue_during_drain_enabled()
                    else f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
                )
            if self._busy_input_mode == "queue":
                logger.debug("PRIORITY queue follow-up for session %s", _quick_key)
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            if self._busy_input_mode == "steer":
                # Steer mode: inject text into the running agent mid-run via
                # agent.steer().  Falls back to queue semantics if the payload
                # is empty, the agent lacks steer(), or steer() rejects.
                steer_text = (event.text or "").strip()
                steered = False
                if steer_text and hasattr(running_agent, "steer"):
                    try:
                        steered = bool(running_agent.steer(steer_text))
                    except Exception as exc:
                        logger.warning("PRIORITY steer failed for session %s: %s", _quick_key, exc)
                        steered = False
                if steered:
                    logger.debug("PRIORITY steer for session %s", _quick_key)
                    return None
                logger.debug("PRIORITY steer-fallback-to-queue for session %s", _quick_key)
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            # #30170 — Subagent protection (PRIORITY path). Same rationale
            # as ``_handle_active_session_busy_message``: an interrupt
            # cascades through ``_active_children`` and aborts in-flight
            # delegate_task work. Demote to queue semantics when the
            # parent is currently driving subagents so a conversational
            # follow-up doesn't destroy minutes of subagent progress.
            # /stop reaches its dedicated handler above, so the operator
            # still has a clean escape hatch.
            if self._agent_has_active_subagents(running_agent):
                logger.info(
                    "PRIORITY interrupt demoted to queue for session %s "
                    "because the running agent has active subagents (#30170)",
                    _quick_key,
                )
                self._queue_or_replace_pending_event(_quick_key, event)
                return None
            logger.debug("PRIORITY interrupt for session %s", _quick_key)
            running_agent.interrupt(event.text)
            # NOTE: self._pending_messages was write-only (never consumed).
            # The actual interrupt message is delivered via adapter._pending_messages
            # which is read by _run_agent. Removed to prevent unbounded growth.
            return None

        # Check for commands
        command = event.get_command()

        from hermes_cli.commands import (
            GATEWAY_KNOWN_COMMANDS,
            is_gateway_known_command,
            resolve_command as _resolve_cmd,
        )

        # Resolve aliases to canonical name so dispatch and hook names
        # don't depend on the exact alias the user typed.
        _cmd_def = _resolve_cmd(command) if command else None
        canonical = _cmd_def.name if _cmd_def else command

        # Expand alias quick commands before built-in dispatch so targets like
        # /model openai/gpt-5.5 --provider openrouter reach the /model handler.
        # Preserve built-in precedence; aliases only need early handling when
        # the typed command is not already known.
        if command and _cmd_def is None:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if isinstance(quick_commands, dict) and command in quick_commands:
                qcmd = quick_commands[command]
                if qcmd.get("type") == "alias":
                    target = qcmd.get("target", "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        _cmd_def = _resolve_cmd(command) if command else None
                        canonical = _cmd_def.name if _cmd_def else command

        # Per-platform slash command access control. Only kicks in when the
        # operator has set ``allow_admin_from`` for the source's scope (DM
        # vs group). When unset → backward-compat: every allowed user can
        # run every command. When set → non-admins can run only commands in
        # ``user_allowed_commands`` (plus the always-allowed floor: /help,
        # /whoami). Plain chat is unaffected — only slash commands gate.
        if command and canonical and is_gateway_known_command(canonical):
            _denied = self._check_slash_access(source, canonical)
            if _denied is not None:
                return _denied

        # Fire the ``command:<canonical>`` hook for any recognized slash
        # command — built-in OR plugin-registered. Handlers can return a
        # dict with ``{"decision": "deny" | "handled" | "rewrite", ...}``
        # to intercept dispatch before core handling runs. This replaces
        # the previous fire-and-forget emit(): return values are now
        # honored, but handlers that return nothing behave exactly as
        # before (telemetry-style hooks keep working).
        if command and is_gateway_known_command(canonical):
            raw_args = event.get_command_args().strip()
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "command": canonical,
                "raw_command": command,
                "args": raw_args,
                "raw_args": raw_args,
            }
            try:
                hook_results = await self.hooks.emit_collect(
                    f"command:{canonical}", hook_ctx
                )
            except Exception as _hook_err:
                logger.debug(
                    "command:%s hook dispatch failed (non-fatal): %s",
                    canonical, _hook_err,
                )
                hook_results = []

            for hook_result in hook_results:
                if not isinstance(hook_result, dict):
                    continue
                decision = str(hook_result.get("decision", "")).strip().lower()
                if not decision or decision == "allow":
                    continue
                if decision == "deny":
                    message = hook_result.get("message")
                    if isinstance(message, str) and message:
                        return message
                    return f"Command `/{command}` was blocked by a hook."
                if decision == "handled":
                    message = hook_result.get("message")
                    return message if isinstance(message, str) and message else None
                if decision == "rewrite":
                    new_command = str(
                        hook_result.get("command_name", "")
                    ).strip().lstrip("/")
                    if not new_command:
                        continue
                    new_args = str(hook_result.get("raw_args", "")).strip()
                    event.text = f"/{new_command} {new_args}".strip()
                    command = event.get_command()
                    _cmd_def = _resolve_cmd(command) if command else None
                    canonical = _cmd_def.name if _cmd_def else command
                    break

        if canonical == "new":
            if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
                return self._telegram_topic_root_new_message()
            async def _do_reset():
                return await self._handle_reset_command(event)
            return await self._maybe_confirm_destructive_slash(
                event=event,
                command="new",
                title="/new",
                detail=(
                    "This starts a fresh session and discards the current "
                    "conversation history."
                ),
                execute=_do_reset,
            )

        if canonical == "topic":
            return await self._handle_topic_command(event)
        
        if canonical == "help":
            return await self._handle_help_command(event)

        if canonical == "start":
            logger.info("Ignoring /start platform ping for session %s", _quick_key)
            return ""

        if canonical == "commands":
            return await self._handle_commands_command(event)
        
        if canonical == "profile":
            return await self._handle_profile_command(event)

        if canonical == "whoami":
            return await self._handle_whoami_command(event)

        if canonical == "status":
            return await self._handle_status_command(event)

        if canonical == "agents":
            return await self._handle_agents_command(event)

        if canonical == "platform":
            return await self._handle_platform_command(event)

        if canonical == "restart":
            return await self._handle_restart_command(event)
        
        if canonical == "stop":
            return await self._handle_stop_command(event)
        
        if canonical == "reasoning":
            return await self._handle_reasoning_command(event)

        if canonical == "memory":
            return await self._handle_memory_command(event)

        if canonical == "skills":
            return await self._handle_skills_command(event)

        if canonical == "learn":
            # Open-ended: rewrite the turn to a standards-guided prompt and fall
            # through to normal agent processing. The live agent gathers the
            # sources the user described (dirs via read_file, URLs via
            # web_extract, this conversation, pasted text) and authors the skill
            # via skill_manage. Mirrors the /blueprint fall-through so role
            # alternation is preserved. No engine, works on any backend.
            from agent.learn_prompt import build_learn_prompt

            _learn_req = event.get_command_args().strip()
            _ack = (
                "Learning a skill from what you described…"
                if _learn_req
                else "Learning a skill from this conversation…"
            )
            try:
                adapter = self.adapters.get(source.platform)
                if adapter:
                    _ack_meta = self._thread_metadata_for_source(source)
                    await adapter.send(str(source.chat_id), _ack, metadata=_ack_meta)
            except Exception:
                logger.debug("learn ack send failed", exc_info=True)
            try:
                event.text = build_learn_prompt(_learn_req)
                # fall through to agent processing
            except Exception:
                return "Could not start /learn — please try again."

        if canonical == "fast":
            return await self._handle_fast_command(event)

        if canonical == "verbose":
            return await self._handle_verbose_command(event)

        if canonical == "footer":
            return await self._handle_footer_command(event)

        if canonical == "yolo":
            return await self._handle_yolo_command(event)

        if canonical == "model":
            return await self._handle_model_command(event)

        if canonical == "codex-runtime":
            return await self._handle_codex_runtime_command(event)

        if canonical == "personality":
            return await self._handle_personality_command(event)

        if canonical == "kanban":
            return await self._handle_kanban_command(event)

        if canonical == "suggestions":
            return await self._handle_suggestions_command(event)

        if canonical == "blueprint":
            _blueprint_result = await self._handle_blueprint_command(event)
            _blueprint_seed = getattr(_blueprint_result, "agent_seed", None)
            if _blueprint_seed:
                # Blueprint matched — rewrite the turn to the seed and fall
                # through to _handle_message_with_agent so the agent asks the
                # user for each slot value conversationally and then calls the
                # cronjob tool (the /steer fall-through pattern). The seed
                # enters as a normal user turn, preserving role alternation.
                # Send the "Setting up X…" ack first so the user gets the same
                # immediate feedback CLI users see, instead of silence until
                # the agent's first question.
                _ack = getattr(_blueprint_result, "text", "") or ""
                if _ack:
                    try:
                        adapter = self.adapters.get(source.platform)
                        if adapter:
                            _ack_meta = self._thread_metadata_for_source(source)
                            await adapter.send(str(source.chat_id), _ack, metadata=_ack_meta)
                    except Exception:
                        logger.debug("blueprint ack send failed", exc_info=True)
                try:
                    event.text = _blueprint_seed
                except Exception:
                    return getattr(_blueprint_result, "text", "") or None
            else:
                return getattr(_blueprint_result, "text", "") or None

        if canonical == "retry":
            return await self._handle_retry_command(event)
        
        if canonical == "undo":
            async def _do_undo():
                return await self._handle_undo_command(event)
            _undo_n = 1
            _undo_raw = event.get_command_args().strip()
            if _undo_raw:
                try:
                    _undo_n = max(1, int(_undo_raw.split()[0]))
                except (ValueError, IndexError):
                    _undo_n = 1
            _undo_detail = (
                "This removes the last user/assistant exchange from history."
                if _undo_n == 1
                else f"This removes the last {_undo_n} user turns from history."
            )
            return await self._maybe_confirm_destructive_slash(
                event=event,
                command="undo",
                title="/undo",
                detail=_undo_detail,
                execute=_do_undo,
            )
        
        if canonical == "sethome":
            return await self._handle_set_home_command(event)

        if canonical == "compress":
            return await self._handle_compress_command(event)

        if canonical == "usage":
            return await self._handle_usage_command(event)

        if canonical == "credits":
            return await self._handle_credits_command(event)

        if canonical == "insights":
            return await self._handle_insights_command(event)

        if canonical == "reload-mcp":
            return await self._handle_reload_mcp_command(event)

        if canonical == "reload-skills":
            return await self._handle_reload_skills_command(event)

        if canonical == "bundles":
            return await self._handle_bundles_command(event)

        if canonical == "approve":
            return await self._handle_approve_command(event)

        if canonical == "deny":
            return await self._handle_deny_command(event)

        if canonical == "update":
            return await self._handle_update_command(event)

        if canonical == "version":
            return await self._handle_version_command(event)

        if canonical == "debug":
            return await self._handle_debug_command(event)

        if canonical == "title":
            return await self._handle_title_command(event)

        if canonical == "resume":
            return await self._handle_resume_command(event)

        if canonical == "sessions":
            return await self._handle_sessions_command(event)

        if canonical == "branch":
            return await self._handle_branch_command(event)

        if canonical == "rollback":
            return await self._handle_rollback_command(event)

        if canonical == "background":
            return await self._handle_background_command(event)

        if canonical == "steer":
            # No active agent — /steer has no tool call to inject into.
            # Strip the prefix so downstream treats it as a normal user
            # message. If the payload is empty, surface the usage hint.
            steer_payload = event.get_command_args().strip()
            if not steer_payload:
                return "Usage: /steer <prompt>  (no agent is running; sending as a normal message)"
            try:
                event.text = steer_payload
            except Exception:
                pass
            # Do NOT return — fall through to _handle_message_with_agent
            # at the end of this function so the rewritten text is sent
            # to the agent as a regular user turn.

        if canonical == "goal":
            return await self._handle_goal_command(event)

        if canonical == "moa":
            # /moa is one-shot sugar only: run a single prompt through the
            # default MoA preset, then restore the prior model. To *switch* to a
            # MoA preset for the session, pick it from the model picker (MoA
            # presets surface as a virtual "Mixture of Agents" provider).
            from hermes_cli.moa_config import (
                moa_usage,
                normalize_moa_config,
            )
            from hermes_cli.config import load_config

            moa_payload = event.get_command_args().strip()
            if not moa_payload:
                return moa_usage()
            try:
                cfg = load_config()
                moa_cfg = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
            except Exception:
                moa_cfg = normalize_moa_config({})
            preset = moa_cfg["default_preset"]
            try:
                event.text = moa_payload
                event._moa_restore_override = self._session_model_overrides.get(_quick_key)
                self._session_model_overrides[_quick_key] = {
                    "provider": "moa",
                    "model": preset,
                    "base_url": "moa://local",
                    "api_key": "moa-virtual-provider",
                    "api_mode": "chat_completions",
                }
                self._evict_cached_agent(_quick_key)
                event._moa_disable_after_turn = True
            except Exception:
                return "Failed to prepare MoA turn."

        if canonical == "subgoal":
            return await self._handle_subgoal_command(event)

        if canonical == "voice":
            return await self._handle_voice_command(event)

        if self._draining:
            return f"⏳ Gateway is {self._status_action_gerund()} and is not accepting new work right now."

        # User-defined quick commands (bypass agent loop, no LLM call)
        if command:
            if isinstance(self.config, dict):
                quick_commands = self.config.get("quick_commands", {}) or {}
            else:
                quick_commands = getattr(self.config, "quick_commands", {}) or {}
            if not isinstance(quick_commands, dict):
                quick_commands = {}
            if command in quick_commands:
                # Quick commands are slash capabilities too — and type:exec
                # ones run a shell command in the gateway process. The early
                # gate above only fires for registry-known commands, so quick
                # commands (never in the registry) would otherwise reach this
                # dispatch sink unchecked. Apply the same admin/user policy to
                # the raw typed name here so non-admins can't invoke admin-only
                # quick commands. (#44727)
                _denied = self._check_slash_access(source, command)
                if _denied is not None:
                    return _denied
                qcmd = quick_commands[command]
                if qcmd.get("type") == "exec":
                    exec_cmd = qcmd.get("command", "")
                    if exec_cmd:
                        try:
                            # Sanitize env to prevent credential leakage —
                            # quick commands run in the gateway process which
                            # has all API keys in os.environ.
                            from tools.environments.local import _sanitize_subprocess_env
                            sanitized_env = _sanitize_subprocess_env(os.environ.copy())
                            proc = await asyncio.create_subprocess_shell(
                                exec_cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.PIPE,
                                env=sanitized_env,
                            )
                            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                            output = (stdout or stderr).decode().strip()
                            # Redact any remaining sensitive patterns in output
                            if output:
                                from agent.redact import redact_sensitive_text
                                output = redact_sensitive_text(output)
                            return output if output else "Command returned no output."
                        except asyncio.TimeoutError:
                            return "Quick command timed out (30s)."
                        except Exception as e:
                            return f"Quick command error: {e}"
                    else:
                        return f"Quick command '/{command}' has no command defined."
                elif qcmd.get("type") == "alias":
                    target = qcmd.get("target", "").strip()
                    if target:
                        target = target if target.startswith("/") else f"/{target}"
                        target_command = target.lstrip("/")
                        user_args = event.get_command_args().strip()
                        event.text = f"{target} {user_args}".strip()
                        command = target_command.split()[0] if target_command else target_command
                        # Fall through to normal command dispatch below
                    else:
                        return f"Quick command '/{command}' has no target defined."
                else:
                    return f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias')."

        # Plugin-registered slash commands
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                # Normalize underscores to hyphens so Telegram's underscored
                # autocomplete form matches plugin commands registered with
                # hyphens. See hermes_cli/commands.py:_build_telegram_menu.
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    user_args = event.get_command_args().strip()
                    result = plugin_handler(user_args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result) if result else None
            except Exception as e:
                logger.warning("Plugin command dispatch failed: %s", e)

        # Skill slash commands: /skill-name loads the skill and sends to agent.
        # resolve_skill_command_key() handles the Telegram underscore/hyphen
        # round-trip so /claude_code from Telegram autocomplete still resolves
        # to the claude-code skill.
        if command:
            # Skill bundles take precedence over individual skill commands —
            # /<bundle> loads multiple skills at once. Mirrors CLI dispatch.
            _bundle_handled = False
            try:
                from agent.skill_bundles import (
                    build_bundle_invocation_message,
                    resolve_bundle_command_key,
                )
                bundle_key = resolve_bundle_command_key(command)
                if bundle_key is not None:
                    user_instruction = event.get_command_args().strip()
                    bundle_result = build_bundle_invocation_message(
                        bundle_key, user_instruction, task_id=_quick_key
                    )
                    if bundle_result:
                        msg, _loaded, missing = bundle_result
                        event.text = msg
                        _bundle_handled = True
                        if missing:
                            logger.info(
                                "Bundle %s skipped missing skills: %s",
                                bundle_key, ", ".join(missing),
                            )
                        # Fall through to normal message processing with bundle content
            except Exception as exc:
                logger.warning("Bundle dispatch failed: %s", exc)

        if command and not locals().get("_bundle_handled", False):
            try:
                from agent.skill_commands import (
                    get_skill_commands,
                    build_skill_invocation_message,
                    resolve_skill_command_key,
                )
                skill_cmds = get_skill_commands()
                cmd_key = resolve_skill_command_key(command)
                if cmd_key is not None:
                    # Check per-platform disabled status before executing.
                    # get_skill_commands() only applies the *global* disabled
                    # list at scan time; per-platform overrides need checking
                    # here because the cache is process-global across platforms.
                    _skill_name = skill_cmds[cmd_key].get("name", "")
                    _plat = source.platform.value if source.platform else None
                    if _plat and _skill_name:
                        from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                        if _skill_name in _get_plat_disabled(platform=_plat):
                            return (
                                f"The **{_skill_name}** skill is disabled for {_plat}.\n"
                                f"Enable it with: `hermes skills config`"
                            )
                    user_instruction = event.get_command_args().strip()
                    msg = build_skill_invocation_message(
                        cmd_key, user_instruction, task_id=_quick_key
                    )
                    if msg:
                        event.text = msg
                        # Fall through to normal message processing with skill content
                else:
                    # Not an active skill — check if it's a known-but-disabled or
                    # uninstalled skill and give actionable guidance.
                    _unavail_msg = _check_unavailable_skill(command)
                    if _unavail_msg:
                        return _unavail_msg
                    # Genuinely unrecognized /command: not a built-in, not a
                    # plugin, not a skill, not a known-inactive skill. Warn
                    # the user instead of silently forwarding it to the LLM
                    # as free text (which leads to silent-failure behavior
                    # like the model inventing a delegate_task call).
                    # Normalize to hyphenated form before checking known
                    # built-ins (command may be an alias target set by the
                    # quick-command block above, so _cmd_def can be stale).
                    if command.replace("_", "-") not in GATEWAY_KNOWN_COMMANDS:
                        logger.warning(
                            "Unrecognized slash command /%s from %s — "
                            "replying with unknown-command notice",
                            command,
                            source.platform.value if source.platform else "?",
                        )
                        return (
                            f"Unknown command `/{command}`. "
                            f"Type /commands to see what's available, "
                            f"or resend without the leading slash to send "
                            f"as a regular message."
                        )
            except Exception as e:
                logger.debug("Skill command check failed (non-fatal): %s", e)
        
        # Pending exec approvals are handled by /approve and /deny commands above.
        # No bare text matching — "yes" in normal conversation must not trigger
        # execution of a dangerous command.

        if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
            # Debounce the lobby reminder so a user who forgets about
            # topic mode and fires ten prompts doesn't get ten copies.
            if self._should_send_telegram_lobby_reminder(source):
                return self._telegram_topic_root_lobby_message()
            return None

        # ── External-drain new-turn gate (Phase 2) ────────────────────
        # When NAS has engaged an external drain (.drain_request.json present,
        # observed by _drain_control_watcher), refuse to START a new turn so
        # the in-flight set can only fall to zero — eliminating the TOCTOU race
        # (D4a: stop accepting new turns FIRST, then NAS polls until
        # active_agents==0). In-flight turns are untouched; this only blocks the
        # claim of a NEW session slot. Internal/system events (restart-recovery
        # replays, background-process completions) bypass the gate — they are
        # not user-initiated new work and must still flow during a drain.
        # Reversible: once the marker is removed the gate opens again.
        if self._external_drain_active and not is_internal:
            logger.info(
                "Refusing new turn for session %s — external drain active.",
                _quick_key,
            )
            return (
                "⏳ This agent is draining for a maintenance action and isn't "
                "accepting new turns right now. It'll be back in a moment — "
                "please resend shortly."
            )

        # ── Claim this session before any await ───────────────────────
        # Between here and _run_agent registering the real AIAgent, there
        # are numerous await points (hooks, vision enrichment, STT,
        # session hygiene compression).  Without this sentinel a second
        # message arriving during any of those yields would pass the
        # "already running" guard and spin up a duplicate agent for the
        # same session — corrupting the transcript.
        _active_session_lease, _limit_message = self._claim_active_session_slot(
            _quick_key,
            source,
        )
        if _limit_message is not None:
            logger.info(
                "Rejecting new active session %s: max_concurrent_sessions reached",
                _quick_key,
            )
            return _limit_message
        if _active_session_lease is not None:
            if not hasattr(self, "_active_session_leases"):
                self._active_session_leases = {}
            self._active_session_leases[_quick_key] = _active_session_lease
        self._running_agents[_quick_key] = _AGENT_PENDING_SENTINEL
        self._running_agents_ts[_quick_key] = time.time()
        self._persist_active_agents()
        _run_generation = self._begin_session_run_generation(_quick_key)

        try:
            _agent_result = await self._handle_message_with_agent(event, source, _quick_key, _run_generation)
            # Goal continuation: after the agent returns a final response
            # for this turn, check any standing /goal — the judge will
            # either mark it done, pause it (budget), or enqueue a
            # continuation prompt back through the adapter FIFO so the
            # next turn makes more progress. Wrapped in try/except so a
            # broken judge never breaks normal message handling.
            try:
                _final_text = ""
                if isinstance(_agent_result, dict):
                    _final_text = str(_agent_result.get("final_response") or "")
                elif isinstance(_agent_result, str):
                    _final_text = _agent_result
                # Skip for empty responses (interrupted / errored) — the
                # judge would almost always say "continue" and we'd loop
                # on error. Let the user drive the next turn.
                if _final_text.strip():
                    try:
                        session_entry = self.session_store.get_or_create_session(source)
                    except Exception:
                        session_entry = None
                    if session_entry is not None:
                        await self._post_turn_goal_continuation(
                            session_entry=session_entry,
                            source=source,
                            final_response=_final_text,
                        )
            except Exception as _goal_exc:
                logger.debug("goal continuation hook failed: %s", _goal_exc)
            return _agent_result
        finally:
            # MoA one-shot restore must run on EVERY exit path, not just
            # success. The restore data lives on the per-turn event object
            # (_moa_restore_override), which is discarded once the event goes
            # out of scope — so if _handle_message_with_agent raises, a restore
            # in the try block would be skipped and the MoA override would leak
            # permanently (every later message silently fans out through MoA).
            # Putting it in finally guarantees the revert on success, exception,
            # and interrupt alike.
            self._restore_moa_one_shot(event, _quick_key)
            # Unconditional release covers every exit path. _release_running_agent_state
            # is idempotent (pop-on-absent is harmless) and, called without a
            # run_generation guard, always clears the slot regardless of which
            # generation it holds. This evicts the zombie left when session_reset
            # bumps the generation (N -> N+1) mid-flight: gen-N's guarded release
            # inside _run_agent returns False, and the old sentinel-only check here
            # missed the leftover real agent — locking the session out forever (#28686).
            self._release_running_agent_state(_quick_key)

    def _restore_moa_one_shot(self, event: "MessageEvent", quick_key: str) -> None:
        """Revert a ``/moa <prompt>`` one-shot model override after its turn.

        Called from the ``finally`` of the message-handling path so the revert
        fires whether the turn succeeded, raised, or was interrupted. A no-op
        unless ``event._moa_disable_after_turn`` is set. ``_moa_restore_override``
        carries the prior per-session override (``None`` means the user had no
        override, so the MoA override is cleared outright).
        """
        if not getattr(event, "_moa_disable_after_turn", False):
            return
        try:
            _restore = getattr(event, "_moa_restore_override", None)
            if _restore is None:
                self._session_model_overrides.pop(quick_key, None)
            else:
                self._session_model_overrides[quick_key] = _restore
            self._evict_cached_agent(quick_key)
        except Exception:
            pass

    async def _prepare_inbound_message_text(
        self,
        *,
        event: MessageEvent,
        source: SessionSource,
        history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Prepare inbound event text for the agent.

        Keep the normal inbound path and the queued follow-up path on the same
        preprocessing pipeline so sender attribution, image enrichment, STT,
        document notes, reply context, and @ references all behave the same.

        Side effect: buffers per-session native image paths when the active
        model supports native vision AND the user has images attached. The
        caller consumes and clears that session-scoped buffer at the
        ``run_conversation`` site to build a multimodal user turn. When the
        list is empty, the ``_enrich_message_with_vision`` text path has
        already run and images are represented in-text.
        """
        history = history or []
        message_text = event.text or ""
        _group_sessions_per_user = getattr(self.config, "group_sessions_per_user", True)
        _thread_sessions_per_user = getattr(self.config, "thread_sessions_per_user", False)
        # Use the same helper every other call site uses so the write key here
        # matches the consume key at the run_conversation site — even if the
        # session store overrides build_session_key's default behavior.
        session_key = self._session_key_for_source(source)
        # Reset only this session's per-call buffer; other sessions may be
        # concurrently preparing multimodal turns on the same runner.
        self._consume_pending_native_image_paths(session_key)

        _is_shared_multi_user = is_shared_multi_user_session(
            source,
            group_sessions_per_user=_group_sessions_per_user,
            thread_sessions_per_user=_thread_sessions_per_user,
        )
        if _is_shared_multi_user and source.user_name:
            message_text = f"[{source.user_name}] {message_text}"

        # Prepend channel context from history backfill (if any).  This
        # happens after sender-prefix so the prefix only applies to the
        # trigger message, not the backfill block.
        if getattr(event, "channel_context", None):
            message_text = f"{event.channel_context}\n\n[New message]\n{message_text}"

        # Declare at outer scope so the audio-file-paths handling block below
        # remains safe when ``event.media_urls`` is empty (no inner block runs).
        audio_file_paths: list[str] = []
        video_paths: list[str] = []

        if event.media_urls:
            image_paths = []
            audio_paths = []
            for i, path in enumerate(event.media_urls):
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                # Classify images per-attachment: trust this attachment's own
                # MIME, and only honour the message-level PHOTO type when the
                # per-attachment MIME is unknown. Otherwise a document (or any
                # non-image) sent alongside an image in the same message gets
                # mis-routed here as an image and the provider 400s.
                if _event_media_is_image(event, i):
                    image_paths.append(path)
                # MessageType.AUDIO = audio file attachment (e.g. .mp3, .m4a) — never STT
                # MessageType.VOICE = voice message (Opus/OGG) — always STT
                if event.message_type == MessageType.AUDIO:
                    audio_file_paths.append(path)
                elif event.message_type == MessageType.VOICE or (
                    mtype.startswith("audio/")
                    and event.message_type not in {MessageType.AUDIO, MessageType.DOCUMENT}
                ):
                    audio_paths.append(path)
                if mtype.startswith("video/") or (not mtype and event.message_type == MessageType.VIDEO):
                    video_paths.append(path)

            if image_paths:
                # Decide routing: native (attach pixels) vs text (vision_analyze
                # pre-run + prepend description).  See agent/image_routing.py.
                _img_mode = self._decide_image_input_mode()
                if _img_mode == "native":
                    # Defer attachment to the run_conversation call site.
                    pending_native = getattr(self, "_pending_native_image_paths_by_session", None)
                    if pending_native is None:
                        pending_native = {}
                        self._pending_native_image_paths_by_session = pending_native
                    pending_native[session_key] = list(image_paths)
                    logger.info(
                        "Image routing: native (model supports vision). %d image(s) will be attached inline.",
                        len(image_paths),
                    )
                else:
                    logger.info(
                        "Image routing: text (mode=%s). Pre-analyzing %d image(s) via vision_analyze.",
                        _img_mode, len(image_paths),
                    )
                    message_text = await self._enrich_message_with_vision(
                        message_text,
                        image_paths,
                    )

            if audio_paths:
                message_text, _successful_transcripts = await self._enrich_message_with_transcription(
                    message_text,
                    audio_paths,
                )
                # Echo each successful transcript back to the user immediately,
                # before the agent loop runs. Lets the user verify STT quality
                # in real-time and see the raw whisper output verbatim.
                if _successful_transcripts:
                    _echo_adapter = self.adapters.get(source.platform)
                    _echo_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))
                    if _echo_adapter:
                        for _tx in _successful_transcripts:
                            try:
                                await _echo_adapter.send(
                                    source.chat_id,
                                    f'🎙️ "{_tx}"',
                                    metadata=_echo_meta,
                                )
                            except Exception as _echo_exc:
                                logger.debug(
                                    "Transcript echo failed (non-fatal): %s", _echo_exc,
                                )
                # NOTE: Previously, when transcription failed (e.g. no STT
                # provider configured), the gateway also emitted a hardcoded
                # English notice via `_stt_adapter.send()`. That bypassed the
                # LLM and produced two replies — one pre-canned English clip
                # (which TTS then spoke aloud, in the wrong language) and one
                # correct, localized LLM reply from the enriched message text.
                # The enrichment step now leaves a single neutral marker in the
                # prompt, so the LLM produces one coherent reply in the user's
                # language. The hardcoded send has therefore been removed.

        if audio_file_paths:
            from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
            for _apath in audio_file_paths:
                _basename = os.path.basename(_apath)
                _parts = _basename.split("_", 2)
                _display = _parts[2] if len(_parts) >= 3 else _basename
                _display = re.sub(r'[^\w.\- ]', '_', _display)
                _agent_path = _to_agent_path(_apath)
                _note = (
                    f"[The user sent an audio file attachment: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the audio contains, transcribe or process it yourself — for "
                    f"example by passing the path to a transcription or media tool — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                )
                message_text = f"{_note}\n\n{message_text}"

        if video_paths:
            from tools.credential_files import to_agent_visible_cache_path as _to_agent_path
            for _vpath in video_paths:
                _basename = os.path.basename(_vpath)
                _parts = _basename.split("_", 2)
                _display = _parts[2] if len(_parts) >= 3 else _basename
                _display = re.sub(r'[^\w.\- ]', '_', _display)
                _agent_path = _to_agent_path(_vpath)
                _note = (
                    f"[The user sent a video attachment: '{_display}'. "
                    f"It is saved at: {_agent_path}. "
                    f"Its content is not inlined here. If the user's request involves "
                    f"what the video contains, inspect or process it yourself — for "
                    f"example by passing the path to a video analysis or media tool — "
                    f"instead of asking the user to describe it. Only ask what to do "
                    f"with it if their intent is genuinely unclear.]"
                )
                message_text = f"{_note}\n\n{message_text}"

        if event.media_urls:
            import mimetypes as _mimetypes
            from tools.credential_files import to_agent_visible_cache_path

            _TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
            for i, path in enumerate(event.media_urls):
                # Per-attachment document handling. Skip anything already routed
                # as image / audio / video by the buckets above — only genuine
                # non-media files get a path-pointing context note. This makes a
                # document mixed into a PHOTO/VOICE message (whole-message type
                # != DOCUMENT) still reach the agent as a readable cached file,
                # instead of being silently dropped because the message-level
                # type wasn't DOCUMENT.
                if (
                    _event_media_is_image(event, i)
                    or _event_media_is_audio(event, i)
                    or _event_media_is_video(event, i)
                ):
                    continue
                mtype = event.media_types[i] if i < len(event.media_types) else ""
                if mtype in {"", "application/octet-stream"}:
                    _ext = os.path.splitext(path)[1].lower()
                    if _ext in _TEXT_EXTENSIONS:
                        mtype = "text/plain"
                    else:
                        guessed, _ = _mimetypes.guess_type(path)
                        if guessed:
                            mtype = guessed
                        else:
                            mtype = "application/octet-stream"
                # Any accepted file gets a path-pointing context note — we accept
                # all file types now, so a non-text/non-application MIME (font/*,
                # model/*, etc.) must still tell the agent the file exists.

                basename = os.path.basename(path)
                parts = basename.split("_", 2)
                display_name = parts[2] if len(parts) >= 3 else basename
                display_name = re.sub(r'[^\w.\- ]', '_', display_name)

                # Translate host cache path to in-container path if running under Docker backend.
                # This ensures the agent receives a path it can open inside its sandbox, as the
                # cache directories are auto-mounted at /root/.hermes/cache/* by get_cache_directory_mounts().
                agent_path = to_agent_visible_cache_path(path)

                context_note = _build_document_context_note(display_name, agent_path, mtype)
                message_text = f"{context_note}\n\n{message_text}"

        # Discord: surface the triggering message id per-turn on the user
        # message rather than in the cached system prompt. message_id changes
        # every turn, so baking it into build_session_context_prompt() would
        # bust the agent-cache signature and rebuild the AIAgent every message
        # (destroying prompt caching). The static IDs block points the agent
        # here; the volatile id rides the per-turn user content.
        if (
            source is not None
            and getattr(source, "platform", None) == Platform.DISCORD
            and getattr(event, "message_id", None)
        ):
            from gateway.session import _discord_tools_loaded as _disc_tools_loaded
            if _disc_tools_loaded():
                message_text = (
                    f"[Triggering message id: `{event.message_id}` — use as "
                    f"`message_id` for reply/react/pin via the discord tools.]\n\n"
                    f"{message_text}"
                )

        if getattr(event, "reply_to_text", None) and event.reply_to_message_id:
            # Always inject the reply-to pointer — even when the quoted text
            # already appears in history. The prefix isn't deduplication, it's
            # disambiguation: it tells the agent *which* prior message the user
            # is referencing. History can contain the same or similar text
            # multiple times, and without an explicit pointer the agent has to
            # guess (or answer for both subjects). Token overhead is minimal.
            reply_snippet = event.reply_to_text[:500]
            if getattr(event, "reply_to_is_own_message", False):
                message_text = (
                    f'[Replying to your previous message: "{reply_snippet}"]\n\n'
                    f"{message_text}"
                )
            else:
                message_text = f'[Replying to: "{reply_snippet}"]\n\n{message_text}'

        if "@" in message_text:
            try:
                from agent.context_references import preprocess_context_references_async
                from agent.model_metadata import get_model_context_length_async

                _msg_cwd = os.environ.get("TERMINAL_CWD", os.path.expanduser("~"))
                _msg_runtime = _resolve_runtime_agent_kwargs()
                _msg_config_ctx = None
                try:
                    _msg_cfg = _load_gateway_config()
                    _msg_model_cfg = _msg_cfg.get("model", {})
                    if isinstance(_msg_model_cfg, dict):
                        _msg_raw_ctx = _msg_model_cfg.get("context_length")
                        if _msg_raw_ctx is not None:
                            _msg_config_ctx = int(_msg_raw_ctx)
                except Exception:
                    pass
                _msg_ctx_len = await get_model_context_length_async(
                    self._model,
                    base_url=self._base_url or _msg_runtime.get("base_url") or "",
                    api_key=_msg_runtime.get("api_key") or "",
                    config_context_length=_msg_config_ctx,
                )
                _ctx_result = await preprocess_context_references_async(
                    message_text,
                    cwd=_msg_cwd,
                    context_length=_msg_ctx_len,
                    allowed_root=_msg_cwd,
                )
                if _ctx_result.blocked:
                    _adapter = self.adapters.get(source.platform)
                    if _adapter:
                        await _adapter.send(
                            source.chat_id,
                            "\n".join(_ctx_result.warnings) or "Context injection refused.",
                        )
                    return None
                if _ctx_result.expanded:
                    message_text = _ctx_result.message
            except Exception as exc:
                logger.debug("@ context reference expansion failed: %s", exc)

        return message_text

    def _consume_pending_native_image_paths(self, session_key: str) -> List[str]:
        pending_native = getattr(self, "_pending_native_image_paths_by_session", None)
        if not pending_native:
            return []
        return list(pending_native.pop(session_key, []) or [])

    def _cache_session_source(self, session_key: str, source) -> None:
        if not session_key or source is None:
            return
        cached_sources = getattr(self, "_session_sources", None)
        if cached_sources is None:
            cached_sources = OrderedDict()
            self._session_sources = cached_sources
        try:
            cached_sources[session_key] = dataclasses.replace(source)
        except Exception:
            logger.debug("Failed to cache live session source for %s", session_key, exc_info=True)
            return
        # LRU: mark as most-recently-used and trim to max size.
        try:
            cached_sources.move_to_end(session_key)
            max_size = getattr(self, "_session_sources_max", 512)
            while len(cached_sources) > max_size:
                cached_sources.popitem(last=False)
        except Exception:
            pass

    def _get_cached_session_source(self, session_key: str):
        if not session_key:
            return None
        cached_sources = getattr(self, "_session_sources", None)
        if not cached_sources:
            return None
        source = cached_sources.get(session_key)
        if source is not None:
            try:
                cached_sources.move_to_end(session_key)
            except Exception:
                pass
        return source

    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        """Inner handler that runs under the _running_agents sentinel guard."""
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        _msg_preview = (event.text or "")[:80].replace("\n", " ")
        _reply_id = getattr(event, "reply_to_message_id", None)
        _reply_txt = (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " ")
        logger.info(
            "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r",
            _platform_name, source.user_name or source.user_id or "unknown",
            source.chat_id or "unknown", _msg_preview, _reply_id, _reply_txt,
        )

        # Get or create session
        # Topic-mode DMs: rewrite a stale/foreign thread_id to the user's
        # last-active topic so a cross-topic Reply or stripped plain reply
        # doesn't fragment the conversation across sessions.
        recovered = await asyncio.to_thread(self._recover_telegram_topic_thread_id, source)
        if recovered is not None:
            logger.info(
                "telegram topic recovery: chat=%s user=%s %r -> %s",
                source.chat_id, source.user_id, source.thread_id, recovered,
            )
            source = dataclasses.replace(source, thread_id=recovered)
            try:
                event.source = source
            except Exception:
                pass

        session_entry = self.session_store.get_or_create_session(source)
        session_key = session_entry.session_key
        self._cache_session_source(session_key, source)
        if await asyncio.to_thread(self._is_telegram_topic_lane, source):
            try:
                binding = (await self._session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )) if self._session_db else None
            except Exception:
                logger.debug("Failed to read Telegram topic binding", exc_info=True)
                binding = None
            if binding:
                bound_session_id = str(binding.get("session_id") or "")
                # Heal bindings that point at a pre-compression parent: walk
                # the compression-continuation chain forward to its tip so the
                # next message resumes the compressed child instead of
                # reloading the oversized parent transcript (#20470/#29712/
                # #33414). Returns the input unchanged when the session isn't
                # a compression parent, so this is cheap and safe.
                if bound_session_id and self._session_db is not None:
                    try:
                        canonical_session_id = await self._session_db.get_compression_tip(
                            bound_session_id,
                        )
                    except Exception:
                        logger.debug(
                            "compression-tip lookup failed for %s",
                            bound_session_id, exc_info=True,
                        )
                        canonical_session_id = bound_session_id
                    if (
                        canonical_session_id
                        and canonical_session_id != bound_session_id
                    ):
                        bound_session_id = canonical_session_id
                if bound_session_id and bound_session_id != session_entry.session_id:
                    # Route the override through SessionStore so the session_key
                    # → session_id mapping is persisted to disk and the previous
                    # lane session is ended cleanly. Mutating session_entry in
                    # place here created a split-brain state where the JSON
                    # index pointed at one id but code downstream used another.
                    switched = self.session_store.switch_session(session_key, bound_session_id)
                    if switched is not None:
                        session_entry = switched
                # If the stored binding pointed at a parent, rewrite it to the
                # canonical descendant now that we've followed the chain.
                if (
                    bound_session_id
                    and bound_session_id != str(binding.get("session_id") or "")
                ):
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-tip-walk",
                    )
            else:
                try:
                    await asyncio.to_thread(self._record_telegram_topic_binding, source, session_entry)
                except Exception:
                    logger.debug("Failed to record Telegram topic binding", exc_info=True)
        # Capture and immediately consume was_auto_reset so it does not
        # re-fire on subsequent messages — preventing the cleanup from
        # wiping model/reasoning overrides set between turns (Closes #48031).
        _was_auto_reset = getattr(session_entry, "was_auto_reset", False)
        if _was_auto_reset:
            # Treat auto-reset as a full conversation boundary — drop every
            # session-scoped transient state so the fresh session does not
            # inherit the previous conversation's model/reasoning overrides
            # or a queued "/model switched" note.
            self._session_model_overrides.pop(session_key, None)
            self._set_session_reasoning_override(session_key, None)
            if hasattr(self, "_pending_model_notes"):
                self._pending_model_notes.pop(session_key, None)
            # Evict the cached agent so the fresh session does not inherit the
            # previous conversation's context_compressor._previous_summary —
            # the cache is keyed on the stable session_key, so an auto-reset
            # otherwise reuses the old agent and leaks prior history into new
            # compaction summaries. Mirrors /reset and the compression-exhausted
            # path (#9893). Covers daily/idle/suspended auto-reset.
            self._evict_cached_agent(session_key)
            session_entry.was_auto_reset = False
        
        # Emit session:start for new or auto-reset sessions
        _is_new_session = (
            session_entry.created_at == session_entry.updated_at
            or _was_auto_reset
            or getattr(session_entry, "is_fresh_reset", False)
        )
        # Consume the is_fresh_reset flag immediately so it doesn't leak
        # onto subsequent messages in the same session (issue #6508).
        if getattr(session_entry, "is_fresh_reset", False):
            session_entry.is_fresh_reset = False
        if _is_new_session:
            await self.hooks.emit("session:start", {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "session_id": session_entry.session_id,
                "session_key": session_key,
            })
        
        # Build session context
        context = build_session_context(source, self.config, session_entry)
        
        # Set session context variables for tools (task-local, concurrency-safe)
        _session_env_tokens = self._set_session_env(context)
        
        # Read privacy.redact_pii from config (re-read per message)
        _redact_pii = False
        persist_user_message = None
        persist_user_timestamp = None
        try:
            _pcfg = _load_gateway_config()
            _redact_pii = bool((_pcfg.get("privacy") or {}).get("redact_pii", False))
        except Exception:
            pass

        # Build the context prompt to inject
        context_prompt = build_session_context_prompt(context, redact_pii=_redact_pii)
        
        # If the previous session expired and was auto-reset, prepend a notice
        # so the agent knows this is a fresh conversation (not an intentional /reset).
        if _was_auto_reset:
            reset_reason = getattr(session_entry, 'auto_reset_reason', None) or 'idle'
            if reset_reason == "suspended":
                context_note = "[System note: The user's previous session was stopped and suspended. This is a fresh conversation with no prior context.]"
            elif reset_reason == "daily":
                context_note = "[System note: The user's session was automatically reset by the daily schedule. This is a fresh conversation with no prior context.]"
            else:
                context_note = "[System note: The user's previous session expired due to inactivity. This is a fresh conversation with no prior context.]"
            context_prompt = context_note + "\n\n" + context_prompt

            # Send a user-facing notification explaining the reset, unless:
            # - notifications are disabled in config
            # - the platform is excluded (e.g. api_server, webhook)
            # - the expired session had no activity (nothing was cleared)
            try:
                policy = self.session_store.config.get_reset_policy(
                    platform=source.platform,
                    session_type=getattr(source, 'chat_type', 'dm'),
                )
                platform_name = source.platform.value if source.platform else ""
                had_activity = getattr(session_entry, 'reset_had_activity', False)
                # Suspended sessions always notify (they were explicitly stopped
                # or crashed mid-operation) — skip the policy check.
                should_notify = reset_reason == "suspended" or (
                    policy.notify
                    and had_activity
                    and platform_name not in policy.notify_exclude_platforms
                )
                if should_notify:
                    adapter = self.adapters.get(source.platform)
                    if adapter:
                        if reset_reason == "suspended":
                            reason_text = "previous session was stopped or interrupted"
                        elif reset_reason == "daily":
                            reason_text = f"daily schedule at {policy.at_hour}:00"
                        else:
                            hours = policy.idle_minutes // 60
                            mins = policy.idle_minutes % 60
                            duration = f"{hours}h" if not mins else f"{hours}h {mins}m" if hours else f"{mins}m"
                            reason_text = f"inactive for {duration}"
                        notice = (
                            f"◐ Session automatically reset ({reason_text}). "
                            f"Conversation history cleared.\n"
                            f"Use /resume to browse and restore a previous session.\n"
                            f"Adjust reset timing in config.yaml under session_reset."
                        )
                        try:
                            session_info = self._format_session_info()
                            if session_info:
                                notice = f"{notice}\n\n{session_info}"
                        except Exception:
                            pass
                        await adapter.send(
                            source.chat_id, notice,
                            metadata=self._thread_metadata_for_source(source),
                        )
            except Exception as e:
                logger.debug("Auto-reset notification failed (non-fatal): %s", e)

            # was_auto_reset is already consumed in the cleanup block above
            # (single source of truth); only the reset reason needs clearing here.
            session_entry.auto_reset_reason = None

        # Auto-load skill(s) for topic/channel bindings (Telegram DM Topics,
        # Discord channel_skill_bindings).  Supports a single name or ordered list.
        # Only inject on NEW sessions — ongoing conversations already have the
        # skill content in their conversation history from the first message.
        _auto = getattr(event, "auto_skill", None)
        if _is_new_session and _auto:
            _skill_names = [_auto] if isinstance(_auto, str) else list(_auto)
            try:
                from agent.skill_commands import _load_skill_payload, _build_skill_message
                _combined_parts: list[str] = []
                _loaded_names: list[str] = []
                for _sname in _skill_names:
                    _loaded = _load_skill_payload(_sname, task_id=_quick_key)
                    if _loaded:
                        _loaded_skill, _skill_dir, _display_name = _loaded
                        _note = (
                            f'[IMPORTANT: The "{_display_name}" skill is auto-loaded. '
                            f"Follow its instructions for this session.]"
                        )
                        _part = _build_skill_message(_loaded_skill, _skill_dir, _note)
                        if _part:
                            _combined_parts.append(_part)
                            _loaded_names.append(_sname)
                    else:
                        logger.warning("[Gateway] Auto-skill '%s' not found", _sname)
                if _combined_parts:
                    # Append the user's original text after all skill payloads
                    _combined_parts.append(event.text)
                    event.text = "\n\n".join(_combined_parts)
                    logger.info(
                        "[Gateway] Auto-loaded skill(s) %s for session %s",
                        _loaded_names, session_key,
                    )
            except Exception as e:
                logger.warning("[Gateway] Failed to auto-load skill(s) %s: %s", _skill_names, e)

        # Load conversation history from transcript
        history = self.session_store.load_transcript(session_entry.session_id)
        
        # -----------------------------------------------------------------
        # Session hygiene: auto-compress pathologically large transcripts
        #
        # Long-lived gateway sessions can accumulate enough history that
        # every new message rehydrates an oversized transcript, causing
        # repeated truncation/context failures.  Detect this early and
        # compress proactively — before the agent even starts.  (#628)
        #
        # Token source priority:
        # 1. Actual API-reported prompt_tokens from the last turn
        #    (stored in session_entry.last_prompt_tokens)
        # 2. Rough char-based estimate (str(msg)//4). Overestimates
        #    by 30-50% on code/JSON-heavy sessions, but that just
        #    means hygiene fires a bit early — safe and harmless.
        # -----------------------------------------------------------------
        if history and len(history) >= 4:
            from agent.model_metadata import (
                estimate_messages_tokens_rough,
                get_model_context_length_async,
            )

            # Read model + compression config from config.yaml.
            # NOTE: hygiene threshold is intentionally HIGHER than the agent's
            # own compressor (0.85 vs 0.50).  Hygiene is a safety net for
            # sessions that grew too large between turns — it fires pre-agent
            # to prevent API failures.  The agent's own compressor handles
            # normal context management during its tool loop with accurate
            # real token counts.  Having hygiene at 0.50 caused premature
            # compression on every turn in long gateway sessions.
            _hyg_model = "anthropic/claude-sonnet-4.6"
            _hyg_threshold_pct = 0.85
            _hyg_compression_enabled = True
            _hyg_hard_msg_limit = 5000
            _hyg_config_context_length = None
            _hyg_provider = None
            _hyg_base_url = None
            _hyg_api_key = None
            _hyg_data = {}
            try:
                _hyg_data = _load_gateway_config()
                if _hyg_data:
                    # Resolve model name (same logic as run_sync)
                    _model_cfg = _hyg_data.get("model", {})
                    if isinstance(_model_cfg, str):
                        _hyg_model = _model_cfg
                    elif isinstance(_model_cfg, dict):
                        _hyg_model = _model_cfg.get("default") or _model_cfg.get("model") or _hyg_model
                        # Read explicit context_length override from model config
                        # (same as run_agent.py lines 995-1005)
                        _raw_ctx = _model_cfg.get("context_length")
                        if _raw_ctx is not None:
                            try:
                                _hyg_config_context_length = int(_raw_ctx)
                            except (TypeError, ValueError):
                                pass
                        # Read provider for accurate context detection
                        _hyg_provider = _model_cfg.get("provider") or None
                        _hyg_base_url = _model_cfg.get("base_url") or None

                    # Read compression settings — only use enabled flag.
                    # The threshold is intentionally separate from the agent's
                    # compression.threshold (hygiene runs higher).
                    _comp_cfg = _hyg_data.get("compression", {})
                    if isinstance(_comp_cfg, dict):
                        _hyg_compression_enabled = str(
                            _comp_cfg.get("enabled", True)
                        ).lower() in {"true", "1", "yes"}
                        _raw_hard_limit = _comp_cfg.get("hygiene_hard_message_limit")
                        if _raw_hard_limit is not None:
                            try:
                                _parsed = int(_raw_hard_limit)
                                if _parsed > 0:
                                    _hyg_hard_msg_limit = _parsed
                            except (TypeError, ValueError):
                                pass

                try:
                    _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                        source=source,
                        session_key=session_key,
                        user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                    )
                    _hyg_provider = _hyg_runtime.get("provider") or _hyg_provider
                    _hyg_base_url = _hyg_runtime.get("base_url") or _hyg_base_url
                    _hyg_api_key = _hyg_runtime.get("api_key") or _hyg_api_key
                except Exception:
                    pass

                # Check custom_providers per-model context_length
                # (same fallback as run_agent.py lines 1171-1189).
                # Must run after runtime resolution so _hyg_base_url is set.
                if _hyg_config_context_length is None and _hyg_base_url:
                    try:
                        try:
                            from hermes_cli.config import get_compatible_custom_providers as _gw_gcp
                            _hyg_custom_providers = _gw_gcp(_hyg_data)
                        except Exception:
                            _hyg_custom_providers = _hyg_data.get("custom_providers")
                            if not isinstance(_hyg_custom_providers, list):
                                _hyg_custom_providers = []
                        for _cp in _hyg_custom_providers:
                            if not isinstance(_cp, dict):
                                continue
                            _cp_url = (_cp.get("base_url") or "").rstrip("/")
                            if _cp_url and _cp_url == _hyg_base_url.rstrip("/"):
                                _cp_models = _cp.get("models", {})
                                if isinstance(_cp_models, dict):
                                    _cp_model_cfg = _cp_models.get(_hyg_model, {})
                                    if isinstance(_cp_model_cfg, dict):
                                        _cp_ctx = _cp_model_cfg.get("context_length")
                                        if _cp_ctx is not None:
                                            _hyg_config_context_length = int(_cp_ctx)
                                break
                    except (TypeError, ValueError):
                        pass
            except Exception:
                pass

            if _hyg_compression_enabled:
                _hyg_context_length = await get_model_context_length_async(
                    _hyg_model,
                    base_url=_hyg_base_url or "",
                    api_key=_hyg_api_key or "",
                    config_context_length=_hyg_config_context_length,
                    provider=_hyg_provider or "",
                )
                _compress_token_threshold = int(
                    _hyg_context_length * _hyg_threshold_pct
                )
                _warn_token_threshold = int(_hyg_context_length * 0.95)

                _msg_count = len(history)

                # Prefer actual API-reported tokens from the last turn
                # (stored in session entry) over the rough char-based estimate.
                _stored_tokens = session_entry.last_prompt_tokens
                if _stored_tokens > 0:
                    _approx_tokens = _stored_tokens
                    _token_source = "actual"
                else:
                    _approx_tokens = estimate_messages_tokens_rough(history)
                    _token_source = "estimated"
                    # Note: rough estimates overestimate by 30-50% for code/JSON-heavy
                    # sessions, but that just means hygiene fires a bit early — which
                    # is safe and harmless.  The 85% threshold already provides ample
                    # headroom (agent's own compressor runs at 50%).  A previous 1.4x
                    # multiplier tried to compensate by inflating the threshold, but
                    # 85% * 1.4 = 119% of context — which exceeds the model's limit
                    # and prevented hygiene from ever firing for ~200K models (GLM-5).

                # Hard safety valve: force compression if message count is
                # extreme, regardless of token estimates.  This breaks the
                # death spiral where API disconnects prevent token data
                # collection, which prevents compression, which causes more
                # disconnects.  5000 messages is far above any normal session
                # but catches truly runaway growth before it becomes
                # unrecoverable.  Set well clear of legitimate large-context
                # (1M+) sessions doing thousands of short turns — those
                # compress on the token threshold, not this count-based floor.
                # Threshold is configurable via
                # compression.hygiene_hard_message_limit.
                # (#2153)
                _HARD_MSG_LIMIT = _hyg_hard_msg_limit
                _needs_compress = (
                    _approx_tokens >= _compress_token_threshold
                    or _msg_count >= _HARD_MSG_LIMIT
                )

                if _needs_compress:
                    logger.info(
                        "Session hygiene: %s messages, ~%s tokens (%s) — auto-compressing "
                        "(threshold: %s%% of %s = %s tokens)",
                        _msg_count, f"{_approx_tokens:,}", _token_source,
                        int(_hyg_threshold_pct * 100),
                        f"{_hyg_context_length:,}",
                        f"{_compress_token_threshold:,}",
                    )

                    _hyg_meta = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

                    try:
                        from run_agent import AIAgent

                        _hyg_model, _hyg_runtime = self._resolve_session_agent_runtime(
                            source=source,
                            session_key=session_key,
                            user_config=_hyg_data if isinstance(_hyg_data, dict) else None,
                        )
                        if _hyg_runtime.get("api_key"):
                            _hyg_msgs = [
                                {"role": m.get("role"), "content": m.get("content")}
                                for m in history
                                if m.get("role") in {"user", "assistant"}
                                and m.get("content")
                            ]

                            if len(_hyg_msgs) >= 4:
                                _hyg_agent = AIAgent(
                                    **_hyg_runtime,
                                    model=_hyg_model,
                                    max_iterations=4,
                                    quiet_mode=True,
                                    skip_memory=True,
                                    enabled_toolsets=["memory"],
                                    session_id=session_entry.session_id,
                                    session_db=getattr(self._session_db, "_db", self._session_db),
                                )
                                try:
                                    # The hygiene agent rotates the session
                                    # forward to a continuation id that becomes
                                    # the gateway session's live row. It must
                                    # never finalize on close() — close() would
                                    # end the newly rotated session the gateway
                                    # entry now points at.
                                    _hyg_agent._end_session_on_close = False
                                    _hyg_agent._print_fn = lambda *a, **kw: None

                                    loop = asyncio.get_running_loop()
                                    _compressed, _ = await loop.run_in_executor(
                                        None,
                                        lambda: _hyg_agent._compress_context(
                                            _hyg_msgs, "",
                                            approx_tokens=_approx_tokens,
                                        ),
                                    )

                                    # _compress_context ends the old session and creates
                                    # a new session_id.  Write compressed messages into
                                    # the NEW session so the old transcript stays intact
                                    # and searchable via session_search.
                                    _hyg_new_sid = _hyg_agent.session_id
                                    _hyg_rotated = _hyg_new_sid != session_entry.session_id
                                    _hyg_in_place = bool(
                                        getattr(_hyg_agent, "_last_compaction_in_place", False)
                                    )
                                    if _hyg_rotated:
                                        session_entry.session_id = _hyg_new_sid
                                        self.session_store._save()
                                        await asyncio.to_thread(
                                            self._sync_telegram_topic_binding,
                                            source, session_entry,
                                            reason="hygiene-compression",
                                        )

                                    # Only rewrite the transcript when rotation produced
                                    # a NEW session id OR in-place compaction succeeded.
                                    # The danger this guards against (mirrors the
                                    # /compress fix #44794/#39704): the hygiene agent is
                                    # built WITHOUT a session_db, so _compress_context
                                    # cannot rotate — if it also wasn't in-place, the
                                    # session_id is unchanged for a FAILURE reason, and an
                                    # unconditional rewrite_transcript() would DELETE the
                                    # original messages and replace them with only the
                                    # compressed summary (permanent data loss, #21301).
                                    if _hyg_rotated or _hyg_in_place:
                                        self.session_store.rewrite_transcript(
                                            session_entry.session_id, _compressed
                                        )
                                        # Reset stored token count — transcript rewritten
                                        session_entry.last_prompt_tokens = 0
                                        history = _compressed
                                        _new_count = len(_compressed)
                                        _new_tokens = estimate_messages_tokens_rough(
                                            _compressed
                                        )
                                    else:
                                        # No rewrite happened — transcript preserved
                                        # unchanged, so the post-compression counts equal
                                        # the pre-compression ones.
                                        _new_count = _msg_count
                                        _new_tokens = _approx_tokens
                                        logger.warning(
                                            "Gateway hygiene compression for session %s "
                                            "did not rotate or compact in place "
                                            "(no session_db on the hygiene agent) — "
                                            "preserving the original transcript instead "
                                            "of overwriting it with the summary (#21301).",
                                            session_entry.session_id,
                                        )

                                    logger.info(
                                        "Session hygiene: compressed %s → %s msgs, "
                                        "~%s → ~%s tokens",
                                        _msg_count, _new_count,
                                        f"{_approx_tokens:,}", f"{_new_tokens:,}",
                                    )

                                    if _new_tokens >= _warn_token_threshold:
                                        logger.warning(
                                            "Session hygiene: still ~%s tokens after "
                                            "compression",
                                            f"{_new_tokens:,}",
                                        )

                                    # If summary generation failed, the
                                    # compressor aborts entirely and returns
                                    # messages unchanged — nothing is dropped.
                                    # Surface a visible warning to the gateway
                                    # user — agent.log alone is invisible on
                                    # TG/Discord/etc. — so they know the chat
                                    # is "frozen" at the current size and can
                                    # /compress to retry or /reset to start
                                    # fresh.
                                    _comp = getattr(_hyg_agent, "context_compressor", None)
                                    if _comp is not None and getattr(_comp, "_last_compress_aborted", False):
                                        _err = getattr(_comp, "_last_summary_error", None) or "unknown error"
                                        _warn_msg = (
                                            "⚠️ Context compression aborted "
                                            f"({_err}). No messages were dropped — "
                                            "conversation is unchanged. Run /compress "
                                            "to retry, /reset for a clean session, or "
                                            "check your auxiliary.compression model "
                                            "configuration."
                                        )
                                        try:
                                            _adapter = self.adapters.get(source.platform)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _warn_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver compression-failure warning to user: %s",
                                                _werr,
                                            )
                                    # Separately: if the user's CONFIGURED aux
                                    # model failed and we recovered by falling
                                    # back to the main model, tell them — a
                                    # misconfigured auxiliary.compression.model
                                    # is something only they can fix, and
                                    # silent recovery would hide it.
                                    elif _comp is not None and getattr(_comp, "_last_aux_model_failure_model", None):
                                        _aux_model = getattr(_comp, "_last_aux_model_failure_model", "")
                                        _aux_err = getattr(_comp, "_last_aux_model_failure_error", None) or "unknown error"
                                        _aux_msg = (
                                            f"ℹ️ Configured compression model `{_aux_model}` "
                                            f"failed ({_aux_err}). Recovered using your main "
                                            "model — context is intact — but you may want to "
                                            "check `auxiliary.compression.model` in config.yaml."
                                        )
                                        try:
                                            _adapter = self.adapters.get(source.platform)
                                            if _adapter and source.chat_id:
                                                await _adapter.send(source.chat_id, _aux_msg, metadata=_hyg_meta)
                                        except Exception as _werr:
                                            logger.warning(
                                                "Failed to deliver aux-model-fallback notice to user: %s",
                                                _werr,
                                            )
                                finally:
                                    # Evict the cached agent so the next turn
                                    # rebuilds its system prompt from current
                                    # SOUL.md, memory, and skills.
                                    self._evict_cached_agent(session_key)
                                    await self._cleanup_agent_resources_off_loop(
                                        _hyg_agent, context="session hygiene"
                                    )

                    except Exception as e:
                        logger.warning(
                            "Session hygiene auto-compress failed: %s", e
                        )

        # First-message onboarding -- only on the very first interaction ever
        if not history and not self.session_store.has_any_sessions():
            # Default first-contact note: a brief self-introduction.
            _intro_note = (
                "\n\n[System note: This is the user's very first message ever. "
                "Briefly introduce yourself and mention that /help shows available commands. "
                "Keep the introduction concise -- one or two sentences max.]"
            )
            # Opt-in structured profile-build path. When enabled (default
            # "ask") and not yet offered on this install, swap the plain intro
            # for a consent-gated directive that offers to build a user
            # profile and persists confirmed facts via memory(target="user").
            # The offer fires at most once (onboarding.seen flag); set
            # onboarding.profile_build: off in config.yaml to disable.
            try:
                from agent.onboarding import (
                    PROFILE_BUILD_FLAG,
                    is_seen,
                    mark_seen,
                    profile_build_directive,
                    profile_build_mode,
                )
                _onb_cfg = _load_gateway_config()
                if (
                    profile_build_mode(_onb_cfg) == "ask"
                    and not is_seen(_onb_cfg, PROFILE_BUILD_FLAG)
                ):
                    context_prompt += profile_build_directive()
                    mark_seen(_hermes_home / "config.yaml", PROFILE_BUILD_FLAG)
                else:
                    context_prompt += _intro_note
            except Exception as _pb_err:
                logger.debug(
                    "Profile-build onboarding directive failed, using plain intro: %s",
                    _pb_err,
                )
                context_prompt += _intro_note
        
        # One-time prompt if no home channel is set for this platform
        # Skip for webhooks - they deliver directly to configured targets (github_comment, etc.)
        if not history and source.platform and source.platform != Platform.LOCAL and source.platform != Platform.WEBHOOK:
            platform_name = source.platform.value
            env_key = _home_target_env_var(platform_name)
            if not os.getenv(env_key):
                # Slack dispatches all Hermes commands through a single
                # parent slash command `/hermes`; bare `/sethome` is not
                # registered and would fail with "app did not respond".
                sethome_cmd = (
                    "/hermes sethome"
                    if source.platform == Platform.SLACK
                    else "/sethome"
                )
                notice = (
                    f"📬 No home channel is set for {platform_name.title()}. "
                    f"A home channel is where Hermes delivers cron job results "
                    f"and cross-platform messages.\n\n"
                    f"Type {sethome_cmd} to make this chat your home channel, "
                    f"or ignore to skip."
                )
                await self._deliver_platform_notice(source, notice)
        
        # -----------------------------------------------------------------
        # Voice channel awareness — inject current voice channel state
        # into context so the agent knows who is in the channel and who
        # is speaking, without needing a separate tool call.
        # -----------------------------------------------------------------
        if source.platform == Platform.DISCORD:
            adapter = self.adapters.get(Platform.DISCORD)
            guild_id = self._get_guild_id(event)
            if guild_id and adapter and hasattr(adapter, "get_voice_channel_context"):
                vc_context = adapter.get_voice_channel_context(guild_id)
                if vc_context:
                    context_prompt += f"\n\n{vc_context}"

        # -----------------------------------------------------------------
        # Auto-analyze images sent by the user
        #
        # If the user attached image(s), we run the vision tool eagerly so
        # the conversation model always receives a text description.  The
        # local file path is also included so the model can re-examine the
        # image later with a more targeted question via vision_analyze.
        #
        # We filter to image paths only (by media_type) so that non-image
        # attachments (documents, audio, etc.) are not sent to the vision
        # tool even when they appear in the same message.
        # -----------------------------------------------------------------
        message_text = await self._prepare_inbound_message_text(
            event=event,
            source=source,
            history=history,
        )
        if message_text is None:
            return

        # Capture the platform event time as message metadata and keep the
        # persisted transcript clean (strip any leading timestamp prefix).
        # This runs regardless of the toggle so storage stays clean and the
        # send-time is preserved. Only the in-context RENDER (prepending the
        # human-readable prefix the model sees) is gated behind
        # gateway.message_timestamps.enabled — default OFF.
        try:
            from hermes_time import get_timezone as _get_evt_tz
            from gateway.message_timestamps import (
                coerce_message_timestamp as _coerce_msg_ts,
                render_user_content_with_timestamp as _render_msg_ts,
                strip_leading_message_timestamps as _strip_msg_ts,
            )
            _evt_tz = _get_evt_tz()
            _evt_ts = getattr(event, "timestamp", None)
            if message_text and isinstance(message_text, str):
                _clean_message_text, _embedded_ts = _strip_msg_ts(
                    message_text, tz=_evt_tz)
                persist_user_message = _clean_message_text
                _event_epoch = _coerce_msg_ts(_evt_ts, tz=_evt_tz)
                persist_user_timestamp = (
                    _event_epoch if _event_epoch is not None else _embedded_ts
                )
                if _message_timestamps_enabled(_load_gateway_config()):
                    message_text = _render_msg_ts(
                        _clean_message_text,
                        persist_user_timestamp,
                        tz=_evt_tz,
                    )
                else:
                    # Toggle off: model sees the clean message; the timestamp
                    # is still stored as metadata for later opt-in.
                    message_text = _clean_message_text
        except Exception as _ts_err:
            logger.debug("Message timestamp injection failed (non-fatal): %s", _ts_err)

        # Bind this gateway run generation to the adapter's active-session
        # event so deferred post-delivery callbacks can be released by the
        # same run that registered them.
        self._bind_adapter_run_generation(
            self.adapters.get(source.platform),
            session_key,
            run_generation,
        )

        try:
            # Emit agent:start hook
            hook_ctx = {
                "platform": source.platform.value if source.platform else "",
                "user_id": source.user_id,
                "chat_id": source.chat_id or "",
                "thread_id": str(getattr(source, "thread_id", None)) if getattr(source, "thread_id", None) else "",
                "chat_type": getattr(source, "chat_type", "") or "",
                "session_id": session_entry.session_id,
                "message": message_text[:500],
            }
            await self.hooks.emit("agent:start", hook_ctx)

            # Run the agent
            agent_result = await self._run_agent(
                message=message_text,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_entry.session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=self._reply_anchor_for_event(event),
                channel_prompt=event.channel_prompt,
                moa_config=getattr(event, "_moa_config", None),
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
            )

            # Stop persistent typing indicator now that the agent is done
            try:
                _typing_adapter = self.adapters.get(source.platform)
                if _typing_adapter and hasattr(_typing_adapter, "stop_typing"):
                    await _typing_adapter.stop_typing(source.chat_id)
            except Exception:
                pass

            if not self._is_session_run_current(_quick_key, run_generation):
                logger.info(
                    "Discarding stale agent result for %s — generation %d is no longer current",
                    _quick_key or "?",
                    run_generation,
                )
                _stale_adapter = self.adapters.get(source.platform)
                if getattr(type(_stale_adapter), "pop_post_delivery_callback", None) is not None:
                    _stale_adapter.pop_post_delivery_callback(
                        _quick_key,
                        generation=run_generation,
                    )
                elif _stale_adapter and hasattr(_stale_adapter, "_post_delivery_callbacks"):
                    _stale_adapter._post_delivery_callbacks.pop(_quick_key, None)
                return None

            response = agent_result.get("final_response") or ""
            try:
                from gateway.response_filters import is_intentional_silence_agent_result
                _intentional_silence = is_intentional_silence_agent_result(
                    agent_result, response,
                )
            except Exception:
                _intentional_silence = False

            # Convert the agent's internal "(empty)" sentinel into a
            # user-friendly message.  "(empty)" means the model failed to
            # produce visible content after exhausting all retries (nudge,
            # prefill, empty-retry, fallback).  Sending the raw sentinel
            # looks like a bug; a short explanation is more helpful.
            if response == "(empty)" and not _intentional_silence:
                response = (
                    "⚠️ The model returned no response after processing tool "
                    "results. This can happen with some models — try again or "
                    "rephrase your question."
                )
            agent_messages = agent_result.get("messages", [])
            _response_time = time.time() - _msg_start_time
            _api_calls = agent_result.get("api_calls", 0)
            _resp_len = len(response)
            logger.info(
                "response ready: platform=%s chat=%s time=%.1fs api_calls=%d response=%d chars",
                _platform_name, source.chat_id or "unknown",
                _response_time, _api_calls, _resp_len,
            )

            # NOTE: the cross-process cache-coherence re-baseline
            # (_refresh_agent_cache_message_count) is intentionally deferred
            # until AFTER this turn's transcript persistence block below — it
            # must include the first-turn `session_meta` marker row and the
            # compression session_id swap, both of which happen later.  See
            # the call site after the `update_session(...)` write.

            # Successful turn — clear any stuck-loop counter for this session.
            # This ensures the counter only accumulates across CONSECUTIVE
            # restarts where the session was active (never completed).
            #
            # Also clear the resume_pending flag (set by drain-timeout
            # shutdown) — the turn ran to completion, so recovery
            # succeeded and subsequent messages should no longer receive
            # the restart-interruption system note.
            if session_key and _should_clear_resume_pending_after_turn(agent_result):
                self._clear_restart_failure_count(session_key)
                try:
                    self.session_store.clear_resume_pending(session_key)
                except Exception as _e:
                    logger.debug(
                        "clear_resume_pending failed for %s: %s",
                        session_key, _e,
                    )

            # Normalize empty responses: surface errors, partial failures, and
            # the case where agent did work but returned no text. Fix for #18765.
            if not _intentional_silence:
                response = _normalize_empty_agent_response(
                    agent_result, response, history_len=len(history),
                )
                response = _sanitize_gateway_final_response(source.platform, response)

            # Ordering contract: the agent thread already updated the contextvar
            # in conversation_compression.py; propagate to SessionEntry + _save().
            # If the agent's session_id changed during compression, update
            # session_entry so transcript writes below go to the right session.
            if agent_result.get("session_id") and agent_result["session_id"] != session_entry.session_id:
                session_entry.session_id = agent_result["session_id"]
                self.session_store._save()
                await asyncio.to_thread(
                    self._sync_telegram_topic_binding,
                    source, session_entry, reason="agent-result-compression",
                )

            # Prepend reasoning/thinking if display is enabled (per-platform).
            # Mattermost requires explicit per-platform opt-in because this is
            # scratch text, not ordinary final-answer content.
            try:
                _show_reasoning_effective = _resolve_gateway_display_bool(
                    _load_gateway_config(),
                    _platform_config_key(source.platform),
                    "show_reasoning",
                    default=bool(getattr(self, "_show_reasoning", False)),
                    platform=source.platform,
                    require_platform_override_for={Platform.MATTERMOST},
                )
            except Exception:
                _show_reasoning_effective = (
                    False
                    if source.platform == Platform.MATTERMOST
                    else getattr(self, "_show_reasoning", False)
                )
            if _show_reasoning_effective and response and not _intentional_silence:
                last_reasoning = agent_result.get("last_reasoning")
                if last_reasoning:
                    # Collapse long reasoning to keep messages readable
                    lines = last_reasoning.strip().splitlines()
                    if len(lines) > 15:
                        display_reasoning = "\n".join(lines[:15])
                        display_reasoning += f"\n_... ({len(lines) - 15} more lines)_"
                    else:
                        display_reasoning = last_reasoning.strip()
                    # Render style is per-platform: Discord defaults to "-# "
                    # subtext (native small grey metadata text); other
                    # platforms keep the fenced code block.
                    try:
                        from gateway.display_config import resolve_display_setting
                        _reasoning_style = resolve_display_setting(
                            _load_gateway_config(),
                            _platform_config_key(source.platform),
                            "reasoning_style",
                            "code",
                        )
                    except Exception:
                        _reasoning_style = "code"
                    if _reasoning_style == "subtext":
                        _quoted = "\n".join(
                            f"-# {ln}" if ln else "-#" for ln in display_reasoning.splitlines()
                        )
                        response = f"-# 💭 Reasoning\n{_quoted}\n\n{response}"
                    elif _reasoning_style == "blockquote":
                        _quoted = "\n".join(
                            f"> {ln}" if ln else ">" for ln in display_reasoning.splitlines()
                        )
                        response = f"> 💭 **Reasoning:**\n{_quoted}\n\n{response}"
                    else:
                        response = f"💭 **Reasoning:**\n```\n{display_reasoning}\n```\n\n{response}"

            # Runtime-metadata footer — only on the FINAL message of the turn.
            # Off by default (display.runtime_footer.enabled=false).  When
            # streaming already delivered the body, we can't mutate the sent
            # text, so we fire a separate trailing send below.
            _footer_line = ""
            try:
                from gateway.runtime_footer import build_footer_line as _bfl
                _footer_line = _bfl(
                    user_config=_load_gateway_config(),
                    platform_key=_platform_config_key(source.platform),
                    model=agent_result.get("model"),
                    context_tokens=agent_result.get("last_prompt_tokens", 0) or 0,
                    context_length=agent_result.get("context_length") or None,
                    cwd=os.environ.get("TERMINAL_CWD", ""),
                )
            except Exception as _footer_err:
                logger.debug("runtime_footer build failed: %s", _footer_err)
                _footer_line = ""
            if _footer_line and response and not agent_result.get("already_sent") and not _intentional_silence:
                response = f"{response}\n\n{_footer_line}"

            # Emit agent:end hook
            await self.hooks.emit("agent:end", {
                **hook_ctx,
                "response": (response or "")[:500],
            })
            
            # Check for pending process watchers (check_interval on background processes)
            try:
                from tools.process_registry import process_registry
                # Detach the current batch atomically (see crash-recovery drain
                # above): reassign to a fresh list so a watcher appended by a
                # concurrent session during the yield isn't dropped by clear().
                watchers = process_registry.pending_watchers
                process_registry.pending_watchers = []
                for i, watcher in enumerate(watchers):
                    asyncio.create_task(self._run_process_watcher(watcher))
                    if i % 100 == 99:
                        await asyncio.sleep(0)
            except Exception as e:
                logger.error("Process watcher setup error: %s", e)

            # Drain watch pattern notifications that arrived during the agent run.
            # Watch events and completions share the same queue; process
            # completions are already handled by the per-process watcher task
            # above, so we only inject watch-type events here.
            #
            # Async-delegation completions ALSO ride this shared queue but are
            # owned by the dedicated _async_delegation_watcher (started at
            # boot), which covers both the idle and post-turn cases with a
            # single consumer — so we leave them on the queue here.
            try:
                from tools.process_registry import process_registry as _pr
                _watch_events = _drain_gateway_watch_events(_pr.completion_queue)
                for evt in _watch_events:
                    synth_text = _format_gateway_process_notification(evt)
                    if synth_text:
                        try:
                            await self._inject_watch_notification(synth_text, evt)
                        except Exception as e2:
                            logger.error("Watch notification injection error: %s", e2)
            except Exception as e:
                logger.debug("Watch queue drain error: %s", e)

            # NOTE: Dangerous command approvals are now handled inline by the
            # blocking gateway approval mechanism in tools/approval.py.  The agent
            # thread blocks until the user responds with /approve or /deny, so by
            # the time we reach here the approval has already been resolved.  The
            # old post-loop pop_pending + approval_hint code was removed in favour
            # of the blocking approach that mirrors CLI's synchronous input().
            
            # Save the full conversation to the transcript, including tool calls.
            # This preserves the complete agent loop (tool_calls, tool results,
            # intermediate reasoning) so sessions can be resumed with full context
            # and transcripts are useful for debugging and training data.
            #
            # IMPORTANT: For context-overflow failures (compression exhausted,
            # generic 400 on large sessions) we must NOT persist the user's
            # message — doing so would grow the session further and cause the
            # same failure on the next attempt, an infinite loop. (#1630, #9893)
            #
            # Transient failures (429, timeout, connection error, provider 5xx)
            # are different: the session is not oversized, and silently dropping
            # the user message causes severe context loss on retry — the agent
            # forgets what was just asked.  Persist the user turn so the
            # conversation is preserved. (#7100)
            agent_failed_early = bool(agent_result.get("failed"))
            _err_str_for_classify = str(agent_result.get("error", "")).lower()
            # Use specific multi-word phrases (not bare "exceed" or "token")
            # to avoid false positives on transient errors like "rate limit
            # exceeded" or "invalid auth token". Matches run_agent.py's
            # own context-length classifier.
            is_context_overflow_failure = agent_failed_early and (
                bool(agent_result.get("compression_exhausted"))
                or any(p in _err_str_for_classify for p in (
                    "context length", "context size", "context window",
                    "maximum context", "token limit", "too many tokens",
                    "reduce the length", "exceeds the limit",
                    "request entity too large", "prompt is too long",
                    "payload too large", "input is too long",
                ))
                or ("400" in _err_str_for_classify and len(history) > 50)
            )
            if is_context_overflow_failure:
                logger.info(
                    "Skipping transcript persistence for context-overflow "
                    "failure in session %s to prevent session growth loop.",
                    session_entry.session_id,
                )
            elif agent_failed_early:
                logger.info(
                    "Transient agent failure in session %s — persisting user "
                    "message so conversation context is preserved on retry.",
                    session_entry.session_id,
                )

            # When compression is exhausted, the session is permanently too
            # large to process.  Auto-reset it so the next message starts
            # fresh instead of replaying the same oversized context in an
            # infinite fail loop.  (#9893)
            if agent_result.get("compression_exhausted") and session_entry and session_key:
                logger.info(
                    "Auto-resetting session %s after compression exhaustion.",
                    session_entry.session_id,
                )
                new_entry = self.session_store.reset_session(session_key)
                self._evict_cached_agent(session_key)
                self._session_model_overrides.pop(session_key, None)
                self._set_session_reasoning_override(session_key, None)
                if hasattr(self, "_pending_model_notes"):
                    self._pending_model_notes.pop(session_key, None)
                if new_entry is not None:
                    # Drop the stale reference to the bloated compressed child and
                    # re-point the Telegram topic binding at the fresh session.
                    # Compression rotated session_entry.session_id to the oversized
                    # compressed child earlier this turn (the agent-result sync
                    # above), and that _sync also rewrote the (chat_id, thread_id)
                    # -> bloated-child binding. reset_session swaps in a clean,
                    # parentless session, but without re-syncing the binding the
                    # next inbound message in this topic gets switch_session'd back
                    # onto the bloated child by the binding-heal walk, reloads the
                    # oversized transcript, and re-triggers compression exhaustion
                    # forever (#35809 — regression of the #9893/#10063 auto-reset).
                    # No-op on non-topic lanes.
                    session_entry = new_entry
                    await asyncio.to_thread(
                        self._sync_telegram_topic_binding,
                        source, session_entry, reason="compression-exhausted-reset",
                    )
                response = (response or "") + (
                    "\n\n🔄 Session auto-reset — the conversation exceeded the "
                    "maximum context size and could not be compressed further. "
                    "Your next message will start a fresh session."
                )

            ts = time.time()  # Unix epoch float — consistent with DB storage
            
            # If this is a fresh session (no history), write the full tool
            # definitions as the first entry so the transcript is self-describing
            # -- the same list of dicts sent as tools=[...] in the API request.
            if is_context_overflow_failure:
                pass  # Skip all transcript writes — don't grow a broken session
            elif not history:
                tool_defs = agent_result.get("tools", [])
                self.session_store.append_to_transcript(
                    session_entry.session_id,
                    {
                        "role": "session_meta",
                        "tools": tool_defs or [],
                        "model": _resolve_gateway_model(),
                        "platform": source.platform.value if source.platform else "",
                        "timestamp": ts,
                    }
                )
            
            # The agent already persisted these messages to SQLite via
            # _flush_messages_to_session_db(), so skip the DB write here
            # to prevent the duplicate-write bug (#860 / #42039). This holds
            # for the codex app-server runtime too: although it early-returns
            # and bypasses conversation_loop's per-step flushes, it flushes its
            # own projected assistant/tool messages before returning and
            # reports agent_persisted=True (see agent/codex_runtime.py). Reading
            # the flag (default = self._session_db is not None) keeps the
            # persistence contract explicit and lets any future non-persisting
            # runtime opt into a gateway-side write by returning False.
            agent_persisted = agent_result.get("agent_persisted", self._session_db is not None)

            # Find only the NEW messages from this turn (skip history we loaded).
            # Use the filtered history length (history_offset) that was actually
            # passed to the agent, not len(history) which includes session_meta
            # entries that were stripped before the agent saw them.
            if is_context_overflow_failure:
                pass  # handled above — skip all transcript writes
            elif agent_failed_early:
                # Transient failure (429/timeout/5xx): persist only the user
                # message so the next message can load a transcript that
                # reflects what was said.  Skip the assistant error text since
                # it's a gateway-generated hint, not model output. (#7100)
                _user_entry = {
                    "role": "user",
                    "content": (
                        persist_user_message
                        if persist_user_message is not None
                        else message_text
                    ),
                    "timestamp": (
                        persist_user_timestamp
                        if persist_user_timestamp is not None
                        else ts
                    ),
                }
                if event.message_id:
                    _user_entry["message_id"] = str(event.message_id)
                # Dedupe: skip if this platform message_id is already in the
                # transcript (prevents duplicate user turns on Telegram retries
                # after transient failures). #47237
                _skip_persist = (
                    event.message_id
                    and self.session_store.has_platform_message_id(
                        session_entry.session_id, str(event.message_id)
                    )
                )
                if _skip_persist:
                    logger.info(
                        "Skipping duplicate user turn "
                        "(message_id=%s) in session %s",
                        event.message_id, session_entry.session_id,
                    )
                else:
                    self.session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
            else:
                history_len = agent_result.get("history_offset", len(history))
                new_messages = agent_messages[history_len:] if len(agent_messages) > history_len else []

                # If no new messages found (edge case), fall back to simple user/assistant
                if not new_messages:
                    _user_entry = {
                        "role": "user",
                        "content": (
                            persist_user_message
                            if persist_user_message is not None
                            else message_text
                        ),
                        "timestamp": (
                            persist_user_timestamp
                            if persist_user_timestamp is not None
                            else ts
                        ),
                    }
                    if event.message_id:
                        _user_entry["message_id"] = str(event.message_id)
                    self.session_store.append_to_transcript(
                        session_entry.session_id,
                        _user_entry,
                        skip_db=agent_persisted,
                    )
                    if response:
                        self.session_store.append_to_transcript(
                            session_entry.session_id,
                            {"role": "assistant", "content": response, "timestamp": ts},
                            skip_db=agent_persisted,
                        )
                else:
                    # Attach the inbound platform message_id to the first user
                    # entry written this turn so platform-level quote-resolution
                    # (e.g. Yuanbao QuoteContextMiddleware's transcript fallback)
                    # can find earlier @bot messages by their original message_id.
                    _user_msg_id_attached = False
                    for msg in new_messages:
                        # Skip system messages (they're rebuilt each run)
                        if msg.get("role") == "system":
                            continue
                        # Add timestamp to each message for debugging
                        entry = {**msg, "timestamp": ts}
                        if (
                            not _user_msg_id_attached
                            and msg.get("role") == "user"
                            and event.message_id
                            and "message_id" not in entry
                        ):
                            entry["message_id"] = str(event.message_id)
                            _user_msg_id_attached = True
                        self.session_store.append_to_transcript(
                            session_entry.session_id, entry,
                            skip_db=agent_persisted,
                        )
            
            # Token counts and model are now persisted by the agent directly.
            # Keep only last_prompt_tokens here for context-window tracking and
            # compression decisions.
            self.session_store.update_session(
                session_entry.session_key,
                last_prompt_tokens=agent_result.get("last_prompt_tokens", 0),
            )

            # Re-baseline the cached agent's message_count snapshot now that
            # ALL of this turn's transcript writes are done — the agent's
            # flushed user/assistant/tool rows AND the first-turn `session_meta`
            # marker appended above.  The cross-process coherence guard (#45966)
            # snapshots the count at agent-BUILD time (before this turn's own
            # writes) and never refreshes it on reuse, so without this the
            # process's own turn grows message_count and the next turn sees a
            # mismatch and rebuilds the agent — destroying prompt caching.
            #
            # This MUST run after the `session_meta` append: that row also
            # increments message_count, so re-baselining before it (the old
            # position) left the snapshot one short and the guard mis-fired on
            # turn 2 of EVERY fresh gateway conversation, rebuilding the cached
            # agent and busting the prompt cache.  Running here also uses the
            # compaction-updated session_id (the agent_result session_id swap
            # above), matching this function's documented contract.  Refreshing
            # here makes the guard fire only on a DIFFERENT process's writes.
            # Fail-safe inside the helper.
            await self._refresh_agent_cache_message_count(
                session_key, session_entry.session_id
            )

            # Intentional silence is a delivery decision, not a transcript
            # mutation.  The agent's [SILENT]/NO_REPLY assistant turn above is
            # still persisted in session history so later turns keep normal
            # user/assistant alternation; only the outbound chat delivery is
            # suppressed.
            if _intentional_silence:
                logger.info(
                    "Suppressing intentional silence marker for session %s",
                    session_entry.session_id,
                )
                response = ""

            # Auto voice reply: send TTS audio before the text response
            _already_sent = bool(agent_result.get("already_sent"))
            if self._should_send_voice_reply(event, response, agent_messages, already_sent=_already_sent):
                await self._send_voice_reply(event, response)

            # If streaming already delivered the response, extract and
            # deliver any MEDIA: files before returning None.  Streaming
            # sends raw text chunks that include MEDIA: tags — the normal
            # post-processing in _process_message_background is skipped
            # when already_sent is True, so media files would never be
            # delivered without this.
            #
            # Never skip when the agent failed — the error message is new
            # content the user hasn't seen (streaming only sent earlier
            # partial output before the failure).  Without this guard,
            # users see the agent "stop responding without explanation."
            if agent_result.get("already_sent") and not agent_result.get("failed"):
                if response:
                    _media_adapter = self.adapters.get(source.platform)
                    if _media_adapter:
                        await self._deliver_media_from_response(
                            response, event, _media_adapter,
                        )
                # Streaming already delivered the body text, but the footer was
                # intentionally held back (see the `not already_sent` gate above).
                # Send it now as a small trailing message so Telegram/Discord/etc.
                # still surface the runtime metadata on the final reply.
                if _footer_line:
                    try:
                        _foot_adapter = self.adapters.get(source.platform)
                        if _foot_adapter:
                            await _foot_adapter.send(
                                source.chat_id,
                                _footer_line,
                                metadata=self._thread_metadata_for_source(source, self._reply_anchor_for_event(event)),
                            )
                    except Exception as _e:
                        logger.debug("trailing footer send failed: %s", _e)
                return None

            return response
            
        except Exception as e:
            # Stop typing indicator on error too
            try:
                _err_adapter = self.adapters.get(source.platform)
                if _err_adapter and hasattr(_err_adapter, "stop_typing"):
                    await _err_adapter.stop_typing(source.chat_id)
            except Exception:
                pass
            logger.exception("Agent error in session %s", session_key)
            # Crash-resilience for failures that happen before AIAgent enters
            # run_conversation() (for example: provider/httpx client init
            # failures). In that path the agent cannot persist the current
            # inbound turn itself, so append the user message here once. If the
            # agent already reached its early turn-start persistence, the latest
            # transcript user row will match and we skip the duplicate.
            try:
                if 'message_text' in locals() and message_text is not None and session_entry is not None:
                    _already_persisted = False
                    try:
                        _recent_transcript = self.session_store.load_transcript(session_entry.session_id)
                    except Exception:
                        _recent_transcript = []
                    for _msg in reversed(_recent_transcript[-10:]):
                        if _msg.get("role") == "user":
                            _expected_user_content = (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            )
                            _already_persisted = (_msg.get("content") == _expected_user_content)
                            break
                    if not _already_persisted:
                        _user_entry = {
                            "role": "user",
                            "content": (
                                persist_user_message
                                if persist_user_message is not None
                                else message_text
                            ),
                            "timestamp": (
                                persist_user_timestamp
                                if persist_user_timestamp is not None
                                else time.time()
                            ),
                        }
                        if getattr(event, "message_id", None):
                            _user_entry["message_id"] = str(event.message_id)
                        self.session_store.append_to_transcript(
                            session_entry.session_id,
                            _user_entry,
                        )
            except Exception:
                logger.debug("Failed to persist inbound user message after agent exception", exc_info=True)
            # Log full details server-side only; never expose raw exception
            # types or messages to end users (info-leakage risk).
            status_hint = ""
            status_code = getattr(e, "status_code", None)
            _hist_len = len(history) if 'history' in locals() else 0
            if status_code == 401:
                status_hint = " Check your API key or run `claude /login` to refresh OAuth credentials."
            elif status_code == 402:
                status_hint = " Your API balance or quota is exhausted. Check your provider dashboard."
            elif status_code == 429:
                # Check if this is a plan usage limit (resets on a schedule) vs a transient rate limit
                _err_body = getattr(e, "response", None)
                _err_json = {}
                try:
                    if _err_body is not None:
                        _err_json = _err_body.json().get("error", {})
                        if not isinstance(_err_json, dict):
                            _err_json = {}
                except Exception:
                    pass
                if _err_json.get("type") == "usage_limit_reached":
                    _resets_in = _err_json.get("resets_in_seconds")
                    if _resets_in and _resets_in > 0:
                        import math
                        _hours = math.ceil(_resets_in / 3600)
                        status_hint = f" Your plan's usage limit has been reached. It resets in ~{_hours}h."
                    else:
                        status_hint = " Your plan's usage limit has been reached. Please wait until it resets."
                else:
                    status_hint = " You are being rate-limited. Please wait a moment and try again."
            elif status_code == 529:
                status_hint = " The API is temporarily overloaded. Please try again shortly."
            elif status_code in {400, 500}:
                # 400 with a large session is context overflow.
                # 500 with a large session often means the payload is too large
                # for the API to process — treat it the same way.
                if _hist_len > 50:
                    return (
                        "⚠️ Session too large for the model's context window.\n"
                        "Use /compact to compress the conversation, or "
                        "/reset to start fresh."
                    )
                elif status_code == 400:
                    status_hint = " The request was rejected by the API."
            return (
                f"Sorry, I encountered an unexpected error.{status_hint}\n"
                "Try again or use /reset to start a fresh session."
            )
        finally:
            # Restore session context variables to their pre-handler state
            self._clear_session_env(_session_env_tokens)

    def _format_session_info(self) -> str:
        """Resolve current model config and return a formatted info block.

        Surfaces model, provider, context length, and endpoint so gateway
        users can immediately see if context detection went wrong (e.g.
        local models falling to the 128K default).
        """
        from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT

        model = _resolve_gateway_model()
        config_context_length = None
        provider = None
        base_url = None
        api_key = None
        custom_provs = None
        data = None

        try:
            data = _load_gateway_config()
            if data:
                model_cfg = data.get("model", {})
                if isinstance(model_cfg, dict):
                    raw_ctx = model_cfg.get("context_length")
                    if raw_ctx is not None:
                        try:
                            config_context_length = int(raw_ctx)
                        except (TypeError, ValueError):
                            pass
                    provider = model_cfg.get("provider") or None
                    base_url = model_cfg.get("base_url") or None
                try:
                    from hermes_cli.config import get_compatible_custom_providers
                    custom_provs = get_compatible_custom_providers(data)
                except Exception:
                    custom_provs = data.get("custom_providers")
        except Exception:
            pass

        # Also check custom_providers for context_length when top-level model.context_length is not set
        if config_context_length is None and data:
            try:
                custom_providers = data.get("custom_providers", [])
                if custom_providers:
                    for cp in custom_providers:
                        if not isinstance(cp, dict):
                            continue
                        cp_model = cp.get("model") or ""
                        cp_models = cp.get("models") or {}
                        # Match provider model to current model
                        if cp_model and cp_model == model:
                            raw_cp_ctx = cp.get("context_length")
                            if raw_cp_ctx is not None:
                                try:
                                    config_context_length = int(raw_cp_ctx)
                                    break
                                except (TypeError, ValueError):
                                    pass
                        # Also check per-model context_length
                        if isinstance(cp_models, dict):
                            model_entry = cp_models.get(model)
                            if isinstance(model_entry, dict):
                                model_ctx = model_entry.get("context_length")
                            else:
                                model_ctx = model_entry
                            if model_ctx is not None and isinstance(model_ctx, (int, float)):
                                try:
                                    config_context_length = int(model_ctx)
                                    break
                                except (TypeError, ValueError):
                                    pass
            except Exception:
                pass

        # Resolve runtime credentials for probing
        try:
            runtime = _resolve_runtime_agent_kwargs()
            provider = provider or runtime.get("provider")
            base_url = base_url or runtime.get("base_url")
            api_key = runtime.get("api_key")
        except Exception:
            pass

        context_length = get_model_context_length(
            model,
            base_url=base_url or "",
            api_key=api_key or "",
            config_context_length=config_context_length,
            provider=provider or "",
            custom_providers=custom_provs,
        )

        # Format context source hint
        if config_context_length is not None:
            ctx_source = "config"
        elif context_length == DEFAULT_FALLBACK_CONTEXT:
            ctx_source = "default — set model.context_length in config to override"
        else:
            ctx_source = "detected"

        # Format context length for display
        if context_length >= 1_000_000:
            ctx_display = f"{context_length / 1_000_000:.1f}M"
        elif context_length >= 1_000:
            ctx_display = f"{context_length // 1_000}K"
        else:
            ctx_display = str(context_length)

        lines = [
            f"◆ Model: `{model}`",
            f"◆ Provider: {provider or 'openrouter'}",
            f"◆ Context: {ctx_display} tokens ({ctx_source})",
        ]

        # Show endpoint for local/custom setups
        if base_url and ("localhost" in base_url or "127.0.0.1" in base_url or "0.0.0.0" in base_url):
            lines.append(f"◆ Endpoint: {base_url}")

        return "\n".join(lines)




    def _check_slash_access(
        self, source: SessionSource, canonical_cmd: str
    ) -> Optional[str]:
        """Return a denial message if ``source`` cannot run ``canonical_cmd``,
        else None. Used by both the cold and running-agent dispatch paths
        in ``_handle_message`` so admin/user gating can't be bypassed by
        an in-flight agent.

        Backward-compat semantics live in
        :func:`gateway.slash_access.policy_for_source` — when the operator
        hasn't set ``allow_admin_from`` for the scope, the policy returns
        ``enabled=False`` and this method always returns None.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        if not canonical_cmd:
            return None
        policy = _policy_for_source(self.config, source)
        if not policy.enabled or policy.can_run(source.user_id, canonical_cmd):
            return None
        logger.info(
            "Slash command /%s denied for %s:%s (not admin, not in user_allowed_commands)",
            canonical_cmd,
            source.platform.value if source.platform else "?",
            source.user_id,
        )
        allowed_preview = sorted(policy.user_allowed_commands)
        if allowed_preview:
            suffix = (
                "You can run: "
                + ", ".join(f"/{c}" for c in allowed_preview[:12])
                + ("…" if len(allowed_preview) > 12 else "")
                + ". Use /whoami for the full list."
            )
        else:
            suffix = (
                "No slash commands are enabled for non-admins on this "
                "platform. Ask an admin to add you to allow_admin_from "
                "or to set user_allowed_commands."
            )
        return f"⛔ /{canonical_cmd} is admin-only here. {suffix}"







    def _sibling_thread_run_keys(self, source: SessionSource, own_key: str) -> list:
        """Find running-agent keys for OTHER participants in the same thread.

        Only applies when the message originates in a thread.  In per-user
        thread mode (``thread_sessions_per_user=True``) each participant gets
        an isolated session key of the form
        ``agent:main:{platform}:{chat_type}:{chat_id}:{thread_id}:{user_id}``,
        so a run started by another user is invisible to the caller's own
        ``/stop``.  This returns the keys of any *actually running* agents
        (not the pending sentinel, not the caller's own key) whose key shares
        the caller's ``{chat_id}:{thread_id}`` prefix.

        Returns an empty list when the source is not in a thread, or when no
        sibling runs exist — callers must still gate on authorization.
        """
        thread_id = getattr(source, "thread_id", None)
        chat_id = getattr(source, "chat_id", None)
        if not thread_id or not chat_id:
            return []
        platform = source.platform.value
        chat_type = getattr(source, "chat_type", None) or ""
        # Prefix that every per-user key in this thread shares, up to and
        # including the thread_id segment.  Matching either the exact
        # shared-thread key or any key with a further (user_id) segment
        # (prefix + ":") avoids cross-matching an unrelated thread whose id
        # merely starts with this one.
        prefix = ":".join(
            ["agent:main", platform, chat_type, str(chat_id), str(thread_id)]
        )
        matches = []
        for key, agent in list(self._running_agents.items()):
            if key == own_key:
                continue
            if agent is _AGENT_PENDING_SENTINEL or not agent:
                continue
            if key == prefix or key.startswith(prefix + ":"):
                matches.append(key)
        return matches




    def _is_stale_restart_redelivery(self, event: MessageEvent) -> bool:
        """Return True if this /restart is a Telegram re-delivery we already handled.

        The previous gateway wrote ``.restart_last_processed.json`` with the
        triggering platform + update_id when it processed the /restart.  If
        we now see a /restart on the same platform with an update_id <= that
        recorded value AND the marker is recent (< 5 minutes), it's a
        redelivery and should be ignored.

        Only applies to Telegram today (the only platform that exposes a
        numeric cross-session update ordering); other platforms return False.
        """
        if event is None or event.source is None:
            return False
        if event.platform_update_id is None:
            return False
        if event.source.platform is None:
            return False
        # Only Telegram populates platform_update_id currently; be explicit
        # so future platforms aren't accidentally gated by this check.
        try:
            platform_value = event.source.platform.value
        except Exception:
            return False
        if platform_value != "telegram":
            return False

        try:
            marker_path = _hermes_home / ".restart_last_processed.json"
            if not marker_path.exists():
                # Belt-and-suspenders for when the dedup marker goes missing
                # (manually cleaned up, or the previous cycle's write failed).
                # Without a marker the update_id comparison below can't run, so
                # a redelivered /restart would sail through and re-restart the
                # gateway — an infinite loop (issue #18528).
                #
                # Suppress ONLY when we can independently confirm we just came
                # out of a restart cycle: this process booted from a
                # chat-originated /restart (_booted_from_restart) AND is still
                # within a short post-boot window. This never swallows a
                # genuine first /restart on a fresh boot (no restart marker on
                # boot → flag stays False). Consume the flag one-shot so a
                # legitimate /restart sent later in the same session is honored.
                if (
                    getattr(self, "_booted_from_restart", False)
                    and time.time() - getattr(self, "_startup_time", 0.0) < 60
                ):
                    self._booted_from_restart = False
                    return True
                return False
            data = json.loads(marker_path.read_text())
        except Exception:
            return False

        if data.get("platform") != platform_value:
            return False
        recorded_uid = data.get("update_id")
        if not isinstance(recorded_uid, int):
            return False
        # Staleness guard: ignore markers older than 5 minutes.  A legitimately
        # old marker (e.g. crash recovery where notify never fired) should not
        # swallow a fresh /restart from the user.
        requested_at = data.get("requested_at")
        if isinstance(requested_at, (int, float)):
            if time.time() - requested_at > 300:
                return False
        return event.platform_update_id <= recorded_uid









    async def _handle_suggestions_command(self, event: MessageEvent) -> str:
        """Handle /suggestions in the gateway.

        Delegates to the shared handler so CLI and gateway never drift. The
        origin is built from the event source so an accepted suggestion's job
        delivers back to this chat/thread.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.suggestions_cmd import handle_suggestions_command

            return handle_suggestions_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("suggestions command failed: %s", e)
            return f"Suggestions command failed: {e}"

    async def _handle_blueprint_command(self, event: MessageEvent):
        """Handle /blueprint in the gateway.

        Delegates to the shared handler so CLI, TUI, and gateway never drift.
        Returns a BlueprintCommandResult: ``text`` is shown to the user, and if
        ``agent_seed`` is set the dispatch site rewrites ``event.text`` to the
        seed and falls through to the agent (the ``/steer`` pattern) so the
        agent gathers the slot values conversationally. Origin is built from the
        event source so a directly created blueprint job delivers back to this chat.
        """
        args = (event.get_command_args() or "").strip()
        source = event.source
        origin = None
        try:
            platform = getattr(source.platform, "value", None) or str(getattr(source, "platform", "") or "")
            chat_id = getattr(source, "chat_id", None)
            if platform and chat_id:
                origin = {
                    "platform": platform,
                    "chat_id": str(chat_id),
                    "chat_name": getattr(source, "chat_name", None),
                    "thread_id": getattr(source, "thread_id", None),
                }
        except Exception:
            origin = None
        try:
            from hermes_cli.blueprint_cmd import handle_blueprint_command

            return handle_blueprint_command(args, origin=origin, surface="gateway")
        except Exception as e:
            logger.debug("blueprint command failed: %s", e)
            from hermes_cli.blueprint_cmd import BlueprintCommandResult

            return BlueprintCommandResult(f"Cron blueprint command failed: {e}")

    # ────────────────────────────────────────────────────────────────
    # /goal — persistent cross-turn goals (Ralph-style loop)
    # ────────────────────────────────────────────────────────────────
    def _goal_max_turns_from_config(self) -> int:
        """Resolve the configured /goal turn budget for gateway sessions.

        GatewayRunner.config is a GatewayConfig dataclass, not the full
        user config mapping. Top-level config blocks such as ``goals`` are
        therefore only available through hermes_cli.config.load_config().
        """
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return a GoalManager bound to the session for this gateway event.

        Returns ``(manager, session_entry)`` or ``(None, None)`` if the
        goals module can't be loaded.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal manager unavailable: %s", exc)
            return None, None
        try:
            session_entry = self.session_store.get_or_create_session(event.source)
        except Exception as exc:
            logger.debug("goal manager: session lookup failed: %s", exc)
            return None, None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None, None
        max_turns = self._goal_max_turns_from_config()
        return GoalManager(session_id=sid, default_max_turns=max_turns), session_entry



    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None

        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s",
                getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The gateway message handler returns the agent response to the platform
        adapter, which sends it after this method's caller has returned.  For a
        natural Discord/Telegram reading order, goal status belongs after that
        send.  Platform adapters provide a one-shot post-delivery callback for
        exactly this boundary; when unavailable, fall back to direct awaited
        delivery rather than silently dropping the notice.
        """
        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None

        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(
                    session_key,
                    _deliver,
                    generation=generation,
                )
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)

        await _deliver()

    async def _post_turn_goal_continuation(
        self,
        *,
        session_entry: Any,
        source: Any,
        final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn and, if still active,
        enqueue a continuation prompt for the same session.

        Called from ``_handle_message_with_agent`` at turn boundary, AFTER
        the response has been delivered. Safe when no goal is set.

        We use the adapter's pending-message / FIFO machinery so any real
        user message that arrives simultaneously is handled by the same
        queue and takes priority naturally.
        """
        try:
            from hermes_cli.goals import GoalManager
        except Exception as exc:
            logger.debug("goal continuation: goals module unavailable: %s", exc)
            return

        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return

        max_turns = self._goal_max_turns_from_config()

        mgr = GoalManager(session_id=sid, default_max_turns=max_turns)
        if not mgr.is_active():
            return

        try:
            from hermes_cli.goals import gather_background_processes as _gather_bg
            _bg_procs = _gather_bg()
        except Exception:
            _bg_procs = None

        decision = mgr.evaluate_after_turn(
            final_response or "",
            user_initiated=True,
            background_processes=_bg_procs,
        )
        msg = decision.get("message") or ""

        # Defer the status line until after the adapter has delivered the
        # agent's visible final response. The judge runs after the response is
        # produced but before BasePlatformAdapter sends it, so sending here
        # would show "✓ Goal achieved" before the answer itself. Registering
        # an awaited post-delivery callback preserves delivery reliability
        # without reversing the user-visible ordering.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

        if not decision.get("should_continue"):
            return

        prompt = decision.get("continuation_prompt") or ""
        if not prompt or source is None:
            return

        # Enqueue via the adapter's FIFO so a user message already in
        # flight preempts the continuation naturally.
        try:
            adapter = self.adapters.get(source.platform)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                cont_event = MessageEvent(
                    text=prompt,
                    message_type=MessageType.TEXT,
                    source=source,
                    message_id=None,
                    channel_prompt=None,
                )
                self._enqueue_fifo(_quick_key, cont_event, adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)



    @staticmethod
    def _get_guild_id(event: MessageEvent) -> Optional[int]:
        """Extract Discord guild_id from the raw message object."""
        raw = getattr(event, "raw_message", None)
        if raw is None:
            return None
        # Slash command interaction
        if hasattr(raw, "guild_id") and raw.guild_id:
            return int(raw.guild_id)
        # Regular message
        if hasattr(raw, "guild") and raw.guild:
            return raw.guild.id
        return None


    async def _handle_voice_channel_join(self, event: MessageEvent) -> str:
        """Join the user's current Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        if not hasattr(adapter, "join_voice_channel"):
            return "Voice channels are not supported on this platform."

        guild_id = self._get_guild_id(event)
        if not guild_id:
            return "This command only works in a Discord server."

        voice_channel = await adapter.get_user_voice_channel(
            guild_id, event.source.user_id
        )
        if not voice_channel:
            return "You need to be in a voice channel first."

        # Wire callbacks BEFORE join so voice input arriving immediately
        # after connection is not lost.
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = self._handle_voice_channel_input
        if hasattr(adapter, "_on_voice_disconnect"):
            adapter._on_voice_disconnect = self._handle_voice_timeout_cleanup
        # Let the adapter's inactivity timer see the live voice-reply mode so it
        # doesn't disconnect a deliberately text-only (/voice off) session.
        if hasattr(adapter, "_voice_mode_getter"):
            adapter._voice_mode_getter = lambda chat_id: self._voice_mode.get(
                self._voice_key(Platform.DISCORD, str(chat_id)), "off"
            )

        try:
            success = await adapter.join_voice_channel(voice_channel)
        except Exception as e:
            logger.warning("Failed to join voice channel: %s", e)
            adapter._voice_input_callback = None
            err_lower = str(e).lower()
            if "pynacl" in err_lower or "nacl" in err_lower or "davey" in err_lower:
                return (
                    "Voice dependencies are missing (PyNaCl / davey). "
                    f"Install with: `{sys.executable} -m pip install PyNaCl`"
                )
            return f"Failed to join voice channel: {e}"

        if success:
            adapter._voice_text_channels[guild_id] = int(event.source.chat_id)
            if hasattr(adapter, "_voice_sources"):
                adapter._voice_sources[guild_id] = event.source.to_dict()
            self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "all"
            self._save_voice_modes()
            self._set_adapter_auto_tts_enabled(adapter, event.source.chat_id, enabled=True)
            return (
                f"Joined voice channel **{voice_channel.name}**.\n"
                f"I'll speak my replies and listen to you. Use /voice leave to disconnect."
            )
        # Join failed — clear callback
        adapter._voice_input_callback = None
        return "Failed to join voice channel. Check bot permissions (Connect + Speak)."

    async def _handle_voice_channel_leave(self, event: MessageEvent) -> str:
        """Leave the Discord voice channel."""
        adapter = self.adapters.get(event.source.platform)
        guild_id = self._get_guild_id(event)

        if not guild_id or not hasattr(adapter, "leave_voice_channel"):
            return "Not in a voice channel."

        if not hasattr(adapter, "is_in_voice_channel") or not adapter.is_in_voice_channel(guild_id):
            return "Not in a voice channel."

        try:
            await adapter.leave_voice_channel(guild_id)
        except Exception as e:
            logger.warning("Error leaving voice channel: %s", e)
        # Always clean up state even if leave raised an exception
        self._voice_mode[self._voice_key(event.source.platform, event.source.chat_id)] = "off"
        self._save_voice_modes()
        self._set_adapter_auto_tts_disabled(adapter, event.source.chat_id, disabled=True)
        if hasattr(adapter, "_voice_input_callback"):
            adapter._voice_input_callback = None
        return "Left voice channel."

    def _handle_voice_timeout_cleanup(self, chat_id: str) -> None:
        """Called by the adapter when a voice channel times out.

        Cleans up runner-side voice_mode state that the adapter cannot reach.
        """
        self._voice_mode[self._voice_key(Platform.DISCORD, chat_id)] = "off"
        self._save_voice_modes()
        adapter = self.adapters.get(Platform.DISCORD)
        self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)

    def _is_duplicate_voice_transcript(self, guild_id: int, user_id: int, transcript: str) -> bool:
        """Suppress repeated STT outputs for the same recent utterance.

        Voice capture can occasionally emit the same utterance twice a few
        seconds apart, which creates a second queued agent run and overlapping
        spoken replies. Dedup exact and near-exact repeats per guild/user over a
        short window while allowing genuinely new turns through.
        """
        from difflib import SequenceMatcher

        normalized = re.sub(r"\s+", " ", transcript).strip().lower()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        if not normalized:
            return False

        now = time.monotonic()
        window_seconds = 12.0
        key = (guild_id, user_id)
        recent_store = getattr(self, "_recent_voice_transcripts", None)
        if not isinstance(recent_store, dict):
            recent_store = {}
            self._recent_voice_transcripts = recent_store
        recent = [
            (ts, txt)
            for ts, txt in recent_store.get(key, [])
            if now - ts <= window_seconds
        ]

        for _, prior in recent:
            if prior == normalized:
                recent_store[key] = recent
                return True
            if len(prior) >= 16 and len(normalized) >= 16:
                if SequenceMatcher(None, prior, normalized).ratio() >= 0.95:
                    recent_store[key] = recent
                    return True

        recent.append((now, normalized))
        recent_store[key] = recent[-5:]
        return False

    async def _handle_voice_channel_input(
        self, guild_id: int, user_id: int, transcript: str
    ):
        """Handle transcribed voice from a user in a voice channel.

        Creates a synthetic MessageEvent and processes it through the
        adapter's full message pipeline (session, typing, agent, TTS reply).
        """
        adapter = self.adapters.get(Platform.DISCORD)
        if not adapter:
            return

        text_ch_id = adapter._voice_text_channels.get(guild_id)
        if not text_ch_id:
            return

        # Build source — reuse the linked text channel's metadata when available
        # so voice input shares the same session as the bound text conversation.
        source_data = getattr(adapter, "_voice_sources", {}).get(guild_id)
        if source_data:
            source = SessionSource.from_dict(source_data)
            source.user_id = str(user_id)
            source.user_name = str(user_id)
        else:
            source = SessionSource(
                platform=Platform.DISCORD,
                chat_id=str(text_ch_id),
                user_id=str(user_id),
                user_name=str(user_id),
                chat_type="channel",
            )

        # Check authorization before processing voice input
        if not self._is_user_authorized(source):
            logger.debug("Unauthorized voice input from user %d, ignoring", user_id)
            return

        if self._is_duplicate_voice_transcript(guild_id, user_id, transcript):
            logger.info(
                "Suppressing duplicate voice transcript for guild=%s user=%s: %s",
                guild_id,
                user_id,
                transcript[:100],
            )
            return

        # Show transcript in text channel (after auth, with mention sanitization)
        try:
            channel = adapter._client.get_channel(text_ch_id)
            if channel:
                safe_text = transcript[:2000].replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere")
                await channel.send(f"**[Voice]** <@{user_id}>: {safe_text}")
        except Exception:
            pass

        # Build a synthetic MessageEvent and feed through the normal pipeline
        # Use SimpleNamespace as raw_message so _get_guild_id() can extract
        # guild_id and _send_voice_reply() plays audio in the voice channel.
        from types import SimpleNamespace
        event = MessageEvent(
            source=source,
            text=transcript,
            message_type=MessageType.VOICE,
            raw_message=SimpleNamespace(guild_id=guild_id, guild=None),
        )

        await adapter.handle_message(event)

    def _should_send_voice_reply(
        self,
        event: MessageEvent,
        response: str,
        agent_messages: list,
        already_sent: bool = False,
    ) -> bool:
        """Decide whether the runner should send a TTS voice reply.

        Returns False when:
        - voice_mode is off for this chat
        - response is empty or an error
        - agent already called text_to_speech tool (dedup)
        - voice input and base adapter auto-TTS already handled it (skip_double)
          UNLESS streaming already consumed the response (already_sent=True),
          in which case the base adapter won't have text for auto-TTS so the
          runner must handle it.
        """
        if not response or response.startswith("Error:"):
            return False

        chat_id = event.source.chat_id
        voice_mode = self._voice_mode.get(self._voice_key(event.source.platform, chat_id), "off")
        is_voice_input = (event.message_type == MessageType.VOICE)

        should = (
            (voice_mode == "all")
            or (voice_mode == "voice_only" and is_voice_input)
        )
        if not should:
            return False

        # Dedup: agent already called TTS tool
        has_agent_tts = any(
            msg.get("role") == "assistant"
            and any(
                tc.get("function", {}).get("name") == "text_to_speech"
                for tc in (msg.get("tool_calls") or [])
            )
            for msg in agent_messages
        )
        if has_agent_tts:
            return False

        # Dedup: base adapter auto-TTS already handles voice input
        # (play_tts plays in VC when connected, so runner can skip).
        # When streaming already delivered the text (already_sent=True),
        # the base adapter will receive None and can't run auto-TTS,
        # so the runner must take over.
        if is_voice_input and not already_sent:
            return False

        return True

    async def _send_voice_reply(self, event: MessageEvent, text: str) -> None:
        """Generate TTS audio and send as a voice message before the text reply."""
        import uuid as _uuid
        audio_path = None
        actual_path = None
        try:
            from tools.tts_tool import text_to_speech_tool, _strip_markdown_for_tts

            tts_text = _strip_markdown_for_tts(text[:4000])
            if not tts_text:
                return

            # Telegram's adapter only sends native voice bubbles for OGG/Opus.
            # Other platforms keep the existing MP3 default.
            audio_ext = "ogg" if event.source.platform == Platform.TELEGRAM else "mp3"
            audio_path = os.path.join(
                tempfile.gettempdir(), "hermes_voice",
                f"tts_reply_{_uuid.uuid4().hex[:12]}.{audio_ext}",
            )
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            result_json = await asyncio.to_thread(
                text_to_speech_tool, text=tts_text, output_path=audio_path
            )
            try:
                result = json.loads(result_json)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Auto voice reply TTS returned invalid JSON: %s", result_json[:200] if result_json else result_json)
                return

            # Use the actual file path from result (may differ after opus conversion)
            actual_path = result.get("file_path", audio_path)
            if not result.get("success") or not os.path.isfile(actual_path):
                logger.warning("Auto voice reply TTS failed: %s", result.get("error"))
                return

            adapter = self.adapters.get(event.source.platform)

            # If connected to a voice channel, play there instead of sending a file
            guild_id = self._get_guild_id(event)
            if (guild_id
                    and hasattr(adapter, "play_in_voice_channel")
                    and hasattr(adapter, "is_in_voice_channel")
                    and adapter.is_in_voice_channel(guild_id)):
                await adapter.play_in_voice_channel(guild_id, actual_path)
            elif adapter and hasattr(adapter, "send_voice"):
                reply_anchor = self._reply_anchor_for_event(event)
                thread_meta = self._thread_metadata_for_source(event.source, reply_anchor)
                # Mark the auto voice reply as notify-worthy.  Mirrors the
                # final-text path in gateway/platforms/base.py which sets
                # ``notify=True`` so platform adapters that gate push
                # notifications (Telegram "important" mode) deliver the
                # final voice reply as a normal notification instead of a
                # silent message.  Clone first so we don't mutate metadata
                # shared with concurrent typing-indicator state.
                if thread_meta is not None:
                    thread_meta = dict(thread_meta)
                    thread_meta["notify"] = True
                else:
                    thread_meta = {"notify": True}
                send_kwargs: Dict[str, Any] = {
                    "chat_id": event.source.chat_id,
                    "audio_path": actual_path,
                    "reply_to": reply_anchor,
                    "metadata": thread_meta,
                }
                await adapter.send_voice(**send_kwargs)
        except Exception as e:
            logger.warning("Auto voice reply failed: %s", e, exc_info=True)
        finally:
            for p in {audio_path, actual_path} - {None}:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    async def _deliver_media_from_response(
        self,
        response: str,
        event: MessageEvent,
        adapter,
    ) -> None:
        """Extract MEDIA: tags and local file paths from a response and deliver them.

        Called after streaming has already sent the text to the user, so the
        text itself is already delivered — this only handles file attachments
        that the normal _process_message_background path would have caught.
        """
        from pathlib import Path
        from urllib.parse import quote as _quote

        try:
            # Capture [[as_document]] before extract_media strips it, so the
            # dispatch partition below can route image-extension files
            # through send_document (preserving bytes) instead of
            # send_multiple_images (Telegram sendPhoto recompresses to ~1280px).
            force_document_attachments = "[[as_document]]" in response

            from gateway.platforms.base import BasePlatformAdapter, should_send_media_as_audio

            media_files, cleaned = adapter.extract_media(response)
            media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
            # Chain the cleaned text through each extractor (extract_media →
            # extract_images → extract_local_files) so MEDIA: tags and image URLs
            # are removed before the bare-path auto-detect runs. Previously the
            # cleaned text from extract_media was dropped (``_``) and
            # extract_local_files scanned text that still contained MEDIA: tags,
            # producing false-positive bare-path matches with the MEDIA: prefix
            # glued on. This matches the chain order in gateway/platforms/base.py.
            _, cleaned = adapter.extract_images(cleaned)
            local_files, _ = adapter.extract_local_files(cleaned)
            local_files = BasePlatformAdapter.filter_local_delivery_paths(local_files)

            _thread_meta = self._thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))

            _VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'}
            _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

            # Partition out images so they can be sent as a single batch
            # (e.g. Signal's multi-attachment RPC). When [[as_document]] was
            # set, image-extension files skip the photo path and route to
            # send_document below — preserving original bytes.
            image_paths: list = []
            non_image_media: list = []
            for media_path, is_voice in media_files:
                ext = Path(media_path).suffix.lower()
                if (ext in _IMAGE_EXTS
                        and not is_voice
                        and not force_document_attachments):
                    image_paths.append(media_path)
                else:
                    non_image_media.append((media_path, is_voice))

            non_image_local: list = []
            for file_path in local_files:
                if (Path(file_path).suffix.lower() in _IMAGE_EXTS
                        and not force_document_attachments):
                    image_paths.append(file_path)
                else:
                    non_image_local.append(file_path)

            if image_paths:
                try:
                    images = [(f"file://{_quote(p)}", "") for p in image_paths]
                    await adapter.send_multiple_images(
                        chat_id=event.source.chat_id,
                        images=images,
                        metadata=_thread_meta,
                    )
                except Exception as e:
                    logger.warning("[%s] Post-stream image batch delivery failed: %s", adapter.name, e)

            for media_path, is_voice in non_image_media:
                try:
                    ext = Path(media_path).suffix.lower()
                    if should_send_media_as_audio(event.source.platform, ext, is_voice=is_voice):
                        await adapter.send_voice(
                            chat_id=event.source.chat_id,
                            audio_path=media_path,
                            metadata=_thread_meta,
                        )
                    elif ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=media_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=media_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream media delivery failed: %s", adapter.name, e)

            for file_path in non_image_local:
                try:
                    ext = Path(file_path).suffix.lower()
                    if ext in _VIDEO_EXTS:
                        await adapter.send_video(
                            chat_id=event.source.chat_id,
                            video_path=file_path,
                            metadata=_thread_meta,
                        )
                    else:
                        await adapter.send_document(
                            chat_id=event.source.chat_id,
                            file_path=file_path,
                            metadata=_thread_meta,
                        )
                except Exception as e:
                    logger.warning("[%s] Post-stream file delivery failed: %s", adapter.name, e)

        except Exception as e:
            logger.warning("Post-stream media extraction failed: %s", e)



    async def _run_background_task(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        from run_agent import AIAgent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self.adapters.get(source.platform)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            from hermes_cli.tools_config import _get_platform_tools
            enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
            agent_cfg = user_config.get("agent") or {}
            disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(source=source)
            self._reasoning_config = reasoning_config
            self._service_tier = self._load_service_tier()
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    fallback_model=self._fallback_model,
                )
                try:
                    return agent.run_conversation(
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )
                    except Exception:
                        pass

                # Send media files, routing each by type so a TTS clip
                # arrives as a voice bubble / a clip as a video rather than
                # a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass







    async def _get_telegram_topic_capabilities(self, source: SessionSource) -> dict:
        """Read Telegram private-topic capability flags via Bot API getMe."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "get_me"):
            return {"checked": False}
        try:
            me = await bot.get_me()
        except Exception:
            logger.debug("Failed to fetch Telegram getMe topic capabilities", exc_info=True)
            return {"checked": False}

        def _field(name: str):
            if hasattr(me, name):
                return getattr(me, name)
            api_kwargs = getattr(me, "api_kwargs", None)
            if isinstance(api_kwargs, dict) and name in api_kwargs:
                return api_kwargs.get(name)
            if isinstance(me, dict):
                return me.get(name)
            return None

        return {
            "checked": True,
            "has_topics_enabled": _field("has_topics_enabled"),
            "allows_users_to_create_topics": _field("allows_users_to_create_topics"),
        }

    async def _ensure_telegram_system_topic(self, source: SessionSource) -> None:
        """Create/pin the managed System topic after /topic activation when possible."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is None or not source.chat_id:
            return

        thread_id = None
        create_topic = getattr(adapter, "_create_dm_topic", None)
        if callable(create_topic):
            try:
                thread_id = await create_topic(int(source.chat_id), "System")
            except Exception:
                logger.debug("Failed to create Telegram System topic", exc_info=True)
        if not thread_id:
            return

        message_id = None
        try:
            send_result = await adapter.send(
                source.chat_id,
                "System topic for Hermes commands and status.",
                metadata={"thread_id": str(thread_id)},
            )
            message_id = getattr(send_result, "message_id", None)
        except Exception:
            logger.debug("Failed to send Telegram System topic intro", exc_info=True)
        if not message_id:
            return

        bot = getattr(adapter, "_bot", None)
        if bot is None or not hasattr(bot, "pin_chat_message"):
            return
        try:
            await bot.pin_chat_message(
                chat_id=int(source.chat_id),
                message_id=int(message_id),
                disable_notification=True,
            )
        except Exception:
            logger.debug("Failed to pin Telegram System topic intro", exc_info=True)

    async def _send_telegram_topic_setup_image(self, source: SessionSource) -> None:
        """Send the bundled BotFather Threads Settings screenshot when available."""
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is None or not source.chat_id or not hasattr(adapter, "send_image_file"):
            return
        image_path = Path(__file__).resolve().parent / "assets" / "telegram-botfather-threads-settings.jpg"
        if not image_path.exists():
            return
        try:
            await adapter.send_image_file(
                chat_id=source.chat_id,
                image_path=str(image_path),
                caption="BotFather → Bot Settings → Threads Settings",
                metadata={"thread_id": str(source.thread_id)} if source.thread_id else None,
            )
        except Exception:
            logger.debug("Failed to send Telegram topic setup image", exc_info=True)

    def _sanitize_telegram_topic_title(self, title: str) -> str:
        """Return a Bot API-safe forum topic name from a generated session title."""
        cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
        if not cleaned:
            return "Hermes Chat"
        # Telegram forum topic names are short (currently 1-128 chars). Keep
        # extra room for multi-byte titles and avoid trailing ellipsis churn.
        if len(cleaned) > 120:
            cleaned = cleaned[:117].rstrip() + "..."
        return cleaned

    async def _rename_telegram_topic_for_session_title(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Best-effort rename of a Telegram DM topic when Hermes auto-titles a session."""
        if not await asyncio.to_thread(self._is_telegram_topic_lane, source) or not source.chat_id or not source.thread_id:
            return

        # Operator can fully disable per-topic auto-rename via
        # extra.disable_topic_auto_rename. Useful when topics are managed
        # by the user (ad-hoc Threaded Mode) and auto-rename would
        # overwrite their chosen names every time the auto-title fires.
        if self._telegram_topic_auto_rename_disabled(source):
            return

        # Skip rename when the topic is operator-declared via
        # extra.dm_topics. Those topics have fixed names chosen by the
        # operator (plus optional skill binding); auto-renaming would
        # silently mutate operator config.
        #
        # Check the class, not the instance — getattr() on MagicMock
        # auto-creates attributes, so `hasattr(adapter, "_get_dm_topic_info")`
        # would return True for every test double.
        adapter = self.adapters.get(source.platform) if getattr(self, "adapters", None) else None
        if adapter is not None:
            get_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_info):
                try:
                    operator_topic = get_info(adapter, str(source.chat_id), str(source.thread_id))
                except Exception:
                    operator_topic = None
                # Only treat dict-shaped returns as operator-declared; a
                # bare MagicMock or other sentinel shouldn't count.
                if isinstance(operator_topic, dict):
                    return

        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                )
                if binding and str(binding.get("session_id") or "") != str(session_id):
                    return
            except Exception:
                logger.debug("Failed to verify Telegram topic binding before rename", exc_info=True)
                return

        if adapter is None:
            return
        topic_name = self._sanitize_telegram_topic_title(title)
        try:
            rename_topic = getattr(adapter, "rename_dm_topic", None)
            if rename_topic is not None:
                await rename_topic(
                    chat_id=str(source.chat_id),
                    thread_id=str(source.thread_id),
                    name=topic_name,
                )
                return

            bot = getattr(adapter, "_bot", None)
            edit_forum_topic = getattr(bot, "edit_forum_topic", None) if bot is not None else None
            if edit_forum_topic is None:
                edit_forum_topic = getattr(bot, "editForumTopic", None) if bot is not None else None
            if edit_forum_topic is None:
                return
            try:
                await edit_forum_topic(
                    chat_id=int(source.chat_id),
                    message_thread_id=int(source.thread_id),
                    name=topic_name,
                )
            except (TypeError, ValueError):
                await edit_forum_topic(
                    chat_id=source.chat_id,
                    message_thread_id=source.thread_id,
                    name=topic_name,
                )
        except Exception:
            logger.debug("Failed to rename Telegram topic for auto-generated title", exc_info=True)

    def _telegram_topic_auto_rename_disabled(self, source: SessionSource) -> bool:
        """Return True when operator disabled per-topic auto-rename for this Telegram chat.

        Controlled via ``gateway.platforms.telegram.extra.disable_topic_auto_rename``.
        Default is False (auto-rename enabled, preserves prior behaviour).
        """
        platform_cfg = (
            self.config.platforms.get(source.platform)
            if getattr(self, "config", None) and getattr(self.config, "platforms", None)
            else None
        )
        if platform_cfg is None:
            return False
        extra = getattr(platform_cfg, "extra", None) or {}
        value = extra.get("disable_topic_auto_rename")
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _schedule_telegram_topic_title_rename(
        self,
        source: SessionSource,
        session_id: str,
        title: str,
    ) -> None:
        """Schedule a topic rename from the auto-title background thread."""
        if not title or not self._is_telegram_topic_lane(source):
            return
        if self._telegram_topic_auto_rename_disabled(source):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = getattr(self, "_gateway_loop", None)
        if loop is None or loop.is_closed():
            return
        try:
            copied_source = dataclasses.replace(source)
        except Exception:
            copied_source = source
        future = safe_schedule_threadsafe(
            self._rename_telegram_topic_for_session_title(copied_source, session_id, title),
            loop,
            logger=logger,
            log_message="Telegram topic title rename failed to schedule",
        )
        if future is None:
            return
        def _log_rename_failure(fut) -> None:
            try:
                fut.result()
            except Exception:
                logger.debug("Telegram topic title rename failed", exc_info=True)

        future.add_done_callback(_log_rename_failure)

    _TELEGRAM_CAPABILITY_HINT_COOLDOWN_S = 300.0

    def _should_send_telegram_capability_hint(self, source: SessionSource) -> bool:
        """Rate-limit the BotFather Threads Settings screenshot.

        If a user sends /topic repeatedly while Threads Settings are still
        off, we shouldn't keep re-uploading the screenshot every time.
        """
        if not hasattr(self, "_telegram_capability_hint_ts"):
            self._telegram_capability_hint_ts = {}
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return True
        import time as _time
        now = _time.monotonic()
        last = self._telegram_capability_hint_ts.get(chat_id, 0.0)
        if now - last < self._TELEGRAM_CAPABILITY_HINT_COOLDOWN_S:
            return False
        self._telegram_capability_hint_ts[chat_id] = now
        return True

    def _telegram_topic_help_text(self) -> str:
        return (
            "/topic — enable multi-session DM mode (one bot, many parallel chats)\n"
            "\n"
            "Usage:\n"
            "  /topic             Enable topic mode, or show status if already on\n"
            "  /topic help        Show this message\n"
            "  /topic off         Disable topic mode and clear topic bindings\n"
            "  /topic <id>        Inside a topic: restore a previous session by ID\n"
            "\n"
            "How it works:\n"
            "1. Run /topic once in this DM — Hermes checks BotFather Threads\n"
            "   Settings are enabled and flips on multi-session mode.\n"
            "2. Tap All Messages at the top of the bot and send any message.\n"
            "   Telegram creates a new topic for that message; each topic is\n"
            "   an independent Hermes session (fresh history, fresh context).\n"
            "3. The root DM becomes a system lobby — send /topic, /status,\n"
            "   /help, /usage there. Normal prompts go in a topic.\n"
            "4. /new inside a topic resets just that topic's session.\n"
            "5. /topic <id> inside a topic restores an old session into it."
        )

    async def _disable_telegram_topic_mode_for_chat(self, source: SessionSource) -> str:
        """Cleanly disable topic mode for a chat via /topic off."""
        if not self._session_db:
            from hermes_state import format_session_db_unavailable
            return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))
        chat_id = str(source.chat_id or "")
        if not chat_id:
            return "Could not determine chat ID."
        # No-op if never enabled.
        try:
            currently_enabled = await self._session_db.is_telegram_topic_mode_enabled(
                chat_id=chat_id,
                user_id=str(source.user_id or ""),
            )
        except Exception:
            currently_enabled = False
        if not currently_enabled:
            return "Multi-session topic mode is not currently enabled for this chat."
        try:
            await self._session_db.disable_telegram_topic_mode(chat_id=chat_id)
        except Exception as exc:
            logger.exception("Failed to disable Telegram topic mode")
            return f"Failed to disable topic mode: {exc}"
        # Reset per-chat debounce state so the user doesn't see a stale
        # cooldown on the next activation.
        for attr in ("_telegram_lobby_reminder_ts", "_telegram_capability_hint_ts"):
            store = getattr(self, attr, None)
            if isinstance(store, dict):
                store.pop(chat_id, None)
        return (
            "Multi-session topic mode is now OFF for this chat.\n\n"
            "Existing topics in Telegram aren't removed — they'll just stop "
            "being gated as independent sessions. The root DM works as a "
            "normal Hermes chat again. Run /topic to re-enable later."
        )


    async def _telegram_topic_root_status_message(self, source: SessionSource) -> str:
        lines = [
            "Telegram multi-session topics are enabled.",
            "",
            "To create a new Hermes chat, open All Messages at the top of this "
            "bot interface and send any message there. Telegram will create a "
            "new topic for it.",
            "",
        ]
        try:
            sessions = await self._session_db.list_unlinked_telegram_sessions_for_user(
                chat_id=str(source.chat_id),
                user_id=str(source.user_id),
                limit=10,
            )
        except Exception:
            logger.debug("Failed to list unlinked Telegram sessions", exc_info=True)
            sessions = []

        if sessions:
            lines.append("Previous unlinked sessions:")
            for session in sessions:
                session_id = str(session.get("id") or "")
                title = str(session.get("title") or "Untitled session")
                preview = str(session.get("preview") or "").strip()
                line = f"- {title} — `{session_id}`"
                if preview:
                    line += f" — {preview}"
                lines.append(line)
            lines.extend([
                "",
                "To restore one:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
                f"Example: Send /topic {sessions[0].get('id')} inside a topic.",
            ])
        else:
            lines.extend([
                "No previous unlinked Telegram sessions found.",
                "",
                "To restore a previous session later:",
                "1. Create or open a topic. To create a new one, open All Messages and send any message there.",
                "2. Send /topic <session-id> inside that topic.",
            ])
        return "\n".join(lines)

    async def _restore_telegram_topic_session(self, event: MessageEvent, raw_session_id: str) -> str:
        """Restore an existing Telegram-owned Hermes session into this topic."""
        source = event.source
        session_id = await self._session_db.resolve_session_id(raw_session_id.strip())
        if not session_id:
            return f"Session not found: {raw_session_id.strip()}"

        session = await self._session_db.get_session(session_id)
        if not session:
            return f"Session not found: {raw_session_id.strip()}"
        if str(session.get("source") or "") != "telegram":
            return "That session is not a Telegram session and cannot be restored into this topic."
        if str(session.get("user_id") or "") != str(source.user_id):
            return "That session does not belong to this Telegram user."

        linked = await self._session_db.is_telegram_session_linked_to_topic(session_id=session_id)
        current_binding = await self._session_db.get_telegram_topic_binding(
            chat_id=str(source.chat_id),
            thread_id=str(source.thread_id),
        )
        if linked:
            if not current_binding or current_binding.get("session_id") != session_id:
                return "That session is already linked to another Telegram topic."

        session_key = self._session_key_for_source(source)
        try:
            await self._session_db.bind_telegram_topic(
                chat_id=str(source.chat_id),
                thread_id=str(source.thread_id),
                user_id=str(source.user_id),
                session_key=session_key,
                session_id=session_id,
                managed_mode="restored",
            )
        except ValueError as exc:
            if "already linked" in str(exc):
                return "That session is already linked to another Telegram topic."
            raise

        title = await self._session_db.get_session_title(session_id) or session_id
        last_assistant = None
        try:
            for message in reversed(await self._session_db.get_messages(session_id)):
                if message.get("role") == "assistant" and message.get("content"):
                    last_assistant = str(message.get("content"))
                    break
        except Exception:
            last_assistant = None

        response = f"Session restored: {title}"
        if last_assistant:
            response += f"\n\nLast Hermes message:\n{last_assistant}"
        return response







    async def _execute_mcp_reload(self, event: MessageEvent) -> str:
        """Actually disconnect, reconnect, and notify MCP tool changes.

        Split out from ``_handle_reload_mcp_command`` so the confirmation
        wrapper can invoke the same path whether the user confirmed via
        button, text reply, or has the confirm gate disabled.
        """
        loop = asyncio.get_running_loop()
        try:
            from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, _servers, _lock

            # Capture old server names before shutdown
            with _lock:
                old_servers = set(_servers.keys())

            # Read new config before shutting down, so we know what will be added/removed
            # Shutdown existing connections
            await loop.run_in_executor(None, shutdown_mcp_servers)

            # Reconnect by discovering tools (reads config.yaml fresh)
            new_tools = await loop.run_in_executor(None, discover_mcp_tools)

            # Compute what changed
            with _lock:
                connected_servers = set(_servers.keys())

            added = connected_servers - old_servers
            removed = old_servers - connected_servers
            reconnected = connected_servers & old_servers

            lines = [t("gateway.reload_mcp.header")]
            if reconnected:
                lines.append(t("gateway.reload_mcp.reconnected", names=", ".join(sorted(reconnected))))
            if added:
                lines.append(t("gateway.reload_mcp.added", names=", ".join(sorted(added))))
            if removed:
                lines.append(t("gateway.reload_mcp.removed", names=", ".join(sorted(removed))))
            if not connected_servers:
                lines.append(t("gateway.reload_mcp.none_connected"))
            else:
                lines.append(t("gateway.reload_mcp.tools_available", tools=len(new_tools), servers=len(connected_servers)))

            # Refresh cached agents so existing sessions see new MCP tools on
            # their next turn — without this, the user has to `/new` (which
            # discards conversation history) to pick up tools from a server
            # that was just added or reconnected. The user has already
            # consented to the prompt-cache invalidation via the slash-confirm
            # gate in _handle_reload_mcp_command before we reach this point.
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools
                _cache = getattr(self, "_agent_cache", None)
                _cache_lock = getattr(self, "_agent_cache_lock", None)
                if _cache_lock is not None and _cache:
                    with _cache_lock:
                        for _sess_key, _entry in list(_cache.items()):
                            try:
                                _agent = _entry[0] if isinstance(_entry, tuple) else _entry
                            except Exception:
                                continue
                            if _agent is None:
                                continue
                            # Preserve each cached agent's build-time toolset
                            # selection EXACTLY: a gateway session built with a
                            # restricted enabled_toolsets (e.g. ["safe"]) must
                            # NOT silently gain tools after a reload. This is the
                            # opposite of the interactive CLI/TUI /reload-mcp,
                            # which is a single user re-applying their own config
                            # edit; gateway agents are per-session and may be
                            # deliberately locked down. (Contract is asserted by
                            # test_reload_mcp_preserves_per_agent_toolset_overrides.)
                            refresh_agent_mcp_tools(_agent, quiet_mode=True)
            except Exception as _exc:
                logger.debug(
                    "Failed to update cached agent tools after MCP reload: %s",
                    _exc,
                )

            # Inject a message at the END of the session history so the
            # model knows tools changed on its next turn.  Appended after
            # all existing messages to preserve prompt-cache for the prefix.
            change_parts = []
            if added:
                change_parts.append(f"Added servers: {', '.join(sorted(added))}")
            if removed:
                change_parts.append(f"Removed servers: {', '.join(sorted(removed))}")
            if reconnected:
                change_parts.append(f"Reconnected servers: {', '.join(sorted(reconnected))}")
            tool_summary = f"{len(new_tools)} MCP tool(s) now available" if new_tools else "No MCP tools available"
            change_detail = ". ".join(change_parts) + ". " if change_parts else ""
            reload_msg = {
                "role": "user",
                "content": f"[IMPORTANT: MCP servers have been reloaded. {change_detail}{tool_summary}. The tool list for this conversation has been updated accordingly.]",
            }
            try:
                session_entry = self.session_store.get_or_create_session(event.source)
                self.session_store.append_to_transcript(
                    session_entry.session_id, reload_msg
                )
            except Exception:
                pass  # Best-effort; don't fail the reload over a transcript write

            return "\n".join(lines)

        except Exception as e:
            logger.warning("MCP reload failed: %s", e)
            return t("gateway.reload_mcp.failed", error=e)



    # ------------------------------------------------------------------
    # Slash-command confirmation primitive (generic)
    # ------------------------------------------------------------------
    # Used by slash commands that have a non-destructive but expensive
    # side effect worth an explicit user confirmation (currently only
    # /reload-mcp, which invalidates the prompt cache).  Two delivery
    # paths:
    #   1. Button UI — adapters that override ``send_slash_confirm``
    #      (Telegram, Discord, Slack, Matrix, Feishu) render three
    #      inline buttons.  The adapter routes the button click back via
    #      ``tools.slash_confirm.resolve(session_key, confirm_id, choice)``.
    #   2. Text fallback — adapters that don't override the hook get a
    #      plain text prompt.  Users reply with /approve, /always, or
    #      /cancel; the early intercept in ``_handle_message`` matches
    #      those replies against ``tools.slash_confirm.get_pending()``.

    async def _maybe_confirm_destructive_slash(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        detail: str,
        execute,
    ) -> Union[str, "EphemeralReply", None]:
        """Gate a destructive session slash command (/new, /reset, /undo).

        ``execute`` is an async callable ``execute() -> str | EphemeralReply``
        that performs the destructive action.  If the
        ``approvals.destructive_slash_confirm`` config gate is off, ``execute``
        runs immediately (returning its result).  Otherwise this routes
        through ``_request_slash_confirm`` — native yes/no buttons on
        Telegram/Discord/Slack, text fallback elsewhere.

        Three-option resolution:

          - ``once``  — run ``execute`` and return its result
          - ``always`` — persist ``approvals.destructive_slash_confirm: false``,
                        then run ``execute``
          - ``cancel`` — return a "cancelled" message; do not run ``execute``
        """
        # Gate check.
        confirm_required = True
        try:
            cfg = self._read_user_config()
            approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
            if isinstance(approvals, dict):
                confirm_required = bool(approvals.get("destructive_slash_confirm", True))
        except Exception:
            pass

        if not confirm_required:
            return await execute()

        session_key = self._session_key_for_source(event.source)

        async def _on_confirm(choice: str):
            if choice == "cancel":
                return f"🟡 /{command} cancelled. Conversation unchanged."
            if choice == "always":
                try:
                    from cli import save_config_value
                    save_config_value("approvals.destructive_slash_confirm", False)
                    logger.info(
                        "User opted out of destructive slash confirm (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to persist destructive_slash_confirm=false: %s", exc,
                    )
            result = await execute()
            if choice == "always":
                note = (
                    "\n\nℹ️ Future /clear, /new, /reset, and /undo will run "
                    "without confirmation. Re-enable via "
                    "`approvals.destructive_slash_confirm: true` in config.yaml."
                )
                if isinstance(result, str):
                    return result + note
                # EphemeralReply or other — leave untouched; the opt-out note
                # would otherwise mangle structured replies.  The persist itself
                # already happened above; user gets the same UX next time.
                return result
            return result

        _p = self._typed_command_prefix_for(event.source.platform)
        prompt_message = (
            f"⚠️ **Confirm /{command}**\n\n"
            f"{detail}\n\n"
            "Choose:\n"
            "• **Approve Once** — proceed this time only\n"
            "• **Always Approve** — proceed and silence this prompt permanently\n"
            "• **Cancel** — keep current conversation\n\n"
            f"_Text fallback: reply `{_p}approve`, `{_p}always`, or `{_p}cancel`._"
        )
        return await self._request_slash_confirm(
            event=event,
            command=command,
            title=title,
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _request_slash_confirm(
        self,
        *,
        event: MessageEvent,
        command: str,
        title: str,
        message: str,
        handler,
    ) -> Optional[str]:
        """Ask the user to confirm an expensive slash command.

        ``handler`` is an async callable ``handler(choice: str) -> str``
        where ``choice`` is ``"once"``, ``"always"``, or ``"cancel"``.
        The handler runs on the event loop when the user responds; its
        return value is sent back as a gateway message.

        Returns a short acknowledgment string to send immediately (before
        the user's response).  If buttons rendered successfully the ack
        is ``None`` (buttons are self-explanatory); if we fell back to
        text the message itself IS the ack.
        """
        from tools import slash_confirm as _slash_confirm_mod

        source = event.source
        session_key = self._session_key_for_source(source)
        # Bare-runner test harnesses (object.__new__(GatewayRunner)) skip
        # __init__ and don't have the counter attribute — fall back to a
        # local counter so tests don't AttributeError.  Real runs always
        # have the instance attribute.
        counter = getattr(self, "_slash_confirm_counter", None)
        if counter is None:
            import itertools as _itertools
            counter = _itertools.count(1)
            self._slash_confirm_counter = counter
        confirm_id = f"{next(counter)}"

        # Register the pending confirm FIRST so a super-fast button click
        # cannot race the send_slash_confirm return.
        _slash_confirm_mod.register(session_key, confirm_id, command, handler)

        adapter = self.adapters.get(source.platform)
        metadata = self._thread_metadata_for_source(source, self._reply_anchor_for_event(event))

        used_buttons = False
        if adapter is not None:
            try:
                button_result = await adapter.send_slash_confirm(
                    chat_id=source.chat_id,
                    title=title,
                    message=message,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    metadata=metadata,
                )
                if button_result and getattr(button_result, "success", False):
                    used_buttons = True
            except Exception as exc:
                logger.debug(
                    "send_slash_confirm failed for %s on %s: %s",
                    command, source.platform, exc,
                )

        if used_buttons:
            # Buttons rendered — no redundant text ack.
            return None
        # Text fallback — return the prompt message as the direct reply.
        return message

    def _read_user_config(self) -> Dict[str, Any]:
        """Read the user's raw config.yaml (cached) for gate lookups.

        Used by slash-confirm gates that must reflect on-disk state changes
        (e.g. a prior "Always Approve" click) without a gateway restart.
        """
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
            return cfg if isinstance(cfg, dict) else {}
        except Exception:
            return {}

    def _thread_metadata_for_source(
        self,
        source,
        reply_to_message_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build the metadata dict platforms need for thread-aware replies."""
        return self._thread_metadata_for_target(
            getattr(source, "platform", None),
            getattr(source, "chat_id", None),
            getattr(source, "thread_id", None),
            chat_type=getattr(source, "chat_type", None),
            reply_to_message_id=reply_to_message_id or getattr(source, "message_id", None),
        )

    def _thread_metadata_for_target(
        self,
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build thread metadata for synthetic sends that only have routing state."""
        if thread_id is None:
            return None
        metadata: Dict[str, Any] = {"thread_id": thread_id}
        if self._is_telegram_dm_topic_target(
            platform,
            chat_id,
            thread_id,
            chat_type=chat_type,
            adapter=adapter,
        ):
            metadata["telegram_dm_topic_reply_fallback"] = True
            # Telegram DM topic lanes need direct_messages_topic_id in metadata
            # so synthetic/queued messages (goal continuations, status notices)
            # route to the correct topic even when reply anchor is unavailable.
            tid = str(thread_id)
            if tid and tid not in {"", "1"}:
                metadata["direct_messages_topic_id"] = tid
            if reply_to_message_id is not None:
                metadata["telegram_reply_to_message_id"] = str(reply_to_message_id)
        return metadata

    @staticmethod
    def _is_telegram_dm_topic_target(
        platform: Optional[Platform],
        chat_id: Optional[str],
        thread_id: Optional[str],
        *,
        chat_type: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> bool:
        """Return True when a target is a Telegram private DM topic lane."""
        if platform != Platform.TELEGRAM or thread_id is None:
            return False
        if chat_type == "dm":
            return True
        # Inspect operator-declared DM topics via the adapter's lookup. Resolve
        # the method on the CLASS, not the instance: getattr() on a MagicMock
        # auto-creates a callable child for any attribute, so an instance-level
        # lookup would report a DM topic for every test double. Only a
        # dict-shaped return counts as an operator-declared topic — a bare
        # MagicMock or other sentinel must not. Mirrors the guard in
        # _rename_telegram_topic_for_session_title.
        if adapter is not None and chat_id:
            get_dm_topic_info = getattr(type(adapter), "_get_dm_topic_info", None)
            if callable(get_dm_topic_info):
                try:
                    topic_info = get_dm_topic_info(adapter, str(chat_id), str(thread_id))
                except Exception:
                    logger.debug("Failed to inspect Telegram DM topic metadata", exc_info=True)
                else:
                    return isinstance(topic_info, dict)
        return False

    @staticmethod
    def _reply_anchor_for_event(event: MessageEvent) -> Optional[str]:
        """Return the platform-specific reply anchor for GatewayRunner sends."""
        return _reply_anchor_for_event(event)


    # ------------------------------------------------------------------
    # /approve & /deny — explicit dangerous-command approval
    # ------------------------------------------------------------------

    _APPROVAL_TIMEOUT_SECONDS = 300  # 5 minutes



    # Built-in messaging platforms where the ``/update`` command is allowed.
    # ACP, API server, and webhooks are programmatic interfaces that should
    # not trigger system updates.  Plugin-migrated platforms (discord,
    # mattermost, teams, irc, line, …) are NOT listed here — they declare
    # ``allow_update_command=True`` on their ``PlatformEntry`` and are
    # honored via the registry fallback at ``_handle_update_command`` below.
    _UPDATE_ALLOWED_PLATFORMS = frozenset({
        Platform.TELEGRAM, Platform.SLACK, Platform.WHATSAPP,
        Platform.SIGNAL, Platform.MATRIX,
        Platform.EMAIL, Platform.SMS, Platform.DINGTALK,
        Platform.FEISHU, Platform.WECOM, Platform.WECOM_CALLBACK, Platform.WEIXIN, Platform.BLUEBUBBLES, Platform.QQBOT, Platform.LOCAL,
    })



    def _schedule_update_notification_watch(self) -> None:
        """Ensure a background task is watching for update completion."""
        existing_task = getattr(self, "_update_notification_task", None)
        if existing_task and not existing_task.done():
            return

        try:
            self._update_notification_task = asyncio.create_task(
                self._watch_update_progress()
            )
        except RuntimeError:
            logger.debug("Skipping update notification watcher: no running event loop")

    async def _watch_update_progress(
        self,
        poll_interval: float = 2.0,
        stream_interval: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the
        user periodically.  Detects ``.update_prompt.json`` (written by the
        update process when it needs user input) and forwards the prompt to
        the messenger.  The user's next message is intercepted by
        ``_handle_message`` and written to ``.update_response``.
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        prompt_path = _hermes_home / ".update_prompt.json"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # Resolve the adapter and chat_id for sending messages
        adapter = None
        chat_id = None
        session_key = None
        metadata = None
        for path in (claimed_path, pending_path):
            if path.exists():
                try:
                    pending = json.loads(path.read_text())
                    platform_str = pending.get("platform")
                    chat_id = pending.get("chat_id")
                    chat_type = pending.get("chat_type")
                    session_key = pending.get("session_key")
                    thread_id = pending.get("thread_id")
                    message_id = pending.get("message_id")
                    if platform_str and chat_id:
                        platform = Platform(platform_str)
                        adapter = self.adapters.get(platform)
                        metadata = self._thread_metadata_for_target(
                            platform,
                            chat_id,
                            thread_id,
                            chat_type=chat_type,
                            reply_to_message_id=message_id,
                            adapter=adapter,
                        )
                        # Fallback session key if not stored (old pending files)
                        if not session_key:
                            session_key = f"{platform_str}:{chat_id}"
                    break
                except Exception:
                    pass

        if not adapter or not chat_id:
            logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
            # Fall back to completion-only: wait for the exit code and send the
            # final notification. _send_update_notification re-resolves the
            # adapter on every call, so when the target platform is still
            # reconnecting it returns False and keeps the markers. Keep polling
            # until it actually delivers (returns True) instead of giving up
            # after the first completion check — otherwise a platform that
            # reconnects a few seconds after completion never gets notified.
            while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
                if exit_code_path.exists() and await self._send_update_notification():
                    return
                await asyncio.sleep(poll_interval)
            if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
                exit_code_path.write_text("124")
                await self._send_update_notification()
            return

        def _strip_ansi(text: str) -> str:
            return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)

        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            """Send buffered output to the user."""
            nonlocal buffer, last_stream_time
            if not buffer.strip():
                buffer = ""
                return
            # Chunk to fit message limits (Telegram: 4096, others: generous)
            clean = _strip_ansi(buffer).strip()
            buffer = ""
            last_stream_time = loop.time()
            if not clean:
                return
            # Split into chunks if too long
            max_chunk = 3500
            chunks = [clean[i:i + max_chunk] for i in range(0, len(clean), max_chunk)]
            for chunk in chunks:
                try:
                    await adapter.send(
                        chat_id,
                        f"```\n{chunk}\n```",
                        metadata=_non_conversational_metadata(metadata, platform=platform),
                    )
                except Exception as e:
                    logger.debug("Update stream send failed: %s", e)

        while loop.time() < deadline:
            # Check for completion
            if exit_code_path.exists():
                # Read any remaining output
                if output_path.exists():
                    try:
                        content = output_path.read_text()
                        if len(content) > bytes_sent:
                            buffer += content[bytes_sent:]
                            bytes_sent = len(content)
                    except OSError:
                        pass
                await _flush_buffer()

                # Send final status
                try:
                    exit_code_raw = exit_code_path.read_text().strip() or "1"
                    exit_code = int(exit_code_raw)
                    if exit_code == 0:
                        await adapter.send(
                            chat_id,
                            "✅ Hermes update finished.",
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    else:
                        await adapter.send(
                            chat_id,
                            "❌ Hermes update failed (exit code {}).".format(exit_code),
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                except Exception as e:
                    logger.warning("Update final notification failed: %s", e)

                # Cleanup
                for p in (pending_path, claimed_path, output_path,
                          exit_code_path, prompt_path):
                    p.unlink(missing_ok=True)
                (_hermes_home / ".update_response").unlink(missing_ok=True)
                self._update_prompt_pending.pop(session_key, None)
                return

            # Check for new output
            if output_path.exists():
                try:
                    content = output_path.read_text()
                    if len(content) > bytes_sent:
                        buffer += content[bytes_sent:]
                        bytes_sent = len(content)
                except OSError:
                    pass

            # Flush buffer periodically
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()

            # Check for prompts — only forward if we haven't already sent
            # one that's still awaiting a response.  Without this guard the
            # watcher would re-read the same .update_prompt.json every poll
            # cycle and spam the user with duplicate prompt messages.
            if (prompt_path.exists() and session_key
                    and not self._update_prompt_pending.get(session_key)):
                try:
                    prompt_data = json.loads(prompt_path.read_text())
                    prompt_text = prompt_data.get("prompt", "")
                    default = prompt_data.get("default", "")
                    if prompt_text:
                        # Flush any buffered output first so the user sees
                        # context before the prompt
                        await _flush_buffer()
                        # Try platform-native buttons first (Discord, Telegram)
                        sent_buttons = False
                        if getattr(type(adapter), "send_update_prompt", None) is not None:
                            try:
                                await adapter.send_update_prompt(
                                    chat_id=chat_id,
                                    prompt=prompt_text,
                                    default=default,
                                    session_key=session_key,
                                    metadata=_non_conversational_metadata(metadata, platform=platform),
                                )
                                sent_buttons = True
                            except Exception as btn_err:
                                logger.debug("Button-based update prompt failed: %s", btn_err)
                        if not sent_buttons:
                            default_hint = f" (default: {default})" if default else ""
                            _p = getattr(adapter, "typed_command_prefix", "/")
                            await adapter.send(
                                chat_id,
                                f"⚕ **Update needs your input:**\n\n"
                                f"{prompt_text}{default_hint}\n\n"
                                f"Reply `{_p}approve` (yes) or `{_p}deny` (no), "
                                f"or type your answer directly.",
                                metadata=_non_conversational_metadata(metadata, platform=platform),
                            )
                        # Keep the prompt marker on disk until the user
                        # answers. If the gateway restarts mid-prompt, the
                        # next watcher can recover by re-forwarding it from
                        # disk. Duplicate sends in the same process are
                        # still suppressed by _update_prompt_pending.
                        self._update_prompt_pending[session_key] = True
                        # .update_response to continue — it doesn't re-check
                        logger.info("Forwarded update prompt to %s: %s", session_key, prompt_text[:80])
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)

            await asyncio.sleep(poll_interval)

        # Timeout
        if not exit_code_path.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            exit_code_path.write_text("124")
            await _flush_buffer()
            try:
                await adapter.send(
                    chat_id,
                    "❌ Hermes update timed out after 30 minutes.",
                    metadata=_non_conversational_metadata(metadata, platform=platform),
                )
            except Exception:
                pass
            for p in (pending_path, claimed_path, output_path,
                      exit_code_path, prompt_path):
                p.unlink(missing_ok=True)
            (_hermes_home / ".update_response").unlink(missing_ok=True)
            self._update_prompt_pending.pop(session_key, None)

    async def _send_update_notification(self) -> bool:
        """If an update finished, notify the user.

        Returns False when the update is still running so a caller can retry
        later. Returns True after a definitive send/skip decision.

        This is the legacy notification path used when the streaming watcher
        cannot resolve the adapter (e.g. after a gateway restart where the
        platform hasn't reconnected yet).
        """
        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"

        if not pending_path.exists() and not claimed_path.exists():
            return False

        cleanup = True
        active_pending_path = claimed_path
        try:
            if pending_path.exists():
                try:
                    pending_path.replace(claimed_path)
                except FileNotFoundError:
                    if not claimed_path.exists():
                        return True
            elif not claimed_path.exists():
                return True

            pending = json.loads(claimed_path.read_text())
            platform_str = pending.get("platform")
            chat_id = pending.get("chat_id")
            chat_type = pending.get("chat_type")
            thread_id = pending.get("thread_id")
            message_id = pending.get("message_id")

            if not exit_code_path.exists():
                logger.info("Update notification deferred: update still running")
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            exit_code_raw = exit_code_path.read_text().strip() or "1"
            exit_code = int(exit_code_raw)

            # Read the captured update output
            output = ""
            if output_path.exists():
                output = output_path.read_text()

            # Resolve adapter
            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)

            if not adapter and chat_id:
                # The update finished, but the target platform has not
                # reconnected yet (common right after the restart that
                # `hermes update` triggers). Treating "adapter missing" as a
                # definitive skip would delete the markers and silently lose the
                # completion notification — the user never learns whether the
                # update succeeded or timed out. Preserve the markers instead so
                # a later retry (the watcher poll loop, or the next gateway
                # startup) can deliver the result once the adapter is back.
                logger.info(
                    "Update notification deferred: %s adapter not connected yet",
                    platform_str,
                )
                cleanup = False
                active_pending_path = pending_path
                claimed_path.replace(pending_path)
                return False

            if adapter and chat_id:
                metadata = self._thread_metadata_for_target(
                    platform,
                    chat_id,
                    thread_id,
                    chat_type=chat_type,
                    reply_to_message_id=message_id,
                    adapter=adapter,
                )
                # Strip ANSI escape codes for clean display
                output = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
                if output:
                    if len(output) > 3500:
                        output = "…" + output[-3500:]
                    if exit_code == 0:
                        msg = f"✅ Hermes update finished.\n\n```\n{output}\n```"
                    else:
                        msg = f"❌ Hermes update failed.\n\n```\n{output}\n```"
                elif exit_code == 0:
                    msg = "✅ Hermes update finished successfully."
                else:
                    msg = "❌ Hermes update failed. Check the gateway logs or run `hermes update` manually for details."
                await adapter.send(
                    chat_id,
                    msg,
                    metadata=_non_conversational_metadata(metadata, platform=platform),
                )
                logger.info(
                    "Sent post-update notification to %s:%s (exit=%s)",
                    platform_str,
                    chat_id,
                    exit_code,
                )
        except Exception as e:
            logger.warning("Post-update notification failed: %s", e)
        finally:
            if cleanup:
                active_pending_path.unlink(missing_ok=True)
                claimed_path.unlink(missing_ok=True)
                output_path.unlink(missing_ok=True)
                exit_code_path.unlink(missing_ok=True)

        return True

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None

        try:
            data = json.loads(notify_path.read_text())
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            chat_type = data.get("chat_type")
            thread_id = data.get("thread_id")
            message_id = data.get("message_id")

            if not platform_str or not chat_id:
                return None

            platform = Platform(platform_str)
            adapter = self.adapters.get(platform)
            if not adapter:
                logger.debug(
                    "Restart notification skipped: %s adapter not connected",
                    platform_str,
                )
                return None

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false",
                    platform_str,
                )
                return None

            metadata = self._thread_metadata_for_target(
                platform,
                chat_id,
                thread_id,
                chat_type=chat_type,
                reply_to_message_id=message_id,
                adapter=adapter,
            )
            result = await adapter.send(
                str(chat_id),
                "♻ Gateway restarted successfully. Your session continues.",
                metadata=_non_conversational_metadata(metadata, platform=platform),
            )
            # adapter.send() catches provider errors (e.g. "Chat not found")
            # and returns SendResult(success=False) rather than raising, so
            # we must inspect the result before claiming success — otherwise
            # the log line is misleading and hides real delivery failures.
            if result is not None and getattr(result, "success", True) is False:
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s",
                    platform_str,
                    chat_id,
                    getattr(result, "error", "send returned success=False"),
                )
                return None

            logger.info(
                "Sent restart notification to %s:%s",
                platform_str,
                chat_id,
            )
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)

    async def _send_home_channel_startup_notifications(
        self,
        *,
        skip_targets: Optional[set[tuple[str, str, Optional[str]]]] = None,
    ) -> set[tuple[str, str, Optional[str]]]:
        """Notify configured home channels that the gateway is back online.

        The notification is best-effort and sent once per connected platform
        home channel. ``skip_targets`` lets startup avoid duplicate messages
        when a more specific restart notification is queued for the same chat.
        """
        delivered: set[tuple[str, str, Optional[str]]] = set()
        skipped = skip_targets or set()
        message = "♻️ Gateway online — Hermes is back and ready."

        for platform, adapter in self.adapters.items():
            home = self.config.get_home_channel(platform)
            if not home or not home.chat_id:
                continue

            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Home-channel startup notification suppressed: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue

            target = (platform.value, str(home.chat_id), str(home.thread_id) if home.thread_id else None)
            if target in skipped or target in delivered:
                continue

            try:
                metadata = self._thread_metadata_for_target(
                    platform,
                    home.chat_id,
                    home.thread_id,
                    adapter=adapter,
                )
                if metadata:
                    result = await adapter.send(
                        str(home.chat_id),
                        message,
                        metadata=_non_conversational_metadata(metadata, platform=platform),
                    )
                else:
                    _startup_meta = _non_conversational_metadata(platform=platform)
                    if _startup_meta:
                        result = await adapter.send(
                            str(home.chat_id),
                            message,
                            metadata=_startup_meta,
                        )
                    else:
                        result = await adapter.send(str(home.chat_id), message)
                if result is not None and getattr(result, "success", True) is False:
                    logger.warning(
                        "Home-channel startup notification failed for %s:%s: %s",
                        platform.value,
                        home.chat_id,
                        getattr(result, "error", "send returned success=False"),
                    )
                    continue

                delivered.add(target)
                logger.info(
                    "Sent home-channel startup notification to %s:%s",
                    platform.value,
                    home.chat_id,
                )
            except Exception as exc:
                logger.warning(
                    "Home-channel startup notification failed for %s:%s: %s",
                    platform.value,
                    home.chat_id,
                    exc,
                )

        return delivered

    def _set_session_env(self, context: SessionContext) -> list:
        """Set session context variables for the current async task.

        Uses ``contextvars`` instead of ``os.environ`` so that concurrent
        gateway messages cannot overwrite each other's session state.

        Returns a list of reset tokens; pass them to ``_clear_session_env``
        in a ``finally`` block.
        """
        from gateway.session_context import set_session_vars
        # Propagate the adapter's async-delivery capability so async tools
        # (terminal notify_on_complete / watch_patterns, delegate_task
        # background=True) know whether this channel can wake a later turn.
        # Default True keeps CLI / unknown paths working; stateless adapters
        # (api_server) declare supports_async_delivery=False. Use getattr so
        # bare runners built via object.__new__ (tests) without self.adapters
        # don't blow up — they simply default to supported.
        _adapters = getattr(self, "adapters", None) or {}
        _adapter = _adapters.get(context.source.platform)
        _async_delivery = getattr(_adapter, "supports_async_delivery", True)
        return set_session_vars(
            platform=context.source.platform.value,
            chat_id=context.source.chat_id,
            chat_name=context.source.chat_name or "",
            thread_id=str(context.source.thread_id) if context.source.thread_id else "",
            user_id=str(context.source.user_id) if context.source.user_id else "",
            user_name=str(context.source.user_name) if context.source.user_name else "",
            session_key=context.session_key,
            message_id=str(context.source.message_id) if context.source.message_id else "",
            async_delivery=_async_delivery,
        )

    def _clear_session_env(self, tokens: list) -> None:
        """Restore session context variables to their pre-handler values."""
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)

    async def _run_in_executor_with_context(self, func, *args):
        """Run blocking work in the thread pool while preserving session contextvars."""
        loop = asyncio.get_running_loop()
        ctx = copy_context()
        return await loop.run_in_executor(
            self._get_executor(),
            ctx.run,
            func,
            *args,
        )

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return the gateway-owned executor for blocking agent work."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._executor_lock = lock

        with lock:
            if getattr(self, "_executor_closing", False):
                raise RuntimeError("Gateway is shutting down; executor unavailable")
            executor = getattr(self, "_executor", None)
            if executor is None or getattr(executor, "_shutdown", False):
                executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=10,
                    thread_name_prefix="hermes-gateway",
                )
                self._executor = executor
            return executor

    def _shutdown_executor(self) -> None:
        """Stop the gateway-owned executor without touching the loop default."""
        lock = getattr(self, "_executor_lock", None)
        if lock is None:
            return

        with lock:
            self._executor_closing = True
            executor = getattr(self, "_executor", None)
            self._executor = None

        if executor is None:
            return

        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)

    def _decide_image_input_mode(self) -> str:
        """Resolve the image-input routing for the currently active model.

        Returns ``"native"`` (attach pixels on the user turn) or ``"text"``
        (pre-analyze with vision_analyze and prepend the description). See
        agent/image_routing.py for the full decision table.

        The active provider/model are read from config.yaml so the decision
        tracks ``/model`` switches automatically on the next message.
        """
        try:
            from agent.image_routing import decide_image_input_mode
            from agent.auxiliary_client import _read_main_model, _read_main_provider
            from hermes_cli.config import load_config

            cfg = load_config()
            provider = _read_main_provider()
            model = _read_main_model()
            return decide_image_input_mode(provider, model, cfg)
        except Exception as exc:
            logger.debug("image_routing: decision failed, falling back to text — %s", exc)
            return "text"

    async def _enrich_message_with_vision(
        self,
        user_text: str,
        image_paths: List[str],
    ) -> str:
        """
        Auto-analyze user-attached images with the vision tool and prepend
        the descriptions to the message text.

        Each image is analyzed with a general-purpose prompt.  The resulting
        description *and* the local cache path are injected so the model can:
          1. Immediately understand what the user sent (no extra tool call).
          2. Re-examine the image with vision_analyze if it needs more detail.

        Args:
            user_text:   The user's original caption / message text.
            image_paths: List of local file paths to cached images.

        Returns:
            The enriched message string with vision descriptions prepended.
        """
        from tools.vision_tools import vision_analyze_tool
        from agent.memory_manager import sanitize_context

        analysis_prompt = (
            "Describe everything visible in this image in thorough detail. "
            "Include any text, code, data, objects, people, layout, colors, "
            "and any other notable visual information."
        )

        enriched_parts = []
        for path in image_paths:
            try:
                logger.debug("Auto-analyzing user image: %s", path)
                result_json = await vision_analyze_tool(
                    image_url=path,
                    user_prompt=analysis_prompt,
                )
                result = json.loads(result_json)
                if result.get("success"):
                    description = result.get("analysis", "")
                    description = sanitize_context(description)
                    enriched_parts.append(
                        f"[The user sent an image~ Here's what I can see:\n{description}]\n"
                        f"[If you need a closer look, use vision_analyze with "
                        f"image_url: {path} ~]"
                    )
                else:
                    enriched_parts.append(
                        "[The user sent an image but I couldn't quite see it "
                        "this time (>_<) You can try looking at it yourself "
                        f"with vision_analyze using image_url: {path}]"
                    )
            except Exception as e:
                logger.error("Vision auto-analysis error: %s", e)
                enriched_parts.append(
                    f"[The user sent an image but something went wrong when I "
                    f"tried to look at it~ You can try examining it yourself "
                    f"with vision_analyze using image_url: {path}]"
                )

        # Combine: vision descriptions first, then the user's original text
        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            if user_text:
                return f"{prefix}\n\n{user_text}"
            return prefix
        return user_text

    async def _enrich_message_with_transcription(
        self,
        user_text: str,
        audio_paths: List[str],
    ) -> tuple[str, List[str]]:
        """
        Auto-transcribe user voice/audio messages using the configured STT provider
        and prepend the transcript to the message text.

        Args:
            user_text:   The user's original caption / message text.
            audio_paths: List of local file paths to cached audio files.

        Returns:
            A tuple of ``(enriched_text, successful_transcripts)``:
              - ``enriched_text``: the message string with transcription wrappers
                prepended (same as before).
              - ``successful_transcripts``: the raw transcript strings for audio
                clips that were successfully transcribed, in input order. Empty
                list if every clip failed or STT is disabled. Callers can use
                this to echo transcripts back to the user before the agent loop.
        """
        if not getattr(self.config, "stt_enabled", True):
            notes = []
            for path in audio_paths:
                abs_path = os.path.abspath(path)
                duration_str = await _probe_audio_duration(abs_path)
                if duration_str:
                    notes.append(
                        f"[The user sent a voice message: {abs_path} (duration: {duration_str})]"
                    )
                else:
                    notes.append(f"[The user sent a voice message: {abs_path}]")
            if not notes:
                return user_text, []
            prefix = "\n\n".join(notes)
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, []
            if user_text:
                return f"{prefix}\n\n{user_text}", []
            return prefix, []

        from tools.transcription_tools import transcribe_audio

        enriched_parts = []
        successful_transcripts: List[str] = []
        for path in audio_paths:
            try:
                logger.debug("Transcribing user voice: %s", path)
                result = await asyncio.to_thread(transcribe_audio, path)
                if result["success"]:
                    transcript = result["transcript"]
                    successful_transcripts.append(transcript)
                    # Pass the transcript through as a plain quoted line. The
                    # earlier wording ("The user sent a voice message~ Here's
                    # what they said: ...") read as a meta-instruction and made
                    # the LLM volunteer commentary about voice mode rather than
                    # reply to the content.
                    enriched_parts.append(f'"{transcript}"')
                else:
                    error = result.get("error", "unknown error")
                    # All failure branches: a single, minimal, neutral marker.
                    # Do NOT mention "no STT provider configured", "setup
                    # instructions", or the "hermes-agent-setup" skill, and do
                    # NOT claim a direct message was sent — those phrases get
                    # persisted in conversation history and poison every later
                    # turn, so the model keeps volunteering STT-setup advice
                    # even after transcription starts working. The cause is
                    # logged for operator diagnosis but kept out of the
                    # LLM-visible prompt.
                    logger.info("Voice transcription failed for %s: %s", path, error)
                    enriched_parts.append("[voice message could not be transcribed]")
            except Exception as e:
                logger.error("Transcription error: %s", e)
                enriched_parts.append("[voice message could not be transcribed]")

        if enriched_parts:
            prefix = "\n\n".join(enriched_parts)
            # Strip the empty-content placeholder from the Discord adapter
            # when we successfully transcribed the audio — it's redundant.
            _placeholder = "(The user sent a message with no text content)"
            if user_text and user_text.strip() == _placeholder:
                return prefix, successful_transcripts
            if user_text:
                return f"{prefix}\n\n{user_text}", successful_transcripts
            return prefix, successful_transcripts
        return user_text, successful_transcripts

    async def _dequeue_pending_with_transcription(
        self,
        adapter,
        session_key: str,
        source,
    ) -> str | None:
        """Dequeue a pending queued message, auto-transcribing audio media.

        When a voice/audio message arrives during an active agent run, the
        adapter stores the event in its pending queue and signals an interrupt
        (see base.BaseAdapter.handle_message). The adapter path bypasses
        _handle_message entirely, so the normal STT pipeline at message-receive
        time never runs.

        This helper fills that gap: when the dequeued event has audio media,
        we transcribe inline, echo the raw transcript back to the user (same
        "🎙️" format as the fresh-message path), and return enriched text.
        Non-audio events fall back to _build_media_placeholder, matching the
        original _dequeue_pending_text behavior.
        """
        event = adapter.get_pending_message(session_key)
        if not event:
            return None

        text = event.text or ""

        audio_paths: List[str] = []
        media_urls = getattr(event, "media_urls", None) or []
        media_types = getattr(event, "media_types", None) or []
        for i, path in enumerate(media_urls):
            mtype = media_types[i] if i < len(media_types) else ""
            is_audio = (
                mtype.startswith("audio/")
                or getattr(event, "message_type", None) in (MessageType.VOICE, MessageType.AUDIO)
            )
            if is_audio:
                audio_paths.append(path)

        if audio_paths:
            enriched_text, successful_transcripts = await self._enrich_message_with_transcription(
                text, audio_paths,
            )
            # Echo raw transcripts back to the user so voice interrupts
            # feel identical to fresh voice messages.
            if successful_transcripts:
                echo_adapter = self.adapters.get(source.platform)
                echo_meta = {"thread_id": source.thread_id} if source.thread_id else None
                if echo_adapter:
                    for tx in successful_transcripts:
                        try:
                            await echo_adapter.send(
                                source.chat_id,
                                f'🎙️ "{tx}"',
                                metadata=echo_meta,
                            )
                        except Exception as echo_exc:
                            logger.debug(
                                "Transcript echo failed (non-fatal): %s", echo_exc,
                            )
            return enriched_text or None

        # Non-audio fallback: preserve original _dequeue_pending_text semantics.
        if not text and media_urls:
            text = _build_media_placeholder(event)
        return text or None

    def _build_process_event_source(self, evt: dict):
        """Resolve the canonical source for a synthetic background-process event.

        Prefer the persisted session-store origin for the event's session key.
        Falling back to the currently active foreground event is what causes
        cross-topic bleed, so don't do that.
        """
        from gateway.session import SessionSource

        session_key = str(evt.get("session_key") or "").strip()
        derived_platform = ""
        derived_chat_type = ""
        derived_chat_id = ""

        if session_key:
            try:
                self.session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                if entry and getattr(entry, "origin", None):
                    return entry.origin
            except Exception as exc:
                logger.debug(
                    "Synthetic process-event session-store lookup failed for %s: %s",
                    session_key,
                    exc,
                )

            cached_source = self._get_cached_session_source(session_key)
            if cached_source is not None:
                return cached_source

            _parsed = _parse_session_key(session_key)
            if _parsed:
                derived_platform = _parsed["platform"]
                derived_chat_type = _parsed["chat_type"]
                derived_chat_id = _parsed["chat_id"]

        platform_name = str(evt.get("platform") or derived_platform or "").strip().lower()
        chat_type = str(evt.get("chat_type") or derived_chat_type or "").strip().lower()
        chat_id = str(evt.get("chat_id") or derived_chat_id or "").strip()
        if not platform_name or not chat_type or not chat_id:
            return None

        try:
            platform = Platform(platform_name)
            # Reject arbitrary strings that create dynamic pseudo-members.
            # Built-in platforms are always valid; plugin platforms must be
            # registered in the platform registry.
            if platform.value not in _BUILTIN_PLATFORM_VALUES:
                try:
                    from gateway.platform_registry import platform_registry
                    if not platform_registry.is_registered(platform.value):
                        raise ValueError(platform_name)
                except Exception:
                    raise ValueError(platform_name)
        except Exception:
            logger.warning(
                "Synthetic process event has invalid platform metadata: %r",
                platform_name,
            )
            return None

        return SessionSource(
            platform=platform,
            chat_id=chat_id,
            chat_type=chat_type,
            thread_id=str(evt.get("thread_id") or "").strip() or None,
            user_id=str(evt.get("user_id") or "").strip() or None,
            user_name=str(evt.get("user_name") or "").strip() or None,
        )

    async def _inject_watch_notification(self, synth_text: str, evt: dict) -> None:
        """Inject a watch-pattern notification as a synthetic message event.

        Routing must come from the queued watch event itself, not from whatever
        foreground message happened to be active when the queue was drained.
        """
        source = self._build_process_event_source(evt)
        if not source:
            logger.warning(
                "Dropping watch notification with no routing metadata for process %s",
                evt.get("session_id", "unknown"),
            )
            return
        platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)
        adapter = None
        for p, a in self.adapters.items():
            if p.value == platform_name:
                adapter = a
                break
        if not adapter:
            return
        try:
            synth_event = MessageEvent(
                text=synth_text,
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
                message_id=str(evt.get("message_id") or "").strip() or None,
            )
            logger.info(
                "Watch pattern notification — injecting for %s chat=%s thread=%s",
                platform_name,
                source.chat_id,
                source.thread_id,
            )
            await adapter.handle_message(synth_event)
        except Exception as e:
            logger.error("Watch notification injection error: %s", e)

    def _enrich_async_delegation_routing(self, evt: dict) -> None:
        """Fill platform/chat_id/thread_id/chat_type on an async-delegation event.

        Async-delegation completion events only carry ``session_key`` (the
        daemon worker has no access to the per-message routing metadata the
        terminal background watcher captures at spawn time). Parse the
        session_key into the routing fields ``_build_process_event_source``
        expects. Best-effort: a CLI-origin event (empty session_key) is left
        as-is and simply won't route on the gateway.
        """
        if evt.get("platform"):
            return  # already enriched
        parsed = _parse_session_key(evt.get("session_key", "") or "")
        if not parsed:
            return
        evt["platform"] = parsed.get("platform", "")
        evt["chat_type"] = parsed.get("chat_type", "")
        evt["chat_id"] = parsed.get("chat_id", "")
        if parsed.get("thread_id"):
            evt["thread_id"] = parsed["thread_id"]

    async def _async_delegation_watcher(self, interval: float = 2.0) -> None:
        """Drain async-delegation completions and inject them as new turns.

        Background subagents (``delegate_task(background=true)``) run on the
        async-delegation daemon executor — they have no per-process watcher
        task, so their completion events would only be seen by the post-turn
        queue drain. This watcher covers the IDLE case: when a background
        subagent finishes while no agent turn is running, its result still
        re-enters the originating session promptly.

        Mirrors the CLI's idle ``process_loop`` drain. Stays silent when the
        queue has nothing for us; ignores non-async event types (those are
        handled by ``_run_process_watcher`` / the post-turn drain).
        """
        await asyncio.sleep(3)  # let platforms finish connecting
        from tools.process_registry import process_registry as _pr
        while self._running:
            try:
                # Peek the queue for async-delegation events. We must NOT
                # consume watch/completion events here (other drains own them),
                # so requeue anything that isn't ours.
                requeue = []
                async_events = []
                while not _pr.completion_queue.empty():
                    try:
                        evt = _pr.completion_queue.get_nowait()
                    except Exception:
                        break
                    if evt.get("type") == "async_delegation":
                        async_events.append(evt)
                    else:
                        requeue.append(evt)
                for evt in requeue:
                    _pr.completion_queue.put(evt)
                for evt in async_events:
                    self._enrich_async_delegation_routing(evt)
                    synth_text = _format_gateway_process_notification(evt)
                    if not synth_text:
                        continue
                    try:
                        await self._inject_watch_notification(synth_text, evt)
                    except Exception as e:
                        logger.error("Async delegation injection error: %s", e)
            except Exception as e:
                logger.debug("Async delegation watcher error: %s", e)
            await asyncio.sleep(interval)

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        user_id = watcher.get("user_id", "")
        user_name = watcher.get("user_name", "")
        message_id = str(watcher.get("message_id") or "").strip() or None
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                # Skip if the agent already consumed the result via wait/log.
                # poll() is read-only and intentionally does NOT mark consumed
                # (#10156) — a status check must not suppress this delivery turn.
                from tools.process_registry import format_process_notification, process_registry as _pr_check
                if agent_notify and not _pr_check.is_completion_consumed(session_id):
                    from tools.ansi_strip import strip_ansi
                    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                    # Truncate at line boundaries so notifications never start
                    # mid-line (fixes #23284). Keep the last ~2000 chars but
                    # snap to the nearest preceding newline, then prepend a
                    # truncation marker when output was cut.
                    _LIMIT = 2000
                    if len(_raw) > _LIMIT:
                        _tail = _raw[-_LIMIT:]
                        _nl = _tail.find("\n")
                        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                    else:
                        _out = _raw
                    synth_text = format_process_notification({
                        "type": "completion",
                        "session_id": session_id,
                        "command": session.command,
                        "exit_code": session.exit_code,
                        "completion_reason": getattr(session, "completion_reason", "exited"),
                        "termination_source": getattr(session, "termination_source", ""),
                        "output": _out,
                    })
                    if not synth_text:
                        break
                    source = self._build_process_event_source({
                        "session_id": session_id,
                        "session_key": session_key,
                        "platform": platform_name,
                        "chat_id": chat_id,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "user_name": user_name,
                    })
                    if not source:
                        logger.warning(
                            "Dropping completion notification with no routing metadata for process %s",
                            session_id,
                        )
                        break

                    adapter = None
                    for p, a in self.adapters.items():
                        if p == source.platform:
                            adapter = a
                            break
                    if adapter and source.chat_id:
                        try:
                            synth_event = MessageEvent(
                                text=synth_text,
                                message_type=MessageType.TEXT,
                                source=source,
                                internal=True,
                                message_id=message_id,
                            )
                            logger.info(
                                "Process %s finished — injecting agent notification for session %s chat=%s thread=%s",
                                session_id,
                                session_key,
                                source.chat_id,
                                source.thread_id,
                            )
                            await adapter.handle_message(synth_event)
                        except Exception as e:
                            logger.error("Agent notify injection error: %s", e)
                    break

                # --- Normal text-only notification ---
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in {"all", "result"}
                    or (notify_mode == "error" and session.exit_code not in {0, None})
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    if new_output:
                        from agent.redact import redact_terminal_output
                        new_output = redact_terminal_output(
                            new_output, getattr(session, "command", "") or ""
                        )
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(
                                chat_id,
                                message_text,
                                metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                            )
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                if new_output:
                    from agent.redact import redact_terminal_output
                    new_output = redact_terminal_output(
                        new_output, getattr(session, "command", "") or ""
                    )
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(
                            chat_id,
                            message_text,
                            metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                        )
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)

    _MAX_INTERRUPT_DEPTH = 3  # Cap recursive interrupt handling (#816)

    # Config keys whose values MUST invalidate the gateway's cached agent
    # when they change.  The agent bakes these into its compressor / context
    # handling at construction time, so a mid-running-gateway config edit
    # would otherwise be silently ignored until the user triggers a
    # different cache eviction (model switch, /reset, etc.).
    #
    # Each entry is a tuple of (section, key) read from the raw config dict.
    # Add more here as new baked-at-construction config settings are added.
    _CACHE_BUSTING_CONFIG_KEYS: tuple = (
        ("model", "context_length"),
        ("model", "max_tokens"),
        ("compression", "enabled"),
        ("compression", "threshold"),
        ("compression", "target_ratio"),
        ("compression", "protect_last_n"),
        ("agent", "disabled_toolsets"),
        ("memory", "provider"),
    )

    _HONCHO_CACHE_BUSTING_KEYS = (
        "honcho.peer_name",
        "honcho.ai_peer",
        "honcho.pin_peer_name",
        "honcho.runtime_peer_prefix",
        "honcho.user_peer_aliases",
    )
    _HONCHO_CACHE_BUSTING_MEMO: dict[tuple[str, int | None], dict[str, Any]] = {}

    @classmethod
    def _empty_honcho_cache_busting_config(cls) -> dict[str, Any]:
        return {key: None for key in cls._HONCHO_CACHE_BUSTING_KEYS}

    @classmethod
    def _extract_honcho_cache_busting_config(cls) -> dict[str, Any]:
        """Extract Honcho identity keys, memoized by honcho.json mtime."""
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path

            path = resolve_config_path()
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                mtime_ns = None
            memo_key = (str(path), mtime_ns)
            cached = cls._HONCHO_CACHE_BUSTING_MEMO.get(memo_key)
            if cached is not None:
                return dict(cached)

            hcfg = HonchoClientConfig.from_global_config(config_path=path)
            aliases = hcfg.user_peer_aliases or {}
            values = {
                "honcho.peer_name": hcfg.peer_name,
                "honcho.ai_peer": hcfg.ai_peer,
                "honcho.pin_peer_name": bool(hcfg.pin_peer_name),
                "honcho.runtime_peer_prefix": hcfg.runtime_peer_prefix or "",
                "honcho.user_peer_aliases": sorted(aliases.items()) if isinstance(aliases, dict) else [],
            }
            cls._HONCHO_CACHE_BUSTING_MEMO = {memo_key: values}
            return dict(values)
        except Exception:
            return cls._empty_honcho_cache_busting_config()

    @classmethod
    def _extract_cache_busting_config(cls, user_config: dict | None) -> dict:
        """Pull values that must bust the cached agent.

        Returns a flat dict keyed by 'section.key'.  Missing config keys and
        non-dict sections yield None values, which still contribute to the
        signature (so 'absent' vs 'present-and-null' differ).

        The live tool registry generation is included too.  MCP reloads and
        dynamic MCP tool-list changes mutate the registry without necessarily
        changing config.yaml.  Cached AIAgent instances freeze their tool
        schemas at construction time, so a registry generation change must
        rebuild the agent before the next turn.
        """
        out: Dict[str, Any] = {}
        cfg = user_config if isinstance(user_config, dict) else {}
        for section, key in cls._CACHE_BUSTING_CONFIG_KEYS:
            section_val = cfg.get(section)
            if isinstance(section_val, dict):
                out[f"{section}.{key}"] = section_val.get(key)
            else:
                out[f"{section}.{key}"] = None
        try:
            from tools.registry import registry

            out["tools.registry_generation"] = getattr(registry, "_generation", None)
        except Exception:
            out["tools.registry_generation"] = None

        # Honcho identity-mapping keys live in honcho.json, not user_config.
        # Only read that file when Honcho is the active memory provider.
        provider = cfg_get(cfg, "memory", "provider")
        if isinstance(provider, str) and provider.lower() == "honcho":
            out.update(cls._extract_honcho_cache_busting_config())
        else:
            out.update(cls._empty_honcho_cache_busting_config())

        return out

    @staticmethod
    def _agent_config_signature(
        model: str,
        runtime: dict,
        enabled_toolsets: list,
        ephemeral_prompt: str,
        cache_keys: dict | None = None,
        user_id: str | None = None,
        user_id_alt: str | None = None,
    ) -> str:
        """Compute a stable string key from agent config values.

        When this signature changes between messages, the cached AIAgent is
        discarded and rebuilt.  When it stays the same, the cached agent is
        reused — preserving the frozen system prompt and tool schemas for
        prompt cache hits.

        ``cache_keys`` is an optional flat dict of additional config values
        that should invalidate the cache when they change.  Callers pass
        the output of ``_extract_cache_busting_config(user_config)`` so
        edits to model.context_length / compression.* in config.yaml are
        picked up on the next gateway message without a manual restart.

        ``user_id`` and ``user_id_alt`` are the runtime user identities
        carried by the current message's gateway source.  They participate
        in the cache key because the Honcho memory provider freezes them
        into ``HonchoSessionManager`` at first-message init (see
        ``plugins/memory/honcho/__init__.py::_do_session_init``).  Without
        them in the signature, a shared-thread session_key (one in which
        ``build_session_key`` intentionally omits the participant ID,
        e.g. ``thread_sessions_per_user=False``) would reuse the cached
        AIAgent across distinct users, causing the second user's messages
        to be attributed to the first user's resolved Honcho peer.  This
        broke #27371's per-user-peer contract in multi-user gateways.
        Per-user agent rebuilds in shared threads trade prompt-cache
        warmth for correct memory attribution.
        """
        import hashlib, json as _j

        # Fingerprint the FULL credential string instead of using a short
        # prefix. OAuth/JWT-style tokens frequently share a common prefix
        # (e.g. "eyJhbGci"), which can cause false cache hits across auth
        # switches if only the first few characters are considered.
        _api_key = str(runtime.get("api_key", "") or "")
        _api_key_fingerprint = hashlib.sha256(_api_key.encode()).hexdigest() if _api_key else ""

        _cache_keys_sorted = sorted((cache_keys or {}).items())

        blob = _j.dumps(
            [
                model,
                _api_key_fingerprint,
                runtime.get("base_url", ""),
                runtime.get("provider", ""),
                runtime.get("api_mode", ""),
                sorted(enabled_toolsets) if enabled_toolsets else [],
                # reasoning_config excluded — it's set per-message on the
                # cached agent and doesn't affect system prompt or tools.
                ephemeral_prompt or "",
                _cache_keys_sorted,
                str(user_id or ""),
                str(user_id_alt or ""),
            ],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _apply_session_model_override(
        self, session_key: str, model: str, runtime_kwargs: dict
    ) -> tuple:
        """Apply /model session overrides if present, returning (model, runtime_kwargs).

        The gateway /model command stores per-session overrides in
        ``_session_model_overrides``.  These must take precedence over
        config.yaml defaults so the switched model is actually used for
        subsequent messages.  Fields with ``None`` values are skipped so
        partial overrides don't clobber valid config defaults.
        """
        override = self._session_model_overrides.get(session_key)
        if not override:
            return model, runtime_kwargs
        model = override.get("model", model)
        for key in ("provider", "api_key", "base_url", "api_mode"):
            val = override.get(key)
            if val is not None:
                runtime_kwargs[key] = val
        return model, runtime_kwargs

    def _is_intentional_model_switch(self, session_key: str, agent_model: str) -> bool:
        """Return True if *agent_model* matches an active /model session override."""
        override = self._session_model_overrides.get(session_key)
        return override is not None and override.get("model") == agent_model

    def _release_running_agent_state(
        self,
        session_key: str,
        *,
        run_generation: Optional[int] = None,
    ) -> bool:
        """Pop ALL per-running-agent state entries for ``session_key``.

        Replaces ad-hoc ``del self._running_agents[key]`` calls scattered
        across the gateway.  Those sites had drifted: some popped only
        ``_running_agents``; some also ``_running_agents_ts``; only one
        path also cleared ``_busy_ack_ts``.  Each missed entry was a
        small, persistent leak — a (str_key → float) tuple per session
        per gateway lifetime.

        Use this at every site that ends a running turn, regardless of
        cause (normal completion, /stop, /reset, /resume, sentinel
        cleanup, stale-eviction).  Per-session state that PERSISTS
        across turns (``_session_model_overrides``, ``_voice_mode``,
        ``_pending_approvals``, ``_update_prompt_pending``) is NOT
        touched here — those have their own lifecycles.

        When ``run_generation`` is provided, only clear the slot if that
        generation is still current for the session.  This prevents an
        older async run whose generation was bumped by /stop or /new from
        clobbering a newer run's state during its own unwind.  Returns
        True when the slot was cleared, False when an ownership guard
        blocked it.
        """
        if not session_key:
            return False
        if run_generation is not None and not self._is_session_run_current(
            session_key, run_generation
        ):
            return False
        lease = getattr(self, "_active_session_leases", {}).pop(session_key, None)
        if lease is not None:
            try:
                lease.release()
            except Exception:
                logger.debug("Failed to release active session slot", exc_info=True)
        self._running_agents.pop(session_key, None)
        self._running_agents_ts.pop(session_key, None)
        if hasattr(self, "_busy_ack_ts"):
            self._busy_ack_ts.pop(session_key, None)
        # Turn boundary: a running-agent slot was just released.  Persist the
        # new (lower) in-flight count so the dashboard readout stays current
        # between lifecycle transitions.  Preserves gateway_state (see
        # _persist_active_agents).
        self._persist_active_agents()
        return True

    def _clear_session_boundary_security_state(self, session_key: str) -> None:
        """Clear per-session control state that must not survive a boundary switch."""
        if not session_key:
            return

        pending_skills_reload_notes = getattr(
            self, "_pending_skills_reload_notes", None
        )
        if isinstance(pending_skills_reload_notes, dict):
            pending_skills_reload_notes.pop(session_key, None)

        pending_approvals = getattr(self, "_pending_approvals", None)
        if isinstance(pending_approvals, dict):
            pending_approvals.pop(session_key, None)

        update_prompt_pending = getattr(self, "_update_prompt_pending", None)
        if isinstance(update_prompt_pending, dict):
            update_prompt_pending.pop(session_key, None)

        try:
            from tools import slash_confirm as _slash_confirm_mod
        except Exception:
            _slash_confirm_mod = None
        if _slash_confirm_mod is not None:
            try:
                _slash_confirm_mod.clear(session_key)
            except Exception as e:
                logger.debug(
                    "Failed to clear slash-confirm state for session boundary %s: %s",
                    session_key,
                    e,
                )

        try:
            from tools.approval import clear_session as _clear_approval_session
        except Exception:
            return

        try:
            _clear_approval_session(session_key)
        except Exception as e:
            logger.debug(
                "Failed to clear approval state for session boundary %s: %s",
                session_key,
                e,
            )

    def _begin_session_run_generation(self, session_key: str) -> int:
        """Claim a fresh run generation token for ``session_key``.

        Every top-level gateway turn gets a monotonically increasing token.
        If a later command like /stop or /new invalidates that token while the
        old worker is still unwinding, the late result can be recognized and
        dropped instead of bleeding into the fresh session.
        """
        if not session_key:
            return 0
        generations = self.__dict__.get("_session_run_generation")
        if generations is None:
            generations = {}
            self._session_run_generation = generations
        next_generation = int(generations.get(session_key, 0)) + 1
        generations[session_key] = next_generation
        return next_generation

    def _invalidate_session_run_generation(self, session_key: str, *, reason: str = "") -> int:
        """Invalidate any in-flight run token for ``session_key``."""
        generation = self._begin_session_run_generation(session_key)
        if reason:
            logger.info(
                "Invalidated run generation for %s → %d (%s)",
                session_key,
                generation,
                reason,
            )
        return generation

    def _is_session_run_current(self, session_key: str, generation: int) -> bool:
        """Return True when ``generation`` is still current for ``session_key``."""
        if not session_key:
            return True
        generations = self.__dict__.get("_session_run_generation") or {}
        return int(generations.get(session_key, 0)) == int(generation)

    def _bind_adapter_run_generation(
        self,
        adapter: Any,
        session_key: str,
        generation: int | None,
    ) -> None:
        """Bind a gateway run generation to the adapter's active-session event."""
        if not adapter or not session_key or generation is None:
            return
        try:
            interrupt_event = getattr(adapter, "_active_sessions", {}).get(session_key)
            if interrupt_event is not None:
                setattr(interrupt_event, "_hermes_run_generation", int(generation))
        except Exception:
            pass

    async def _interrupt_and_clear_session(
        self,
        session_key: str,
        source: SessionSource,
        *,
        interrupt_reason: str,
        invalidation_reason: str,
        release_running_state: bool = True,
    ) -> None:
        """Interrupt the current run and clear queued session state consistently."""
        if not session_key:
            return
        running_agent = self._running_agents.get(session_key)
        if running_agent and running_agent is not _AGENT_PENDING_SENTINEL:
            running_agent.interrupt(interrupt_reason)
        self._invalidate_session_run_generation(session_key, reason=invalidation_reason)
        adapter = self.adapters.get(source.platform)
        if adapter and hasattr(adapter, "interrupt_session_activity"):
            await adapter.interrupt_session_activity(session_key, source.chat_id)
        if adapter and hasattr(adapter, "get_pending_message"):
            adapter.get_pending_message(session_key)  # consume and discard
        self._pending_messages.pop(session_key, None)
        if release_running_state:
            self._release_running_agent_state(session_key)

    async def _refresh_agent_cache_message_count(
        self, session_key: str, session_id: Optional[str]
    ) -> None:
        """Re-baseline a cached agent's stored message_count after THIS turn.

        The cross-process coherence guard (#45966) compares the session's
        on-disk ``message_count`` against the count snapshotted next to the
        cached agent, and rebuilds the agent on a mismatch.  But the snapshot
        is taken at agent-BUILD time — before this turn writes its own user +
        assistant (+ tool) rows — and the cache entry is never rewritten on a
        reuse.  So without this re-baseline, THIS process's own turn would
        grow ``message_count`` and the very next turn would see a mismatch
        and rebuild the agent — every turn, for every conversation — silently
        destroying the per-conversation prompt caching the cache exists to
        protect.

        Call this once a turn has completed and the agent has flushed its
        rows to the SessionDB.  It snapshots the now-current count (which
        includes this process's own writes) so the guard only fires when a
        DIFFERENT process changes the transcript out from under us.  The
        ``_sig`` is left untouched; only the count element is refreshed, and
        only when the same agent is still cached (no rebuild/eviction raced
        in between).  Fail-safe: any DB error leaves the snapshot as-is, which
        at worst costs one unnecessary rebuild on the next turn.

        When the cache entry records a ``session_id`` (4-tuple form, #54947)
        that differs from the current ``session_id`` — meaning the cache
        was built for a DIFFERENT conversation under the same ``session_key``
        — the snapshot is intentionally left untouched.  Overwriting it with
        the current session's count would corrupt the original conversation's
        baseline and cause the next switch back to fire the cross-process
        guard spuriously.  Fail-safe: the legacy 3-tuple shape (no
        ``session_id``) is still re-baselined as before.
        """
        if self._session_db is None or not session_id:
            return
        _cache_lock = getattr(self, "_agent_cache_lock", None)
        _cache = getattr(self, "_agent_cache", None)
        if not _cache_lock or _cache is None:
            return
        try:
            _sess_row = await self._session_db.get_session(session_id)
            _live = _sess_row.get("message_count", 0) if _sess_row else None
        except Exception:
            return
        if _live is None:
            return
        with _cache_lock:
            cached = _cache.get(session_key)
            # Only re-baseline a live 3-tuple entry; skip pending sentinels,
            # legacy 2-tuples (they intentionally opt out of the guard), and
            # the case where the entry was evicted/rebuilt mid-turn.
            if (
                isinstance(cached, tuple)
                and len(cached) > 2
                and cached[0] is not _AGENT_PENDING_SENTINEL
            ):
                # If the snapshot was taken for a different session_id
                # (same session_key, different conversation), leave the
                # snapshot alone — the current session_id's count belongs
                # to a different DB row (#54947).
                _snapshot_sid = cached[3] if len(cached) > 3 else None
                if _snapshot_sid is not None and _snapshot_sid != session_id:
                    return
                if cached[2] != _live:
                    if _snapshot_sid is None:
                        # Legacy 3-tuple: preserve the original 3-element
                        # shape so existing entries stay compatible with
                        # callers that index ``cached[2]`` directly.
                        _cache[session_key] = (cached[0], cached[1], _live)
                    else:
                        _cache[session_key] = (
                            cached[0], cached[1], _live, _snapshot_sid,
                        )

    def _evict_cached_agent(self, session_key: str) -> None:
        """Remove a cached agent for a session (called on /new, /model, etc).

        Pops the entry AND soft-releases the evicted agent's LLM client
        pool so the httpx connection (sockets + held buffers) is freed
        promptly rather than waiting on CPython GC — AIAgent holds
        reference cycles (callbacks, tool state) that delay refcount
        collection, so a manual release is required to keep gateway RSS
        flat across many /new, /model, undo and reset operations (#29298,
        same leak class as #25315).

        The release is soft (``release_clients()``): it frees the client
        pool and per-turn child subagents but PRESERVES the session's
        terminal sandbox, browser daemon, and tracked bg processes (keyed
        on task_id), because the session may resume with a freshly-built
        agent.  Call sites that want a hard teardown (true conversation
        boundaries like /new) already call ``_cleanup_agent_resources``
        before evicting; ``release_clients`` is idempotent and safe to
        run again after that (the client is already None).

        Cleanup runs on a daemon thread so we never block holding
        ``_agent_cache_lock`` on slow socket teardown — mirrors the
        cap-enforcer and idle-sweeper paths.
        """
        _lock = getattr(self, "_agent_cache_lock", None)
        evicted = None
        if _lock:
            with _lock:
                evicted = self._agent_cache.pop(session_key, None)
        else:
            _cache = getattr(self, "_agent_cache", None)
            if _cache is not None:
                evicted = _cache.pop(session_key, None)

        agent = evicted[0] if isinstance(evicted, tuple) and evicted else evicted
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return

        # Don't tear down an agent that's actively mid-turn — its client,
        # sandbox and child subagents are in use by the running request.
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }
        if id(agent) in running_ids:
            return

        try:
            threading.Thread(
                target=self._release_evicted_agent_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-evict-{str(session_key)[:24]}",
            ).start()
        except Exception:
            # If we can't spawn a thread (interpreter shutdown), release
            # inline as a best-effort fallback.
            try:
                self._release_evicted_agent_soft(agent)
            except Exception:
                pass

    @staticmethod
    def _init_cached_agent_for_turn(agent: Any, interrupt_depth: int) -> None:
        """Reset per-turn state on a cached agent before a new turn starts.

        Both _last_activity_ts and _last_activity_desc are only reset for
        fresh external turns (depth 0); they are semantically paired —
        desc describes the activity *at* ts, so updating one without the
        other would make get_activity_summary() misleading.
        For interrupt-recursive turns both are preserved so the inactivity
        watchdog can accumulate stuck-turn idle time and fire the 30-min
        timeout (#15654).  The depth-0 reset is still needed: a session
        idle for 29 min would otherwise trip the watchdog before the new
        turn makes its first API call (#9051).
        """
        if interrupt_depth == 0:
            agent._last_activity_ts = time.time()
            agent._last_activity_desc = "starting new turn (cached)"
            # Reset the SessionDB flush cursor so the new turn's messages are
            # fully persisted — a stale value from the previous turn would
            # cause `_flush_messages_to_session_db` to skip new rows (#44327).
            if hasattr(agent, "_last_flushed_db_idx"):
                agent._last_flushed_db_idx = 0
        agent._api_call_count = 0

    def _release_evicted_agent_soft(self, agent: Any) -> None:
        """Soft cleanup for cache-evicted agents — preserves session tool state.

        Called from _enforce_agent_cache_cap and _sweep_idle_cached_agents.
        Distinct from _cleanup_agent_resources (full teardown) because a
        cache-evicted session may resume at any time — its terminal
        sandbox, browser daemon, and tracked bg processes must outlive
        the Python AIAgent instance so the next agent built for the
        same task_id inherits them.
        """
        if agent is None:
            return
        try:
            if hasattr(agent, "release_clients"):
                agent.release_clients()
            else:
                # Older agent instance (shouldn't happen in practice) —
                # fall back to the legacy full-close path.
                self._cleanup_agent_resources(agent)
        except Exception:
            pass
        # Free conversation history memory — can be tens of MB with tool
        # outputs (file reads, terminal output, search results) on heavy
        # 100+-tool-call sessions. release_clients() deliberately preserves
        # session tool state for resume, but the message list is rebuilt from
        # persisted session JSON on the next turn, so dropping it here is safe.
        if hasattr(agent, "_session_messages"):
            agent._session_messages = []

    def _enforce_agent_cache_cap(self) -> None:
        """Evict oldest cached agents when cache exceeds _AGENT_CACHE_MAX_SIZE.

        Must be called with _agent_cache_lock held.  Resource cleanup
        (memory provider shutdown, tool resource close) is scheduled
        on a daemon thread so the caller doesn't block on slow teardown
        while holding the cache lock.

        Agents currently in _running_agents are SKIPPED — their clients,
        terminal sandboxes, background processes, and child subagents
        are all in active use by the running turn.  Evicting them would
        tear down those resources mid-turn and crash the request.  If
        every candidate in the LRU order is active, we simply leave the
        cache over the cap; it will be re-checked on the next insert.
        """
        _cache = getattr(self, "_agent_cache", None)
        if _cache is None:
            return
        # OrderedDict.popitem(last=False) pops oldest; plain dict lacks the
        # arg so skip enforcement if a test fixture swapped the cache type.
        if not hasattr(_cache, "move_to_end"):
            return

        # Snapshot of agent instances that are actively mid-turn.  Use id()
        # so the lookup is O(1) and doesn't depend on AIAgent.__eq__ (which
        # MagicMock overrides in tests).
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }

        # Walk LRU → MRU and evict excess-LRU entries that aren't mid-turn.
        # We only consider entries in the first (size - cap) LRU positions
        # as eviction candidates.  If one of those slots is held by an
        # active agent, we SKIP it without compensating by evicting a
        # newer entry — that would penalise a freshly-inserted session
        # (which has no cache history to retain) while protecting an
        # already-cached long-running one.  The cache may therefore stay
        # temporarily over cap; it will re-check on the next insert,
        # after active turns have finished.
        excess = max(0, len(_cache) - _AGENT_CACHE_MAX_SIZE)
        evict_plan: List[tuple] = []  # [(key, agent), ...]
        if excess > 0:
            ordered_keys = list(_cache.keys())
            for key in ordered_keys[:excess]:
                entry = _cache.get(key)
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is not None and id(agent) in running_ids:
                    continue  # active mid-turn; don't evict, don't substitute
                evict_plan.append((key, agent))

        for key, _ in evict_plan:
            _cache.pop(key, None)

        remaining_over_cap = len(_cache) - _AGENT_CACHE_MAX_SIZE
        if remaining_over_cap > 0:
            logger.warning(
                "Agent cache over cap (%d > %d); %d excess slot(s) held by "
                "mid-turn agents — will re-check on next insert.",
                len(_cache), _AGENT_CACHE_MAX_SIZE, remaining_over_cap,
            )

        for key, agent in evict_plan:
            logger.info(
                "Agent cache at cap; evicting LRU session=%s (cache_size=%d)",
                key, len(_cache),
            )
            if agent is not None:
                threading.Thread(
                    target=self._release_evicted_agent_soft,
                    args=(agent,),
                    daemon=True,
                    name=f"agent-cache-evict-{key[:24]}",
                ).start()

    def _sweep_idle_cached_agents(self) -> int:
        """Evict cached agents whose AIAgent has been idle > _AGENT_CACHE_IDLE_TTL_SECS.

        Safe to call from the session expiry watcher without holding the
        cache lock — acquires it internally.  Returns the number of entries
        evicted.  Resource cleanup is scheduled on daemon threads.

        Agents currently in _running_agents are SKIPPED for the same reason
        as _enforce_agent_cache_cap: tearing down an active turn's clients
        mid-flight would crash the request.
        """
        _cache = getattr(self, "_agent_cache", None)
        _lock = getattr(self, "_agent_cache_lock", None)
        if _cache is None or _lock is None:
            return 0
        now = time.time()
        to_evict: List[tuple] = []
        running_ids = {
            id(a)
            for a in getattr(self, "_running_agents", {}).values()
            if a is not None and a is not _AGENT_PENDING_SENTINEL
        }
        with _lock:
            for key, entry in list(_cache.items()):
                agent = entry[0] if isinstance(entry, tuple) and entry else None
                if agent is None:
                    continue
                if id(agent) in running_ids:
                    continue  # mid-turn — don't tear it down
                last_activity = getattr(agent, "_last_activity_ts", None)
                if last_activity is None:
                    continue
                if (now - last_activity) > _AGENT_CACHE_IDLE_TTL_SECS:
                    to_evict.append((key, agent))
            for key, _ in to_evict:
                _cache.pop(key, None)
        for key, agent in to_evict:
            logger.info(
                "Agent cache idle-TTL evict: session=%s (idle=%.0fs)",
                key, now - getattr(agent, "_last_activity_ts", now),
            )
            threading.Thread(
                target=self._release_evicted_agent_soft,
                args=(agent,),
                daemon=True,
                name=f"agent-cache-idle-{key[:24]}",
            ).start()
        return len(to_evict)

    # ------------------------------------------------------------------
    # Proxy mode: forward messages to a remote Hermes API server
    # ------------------------------------------------------------------

    def _get_proxy_url(self) -> Optional[str]:
        """Return the proxy URL if proxy mode is configured, else None.

        Checks GATEWAY_PROXY_URL env var first (convenient for Docker),
        then ``gateway.proxy_url`` in config.yaml.
        """
        url = os.getenv("GATEWAY_PROXY_URL", "").strip()
        if url:
            return url.rstrip("/")
        cfg = _load_gateway_config()
        url = (cfg.get("gateway") or {}).get("proxy_url", "").strip()
        if url:
            return url.rstrip("/")
        return None

    async def _run_agent_via_proxy(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: "SessionSource",
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        event_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Forward the message to a remote Hermes API server instead of
        running a local AIAgent.

        When ``GATEWAY_PROXY_URL`` (or ``gateway.proxy_url`` in config.yaml)
        is set, the gateway becomes a thin relay: it handles platform I/O
        (encryption, threading, media) and delegates all agent work to the
        remote server via ``POST /v1/chat/completions`` with SSE streaming.

        This lets a Docker container handle Matrix E2EE while the actual
        agent runs on the host with full access to local files, memory,
        skills, and a unified session store.
        """
        try:
            from aiohttp import ClientSession as _AioClientSession, ClientTimeout
        except ImportError:
            return {
                "final_response": "⚠️ Proxy mode requires aiohttp. Install with: pip install aiohttp",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        proxy_url = self._get_proxy_url()
        if not proxy_url:
            return {
                "final_response": "⚠️ Proxy URL not configured (GATEWAY_PROXY_URL or gateway.proxy_url)",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        proxy_key = os.getenv("GATEWAY_PROXY_KEY", "").strip()

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)

        # Build messages in OpenAI chat format --------------------------
        #
        # The remote api_server can maintain session continuity via
        # X-Hermes-Session-Id, so it loads its own history.  We only
        # need to send the current user message.  If the remote has
        # no history for this session yet, include what we have locally
        # so the first exchange has context.
        #
        # We always include the current message.  For history, send a
        # compact version (text-only user/assistant turns) — the remote
        # handles tool replay and system prompts.
        api_messages: List[Dict[str, str]] = []

        if context_prompt:
            api_messages.append({"role": "system", "content": context_prompt})

        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in {"user", "assistant"} and content:
                api_messages.append({"role": role, "content": content})

        api_messages.append({"role": "user", "content": message})

        # HTTP headers ---------------------------------------------------
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if proxy_key:
            headers["Authorization"] = f"Bearer {proxy_key}"
        if session_id:
            headers["X-Hermes-Session-Id"] = session_id

        body = {
            "model": "hermes-agent",
            "messages": api_messages,
            "stream": True,
        }

        # Set up platform streaming if available -------------------------
        _stream_consumer = None
        _scfg = getattr(getattr(self, "config", None), "streaming", None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        platform_key = _platform_config_key(source.platform)
        user_config = _load_gateway_config()
        from gateway.display_config import resolve_display_setting
        _plat_streaming = resolve_display_setting(
            user_config, platform_key, "streaming"
        )
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )

        _thread_metadata: Optional[Dict[str, Any]] = self._thread_metadata_for_source(source, event_message_id)

        if _streaming_enabled:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                _adapter = self.adapters.get(source.platform)
                if _adapter:
                    _pause_typing_before_finalize = None
                    if source.platform == Platform.TELEGRAM and hasattr(_adapter, "pause_typing_for_chat"):
                        def _pause_typing_before_finalize(
                            _adapter=_adapter,
                            _chat_id=source.chat_id,
                        ) -> None:
                            _adapter.pause_typing_for_chat(_chat_id)
                    _adapter_supports_edit = getattr(_adapter, "SUPPORTS_MESSAGE_EDITING", True)
                    _effective_cursor = _scfg.cursor if _adapter_supports_edit else ""
                    _buffer_only = False
                    if source.platform == Platform.MATRIX:
                        _effective_cursor = ""
                        _buffer_only = True
                    # Fresh-final applies to Telegram only — other
                    # platforms either edit in place cheaply (Discord,
                    # Slack) or don't have the timestamp-on-edit
                    # problem.  (Ported from openclaw/openclaw#72038.)
                    _fresh_final_secs = (
                        float(getattr(_scfg, "fresh_final_after_seconds", 0.0) or 0.0)
                        if source.platform == Platform.TELEGRAM
                        else 0.0
                    )
                    _consumer_cfg = StreamConsumerConfig(
                        edit_interval=_scfg.edit_interval,
                        buffer_threshold=_scfg.buffer_threshold,
                        cursor=_effective_cursor,
                        buffer_only=_buffer_only,
                        fresh_final_after_seconds=_fresh_final_secs,
                        transport=_scfg.transport or "edit",
                        chat_type=getattr(source, "chat_type", "") or "",
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=source.chat_id,
                        config=_consumer_cfg,
                        metadata=_thread_metadata,
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=event_message_id,
                        run_still_current=_run_still_current,
                    )
            except Exception as _sc_err:
                logger.debug("Proxy: could not set up stream consumer: %s", _sc_err)

        # Run the stream consumer task in the background
        stream_task = None
        if _stream_consumer:
            stream_task = asyncio.create_task(_stream_consumer.run())

        # Send typing indicator
        _adapter = self.adapters.get(source.platform)
        if _adapter:
            try:
                await _adapter.send_typing(source.chat_id, metadata=_thread_metadata)
            except Exception:
                pass

        # Make the HTTP request with SSE streaming -----------------------
        full_response = ""
        _start = time.time()

        try:
            _timeout = ClientTimeout(total=0, sock_read=1800)
            async with _AioClientSession(timeout=_timeout) as session:
                async with session.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.warning(
                            "Proxy error (%d) from %s: %s",
                            resp.status, proxy_url, error_text[:500],
                        )
                        return {
                            "final_response": f"⚠️ Proxy error ({resp.status}): {error_text[:300]}",
                            "messages": [],
                            "api_calls": 0,
                            "tools": [],
                        }

                    # Parse SSE stream
                    buffer = ""
                    async for chunk in resp.content.iter_any():
                        if not _run_still_current():
                            logger.info(
                                "Discarding stale proxy stream for %s — generation %d is no longer current",
                                session_key or "?",
                                run_generation or 0,
                            )
                            return {
                                "final_response": "",
                                "messages": [],
                                "api_calls": 0,
                                "tools": [],
                                "history_offset": len(history),
                                "session_id": session_id,
                                "response_previewed": False,
                            }
                        text = chunk.decode("utf-8", errors="replace")
                        buffer += text

                        # Process complete SSE lines
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(data)
                                    choices = obj.get("choices", [])
                                    if choices:
                                        delta = choices[0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            full_response += content
                                            if _stream_consumer:
                                                _stream_consumer.on_delta(content)
                                except json.JSONDecodeError:
                                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Proxy connection error to %s: %s", proxy_url, e)
            if not full_response:
                return {
                    "final_response": f"⚠️ Proxy connection error: {e}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }
            # Partial response — return what we got
        finally:
            # Finalize stream consumer
            if _stream_consumer:
                _stream_consumer.finish()
            if stream_task:
                try:
                    await asyncio.wait_for(stream_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stream_task.cancel()

        _elapsed = time.time() - _start
        if not _run_still_current():
            logger.info(
                "Discarding stale proxy result for %s — generation %d is no longer current",
                session_key or "?",
                run_generation or 0,
            )
            return {
                "final_response": "",
                "messages": [],
                "api_calls": 0,
                "tools": [],
                "history_offset": len(history),
                "session_id": session_id,
                "response_previewed": False,
            }
        logger.info(
            "proxy response: url=%s session=%s time=%.1fs response=%d chars",
            proxy_url, (session_id or "")[:20], _elapsed, len(full_response),
        )

        return {
            "final_response": full_response or "(No response from remote agent)",
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response},
            ],
            "api_calls": 1,
            "tools": [],
            "history_offset": len(history),
            "session_id": session_id,
            "response_previewed": _stream_consumer is not None and bool(full_response),
        }

    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[str] = None,
        persist_user_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Profile-scoping wrapper around the agent run.

        When multiplexing is active, resolve the inbound source's profile and
        run the whole turn inside ``_profile_runtime_scope`` so config/skills/
        memory resolve to that profile's home AND credentials resolve from that
        profile's secret scope (never the process-global ``os.environ``). When
        multiplexing is off this is a transparent pass-through — zero behavior
        change for single-profile gateways.
        """
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
            )

        profile_home = self._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            return await self._run_agent_inner(
                message, context_prompt, history, source, session_id,
                session_key=session_key, run_generation=run_generation,
                _interrupt_depth=_interrupt_depth, event_message_id=event_message_id,
                channel_prompt=channel_prompt, moa_config=moa_config,
                persist_user_message=persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
            )

    def _resolve_profile_home_for_source(self, source: SessionSource) -> "Path":
        """Resolve which profile's HERMES_HOME should serve this inbound source.

        Prefers the profile the source was routed to (``source.profile`` — set
        by the /p/<profile>/ URL prefix or a per-credential adapter), falling
        back to the active profile (the multiplexer's own home).
        """
        from hermes_cli.profiles import get_active_profile_name, get_profile_dir
        try:
            name = (source.profile or "").strip() or get_active_profile_name() or "default"
            return get_profile_dir(name)
        except Exception:
            from hermes_constants import get_hermes_home
            return get_hermes_home()

    async def _run_agent_inner(
        self,
        message: str,
        context_prompt: str,
        history: List[Dict[str, Any]],
        source: SessionSource,
        session_id: str,
        session_key: str = None,
        run_generation: Optional[int] = None,
        _interrupt_depth: int = 0,
        event_message_id: Optional[str] = None,
        channel_prompt: Optional[str] = None,
        moa_config: Optional[dict] = None,
        persist_user_message: Optional[str] = None,
        persist_user_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run the agent with the given message and context.
        
        Returns the full result dict from run_conversation, including:
          - "final_response": str (the text to send back)
          - "messages": list (full conversation including tool calls)
          - "api_calls": int
          - "completed": bool
        
        This is run in a thread pool to not block the event loop.
        Supports interruption via new messages.
        """
        # ---- Proxy mode: delegate to remote API server ----
        if self._get_proxy_url():
            return await self._run_agent_via_proxy(
                message=message,
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_id,
                session_key=session_key,
                run_generation=run_generation,
                event_message_id=event_message_id,
            )

        from run_agent import AIAgent
        import queue

        def _run_still_current() -> bool:
            if run_generation is None or not session_key:
                return True
            return self._is_session_run_current(session_key, run_generation)
        
        user_config = _load_gateway_config()
        platform_key = _platform_config_key(source.platform)

        from hermes_cli.tools_config import _get_platform_tools
        enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
        agent_cfg_local = user_config.get("agent") or {}
        disabled_toolsets = agent_cfg_local.get("disabled_toolsets") or None

        display_config = user_config.get("display", {})
        if not isinstance(display_config, dict):
            display_config = {}

        # Per-platform display settings — resolve via display_config module
        # which checks display.platforms.<platform>.<key> first, then
        # display.<key> global, then built-in platform defaults.
        from gateway.display_config import resolve_display_setting

        # Apply tool preview length config (0 = no limit)
        try:
            from agent.display import set_tool_preview_max_len
            _tpl = resolve_display_setting(user_config, platform_key, "tool_preview_length", 0)
            set_tool_preview_max_len(int(_tpl) if _tpl else 0)
        except Exception:
            pass

        # Apply friendly tool labels config (default on) — per-platform aware
        try:
            from agent.display import set_friendly_tool_labels
            _ftl = resolve_display_setting(user_config, platform_key, "friendly_tool_labels", True)
            set_friendly_tool_labels(bool(_ftl))
        except Exception:
            pass

        # Tool progress mode — resolved per-platform with env var fallback
        _resolved_tp = resolve_display_setting(user_config, platform_key, "tool_progress")
        _env_tp = os.getenv("HERMES_TOOL_PROGRESS_MODE")
        _display_cfg = display_config if isinstance(display_config, dict) else {}
        _platforms_cfg = _display_cfg.get("platforms") or {}
        _platform_cfg = _platforms_cfg.get(platform_key) or {}
        _legacy_tp_overrides = _display_cfg.get("tool_progress_overrides") or {}
        _tool_progress_configured = (
            "tool_progress" in _display_cfg
            or (
                isinstance(_platform_cfg, dict)
                and "tool_progress" in _platform_cfg
            )
            or (
                isinstance(_legacy_tp_overrides, dict)
                and platform_key in _legacy_tp_overrides
            )
        )
        progress_mode = (
            _env_tp
            if _env_tp and not _tool_progress_configured
            else (_resolved_tp or _env_tp or "all")
        )
        # Tool progress grouping: "accumulate" (edit one bubble) or "separate" (one msg per tool)
        progress_grouping = resolve_display_setting(user_config, platform_key, "tool_progress_grouping") or "accumulate"
        # Disable tool progress for webhooks - they don't support message editing,
        # so each progress line would be sent as a separate message.
        from gateway.config import Platform
        tool_progress_enabled = progress_mode != "off" and source.platform != Platform.WEBHOOK
        # Natural assistant status messages are intentionally independent from
        # tool progress and token streaming. Users can keep tool_progress quiet
        # in chat platforms while opting into concise mid-turn updates.
        interim_assistant_messages_enabled = (
            source.platform != Platform.WEBHOOK
            and _resolve_gateway_display_bool(
                user_config,
                platform_key,
                "interim_assistant_messages",
                default=True,
                platform=source.platform,
                require_platform_override_for={Platform.MATTERMOST},
            )
        )
        # thinking_progress is independent — if enabled, we need the progress
        # queue even when tool_progress is off (thinking relay uses same infra).
        # Mattermost requires a per-platform opt-in: global scratch-text display
        # is too easy to leak into busy public threads.
        _thinking_enabled = _resolve_gateway_display_bool(
            user_config,
            platform_key,
            "thinking_progress",
            default=False,
            platform=source.platform,
            require_platform_override_for={Platform.MATTERMOST},
        )
        needs_progress_queue = tool_progress_enabled or _thinking_enabled


        # Queue for progress messages (thread-safe)
        progress_queue = queue.Queue() if needs_progress_queue else None
        last_tool = [None]  # Mutable container for tracking in closure
        last_progress_msg = [None]  # Track last message for dedup
        repeat_count = [0]  # How many times the same message repeated
        # True when the previously enqueued progress line was a terminal
        # fenced code block — consecutive terminal calls then drop the
        # repeated "💻 terminal" header and render back-to-back blocks.
        last_was_terminal_block = [False]

        # ── Discord voice "verbal ack before tool calls" ────────────────
        # When the bot is in a voice channel with the continuous mixer
        # installed (discord.voice_fx.enabled), speak a short phrase ("let me
        # look into that") over the ambient idle bed on the FIRST tool call of
        # the turn.  Fires from tool_start_callback (independent of the
        # tool-progress text gate), at most once per turn.  No-op on every
        # other platform / when not in a voice channel.
        _voice_ack_fired = [False]
        _voice_ack_guild: List[Optional[int]] = [None]
        if source.platform == Platform.DISCORD:
            _va = self.adapters.get(Platform.DISCORD)
            # source.chat_id is the linked text channel; resolve the guild whose
            # voice connection is bound to it (mirrors DiscordAdapter.play_tts).
            _vtc = getattr(_va, "_voice_text_channels", None)
            if isinstance(_vtc, dict) and hasattr(_va, "voice_mixer_active"):
                for _gid, _tc in _vtc.items():
                    if str(_tc) == str(source.chat_id) and _va.voice_mixer_active(_gid):
                        _voice_ack_guild[0] = _gid
                        break
        _voice_ack_loop = asyncio.get_running_loop()

        def voice_ack_callback(call_id, tool_name, args):
            """tool_start_callback: speak a one-time ack in the voice channel."""
            if _voice_ack_fired[0] or _voice_ack_guild[0] is None:
                return
            if not _run_still_current():
                return
            _voice_ack_fired[0] = True
            _adapter = self.adapters.get(Platform.DISCORD)
            if _adapter is None or not hasattr(_adapter, "play_ack_in_voice"):
                return
            try:
                safe_schedule_threadsafe(
                    _adapter.play_ack_in_voice(_voice_ack_guild[0]),
                    _voice_ack_loop,
                    logger=logger,
                    log_message="voice ack scheduling error",
                )
            except Exception as _ack_err:
                logger.debug("voice ack schedule failed: %s", _ack_err)

        # Auto-cleanup of temporary progress bubbles (Telegram + any adapter
        # that implements ``delete_message``). When enabled via
        # ``display.platforms.<platform>.cleanup_progress: true``, message IDs
        # from the tool-progress / "⏳ Working — N min" / status-callback bubbles
        # are collected here and deleted after the final response lands.
        # Failed runs skip cleanup so the bubbles remain as breadcrumbs.
        _cleanup_progress = bool(
            resolve_display_setting(user_config, platform_key, "cleanup_progress")
        )
        _cleanup_adapter = self.adapters.get(source.platform) if _cleanup_progress else None
        if _cleanup_adapter is not None and (
            type(_cleanup_adapter).delete_message is BasePlatformAdapter.delete_message
        ):
            # Adapter doesn't support deletion — silently disable.
            _cleanup_progress = False
            _cleanup_adapter = None
        _cleanup_msg_ids: List[str] = []
        # First-touch onboarding latch: fires at most once per run, even if
        # several tools exceed the threshold.
        long_tool_hint_fired = [False]
        _LONG_TOOL_THRESHOLD_S = 30.0

        def progress_callback(event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
            """Callback invoked by agent on tool lifecycle events."""
            if not progress_queue or not _run_still_current():
                return

            # First-touch onboarding: the first time a tool takes longer than
            # _LONG_TOOL_THRESHOLD_S during a run that's streaming every tool
            # (progress_mode == "all"), append a one-time hint suggesting
            # /verbose.  We only fire when (a) the user hasn't seen the hint
            # before and (b) /verbose is actually usable on this platform
            # (gateway gate must be open).  The CLI has its own trigger.
            if event_type == "tool.completed" and not long_tool_hint_fired[0]:
                try:
                    duration = kwargs.get("duration") or 0
                    if duration >= _LONG_TOOL_THRESHOLD_S and progress_mode == "all":
                        from agent.onboarding import (
                            TOOL_PROGRESS_FLAG,
                            is_seen,
                            mark_seen,
                            tool_progress_hint_gateway,
                        )
                        _cfg = _load_gateway_config()
                        gate_on = is_truthy_value(
                            cfg_get(_cfg, "display", "tool_progress_command"),
                            default=False,
                        )
                        if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                            long_tool_hint_fired[0] = True
                            progress_queue.put(tool_progress_hint_gateway())
                            mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
                except Exception as _hint_err:
                    logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
                return

            # "_thinking" is assistant scratch text between tool calls.  It
            # is never ordinary tool progress: only relay it when the platform
            # explicitly opted into thinking_progress.  Handle both legacy
            # callback shapes: ("_thinking", text) and
            # ("reasoning.available", "_thinking", text, ...).
            if event_type == "_thinking" or tool_name == "_thinking":
                if not _thinking_enabled:
                    return
                thinking_text = preview if tool_name == "_thinking" else tool_name
                msg = f"💬 {thinking_text}" if thinking_text else None
                if msg:
                    progress_queue.put(msg)
                return

            # If tool_progress is off, only _thinking passes through (above).
            # Regular tool calls are suppressed.
            if not tool_progress_enabled:
                return

            # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
            if event_type not in {"tool.started",}:
                return

            # Suppress tool-progress bubbles once the user has sent `stop`.
            # When the LLM response carries N parallel tool calls, the agent
            # fires N "tool.started" events back-to-back before checking for
            # interrupts — without this guard, a late `stop` still renders
            # all N as 🔍 bubbles, making the interrupt feel ignored.
            # (agent lives in run_sync's scope; agent_holder[0] is the shared
            # handle across nested scopes — see line ~9607.)
            try:
                _agent_for_interrupt = agent_holder[0] if agent_holder else None
                if _agent_for_interrupt is not None and getattr(
                    _agent_for_interrupt, "is_interrupted", False
                ):
                    return
            except Exception:
                pass

            # "new" mode: only report when tool changes
            if progress_mode == "new" and tool_name == last_tool[0]:
                return
            last_tool[0] = tool_name
            
            # Build progress message with primary argument preview
            from agent.display import get_tool_emoji
            emoji = get_tool_emoji(tool_name, default="⚙️")

            # Markdown-capable platforms render a terminal command as a fenced
            # code block instead of the compact `terminal: "cmd…"` preview.
            # Gated on the adapter's ``supports_code_blocks`` capability so
            # plain-text platforms keep the short line.  No language tag is
            # emitted — Slack mrkdwn renders the tag as a literal first code
            # line ("bash"), and a bare fence renders correctly everywhere
            # that supports blocks.
            #
            # Verbose mode shows the FULL command.  Non-verbose ("all"/"new")
            # modes still wrap in a fence but truncate to a single line capped
            # at ``tool_preview_length`` (default 40) so a long or multi-line
            # command doesn't render as a huge block — matching the budget the
            # non-terminal preview path already applies (#42634).
            _code_block_full = None
            _code_block_short = None
            try:
                _progress_adapter = self.adapters.get(source.platform)
            except Exception:
                _progress_adapter = None
            if (
                getattr(_progress_adapter, "supports_code_blocks", False)
                and tool_name == "terminal"
                and isinstance(args, dict)
                and isinstance(args.get("command"), str)
                and args["command"].strip()
            ):
                from agent.display import get_tool_preview_max_len
                _cmd_full = args["command"].rstrip()
                # Consecutive terminal calls: drop the repeated
                # "💻 terminal" header so back-to-back commands render as
                # adjacent code blocks under a single header.
                _block_header = (
                    "" if last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
                )
                _code_block_full = f"{_block_header}```\n{_cmd_full}\n```"
                # Single-line, capped preview for non-verbose modes.
                _pl = get_tool_preview_max_len()
                _cap = _pl if _pl > 0 else 40
                _lines = _cmd_full.splitlines()
                _cmd_short = _lines[0] if _lines else _cmd_full
                _multiline = len(_lines) > 1
                if len(_cmd_short) > _cap:
                    _cmd_short = _cmd_short[:_cap - 3] + "..."
                elif _multiline:
                    _cmd_short = _cmd_short + " ..."
                _code_block_short = f"{_block_header}```\n{_cmd_short}\n```"

            # Verbose mode: show detailed arguments, respects tool_preview_length
            if progress_mode == "verbose":
                if _code_block_full is not None:
                    last_was_terminal_block[0] = True
                    progress_queue.put(_code_block_full)
                    return
                last_was_terminal_block[0] = False
                if args:
                    from agent.display import get_tool_preview_max_len
                    _pl = get_tool_preview_max_len()
                    args_str = json.dumps(args, ensure_ascii=False, default=str)
                    # When tool_preview_length is 0 (default), don't truncate
                    # in verbose mode — the user explicitly asked for full
                    # detail.  Platform message-length limits handle the rest.
                    if _pl > 0 and len(args_str) > _pl:
                        args_str = args_str[:_pl - 3] + "..."
                    msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
                elif preview:
                    msg = f"{emoji} {tool_name}: \"{preview}\""
                else:
                    msg = f"{emoji} {tool_name}..."
                progress_queue.put(msg)
                return
            
            # "all" / "new" modes: short preview, respects tool_preview_length
            # config (defaults to 40 chars when unset to keep gateway messages
            # compact — unlike CLI spinners, these persist as permanent messages).
            # Terminal commands on markdown platforms get a single-line capped
            # fenced block (built above) instead of the truncated preview.
            if _code_block_short is not None:
                msg = _code_block_short
                last_was_terminal_block[0] = True
            elif preview:
                from agent.display import (
                    get_tool_preview_max_len,
                    get_tool_verb,
                    tool_verb_connector,
                    verb_drops_preview,
                )
                _pl = get_tool_preview_max_len()
                _cap = _pl if _pl > 0 else 40
                if len(preview) > _cap:
                    preview = preview[:_cap - 3] + "..."
                # Friendly labels: render a human-phrased line for built-in
                # tools ("🔍 Searching the web for ...") by prefixing the verb
                # onto the preview the callback already computed (so the
                # command/url/query is preserved).  Custom/plugin/MCP tools
                # have no verb and fall back to the raw "tool_name: ..." form.
                _verb = get_tool_verb(tool_name)
                if _verb:
                    if verb_drops_preview(tool_name):
                        msg = f"{emoji} {_verb}"
                    else:
                        msg = f"{emoji} {_verb}{tool_verb_connector(tool_name)}{preview}"
                else:
                    msg = f"{emoji} {tool_name}: \"{preview}\""
                last_was_terminal_block[0] = False
            else:
                msg = f"{emoji} {tool_name}..."
                last_was_terminal_block[0] = False
            
            # Dedup: collapse consecutive identical progress messages.
            # Common with execute_code where models iterate with the same
            # code (same boilerplate imports → identical previews).
            if msg == last_progress_msg[0]:
                repeat_count[0] += 1
                # Update the last line in progress_lines with a counter
                # via a special "dedup" queue message.
                progress_queue.put(("__dedup__", msg, repeat_count[0]))
                return
            last_progress_msg[0] = msg
            repeat_count[0] = 0
            
            progress_queue.put(msg)
        
        # Background task to send progress messages
        # Accumulates tool lines into a single message that gets edited.
        #
        # Threading metadata is platform-specific:
        # - Slack DM threading needs event_message_id fallback (reply thread)
        # - Telegram forum topics use message_thread_id; Hermes-created private
        #   DM topic lanes require both thread metadata and a reply anchor
        # - Feishu only honors reply_in_thread when sending a reply, so topic
        #   progress uses the triggering event message as the reply target
        # - Other platforms should use explicit source.thread_id only
        _progress_thread_id = _resolve_progress_thread_id(
            source.platform, source.thread_id, event_message_id,
        )
        _progress_metadata = (
            self._thread_metadata_for_source(source, event_message_id)
            if _progress_thread_id == source.thread_id
            else {"thread_id": _progress_thread_id}
        ) if _progress_thread_id else None
        _progress_metadata = _non_conversational_metadata(_progress_metadata, platform=source.platform)
        _progress_reply_to = (
            event_message_id
            if source.platform in (Platform.FEISHU, Platform.MATTERMOST) and source.thread_id and event_message_id
            else None
        )

        async def send_progress_messages():
            if not progress_queue:
                return

            adapter = self.adapters.get(source.platform)
            if not adapter:
                return

            # Skip tool progress for platforms that don't support message
            # editing (e.g. iMessage/BlueBubbles) — each progress update
            # would become a separate message bubble, which is noisy.
            if type(adapter).edit_message is BasePlatformAdapter.edit_message:
                while not progress_queue.empty():
                    try:
                        progress_queue.get_nowait()
                    except Exception:
                        break
                return

            progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
            progress_msg_id = None   # ID of the current progress message to edit
            can_edit = progress_grouping != "separate"  # "separate" = one message per tool (pre-v0.9 behavior)
            _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
            _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

            _progress_len_fn = (
                adapter.message_len_fn
                if isinstance(adapter, BasePlatformAdapter)
                else len
            )
            try:
                _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
            except Exception:
                _raw_progress_limit = 4000
            # Leave a little room for platform quirks / formatting.  For tiny
            # test adapters keep the limit usable instead of clamping to 500+.
            _PROGRESS_TEXT_LIMIT = max(
                1,
                _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
            )

            # Detect whether the adapter's edit_message accepts metadata so
            # overflow edits preserve Telegram topic/thread routing (#27487).
            _edit_accepts_metadata = False
            if _progress_metadata:
                try:
                    _edit_params = inspect.signature(adapter.edit_message).parameters
                    _edit_accepts_metadata = (
                        "metadata" in _edit_params
                        or any(
                            param.kind is inspect.Parameter.VAR_KEYWORD
                            for param in _edit_params.values()
                        )
                    )
                except (TypeError, ValueError):
                    _edit_accepts_metadata = False

            async def _edit_progress_message(message_id: str, content: str):
                kwargs = {
                    "chat_id": source.chat_id,
                    "message_id": message_id,
                    "content": content,
                }
                if getattr(adapter, "REQUIRES_EDIT_FINALIZE", False):
                    kwargs["finalize"] = True
                if _edit_accepts_metadata:
                    kwargs["metadata"] = _progress_metadata
                return await adapter.edit_message(**kwargs)

            def _progress_text(lines: list) -> str:
                return "\n".join(str(line) for line in lines)

            def _split_progress_groups(lines: list) -> list[list]:
                """Partition progress lines into platform-sized editable bubbles."""
                groups: list[list] = []
                current: list = []
                for line in lines:
                    candidate = current + [line]
                    if current and _progress_len_fn(_progress_text(candidate)) > _PROGRESS_TEXT_LIMIT:
                        groups.append(current)
                        current = [line]
                    else:
                        current = candidate
                if current:
                    groups.append(current)
                return groups

            def _track_progress_result(result) -> None:
                if (
                    _cleanup_progress
                    and getattr(result, "success", False)
                    and getattr(result, "message_id", None)
                ):
                    _cleanup_msg_ids.append(str(result.message_id))

            async def _send_progress_text(text: str):
                result = await adapter.send(
                    chat_id=source.chat_id,
                    content=text,
                    reply_to=_progress_reply_to,
                    metadata=_progress_metadata,
                )
                _track_progress_result(result)
                return result

            async def _roll_progress_overflow_if_needed() -> bool:
                """Start fresh editable progress bubbles before a bubble exceeds limit.

                Returns True when it delivered/split the current buffer and the
                caller should skip the normal send/edit path for this tick.
                """
                nonlocal progress_msg_id, progress_lines, can_edit
                if not progress_lines or not can_edit:
                    return False
                groups = _split_progress_groups(progress_lines)
                if len(groups) <= 1:
                    return False

                first_text = _progress_text(groups[0])
                if progress_msg_id is not None:
                    result = await _edit_progress_message(progress_msg_id, first_text)
                    if not result.success:
                        can_edit = False
                        # Fall back to the existing non-edit behavior below.
                        return False
                else:
                    result = await _send_progress_text(first_text)
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id

                for group in groups[1:]:
                    result = await _send_progress_text(_progress_text(group))
                    if result.success and result.message_id:
                        progress_msg_id = result.message_id

                # The newest continuation is now the only mutable bubble.  Keep
                # just its lines so subsequent edits update it instead of
                # replaying the full historical transcript into new messages.
                progress_lines = groups[-1]
                return True

            while True:
                try:
                    if not _run_still_current():
                        while not progress_queue.empty():
                            try:
                                progress_queue.get_nowait()
                            except Exception:
                                break
                        return

                    raw = progress_queue.get_nowait()

                    # Drain silently when interrupted: events queued in the
                    # window between tool parse and interrupt processing
                    # should not render as bubbles.  The "⚡ Interrupting
                    # current task" message is sent separately and is the
                    # last progress-flavored bubble the user should see.
                    try:
                        _agent_for_interrupt = agent_holder[0] if agent_holder else None
                        if _agent_for_interrupt is not None and getattr(
                            _agent_for_interrupt, "is_interrupted", False
                        ):
                            # Drop this event and continue draining.
                            await asyncio.sleep(0)
                            continue
                    except Exception:
                        pass

                    # Handle dedup messages: update last line with repeat counter
                    if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                        _, base_msg, count = raw
                        if progress_lines:
                            progress_lines[-1] = f"{base_msg} (×{count + 1})"
                        msg = progress_lines[-1] if progress_lines else base_msg
                    elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                        # Content bubble just landed on the platform — close off
                        # the current tool-progress bubble so the next tool
                        # starts a fresh bubble below the content. Without this,
                        # tool lines keep editing the ORIGINAL progress message
                        # above the new content, making the chat appear out of
                        # order. Mirrors GatewayStreamConsumer.on_segment_break
                        # on the content side. (Issue: tool + content
                        # linearization regression after PR #7885.)
                        progress_msg_id = None
                        progress_lines = []
                        last_progress_msg[0] = None
                        repeat_count[0] = 0
                        continue
                    else:
                        msg = raw
                        progress_lines.append(msg)

                    if await _roll_progress_overflow_if_needed():
                        _last_edit_ts = time.monotonic()
                        await asyncio.sleep(0.3)
                        if _run_still_current():
                            await adapter.send_typing(source.chat_id, metadata=_progress_metadata)
                        continue

                    # Throttle edits: batch rapid tool updates into fewer
                    # API calls to avoid hitting Telegram flood control.
                    # (grammY auto-retry pattern: proactively rate-limit
                    # instead of reacting to 429s.)
                    _now = time.monotonic()
                    _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                    if _remaining > 0:
                        # Wait out the throttle interval, then loop back to
                        # drain any additional queued messages before sending
                        # a single batched edit.
                        await asyncio.sleep(_remaining)
                        continue

                    if not _run_still_current():
                        return

                    if can_edit and progress_msg_id is not None:
                        # Try to edit the existing progress message
                        full_text = "\n".join(progress_lines)
                        result = await _edit_progress_message(progress_msg_id, full_text)
                        if not result.success:
                            _err = (getattr(result, "error", "") or "").lower()
                            # Transient network errors (ConnectError, timeouts)
                            # must not permanently disable progress-message
                            # editing — the next cycle can catch up.  Only
                            # permanent failures (flood control, message not
                            # found, permissions) should set can_edit = False.
                            if getattr(result, "retryable", False):
                                logger.debug(
                                    "[%s] Transient edit failure — keeping can_edit=True",
                                    adapter.name,
                                )
                                continue
                            if "flood" in _err or "retry after" in _err:
                                # Flood control hit — backoff but keep editing.
                                # Only disable edits for non-recoverable errors.
                                logger.info(
                                    "[%s] Progress edit flood control, backing off",
                                    adapter.name,
                                )
                                _last_edit_ts = time.monotonic()
                            else:
                                can_edit = False
                            _flood_result = await adapter.send(
                                chat_id=source.chat_id,
                                content=msg,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                            if (
                                _cleanup_progress
                                and getattr(_flood_result, "success", False)
                                and getattr(_flood_result, "message_id", None)
                            ):
                                _cleanup_msg_ids.append(str(_flood_result.message_id))
                    else:
                        if can_edit:
                            # First tool: send all accumulated text as new message
                            full_text = "\n".join(progress_lines)
                            result = await adapter.send(
                                chat_id=source.chat_id,
                                content=full_text,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                        else:
                            # Editing unsupported: send just this line
                            result = await adapter.send(
                                chat_id=source.chat_id,
                                content=msg,
                                reply_to=_progress_reply_to,
                                metadata=_progress_metadata,
                            )
                        if result.success and result.message_id:
                            progress_msg_id = result.message_id
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(str(result.message_id))

                    _last_edit_ts = time.monotonic()

                    # Restore typing indicator
                    await asyncio.sleep(0.3)
                    if _run_still_current():
                        await adapter.send_typing(source.chat_id, metadata=_progress_metadata)

                except queue.Empty:
                    await asyncio.sleep(0.3)
                except asyncio.CancelledError:
                    # Drain remaining queued messages
                    while not progress_queue.empty():
                        try:
                            raw = progress_queue.get_nowait()
                            if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                                _, base_msg, count = raw
                                if progress_lines:
                                    progress_lines[-1] = f"{base_msg} (×{count + 1})"
                                    await _roll_progress_overflow_if_needed()
                            elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                                # Content-bubble marker during drain: close off
                                # the current progress bubble and start a fresh
                                # one for any tool lines that arrived after.
                                await _roll_progress_overflow_if_needed()
                                if can_edit and progress_lines and progress_msg_id:
                                    _pending_text = _progress_text(progress_lines)
                                    try:
                                        await _edit_progress_message(progress_msg_id, _pending_text)
                                    except Exception:
                                        pass
                                progress_msg_id = None
                                progress_lines = []
                                last_progress_msg[0] = None
                                repeat_count[0] = 0
                            else:
                                progress_lines.append(raw)
                                await _roll_progress_overflow_if_needed()
                        except Exception:
                            break
                    # Final edit with all remaining tools (only if editing works)
                    if can_edit and progress_lines and progress_msg_id:
                        await _roll_progress_overflow_if_needed()
                    if can_edit and progress_lines and progress_msg_id:
                        full_text = _progress_text(progress_lines)
                        try:
                            await _edit_progress_message(progress_msg_id, full_text)
                        except Exception:
                            pass
                    return
                except Exception as e:
                    logger.error("Progress message error: %s", e)
                    await asyncio.sleep(1)
        
        # We need to share the agent instance for interrupt support
        agent_holder = [None]  # Mutable container for the agent instance
        result_holder = [None]  # Mutable container for the result
        tools_holder = [None]   # Mutable container for the tool definitions
        stream_consumer_holder = [None]  # Mutable container for stream consumer
        
        # Bridge sync step_callback → async hooks.emit for agent:step events
        _loop_for_step = asyncio.get_running_loop()
        _hooks_ref = self.hooks

        def _step_callback_sync(iteration: int, prev_tools: list) -> None:
            if not _run_still_current():
                return
            # prev_tools may be list[str] or list[dict] with "name"/"result"
            # keys.  Normalise to keep "tool_names" backward-compatible for
            # user-authored hooks that do ', '.join(tool_names)'.
            _names: list[str] = []
            for _t in (prev_tools or []):
                if isinstance(_t, dict):
                    _names.append(_t.get("name") or "")
                else:
                    _names.append(str(_t))
            safe_schedule_threadsafe(
                _hooks_ref.emit("agent:step", {
                    "platform": source.platform.value if source.platform else "",
                    "user_id": source.user_id,
                    "session_id": session_id,
                    "iteration": iteration,
                    "tool_names": _names,
                    "tools": prev_tools,
                }),
                _loop_for_step,
                logger=logger,
                log_message="agent:step hook scheduling error",
            )

        # Bridge sync event_callback → async hooks.emit for lifecycle events
        # (e.g. session:compress fires after context compression splits a session)
        def _event_callback_sync(event_type: str, context: dict) -> None:
            try:
                asyncio.run_coroutine_threadsafe(
                    _hooks_ref.emit(event_type, context),
                    _loop_for_step,
                )
            except Exception as _e:
                logger.debug("event_callback hook error: %s", _e)

        # Bridge sync status_callback → async adapter.send for context pressure
        _status_adapter = self.adapters.get(source.platform)
        _status_chat_id = source.chat_id
        if source.platform == Platform.FEISHU and source.thread_id and event_message_id:
            # Feishu topics only keep messages inside the topic when they are
            # sent via the reply API with reply_in_thread=true. Status/interim,
            # approval, and stream-consumer paths usually only receive metadata,
            # so carry the triggering message id as a Feishu-specific fallback.
            _status_thread_metadata: Optional[Dict[str, Any]] = {
                "thread_id": _progress_thread_id,
                "reply_to_message_id": event_message_id,
            }
        else:
            _status_thread_metadata = self._thread_metadata_for_source(source, event_message_id) if _progress_thread_id else None

        def _status_callback_sync(event_type: str, message: str) -> None:
            if not _status_adapter or not _run_still_current():
                return
            prepared_message = _prepare_gateway_status_message(
                source.platform,
                event_type,
                message,
            )
            if prepared_message is None:
                logger.debug(
                    "status_callback suppressed for %s/%s: %s",
                    source.platform.value if source.platform else "unknown",
                    event_type,
                    _redact_gateway_user_facing_secrets(str(message or ""))[:160],
                )
                return
            _fut = safe_schedule_threadsafe(
                _send_or_update_status_coro(_status_adapter, _status_chat_id, event_type, prepared_message, _status_thread_metadata),
                _loop_for_step,
                logger=logger,
                log_message=f"status_callback ({event_type}) scheduling error",
            )
            if _fut is None:
                return
            if _cleanup_progress:
                def _track_status_id(fut) -> None:
                    try:
                        res = fut.result()
                    except Exception:
                        return
                    mid = getattr(res, "message_id", None)
                    if getattr(res, "success", False) and mid:
                        _cleanup_msg_ids.append(str(mid))
                _fut.add_done_callback(_track_status_id)

        def run_sync():
            # The conditional re-assignment of `message` further below
            # (prepending model-switch notes) makes Python treat it as a
            # local variable in the entire function.  `nonlocal` lets us
            # read *and* reassign the outer `_run_agent` parameter without
            # triggering an UnboundLocalError on the earlier read at
            # `_resolve_turn_agent_config(message, …)`.
            nonlocal message

            # session_key is propagated via contextvars in _set_session_env()
            # (_SESSION_KEY) and via set_current_session_key() (_approval_session_key)
            # below — both concurrency-safe and inherited by tool worker threads.
            # We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
            # os.environ is process-global, so concurrent gateway sessions (e.g.
            # two Discord threads) would clobber each other's value, and a tool
            # thread whose contextvar is unset would fall back to os.environ and
            # read the wrong session key — misrouting command-approval prompts to
            # the wrong thread (#24100). The non-gateway surfaces don't depend on
            # this write: CLI and cron bind the session via contextvars
            # (set_current_session_key / session context), and only the TUI
            # slash-worker *subprocess* exports HERMES_SESSION_KEY (from its own
            # --session-key argv, a separate process) — so removing this in-process
            # gateway write does not affect any of them.

            # Map platform enum to the platform hint key the agent understands.
            # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
            platform_key = "cli" if source.platform == Platform.LOCAL else source.platform.value
            
            # Combine platform context, per-channel context, and the user-configured
            # ephemeral system prompt.
            combined_ephemeral = context_prompt or ""
            event_channel_prompt = (channel_prompt or "").strip()
            if event_channel_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
            if self._ephemeral_system_prompt:
                combined_ephemeral = (combined_ephemeral + "\n\n" + self._ephemeral_system_prompt).strip()

            max_iterations = _current_max_iterations()

            try:
                model, runtime_kwargs = self._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                    user_config=user_config,
                )
                logger.debug(
                    "run_agent resolved: model=%s provider=%s session=%s",
                    model, runtime_kwargs.get("provider"), session_key or "",
                )
            except Exception as exc:
                return {
                    "final_response": f"⚠️ Provider authentication failed: {exc}",
                    "messages": [],
                    "api_calls": 0,
                    "tools": [],
                }

            pr = self._provider_routing
            reasoning_config = self._resolve_session_reasoning_config(
                source=source,
                session_key=session_key,
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._load_service_tier()
            # Set up stream consumer for token streaming or interim commentary.
            _stream_consumer = None
            _stream_delta_cb = None
            _scfg = getattr(getattr(self, 'config', None), 'streaming', None)
            if _scfg is None:
                from gateway.config import StreamingConfig
                _scfg = StreamingConfig()

            # Per-platform streaming gate: display.platforms.<plat>.streaming
            # can disable streaming for specific platforms even when the global
            # streaming config is enabled.
            _plat_streaming = resolve_display_setting(
                user_config, platform_key, "streaming"
            )
            # None = no per-platform override → follow global config
            _streaming_enabled = (
                _scfg.enabled and _scfg.transport != "off"
                if _plat_streaming is None
                else bool(_plat_streaming)
            )
            _want_stream_deltas = _streaming_enabled
            _want_interim_messages = interim_assistant_messages_enabled
            _want_interim_consumer = _want_interim_messages
            if _want_stream_deltas or _want_interim_consumer:
                try:
                    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
                    _adapter = self.adapters.get(source.platform)
                    if _adapter:
                        _pause_typing_before_finalize = None
                        if source.platform == Platform.TELEGRAM and hasattr(_adapter, "pause_typing_for_chat"):
                            def _pause_typing_before_finalize(
                                _adapter=_adapter,
                                _chat_id=source.chat_id,
                            ) -> None:
                                _adapter.pause_typing_for_chat(_chat_id)
                        # Platforms that don't support editing sent messages
                        # (e.g. QQ, WeChat) should skip streaming entirely —
                        # without edit support, the consumer sends a partial
                        # first message that can never be updated, resulting in
                        # duplicate messages (partial + final).
                        _adapter_supports_edit = getattr(_adapter, "SUPPORTS_MESSAGE_EDITING", True)
                        if not _adapter_supports_edit:
                            raise RuntimeError("skip streaming for non-editable platform")
                        _effective_cursor = _scfg.cursor
                        # Some Matrix clients render the streaming cursor
                        # as a visible tofu/white-box artifact.  Keep
                        # streaming text on Matrix, but suppress the cursor.
                        _buffer_only = False
                        if source.platform == Platform.MATRIX:
                            _effective_cursor = ""
                            _buffer_only = True
                        # Fresh-final applies to Telegram only — other
                        # platforms either edit in place cheaply or don't
                        # have the edit-timestamp-stays-stale problem.
                        # (Ported from openclaw/openclaw#72038.)
                        _fresh_final_secs = (
                            float(getattr(_scfg, "fresh_final_after_seconds", 0.0) or 0.0)
                            if source.platform == Platform.TELEGRAM
                            else 0.0
                        )
                        _consumer_cfg = StreamConsumerConfig(
                            edit_interval=_scfg.edit_interval,
                            buffer_threshold=_scfg.buffer_threshold,
                            cursor=_effective_cursor,
                            buffer_only=_buffer_only,
                            fresh_final_after_seconds=_fresh_final_secs,
                            transport=_scfg.transport or "edit",
                            chat_type=getattr(source, "chat_type", "") or "",
                        )
                        _stream_consumer = GatewayStreamConsumer(
                            adapter=_adapter,
                            chat_id=source.chat_id,
                            config=_consumer_cfg,
                            metadata=_status_thread_metadata,
                            on_new_message=(
                                (lambda: progress_queue.put(("__reset__",)))
                                if progress_queue is not None
                                else None
                            ),
                            on_before_finalize=_pause_typing_before_finalize,
                            initial_reply_to_id=event_message_id,
                            run_still_current=_run_still_current,
                        )
                        if _want_stream_deltas:
                            def _stream_delta_cb(text: str) -> None:
                                if _run_still_current():
                                    _stream_consumer.on_delta(text)
                        stream_consumer_holder[0] = _stream_consumer
                except Exception as _sc_err:
                    logger.debug("Could not set up stream consumer: %s", _sc_err)

            def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
                if not _run_still_current():
                    return
                if _stream_consumer is not None:
                    if already_streamed:
                        _stream_consumer.on_segment_break()
                    else:
                        _stream_consumer.on_commentary(text)
                    return
                if already_streamed or not _status_adapter or not str(text or "").strip():
                    return
                safe_schedule_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        text,
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="interim_assistant_callback scheduling error",
                )

            turn_route = self._resolve_turn_agent_config(message, model, runtime_kwargs)

            # Check agent cache — reuse the AIAgent from the previous message
            # in this session to preserve the frozen system prompt and tool
            # schemas for prompt cache hits.
            _sig = self._agent_config_signature(
                turn_route["model"],
                turn_route["runtime"],
                enabled_toolsets,
                combined_ephemeral,
                cache_keys=self._extract_cache_busting_config(user_config),
                user_id=getattr(source, "user_id", None),
                user_id_alt=getattr(source, "user_id_alt", None),
            )
            agent = None
            reused_cached_agent = False
            _cache_lock = getattr(self, "_agent_cache_lock", None)
            _cache = getattr(self, "_agent_cache", None)

            # Detect cross-process writes: when another process (e.g. hermes
            # dashboard) appends to the same session in the shared SessionDB,
            # the cached agent's in-memory transcript becomes stale.  Compare
            # the session's current message_count against the count recorded
            # when the agent was cached; on mismatch, invalidate the cache
            # so a fresh agent re-reads from disk. (#45966)
            _current_msg_count = None
            if self._session_db is not None and session_id:
                try:
                    # run_sync is off-loop (executor); sync DB is fine.
                    _sess_row = self._session_db._db.get_session(session_id)
                    if _sess_row:
                        _current_msg_count = _sess_row.get("message_count", 0)
                except Exception:
                    pass

            _xproc_evicted_agent = None
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    cached = _cache.get(session_key)
                    if cached and cached[1] == _sig:
                        # cached[2] is the message_count at cache time;
                        # stale when a second process appended rows.
                        # cached[3] (when present) is the session_id the
                        # snapshot was taken for — used to skip the guard
                        # when the active session_id differs (#54947).
                        _cached_mc = cached[2] if len(cached) > 2 else None
                        _cached_sid = cached[3] if len(cached) > 3 else None
                        # If the snapshot belongs to a different session_id
                        # (same session_key, different conversation), the
                        # message_count comparison is meaningless — the
                        # counts track DIFFERENT DB rows.  REUSE the cached
                        # agent rather than rebuild and bust the prompt cache
                        # on every session switch (#54947).
                        _session_id_mismatch = (
                            _cached_sid is not None
                            and session_id is not None
                            and _cached_sid != session_id
                        )
                        if (
                            not _session_id_mismatch
                            and _cached_mc is not None
                            and _current_msg_count is not None
                            and _current_msg_count != _cached_mc
                        ):
                            # Cross-process write detected — discard stale
                            # agent so it rebuilds from fresh DB transcript.
                            logger.info(
                                "Agent cache invalidated for session %s: "
                                "message_count changed (%s -> %s), "
                                "possible cross-process write",
                                session_key, _cached_mc, _current_msg_count,
                            )
                            evicted = self._agent_cache.pop(session_key, None)
                            _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                            if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                                # Defer cleanup until AFTER the lock is
                                # released — _cleanup_agent_resources /
                                # release_clients can block on memory-provider
                                # shutdown and socket teardown, and running it
                                # here would stall the gateway event loop while
                                # _sweep_idle_cached_agents (session-expiry
                                # watcher) waits on the same lock, blocking
                                # Discord heartbeats (#52197).  The same session
                                # rebuilds a fresh agent immediately below, so
                                # use the SOFT release that preserves the
                                # session's terminal sandbox / browser / bg
                                # processes for the rebuilt agent to inherit —
                                # mirrors _evict_cached_agent / idle-sweep.
                                _xproc_evicted_agent = _ev_agent
                        else:
                            agent = cached[0]
                            # Refresh LRU order so the cap enforcement evicts
                            # truly-oldest entries, not the one we just used.
                            if hasattr(_cache, "move_to_end"):
                                try:
                                    _cache.move_to_end(session_key)
                                except KeyError:
                                    pass
                            self._init_cached_agent_for_turn(agent, _interrupt_depth)
                            # Refresh agent max_iterations from current config
                            # (cached agent may have been created with old config)
                            agent.max_iterations = max_iterations
                            logger.debug("Reusing cached agent for session %s", session_key)
                            reused_cached_agent = True

            # Lock released — now schedule cleanup of any cross-process-evicted
            # agent on a daemon thread so memory-provider shutdown / socket
            # teardown never blocks the gateway event loop or the cache lock
            # the session-expiry watcher needs (#52197).
            if _xproc_evicted_agent is not None:
                try:
                    threading.Thread(
                        target=self._release_evicted_agent_soft,
                        args=(_xproc_evicted_agent,),
                        daemon=True,
                        name=f"agent-xproc-evict-{str(session_key)[:24]}",
                    ).start()
                except Exception:
                    # Interpreter shutdown or thread-spawn failure — release
                    # inline as a best-effort fallback.
                    try:
                        self._release_evicted_agent_soft(_xproc_evicted_agent)
                    except Exception:
                        pass

            if agent is None:
                # Config changed or first message — create fresh agent
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    ephemeral_system_prompt=combined_ephemeral or None,
                    prefill_messages=self._prefill_messages or None,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=session_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    gateway_session_key=session_key,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    fallback_model=self._fallback_model,
                )
                if _cache_lock and _cache is not None:
                    with _cache_lock:
                        # Record the session_id the snapshot was taken for
                        # alongside the message_count, so the cross-process
                        # guard can skip the (meaningless) count comparison
                        # when the active session_id later switches under
                        # the same session_key (#54947).
                        _cache[session_key] = (
                            agent, _sig, _current_msg_count, session_id,
                        )
                        self._enforce_agent_cache_cap()
                logger.debug("Created new agent for session %s (sig=%s)", session_key, _sig)

            # Per-message state — callbacks and reasoning config change every
            # turn and must not be baked into the cached agent constructor.
            # Gate on needs_progress_queue (tool_progress OR thinking_progress)
            # rather than tool_progress alone: the progress_callback also relays
            # _thinking assistant scratch text, which is gated on
            # thinking_progress and is intentionally independent of tool
            # progress. With the old `tool_progress_enabled`-only gate, a user
            # who set thinking_progress:true but kept tool_progress:off got a
            # None callback — so _thinking scratch bubbles never relayed even
            # though the progress queue was created for them.
            agent.tool_progress_callback = progress_callback if needs_progress_queue else None
            # Discord voice verbal-ack hook (fires once per turn on first tool
            # call; armed only when in a voice channel with the mixer running).
            agent.tool_start_callback = (
                voice_ack_callback if _voice_ack_guild[0] is not None else None
            )
            agent.step_callback = _step_callback_sync if _hooks_ref.loaded_hooks else None
            agent.stream_delta_callback = _stream_delta_cb
            agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
            agent.status_callback = _status_callback_sync
            # Credits / out-of-band notices (usage bands, depletion, restored).
            # Messaging has no persistent status bar, so each notice is a
            # standalone push: render to a single plaintext line and deliver via
            # the shared _deliver_platform_notice rail (honors private/public +
            # thread metadata). Fires from the agent's sync worker thread, so we
            # hop onto the gateway loop with safe_schedule_threadsafe - same
            # pattern as _status_callback_sync. The fired-once latch lives on the
            # cached agent and persists across turns, so a band crosses -> one
            # push (no per-turn re-nag). Recovery ("✓ Credit access restored")
            # rides the same show path (it's emitted as a success notice, not a
            # clear). The clear callback is a no-op: a sent platform message
            # can't be cleanly retracted, and the band already fired once.
            def _notice_callback_sync(notice) -> None:
                if not _status_adapter or not _run_still_current():
                    return
                try:
                    line = render_notice_line(notice)
                except Exception:
                    logger.debug("render_notice_line failed", exc_info=True)
                    return
                if not line:
                    return
                safe_schedule_threadsafe(
                    self._deliver_platform_notice(source, line),
                    _loop_for_step,
                    logger=logger,
                    log_message="notice_callback delivery scheduling error",
                )

            agent.notice_callback = _notice_callback_sync
            agent.notice_clear_callback = None
            agent.event_callback = _event_callback_sync
            agent.reasoning_config = reasoning_config
            agent.service_tier = self._service_tier
            agent.request_overrides = turn_route.get("request_overrides") or {}

            _bg_review_release = threading.Event()
            _bg_review_pending: list[str] = []
            _bg_review_pending_lock = threading.Lock()

            def _deliver_bg_review_message(message: str) -> None:
                if not _status_adapter or not _run_still_current():
                    return
                safe_schedule_threadsafe(
                    _status_adapter.send(
                        _status_chat_id,
                        message,
                        metadata=_non_conversational_metadata(_status_thread_metadata, platform=source.platform),
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="background_review_callback scheduling error",
                )

            def _release_bg_review_messages() -> None:
                _bg_review_release.set()
                with _bg_review_pending_lock:
                    pending = list(_bg_review_pending)
                    _bg_review_pending.clear()
                for queued in pending:
                    _deliver_bg_review_message(queued)

            # Background review delivery — send "💾 Memory updated" etc. to user
            def _bg_review_send(message: str) -> None:
                if not _status_adapter or not _run_still_current():
                    return
                if not _bg_review_release.is_set():
                    with _bg_review_pending_lock:
                        if not _bg_review_release.is_set():
                            _bg_review_pending.append(message)
                            return
                _deliver_bg_review_message(message)

            agent.background_review_callback = _bg_review_send
            # Register the release hook on the adapter so base.py's finally
            # block can fire it after delivering the main response.
            if _status_adapter and session_key:
                if getattr(type(_status_adapter), "register_post_delivery_callback", None) is not None:
                    _status_adapter.register_post_delivery_callback(
                        session_key,
                        _release_bg_review_messages,
                        generation=run_generation,
                    )
                else:
                    _pdc = getattr(_status_adapter, "_post_delivery_callbacks", None)
                    if _pdc is not None:
                        _pdc[session_key] = _release_bg_review_messages
            # Memory update notifications in chat.  Config: display.memory_notifications
            #   off     — no chat notification (still logged to stdout)
            #   on      — generic "💾 Memory updated" (default)
            #   verbose — content preview: "💾 Memory ➕ Hermes Repo..."
            _mem_notif = user_config.get("display", {}).get("memory_notifications")
            if isinstance(_mem_notif, bool):
                _mem_notif = "on" if _mem_notif else "off"
            agent.memory_notifications = str(_mem_notif).lower() if _mem_notif else "on"

            # ------------------------------------------------------------------
            # Clarify callback: present a clarify prompt and block on a response.
            #
            # Runs on the agent's worker thread (see clarify_tool's synchronous
            # callback contract).  Bridges sync→async by scheduling the
            # adapter's send_clarify on the gateway event loop, then blocks on
            # the clarify primitive's threading.Event with a configurable
            # timeout.  Returns the user's response string, or a sentinel
            # explaining that no response arrived (so the agent can adapt
            # rather than hang forever).
            # ------------------------------------------------------------------
            def _clarify_callback_sync(question: str, choices) -> str:
                from tools import clarify_gateway as _clarify_mod
                import uuid as _uuid

                if not _status_adapter:
                    return ""

                clarify_id = _uuid.uuid4().hex[:10]
                _clarify_mod.register(
                    clarify_id=clarify_id,
                    session_key=session_key or "",
                    question=question,
                    choices=list(choices) if choices else None,
                )

                # Pause typing — like approval, we don't want a "thinking..."
                # status to obscure the prompt or block the user from typing
                # an "Other" response on platforms that disable input while
                # typing is active (Slack Assistant API).
                try:
                    _status_adapter.pause_typing_for_chat(_status_chat_id)
                except Exception:
                    pass

                send_ok = False
                fut = safe_schedule_threadsafe(
                    _status_adapter.send_clarify(
                        chat_id=_status_chat_id,
                        question=question,
                        choices=list(choices) if choices else None,
                        clarify_id=clarify_id,
                        session_key=session_key or "",
                        metadata=_status_thread_metadata,
                    ),
                    _loop_for_step,
                    logger=logger,
                    log_message="Clarify send failed to schedule",
                )
                if fut is None:
                    send_ok = False
                else:
                    try:
                        result = fut.result(timeout=15)
                        send_ok = bool(getattr(result, "success", False))
                    except Exception as exc:
                        logger.warning("Clarify send failed: %s", exc)
                        send_ok = False

                if not send_ok:
                    # Couldn't deliver the prompt — clean up and return
                    # sentinel so the agent can fall back to a sensible
                    # default rather than hanging.
                    _clarify_mod.clear_session(session_key or "")
                    return "[clarify prompt could not be delivered]"

                timeout = _clarify_mod.get_clarify_timeout()
                response = _clarify_mod.wait_for_response(clarify_id, timeout=float(timeout))
                if response is None or response == "":
                    # Timeout or session-boundary cancellation
                    return f"[user did not respond within {int(timeout / 60)}m]"
                return response

            agent.clarify_callback = _clarify_callback_sync

            # Show assistant thinking between tool calls — independent of
            # tool_progress mode. Mattermost needs an explicit per-platform
            # opt-in so global scratch-text display does not leak into threads.
            agent.thinking_progress = _thinking_enabled
            # Store agent reference for interrupt support
            agent_holder[0] = agent
            # Capture the full tool definitions for transcript logging
            tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None
            
            # Convert history to agent format.
            # Two cases:
            #   1. Normal path (from transcript): simple {role, content, timestamp} dicts
            #      - Strip timestamps, keep role+content
            #   2. Interrupt path (from agent result["messages"]): full agent messages
            #      that may include tool_calls, tool_call_id, reasoning, etc.
            #      - These must be passed through intact so the API sees valid
            #        assistant→tool sequences (dropping tool_calls causes 500 errors)
            #
            # Telegram observed group context is handled structurally here:
            # observed=True transcript rows are withheld from replayable
            # history and attached to the current addressed message as
            # API-only context, so persisted history stores only the real
            # addressed user turn.
            agent_history, observed_group_context = _build_gateway_agent_history(
                history,
                channel_prompt=channel_prompt,
                inject_timestamps=_message_timestamps_enabled(_load_gateway_config()),
            )

            # FTS write-corruption guard (#50502): when message persistence
            # fails silently through corrupt FTS triggers, the reloaded
            # transcript above is stale/empty even though the SAME cached agent
            # still holds the full live conversation in `_session_messages`.
            # Replacing the live transcript with that shorter copy causes
            # immediate same-session amnesia. Only applies when we reused a
            # cached agent bound to this exact session_id.
            if reused_cached_agent and getattr(agent, "session_id", None) == session_id:
                _selected = _select_cached_agent_history(
                    agent_history, getattr(agent, "_session_messages", None)
                )
                if _selected is not agent_history:
                    logger.warning(
                        "Persisted transcript lagged live cached history for "
                        "session %s (disk=%d, memory=%d); preserving live "
                        "conversation context (possible FTS write corruption)",
                        session_key, len(agent_history), len(_selected),
                    )
                    agent_history = _selected
            
            # Collect MEDIA paths already in history so we can exclude them
            # from the current turn's extraction. This is compression-safe:
            # even if the message list shrinks, we know which paths are old.
            _history_media_paths: set = _collect_history_media_paths(agent_history)
            
            # Register per-session gateway approval callback so dangerous
            # command approval blocks the agent thread (mirrors CLI input()).
            # The callback bridges sync→async to send the approval request
            # to the user immediately.
            from tools.approval import (
                register_gateway_notify,
                reset_current_session_key,
                set_current_session_key,
                unregister_gateway_notify,
            )

            def _approval_notify_sync(approval_data: dict) -> None:
                """Send the approval request to the user from the agent thread.

                If the adapter supports interactive button-based approvals
                (e.g. Discord's ``send_exec_approval``), use that for a richer
                UX.  Otherwise fall back to a plain text message with
                ``/approve`` instructions.
                """
                # Pause the typing indicator while the agent waits for
                # user approval.  Critical for Slack's Assistant API where
                # assistant_threads_setStatus disables the compose box — the
                # user literally cannot type /approve while "is thinking..."
                # is active.  The approval message send auto-clears the Slack
                # status; pausing prevents _keep_typing from re-setting it.
                # Typing resumes in _handle_approve_command/_handle_deny_command.
                _status_adapter.pause_typing_for_chat(_status_chat_id)

                cmd = approval_data.get("command", "")
                desc = approval_data.get("description", "dangerous command")

                # Redact credentials from the command before displaying it in
                # the approval prompt — Tirith's findings are already redacted,
                # but the raw command string still leaks secrets to the chat
                # platform (#48456). Applied here so BOTH the button-based
                # (send_exec_approval) and plain-text fallback paths below use
                # the redacted value.
                cmd = _redact_approval_command(cmd)

                # Prefer button-based approval when the adapter supports it.
                # Check the *class* for the method, not the instance — avoids
                # false positives from MagicMock auto-attribute creation in tests.
                if getattr(type(_status_adapter), "send_exec_approval", None) is not None:
                    try:
                        _approval_fut = safe_schedule_threadsafe(
                            _status_adapter.send_exec_approval(
                                chat_id=_status_chat_id,
                                command=cmd,
                                session_key=_approval_session_key,
                                description=desc,
                                metadata=_status_thread_metadata,
                            ),
                            _loop_for_step,
                            logger=logger,
                            log_message="send_exec_approval scheduling error",
                        )
                        if _approval_fut is None:
                            raise RuntimeError("send_exec_approval: loop unavailable")
                        _approval_result = _approval_fut.result(timeout=15)
                        if _approval_result.success:
                            return
                        logger.warning(
                            "Button-based approval failed (send returned error), falling back to text: %s",
                            _approval_result.error,
                        )
                    except Exception as _e:
                        logger.warning(
                            "Button-based approval failed, falling back to text: %s", _e
                        )

                # Fallback: plain text approval prompt.  Use the adapter's
                # typed prefix so Slack/Matrix users are told the form they
                # can actually type (`!approve`) — typed "/" is blocked in
                # Slack threads and reserved by Matrix clients.
                _p = getattr(_status_adapter, "typed_command_prefix", "/")
                cmd_preview = cmd[:200] + "..." if len(cmd) > 200 else cmd
                msg = (
                    f"⚠️ **Dangerous command requires approval:**\n"
                    f"```\n{cmd_preview}\n```\n"
                    f"Reason: {desc}\n\n"
                    f"Reply `{_p}approve` to execute, `{_p}approve session` to approve this pattern "
                    f"for the session, `{_p}approve always` to approve permanently, or `{_p}deny` to cancel."
                )
                try:
                    _approval_send_fut = safe_schedule_threadsafe(
                        _status_adapter.send(
                            _status_chat_id,
                            msg,
                            metadata=_status_thread_metadata,
                        ),
                        _loop_for_step,
                        logger=logger,
                        log_message="Approval text-send scheduling error",
                    )
                    if _approval_send_fut is not None:
                        _approval_send_fut.result(timeout=15)
                except Exception as _e:
                    logger.error("Failed to send approval request: %s", _e)

            # Keep real user text separate from API-only recovery guidance.  If
            # an auto-continue note is prepended below, persist the original
            # message so stale guidance never replays as user-authored text.
            _persist_user_message_override: Optional[Any] = persist_user_message
            _persist_user_timestamp_override: Optional[float] = persist_user_timestamp

            # Prepend pending model switch note so the model knows about the switch
            _pending_notes = getattr(self, '_pending_model_notes', {})
            _msn = _pending_notes.pop(session_key, None) if session_key else None
            if _msn:
                message = _msn + "\n\n" + message

            # Auto-continue: if the loaded history ends with a tool result,
            # the previous agent turn was interrupted mid-work (gateway
            # restart, crash, SIGTERM).  Prepend a system note so the model
            # finishes processing the pending tool results before addressing
            # the user's new message.  (#4493)
            #
            # Session-level resume_pending (set on drain-timeout shutdown)
            # escalates the wording — the transcript's last role may be
            # anything (tool, assistant with unfinished work, etc.), so we
            # give a stronger, reason-aware instruction that subsumes the
            # tool-tail case.
            #
            # Freshness gate (#16802): both branches are gated on the age
            # of the last persisted transcript row.  That is the correct
            # "when did we last do anything here" signal for both the
            # resume_pending path (restart watchdog) and the tool-tail
            # path (in-flight tool loop killed).  We read ``history[-1]``
            # here because ``agent_history`` has already stripped the
            # ``timestamp`` field off tool/tool_call rows for API purity
            # (see the `k != "timestamp"` filter above).  Rows without a
            # timestamp (legacy transcripts) are treated as fresh so the
            # historical auto-continue behaviour is preserved.
            _freshness_window = _auto_continue_freshness_window()
            _interruption_is_fresh = _is_fresh_gateway_interruption(
                _last_transcript_timestamp(history),
                window_secs=_freshness_window,
            )

            _resume_entry = None
            if session_key:
                try:
                    _resume_entry = self.session_store._entries.get(session_key)
                except Exception:
                    _resume_entry = None

            # resume_pending freshness uses a SECOND signal in addition to the
            # transcript clock above.  The restart watchdog stamps the session
            # with ``last_resume_marked_at`` at interrupt time — that is the
            # correct "when were we interrupted" signal.  The transcript clock
            # (_interruption_is_fresh) can be far older: an active thread you
            # return to may have its last persisted row hours back, even though
            # the interruption itself just happened.  Gating resume_pending on
            # the transcript clock alone makes the recovery note silently drop,
            # and because the startup auto-resume turn carries empty text
            # (_schedule_resume_pending_sessions), the model then receives a
            # blank user message and replies with confused "the message came
            # through blank" noise.  Treat the marker as fresh when
            # EITHER signal is fresh so the two freshness checks agree.
            _resume_mark_is_fresh = False
            if _resume_entry is not None and getattr(_resume_entry, "resume_pending", False):
                _resume_mark_is_fresh = _is_fresh_gateway_interruption(
                    getattr(_resume_entry, "last_resume_marked_at", None),
                    window_secs=_freshness_window,
                )
            _is_resume_pending = bool(
                _resume_entry is not None
                and getattr(_resume_entry, "resume_pending", False)
                and (_interruption_is_fresh or _resume_mark_is_fresh)
            )
            _has_fresh_tool_tail = bool(
                agent_history
                and agent_history[-1].get("role") == "tool"
                and _interruption_is_fresh
            )

            if _is_resume_pending:
                _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
                _reason_phrase = (
                    "a gateway restart"
                    if _reason == "restart_timeout"
                    else "a gateway shutdown"
                    if _reason == "shutdown_timeout"
                    else "a gateway interruption"
                )
                _persist_user_message_override = message
                # The empty-message case is the auto-resume startup turn
                # synthesized by _schedule_resume_pending_sessions — there is
                # no NEW user message to address, so tell the model to report
                # recovery instead of the (nonexistent) "new message".
                if message:
                    _resume_guidance = (
                        "Address the user's NEW message below FIRST and focus "
                        "on what the user is asking now."
                    )
                else:
                    _resume_guidance = (
                        "Report to the user that the session was restored "
                        "successfully and ask what they would like to do next."
                    )
                message = (
                    f"[System note: The previous turn was interrupted by "
                    f"{_reason_phrase}; the gateway is now back online. "
                    f"Any restart/shutdown command in the history has already "
                    f"run — do NOT re-execute or verify it. {_resume_guidance} "
                    f"Do NOT re-execute old tool calls — skip any unfinished "
                    f"work from the conversation history.]"
                    + (f"\n\n{message}" if message else "")
                )
            elif _has_fresh_tool_tail:
                _persist_user_message_override = message
                message = (
                    "[System note: A new message has arrived. The conversation "
                    "history contains pending tool outputs from an interrupted turn. "
                    "IGNORE those pending results. Address the user's NEW message "
                    "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                    + message
                )

            # Consume one-shot /reload-skills note (if the user ran
            # /reload-skills since their last turn in this session). Same
            # queue pattern as CLI: prepend to the NEXT user message, then
            # clear. Nothing was written to the transcript out-of-band, so
            # message alternation stays intact.
            _pending_notes = getattr(self, "_pending_skills_reload_notes", None)
            if _pending_notes and session_key and session_key in _pending_notes:
                _srn = _pending_notes.pop(session_key, None)
                if _srn:
                    message = _srn + "\n\n" + message

            # Safety net: a startup auto-resume event carries empty
            # text and relies on the resume_pending branch above to supply the
            # recovery note.  If that branch did not fire for any reason (e.g.
            # both freshness signals disagreed, or the marker was cleared
            # between scheduling and dispatch) we must NOT hand the model a
            # blank user turn — it responds with confused "the message came
            # through blank" noise.  Restricted to resume_pending sessions so
            # legitimately empty user turns (e.g. an image with no caption,
            # wrapped as native content below) are untouched.
            if (
                isinstance(message, str)
                and not message.strip()
                and _resume_entry is not None
                and getattr(_resume_entry, "resume_pending", False)
            ):
                _sn_reason = (
                    getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
                )
                _sn_reason_phrase = (
                    "a gateway restart"
                    if _sn_reason == "restart_timeout"
                    else "a gateway shutdown"
                    if _sn_reason == "shutdown_timeout"
                    else "a gateway interruption"
                )
                message = (
                    f"[System note: The previous turn was interrupted by "
                    f"{_sn_reason_phrase}; the gateway is now back online. "
                    f"Any restart/shutdown command in the history has already "
                    f"run — do NOT re-execute or verify it. Report to the user "
                    f"that the session was restored successfully and ask what "
                    f"they would like to do next. Do NOT re-execute old tool "
                    f"calls — skip any unfinished work from the conversation "
                    f"history.]"
                )

            _approval_session_key = session_key or ""
            _approval_session_token = set_current_session_key(_approval_session_key)
            register_gateway_notify(_approval_session_key, _approval_notify_sync)
            try:
                # If _prepare_inbound_message_text buffered image paths for native
                # attachment, wrap the user turn as an OpenAI-style multimodal
                # content list. Consume-and-clear so subsequent turns on the same
                # runner instance don't re-attach stale images.
                _native_imgs = self._consume_pending_native_image_paths(session_key)
                if _native_imgs:
                    try:
                        from agent.image_routing import build_native_content_parts
                        _parts, _skipped = build_native_content_parts(
                            message,
                            _native_imgs,
                        )
                        if _skipped:
                            logger.warning(
                                "Native image attachment: skipped %d unreadable path(s): %s",
                                len(_skipped), _skipped,
                            )
                        if any(p.get("type") == "image_url" for p in _parts):
                            _run_message: Any = _parts
                        else:
                            # All images failed to read — fall back to plain text.
                            _run_message = message
                    except Exception as _img_exc:
                        logger.warning(
                            "Native image attachment failed, falling back to text: %s",
                            _img_exc,
                        )
                        _run_message = message
                else:
                    _run_message = message

                _api_run_message = _wrap_current_message_with_observed_context(
                    _run_message,
                    observed_group_context,
                )
                _conversation_kwargs = {
                    "conversation_history": agent_history,
                    "task_id": session_id,
                }
                if _persist_user_message_override is not None:
                    _conversation_kwargs["persist_user_message"] = _persist_user_message_override
                elif observed_group_context:
                    _conversation_kwargs["persist_user_message"] = message
                if moa_config is not None:
                    _conversation_kwargs["moa_config"] = moa_config
                if _persist_user_timestamp_override is not None:
                    _conversation_kwargs["persist_user_timestamp"] = _persist_user_timestamp_override
                result = agent.run_conversation(_api_run_message, **_conversation_kwargs)
            finally:
                unregister_gateway_notify(_approval_session_key)
                # Cancel any pending clarify entries so blocked agent
                # threads don't hang past the end of the run (interrupt,
                # completion, gateway shutdown).  Idempotent.
                try:
                    from tools.clarify_gateway import clear_session as _clear_clarify_session
                    _clear_clarify_session(_approval_session_key)
                except Exception:
                    pass
                reset_current_session_key(_approval_session_token)
            result_holder[0] = result

            # Signal the stream consumer that the agent is done
            if _stream_consumer is not None:
                _stream_consumer.finish()
            
            # Return final response, or a message if something went wrong
            final_response = result.get("final_response")

            # Extract actual token counts from the agent instance used for this run
            _last_prompt_toks = 0
            _input_toks = 0
            _output_toks = 0
            _context_length = 0
            _agent = agent_holder[0]
            if _agent and hasattr(_agent, "context_compressor"):
                _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
                _input_toks = getattr(_agent, "session_prompt_tokens", 0)
                _output_toks = getattr(_agent, "session_completion_tokens", 0)
                _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
            _resolved_model = getattr(_agent, "model", None) if _agent else None

            # Sync session_id immediately after run_conversation(). Compression
            # can rotate before a follow-up model call fails; the failure return
            # below must still point the gateway at the compressed child.
            agent = agent_holder[0]
            _session_was_split = False
            # In-place compaction (compression.in_place / #38763) compacts the
            # transcript WITHOUT rotating the id, so the id-change diff below
            # can't detect it. compress_context() sets this rotation-independent
            # flag on the agent; the gateway uses it to re-baseline transcript
            # handling (history_offset=0 + rewrite the JSONL transcript) the
            # same way a split would, even though the session_id is unchanged.
            _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
            agent_session_id = getattr(agent, 'session_id', session_id) if agent else session_id
            if agent and session_key and agent_session_id != session_id:
                _session_was_split = True
                logger.info(
                    "Session split detected: %s → %s (compression)",
                    session_id, agent_session_id,
                )
                entry = self.session_store._entries.get(session_key)
                if entry:
                    entry.session_id = agent_session_id
                    self.session_store._save()

                # If this is a Telegram DM and source.thread_id was lost during
                # the session split (synthetic / recovered event), restore it
                # from the binding so _thread_metadata_for_source produces the
                # correct message_thread_id instead of routing to the General
                # thread.  Failure here is non-fatal — we log and continue;
                # worst case the message lands in General, which is the
                # pre-fix behaviour.
                if (
                    getattr(source, "platform", None) == Platform.TELEGRAM
                    and getattr(source, "chat_type", None) == "dm"
                    and getattr(source, "thread_id", None) is None
                    and self._session_db is not None
                ):
                    try:
                        # run_sync is off-loop (executor); sync DB is fine.
                        _binding = self._session_db._db.get_telegram_topic_binding_by_session(
                            session_id=agent_session_id,
                        )
                        if _binding and _binding.get("thread_id"):
                            source.thread_id = str(_binding["thread_id"])
                            logger.debug(
                                "Restored source.thread_id=%s from binding after session split %s → %s",
                                source.thread_id,
                                session_id,
                                agent_session_id,
                            )
                    except Exception:
                        logger.debug(
                            "Failed to restore thread_id from binding after session split",
                            exc_info=True,
                        )
                if entry:
                    self._sync_telegram_topic_binding(
                        source, entry, reason="agent-run-compression",
                    )

            effective_session_id = agent_session_id
            self._sync_session_model_from_agent(effective_session_id, agent)
            # history_offset=0 whenever the agent's message list no longer has
            # the original history prefix — i.e. on rotation (split) OR in-place
            # compaction. In both cases the returned `messages` is the compacted
            # set, so the gateway must persist all of it (offset 0), not slice
            # past the pre-compaction length (which would drop everything).
            _effective_history_offset = (
                0 if (_session_was_split or _compacted_in_place) else len(agent_history)
            )

            if not final_response:
                final_response = _normalize_empty_agent_response(
                    result, final_response or "", history_len=len(agent_history),
                )
                final_response = _sanitize_gateway_final_response(source.platform, final_response)
                if not final_response:
                    final_response = f"⚠️ {result['error']}" if result.get("error") else ""
                return {
                    "final_response": final_response,
                    "messages": result.get("messages", []),
                    "api_calls": result.get("api_calls", 0),
                    "failed": result.get("failed", False),
                    "partial": result.get("partial", False),
                    "completed": result.get("completed"),
                    "interrupted": result.get("interrupted", False),
                    "interrupt_message": result.get("interrupt_message"),
                    "error": result.get("error"),
                    "compression_exhausted": result.get("compression_exhausted", False),
                    "tools": tools_holder[0] or [],
                    "history_offset": _effective_history_offset,
                    "compacted_in_place": _compacted_in_place,
                    "session_id": effective_session_id,
                    "last_prompt_tokens": _last_prompt_toks,
                    "input_tokens": _input_toks,
                    "output_tokens": _output_toks,
                    "model": _resolved_model,
                    "context_length": _context_length,
                }
            
            # Scan tool results for MEDIA:<path> tags that need to be delivered
            # as native audio/file attachments.  The TTS tool embeds MEDIA: tags
            # in its JSON response, but the model's final text reply usually
            # doesn't include them.  We collect unique tags from tool results and
            # append any that aren't already present in the final response, so the
            # adapter's extract_media() can find and deliver the files exactly once.
            #
            # Scope the scan to THIS turn's tool results only. ``agent_history``
            # was passed into run_conversation as ``conversation_history``, so the
            # agent's returned ``messages`` list is ``agent_history`` followed by
            # the messages produced this turn. Slicing at ``len(agent_history)``
            # isolates the current turn precisely, so a stale MEDIA: path emitted
            # by a tool several turns earlier (still present in the full message
            # list) can never leak onto a later text-only reply. (Fixes #34608)
            #
            # Path-based deduplication against _history_media_paths (collected
            # before run_conversation) is retained as a secondary guard. It is
            # also the sole guard on the fallback branch taken when mid-run
            # context compression shrinks the message list below the original
            # history length, preserving the compression-safe behaviour of #160.
            if "MEDIA:" not in final_response:
                media_tags, has_voice_directive = _collect_auto_append_media_tags(
                    result.get("messages", []),
                    history_offset=len(agent_history),
                    history_media_paths=_history_media_paths,
                )

                if media_tags:
                    seen = set()
                    unique_tags = []
                    for tag in media_tags:
                        if tag not in seen:
                            seen.add(tag)
                            unique_tags.append(tag)
                    if has_voice_directive:
                        unique_tags.insert(0, "[[audio_as_voice]]")
                    final_response = final_response + "\n" + "\n".join(unique_tags)
            
            # Auto-generate session title after first exchange (non-blocking)
            if final_response and self._session_db:
                try:
                    from agent.title_generator import maybe_auto_title
                    all_msgs = result_holder[0].get("messages", []) if result_holder[0] else []
                    # In Gateway mode, auto-title failures must NOT be
                    # surfaced as user-visible messages (fixes #23246).
                    # Log them at debug level only — they are not actionable
                    # to the end user. CLI mode keeps the existing behaviour
                    # via the agent's _emit_auxiliary_failure path.
                    def _title_failure_cb(task: str, exc: BaseException) -> None:
                        logger.debug(
                            "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                            task, exc,
                        )
                    maybe_auto_title_kwargs = {
                        "failure_callback": _title_failure_cb,
                        "main_runtime": {
                            "model": getattr(agent, "model", None),
                            "provider": getattr(agent, "provider", None),
                            "base_url": getattr(agent, "base_url", None),
                            "api_key": getattr(agent, "api_key", None),
                            "api_mode": getattr(agent, "api_mode", None),
                        } if agent else None,
                    }
                    if self._is_telegram_topic_lane(source):
                        maybe_auto_title_kwargs["title_callback"] = lambda title: self._schedule_telegram_topic_title_rename(
                            source,
                            effective_session_id,
                            title,
                        )
                    maybe_auto_title(
                        getattr(self._session_db, "_db", self._session_db),
                        effective_session_id,
                        message,
                        final_response,
                        all_msgs,
                        **maybe_auto_title_kwargs,
                    )
                except Exception:
                    pass

            return {
                "final_response": final_response,
                "last_reasoning": result.get("last_reasoning"),
                "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                "api_calls": result_holder[0].get("api_calls", 0) if result_holder[0] else 0,
                "completed": result_holder[0].get("completed") if result_holder[0] else None,
                "interrupted": result_holder[0].get("interrupted", False) if result_holder[0] else False,
                "partial": result_holder[0].get("partial", False) if result_holder[0] else False,
                "error": result_holder[0].get("error") if result_holder[0] else None,
                "interrupt_message": result_holder[0].get("interrupt_message") if result_holder[0] else None,
                "tools": tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "compacted_in_place": _compacted_in_place,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
                "session_id": effective_session_id,
                "response_previewed": result.get("response_previewed", False),
                "response_transformed": result.get("response_transformed", False),
                # Pass through the agent_persisted flag so the persistence block
                # above can correctly determine whether the codex app-server path
                # self-persisted (it didn't — see codex_runtime.py).  Default
                # True preserves the skip-db behaviour for the standard runtime.
                "agent_persisted": (result_holder[0].get("agent_persisted", True) if result_holder[0] else True),
            }
        
        # Start progress message sender if enabled. Gate on needs_progress_queue
        # (tool_progress OR thinking_progress), not tool_progress alone: the
        # sender drains BOTH tool-progress lines and _thinking scratch bubbles.
        # With the old tool_progress-only gate, a thinking_progress:true /
        # tool_progress:off user had the callback queue _thinking messages that
        # no task ever drained — so they silently never appeared.
        progress_task = None
        if needs_progress_queue:
            progress_task = asyncio.create_task(send_progress_messages())

        # Start stream consumer task — polls for consumer creation since it
        # happens inside run_sync (thread pool) after the agent is constructed.
        stream_task = None

        async def _start_stream_consumer():
            """Wait for the stream consumer to be created, then run it."""
            for _ in range(200):  # Up to 10s wait
                if stream_consumer_holder[0] is not None:
                    await stream_consumer_holder[0].run()
                    return
                await asyncio.sleep(0.05)

        stream_task = asyncio.create_task(_start_stream_consumer())
        
        # Track this agent as running for this session (for interrupt support)
        # We do this in a callback after the agent is created
        async def track_agent():
            # Wait for agent to be created
            while agent_holder[0] is None:
                await asyncio.sleep(0.05)
            if not session_key:
                return
            # Only promote the sentinel to the real agent if this run is still
            # current.  If /stop or /new bumped the generation while we were
            # spinning up, leave the newer run's slot alone — we'll be
            # discarded by the stale-result check in _handle_message_with_agent.
            if run_generation is not None and not self._is_session_run_current(
                session_key, run_generation
            ):
                logger.info(
                    "Skipping stale agent promotion for %s — generation %s is no longer current",
                    session_key or "",
                    run_generation,
                )
                return
            self._running_agents[session_key] = agent_holder[0]
            if self._draining:
                self._update_runtime_status("draining")
        
        tracking_task = asyncio.create_task(track_agent())
        
        # Monitor for interrupts from the adapter (new messages arriving).
        # This is the PRIMARY interrupt path for regular text messages —
        # Level 1 (base.py) catches them before _handle_message() is reached,
        # so the Level 2 running_agent.interrupt() path never fires.
        # The inactivity poll loop below has a BACKUP check in case this
        # task dies (no error handling = silent death = lost interrupts).
        _interrupt_detected = asyncio.Event()  # shared with backup check

        async def monitor_for_interrupt():
            if not session_key:
                return

            while True:
                await asyncio.sleep(0.2)  # Check every 200ms
                try:
                    # Re-resolve adapter each iteration so reconnects don't
                    # leave us holding a stale reference.
                    _adapter = self.adapters.get(source.platform)
                    if not _adapter:
                        continue
                    # Check if adapter has a pending interrupt for this session.
                    # Must use session_key (build_session_key output) — NOT
                    # source.chat_id — because the adapter stores interrupt events
                    # under the full session key.
                    if hasattr(_adapter, 'has_pending_interrupt') and _adapter.has_pending_interrupt(session_key):
                        agent = agent_holder[0]
                        if agent:
                            # Peek at the pending message text WITHOUT consuming it.
                            # The message must remain in _pending_messages so the
                            # post-run dequeue at _dequeue_pending_event() can
                            # retrieve the full MessageEvent (with media metadata).
                            # If we pop here, a race exists: the agent may finish
                            # before checking _interrupt_requested, and the message
                            # is lost — neither the interrupt path nor the dequeue
                            # path finds it.
                            _peek_event = _adapter._pending_messages.get(session_key)
                            pending_text = None
                            if _peek_event is not None:
                                pending_text = _peek_event.text or ""
                                # Transcribe audio media BEFORE signaling the
                                # agent, so voice messages interrupt with the
                                # real transcript instead of an empty string
                                # (or file-path placeholder). Matches the UX
                                # of fresh voice messages including the
                                # 🎙️ echo back to the user.
                                _media_urls = getattr(_peek_event, "media_urls", None) or []
                                _media_types = getattr(_peek_event, "media_types", None) or []
                                _audio_paths = []
                                for _i, _path in enumerate(_media_urls):
                                    _mtype = _media_types[_i] if _i < len(_media_types) else ""
                                    _is_audio = (
                                        _mtype.startswith("audio/")
                                        or getattr(_peek_event, "message_type", None) in (MessageType.VOICE, MessageType.AUDIO)
                                    )
                                    if _is_audio:
                                        _audio_paths.append(_path)
                                if _audio_paths:
                                    try:
                                        _enriched, _transcripts = await self._enrich_message_with_transcription(
                                            pending_text, _audio_paths,
                                        )
                                        pending_text = _enriched
                                        if _transcripts:
                                            _echo_meta = {"thread_id": source.thread_id} if source.thread_id else None
                                            for _tx in _transcripts:
                                                try:
                                                    await _adapter.send(
                                                        source.chat_id,
                                                        f'🎙️ "{_tx}"',
                                                        metadata=_echo_meta,
                                                    )
                                                except Exception as _echo_exc:
                                                    logger.debug(
                                                        "Voice-interrupt echo failed (non-fatal): %s",
                                                        _echo_exc,
                                                    )
                                    except Exception as _trans_exc:
                                        logger.warning(
                                            "Voice-interrupt transcription failed: %s", _trans_exc,
                                        )
                                elif not pending_text and _media_urls:
                                    pending_text = _build_media_placeholder(_peek_event)
                            logger.debug("Interrupt detected from adapter, signaling agent...")
                            agent.interrupt(pending_text)
                            _interrupt_detected.set()
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as _mon_err:
                    logger.debug("monitor_for_interrupt error (will retry): %s", _mon_err)
        
        interrupt_monitor = asyncio.create_task(monitor_for_interrupt())

        # Periodic "still working" notifications for long-running tasks.
        # Fires every N seconds so the user knows the agent hasn't died.
        # Config: agent.gateway_notify_interval in config.yaml, or
        # HERMES_AGENT_NOTIFY_INTERVAL env var.  Default 180s (3 min).
        # 0 = disable notifications.
        _NOTIFY_INTERVAL_RAW = _float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180)
        _NOTIFY_INTERVAL = _NOTIFY_INTERVAL_RAW if _NOTIFY_INTERVAL_RAW > 0 else None
        if not bool(
            resolve_display_setting(
                user_config,
                platform_key,
                "long_running_notifications",
                True,
            )
        ):
            _NOTIFY_INTERVAL = None
        _notify_start = time.time()

        async def _notify_long_running():
            if _NOTIFY_INTERVAL is None:
                return  # Notifications disabled (gateway_notify_interval: 0)
            _notify_adapter = self.adapters.get(source.platform)
            if not _notify_adapter:
                return
            # Track the heartbeat message id so we can edit-in-place on
            # platforms that support it (Telegram, Discord, Slack, etc.)
            # instead of spamming a new "Still working" bubble every
            # interval. Falls back to send-new when edit fails or isn't
            # supported by the adapter.
            _heartbeat_msg_id: Optional[str] = None
            while True:
                await asyncio.sleep(_NOTIFY_INTERVAL)
                # Stop heartbeating once this run no longer owns the session
                # slot or the executor has finished — otherwise a stale
                # "running: delegate_task" bubble can outlive the run that
                # spawned it (#12029). _executor_task is a closure var bound
                # just after this task is scheduled; tolerate the brief window
                # before then (the first wake is _NOTIFY_INTERVAL away anyway).
                try:
                    _exec_ref = _executor_task
                except NameError:
                    _exec_ref = None
                if not self._should_emit_long_running_notification(
                    session_key, agent_holder[0], _exec_ref
                ):
                    break
                _elapsed_mins = int((time.time() - _notify_start) // 60)
                # Include agent activity context if available. Default
                # heartbeat is terse: elapsed + current tool. Verbose
                # iteration counter is gated on busy_ack_detail so users
                # who want it can opt in per platform.
                _agent_ref = agent_holder[0]
                _status_detail = ""
                _want_iteration_detail = bool(
                    resolve_display_setting(
                        user_config,
                        platform_key,
                        "busy_ack_detail",
                        True,
                    )
                )
                if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                    try:
                        _a = _agent_ref.get_activity_summary()
                        _parts = []
                        if _want_iteration_detail:
                            _parts.append(
                                f"iteration {_a['api_call_count']}/{_a['max_iterations']}"
                            )
                        _action = _a.get("current_tool") or _a.get("last_activity_desc")
                        if _action:
                            _parts.append(str(_action))
                        if _parts:
                            _status_detail = " — " + ", ".join(_parts)
                    except Exception:
                        pass
                _heartbeat_text = f"⏳ Working — {_elapsed_mins} min{_status_detail}"
                try:
                    _notify_res = None
                    if _heartbeat_msg_id:
                        try:
                            _notify_res = await _notify_adapter.edit_message(
                                source.chat_id,
                                _heartbeat_msg_id,
                                _heartbeat_text,
                            )
                        except Exception as _ee:
                            logger.debug("Heartbeat edit failed: %s", _ee)
                            _notify_res = None
                    if not (_notify_res and getattr(_notify_res, "success", False)):
                        _notify_res = await _notify_adapter.send(
                            source.chat_id,
                            _heartbeat_text,
                            metadata=_non_conversational_metadata(_status_thread_metadata, platform=source.platform),
                        )
                        if getattr(_notify_res, "success", False) and getattr(
                            _notify_res, "message_id", None
                        ):
                            _heartbeat_msg_id = str(_notify_res.message_id)
                            if _cleanup_progress:
                                _cleanup_msg_ids.append(_heartbeat_msg_id)
                except Exception as _ne:
                    logger.debug("Long-running notification error: %s", _ne)

        _notify_task = asyncio.create_task(_notify_long_running())

        def _stream_confirmed_final_delivery(
            consumer,
            final_text: str,
            *,
            previewed: bool = False,
        ) -> bool:
            """Return True only when the actual final reply reached the user."""
            if consumer is None:
                return False
            if getattr(consumer, "final_response_sent", False):
                return True
            if previewed:
                has_delivered_text = getattr(consumer, "has_delivered_text", None)
                if callable(has_delivered_text):
                    try:
                        return bool(has_delivered_text(final_text))
                    except Exception:
                        return False
            return False

        try:
            # Run in thread pool to not block.  Use an *inactivity*-based
            # timeout instead of a wall-clock limit: the agent can run for
            # hours if it's actively calling tools / receiving stream tokens,
            # but a hung API call or stuck tool with no activity for the
            # configured duration is caught and killed.  (#4815)
            #
            # Config: agent.gateway_timeout in config.yaml, or
            # HERMES_AGENT_TIMEOUT env var (env var takes precedence).
            # Default 1800s (30 min inactivity).  0 = unlimited.
            _agent_timeout_raw = _float_env("HERMES_AGENT_TIMEOUT", 1800)
            _agent_timeout = _agent_timeout_raw if _agent_timeout_raw > 0 else None
            _agent_warning_raw = _float_env("HERMES_AGENT_TIMEOUT_WARNING", 900)
            _agent_warning = _agent_warning_raw if _agent_warning_raw > 0 else None
            _warning_fired = False
            _executor_task = asyncio.ensure_future(
                self._run_in_executor_with_context(run_sync)
            )

            _inactivity_timeout = False
            _POLL_INTERVAL = 5.0

            if _agent_timeout is None:
                # Unlimited — still poll periodically for backup interrupt
                # detection in case monitor_for_interrupt() silently died.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Backup interrupt check: if the monitor task died or
                    # missed the interrupt, catch it here.
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self.adapters.get(source.platform)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()
            else:
                # Poll loop: check the agent's built-in activity tracker
                # (updated by _touch_activity() on every tool call, API
                # call, and stream delta) every few seconds.
                response = None
                while True:
                    done, _ = await asyncio.wait(
                        {_executor_task}, timeout=_POLL_INTERVAL
                    )
                    if done:
                        response = _executor_task.result()
                        break
                    # Agent still running — check inactivity.
                    _agent_ref = agent_holder[0]
                    _idle_secs = 0.0
                    if _agent_ref and hasattr(_agent_ref, "get_activity_summary"):
                        try:
                            _act = _agent_ref.get_activity_summary()
                            _idle_secs = _act.get("seconds_since_activity", 0.0)
                        except Exception:
                            pass
                    # Staged warning: fire once before escalating to full timeout.
                    if (not _warning_fired and _agent_warning is not None
                            and _idle_secs >= _agent_warning):
                        _warning_fired = True
                        _warn_adapter = self.adapters.get(source.platform)
                        if _warn_adapter:
                            _elapsed_warn = int(_agent_warning // 60) or 1
                            _remaining_mins = int((_agent_timeout - _agent_warning) // 60) or 1
                            try:
                                await _warn_adapter.send(
                                    source.chat_id,
                                    f"⚠️ No activity for {_elapsed_warn} min. "
                                    f"If the agent does not respond soon, it will "
                                    f"be timed out in {_remaining_mins} min. "
                                    f"You can continue waiting or use /reset.",
                                    metadata=_status_thread_metadata,
                                )
                            except Exception as _warn_err:
                                logger.debug("Inactivity warning send error: %s", _warn_err)
                    if _idle_secs >= _agent_timeout:
                        _inactivity_timeout = True
                        break
                    # Backup interrupt check (same as unlimited path).
                    if not _interrupt_detected.is_set() and session_key:
                        _backup_adapter = self.adapters.get(source.platform)
                        _backup_agent = agent_holder[0]
                        if (_backup_adapter and _backup_agent
                                and hasattr(_backup_adapter, 'has_pending_interrupt')
                                and _backup_adapter.has_pending_interrupt(session_key)):
                            _bp_event = _backup_adapter._pending_messages.get(session_key)
                            _bp_text = _bp_event.text if _bp_event else None
                            logger.info(
                                "Backup interrupt detected for session %s "
                                "(monitor task state: %s)",
                                session_key,
                                "done" if interrupt_monitor.done() else "running",
                            )
                            _backup_agent.interrupt(_bp_text)
                            _interrupt_detected.set()

            if _inactivity_timeout:
                # Build a diagnostic summary from the agent's activity tracker.
                _timed_out_agent = agent_holder[0]
                _activity = {}
                if _timed_out_agent and hasattr(_timed_out_agent, "get_activity_summary"):
                    try:
                        _activity = _timed_out_agent.get_activity_summary()
                    except Exception:
                        pass

                _last_desc = _activity.get("last_activity_desc", "unknown")
                _secs_ago = _activity.get("seconds_since_activity", 0)
                _cur_tool = _activity.get("current_tool")
                _iter_n = _activity.get("api_call_count", 0)
                _iter_max = _activity.get("max_iterations", 0)

                logger.error(
                    "Agent idle for %.0fs (timeout %.0fs) in session %s "
                    "| last_activity=%s | iteration=%s/%s | tool=%s",
                    _secs_ago, _agent_timeout, session_key,
                    _last_desc, _iter_n, _iter_max,
                    _cur_tool or "none",
                )

                # Interrupt the agent if it's still running so the thread
                # pool worker is freed.
                if _timed_out_agent and hasattr(_timed_out_agent, "interrupt"):
                    _timed_out_agent.interrupt(_INTERRUPT_REASON_TIMEOUT)

                _timeout_mins = int(_agent_timeout // 60) or 1

                # Construct a user-facing message with diagnostic context.
                _diag_lines = [
                    f"⏱️ Agent inactive for {_timeout_mins} min — no tool calls "
                    f"or API responses."
                ]
                if _cur_tool:
                    _diag_lines.append(
                        f"The agent appears stuck on tool `{_cur_tool}` "
                        f"({_secs_ago:.0f}s since last activity, "
                        f"iteration {_iter_n}/{_iter_max})."
                    )
                else:
                    _diag_lines.append(
                        f"Last activity: {_last_desc} ({_secs_ago:.0f}s ago, "
                        f"iteration {_iter_n}/{_iter_max}). "
                        "The agent may have been waiting on an API response."
                    )
                _diag_lines.append(
                    "To increase the limit, set agent.gateway_timeout in config.yaml "
                    "(value in seconds, 0 = no limit) and restart the gateway.\n"
                    "Try again, or use /reset to start fresh."
                )

                response = {
                    "final_response": "\n".join(_diag_lines),
                    "messages": result_holder[0].get("messages", []) if result_holder[0] else [],
                    "api_calls": _iter_n,
                    "tools": tools_holder[0] or [],
                    "history_offset": 0,
                    "failed": True,
                }

            # Track fallback model state: if the agent switched to a
            # fallback model during this run, persist it so /model shows
            # the actually-active model instead of the config default.
            # Skip eviction when the run failed — evicting a failed agent
            # forces MCP reinit on the next message for no benefit (the
            # same error will recur).  This was the root cause of #7130:
            # a bad model ID triggered fallback → eviction → recreation →
            # MCP reinit → same 400 → loop, burning 91% CPU for hours.
            _agent = agent_holder[0]
            _result_for_fb = result_holder[0]
            _run_failed = _result_for_fb.get("failed") if _result_for_fb else False
            if _agent is not None and hasattr(_agent, 'model') and not _run_failed:
                _cfg_model = _resolve_gateway_model()
                # Normalize _cfg_model the same way AIAgent.__init__ does, so a
                # vendor-prefixed config value (e.g. "deepseek/deepseek-v4-pro")
                # matches the agent's stripped model ("deepseek-v4-pro") on
                # native providers. Without this, _agent.model != _cfg_model is
                # always true for vendor-prefixed config and the cached agent is
                # evicted on every successful turn — destroying prompt caching.
                # Aggregators (openrouter, etc.) keep the vendor/model slug, so
                # they're left untouched.
                try:
                    from hermes_cli.model_normalize import (
                        _AGGREGATOR_PROVIDERS,
                        normalize_model_for_provider,
                    )
                    _agent_provider = getattr(_agent, 'provider', '') or ''
                    if _agent_provider and _agent_provider not in _AGGREGATOR_PROVIDERS:
                        _cfg_model = normalize_model_for_provider(_cfg_model, _agent_provider)
                except Exception:
                    pass
                if _agent.model != _cfg_model and not self._is_intentional_model_switch(session_key, _agent.model):
                    # Fallback activated on a successful run — evict cached
                    # agent so the next message retries the primary model.
                    self._evict_cached_agent(session_key)

            # Check if we were interrupted OR have a queued message (/queue).
            result = result_holder[0]
            adapter = self.adapters.get(source.platform)
            
            # Get pending message from adapter.
            # Use session_key (not source.chat_id) to match adapter's storage keys.
            pending_event = None
            pending = None
            if result and adapter and session_key:
                pending_event = _dequeue_pending_event(adapter, session_key)
                # /queue overflow: after consuming the adapter's "next-up"
                # slot, promote the next queued event into it so the
                # recursive run's drain will see it.  This keeps the slot
                # occupied for the full FIFO chain, which (a) preserves
                # order, and (b) causes any mid-chain /queue to correctly
                # route to overflow rather than jumping the queue.
                pending_event = self._promote_queued_event(session_key, adapter, pending_event)
                if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                    interrupt_message = result.get("interrupt_message")
                    if _is_control_interrupt_message(interrupt_message):
                        logger.info(
                            "Ignoring control interrupt message for session %s: %s",
                            session_key or "?",
                            interrupt_message,
                        )
                    else:
                        pending = interrupt_message
                elif pending_event:
                    # Transcribe audio media on the dequeued event BEFORE it is
                    # handed back as the next user turn, so queued/interrupting
                    # voice messages drain with the real transcript instead of
                    # a file-path placeholder. Echo each transcript back to the
                    # user (same 🎙️ format as fresh voice messages) so voice
                    # interrupts feel identical to text interrupts.
                    _pending_text = pending_event.text or ""
                    _media_urls = getattr(pending_event, "media_urls", None) or []
                    _media_types = getattr(pending_event, "media_types", None) or []
                    _audio_paths = []
                    for _i, _path in enumerate(_media_urls):
                        _mtype = _media_types[_i] if _i < len(_media_types) else ""
                        _is_audio = (
                            _mtype.startswith("audio/")
                            or getattr(pending_event, "message_type", None) in (MessageType.VOICE, MessageType.AUDIO)
                        )
                        if _is_audio:
                            _audio_paths.append(_path)
                    if _audio_paths:
                        try:
                            _enriched, _transcripts = await self._enrich_message_with_transcription(
                                _pending_text, _audio_paths,
                            )
                            pending = _enriched or None
                            if _transcripts:
                                _echo_meta = {"thread_id": source.thread_id} if source.thread_id else None
                                for _tx in _transcripts:
                                    try:
                                        await adapter.send(
                                            source.chat_id,
                                            f'🎙️ "{_tx}"',
                                            metadata=_echo_meta,
                                        )
                                    except Exception as _echo_exc:
                                        logger.debug(
                                            "Voice-drain echo failed (non-fatal): %s", _echo_exc,
                                        )
                        except Exception as _trans_exc:
                            logger.warning(
                                "Voice-drain transcription failed: %s", _trans_exc,
                            )
                            pending = _pending_text or _build_media_placeholder(pending_event)
                    else:
                        pending = _pending_text or _build_media_placeholder(pending_event)
                    if pending:
                        logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

            # Leftover /steer: if a steer arrived after the last tool batch
            # (e.g. during the final API call), the agent couldn't inject it
            # and returned it in result["pending_steer"]. Deliver it as the
            # next user turn so it isn't silently dropped.
            if result and not pending and not pending_event:
                _leftover_steer = result.get("pending_steer")
                if _leftover_steer:
                    pending = _leftover_steer
                    logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

            # Safety net: if the pending text is a slash command (e.g. "/stop",
            # "/new"), discard it — commands should never be passed to the agent
            # as user input.  The primary fix is in base.py (commands bypass the
            # active-session guard), but this catches edge cases where command
            # text leaks through the interrupt_message fallback.
            if pending and pending.strip().startswith("/"):
                _pending_parts = pending.strip().split(None, 1)
                _pending_cmd_word = _pending_parts[0][1:].lower() if _pending_parts else ""
                if _pending_cmd_word:
                    try:
                        from hermes_cli.commands import resolve_command as _rc_pending
                        if _rc_pending(_pending_cmd_word):
                            logger.info(
                                "Discarding command '/%s' from pending queue — "
                                "commands must not be passed as agent input",
                                _pending_cmd_word,
                            )
                            pending_event = None
                            pending = None
                    except Exception:
                        pass

            if self._draining and (pending_event or pending):
                logger.info(
                    "Discarding pending follow-up for session %s during gateway %s",
                    session_key or "?",
                    self._status_action_label(),
                )
                pending_event = None
                pending = None

            if pending_event or pending:
                logger.debug("Processing pending message: '%s...'", pending[:40])

                # Clear the adapter's interrupt event so the next _run_agent call
                # doesn't immediately re-trigger the interrupt before the new agent
                # even makes its first API call (this was causing an infinite loop).
                if adapter and hasattr(adapter, '_active_sessions') and session_key and session_key in adapter._active_sessions:
                    adapter._active_sessions[session_key].clear()

                # Cap recursion depth to prevent resource exhaustion when the
                # user sends multiple messages while the agent keeps failing. (#816)
                if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                    logger.warning(
                        "Interrupt recursion depth %d reached for session %s — "
                        "queueing message instead of recursing.",
                        _interrupt_depth, session_key,
                    )
                    adapter = self.adapters.get(source.platform)
                    if adapter and pending_event:
                        merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
                    elif adapter and hasattr(adapter, 'queue_message'):
                        adapter.queue_message(session_key, pending)
                    return result_holder[0] or {"final_response": response, "messages": history}

                was_interrupted = result.get("interrupted")
                if not was_interrupted:
                    # Queued message after normal completion — deliver the first
                    # response before processing the queued follow-up.
                    # Skip if streaming already delivered it.
                    _sc = stream_consumer_holder[0]
                    if _sc and stream_task:
                        try:
                            await asyncio.wait_for(stream_task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            stream_task.cancel()
                            try:
                                await stream_task
                            except asyncio.CancelledError:
                                pass
                        except Exception as e:
                            logger.debug("Stream consumer wait before queued message failed: %s", e)
                    _previewed = bool(result.get("response_previewed"))
                    first_response = result.get("final_response", "")
                    _already_streamed = _stream_confirmed_final_delivery(
                        _sc,
                        first_response,
                        previewed=_previewed,
                    )
                    if first_response and not _already_streamed:
                        try:
                            logger.info(
                                "Queued follow-up for session %s: final stream delivery not confirmed; sending first response before continuing.",
                                session_key or "?",
                            )
                            await adapter.send(
                                source.chat_id,
                                first_response,
                                metadata=_status_thread_metadata,
                            )
                        except Exception as e:
                            logger.warning("Failed to send first response before queued message: %s", e)
                    elif first_response:
                        logger.info(
                            "Queued follow-up for session %s: skipping resend because final streamed delivery was confirmed.",
                            session_key or "?",
                        )
                    # Release deferred bg-review notifications now that the
                    # first response has been delivered.  Pop from the
                    # adapter's callback dict (prevents double-fire in
                    # base.py's finally block) and call it.
                    if getattr(type(adapter), "pop_post_delivery_callback", None) is not None:
                        _bg_cb = adapter.pop_post_delivery_callback(
                            session_key,
                            generation=run_generation,
                        )
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                    elif adapter and hasattr(adapter, "_post_delivery_callbacks"):
                        _bg_cb = adapter._post_delivery_callbacks.pop(session_key, None)
                        if callable(_bg_cb):
                            try:
                                _bg_result = _bg_cb()
                                if inspect.isawaitable(_bg_result):
                                    await _bg_result
                            except Exception:
                                pass
                # else: interrupted — discard the interrupted response ("Operation
                # interrupted." is just noise; the user already knows they sent a
                # new message).

                updated_history = result.get("messages", history)
                next_source = source
                next_message = pending
                next_message_id = None
                next_channel_prompt = None
                if pending_event is not None:
                    next_source = getattr(pending_event, "source", None) or source
                    if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                        logger.info(
                            "Discarding stale goal continuation for session %s — goal is no longer active",
                            session_key or "?",
                        )
                        return result
                    next_message = await self._prepare_inbound_message_text(
                        event=pending_event,
                        source=next_source,
                        history=updated_history,
                    )
                    if next_message is None:
                        return result
                    next_message_id = self._reply_anchor_for_event(pending_event)
                    next_channel_prompt = getattr(pending_event, "channel_prompt", None)

                # Restart typing indicator so the user sees activity while
                # the follow-up turn runs.  The outer _process_message_background
                # typing task is still alive but may be stale.
                _followup_adapter = self.adapters.get(source.platform)
                if _followup_adapter:
                    try:
                        await _followup_adapter.send_typing(
                            source.chat_id,
                            metadata=_status_thread_metadata,
                        )
                    except Exception:
                        pass

                # Re-baseline the cached agent's message_count snapshot before
                # recursing into the in-band queued (/queue) follow-up turn.
                # The first turn has completed and flushed its own user +
                # assistant rows to the SessionDB, so the cross-process
                # coherence guard (#45966) — which this recursive _run_agent
                # call re-enters — would otherwise see the grown on-disk count
                # against the stale build-time snapshot and rebuild the agent
                # on THIS process's OWN writes, destroying the prompt-cache
                # prefix #46237 was merged to preserve.  The existing
                # re-baseline in _handle_message_with_agent only runs after the
                # whole _run_agent chain unwinds — too late for the in-band
                # follow-up.  Use the same (session_key, session_id) the
                # recursive call runs under so the snapshot matches exactly
                # what the follow-up's guard will consult.  Fail-safe in helper.
                await self._refresh_agent_cache_message_count(session_key, session_id)

                followup_result = await self._run_agent(
                    message=next_message,
                    context_prompt=context_prompt,
                    history=updated_history,
                    source=next_source,
                    session_id=session_id,
                    session_key=session_key,
                    run_generation=run_generation,
                    _interrupt_depth=_interrupt_depth + 1,
                    event_message_id=next_message_id,
                    channel_prompt=next_channel_prompt,
                )
                return _preserve_queued_followup_history_offset(result, followup_result)
        finally:
            # Stop progress sender, interrupt monitor, and notification task
            if progress_task:
                progress_task.cancel()
            interrupt_monitor.cancel()
            _notify_task.cancel()

            # Wait for stream consumer to finish its final edit
            if stream_task:
                # If the agent never created a stream consumer (e.g. non-
                # streaming code path, or a test stub returning synchronously)
                # there is nothing to flush — cancel immediately instead of
                # waiting out the 5s timeout on a task that's just polling for
                # a consumer that will never arrive.  This was a 5-second
                # cost per non-streaming test run.
                _has_stream_consumer = (
                    stream_consumer_holder
                    and stream_consumer_holder[0] is not None
                )
                if not _has_stream_consumer:
                    stream_task.cancel()
                    try:
                        await stream_task
                    except asyncio.CancelledError:
                        pass
                else:
                    try:
                        await asyncio.wait_for(stream_task, timeout=5.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        stream_task.cancel()
                        try:
                            await stream_task
                        except asyncio.CancelledError:
                            pass
            
            # Clean up tracking
            tracking_task.cancel()
            if session_key:
                # Only release the slot if this run's generation still owns
                # it.  A /stop or /new that bumped the generation while we
                # were unwinding has already installed its own state; this
                # guard prevents an old run from clobbering it on the way
                # out.
                self._release_running_agent_state(
                    session_key, run_generation=run_generation
                )
            if self._draining:
                self._update_runtime_status("draining")
            
            # Wait for cancelled tasks
            for task in [progress_task, interrupt_monitor, tracking_task, _notify_task]:
                if task:
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # If streaming already delivered the response, mark it so the
        # caller's send() is skipped (avoiding duplicate messages).
        # BUT: never suppress delivery when the agent failed — the error
        # message is new content the user hasn't seen, and it must reach
        # them even if streaming had sent earlier partial output.
        #
        # Also never suppress when the final response is "(empty)" — this
        # means the model failed to produce content after tool calls (common
        # with mimo-v2-pro, GLM-5, etc.).  The stream consumer may have
        # sent intermediate text ("Let me search for that…") alongside the
        # tool call, setting already_sent=True, but that text is NOT the
        # final answer.  Suppressing delivery here leaves the user staring
        # at silence.  (#10xxx — "agent stops after web search")
        _sc = stream_consumer_holder[0]
        if isinstance(response, dict) and not response.get("failed"):
            _final = response.get("final_response") or ""
            _is_empty_sentinel = not _final or _final == "(empty)"
            # response_previewed means the interim_assistant_callback already
            # saw the final text, but only suppress the normal send if that
            # exact final text was delivered. Unrelated commentary/progress
            # must not be mistaken for the final response (#14238).
            _previewed = bool(response.get("response_previewed"))
            _content_delivered = bool(
                _sc and getattr(_sc, "final_content_delivered", False)
            )
            # Plugin hooks (e.g. transform_llm_output) may have appended content
            # after streaming finished — when the response was transformed, always
            # send the final version so the appended content reaches the client.
            _transformed = bool(response.get("response_transformed"))
            # Only suppress the normal send when the actual final reply reached
            # the user: the stream consumer streamed it (final_response_sent /
            # final_content_delivered), or the interim preview delivered that
            # *exact* final text. Unrelated commentary/progress shown during a
            # compression/session split must not be mistaken for the final
            # response (#14238).
            _streamed = _stream_confirmed_final_delivery(
                _sc,
                _final,
                previewed=_previewed,
            )
            if not _is_empty_sentinel and not _transformed and (_streamed or _content_delivered):
                logger.info(
                    "Suppressing normal final send for session %s: final delivery already confirmed (streamed=%s previewed=%s content_delivered=%s).",
                    session_key or "?",
                    _streamed,
                    _previewed,
                    _content_delivered,
                )
                response["already_sent"] = True
            elif not _is_empty_sentinel and _transformed and _sc is not None:
                # Plugin hooks transformed the response after streaming — edit the
                # existing streamed message instead of sending a duplicate.
                _sc_msg_id = _sc.message_id
                if _sc_msg_id:
                    try:
                        await _sc.adapter.edit_message(
                            chat_id=source.chat_id,
                            message_id=_sc_msg_id,
                            content=response["final_response"],
                            finalize=True,
                        )
                        response["already_sent"] = True
                        logger.info(
                            "Edited streamed message %s for session %s to include plugin-transformed content.",
                            _sc_msg_id, session_key or "?",
                        )
                    except Exception as _edit_err:
                        logger.warning(
                            "Failed to edit streamed message for session %s: %s",
                            session_key or "?", _edit_err,
                        )

        # Schedule deletion of tracked temporary progress bubbles after the
        # final response lands. Failed runs skip this so bubbles remain as
        # breadcrumbs for the user to see what work happened. Only fires on
        # adapters that support ``delete_message`` (see init above); failures
        # are swallowed — deletion is best-effort.
        if (
            _cleanup_progress
            and _cleanup_adapter is not None
            and _cleanup_msg_ids
            and session_key
            and isinstance(response, dict)
            and not response.get("failed")
            and hasattr(_cleanup_adapter, "register_post_delivery_callback")
        ):
            _ids_snapshot = list(_cleanup_msg_ids)
            _chat_id_snapshot = source.chat_id
            _adapter_snapshot = _cleanup_adapter
            _loop_snapshot = asyncio.get_running_loop()

            def _cleanup_temp_bubbles() -> None:
                async def _delete_all() -> None:
                    for _mid in _ids_snapshot:
                        try:
                            await _adapter_snapshot.delete_message(
                                _chat_id_snapshot, _mid
                            )
                        except Exception:
                            pass
                try:
                    safe_schedule_threadsafe(
                        _delete_all(), _loop_snapshot,
                        logger=logger,
                        log_message="Temp bubble cleanup scheduling error",
                    )
                except Exception:
                    pass

            try:
                _cleanup_adapter.register_post_delivery_callback(
                    session_key,
                    _cleanup_temp_bubbles,
                    generation=run_generation,
                )
            except Exception as _rpe:
                logger.debug("Post-delivery cleanup registration failed: %s", _rpe)

        return response


def _run_planned_stop_watcher(
    stop_event: threading.Event,
    runner,
    loop: asyncio.AbstractEventLoop,
    shutdown_handler,
    *,
    poll_interval: float = 0.5,
) -> None:
    """Poll for the planned-stop marker and trigger graceful shutdown.

    On Windows, ``asyncio.add_signal_handler`` raises NotImplementedError
    for SIGTERM/SIGINT, so the standard signal-driven shutdown path
    never runs when ``hermes gateway stop`` signals the gateway. The
    consequence is that the drain loop is skipped — in-flight agent
    sessions are killed mid-turn and ``resume_pending`` is never set,
    so the next gateway boot has no idea those sessions need to be
    auto-resumed (issue #33778, v0.13.0 session-resume feature broken
    on native Windows).

    This watcher runs on every platform (cheap, defensive) and bridges
    the gap on Windows by translating a filesystem marker into the
    same shutdown-handler invocation a real SIGTERM would have produced
    on POSIX. The CLI's ``hermes_cli.gateway_windows.stop()`` writes
    the marker via ``write_planned_stop_marker(pid)`` and then waits
    for the gateway PID to exit; this watcher is what makes that
    exit happen cleanly.

    On POSIX this is a no-op safety net — the signal handler always
    races us to consuming the marker file because it fires synchronously
    from the kernel's signal delivery.

    Args:
        stop_event: cleared by start_gateway() during normal shutdown
            to tell the watcher to exit.
        runner: the GatewayRunner instance; we check ``_running`` and
            ``_draining`` to avoid triggering shutdown if the gateway
            is already in one of those states.
        loop: the asyncio event loop the shutdown handler must run on.
        shutdown_handler: same callable that's wired to SIGTERM —
            tolerates a ``None`` signal argument (planned stop case)
            and consumes the marker via
            ``consume_planned_stop_marker_for_self()``.
        poll_interval: seconds between marker checks. 0.5s gives a
            responsive shutdown without burning CPU.
    """
    from gateway.status import (
        _get_planned_stop_marker_path,
        planned_stop_marker_targets_self,
    )
    marker_path = _get_planned_stop_marker_path()
    while not stop_event.is_set():
        try:
            if (
                marker_path.exists()
                and not getattr(runner, "_draining", False)
                and getattr(runner, "_running", False)
            ):
                # A marker existing is NOT sufficient — it may have been
                # written for a PREVIOUS gateway instance (different PID)
                # and left behind because that process exited before the
                # CLI's stop() could clean it up. Firing the handler on a
                # stale/foreign marker drives the gateway into shutdown,
                # then consume_planned_stop_marker_for_self() correctly
                # reports a PID mismatch — but by then we're already
                # stopping, so it's logged as an unexpected "UNKNOWN" exit
                # and the watchdog crash-loops the gateway (issue #34597,
                # a regression from PR #33798 which added this watcher
                # without the PID check).
                #
                # Only fire when the marker actually targets us. The probe
                # is non-destructive on a match (the handler does the
                # authoritative consume on the loop thread) and self-heals
                # by unlinking stale/malformed markers so they cannot wedge
                # a freshly booted gateway.
                if not planned_stop_marker_targets_self():
                    stop_event.wait(poll_interval)
                    continue
                # Drive the same path as a real signal handler.
                # Pass signal=None — the handler tolerates that and consumes
                # the marker via consume_planned_stop_marker_for_self,
                # which also validates target_pid + start_time match us.
                loop.call_soon_threadsafe(shutdown_handler, None)
                # Done — the handler will set _draining; we exit on next tick.
                break
        except Exception as _e:
            logger.debug("Planned-stop watcher tick error: %s", _e)
        stop_event.wait(poll_interval)


def _start_gateway_housekeeping(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """Background thread for gateway-only periodic chores (NOT cron).

    Split out of the historical ``_start_cron_ticker`` so the cron *trigger*
    can live behind the ``CronScheduler`` provider (built-in or external) while
    these gateway-specific chores keep running independently of which provider
    fires cron. An external scale-to-zero provider has no 60s loop at all, but
    this housekeeping still wants its hourly cadence — so it owns its own loop.

    Refreshes the channel directory every 5 minutes and prunes the
    image/audio/document cache + expired ``hermes debug share`` pastes once per
    hour, and polls the curator hourly (its inner gate enforces the real
    weekly cadence).
    """
    from gateway.platforms.base import cleanup_image_cache, cleanup_document_cache
    from hermes_cli.debug import _sweep_expired_pastes

    IMAGE_CACHE_EVERY = 60   # ticks — once per hour at default 60s interval
    CHANNEL_DIR_EVERY = 5    # ticks — every 5 minutes
    PASTE_SWEEP_EVERY = 60   # ticks — once per hour
    CURATOR_EVERY = 60       # ticks — poll hourly (inner gate handles the real cadence)

    logger.info("Gateway housekeeping started (interval=%ds)", interval)
    tick_count = 0
    while not stop_event.is_set():
        tick_count += 1

        if tick_count % CHANNEL_DIR_EVERY == 0 and adapters:
            try:
                from gateway.channel_directory import build_channel_directory
                if loop is not None:
                    # build_channel_directory is async (Slack web calls), and
                    # this runs in a background thread. Schedule onto the
                    # gateway event loop and wait briefly for completion so
                    # refresh failures are still logged via the except.
                    fut = safe_schedule_threadsafe(
                        build_channel_directory(adapters), loop,
                        logger=logger,
                        log_message="Channel directory refresh scheduling error",
                    )
                    if fut is not None:
                        fut.result(timeout=30)
            except Exception as e:
                logger.debug("Channel directory refresh error: %s", e)

        if tick_count % IMAGE_CACHE_EVERY == 0:
            try:
                removed = cleanup_image_cache(max_age_hours=24)
                if removed:
                    logger.info("Image cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Image cache cleanup error: %s", e)
            try:
                removed = cleanup_document_cache(max_age_hours=24)
                if removed:
                    logger.info("Document cache cleanup: removed %d stale file(s)", removed)
            except Exception as e:
                logger.debug("Document cache cleanup error: %s", e)

        if tick_count % PASTE_SWEEP_EVERY == 0:
            try:
                deleted, remaining = _sweep_expired_pastes()
                if deleted:
                    logger.info(
                        "Paste sweep: deleted %d expired paste(s), %d pending",
                        deleted, remaining,
                    )
            except Exception as e:
                logger.debug("Paste sweep error: %s", e)

        # Curator — piggy-back on the housekeeping loop so long-running
        # gateways get weekly skill maintenance without needing restarts.
        # maybe_run_curator() is internally gated by config.interval_hours
        # (7 days by default), so CURATOR_EVERY is just the poll rate — the
        # real work only fires once per config interval.
        if tick_count % CURATOR_EVERY == 0:
            try:
                from agent.curator import maybe_run_curator
                maybe_run_curator(
                    idle_for_seconds=float("inf"),
                    on_summary=lambda msg: logger.info("curator: %s", msg),
                )
            except Exception as e:
                logger.debug("Curator tick error: %s", e)

        stop_event.wait(timeout=interval)
    logger.info("Gateway housekeeping stopped")


def _start_cron_ticker(stop_event: threading.Event, adapters=None, loop=None, interval: int = 60):
    """DEPRECATED shim — preserved for backward compatibility.

    The cron trigger now lives behind the ``CronScheduler`` provider
    (``cron.scheduler_provider``); the gateway resolves a provider and runs its
    ``start()`` directly (see ``start_gateway``). This shim runs ONLY the
    built-in in-process tick loop, exactly as before, for any external caller
    or test that still references this symbol (e.g. hermes_cli/debug.py). It no
    longer runs gateway housekeeping — that moved to
    ``_start_gateway_housekeeping``.
    """
    from cron.scheduler_provider import InProcessCronScheduler
    InProcessCronScheduler().start(stop_event, adapters=adapters, loop=loop, interval=interval)


async def start_gateway(config: Optional[GatewayConfig] = None, replace: bool = False, verbosity: Optional[int] = 0) -> bool:
    """
    Start the gateway and run until interrupted.
    
    This is the main entry point for running the gateway.
    Returns True if the gateway ran successfully, False if it failed to start.
    A False return causes a non-zero exit code so systemd can auto-restart.
    
    Args:
        config: Optional gateway configuration override.
        replace: If True, kill any existing gateway instance before starting.
                 Useful for systemd services to avoid restart-loop deadlocks
                 when the previous process hasn't fully exited yet.
    """
    # Snapshot the checkout revision now, while sys.modules still matches disk,
    # so a later `git pull` under this long-lived process can be detected (and
    # risky work like model switching refused) instead of crashing on a stale
    # in-memory module.
    from gateway.code_skew import record_boot_fingerprint
    record_boot_fingerprint()

    # ── Duplicate-instance guard ──────────────────────────────────────
    # Prevent two gateways from running under the same HERMES_HOME.
    # The PID file is scoped to HERMES_HOME, so future multi-profile
    # setups (each profile using a distinct HERMES_HOME) will naturally
    # allow concurrent instances without tripping this guard.
    from gateway.status import (
        acquire_gateway_runtime_lock,
        get_running_pid,
        get_process_start_time,
        release_gateway_runtime_lock,
        remove_pid_file,
        terminate_pid,
    )
    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        if replace:
            existing_start_time = get_process_start_time(existing_pid)
            logger.info(
                "Replacing existing gateway instance (PID %d) with --replace.",
                existing_pid,
            )
            # Record a takeover marker so the target's shutdown handler
            # recognises its SIGTERM as a planned takeover and exits 0
            # (rather than exit 1, which would trigger systemd's
            # Restart=on-failure and start a flap loop against us).
            # Best-effort — proceed even if the write fails.
            try:
                from gateway.status import write_takeover_marker
                write_takeover_marker(existing_pid)
            except Exception as e:
                logger.debug("Could not write takeover marker: %s", e)
            try:
                terminate_pid(existing_pid, force=False)
            except ProcessLookupError:
                pass  # Already gone
            except (PermissionError, OSError):
                logger.error(
                    "Permission denied killing PID %d. Cannot replace.",
                    existing_pid,
                )
                # Marker is scoped to a specific target; clean it up on
                # give-up so it doesn't grief an unrelated future shutdown.
                try:
                    from gateway.status import clear_takeover_marker
                    clear_takeover_marker()
                except Exception:
                    pass
                return False
            # Wait up to 10 seconds for the old process to exit.
            # ``os.kill(pid, 0)`` on Windows is NOT a no-op — use the
            # handle-based existence check instead.
            from gateway.status import _pid_exists
            old_gateway_exited = False
            for _ in range(20):
                if not _pid_exists(existing_pid):
                    old_gateway_exited = True
                    break  # Process is gone
                time.sleep(0.5)
            else:
                # Still alive after 10s — force kill
                logger.warning(
                    "Old gateway (PID %d) did not exit after SIGTERM, sending SIGKILL.",
                    existing_pid,
                )
                try:
                    terminate_pid(existing_pid, force=True)
                except ProcessLookupError:
                    old_gateway_exited = True
                except (PermissionError, OSError):
                    pass
                # Confirm the force-kill actually reaped the process before we
                # clear its PID file / scoped locks. SIGKILL can fail to take
                # (e.g. an uninterruptible-sleep or zombie-reaping parent), and
                # if we blindly clear the metadata and start a fresh instance
                # we end up with two live gateways fighting over the same
                # token — the duplicate-gateway failure in #19471.
                if not old_gateway_exited:
                    for _ in range(20):
                        if not _pid_exists(existing_pid):
                            old_gateway_exited = True
                            break
                        time.sleep(0.25)
                if not old_gateway_exited:
                    logger.error(
                        "Old gateway (PID %d) still appears alive after SIGKILL; "
                        "aborting replacement to avoid a duplicate gateway.",
                        existing_pid,
                    )
                    try:
                        from gateway.status import clear_takeover_marker
                        clear_takeover_marker()
                    except Exception:
                        pass
                    return False
            remove_pid_file()
            # remove_pid_file() is a no-op when the PID doesn't match.
            # Force-unlink to cover the old-process-crashed case.
            try:
                (get_hermes_home() / "gateway.pid").unlink(missing_ok=True)
            except Exception:
                pass
            # Clean up any takeover marker the old process didn't consume
            # (e.g. SIGKILL'd before its shutdown handler could read it).
            try:
                from gateway.status import clear_takeover_marker
                clear_takeover_marker()
            except Exception:
                pass
            # Also release all scoped locks left by the old process.
            # Stopped (Ctrl+Z) processes don't release locks on exit,
            # leaving stale lock files that block the new gateway from starting.
            try:
                from gateway.status import release_all_scoped_locks
                _released = release_all_scoped_locks(
                    owner_pid=existing_pid,
                    owner_start_time=existing_start_time,
                )
                if _released:
                    logger.info("Released %d stale scoped lock(s) from old gateway.", _released)
            except Exception:
                pass
        else:
            hermes_home = str(get_hermes_home())
            logger.error(
                "Another gateway instance is already running (PID %d, HERMES_HOME=%s). "
                "Use 'hermes gateway restart' to replace it, or 'hermes gateway stop' first.",
                existing_pid, hermes_home,
            )
            print(
                f"\n❌ Gateway already running (PID {existing_pid}).\n"
                f"   Use 'hermes gateway restart' to replace it,\n"
                f"   or 'hermes gateway stop' to kill it first.\n"
                f"   Or use 'hermes gateway run --replace' to auto-replace.\n"
            )
            return False

    # Sync bundled skills on gateway start (fast -- skips unchanged)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    except Exception:
        pass

    # Centralized logging — agent.log (INFO+), errors.log (WARNING+),
    # and gateway.log (INFO+, gateway-component records only).
    # Idempotent, so repeated calls from AIAgent.__init__ won't duplicate.
    from hermes_logging import setup_logging, _safe_stderr
    setup_logging(hermes_home=_hermes_home, mode="gateway")

    # Startup security posture audit — warn-on-load, never blocks. Surfaces
    # root / weak-SSH / ephemeral-container / unauthenticated-listener posture
    # so operators get the "you're exposed" signal the June 2026 MCP-config
    # persistence campaign victims never had.
    try:
        from hermes_cli.security_audit_startup import log_startup_security_warnings

        _audit_cfg = None
        try:
            from hermes_cli.config import read_raw_config

            _audit_cfg = read_raw_config()
        except Exception:
            _audit_cfg = None
        log_startup_security_warnings(hermes_home=_hermes_home, config=_audit_cfg)
    except Exception as _audit_exc:
        logger.debug("Startup security audit failed (non-fatal): %s", _audit_exc)

    # Optional stderr handler — level driven by -v/-q flags on the CLI.
    # verbosity=None (-q/--quiet): no stderr output
    # verbosity=0    (default):    WARNING and above
    # verbosity=1    (-v):         INFO and above
    # verbosity=2+   (-vv/-vvv):   DEBUG
    if verbosity is not None:
        from agent.redact import RedactingFormatter

        _stderr_level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
        _stderr_handler = logging.StreamHandler(_safe_stderr())
        _stderr_handler.setLevel(_stderr_level)
        _stderr_handler.setFormatter(RedactingFormatter('%(levelname)s %(name)s: %(message)s'))
        logging.getLogger().addHandler(_stderr_handler)
        # Lower root logger level if needed so DEBUG records can reach the handler
        if _stderr_level < logging.getLogger().level:
            logging.getLogger().setLevel(_stderr_level)

    runner = GatewayRunner(config)
    
    # Track whether an unexpected signal initiated the shutdown. When an
    # unexpected SIGTERM kills the gateway, we exit non-zero so service
    # managers can revive the process. Planned stop paths write a marker
    # before signalling us so they can exit cleanly instead.
    _signal_initiated_shutdown = False

    # Set up signal handlers
    def shutdown_signal_handler(received_signal=None):
        nonlocal _signal_initiated_shutdown
        # Planned --replace takeover check: when a sibling gateway is
        # taking over via --replace, it wrote a marker naming this PID
        # before sending SIGTERM. If present, treat the signal as a
        # planned shutdown and exit 0 so systemd's Restart=on-failure
        # doesn't revive us (which would flap-fight the replacer when
        # both services are enabled, e.g. hermes.service + hermes-
        # gateway.service from pre-rename installs).
        planned_takeover = False
        try:
            from gateway.status import consume_takeover_marker_for_self
            planned_takeover = consume_takeover_marker_for_self()
        except Exception as e:
            logger.debug("Takeover marker check failed: %s", e)

        # Planned stop check: service managers and `hermes gateway stop`
        # also send SIGTERM, which is indistinguishable from an unexpected
        # external kill unless the CLI marks it first. SIGINT comes from an
        # interactive Ctrl+C and is likewise an intentional foreground stop.
        planned_stop = False
        if received_signal == signal.SIGINT:
            planned_stop = True
        elif not planned_takeover:
            try:
                from gateway.status import consume_planned_stop_marker_for_self
                planned_stop = consume_planned_stop_marker_for_self()
            except Exception as e:
                logger.debug("Planned stop marker check failed: %s", e)

        # Fast (<10ms) snapshot of who's asking us to shut down — runs
        # synchronously inside the asyncio signal handler, so we keep it
        # purely stdlib + /proc reads, no subprocesses.  See PR #15826
        # (May 2026): the previous implementation called `ps aux` here
        # synchronously, blocking the event loop for up to 3s while
        # adapter teardown couldn't begin.
        try:
            from gateway.shutdown_forensics import (
                format_context_for_log,
                snapshot_shutdown_context,
                spawn_async_diagnostic,
            )
            _shutdown_ctx = snapshot_shutdown_context(received_signal)
        except Exception as _e:
            _shutdown_ctx = None
            logger.debug("snapshot_shutdown_context failed: %s", _e)

        if planned_takeover:
            logger.info(
                "Received %s as a planned --replace takeover — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM",
            )
        elif planned_stop:
            logger.info(
                "Received %s as a planned gateway stop — exiting cleanly",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT",
            )
        else:
            _signal_initiated_shutdown = True
            # Mirror onto the runner so _stop_impl can suppress the
            # gateway_state=stopped persist for unexpected signals
            # (container/s6 SIGTERM on restart, OOM, bare kill) — see
            # issue #42675. Operator-initiated stops set a planned-stop
            # marker first, land in the `planned_stop` branch above, and
            # leave this flag False so they DO persist "stopped".
            runner._signal_initiated_shutdown = True
            logger.info(
                "Received %s — initiating shutdown",
                _shutdown_ctx["signal"] if _shutdown_ctx else "SIGTERM/SIGINT",
            )

        # Always log who/what triggered the signal — most useful single
        # line when diagnosing "the gateway keeps dying" tickets.  Format
        # is one line, key=value, parent_cmdline last (often long).
        if _shutdown_ctx is not None:
            try:
                logger.warning(
                    "Shutdown context: %s", format_context_for_log(_shutdown_ctx)
                )
            except Exception as _e:
                logger.debug("format_context_for_log failed: %s", _e)

            # Spawn the heavyweight diagnostic (ps auxf, pstree, dmesg) in
            # a detached subprocess so it can finish writing to disk even
            # if our cgroup is being torn down.  Bounded by an internal
            # timeout; never blocks the event loop here.
            try:
                _diag_log = _hermes_home / "logs" / "gateway-shutdown-diag.log"
                spawn_async_diagnostic(
                    _diag_log, _shutdown_ctx["signal"], timeout_seconds=5.0
                )
            except Exception as _e:
                logger.debug("spawn_async_diagnostic failed: %s", _e)
        asyncio.create_task(runner.stop())

    def restart_signal_handler():
        runner.request_restart(detached=False, via_service=True)
    
    loop = asyncio.get_running_loop()

    # Install a loop-level exception handler that swallows transient
    # network errors from background tasks. Issues #31066 / #31110:
    # an unhandled ``telegram.error.TimedOut`` (or peer NetworkError /
    # httpx connection error) in any awaited coroutine would propagate
    # to the loop and kill the gateway process, taking down every
    # profile attached to the same runner. systemd then restarts the
    # service after ~5s but the active conversation turn is lost.
    #
    # The fix is intentionally narrow: only well-known transient
    # network errors are swallowed (and logged with full traceback so
    # the originating call site is still discoverable). Anything else
    # is forwarded to the default handler so real bugs still surface.
    loop.set_exception_handler(_gateway_loop_exception_handler)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, shutdown_signal_handler, sig)  # windows-footgun: ok — wrapped in try/except NotImplementedError for Windows
            except NotImplementedError:
                pass
        if hasattr(signal, "SIGUSR1"):
            try:
                loop.add_signal_handler(signal.SIGUSR1, restart_signal_handler)  # windows-footgun: ok — POSIX signal, guarded by hasattr above + try/except NotImplementedError
            except NotImplementedError:
                pass
    else:
        logger.info("Skipping signal handlers (not running in main thread).")

    # Windows fallback: asyncio.add_signal_handler raises NotImplementedError
    # on Windows, so `hermes gateway stop`'s SIGTERM (which Python maps to
    # TerminateProcess on Windows) never invokes shutdown_signal_handler.
    # That means the drain loop never runs, mark_resume_pending never fires,
    # and sessions are silently lost across restarts (issue #33778).
    #
    # The fix is a marker-polling thread: `hermes gateway stop` writes the
    # planned-stop marker BEFORE killing, and this thread notices it and
    # drives the same shutdown path the signal handler would have.  Runs
    # on every platform (cheap, defensive) so non-signal-bearing
    # environments (Windows native, sandboxed CI runners that mask
    # SIGTERM) still get a clean drain.
    _planned_stop_watcher_stop = threading.Event()
    _planned_stop_watcher_thread = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(_planned_stop_watcher_stop, runner, loop, shutdown_signal_handler),
        daemon=True,
        name="planned-stop-watcher",
    )
    _planned_stop_watcher_thread.start()

    # Claim the PID file BEFORE bringing up any platform adapters.
    # This closes the --replace race window: two concurrent `gateway run
    # --replace` invocations both pass the termination-wait above, but
    # only the winner of the O_CREAT|O_EXCL race below will ever open
    # Telegram polling, Discord gateway sockets, etc. The loser exits
    # cleanly before touching any external service.
    import atexit
    from gateway.status import write_pid_file, remove_pid_file, get_running_pid
    _current_pid = get_running_pid()
    if _current_pid is not None and _current_pid != os.getpid():
        logger.error(
            "Another gateway instance (PID %d) started during our startup. "
            "Exiting to avoid double-running.", _current_pid
        )
        return False
    if not acquire_gateway_runtime_lock():
        logger.error(
            "Gateway runtime lock is already held by another instance. Exiting."
        )
        return False
    try:
        write_pid_file()
    except FileExistsError:
        release_gateway_runtime_lock()
        logger.error(
            "PID file race lost to another gateway instance. Exiting."
        )
        return False
    atexit.register(remove_pid_file)
    atexit.register(release_gateway_runtime_lock)

    try:
        from hermes_cli.nous_auth_keepalive import start_nous_auth_keepalive

        start_nous_auth_keepalive()
    except Exception as exc:
        logger.debug("Nous auth keepalive did not start: %s", exc)

    _ensure_windows_gateway_venv_imports()

    # MCP tool discovery — run in an executor so the asyncio event loop
    # stays responsive even when a configured MCP server is slow or
    # unreachable.  discover_mcp_tools() uses a blocking 120s wait
    # internally; calling it from the loop thread would freeze platform
    # heartbeats (Discord shard, Telegram polling) until it returned.
    # See #16856.
    try:
        from tools.mcp_tool import discover_mcp_tools
        _loop = asyncio.get_running_loop()
        await _loop.run_in_executor(None, discover_mcp_tools)
    except Exception as e:
        logger.debug("MCP tool discovery failed: %s", e)

    # Start the gateway
    success = await runner.start()
    if not success:
        return False
    if runner.should_exit_cleanly:
        if runner.exit_reason:
            logger.error("Gateway exiting cleanly: %s", runner.exit_reason)
        # A clean exit that carries an explicit exit code (e.g. a fatal
        # config error stamped with GATEWAY_FATAL_CONFIG_EXIT_CODE) must
        # propagate that code to the process so the s6 finish script can
        # translate it (78 → 125) and stop the supervisor restart loop.
        # Without this, the early `return True` below makes main() exit 0,
        # the finish script's `[ "$1" = "78" ]` check never matches, and
        # s6 crash-loops the gateway anyway (#51228).
        if runner.exit_code is not None:
            raise SystemExit(runner.exit_code)
        return True
    if not runner._running:
        # Startup was intentionally aborted by restart/shutdown before entering
        # running mode; preserve that lifecycle path without starting cron.
        await runner.wait_for_shutdown()
        if runner.should_exit_with_failure:
            if runner.exit_reason:
                logger.error("Gateway exiting with failure: %s", runner.exit_reason)
            return False
        try:
            from tools.mcp_tool import shutdown_mcp_servers
            shutdown_mcp_servers()
        except Exception:
            pass
        if runner.exit_code is not None:
            raise SystemExit(runner.exit_code)
        return True
    
    # Start the background cron scheduler via the resolved provider so
    # scheduled jobs fire automatically. The built-in provider is the
    # historical in-process 60s ticker; an external provider (e.g. chronos)
    # may arm a schedule and return. Pass the event loop so cron delivery can
    # use live adapters (E2EE support).
    from cron.scheduler_provider import resolve_cron_scheduler
    cron_stop = threading.Event()
    cron_provider = resolve_cron_scheduler()
    cron_thread = threading.Thread(
        target=cron_provider.start,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop()},
        daemon=True,
        name="cron-scheduler",
    )
    cron_thread.start()

    # Gateway-only periodic housekeeping (channel dir, cache cleanup, paste
    # sweep, curator) — runs independently of which cron provider is active.
    # Shares cron_stop as the shutdown signal.
    housekeeping_thread = threading.Thread(
        target=_start_gateway_housekeeping,
        args=(cron_stop,),
        kwargs={"adapters": runner.adapters, "loop": asyncio.get_running_loop()},
        daemon=True,
        name="gateway-housekeeping",
    )
    housekeeping_thread.start()
    
    # Wait for shutdown
    await runner.wait_for_shutdown()

    try:
        from hermes_cli.nous_auth_keepalive import stop_nous_auth_keepalive

        stop_nous_auth_keepalive()
    except Exception:
        pass

    if runner.should_exit_with_failure:
        if runner.exit_reason:
            logger.error("Gateway exiting with failure: %s", runner.exit_reason)
        return False
    
    # Stop cron scheduler + housekeeping cleanly
    cron_stop.set()
    try:
        cron_provider.stop()
    except Exception as e:
        logger.debug("Cron provider stop() error: %s", e)
    cron_thread.join(timeout=5)
    housekeeping_thread.join(timeout=5)

    # Stop the planned-stop watcher (daemon=True so this is belt-and-suspenders).
    _planned_stop_watcher_stop.set()
    _planned_stop_watcher_thread.join(timeout=2)

    # Close MCP server connections
    try:
        from tools.mcp_tool import shutdown_mcp_servers
        shutdown_mcp_servers()
    except Exception:
        pass

    if runner.exit_code is not None:
        raise SystemExit(runner.exit_code)

    # When an unexpected SIGTERM caused the shutdown and it wasn't a planned
    # restart (/restart, /update, SIGUSR1), exit non-zero so systemd's
    # Restart=on-failure revives the process.  This covers:
    #   - hermes update killing the gateway mid-work
    #   - External kill commands
    #   - WSL2/container runtime sending unexpected signals
    # `hermes gateway stop` and interactive Ctrl+C are handled above as
    # planned stops and should not trigger service-manager revival.
    if _signal_initiated_shutdown and not runner._restart_requested:
        logger.info(
            "Exiting with code 1 (signal-initiated shutdown without restart "
            "request) so systemd Restart=on-failure can revive the gateway."
        )
        return False  # → sys.exit(1) in the caller

    # Older restart paths may reach here without ``runner.exit_code`` set.
    # Keep the historical non-zero fallback for service-managed restarts.
    if runner._restart_via_service:
        logger.info(
            "Exiting with code 75 (service-restart requested) so the service "
            "manager relaunches the gateway."
        )
        raise SystemExit(75)

    return True


def main():
    """CLI entry point for the gateway."""
    # Force UTF-8 stdio on Windows — gateway logs and startup banner would
    # otherwise UnicodeEncodeError on cp1252 consoles.  No-op on POSIX.
    try:
        from hermes_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    import argparse
    
    parser = argparse.ArgumentParser(description="Hermes Gateway - Multi-platform messaging")
    parser.add_argument("--config", "-c", help="Path to gateway config file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            config = GatewayConfig.from_dict(data)
    
    # start_gateway() performs the full graceful teardown (adapters
    # disconnected, sessions saved + flushed, SQLite closed, cron/MCP stopped,
    # PID file + runtime lock released) before it returns OR raises SystemExit
    # with an explicit code. Force-exit afterwards so a wedged non-daemon worker
    # thread (e.g. a ThreadPoolExecutor tool/LLM call blocked with no timeout)
    # cannot block interpreter finalization (Py_FinalizeEx joins all non-daemon
    # threads, incl. concurrent.futures' _python_exit) and strand the gateway
    # half-shut down with the supervisor unable to restart it (#53107).
    #
    # SystemExit is caught explicitly: start_gateway raises it on the
    # clean-fatal-config (#51228), planned-restart, and service-restart paths,
    # all of which complete teardown first. Routing those codes through the
    # same os._exit backstop means EVERY exit path is wedge-proof, not just the
    # boolean-return ones.
    try:
        success = asyncio.run(start_gateway(config))
        exit_code = 0 if success else 1
    except SystemExit as e:
        # e.code may be None (→ 0), an int, or a str (→ 1, like CPython).
        if e.code is None:
            exit_code = 0
        elif isinstance(e.code, int):
            exit_code = e.code
        else:
            exit_code = 1
    _exit_after_graceful_shutdown(exit_code)


def _exit_after_graceful_shutdown(exit_code: int) -> None:
    """Flush stdio, release the PID file + runtime lock, then hard-exit.

    Graceful teardown is already complete by the time this runs, so there is
    nothing left that needs a clean interpreter shutdown. We deliberately use
    ``os._exit`` (not ``sys.exit``): ``sys.exit`` raises ``SystemExit``, which
    triggers ``Py_FinalizeEx`` → ``wait_for_thread_shutdown`` and joins every
    non-daemon thread — exactly the hang (#53107) a wedged tool-worker causes.

    ``os._exit`` bypasses ``atexit`` handlers, so we cannot rely on the
    ``atexit``-registered ``remove_pid_file`` / ``release_gateway_runtime_lock``
    (registered in ``start_gateway``) to run. The full-shutdown path releases
    both explicitly in ``_stop_impl``, but the EARLY exit paths —
    clean-fatal-config (#51228) and startup-aborted-before-running — raise
    ``SystemExit`` right after ``runner.start()`` without going through
    ``_stop_impl``, so on those paths ``atexit`` was the only thing releasing
    them. Now that those paths are routed through this backstop (#53107),
    release both here explicitly. Both calls are idempotent —
    ``remove_pid_file`` only unlinks a PID file that belongs to this process,
    and ``release_gateway_runtime_lock`` no-ops when the lock is already
    released — so this is a no-op on the normal shutdown path and the actual
    cleanup on the early-exit paths.

    Logging is not flushed here: the gateway's handlers are synchronous
    ``RotatingFileHandler``s that write each record immediately (no
    ``MemoryHandler``/``QueueHandler`` buffering), so there is nothing pending.
    Only stdio is buffered, so only stdio is flushed.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    # Guaranteed cleanup chokepoint: os._exit skips atexit, and the early
    # SystemExit exit paths never run _stop_impl, so release here (idempotent).
    try:
        from gateway.status import remove_pid_file, release_gateway_runtime_lock
        remove_pid_file()
        release_gateway_runtime_lock()
    except Exception:
        pass
    os._exit(exit_code)


if __name__ == "__main__":
    main()
