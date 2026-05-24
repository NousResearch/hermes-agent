"""Block and redact GitHub PAT-like credentials.

The plugin protects multiple boundaries:

- ``pre_gateway_dispatch`` blocks inbound gateway messages before they reach
  the agent loop.
- ``pre_tool_call`` blocks tool calls whose arguments contain sensitive tokens
  or upstream scanner notices.
- transform hooks redact credentials from tool results, terminal output, and
  final assistant output.

Detection records are written without payloads or matched substrings.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from hermes_constants import get_hermes_home

LOGGER = logging.getLogger(__name__)
REDACTION = "[REDACTED_GITHUB_PAT]"
SCAN_NOTICE_REDACTION = "[REDACTED_SECURITY_SCAN_NOTICE]"

# GitHub token families are matched by provider-specific prefixes. The regex
# strings are assembled from fragments so source scanners do not flag this file
# as containing a token-like literal.
SHORT_PREFIX_PATTERN = r"\b" + "gh" + r"[pousr]_[A-Za-z0-9_]{30,255}\b"
FINE_GRAINED_PREFIX_PATTERN = r"\b" + "github" + r"_pat_[A-Za-z0-9_]{50,255}\b"
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(SHORT_PREFIX_PATTERN),
    re.compile(FINE_GRAINED_PREFIX_PATTERN),
)

# Some frontends/security layers replace the actual secret with a diagnostic
# notice before Hermes sees the text. Treat those as sensitive too: the token
# is already gone, but we still want to skip the agent/tool path and tell the
# user to revoke/rotate. These regex strings are also assembled from fragments
# to keep repository scanners quiet.
SCAN_NOTICE_PATTERN = r"Security scan\s*[—:-].*" + "Git" + r"Hub PAT detected"
PROVIDER_NOTICE_PATTERN = "credential matching a known " + "provider pattern"
SECURITY_SCAN_NOTICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(SCAN_NOTICE_PATTERN, re.IGNORECASE | re.DOTALL),
    re.compile(PROVIDER_NOTICE_PATTERN, re.IGNORECASE),
)

INBOUND_WARNING = (
    "⚠️ This message appears to contain a GitHub PAT/token. "
    "It will not be passed to the agent or tools. "
    "If this was a real token, revoke/rotate it in GitHub immediately. "
    "Store secrets in `~/.hermes/.env`, a credential pool, or environment variables instead."
)


def _log_file():
    return get_hermes_home() / "logs" / "secret_guard.log"


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _has_secret(text: str | None) -> bool:
    candidate = text or ""
    return any(pattern.search(candidate) for pattern in SECRET_PATTERNS)


def _has_security_scan_notice(text: str | None) -> bool:
    candidate = text or ""
    return any(pattern.search(candidate) for pattern in SECURITY_SCAN_NOTICE_PATTERNS)


def _should_block_sensitive(text: str | None) -> bool:
    return _has_secret(text) or _has_security_scan_notice(text)


def _redact(text: str | None) -> str | None:
    if text is None:
        return None
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(REDACTION, redacted)
    for pattern in SECURITY_SCAN_NOTICE_PATTERNS:
        redacted = pattern.sub(SCAN_NOTICE_REDACTION, redacted)
    return redacted


def _platform_name(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").lower()


def _thread_metadata(source: Any, event: Any) -> dict[str, Any] | None:
    """Best-effort metadata to reply in the same Telegram topic/thread."""
    thread_id = getattr(source, "thread_id", None)
    if thread_id is None:
        return None

    metadata: dict[str, Any] = {"thread_id": thread_id}
    if _platform_name(getattr(source, "platform", None)) == "telegram" and getattr(source, "chat_type", None) == "dm":
        metadata["telegram_dm_topic_reply_fallback"] = True
        tid = str(thread_id)
        if tid and tid != "1":
            metadata["direct_messages_topic_id"] = tid
        anchor = getattr(event, "message_id", None) or getattr(source, "message_id", None)
        if anchor is not None:
            metadata["telegram_reply_to_message_id"] = str(anchor)
    return metadata


def _log_detection(where: str, **fields: Any) -> None:
    try:
        log_file = _log_file()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "where": where,
            **fields,
        }
        # Never log payloads or matched substrings here.
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - logging must never break flow
        LOGGER.debug("secret-guard log write failed: %s", exc)


def _schedule_warning(gateway: Any, source: Any, event: Any) -> None:
    try:
        adapter = getattr(gateway, "adapters", {}).get(getattr(source, "platform", None))
        chat_id = getattr(source, "chat_id", None)
        if not adapter or not chat_id:
            return

        coro = adapter.send(
            chat_id,
            INBOUND_WARNING,
            metadata=_thread_metadata(source, event),
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
        except RuntimeError:
            # CLI/unit-test fallback: no running event loop, so just close the coroutine.
            try:
                coro.close()
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - notification failure should not pass secret onward
        LOGGER.warning("secret-guard failed to schedule gateway warning: %s", exc)


def pre_gateway_dispatch(event: Any, gateway: Any, session_store: Any = None, **kwargs: Any):
    text = getattr(event, "text", "") or ""
    if not _should_block_sensitive(text):
        return None

    source = getattr(event, "source", None)
    _log_detection(
        "pre_gateway_dispatch",
        platform=_platform_name(getattr(source, "platform", None)),
        chat_id=getattr(source, "chat_id", None),
        user_id=getattr(source, "user_id", None),
        message_id=getattr(event, "message_id", None),
        scan_notice=_has_security_scan_notice(text),
    )
    _schedule_warning(gateway, source, event)
    return {"action": "skip", "reason": "secret-guard: sensitive-credential-detected"}


def block_tool_call(tool_name: str, args: dict, task_id: str = "", **kwargs: Any):
    text = _to_text(args)
    if not _should_block_sensitive(text):
        return None

    _log_detection(
        "pre_tool_call",
        tool_name=tool_name,
        task_id=task_id,
        scan_notice=_has_security_scan_notice(text),
    )
    return {
        "action": "block",
        "message": (
            "Security policy blocked this tool call: GitHub PAT-like credential "
            "was detected in tool arguments. If this was a real token, revoke/rotate it; "
            "use ~/.hermes/.env, credential pools, or environment variables instead."
        ),
    }


def redact_tool_result(
    tool_name: str,
    result: str,
    args: dict | None = None,
    task_id: str | None = None,
    **kwargs: Any,
):
    # model_tools invokes this hook with keyword `args=...`.
    # Keep **kwargs for forward compatibility with future hook fields.
    redacted = _redact(result)
    if redacted != result:
        _log_detection("transform_tool_result", tool_name=tool_name, task_id=task_id)
        return redacted
    return None


def redact_terminal_output(
    command: str = "",
    output: str = "",
    exit_code: int | None = None,
    cwd: str = "",
    task_id: str | None = None,
    **kwargs: Any,
):
    # Terminal hook call sites have evolved; keep all fields optional and
    # accept **kwargs so the guard remains compatible across Hermes versions.
    if not output and isinstance(kwargs.get("result"), str):
        output = kwargs["result"]
    redacted = _redact(output)
    if redacted != output:
        _log_detection("transform_terminal_output", exit_code=exit_code, cwd=cwd, task_id=task_id)
        return redacted
    return None


def redact_llm_output(
    response_text: str,
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **kwargs: Any,
):
    redacted = _redact(response_text)
    if redacted != response_text:
        _log_detection("transform_llm_output", session_id=session_id, model=model, platform=platform)
        return redacted
    return None


def register(ctx):
    LOGGER.info("secret-guard plugin loaded: registering gateway/tool/output hooks")
    ctx.register_hook("pre_gateway_dispatch", pre_gateway_dispatch)
    ctx.register_hook("pre_tool_call", block_tool_call)
    ctx.register_hook("transform_tool_result", redact_tool_result)
    ctx.register_hook("transform_terminal_output", redact_terminal_output)
    ctx.register_hook("transform_llm_output", redact_llm_output)
