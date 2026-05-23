"""Gateway context spill and quarantine helpers.

Deterministic admission control for gateway sessions that have grown too
large to send safely to an LLM.  This module is intentionally file-only: it
must not call auxiliary models while deciding or writing a spill.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agent.model_metadata import estimate_messages_tokens_rough
from hermes_constants import get_hermes_home


_DEFAULT_HARD_TOKEN_LIMIT = 180_000
_DEFAULT_HARD_MESSAGE_LIMIT = 180
_DEFAULT_HARD_CHAR_LIMIT = 500_000
_DEFAULT_REQUEST_BYTE_LIMIT = 1_800_000
_DEFAULT_EXCERPT_CHARS = 24_000
_DEFAULT_RECOVERY_ASK_CHARS = 2_000
_MAX_RECOVERY_MESSAGE_CHARS = 4_000
_MAX_USER_NOTICE_CHARS = 500
_MAX_TRANSCRIPT_MESSAGE_CHARS = 1_200
_MAX_TOOL_EXCERPT_CHARS = 500
_DEFAULT_TOOL_SCHEMA_OVERHEAD_TOKENS = 20_000
_DEFAULT_IMAGE_TOKEN_BUDGET = 2_000


@dataclass(frozen=True)
class ContextSpillConfig:
    enabled: bool = True
    mode: str = "enforce"
    wiki_dir: Path = Path("~/wiki/outputs/gateway-context-spills")
    raw_state_dir: Path = Path("~/.hermes/state/context-spills")
    token_threshold_ratio: float = 0.70
    hard_token_limit: int = _DEFAULT_HARD_TOKEN_LIMIT
    hard_message_limit: int = _DEFAULT_HARD_MESSAGE_LIMIT
    hard_char_limit: int = _DEFAULT_HARD_CHAR_LIMIT
    hard_request_byte_limit: int = _DEFAULT_REQUEST_BYTE_LIMIT
    include_raw_state: bool = True
    include_redacted_wiki_excerpt_chars: int = _DEFAULT_EXCERPT_CHARS
    reset_live_session: bool = True
    notify_user: bool = True
    fallback_safe: bool = True
    retention_days: int = 30
    allow_unsafe_paths: bool = False


@dataclass(frozen=True)
class ContextSpillDecision:
    should_spill: bool
    reason: str
    approx_tokens: int
    char_count: int
    message_count: int
    threshold_tokens: int
    request_bytes: int = 0
    stage: str = "unknown"


@dataclass(frozen=True)
class ContextSpillResult:
    spill_id: str
    wiki_path: Optional[Path]
    raw_path: Optional[Path]
    manifest_path: Optional[Path]
    recovery_message: str
    user_notice: str
    decision: ContextSpillDecision


class ContextSpillWriteError(RuntimeError):
    """Raised when a required spill artifact cannot be written safely."""


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enforce"}


def _coerce_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _coerce_ratio(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0 or parsed > 1:
        return default
    return parsed


def _resolve_path(value: Any, default: str) -> Path:
    raw = str(value or default)
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _validate_spill_paths(wiki_dir: Path, raw_dir: Path, hermes_home: Path, *, allow_unsafe: bool) -> None:
    if allow_unsafe:
        return
    home_wiki = (Path.home() / "wiki").resolve()
    default_raw_root = (hermes_home / "state" / "context-spills").resolve()
    if wiki_dir.exists() and wiki_dir.is_symlink():
        raise ValueError(f"gateway_context_spill.wiki_dir cannot be a symlink: {wiki_dir}")
    if raw_dir.exists() and raw_dir.is_symlink():
        raise ValueError(f"gateway_context_spill.raw_state_dir cannot be a symlink: {raw_dir}")
    if not _path_is_under(wiki_dir, home_wiki):
        raise ValueError(f"gateway_context_spill.wiki_dir must stay under {home_wiki}: {wiki_dir}")
    if not _path_is_under(raw_dir, default_raw_root):
        raise ValueError(f"gateway_context_spill.raw_state_dir must stay under {default_raw_root}: {raw_dir}")


def load_context_spill_config(
    user_config: Optional[dict[str, Any]] = None,
    hermes_home: Optional[Path] = None,
    *,
    allow_unsafe_paths: bool = False,
) -> ContextSpillConfig:
    """Load and validate gateway context-spill config from config.yaml data."""
    hermes_home = Path(hermes_home or get_hermes_home()).resolve()
    user_config = user_config or {}
    raw_cfg = user_config.get("gateway_context_spill", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raw_cfg = {}

    mode = str(raw_cfg.get("mode") or "enforce").strip().lower()
    if mode not in {"enforce", "shadow", "off"}:
        mode = "enforce"
    enabled = _coerce_bool(raw_cfg.get("enabled"), True) and mode != "off"
    env_unsafe = os.getenv("HERMES_CONTEXT_SPILL_ALLOW_UNSAFE_PATHS", "").strip().lower() in {"1", "true", "yes"}
    allow_unsafe = bool(allow_unsafe_paths or raw_cfg.get("allow_unsafe_paths") or env_unsafe)

    wiki_dir = _resolve_path(raw_cfg.get("wiki_dir"), "~/wiki/outputs/gateway-context-spills")
    raw_state_dir = _resolve_path(raw_cfg.get("raw_state_dir"), str(hermes_home / "state" / "context-spills"))
    _validate_spill_paths(wiki_dir, raw_state_dir, hermes_home, allow_unsafe=allow_unsafe)

    return ContextSpillConfig(
        enabled=enabled,
        mode=mode,
        wiki_dir=wiki_dir,
        raw_state_dir=raw_state_dir,
        token_threshold_ratio=_coerce_ratio(raw_cfg.get("token_threshold_ratio"), 0.70),
        hard_token_limit=_coerce_int(raw_cfg.get("hard_token_limit"), _DEFAULT_HARD_TOKEN_LIMIT, minimum=1),
        hard_message_limit=_coerce_int(raw_cfg.get("hard_message_limit"), _DEFAULT_HARD_MESSAGE_LIMIT, minimum=1),
        hard_char_limit=_coerce_int(raw_cfg.get("hard_char_limit"), _DEFAULT_HARD_CHAR_LIMIT, minimum=1),
        hard_request_byte_limit=_coerce_int(raw_cfg.get("hard_request_byte_limit"), _DEFAULT_REQUEST_BYTE_LIMIT, minimum=1),
        include_raw_state=_coerce_bool(raw_cfg.get("include_raw_state"), True),
        include_redacted_wiki_excerpt_chars=_coerce_int(raw_cfg.get("include_redacted_wiki_excerpt_chars"), _DEFAULT_EXCERPT_CHARS, minimum=0),
        reset_live_session=_coerce_bool(raw_cfg.get("reset_live_session"), True),
        notify_user=_coerce_bool(raw_cfg.get("notify_user"), True),
        fallback_safe=_coerce_bool(raw_cfg.get("fallback_safe"), True),
        retention_days=_coerce_int(raw_cfg.get("retention_days"), 30, minimum=1),
        allow_unsafe_paths=allow_unsafe,
    )


def _message_content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def estimate_gateway_payload(
    history: list[dict[str, Any]],
    message_text: str = "",
    *,
    system_prompt: str = "",
    channel_prompt: str = "",
    tool_schema_count: int = 0,
    image_count: int = 0,
    extra_overhead_tokens: int = _DEFAULT_TOOL_SCHEMA_OVERHEAD_TOKENS,
) -> tuple[int, int, int, int]:
    """Return rough tokens, chars, message count, and request bytes."""
    safe_history = history if isinstance(history, list) else []
    current = []
    if message_text:
        current.append({"role": "user", "content": message_text})
    messages = safe_history + current
    approx_tokens = estimate_messages_tokens_rough(messages)
    approx_tokens += max(0, int(extra_overhead_tokens or 0))
    approx_tokens += max(0, int(tool_schema_count or 0)) * 750
    approx_tokens += max(0, int(image_count or 0)) * _DEFAULT_IMAGE_TOKEN_BUDGET
    approx_tokens += max(0, (len(system_prompt or "") + len(channel_prompt or "")) // 4)

    char_count = sum(_message_content_chars(m) for m in safe_history)
    char_count += len(message_text or "") + len(system_prompt or "") + len(channel_prompt or "")
    message_count = len(safe_history) + (1 if message_text else 0)
    request_bytes = len(json.dumps(messages, ensure_ascii=False, default=str).encode("utf-8", "ignore"))
    request_bytes += len((system_prompt or "").encode("utf-8", "ignore"))
    request_bytes += len((channel_prompt or "").encode("utf-8", "ignore"))
    return approx_tokens, char_count, message_count, request_bytes


def decide_context_spill(
    history: list[dict[str, Any]],
    message_text: str = "",
    *,
    context_length: int,
    config: ContextSpillConfig,
    stage: str = "unknown",
    system_prompt: str = "",
    channel_prompt: str = "",
    tool_schema_count: int = 0,
    image_count: int = 0,
    extra_overhead_tokens: int = _DEFAULT_TOOL_SCHEMA_OVERHEAD_TOKENS,
) -> ContextSpillDecision:
    """Decide whether the gateway must spill/quarantine before model call."""
    context_length = int(context_length or 200_000)
    threshold_tokens = max(1, min(config.hard_token_limit, max(50_000, int(context_length * config.token_threshold_ratio))))
    approx_tokens, char_count, message_count, request_bytes = estimate_gateway_payload(
        history,
        message_text,
        system_prompt=system_prompt,
        channel_prompt=channel_prompt,
        tool_schema_count=tool_schema_count,
        image_count=image_count,
        extra_overhead_tokens=extra_overhead_tokens,
    )
    reasons: list[str] = []
    if message_count >= config.hard_message_limit:
        reasons.append(f"message_limit:{message_count}>={config.hard_message_limit}")
    if approx_tokens >= threshold_tokens:
        reasons.append(f"token_limit:{approx_tokens}>={threshold_tokens}")
    if char_count >= config.hard_char_limit:
        reasons.append(f"char_limit:{char_count}>={config.hard_char_limit}")
    if request_bytes >= config.hard_request_byte_limit:
        reasons.append(f"request_byte_limit:{request_bytes}>={config.hard_request_byte_limit}")
    return ContextSpillDecision(
        should_spill=bool(config.enabled and reasons),
        reason=",".join(reasons) if reasons else "within_limits",
        approx_tokens=approx_tokens,
        char_count=char_count,
        message_count=message_count,
        threshold_tokens=threshold_tokens,
        request_bytes=request_bytes,
        stage=stage,
    )


def _slug(value: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value or "session").strip("-._")
    return (slug or "session")[:max_len]


def _atomic_write_text(path: Path, text: str, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str), mode=mode)


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text or "", force=True)


def _metadata_for(source: Any, session_entry: Any) -> dict[str, Any]:
    platform = getattr(getattr(source, "platform", None), "value", None) or str(getattr(source, "platform", "") or "")
    return {
        "platform": platform,
        "chat_id": getattr(source, "chat_id", None),
        "thread_id": getattr(source, "thread_id", None),
        "user_id": getattr(source, "user_id", None),
        "chat_type": getattr(source, "chat_type", None),
        "session_key": getattr(session_entry, "session_key", None),
        "session_id": getattr(session_entry, "session_id", None),
    }


def _compact_prior_gateway_handoff(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("[GATEWAY HANDOFF]"):
        return text
    lines = stripped.splitlines()
    kept: list[str] = []
    for line in lines:
        kept.append(line)
        if line.startswith("Current user ask"):
            break
    ask_lines: list[str] = []
    in_ask = False
    for line in lines:
        if line.startswith("Current user ask"):
            in_ask = True
            continue
        if in_ask and line.startswith("Instructions:"):
            break
        if in_ask:
            ask_lines.append(line)
    ask = "\n".join(ask_lines).strip()
    if ask:
        return "\n".join(kept) + f"\n{ask[:_MAX_TRANSCRIPT_MESSAGE_CHARS]}\n[prior gateway handoff body omitted; use its linked wiki handoff if needed]"
    return "[prior gateway handoff omitted; use its linked wiki handoff if needed]"


def _compact_tool_excerpt(text: str) -> str:
    prefix = "[tool output omitted from wiki handoff; raw 0600 bundle preserves full content]"
    if not text:
        return prefix
    tool_name = "tool"
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("tool", "name", "function", "recipient_name"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    tool_name = value.strip()
                    break
            error = payload.get("error")
            if error:
                err = str(error)[:_MAX_TOOL_EXCERPT_CHARS]
                return f"[{tool_name} output omitted from wiki handoff; raw 0600 bundle preserves full content]\nerror: {err}"
    except Exception:
        pass
    return f"[{tool_name} output omitted from wiki handoff; {len(text)} chars; raw 0600 bundle preserves full content]"


def _message_excerpt_text(role: str, content: Any) -> str:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    role_norm = (role or "").lower()
    if role_norm == "tool":
        return _compact_tool_excerpt(text)
    text = _compact_prior_gateway_handoff(text)
    if len(text) > _MAX_TRANSCRIPT_MESSAGE_CHARS:
        return text[:_MAX_TRANSCRIPT_MESSAGE_CHARS] + f"\n[… {len(text) - _MAX_TRANSCRIPT_MESSAGE_CHARS} chars omitted from wiki excerpt; raw bundle preserves full content]"
    return text


def _render_transcript_excerpt(history: list[dict[str, Any]], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if not history:
        return "(empty)"
    lines: list[str] = []
    head = history[:12]
    tail = history[-12:] if len(history) > 24 else []
    selected = head + ([{"role": "system", "content": f"... {len(history) - 24} messages omitted ..."}] if tail else []) + tail
    for idx, msg in enumerate(selected, 1):
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        text = _message_excerpt_text(str(role), content)
        lines.append(f"## {idx}. {role}\n\n{text}")
    excerpt = "\n\n".join(lines)
    return excerpt[:max_chars]


def _render_wiki_markdown(
    *,
    spill_id: str,
    timestamp: str,
    history: list[dict[str, Any]],
    message_text: str,
    metadata: dict[str, Any],
    decision: ContextSpillDecision,
    raw_path: Optional[Path],
    config: ContextSpillConfig,
) -> str:
    current_ask = _redact(_compact_prior_gateway_handoff(message_text or "(empty)"))[: config.include_redacted_wiki_excerpt_chars]
    transcript_excerpt = _redact(_render_transcript_excerpt(history, config.include_redacted_wiki_excerpt_chars))
    return (
        f"# Gateway context spill — {timestamp[:10]} — {_slug(str(metadata.get('session_key') or 'session'))}\n\n"
        "This handoff is source of truth for recovering this gateway session; do not replay the raw transcript into a model.\n\n"
        f"- Spill ID: `{spill_id}`\n"
        f"- Original session id: `{metadata.get('session_id')}`\n"
        f"- Session key: `{metadata.get('session_key')}`\n"
        f"- Platform: `{metadata.get('platform')}`\n"
        f"- Chat/thread: `{metadata.get('chat_id')}` / `{metadata.get('thread_id')}`\n"
        f"- Stage: `{decision.stage}`\n"
        f"- Reason: `{decision.reason}`\n"
        f"- Approx tokens: `{decision.approx_tokens}`\n"
        f"- Threshold tokens: `{decision.threshold_tokens}`\n"
        f"- Messages/chars/request bytes: `{decision.message_count}` / `{decision.char_count}` / `{decision.request_bytes}`\n"
        f"- Raw local bundle: `{raw_path if raw_path else 'none'}`\n\n"
        "## Current user ask (redacted)\n\n"
        f"```text\n{current_ask}\n```\n\n"
        "## Recovery instructions\n\n"
        "1. Continue from this handoff.\n"
        "2. Do not ask the user to repeat themselves.\n"
        "3. Use this wiki handoff for older context.\n"
        "4. Do not load or paste the raw full transcript into a model.\n\n"
        "## Redacted transcript excerpt\n\n"
        f"{transcript_excerpt}\n"
    )


def _build_recovery_message(
    *,
    spill_id: str,
    wiki_path: Optional[Path],
    raw_path: Optional[Path],
    old_session_id: str,
    reason: str,
    current_ask: str,
) -> str:
    try:
        redacted_ask = _redact(_compact_prior_gateway_handoff(current_ask or ""))[:_DEFAULT_RECOVERY_ASK_CHARS]
    except Exception:
        redacted_ask = "[redaction failed; ask omitted]"
    message = (
        "[GATEWAY HANDOFF]\n"
        "The prior gateway session exceeded safe context limits and was spilled before model call.\n"
        f"- Spill ID: {spill_id}\n"
        f"- Wiki handoff: {wiki_path if wiki_path else 'none'}\n"
        f"- Raw local bundle: {raw_path if raw_path else 'none'}\n"
        f"- Original session id: {old_session_id}\n"
        f"- Reason: {reason}\n\n"
        "Current user ask, redacted excerpt:\n"
        f"{redacted_ask}\n\n"
        "Instructions:\n"
        "1. Continue from this handoff.\n"
        "2. Do not ask the user to repeat themselves.\n"
        "3. If older context is needed, read the wiki handoff file with file tools.\n"
        "4. Do not load or paste the raw full transcript into the model.\n"
    )
    return message[:_MAX_RECOVERY_MESSAGE_CHARS]


def _build_user_notice(wiki_path: Optional[Path], raw_path: Optional[Path]) -> str:
    target = str(wiki_path) if wiki_path else f"raw local spill only: {raw_path}"
    notice = (
        "Context was too large to send safely. I wrote a recovery handoff and reset this lane so it can continue without replaying the broken transcript:\n"
        f"{target}"
    )
    try:
        notice = _redact(notice)
    except Exception:
        notice = "Context was too large to send safely. I wrote a local recovery handoff and reset this lane."
    return notice[:_MAX_USER_NOTICE_CHARS]


def write_context_spill(
    *,
    history: list[dict[str, Any]],
    message_text: str,
    source: Any,
    session_entry: Any,
    decision: ContextSpillDecision,
    config: ContextSpillConfig,
) -> ContextSpillResult:
    """Write raw/wiki spill artifacts and return the reduced handoff message.

    Raw write is load-bearing when enabled.  If raw cannot be written, callers
    must fail closed and avoid calling a model with the oversized payload.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_key = str(getattr(session_entry, "session_key", "session") or "session")
    old_session_id = str(getattr(session_entry, "session_id", "") or "")
    spill_id = f"{timestamp}-{_slug(session_key, 24)}-{uuid.uuid4().hex[:8]}"
    metadata = _metadata_for(source, session_entry)

    config.raw_state_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(config.raw_state_dir, 0o700)
    manifest_dir = config.raw_state_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(manifest_dir, 0o700)

    raw_path: Optional[Path] = None
    if config.include_raw_state:
        raw_path = config.raw_state_dir / f"{spill_id}.json"
        raw_payload = {
            "spill_id": spill_id,
            "timestamp": timestamp,
            "metadata": metadata,
            "decision": asdict(decision),
            "history": history,
            "message_text": message_text,
        }
        try:
            _atomic_write_json(raw_path, raw_payload, mode=0o600)
        except Exception as exc:
            raise ContextSpillWriteError(f"raw spill write failed: {type(exc).__name__}: {exc}") from exc

    wiki_path: Optional[Path] = None
    wiki_error: Optional[str] = None
    try:
        wiki_path = config.wiki_dir / f"{spill_id}.md"
        markdown = _render_wiki_markdown(
            spill_id=spill_id,
            timestamp=timestamp,
            history=history,
            message_text=message_text,
            metadata=metadata,
            decision=decision,
            raw_path=raw_path,
            config=config,
        )
        _atomic_write_text(wiki_path, markdown, mode=0o644)
    except Exception as exc:
        wiki_error = f"wiki handoff blocked: {type(exc).__name__}: {exc}"
        wiki_path = None
        if raw_path is None:
            raise ContextSpillWriteError(wiki_error) from exc

    manifest_path = manifest_dir / f"{spill_id}.json"
    manifest = {
        "spill_id": spill_id,
        "timestamp": timestamp,
        "session_id": old_session_id,
        "session_key": session_key,
        "wiki_path": str(wiki_path) if wiki_path else None,
        "raw_path": str(raw_path) if raw_path else None,
        "reason": decision.reason,
        "stage": decision.stage,
        "wiki_error": wiki_error,
    }
    try:
        _atomic_write_json(manifest_path, manifest, mode=0o600)
    except Exception as exc:
        raise ContextSpillWriteError(f"manifest write failed: {type(exc).__name__}: {exc}") from exc

    recovery_message = _build_recovery_message(
        spill_id=spill_id,
        wiki_path=wiki_path,
        raw_path=raw_path,
        old_session_id=old_session_id,
        reason=decision.reason,
        current_ask=message_text,
    )
    if wiki_error and not wiki_path:
        recovery_message += "\n[System note: wiki handoff was blocked by redaction/path safety; use the raw local bundle only if explicitly needed and do not paste it wholesale.]"
        recovery_message = recovery_message[:_MAX_RECOVERY_MESSAGE_CHARS]
    user_notice = _build_user_notice(wiki_path, raw_path) if config.notify_user else ""

    return ContextSpillResult(
        spill_id=spill_id,
        wiki_path=wiki_path,
        raw_path=raw_path,
        manifest_path=manifest_path,
        recovery_message=recovery_message,
        user_notice=user_notice,
        decision=decision,
    )


def request_pressure_from_api_kwargs(api_kwargs: dict[str, Any], *, context_length: int) -> dict[str, int | bool]:
    """Estimate final provider request pressure from built API kwargs."""
    messages = api_kwargs.get("messages") if isinstance(api_kwargs, dict) else []
    if not isinstance(messages, list):
        messages = api_kwargs.get("input") if isinstance(api_kwargs, dict) else []
    if not isinstance(messages, list):
        messages = []
    tools = api_kwargs.get("tools") if isinstance(api_kwargs, dict) else None
    system_prompt = ""
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        system_prompt = str(messages[0].get("content") or "")
    try:
        from agent.model_metadata import estimate_request_tokens_rough
        approx_tokens = estimate_request_tokens_rough(
            messages,
            system_prompt=system_prompt,
            tools=tools if isinstance(tools, list) else None,
        )
    except Exception:
        approx_tokens = estimate_messages_tokens_rough(messages)
    try:
        request_bytes = len(json.dumps(api_kwargs, ensure_ascii=False, default=str).encode("utf-8", "ignore"))
    except Exception:
        request_bytes = len(str(api_kwargs).encode("utf-8", "ignore"))
    context_length = int(context_length or 200_000)
    # This is an agent-level final guard over the *actual* provider request,
    # not the gateway transcript-spill threshold.  Do not cap large-context
    # models at the historical 180k gateway spill default — that produces a
    # false "context exceeded" failure for models whose usable window is larger
    # (for example Codex-routed GPT-5.5 at ~272k).  The byte limit below still
    # catches real transport/request-buffer hazards.
    threshold_tokens = max(1, int(context_length * 0.90))
    too_large = approx_tokens >= threshold_tokens or request_bytes >= _DEFAULT_REQUEST_BYTE_LIMIT
    return {
        "approx_tokens": int(approx_tokens),
        "request_bytes": int(request_bytes),
        "context_length": int(context_length),
        "threshold_tokens": int(threshold_tokens),
        "too_large": bool(too_large),
    }
