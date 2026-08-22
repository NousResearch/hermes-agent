"""Telegram final-answer-first reporting mode.

Only Telegram presentation is changed. Tools, approvals, authorization, project
selection, credentials, and execution behavior remain owned by Hermes core.
"""

from __future__ import annotations

import contextvars
import math
import re
from typing import Any

_DEFAULT_MODE = "brief"
_MAX_BRIEF_LINES = 500
_MAX_BRIEF_CHARS = 30000
_MAX_FIELD_CHARS = 10000
_MAX_MEDIA = 10
_MAX_MEDIA_PATH_CHARS = 300
_MAX_MEDIA_PATH_TOTAL = 1200
_MAX_SERIALIZED_CHARS = 32000

_modes_by_identity: dict[str, str] = {}
_turn_identity: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "telegram_brief_turn_identity", default=None
)
_turn_allows_detail: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "telegram_brief_turn_allows_detail", default=False
)
_final_output_allows_detail: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "telegram_brief_final_output_allows_detail", default=False
)

_BRIEF_CONTEXT = """[TELEGRAM REPORT MODE: BRIEF]
This instruction controls only the final user-facing Telegram response; retain and use every normal agent capability, tool, approval, permission, secret rule, and project rule.
Do not narrate progress. Do not expose reasoning, tool calls, tool output, assistant deltas, commands, diffs, logs, stack traces, or file-by-file edits. Do not include complete code or a code fence.
Return only the final conclusion, normally within 10 lines and in Traditional Chinese when the user writes Chinese. Use this schema when it fits:
結果：完成／部分完成／失敗
變更：1～3 sentences
驗證：tests, build, and real inspection results
交付：commit, push, and deploy state (omit when none)
阻礙：only when a real blocker exists
下一步：only when the user must decide or act
On failure, include only the key error, root cause, impact, and next step; keep full logs locally and provide only their path. Do not make another model call merely to summarize.
Never replace an explicitly requested answer, value, list, error reason, or verification result with a status word or an omission notice. If the user asks to list information, include the requested list in the final response. Concision may remove process narration and duplicate logs, never the answer itself.
"""

_DETAIL_CONTEXT = """[TELEGRAM REPORT MODE: DETAIL]
A detailed final technical explanation is allowed for this turn. Never expose private reasoning or hidden chain-of-thought. Do not stream tool calls, tool output, assistant deltas, or progress narration; send technical detail only in the final response.
"""

# Exact, whole-message requests only. Analytical or quoted text with a trigger
# phrase must not silently disable brief sanitization.
_DIRECT_DETAIL_RE = re.compile(
    r"^\s*(?:"
    r"(?:請|麻煩|這次請)\s*(?:貼(?:出)?程式碼|顯示\s*diff|提供(?:完整)?程式碼|"
    r"提供完整(?:錯誤|log|日誌|技術說明)|詳細說明)|"
    r"(?:please\s+)?(?:show|paste)\s+(?:the\s+)?(?:code|diff)|"
    r"(?:please\s+)?provide\s+(?:the\s+)?(?:full\s+)?(?:code|error|log|stack\s*trace)|"
    r"(?:please\s+)?(?:give|write)\s+(?:a\s+)?detailed\s+(?:technical\s+)?explanation"
    r")\s*(?:please|給我|for\s+this\s+turn)?\s*[.!。！]?\s*$",
    re.IGNORECASE,
)
_FIELD_RE = re.compile(r"^(結果|變更|驗證|交付|阻礙|下一步)[：:]\s*(.*)$")
_FIELD_ORDER = ("結果", "變更", "驗證", "交付", "阻礙", "下一步")
_REQUIRED_TELEGRAM_DISPLAY = {
    "streaming": False,
    "tool_progress": "off",
    "show_reasoning": False,
    "thinking_progress": False,
    "interim_assistant_messages": False,
    "busy_ack_detail": False,
    "long_running_notifications": False,
}
_UNSAFE_VALUE_RE = re.compile(
    r"MEDIA:|```|~~~|diff\s+--git|traceback|stack\s*trace|tool\s*(?:call|output|result)|"
    r"assistant\s*delta|private\s*reasoning|思考過程|"
    r"(?:^|\s)(?:def|class|function|import|from|const|let|var)\s+[A-Za-z_$]|"
    r"\b[A-Za-z_$][\w$]*\([^\n)]*\)|"
    r"(?:^|\s)(?:git|python|npm|pnpm|yarn|curl|powershell|cmd\.exe|bash|pytest)\s+[-/A-Za-z]|"
    r"=>|@@\s|\b(?:secret|token|password|credential|api[_ -]?key|private[_ -]?key)\b|"
    r"(?:密碼|權杖|憑證|私鑰|金鑰)",
    re.IGNORECASE,
)
_CREDENTIAL_SHAPED_RE = re.compile(
    r"(?:\bBearer\s+\S{16,}|"
    r"\b(?:sk-(?:proj-)?|gh[pousr]_|xox[baprs]-|hf_)[A-Za-z0-9_-]{16,}|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}|"
    r"\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*\S{8,})",
    re.IGNORECASE,
)
_OPAQUE_TOKEN_RE = re.compile(r"\S{32,}")


def _has_high_entropy_token(value: str) -> bool:
    for match in _OPAQUE_TOKEN_RE.finditer(value):
        # Evaluate the credential material across arbitrary non-whitespace
        # delimiters so attackers cannot split a token into sub-threshold
        # slash/colon/pipe/etc. segments. Punctuation itself contributes no
        # entropy and therefore cannot manufacture a false positive.
        candidate = re.sub(r"[^A-Za-z0-9]", "", match.group(0))
        if len(candidate) < 32:
            continue
        counts = {char: candidate.count(char) for char in set(candidate)}
        entropy = -sum(
            (count / len(candidate)) * math.log2(count / len(candidate))
            for count in counts.values()
        )
        if len(counts) >= 10 and entropy >= 3.5:
            return True
    return False


def _contains_sensitive_material(value: str) -> bool:
    value = str(value or "")
    # Evaluate the exact character stream controls could collapse into at any
    # later output-normalization boundary, including detail-mode callers.
    screened = re.sub(r"[\x00-\x1f\x7f]", "", value)
    if _CREDENTIAL_SHAPED_RE.search(screened) or _has_high_entropy_token(screened):
        return True
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(screened, force=True) != screened
    except Exception:
        # Secret screening is part of the security boundary.
        return True


# Natural-language summaries only: reject code delimiters/operators, shell
# quoting, Windows paths, and serialized structures instead of guessing.
_SAFE_VALUE_RE = re.compile(
    r"^[\w\s\u3000-\u303f，。！？：；、（）「」『』《》〈〉…·％%＋+\-－／/：:,.!?()'\";–—’“”|*]+$",
    re.UNICODE,
)
_MARKDOWN_PREFIX_RE = re.compile(r"^(?:[-*•]\s+|\d+[.)、]\s*|>\s*|\|\s*)")
_TOKEN_FRAGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{12,}$")
_UNSAFE_ANSWER_LINE_RE = re.compile(
    r"^(?:return\b|RuntimeError\s*:|KeyError\s*:|REASONING\s*:)|"
    r"internal\s+stack|\blog\s+output\b|(?:^|\s)[A-Za-z]:[/\\](?:private|secret)(?:[/\\]|$)",
    re.IGNORECASE,
)


def _platform_name(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _identity_from_source(source: Any) -> str:
    platform = _platform_name(getattr(source, "platform", None))
    profile = str(getattr(source, "profile", None) or "default")
    user = getattr(source, "user_id", None) or getattr(source, "user_id_alt", None)
    if user not in (None, ""):
        return f"{profile}:{platform}:user:{user}"
    chat = str(getattr(source, "chat_id", None) or "unknown-chat")
    thread = str(getattr(source, "thread_id", None) or "root")
    return f"{profile}:{platform}:chat:{chat}:thread:{thread}"


def _identity_for_turn(sender_id: Any = None) -> str:
    identity = _turn_identity.get()
    if identity:
        return identity
    if sender_id not in (None, ""):
        return f"default:telegram:user:{sender_id}"
    return "default:telegram:chat:unknown-chat:thread:root"


def _mode_for_identity(identity: str) -> str:
    return _modes_by_identity.get(str(identity), _DEFAULT_MODE)


def _is_explicit_detail_request(text: str) -> bool:
    return bool(_DIRECT_DETAIL_RE.fullmatch(str(text or "")))


def _transport_is_safe(source: Any, gateway: Any) -> bool:
    """Read-only, profile-scoped fail-closed transport verification."""
    try:
        from gateway.run import _profile_runtime_scope
        from hermes_cli.config import load_config

        profile_home = gateway._resolve_profile_home_for_source(source)
        with _profile_runtime_scope(profile_home):
            config = load_config()
        from gateway.runtime_footer import resolve_footer_config

        if resolve_footer_config(config, "telegram").get("enabled"):
            return False
        actual = (((config.get("display") or {}).get("platforms") or {}).get("telegram") or {})
        for key, required in _REQUIRED_TELEGRAM_DISPLAY.items():
            current = actual.get(key)
            if key == "tool_progress":
                if current not in (False, "off"):
                    return False
            elif current is not required and current != required:
                return False
        return True
    except Exception:
        return False


def _on_pre_gateway_dispatch(**kwargs: Any) -> dict[str, str] | None:
    event = kwargs.get("event")
    source = getattr(event, "source", None)
    if _platform_name(getattr(source, "platform", None)) != "telegram":
        _turn_identity.set(None)
        _turn_allows_detail.set(False)
        return None
    gateway = kwargs.get("gateway")
    if not _transport_is_safe(source, gateway):
        _turn_identity.set(None)
        _turn_allows_detail.set(False)
        return {"action": "skip", "reason": "Telegram brief transport settings are unsafe"}
    _turn_identity.set(_identity_from_source(source))
    return None


def _command_identity(event: Any) -> str | None:
    source = getattr(event, "source", None)
    if _platform_name(getattr(source, "platform", None)) != "telegram":
        return None
    return _identity_from_source(source)


def _handle_brief(_args: str, *, event: Any = None, **_kwargs: Any) -> str:
    identity = _command_identity(event)
    if identity is None:
        return "此指令僅適用於 Telegram。"
    _modes_by_identity[identity] = "brief"
    return "回報模式：brief（Telegram 僅傳送精簡最終結論）"


def _handle_detail(_args: str, *, event: Any = None, **_kwargs: Any) -> str:
    identity = _command_identity(event)
    if identity is None:
        return "此指令僅適用於 Telegram。"
    _modes_by_identity[identity] = "detail"
    return "回報模式：detail（允許完整最終技術說明；仍不顯示私人 reasoning）"


def _extend_gateway_status(**kwargs: Any) -> str | None:
    event = kwargs.get("event")
    source = getattr(event, "source", None)
    if _platform_name(getattr(source, "platform", None)) != "telegram":
        return None
    status = str(kwargs.get("status") or "")
    mode = _mode_for_identity(_identity_from_source(source))
    return f"{status}\n回報模式：{mode}" if status else f"回報模式：{mode}"


def _on_pre_llm_call(**kwargs: Any) -> dict[str, str] | None:
    if _platform_name(kwargs.get("platform")) != "telegram":
        _turn_allows_detail.set(False)
        return None
    identity = _identity_for_turn(kwargs.get("sender_id"))
    explicit_detail = _is_explicit_detail_request(str(kwargs.get("user_message") or ""))
    allow_detail = explicit_detail or _mode_for_identity(identity) == "detail"
    _turn_allows_detail.set(allow_detail)
    return {"context": _DETAIL_CONTEXT if allow_detail else _BRIEF_CONTEXT}


def _safe_field_value(value: str) -> str:
    value = str(value or "")
    # Screen the exact normalized value that can be emitted. Otherwise removed
    # control characters could join individually short fragments into a secret
    # only after the security check had already accepted them.
    value = re.sub(r"[\x00-\x1f\x7f]", "", value)
    value = re.sub(r"\s+", " ", value).strip().replace("`", "")
    if _contains_sensitive_material(value):
        return "敏感內容已遮蔽。"
    if not value or _UNSAFE_VALUE_RE.search(value) or not _SAFE_VALUE_RE.fullmatch(value):
        return "不安全內容已遮蔽。" if value else ""
    return value[: _MAX_FIELD_CHARS - 1] + "…" if len(value) > _MAX_FIELD_CHARS else value


def _normalized_secret_stream(lines: list[str]) -> str:
    """Join only credential-like line fragments for cross-line screening."""
    fragments: list[str] = []
    for line in lines:
        candidate = _MARKDOWN_PREFIX_RE.sub("", line.strip())
        candidate = candidate.strip("|* `")
        if _TOKEN_FRAGMENT_RE.fullmatch(candidate):
            fragments.append(candidate)
    return "".join(fragments)


def _sanitize_brief_response(text: str) -> str:
    original = str(text or "")
    fields: dict[str, str] = {}
    answer_lines: list[str] = []
    blocked = {"敏感內容已遮蔽。", "不安全內容已遮蔽。"}
    for raw_line in original.splitlines():
        stripped = raw_line.strip()
        match = _FIELD_RE.match(stripped)
        if match and match.group(1) not in fields:
            value = _safe_field_value(match.group(2))
            if value:
                fields[match.group(1)] = value
        elif (
            stripped
            and not stripped.startswith(("MEDIA:", "[[audio_as_voice]]", "[[as_document]]"))
            and not _UNSAFE_ANSWER_LINE_RE.search(stripped)
        ):
            value = _safe_field_value(stripped)
            if value and value not in blocked:
                answer_lines.append(value)

    secret_stream = _normalized_secret_stream(answer_lines)
    if secret_stream and _contains_sensitive_material(secret_stream):
        answer_lines = ["敏感內容已遮蔽。"]

    if fields:
        lines = [f"{name}：{fields[name]}" for name in _FIELD_ORDER if name in fields]
        lines.extend(answer_lines)
    else:
        # Brief is a presentation preference, not a schema gate. Preserve every
        # safe answer when the model used prose, values, lists, or tables.
        lines = answer_lines
    lines = lines[:_MAX_BRIEF_LINES]
    while lines and len("\n".join(lines)) > _MAX_BRIEF_CHARS:
        lines.pop()
    summary = "\n".join(lines) or "結果：無法安全顯示\n變更：回覆未包含可安全傳送的文字。"

    # Canonical parser is the sole media syntax allowlist. Bound paths, count,
    # aggregate size, and final serialized size before reconstructing tags.
    bounded: list[tuple[str, bool]] = []
    path_total = 0
    try:
        from gateway.platforms.base import BasePlatformAdapter

        media_files, _ = BasePlatformAdapter.extract_media(original[:20000])
        for path, is_voice in media_files:
            path = str(path)
            if (
                not path
                or len(path) > _MAX_MEDIA_PATH_CHARS
                or re.search(r"[\x00-\x1f\x7f]", path)
                or _contains_sensitive_material(path)
            ):
                continue
            if len(bounded) >= _MAX_MEDIA or path_total + len(path) > _MAX_MEDIA_PATH_TOTAL:
                break
            bounded.append((path, bool(is_voice)))
            path_total += len(path)
    except Exception:
        bounded = []

    if not bounded:
        return summary
    media_lines: list[str] = []
    if any(is_voice for _, is_voice in bounded):
        media_lines.append("[[audio_as_voice]]")
    if "[[as_document]]" in original:
        media_lines.append("[[as_document]]")
    for path, _ in bounded:
        candidate = media_lines + [f"MEDIA:{path}"]
        if len(summary) + 1 + len("\n".join(candidate)) > _MAX_SERIALIZED_CHARS:
            break
        media_lines = candidate
    return summary + (("\n" + "\n".join(media_lines)) if any(line.startswith("MEDIA:") for line in media_lines) else "")


def _transform_llm_output(**kwargs: Any) -> str | None:
    if _platform_name(kwargs.get("platform")) != "telegram":
        _turn_allows_detail.set(False)
        _final_output_allows_detail.set(False)
        return None
    allow_detail = _turn_allows_detail.get()
    _turn_allows_detail.set(False)
    _final_output_allows_detail.set(allow_detail)
    if allow_detail:
        return None
    original = str(kwargs.get("response_text") or "")
    transformed = _sanitize_brief_response(original)
    return transformed if transformed != original else None


def _validate_llm_output(**kwargs: Any) -> bool:
    """Terminally validate the immutable result after every finalizer callback."""
    if _platform_name(kwargs.get("platform")) != "telegram":
        _final_output_allows_detail.set(False)
        return True
    allow_detail = _final_output_allows_detail.get()
    _final_output_allows_detail.set(False)
    output = str(kwargs.get("response_text") or "")
    if allow_detail:
        if _contains_sensitive_material(output):
            raise RuntimeError("Telegram terminal validation rejected sensitive detail output")
        return True
    if _sanitize_brief_response(output) != output:
        raise RuntimeError("Telegram terminal validation rejected post-sanitizer output")
    return True


def _gateway_allows_detail(**kwargs: Any) -> bool:
    if kwargs.get("force_brief"):
        return False
    source = kwargs.get("source")
    if _platform_name(getattr(source, "platform", kwargs.get("platform"))) != "telegram":
        return False
    identity = _identity_from_source(source)
    return _mode_for_identity(identity) == "detail" or _is_explicit_detail_request(
        str(kwargs.get("user_message") or "")
    )


def _revalidate_gateway_transport(**kwargs: Any) -> None:
    source = kwargs.get("source")
    gateway = kwargs.get("gateway")
    if gateway is not None and not _transport_is_safe(source, gateway):
        raise RuntimeError("Telegram gateway transport configuration became unsafe")


def _transform_gateway_output(**kwargs: Any) -> str | None:
    """Re-sanitize after every gateway mutation at the true send boundary."""
    if _platform_name(kwargs.get("platform")) != "telegram":
        return None
    _revalidate_gateway_transport(**kwargs)
    if _gateway_allows_detail(**kwargs):
        return None
    original = str(kwargs.get("response_text") or "")
    transformed = _sanitize_brief_response(original)
    return transformed if transformed != original else None


def _validate_gateway_output(**kwargs: Any) -> bool:
    if _platform_name(kwargs.get("platform")) != "telegram":
        return True
    _revalidate_gateway_transport(**kwargs)
    output = str(kwargs.get("response_text") or "")
    if _gateway_allows_detail(**kwargs):
        if _contains_sensitive_material(output):
            raise RuntimeError("Telegram gateway validation rejected sensitive detail output")
        return True
    if _sanitize_brief_response(output) != output:
        raise RuntimeError("Telegram gateway validation rejected noncanonical output")
    return True


def _reset_state_for_tests() -> None:
    _modes_by_identity.clear()
    _turn_identity.set(None)
    _turn_allows_detail.set(False)
    _final_output_allows_detail.set(False)


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_transport", _on_pre_gateway_dispatch)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("finalize_llm_output", _transform_llm_output)
    ctx.register_hook("validate_llm_output", _validate_llm_output)
    ctx.register_hook("finalize_gateway_output", _transform_gateway_output)
    ctx.register_hook("validate_gateway_output", _validate_gateway_output)
    ctx.register_hook("extend_gateway_status", _extend_gateway_status)
    ctx.register_command("brief", _handle_brief, description="Use concise final-only Telegram reporting.")
    ctx.register_command("detail", _handle_detail, description="Allow detailed final Telegram explanations.")
