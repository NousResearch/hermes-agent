"""Generated voice-only acknowledgement harness for turn-start room audio.

The generated acknowledgement path is deliberately isolated from gateway text,
session history, and the main agent run.  It accepts an injectable generator for
tests/production adapters, validates the candidate aggressively, runs ambient
voice policy, and degrades to silence on every failure.  There is no canned
fallback phrase.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
import re
import threading
import time

from gateway.ambient_voice_policy import AmbientVoicePolicy, VoiceContext as AmbientVoiceContext

AckMethod = Literal["generated", "silence"]
AckMode = Literal["generated", "off"]


class AckGenerator(Protocol):
    """Callable shape for bounded acknowledgement generation."""

    def __call__(self, prompt: str, *, timeout_ms: int) -> str:
        ...

_DEFAULT_TIMEOUT_MS = 1000
_DEFAULT_MAX_WORDS = 12
_DEFAULT_MAX_SPOKEN_CHARS = 120
_DEFAULT_MAX_SECONDS = 2
_GENERATOR_MAX_WORKERS = 4
_GENERATOR_EXECUTOR = ThreadPoolExecutor(max_workers=_GENERATOR_MAX_WORKERS, thread_name_prefix="generated-ack")
_GENERATOR_SLOTS = threading.BoundedSemaphore(_GENERATOR_MAX_WORKERS)
_AMBIENT_POLICY = AmbientVoicePolicy()

GENERATED_ACK_PROMPT_CONTRACT = """Generate a short natural spoken acknowledgement as Eon.

Hard requirements:
- Maximum {max_words} words.
- One sentence only.
- No canned acknowledgement phrases.
- No generic assistant status language.
- Do not promise completion, fixes, tests, tool use, or results.
- Do not mention tools unless the user's request is explicitly about tools.
- Never speak secrets, raw logs, code, file paths, URLs, or long content.
- Match Brenno's energy: direct, sharp, grounded, not corporate.
- This is room audio, not Discord text.
- Return only the acknowledgement text, or an empty string if no safe line exists.

User turn:
{user_message}
"""

_SECRET_RE = re.compile(
    r"(?ix)("
    r"\b(?:bearer|authorization)\b\s*[:=]?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}"
    r"|\b(?:api[_ -]?key|secret(?:[_ -]?key)?|password|passwd|pwd|token|access[_ -]?key|aws[_ -]?key)\b\s*(?:is|=|:)?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}"
    r"|\b(?:sk-[A-Za-z0-9._-]{6,}|gh[pousr]_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,})\b"
    r"|\b(?:xox[baprs]-(?:[A-Za-z0-9-]{10,}|\[REDACTED\])|hf_(?:[A-Za-z0-9]{10,}|\[REDACTED\])|glpat-(?:[A-Za-z0-9_-]{10,}|\[REDACTED\]))(?=$|\W)"
    r"|\bhk_(?:[A-Za-z0-9._-]{10,}|\[REDACTED\])(?=$|\W)"
    r"|\b[a-z]{2,}_(?:test|live|prod|secret|key)_(?:[A-Za-z0-9]{10,}|\[REDACTED\])(?=$|\W)"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{10,}\b"
    r")"
)
_RAW_PATH_RE = re.compile(
    r"(?<![\w@])(?:~|/(?:Users|tmp|var|private|Volumes|home|opt|etc|usr|mnt|workspace|workspaces))"
    r"(?:/[A-Za-z0-9._@%+=:,~-]+)+|\b[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\?)+"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_CODE_OR_LOG_RE = re.compile(
    r"(?ims)(```.*?```|Traceback \(most recent call last\):.*|^\s*(?:\$|>>>|FAILED|ERROR|WARN|WARNING|INFO|DEBUG|E\s+|File \"[^\"]+\", line \d+).*$)"
)
_MEDIA_TAG_RE = re.compile(r"MEDIA:\S+", re.IGNORECASE)
_FILE_REF_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|json|ya?ml|toml|md|txt|log|js|jsx|ts|tsx|css|html|sh|env)\b",
    re.IGNORECASE,
)
_ENV_SECRET_RE = re.compile(
    r"(?i)\b[A-Z][A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|ACCESS[_-]?KEY)\s*=\s*\S+"
)
_SECRET_KEYWORD_VALUE_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|secret(?:[_ -]?key)?|password|passwd|pwd|token|access[_ -]?key|private[_ -]?key|ssh[_ -]?key|passphrase|credential)\b\s*(?:(?:is|=|:)\s*)?\S+"
)
_PRIVATE_KEY_RE = re.compile(r"(?is)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----")
_PRIVATE_KEY_HEADER_RE = re.compile(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
_RELATIVE_PATH_RE = re.compile(r"(?<![\w.~-])(?:\.\.?/|[A-Za-z0-9_.-]+/)[^\s,;:)\]'\"]+(?:/[^\s,;:)\]'\"]+)*")
_UNFENCED_CODE_LINE_RE = re.compile(
    r"(?im)^\s*(?:def\s+\w+\s*\(|class\s+\w+|import\s+\w+|from\s+\S+\s+import\s+|function\s+\w+\s*\(|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*=|return\s+.+|if\s+.+:\s*$|for\s+.+:\s*$|while\s+.+:\s*$)"
)
_INLINE_CODE_RE = re.compile(
    r"(?im)(?:\b(?:print|console\.(?:log|error|warn)|require|eval|exec)\s*\(|\b\w+[\w.]*\s*=\s*[^\s,.;!?]+|[{};]|\b\w+\([^)]*\)|\b\w+\s*(?:\+|-|\*|/|==|!=|<=|>=|<|>)\s*\w+)"
)
_BACKTICK_TEXT_RE = re.compile(r"`[^`]+`")
_SQL_SNIPPET_RE = re.compile(r"(?i)\b(?:select\s+.+\s+from|insert\s+into|update\s+\w+\s+set|delete\s+from|create\s+table|drop\s+table)\b")
_SHELL_SNIPPET_RE = re.compile(
    r"(?im)^\s*(?:sudo\s+|uv\s+run\b|python\s+-m\b|pip\s+install\b|npm\s+\w+\b|pnpm\s+\w+\b|yarn\s+\w+\b|git\s+\w+\b|ls\s+-|cd\s+\S+|grep\s+|rg\s+|curl\s+|wget\s+|ssh\s+|scp\s+|docker\s+|kubectl\s+)"
)
_TIMESTAMP_LOG_RE = re.compile(
    r"(?im)^\s*(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?|\d{2}:\d{2}:\d{2}(?:\.\d+)?|\[[^\]]*(?:ERROR|WARN|WARNING|INFO|DEBUG)[^\]]*\])\s*(?:\[[^\]]+\]\s*)*(?:ERROR|WARN|WARNING|INFO|DEBUG)?\b.*$"
)
_BRACKET_LOG_RE = re.compile(r"(?im)^\s*(?:\[[^\]]+\]\s*)*\[(?:ERROR|WARN|WARNING|INFO|DEBUG)\]\s*.*$")
_SENTENCE_END_RE = re.compile(r"[.!?。！？]")
_WORD_RE = re.compile(r"[\w’'-]+", re.UNICODE)
_CANNED_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")

_CANNED_PHRASES = {
    "im here",
    "i am here",
    "got you",
    "gotcha",
    "im on it",
    "i am on it",
    "lets see",
    "sure i can help with that",
    "okay",
    "ok",
    "processing your request",
}
_GENERIC_PATTERNS = (
    re.compile(r"(?i)\b(?:happy to help|how can i help|assist you|your request|working on it|one moment|please wait)\b"),
    re.compile(r"(?i)\b(?:as an ai|assistant|processing|analyz(?:e|ing)|thinking)\b"),
)
_PROMISE_PATTERNS = (
    re.compile(r"(?i)\b(?:i\s+will|i['’]ll|i\s+am\s+going\s+to|i['’]m\s+going\s+to)\b"),
    re.compile(r"(?i)\b(?:done|fixed|implemented|completed|verified|tests?\s+pass(?:ed)?|run\s+the\s+tests|open(?:ed)?\s+a\s+pr)\b"),
)


def _clamp_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _normalize_phrase(text: str) -> str:
    normalized = str(text or "").lower().replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("'", "")
    return _CANNED_NORMALIZE_RE.sub(" ", normalized).strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _unsafe_raw_context_reason(text: str) -> str | None:
    original = str(text or "")
    checks: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("secret_like", _SECRET_RE),
        ("secret_like", _SECRET_KEYWORD_VALUE_RE),
        ("env_secret", _ENV_SECRET_RE),
        ("private_key", _PRIVATE_KEY_RE),
        ("private_key", _PRIVATE_KEY_HEADER_RE),
        ("jwt_like", _JWT_RE),
        ("raw_path", _RAW_PATH_RE),
        ("file_ref", _FILE_REF_RE),
        ("relative_path", _RELATIVE_PATH_RE),
        ("url", _URL_RE),
        ("media", _MEDIA_TAG_RE),
        ("code_or_log", _CODE_OR_LOG_RE),
        ("code_or_log", _UNFENCED_CODE_LINE_RE),
        ("code_or_log", _INLINE_CODE_RE),
        ("code_or_log", _BACKTICK_TEXT_RE),
        ("code_or_log", _SQL_SNIPPET_RE),
        ("code_or_log", _SHELL_SNIPPET_RE),
        ("code_or_log", _TIMESTAMP_LOG_RE),
        ("code_or_log", _BRACKET_LOG_RE),
    )
    for reason, pattern in checks:
        if pattern.search(original):
            return reason
    return None


def _sanitize_prompt_user_message(text: str, *, limit: int | None = None) -> str:
    """Return bounded provider-prompt context with no raw secrets/paths/logs.

    Candidate validation protects what we publish. This protects what we send to
    the acknowledgement generator itself: raw user turns may contain credentials,
    local paths, pasted logs, code, or media tags. Any unsafe evidence makes this
    fail closed to a coarse placeholder rather than a partially redacted prompt.
    """
    original = str(text or "")
    if _unsafe_raw_context_reason(original):
        return "User asked about technical content; sensitive details omitted."
    redacted = re.sub(r"\s+", " ", original).strip()
    if limit is not None and len(redacted) > limit:
        redacted = redacted[: limit - 1].rstrip() + "…"
    if not redacted:
        return "User asked for voice assistance."
    return redacted


def _safe_policy_metadata(decision: Any) -> dict[str, Any]:
    classifiers = getattr(decision, "classifiers", {}) or {}
    return {
        "allowed": bool(getattr(decision, "allowed", False)),
        "sanitized": bool(getattr(decision, "sanitized", False)),
        "truncated": bool(getattr(decision, "truncated", False)),
        "suppressed": bool(getattr(decision, "suppressed", False)),
        "reason_codes": [str(r) for r in (getattr(decision, "reasons", ()) or ()) if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(r))],
        "rule_profile": str(getattr(decision, "rule_profile", "living_room_default") or "living_room_default"),
        "classifiers": {
            "code": bool(classifiers.get("code")),
            "command_log": bool(classifiers.get("command_log")),
            "raw_path": bool(classifiers.get("raw_path")),
            "secret_like": bool(classifiers.get("secret_like")),
            "sensitive_topic": bool(classifiers.get("sensitive_topic")),
            "stack_trace": bool(classifiers.get("stack_trace")),
        },
    }


@dataclass(frozen=True)
class AckContext:
    user_message: str
    session_id: str | None = None
    platform: str | None = None
    chat_id: str | None = None
    channel_id: str | None = None
    thread_id: str | None = None
    source_message_id: str | None = None
    input_modality: str | None = None
    output_device: str | None = None
    voice_profile: str = "eon"
    timeout_ms: int = _DEFAULT_TIMEOUT_MS
    max_words: int = _DEFAULT_MAX_WORDS
    max_spoken_chars: int = _DEFAULT_MAX_SPOKEN_CHARS
    max_seconds: int = _DEFAULT_MAX_SECONDS
    context_window_chars: int = 500
    config_scope: str = "living_room_default"
    explicit_spoken_request: bool = False
    is_private_context: bool = False

    def bounded_user_message(self) -> str:
        limit = _clamp_int(self.context_window_chars, 500, minimum=80, maximum=2000)
        text = re.sub(r"\s+", " ", str(self.user_message or "")).strip()
        if len(text) > limit:
            return text[: limit - 1].rstrip() + "…"
        return text


@dataclass(frozen=True)
class GeneratedAckResult:
    text: str
    method: AckMethod
    reason: str | None = None
    elapsed_ms: int = 0
    policy: dict[str, Any] = field(default_factory=dict)


class GeneratedAckHarness:
    """Generate and validate one voice-only acknowledgement, or return silence."""

    def __init__(
        self,
        generator: AckGenerator | None = None,
        *,
        mode: str = "generated",
        ambient_policy: AmbientVoicePolicy | None = None,
    ) -> None:
        self.generator = generator
        self.mode = "off" if str(mode or "generated").strip().lower() == "off" else "generated"
        self.ambient_policy = ambient_policy or _AMBIENT_POLICY

    def build_prompt(self, context: AckContext) -> str:
        return GENERATED_ACK_PROMPT_CONTRACT.format(
            max_words=_clamp_int(context.max_words, _DEFAULT_MAX_WORDS, minimum=1, maximum=20),
            user_message=_sanitize_prompt_user_message(
                context.user_message,
                limit=_clamp_int(context.context_window_chars, 500, minimum=80, maximum=2000),
            ),
        )

    def generate(self, context: AckContext) -> GeneratedAckResult:
        started = time.perf_counter()

        def silence(reason: str, policy: dict[str, Any] | None = None) -> GeneratedAckResult:
            return GeneratedAckResult(
                text="",
                method="silence",
                reason=reason,
                elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
                policy=policy or {},
            )

        if self.mode == "off":
            return silence("off")
        if self.generator is None:
            return silence("no_generator")

        timeout_ms = _clamp_int(context.timeout_ms, _DEFAULT_TIMEOUT_MS, minimum=1, maximum=1500)
        prompt = self.build_prompt(context)
        if not _GENERATOR_SLOTS.acquire(blocking=False):
            return silence("generated_saturated")
        future = _GENERATOR_EXECUTOR.submit(self.generator, prompt, timeout_ms=timeout_ms)

        def release_slot(_future: Any) -> None:
            try:
                _GENERATOR_SLOTS.release()
            except ValueError:
                pass

        future.add_done_callback(release_slot)
        try:
            raw = future.result(timeout=timeout_ms / 1000)
        except TimeoutError:
            future.cancel()
            return silence("generated_timeout")
        except Exception:
            return silence("generated_error")

        candidate = str(raw or "").strip().strip('"“”')
        valid, reason = self.validate_generated_ack(candidate, context)
        if not valid:
            return silence(reason or "generated_invalid")

        decision = self.ambient_policy.evaluate(
            candidate,
            AmbientVoiceContext(
                source="ack",
                platform=context.platform,
                channel_id=context.channel_id,
                chat_id=context.chat_id,
                thread_id=context.thread_id,
                source_message_id=context.source_message_id,
                input_modality=context.input_modality,
                output_device=context.output_device,
                profile=context.voice_profile,
                explicit_spoken_request=context.explicit_spoken_request,
                is_private_context=context.is_private_context,
                config_scope=context.config_scope,
            ),
        )
        policy = _safe_policy_metadata(decision)
        if not getattr(decision, "allowed", False) or not str(getattr(decision, "text", "") or "").strip():
            return silence("ambient_policy_denied", policy)

        approved = str(decision.text).strip()
        return GeneratedAckResult(
            text=approved,
            method="generated",
            reason=None,
            elapsed_ms=max(0, int((time.perf_counter() - started) * 1000)),
            policy=policy,
        )

    def validate_generated_ack(self, candidate: str, context: AckContext) -> tuple[bool, str | None]:
        text = str(candidate or "").strip()
        if not text:
            return False, "generated_invalid: empty"
        max_chars = _clamp_int(context.max_spoken_chars, _DEFAULT_MAX_SPOKEN_CHARS, minimum=1, maximum=240)
        if len(text) > max_chars:
            return False, "generated_invalid: too_many_chars"
        if _word_count(text) > _clamp_int(context.max_words, _DEFAULT_MAX_WORDS, minimum=1, maximum=20):
            return False, "generated_invalid: too_many_words"
        # Multiple sentence terminators usually means it is no longer an acknowledgement.
        if len(_SENTENCE_END_RE.findall(text)) > 1:
            return False, "generated_invalid: multiple_sentences"
        normalized = _normalize_phrase(text)
        if normalized in _CANNED_PHRASES:
            return False, "generated_invalid: canned_phrase"
        if any(pattern.search(text) for pattern in _GENERIC_PATTERNS):
            return False, "generated_invalid: generic_assistant_language"
        if any(pattern.search(text) for pattern in _PROMISE_PATTERNS):
            return False, "generated_invalid: completion_promise"
        if _SECRET_RE.search(text):
            return False, "generated_invalid: secret_like"
        unsafe_reason = _unsafe_raw_context_reason(text)
        if unsafe_reason:
            return False, f"generated_invalid: {unsafe_reason}"
        return True, None
