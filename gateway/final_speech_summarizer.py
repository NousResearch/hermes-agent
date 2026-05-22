"""Final assistant response to room-safe speech summarization.

The summarizer deliberately keeps deterministic safety gates around any generated
summary.  Generation is optional and injectable so tests never need network/model
calls; production callers can wire a fast local/auxiliary summarizer later without
changing the public ``summarize(final_response, context)`` shape.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from typing import Callable, Literal
import re

_MAX_TEXT_CHARS = 180

Kind = Literal["completion", "question", "error"]
SummaryMethod = Literal["generated", "deterministic", "silence"]
SummaryMode = Literal["deterministic", "generated", "hybrid", "off"]

_MEDIA_RE = re.compile(r"MEDIA:\S+", re.IGNORECASE)
_DIRECTIVE_RE = re.compile(r"\[\[[^\]]+\]\]")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]{1,120})`")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_TOOL_LOG_LINE_RE = re.compile(
    r'(?m)^\s*(?:\$ .+|>>> .+|FAILED .+|Traceback .+|File "[^"]+", line \d+.*|E\s+.+|={5,}.*)$'
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/][^\s,;:)\]'\"]+|(?:~?/|/)[^\s,;:)\]'\"]+)")
_FILE_REF_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|json|ya?ml|toml|md|txt|log|js|jsx|ts|tsx|css|html|sh|env)\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"""
    (?:
        \b(?:bearer|authorization)\b\s*[:=]?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}
      | \b(?:api[_ -]?key|secret(?:[_ -]?key)?|password|passwd|pwd|token|access[_ -]?key|aws[_ -]?key)\b
        \s*(?:is|=|:|starts\s+with)?\s+[A-Za-z0-9][A-Za-z0-9._~+/\-]{7,}
      | \b(?:sk-[A-Za-z0-9._-]{6,}|sk-[A-Za-z0-9]{2,}\.\.\.[A-Za-z0-9]{2,}|gh[pousr]_[A-Za-z0-9_]{10,})\b
      | \b(?:AKIA|ASIA)[A-Z0-9.]{10,}\b
      | \bhk_[A-Za-z0-9._-]{10,}\b
      | \b[a-z]{2,}_(?:test|live|prod|secret|key)_[A-Za-z0-9]{10,}\b
      | \beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:]\d+)?\b")
_SENTENCE_RE = re.compile(r"^.{1,150}?[.!?。！？](?=\s|$)")
_FUTURE_PROMISE_RE = re.compile(
    r"(?i)\b(?:i\s+will|i['’]ll|i\s+am\s+going\s+to|i['’]m\s+going\s+to)\b"
)
_ERROR_SIGNAL_RE = re.compile(
    r"(?i)\b(?:error|failed?|failure|timed\s+out|timeout|blocked|can't|cannot|couldn't|unable|unavailable|interrupted|sorry)\b"
)
_ACTION_TERMS = {
    "deployed",
    "deploy",
    "sent",
    "closed",
    "opened",
    "merged",
    "created",
    "deleted",
    "fixed",
    "updated",
    "changed",
    "implemented",
    "ran",
    "passed",
    "verified",
}
_COMPLETION_VERBS = {
    "tests pass",
    "tests passed",
    "ran tests",
    "ran the tests",
    "files changed",
    "changed files",
    "pr opened",
    "opened a pr",
    "deployment complete",
    "deployed",
    "issue closed",
    "closed the issue",
    "message sent",
    "sent the message",
}

FINAL_SPEECH_PROMPT_CONTRACT = """You are Eon speaking aloud in Brenno's living room.
Convert the final assistant response into one short spoken line.

Hard requirements:
- Use only facts explicitly present in the final response.
- Do not add new actions, results, numbers, file names, tests, statuses, or promises.
- Preserve the most important user-facing outcome: done, blocked, needs review, error, or question.
- If tests, review, blockers, or created tasks are central in the final response, keep that fact.
- Do not speak secrets, tokens, credentials, raw paths, code, stack traces, logs, MEDIA tags, or platform directives.
- Do not mention internal summarization, policies, prompts, tools, or JSONL.
- No canned assistant phrases. No “as an AI”. No corporate support voice.
- One natural sentence, preferably 8-18 words, maximum {max_chars} characters.
- If no safe aligned summary exists, return an empty string.

Style:
- Calm, direct, grounded.
- Eon, not a notification bot.
- One breath for a room, not a Discord message.

Final assistant response:
{final_response}
"""


@dataclass(frozen=True)
class VoiceContext:
    session_id: str | None = None
    platform: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    source_message_id: str | None = None
    voice_profile: str = "eon"
    max_spoken_chars: int = _MAX_TEXT_CHARS
    max_seconds: int = 4
    timeout_ms: int = 1000
    room_context: str = "living_room"


@dataclass(frozen=True)
class VoiceSummaryResult:
    kind: Kind
    text: str
    method: SummaryMethod
    policy: dict[str, bool] = field(default_factory=dict)
    reason: str | None = None


@dataclass(frozen=True)
class _Sanitized:
    text: str
    policy: dict[str, bool]


def _base_policy() -> dict[str, bool]:
    return {
        "pre_sanitized": True,
        "post_sanitized": True,
        "truncated": False,
        "blocked_sensitive_content": False,
        "dropped_tool_logs": False,
        "dropped_code": False,
        "dropped_media_tags": False,
        "dropped_paths": False,
    }


def _first_sentence(text: str) -> str:
    match = _SENTENCE_RE.match(text)
    return match.group(0) if match else text


def sanitize_voice_text(
    text: str,
    *,
    max_chars: int = _MAX_TEXT_CHARS,
    one_sentence: bool = True,
) -> _Sanitized:
    """Strip material that is unsafe or awkward for room audio."""
    raw = str(text or "")
    policy = _base_policy()
    policy["dropped_code"] = bool(_CODE_FENCE_RE.search(raw) or _INLINE_CODE_RE.search(raw))
    policy["dropped_media_tags"] = bool(_MEDIA_RE.search(raw))
    policy["dropped_tool_logs"] = bool(_TOOL_LOG_LINE_RE.search(raw))
    policy["blocked_sensitive_content"] = bool(_SECRET_RE.search(raw))
    policy["dropped_paths"] = bool(_PATH_RE.search(raw) or _FILE_REF_RE.search(raw))

    cleaned = _CODE_FENCE_RE.sub("", raw)
    cleaned = _TOOL_LOG_LINE_RE.sub("", cleaned)
    cleaned = _MEDIA_RE.sub("", cleaned)
    cleaned = _DIRECTIVE_RE.sub("", cleaned)
    cleaned = _URL_RE.sub("", cleaned)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub("", cleaned)
    cleaned = _PATH_RE.sub("", cleaned)
    cleaned = _FILE_REF_RE.sub("", cleaned)
    cleaned = _SECRET_RE.sub("", cleaned)
    cleaned = re.sub(r"^[#>*\-•\s]+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or policy["blocked_sensitive_content"]:
        return _Sanitized("", policy)

    if one_sentence:
        cleaned = _first_sentence(cleaned).strip()
    max_chars = max(1, int(max_chars or _MAX_TEXT_CHARS))
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip(" ,.;:") + "…"
        policy["truncated"] = True
    return _Sanitized(cleaned, policy)


def classify_final_response(final_response: str) -> Kind:
    safe = sanitize_voice_text(final_response).text
    lowered = safe.lower()
    raw_lowered = str(final_response or "").strip().lower()
    if lowered.startswith(("⚠", "error", "failed", "sorry")) or raw_lowered.startswith(
        ("⚠", "error", "failed", "sorry")
    ):
        return "error"
    if "?" in safe and len(safe) <= 160:
        return "question"
    return "completion"


class FinalSpeechSummarizer:
    """Hybrid final-speech summarizer with deterministic safety fallback."""

    def __init__(
        self,
        generator: Callable[[str], str] | Callable[..., str] | None = None,
        *,
        mode: SummaryMode = "hybrid",
    ) -> None:
        self.generator = generator
        self.mode = mode

    def summarize(self, final_response: str, context: VoiceContext | None = None) -> VoiceSummaryResult:
        context = context or VoiceContext()
        kind = classify_final_response(final_response)
        if self.mode == "off":
            return VoiceSummaryResult(kind=kind, text="", method="silence", policy=_base_policy(), reason="off")

        pre = sanitize_voice_text(
            final_response,
            max_chars=max(context.max_spoken_chars * 4, context.max_spoken_chars),
            one_sentence=False,
        )
        if not pre.text:
            return VoiceSummaryResult(
                kind=kind,
                text="",
                method="silence",
                policy=pre.policy,
                reason="empty_safe_output",
            )

        if self.mode in {"generated", "hybrid"} and self.generator is not None:
            generated, failure_reason = self._try_generated(pre.text, final_response, context)
            if generated is not None:
                return VoiceSummaryResult(kind=kind, text=generated.text, method="generated", policy=generated.policy)
            fallback = self._deterministic(pre.text, context)
            return VoiceSummaryResult(
                kind=kind,
                text=fallback.text,
                method="deterministic" if fallback.text else "silence",
                policy={**pre.policy, **fallback.policy},
                reason=failure_reason if fallback.text else "empty_safe_output",
            )

        fallback = self._deterministic(pre.text, context)
        return VoiceSummaryResult(
            kind=kind,
            text=fallback.text,
            method="deterministic" if fallback.text else "silence",
            policy={**pre.policy, **fallback.policy},
            reason=None if fallback.text else "empty_safe_output",
        )

    def _deterministic(self, safe_final_response: str, context: VoiceContext) -> _Sanitized:
        return sanitize_voice_text(safe_final_response, max_chars=context.max_spoken_chars)

    def _try_generated(
        self, safe_final_response: str, original_final_response: str, context: VoiceContext
    ) -> tuple[_Sanitized | None, str | None]:
        prompt = FINAL_SPEECH_PROMPT_CONTRACT.format(
            final_response=safe_final_response,
            max_chars=context.max_spoken_chars,
        )
        timeout_ms = max(1, int(context.timeout_ms or 1))
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._call_generator, prompt, timeout_ms)
        try:
            raw = future.result(timeout=timeout_ms / 1000)
        except TimeoutError:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            return None, "generated_timeout"
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            return None, "generated_exception"
        finally:
            if future.done():
                executor.shutdown(wait=False, cancel_futures=True)
        valid, reason = self.validate_generated_summary(raw, original_final_response, context)
        if not valid:
            return None, f"generated_invalid: {reason}"
        sanitized = sanitize_voice_text(raw, max_chars=context.max_spoken_chars)
        if not sanitized.text:
            return None, "generated_invalid: empty"
        return sanitized, None

    def _call_generator(self, prompt: str, timeout_ms: int) -> str:
        if self.generator is None:
            return ""
        try:
            return str(self.generator(prompt, timeout_ms=timeout_ms))  # type: ignore[misc]
        except TypeError:
            return str(self.generator(prompt))  # type: ignore[misc]

    def validate_generated_summary(
        self, generated_summary: str, final_response: str, context: VoiceContext | None = None
    ) -> tuple[bool, str | None]:
        context = context or VoiceContext()
        raw = str(generated_summary or "")
        if not raw.strip():
            return False, "empty"
        if len(raw.strip()) > context.max_spoken_chars:
            return False, "too_long"
        if _CODE_FENCE_RE.search(raw) or _INLINE_CODE_RE.search(raw):
            return False, "code"
        if _MEDIA_RE.search(raw):
            return False, "media"
        if _URL_RE.search(raw):
            return False, "url"
        if _PATH_RE.search(raw):
            return False, "path"
        if _FILE_REF_RE.search(raw):
            return False, "file_ref"
        if _SECRET_RE.search(raw):
            return False, "secret"
        if _TOOL_LOG_LINE_RE.search(raw):
            return False, "logs"
        if _DIRECTIVE_RE.search(raw):
            return False, "directive"

        safe_generated = sanitize_voice_text(raw, max_chars=context.max_spoken_chars)
        if not safe_generated.text:
            return False, "empty"
        safe_final = sanitize_voice_text(
            final_response,
            max_chars=max(context.max_spoken_chars * 8, 1000),
            one_sentence=False,
        )
        final_text = safe_final.text.lower()
        generated_text = safe_generated.text.lower()

        if classify_final_response(final_response) == "question" and "?" not in safe_generated.text:
            return False, "question_downgrade"
        if classify_final_response(final_response) == "error" and not _ERROR_SIGNAL_RE.search(generated_text):
            return False, "error_downgrade"

        final_numbers = set(_NUMBER_RE.findall(final_text))
        for number in _NUMBER_RE.findall(generated_text):
            if number not in final_numbers:
                return False, "unsupported_number"

        final_words = set(re.findall(r"[a-z]+", final_text))
        generated_words = set(re.findall(r"[a-z]+", generated_text))
        unsupported_actions = (_ACTION_TERMS & generated_words) - final_words
        if unsupported_actions:
            passed_is_grounded_by_ran_tests = (
                unsupported_actions == {"passed"}
                and "tests" in generated_words
                and "tests" in final_words
                and ({"ran", "pass", "passed", "verified"} & final_words)
            )
            if not passed_is_grounded_by_ran_tests:
                return False, "unsupported_action"

        for phrase in _COMPLETION_VERBS:
            if phrase in generated_text and phrase not in final_text:
                tests_passed_is_grounded_by_ran_tests = (
                    phrase in {"tests pass", "tests passed"}
                    and "tests" in final_words
                    and ({"ran", "pass", "passed", "verified"} & final_words)
                )
                if not tests_passed_is_grounded_by_ran_tests:
                    return False, "unsupported_action"

        if _FUTURE_PROMISE_RE.search(generated_text) and not _FUTURE_PROMISE_RE.search(final_text):
            return False, "future_promise"

        return True, None


_DEFAULT_SUMMARIZER = FinalSpeechSummarizer()


def summarize_final_speech(final_response: str, context: VoiceContext | None = None) -> VoiceSummaryResult:
    """Module-level convenience API for existing gateway call sites."""
    return _DEFAULT_SUMMARIZER.summarize(final_response, context or VoiceContext())
