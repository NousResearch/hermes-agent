"""Deterministic policy boundary for ambient voice output.

The policy in this module is intentionally conservative: it decides whether a
candidate assistant response may become spoken room audio, produces a sanitized
speech string when allowed, and exposes only safe reason-code metadata.  It must
never include raw blocked text, secret values, stack traces, file paths, or other
matched excerpts in decisions or metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


_DEFAULT_MAX_SECONDS = {
    "ack": 2,
    "completion": 4,
    "question": 6,
    "error": 3,
    "progress": 2,
}

_DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "living_room_default": {
        "max_seconds": _DEFAULT_MAX_SECONDS,
        "max_chars": 180,
        "allow_code": False,
        "allow_tool_logs": False,
        "allow_raw_paths": False,
        "allow_sensitive_topics": False,
        "allow_secret_like_text": False,
        "require_explicit_for_sensitive_topics": True,
        "suppress_errors_with_stack_traces": True,
        "one_sentence": True,
    },
    "airpods_private": {
        "max_seconds": {
            "ack": 3,
            "completion": 8,
            "question": 10,
            "error": 5,
            "progress": 3,
        },
        "max_chars": 320,
        "allow_code": False,
        "allow_tool_logs": False,
        "allow_raw_paths": False,
        "allow_sensitive_topics": "explicit_only",
        "allow_secret_like_text": False,
        "require_explicit_for_sensitive_topics": True,
        "suppress_errors_with_stack_traces": True,
        "one_sentence": True,
    },
    "chat_attachment": {
        "max_seconds": {"completion": 10, "question": 10, "error": 5, "ack": 3, "progress": 3},
        "max_chars": 500,
        "allow_code": False,
        "allow_tool_logs": False,
        "allow_raw_paths": False,
        "allow_sensitive_topics": "explicit_only",
        "allow_secret_like_text": False,
        "require_explicit_for_sensitive_topics": True,
        "suppress_errors_with_stack_traces": True,
        "one_sentence": True,
    },
    "discord_voice": {
        "max_seconds": _DEFAULT_MAX_SECONDS,
        "max_chars": 180,
        "allow_code": False,
        "allow_tool_logs": False,
        "allow_raw_paths": False,
        "allow_sensitive_topics": False,
        "allow_secret_like_text": False,
        "require_explicit_for_sensitive_topics": True,
        "suppress_errors_with_stack_traces": True,
        "one_sentence": True,
    },
}

_CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]{1,200}`")
_MEDIA_RE = re.compile(r"MEDIA:\S+")
_DIRECTIVE_RE = re.compile(r"\[\[[^\]]+\]\]")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_STACK_TRACE_RE = re.compile(
    r"(?im)(traceback \(most recent call last\):|^\s*File \"[^\"]+\", line \d+|^[A-Za-z_][\w.]*Error:|^\s*at .+\(.+:\d+:\d+\))"
)
_COMMAND_LOG_RE = re.compile(
    r"(?im)^\s*(?:\$\s+.+|>>>\s+.+|FAILED\s+.+|ERROR\s+.+|={5,}.+|-{5,}.+|E\s+.+)$"
)
_RAW_PATH_RE = re.compile(
    r"(?<![\w@])(?:~|/(?:Users|tmp|var|private|Volumes|home|opt|etc|usr|mnt|workspace|workspaces))"
    r"(?:/[A-Za-z0-9._@%+=:,~-]+)+"
)
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s\\/:*?\"<>|]+\\?)+")
_SECRET_LIKE_RE = re.compile(
    r"(?ix)("
    r"\b(?:sk|ghp|gho|github_pat|glpat|hf)_[A-Za-z0-9_\-]{12,}"
    r"|\b(?:xoxb|xoxp|xoxa|xoxr)-[A-Za-z0-9\-]{12,}"
    r"|\bsk-[A-Za-z0-9_\-]{12,}"
    r"|\bAKIA[0-9A-Z]{16}\b"
    r"|\b[A-Z0-9_]*(?:AWS_SECRET_ACCESS_KEY|SECRET_ACCESS_KEY|API_KEY|TOKEN|PASSWORD|PRIVATE_KEY)\s*=\s*\S+"
    r"|-----BEGIN\s+(?:RSA\s+|OPENSSH\s+|EC\s+)?PRIVATE\s+KEY-----"
    r")"
)
_SENSITIVE_TOPIC_RE = re.compile(
    r"(?i)\b("
    r"pnl|p&l|profit\s+and\s+loss|portfolio|drawdown|trading|trade|position|leverage|liquidation|"
    r"bank\s+account|credit\s+card|salary|net\s+worth|tax|medical|diagnosis|prescription|"
    r"social\s+security|ssn|passport|relationship|divorce|therapy|credential|password|private\s+key"
    r")\b"
)
_SENTENCE_RE = re.compile(r"^.{1,260}?[.!?。！？](?=\s|$)")


@dataclass(frozen=True)
class VoiceContext:
    source: str
    platform: str | None = None
    channel_id: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    source_message_id: str | None = None
    input_modality: str | None = None
    output_device: str | None = None
    profile: str | None = None
    explicit_spoken_request: bool = False
    is_private_context: bool = False
    config_scope: str = "living_room_default"


@dataclass(frozen=True)
class VoicePolicyDecision:
    allowed: bool
    text: str
    original_chars: int
    sanitized: bool
    truncated: bool
    suppressed: bool
    max_seconds: int
    reasons: tuple[str, ...]
    classifiers: dict[str, bool]
    rule_profile: str

    def to_metadata(self) -> dict[str, Any]:
        """Return safe-to-display policy metadata with reason codes only."""
        return {
            "allowed": self.allowed,
            "sanitized": self.sanitized,
            "truncated": self.truncated,
            "suppressed": self.suppressed,
            "reason_codes": self.reasons,
            "rule_profile": self.rule_profile,
            "blocked_secret_like": bool(self.classifiers.get("secret_like")),
            "dropped_code": bool(self.classifiers.get("code")),
            "blocked_stack_trace": bool(self.classifiers.get("stack_trace")),
            "dropped_tool_logs": bool(self.classifiers.get("command_log")),
            "dropped_paths": bool(self.classifiers.get("raw_path")),
            "blocked_sensitive_topic": bool(self.classifiers.get("sensitive_topic")),
        }


@dataclass
class AmbientVoicePolicy:
    """Evaluate candidate speech for deterministic ambient-safety rules."""

    rule_profiles: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: _DEFAULT_PROFILES)
    default_rule_profile: str = "living_room_default"

    def evaluate(self, text: str, context: VoiceContext) -> VoicePolicyDecision:
        original = str(text or "")
        original_chars = len(original)
        rule_profile = self._rule_profile_name(context)
        profile = self._profile(rule_profile)
        classifiers = self.classify(original)
        sanitized_text, sanitize_reasons, sanitized_changed = self.sanitize(original, context, classifiers)

        reasons: list[str] = []
        allowed = True
        sensitive_summarized = False

        if classifiers["secret_like"] and not profile.get("allow_secret_like_text", False):
            allowed = False
            self._append_reason(reasons, "secret_like")
        if classifiers["stack_trace"] and profile.get("suppress_errors_with_stack_traces", True):
            allowed = False
            self._append_reason(reasons, "stack_trace")
        if classifiers["command_log"] and not profile.get("allow_tool_logs", False):
            # Command/log output is too risky for ambient speech. If mixed with
            # a safe sentence, the caller can pass that sentence separately.
            allowed = False
            self._append_reason(reasons, "command_log")
        if classifiers["code"] and not sanitized_text:
            allowed = False
        if classifiers["sensitive_topic"]:
            if self._can_speak_sensitive_topic(profile, context):
                sanitized_text = "I can discuss that sensitive topic in this private voice context."
                sensitive_summarized = True
                sanitized_changed = True
                self._append_reason(reasons, "sensitive_topic_summarized")
            else:
                allowed = False
                self._append_reason(reasons, "sensitive_topic")

        for reason in sanitize_reasons:
            self._append_reason(reasons, reason)

        bounded_text, truncated, bound_reasons = self._bound_text(sanitized_text, profile)
        for reason in bound_reasons:
            self._append_reason(reasons, reason)
        sanitized_text = bounded_text

        if not sanitized_text:
            allowed = False
            self._append_reason(reasons, "empty_after_sanitization")

        if not allowed:
            sanitized_text = ""
            self._append_reason(reasons, "empty_after_sanitization")

        if classifiers["secret_like"] and "secret_like" in reasons:
            reasons = ["secret_like", *[reason for reason in reasons if reason != "secret_like"]]

        sanitized = sanitized_changed or sanitized_text != original or sensitive_summarized
        return VoicePolicyDecision(
            allowed=allowed,
            text=sanitized_text,
            original_chars=original_chars,
            sanitized=sanitized,
            truncated=truncated,
            suppressed=not allowed,
            max_seconds=self.max_seconds(context.source, context),
            reasons=tuple(reasons),
            classifiers=classifiers,
            rule_profile=rule_profile,
        )

    def classify(self, text: str) -> dict[str, bool]:
        candidate = str(text or "")
        return {
            "secret_like": bool(_SECRET_LIKE_RE.search(candidate)),
            "code": bool(_CODE_FENCE_RE.search(candidate) or _INLINE_CODE_RE.search(candidate)),
            "stack_trace": bool(_STACK_TRACE_RE.search(candidate)),
            "command_log": bool(_COMMAND_LOG_RE.search(candidate)),
            "raw_path": bool(_RAW_PATH_RE.search(candidate) or _WINDOWS_PATH_RE.search(candidate)),
            "long_response": len(candidate) > 180 or candidate.count("\n") >= 2 or len(re.findall(r"[.!?。！？]", candidate)) > 1,
            "sensitive_topic": bool(_SENSITIVE_TOPIC_RE.search(candidate)),
        }

    def sanitize(
        self,
        text: str,
        context: VoiceContext,
        classifiers: Mapping[str, bool] | None = None,
    ) -> tuple[str, tuple[str, ...], bool]:
        del context  # Reserved for future context-specific sanitization.
        original = str(text or "")
        cleaned = original
        flags = classifiers or self.classify(original)
        reasons: list[str] = []

        if flags.get("code"):
            cleaned = _CODE_FENCE_RE.sub(" ", cleaned)
            cleaned = _INLINE_CODE_RE.sub(" ", cleaned)
            self._append_reason(reasons, "code_stripped")
        if flags.get("command_log"):
            cleaned = _COMMAND_LOG_RE.sub(" ", cleaned)
            self._append_reason(reasons, "command_log_stripped")
        if flags.get("stack_trace"):
            cleaned = _STACK_TRACE_RE.sub(" ", cleaned)
            self._append_reason(reasons, "stack_trace_stripped")
        if flags.get("raw_path"):
            cleaned = _RAW_PATH_RE.sub("a local file", cleaned)
            cleaned = _WINDOWS_PATH_RE.sub("a local file", cleaned)
            self._append_reason(reasons, "raw_path_stripped")

        cleaned = _MEDIA_RE.sub(" ", cleaned)
        cleaned = _DIRECTIVE_RE.sub(" ", cleaned)
        cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
        cleaned = re.sub(r"^[#>*\-•\s]+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\b(?:and|or)\b\s*$", "", cleaned.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;:\n\t")
        return cleaned, tuple(reasons), cleaned != original

    def max_seconds(self, kind: str, context: VoiceContext) -> int:
        rule_profile = self._rule_profile_name(context)
        profile = self._profile(rule_profile)
        max_seconds = profile.get("max_seconds", {})
        source_kind = str(kind or context.source or "completion").lower()
        if source_kind.startswith("assistant"):
            source_kind = "completion"
        return int(max_seconds.get(source_kind, max_seconds.get("completion", 4)) or 4)

    def _bound_text(self, text: str, profile: Mapping[str, Any]) -> tuple[str, bool, tuple[str, ...]]:
        if not text:
            return "", False, ()
        reasons: list[str] = []
        bounded = text.strip()
        truncated = False
        if profile.get("one_sentence", True):
            match = _SENTENCE_RE.match(bounded)
            if match and match.group(0).strip() != bounded:
                bounded = match.group(0).strip()
                truncated = True
                self._append_reason(reasons, "bounded_to_one_sentence")
        max_chars = int(profile.get("max_chars", 180) or 180)
        if len(bounded) > max_chars:
            bounded = bounded[: max_chars - 1].rstrip(" ,.;:") + "…"
            truncated = True
            self._append_reason(reasons, "bounded_to_max_chars")
        return bounded, truncated, tuple(reasons)

    def _can_speak_sensitive_topic(self, profile: Mapping[str, Any], context: VoiceContext) -> bool:
        policy = profile.get("allow_sensitive_topics", False)
        if policy is True:
            return True
        if policy == "explicit_only":
            return bool(context.explicit_spoken_request and context.is_private_context)
        return False

    def _rule_profile_name(self, context: VoiceContext) -> str:
        requested = str(context.config_scope or "").strip()
        if requested and requested in self.rule_profiles:
            return requested
        device = str(context.output_device or "").strip().lower()
        if device == "airpods":
            return "airpods_private"
        if device == "chat_attachment":
            return "chat_attachment"
        if device == "discord_voice":
            return "discord_voice"
        return self.default_rule_profile

    def _profile(self, rule_profile: str) -> Mapping[str, Any]:
        return self.rule_profiles.get(rule_profile) or self.rule_profiles[self.default_rule_profile]

    @staticmethod
    def _append_reason(reasons: list[str], reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)
