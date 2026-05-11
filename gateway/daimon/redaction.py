"""Regex-based post-response filter for redacting sensitive tokens."""

import re

# Patterns ordered from most specific to least specific.
# More specific patterns (e.g., sk-proj-, sk-ant-) must come before
# the generic sk- pattern to avoid greedy matching.
_REDACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # OpenAI project key (most specific sk- variant)
    (re.compile(r"sk-proj-[a-zA-Z0-9\-_]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
    # Anthropic key (sk-ant- before generic sk-)
    (re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
    # Generic OpenAI key
    (re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE), "[REDACTED_OPENAI_KEY]"),
    # GitHub PAT (most specific GitHub variant)
    (re.compile(r"github_pat_[a-zA-Z0-9_]{20,}", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
    # GitHub personal access token
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
    # GitHub OAuth token
    (re.compile(r"gho_[a-zA-Z0-9]{36,}", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
    # xAI key
    (re.compile(r"xai-[a-zA-Z0-9]{20,}", re.IGNORECASE), "[REDACTED_XAI_KEY]"),
    # Google API key
    (re.compile(r"AIza[a-zA-Z0-9\-_]{30,}"), "[REDACTED_GOOGLE_KEY]"),
    # AWS access key (always uppercase by spec)
    (re.compile(r"AKIA[A-Z0-9]{16}"), "[REDACTED_AWS_KEY]"),
    # Discord/Slack bot token
    (re.compile(r"Bot\s+[A-Za-z0-9._\-]{50,}", re.IGNORECASE), "[REDACTED_BOT_TOKEN]"),
]


def redact_response(text: str) -> str:
    """Redact sensitive tokens from the given text.

    Applies compiled regex patterns in order, replacing matches
    with appropriate redaction placeholders.
    """
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
