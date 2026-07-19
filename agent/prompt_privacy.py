"""
Prompt-level PII anonymization middleware.

When ``privacy.anonymize_prompts`` is enabled in config.yaml, this module
scrubs personally identifiable information (PII) from prompts before they
reach the model provider, then restores the original values in the response.

This is distinct from ``agent/redact.py``, which handles secret redaction in
tool output and logs — ``prompt_privacy`` operates at the API boundary,
protecting user data from upstream model providers.

Architecture
------------
::

    Prompt Builder → PrivacyMiddleware.scrub() → Provider API
    Provider API   → PrivacyMiddleware.restore() → User

Only the provider sees anonymized text.  The user never notices the swap.

Config (in ~/.hermes/config.yaml)
----------------------------------
.. code-block:: yaml

    privacy:
      anonymize_prompts: false   # opt-in — off by default
      scrub_emails: true
      scrub_tokens: true         # API keys, GitHub tokens, etc.
      scrub_ips: true
      scrub_phones: true
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from hermes_cli.config import HermesConfig  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── PII Detection Patterns ───────────────────────────────────────────

PII_PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # (category, regex, placeholder_prefix)
    #
    # GitHub tokens — ghp_, gho_, ghu_, ghs_, ghr_
    ("token", re.compile(
        r"\bgh[opusr]_[A-Za-z0-9_]{36}\b"
    ), "GH_TOKEN"),
    # OpenAI / Anthropic / generic sk- tokens
    ("token", re.compile(
        r"\bsk-[A-Za-z0-9+/=]{20,}\b"
    ), "API_KEY"),
    # Bearer auth headers
    ("token", re.compile(
        r"\bBearer\s+[A-Za-z0-9+/=]{30,}\b"
    ), "BEARER"),
    # Email addresses
    ("email", re.compile(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
    ), "EMAIL"),
    # IPv4 addresses (local + public)
    ("ip", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ), "IP"),
    # French phone numbers — +33 prefix followed by exactly 9 digits (8 pairs)
    # No leading \b because + is a non-word character, so a word boundary
    # would never exist between a space and +.  Trailing \b is fine.
    ("phone", re.compile(
        r"(?:\+33)[1-9](?:\d{2}){4}\b"
    ), "PHONE"),
    # URLs containing sensitive query params (token, key, secret, auth)
    ("url", re.compile(
        r"https?://[^\s]*?[?&](?:token|key|secret|auth|api[_-]?key|pat)="
        r"[^&\s]+",
        re.IGNORECASE,
    ), "URL"),
]

# ── Cache ─────────────────────────────────────────────────────────────

class _PrivacyCache:
    """In-memory placeholder ↔ original value mapping.

    Uses a stable hash-derived placeholder so each original value always maps
    to the same placeholder within a single scrub/restore cycle.
    """

    def __init__(self) -> None:
        self._placeholder_map: Dict[str, str] = {}  # original → placeholder
        self._restore_map: Dict[str, str] = {}      # placeholder → original

    def add(self, original: str) -> str:
        """Map an original PII value to a stable anonymous placeholder."""
        if original in self._placeholder_map:
            return self._placeholder_map[original]
        stable_id = hashlib.sha256(original.encode()).hexdigest()[:12]
        placeholder = f"[{original[:6]}_{stable_id}]"
        self._placeholder_map[original] = placeholder
        self._restore_map[placeholder] = original
        return placeholder

    def restore(self, text: str) -> str:
        """Replace all placeholders in *text* with their original values."""
        result = text
        for placeholder, original in self._restore_map.items():
            result = result.replace(placeholder, original)
        return result

    def clear(self) -> None:
        self._placeholder_map.clear()
        self._restore_map.clear()


# ── Middleware ────────────────────────────────────────────────────────

class PrivacyMiddleware:
    """Prompt-level PII anonymization for the agent pipeline.

    Parameters
    ----------
    scrub_emails : bool
        Whether to scrub email addresses (default: True).
    scrub_tokens : bool
        Whether to scrub API keys and tokens (default: True).
    scrub_ips : bool
        Whether to scrub IPv4 addresses (default: True).
    scrub_phones : bool
        Whether to scrub phone numbers (default: True).
    """

    def __init__(
        self,
        scrub_emails: bool = True,
        scrub_tokens: bool = True,
        scrub_ips: bool = True,
        scrub_phones: bool = True,
    ) -> None:
        self._enabled_categories: frozenset = frozenset(
            cat
            for cat, enabled in {
                "email": scrub_emails,
                "token": scrub_tokens,
                "ip": scrub_ips,
                "phone": scrub_phones,
                "url": True,  # always scrub URLs with credentials
            }.items()
            if enabled
        )
        self._patterns: List[Tuple[str, re.Pattern, str]] = [
            (name, regex, prefix)
            for name, regex, prefix in PII_PATTERNS
            if name in self._enabled_categories
        ]
        self.cache = _PrivacyCache()

    # ------------------------------------------------------------------
    def scrub(self, text: str) -> str:
        """Anonymize a single text string before sending to the model."""
        self.cache.clear()
        result = text
        for _, regex, _ in self._patterns:

            def _replace(m: re.Match) -> str:
                return self.cache.add(m.group(0))

            result = regex.sub(_replace, result)
        return result

    # ------------------------------------------------------------------
    def restore(self, text: str) -> str:
        """Restore original PII in the model's response text."""
        return self.cache.restore(text)

    # ------------------------------------------------------------------
    def scrub_messages(
        self, messages: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        """Anonymize a full OpenAI-format messages array.

        Handles both ``content: str`` and multi-modal ``content: list``.
        """
        scrubbed: List[Dict[str, object]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                scrubbed.append(msg)
                continue

            content = msg.get("content")
            if isinstance(content, str):
                scrubbed.append({**msg, "content": self.scrub(content)})
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        new_parts.append({
                            **part,
                            "text": self.scrub(part["text"]),
                        })
                    else:
                        new_parts.append(part)
                scrubbed.append({**msg, "content": new_parts})
            else:
                scrubbed.append(msg)
        return scrubbed

    # ------------------------------------------------------------------
    def restore_response(self, response: object) -> object:
        """Restore PII in a model response object."""
        self._restore_obj(response)
        return response

    def _restore_obj(self, obj: object) -> None:
        """Recursively walk *obj* and restore placeholders in-place."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    obj[key] = self.restore(value)
                else:
                    self._restore_obj(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = self.restore(item)
                else:
                    self._restore_obj(item)
        elif hasattr(obj, "__dict__") and not isinstance(obj, str):
            for key, value in obj.__dict__.items():
                if isinstance(value, str):
                    setattr(obj, key, self.restore(value))
                else:
                    self._restore_obj(value)


# ── Singleton access ─────────────────────────────────────────────────

_middleware: Optional[PrivacyMiddleware] = None


def get_middleware(
    config: Optional["HermesConfig"] = None,
) -> Optional[PrivacyMiddleware]:
    """Get or create the privacy middleware from Hermes config.

    Returns ``None`` when ``privacy.anonymize_prompts`` is disabled.
    """
    global _middleware
    if _middleware is not None:
        return _middleware

    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            return None

    privacy = config.get("privacy", {})
    if not privacy.get("anonymize_prompts", False):
        return None

    _middleware = PrivacyMiddleware(
        scrub_emails=privacy.get("scrub_emails", True),
        scrub_tokens=privacy.get("scrub_tokens", True),
        scrub_ips=privacy.get("scrub_ips", True),
        scrub_phones=privacy.get("scrub_phones", True),
    )
    logger.info("Privacy middleware enabled — prompts will be anonymized before API calls")
    return _middleware


def clear_middleware() -> None:
    """Reset the privacy middleware (called at end of each API call)."""
    global _middleware
    if _middleware is not None:
        _middleware.cache.clear()
    _middleware = None


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mw = PrivacyMiddleware()

    # Use real-looking test data
    real_token = "ghp_" + "a" * 36
    phone = "+336" + "12345678"
    test = (
        f"Contact: alice@example.com\n"
        f"GitHub: {real_token}\n"
        f"Server: 10.0.0.1\n"
        f"Phone: {phone}\n"
        "API: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghijklmnopQRSTUVWXYZ0123456789====\n"
        "URL: https://api.example.com?token=secret123&user=test\n"
    )

    print("BEFORE:", repr(test[:120]), "...")
    s = mw.scrub(test)
    print("AFTER: ", repr(s[:120]), "...")
    r = mw.restore(s)
    print("RESTORED:", repr(r[:120]), "...")

    errors = []
    for pii in [real_token, phone, "alice@example.com", "10.0.0.1", "secret123"]:
        if pii in s:
            errors.append(f"LEAK: {pii[:30]} still present in scrubbed output!")

    for pii in [real_token, phone, "alice@example.com", "10.0.0.1"]:
        if pii not in r:
            errors.append(f"LOST: {pii[:30]} not restored!")

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAll assertions passed.")