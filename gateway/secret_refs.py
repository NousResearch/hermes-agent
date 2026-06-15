"""Secret-reference contracts and redaction helpers for gateway features.

This module is intentionally offline-safe.  It defines the runtime-only token
wrapper and resolver protocol used by Discord native multi-bot identities, plus
helpers for validating configured secret references and redacting diagnostics.
It does not call MCP secret handoff, Discord, or any network service.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any, Mapping, Protocol, SupportsIndex, runtime_checkable

try:  # Keep this module usable in small import contexts.
    from agent.redact import redact_sensitive_text
except Exception:  # pragma: no cover - defensive fallback
    redact_sensitive_text = None  # type: ignore[assignment]

DEFAULT_SECRET_REF_SCHEMES = ("secret://",)
ENV_SECRET_REF_SCHEME = "env://"
DEV_SECRET_REF_SCHEMES = DEFAULT_SECRET_REF_SCHEMES + (ENV_SECRET_REF_SCHEME,)

FORBIDDEN_DISCORD_TOKEN_KEYS = frozenset(
    {"token", "bottoken", "discordtoken", "discordbottoken"}
)

_SECRET_REF_RE = re.compile(r"\b(secret|env)://[^\s\"'<>),]+")
_DISCORD_DOT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:Bot\s+)?[A-Za-z0-9_-]{3,}\."
    r"[A-Za-z0-9_-]{3,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_TOKEN_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?:token|bot_token|discord_token|DISCORD_BOT_TOKEN)\b\s*[:=]\s*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s'\",}]+)(?P=quote)",
    re.IGNORECASE,
)
_SENSITIVE_KEY_PARTS = ("token", "secret", "password", "credential", "auth")
_NON_SECRET_KEY_NAMES = frozenset(
    {
        "delivery_key",
        "idempotency_key",
        "route_key",
        "scope_key",
        "session_key",
        "state_key",
        "secret_ref",
        "secret_refs",
    }
)
_SENSITIVE_KEY_RE = re.compile(
    r"(^api[_-]?key$|[_\-.:-]key$|^(?:access|client|private|public|ssh|openai|anthropic)[_-]?key$)",
    re.IGNORECASE,
)


class SecretRefError(ValueError):
    """Raised when a configured secret reference is invalid."""


class SecretResolutionError(RuntimeError):
    """Raised when a runtime secret reference cannot be resolved safely."""


class SensitiveToken:
    """Runtime-only wrapper for resolved token material.

    ``str()`` and ``repr()`` are redacted.  Callers that truly need the value for
    runtime clients must opt in explicitly via ``reveal()``.  The object is not
    JSON-serializable by default and rejects pickle-style state extraction.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise SecretResolutionError("resolved secret token must be a non-empty string")
        self._value = value

    def reveal(self) -> str:
        """Return the plaintext token for immediate runtime use only."""

        return self._value

    def __str__(self) -> str:
        return "<SensitiveToken redacted>"

    def __repr__(self) -> str:
        return "SensitiveToken(<redacted>)"

    def __getstate__(self) -> None:
        raise TypeError("SensitiveToken is runtime-only and cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:  # noqa: ARG002
        raise TypeError("SensitiveToken is runtime-only and cannot be serialized")


@runtime_checkable
class SecretResolver(Protocol):
    """Runtime secret resolver contract.

    Implementations return ``SensitiveToken``.  Legacy tests may still return a
    string while callers migrate; gateway code unwraps either form without
    logging or persisting the plaintext.
    """

    def resolve(self, ref: str) -> SensitiveToken:
        """Resolve a validated secret reference into runtime-only token material."""
        raise NotImplementedError


def _normalize_config_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def is_raw_discord_token_like(value: Any) -> bool:
    """Return True for obvious raw Discord bot token-shaped values."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    return bool(_DISCORD_DOT_TOKEN_RE.search(text))


def reject_forbidden_credential_keys(
    data: Mapping[str, Any],
    *,
    context: str = "identity",
) -> None:
    """Reject v2 config mappings that attempt to include raw token keys."""

    for key in data:
        normalized = _normalize_config_key(key)
        if normalized in FORBIDDEN_DISCORD_TOKEN_KEYS:
            raise SecretRefError(f"{context} uses forbidden credential key {key!r}")


def reject_raw_discord_token_values(value: Any, *, context: str = "identity") -> None:
    """Reject obvious raw Discord token material in config-shaped data."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) == "token_secret_ref":
                continue
            reject_raw_discord_token_values(item, context=context)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            reject_raw_discord_token_values(item, context=context)
        return
    if not isinstance(value, str):
        return
    if value.strip().startswith(("secret://", "env://")):
        return
    if is_raw_discord_token_like(value):
        raise SecretRefError(f"{context} contains raw Discord token-looking value")


def _allowed_schemes(*, allow_env: bool) -> tuple[str, ...]:
    return DEV_SECRET_REF_SCHEMES if allow_env else DEFAULT_SECRET_REF_SCHEMES


def validate_secret_ref(
    ref: Any,
    *,
    allow_env: bool = False,
    field_name: str = "secret_ref",
) -> str:
    """Validate and normalize a secret reference.

    ``secret://...`` is always allowed.  ``env://ENV_VAR`` is accepted only when
    ``allow_env=True`` for explicit dev/test resolver policy.
    """

    if not isinstance(ref, str) or not ref.strip():
        raise SecretRefError(f"{field_name} is required")
    value = ref.strip()
    matching_scheme = next(
        (scheme for scheme in _allowed_schemes(allow_env=allow_env) if value.startswith(scheme)),
        None,
    )
    if matching_scheme is None or not value[len(matching_scheme) :].strip():
        raise SecretRefError(
            f"{field_name} must use an allowed secret-ref scheme with a non-empty secret path"
        )
    if matching_scheme == ENV_SECRET_REF_SCHEME:
        env_name = value[len(ENV_SECRET_REF_SCHEME) :].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            raise SecretRefError(f"{field_name} env ref must name a single environment variable")
    return value


def redact_secret_ref(ref: str | None) -> str | None:
    """Redact a configured secret reference while preserving its scheme."""

    if not ref:
        return None
    for scheme in DEV_SECRET_REF_SCHEMES:
        if str(ref).startswith(scheme):
            return f"{scheme}<redacted>"
    return "<redacted>"


def redact_sensitive_text_value(value: Any) -> str:
    """Redact secret refs and Discord-token-like strings in free-form text."""

    if isinstance(value, SensitiveToken):
        return str(value)
    text = str(value)
    if not text:
        return text
    if text.startswith(("secret://", "env://")):
        return redact_secret_ref(text) or "<redacted>"
    if redact_sensitive_text is not None:
        text = redact_sensitive_text(text, force=True)
    text = _SECRET_REF_RE.sub(lambda m: redact_secret_ref(m.group(0)) or "<redacted>", text)
    text = _DISCORD_DOT_TOKEN_RE.sub("<redacted>", text)
    text = _TOKEN_ASSIGNMENT_RE.sub(
        lambda m: f"{m.group('prefix')}{m.group('quote')}<redacted>{m.group('quote')}",
        text,
    )
    return text


def _is_sensitive_output_key(key: Any) -> bool:
    lowered = str(key).lower()
    normalized = re.sub(r"[^a-z0-9]", "", lowered)
    if lowered in _NON_SECRET_KEY_NAMES:
        return False
    return any(part in lowered for part in _SENSITIVE_KEY_PARTS) or bool(
        _SENSITIVE_KEY_RE.search(lowered)
        or normalized in {"apikey", "accesskey", "clientkey", "privatekey", "publickey", "sshkey"}
    )


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact secret/token-looking values from diagnostic data."""

    if isinstance(value, SensitiveToken):
        return str(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if text_key == "token_secret_ref" and isinstance(item, str):
                redacted[text_key] = redact_secret_ref(item)
            elif _is_sensitive_output_key(text_key):
                if isinstance(item, str) and item.startswith(("secret://", "env://")):
                    redacted[text_key] = redact_secret_ref(item)
                else:
                    redacted[text_key] = "<redacted>"
            else:
                redacted[text_key] = redact_sensitive_data(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text_value(value)
    return value


class StaticSecretResolver:
    """Fake/test resolver backed by an in-memory mapping."""

    def __init__(self, secrets: Mapping[str, str]) -> None:
        self._secrets = dict(secrets)
        self.calls: list[str] = []

    def resolve(self, ref: str) -> SensitiveToken:
        ref = validate_secret_ref(ref)
        self.calls.append(ref)
        try:
            return SensitiveToken(self._secrets[ref])
        except KeyError:
            raise SecretResolutionError(
                f"secret not found for ref {redact_secret_ref(ref)!r}"
            ) from None


class EnvSecretResolver:
    """Explicit dev/test resolver for ``env://ENV_VAR`` refs.

    ``allow_env`` defaults to False so env refs fail closed unless a test or dev
    caller opts in deliberately.  ``secret://`` refs are validated but not backed
    by this resolver because real secret backends are out of scope for Slice 1.3.
    """

    def __init__(
        self,
        *,
        allow_env: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.allow_env = allow_env
        self._environ = environ if environ is not None else os.environ
        self.calls: list[str] = []

    def resolve(self, ref: str) -> SensitiveToken:
        ref = validate_secret_ref(ref, allow_env=self.allow_env)
        self.calls.append(ref)
        if not ref.startswith(ENV_SECRET_REF_SCHEME):
            raise SecretResolutionError(
                f"no secret backend configured for ref {redact_secret_ref(ref)!r}"
            )
        env_name = ref[len(ENV_SECRET_REF_SCHEME) :]
        token = self._environ.get(env_name)
        if not token:
            raise SecretResolutionError(
                f"environment secret {redact_secret_ref(ref)!r} is not set"
            )
        return SensitiveToken(token)


class GatewaySecretResolver:
    """Production gateway resolver for ``secret://...`` references.

    This resolver intentionally does not read legacy Discord credential names
    such as ``DISCORD_BOT_TOKEN``. Operators can provide refs either via a JSON
    mapping in ``HERMES_SECRET_REFS_JSON`` or by per-ref environment variables:

    - ``HERMES_SECRET_REF_<SHA256(secret://...)>`` (preferred, no path in name)
    - ``HERMES_SECRET_<SANITIZED_SECRET_PATH>`` (human-readable fallback)

    Resolved values are returned as ``SensitiveToken`` and are not cached.
    """

    def __init__(self, *, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ
        self.calls: list[str] = []

    @staticmethod
    def hashed_env_name(ref: str) -> str:
        digest = hashlib.sha256(ref.encode("utf-8")).hexdigest().upper()
        return f"HERMES_SECRET_REF_{digest}"

    @staticmethod
    def readable_env_name(ref: str) -> str:
        path = ref[len(DEFAULT_SECRET_REF_SCHEMES[0]) :]
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_").upper()
        if not normalized:
            normalized = "UNNAMED"
        return f"HERMES_SECRET_{normalized}"

    def _resolve_from_json_mapping(self, ref: str) -> str | None:
        raw = self._environ.get("HERMES_SECRET_REFS_JSON")
        if not raw:
            return None
        try:
            mapping = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretResolutionError("HERMES_SECRET_REFS_JSON is not valid JSON") from exc
        if not isinstance(mapping, Mapping):
            raise SecretResolutionError("HERMES_SECRET_REFS_JSON must be a JSON object")
        path = ref[len(DEFAULT_SECRET_REF_SCHEMES[0]) :]
        value = mapping.get(ref, mapping.get(path))
        return value if isinstance(value, str) and value else None

    def resolve(self, ref: str) -> SensitiveToken:
        ref = validate_secret_ref(ref, allow_env=False)
        self.calls.append(ref)
        token = self._resolve_from_json_mapping(ref)
        if token is None:
            for env_name in (self.hashed_env_name(ref), self.readable_env_name(ref)):
                token = self._environ.get(env_name)
                if token:
                    break
        if not token:
            raise SecretResolutionError(
                f"secret not found for ref {redact_secret_ref(ref)!r}; configure "
                "HERMES_SECRET_REFS_JSON or a HERMES_SECRET_REF_<sha256> env var"
            )
        return SensitiveToken(token)


class SecretRedactionFilter(logging.Filter):
    """Logging filter that scrubs secret refs and token-looking values."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_sensitive_text_value(record.getMessage())
            record.args = ()
        except Exception:  # pragma: no cover - defensive logging path
            record.msg = redact_sensitive_text_value(record.msg)
            if isinstance(record.args, dict):
                record.args = redact_sensitive_data(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_sensitive_data(arg) for arg in record.args)
            elif record.args:
                record.args = redact_sensitive_data(record.args)
        return True


def install_gateway_secret_redaction_filter(
    logger: logging.Logger | None = None,
) -> SecretRedactionFilter:
    """Install a secret-redaction filter on a gateway logger if absent."""

    target = logger or logging.getLogger("gateway")
    for existing in target.filters:
        if isinstance(existing, SecretRedactionFilter):
            return existing
    new_filter = SecretRedactionFilter()
    target.addFilter(new_filter)
    return new_filter
