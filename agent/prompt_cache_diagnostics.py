"""Prompt-cache diagnostics that do not mutate provider payloads.

The helpers in this module compute stable, redacted fingerprints for the parts
of an LLM request that commonly determine provider-side prompt-cache reuse.
They are intentionally observational: no cache_control markers are added here
and no request payload is modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


_HASH_PREFIX_LEN = 16
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "api_token",
    "token",
    "secret",
    "password",
    "credential",
    "auth",
)
_MAX_HASH_BYTES = 1_000_000


@dataclass(frozen=True)
class PromptCacheDiagnostics:
    """Local fingerprints for cache observability.

    The hashes are best-effort local signals. They must be treated as possible
    cache-bust evidence only because providers keep their exact cache keys
    private and TTL/routing can invalidate a cache even when these hashes are
    unchanged.
    """

    system_prompt_hash: str | None = None
    tools_hash: str | None = None
    skills_hash: str | None = None
    message_prefix_hash: str | None = None
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None
    system_prompt_tokens_estimate: int = 0
    tools_tokens_estimate: int = 0
    message_prefix_tokens_estimate: int = 0
    possible_bust_causes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_json(value: Any) -> str:
    """Return deterministic JSON for hashing serializable-ish values."""

    def default(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            return {
                k: v
                for k, v in vars(obj).items()
                if not k.startswith("_")
                and not any(fragment in k.lower() for fragment in _SECRET_KEY_FRAGMENTS)
            }
        rendered = repr(obj)
        return rendered[:4096]

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default)


def stable_hash(value: Any) -> str | None:
    """Return a short sha256 fingerprint for value, or None when empty."""

    if value is None or value == "":
        return None
    encoded = _stable_json(value).encode("utf-8", errors="replace")[:_MAX_HASH_BYTES]
    return hashlib.sha256(encoded).hexdigest()[:_HASH_PREFIX_LEN]


def _rough_token_estimate(value: Any) -> int:
    if value is None:
        return 0
    # Rough, deterministic estimate aligned with existing logging style. Avoid
    # importing tokenizer-heavy modules here; diagnostics must never fail the
    # request path.
    return max(1, len(_stable_json(value)) // 4)


def _system_message(messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if messages and messages[0].get("role") == "system":
        return messages[0]
    return None


def build_prompt_cache_diagnostics(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
    previous: PromptCacheDiagnostics | Mapping[str, Any] | None = None,
    skills_snapshot: Any | None = None,
) -> PromptCacheDiagnostics:
    """Compute local cache fingerprints and possible bust causes.

    ``previous`` should be the prior diagnostics object for the same live
    conversation. The resulting causes are deliberately phrased as possible
    causes, never as definitive provider cache-bust reasons.
    """

    system = _system_message(messages)
    non_system = list(messages[1:] if system else messages)
    # Prefix signal: all non-system messages except the current/tail message.
    # This changes when history/compression/prefills alter the reusable prefix,
    # but avoids treating every new user turn as an automatic full bust signal.
    message_prefix = non_system[:-1] if len(non_system) > 1 else []

    current = PromptCacheDiagnostics(
        system_prompt_hash=stable_hash(system.get("content") if system else None),
        tools_hash=stable_hash(tools or None),
        skills_hash=stable_hash(skills_snapshot),
        message_prefix_hash=stable_hash(message_prefix or None),
        provider=provider,
        model=model,
        session_id=session_id,
        system_prompt_tokens_estimate=_rough_token_estimate(system.get("content") if system else None),
        tools_tokens_estimate=_rough_token_estimate(tools or None),
        message_prefix_tokens_estimate=_rough_token_estimate(message_prefix or None),
    )

    causes: list[str] = []
    if previous:
        prev = previous.to_dict() if isinstance(previous, PromptCacheDiagnostics) else dict(previous)
        comparisons = (
            ("system_prompt_hash", "system prompt rebuilt or changed"),
            ("tools_hash", "tool schema hash changed"),
            ("skills_hash", "skill bundle hash changed"),
            ("message_prefix_hash", "conversation prefix changed or was compressed"),
            ("provider", "provider changed"),
            ("model", "model changed"),
            ("session_id", "session changed"),
        )
        for field, reason in comparisons:
            old = prev.get(field)
            new = getattr(current, field)
            if old and new and old != new:
                causes.append(reason)
            elif old and new is None:
                causes.append(reason)
            elif old is None and new:
                # The first appearance of a hash after an empty prior value may
                # affect prefix shape, but keep it a weak local signal.
                causes.append(reason)
    else:
        causes.append("first request in local diagnostics window")

    return PromptCacheDiagnostics(
        **{k: v for k, v in current.to_dict().items() if k != "possible_bust_causes"},
        possible_bust_causes=tuple(causes),
    )


def format_prompt_cache_diagnostics(diag: PromptCacheDiagnostics) -> str:
    """Human-readable one-line diagnostics for verbose logs."""

    causes = ", ".join(diag.possible_bust_causes) if diag.possible_bust_causes else "none detected"
    return (
        "Prompt cache diagnostics: "
        f"system={diag.system_prompt_hash or '-'} "
        f"tools={diag.tools_hash or '-'} "
        f"skills={diag.skills_hash or '-'} "
        f"prefix={diag.message_prefix_hash or '-'} "
        f"system_tokens~{diag.system_prompt_tokens_estimate:,} "
        f"tools_tokens~{diag.tools_tokens_estimate:,} "
        f"prefix_tokens~{diag.message_prefix_tokens_estimate:,}; "
        f"possible bust causes: {causes}"
    )
