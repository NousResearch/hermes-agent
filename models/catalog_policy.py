"""Catalogue entitlement filtering over caller-supplied policy facts."""

from __future__ import annotations

from typing import Any, Optional

_NOUS_POLICY_APPEND_MAX = 64

def restrict_to_nous_policy(
    model_ids: list[str],
    allowed: Optional[set[str]],
    *,
    rescue_empty: bool = False,
) -> list[str]:
    """*model_ids* narrowed to *allowed*, preserving order. A ``:free`` sibling is kept when its
    base model is reachable (the gateway admits a row when any requestable id passes); over-listing
    costs a 403 from the authoritative gate, hiding a servable row is unrecoverable client-side.
    *rescue_empty*: an allowlist naming only models the curated manifest lacks would leave an empty
    picker — worse than no filter — so return the allowlist itself. Opt-in per list: an already-
    empty list (a paid tier's gated models) means "nothing to gate", not "nothing survived"."""
    if not allowed:
        return list(model_ids)
    kept = [mid for mid in model_ids if mid in allowed or mid.split(":", 1)[0] in allowed]
    if rescue_empty and not kept and len(allowed) <= _NOUS_POLICY_APPEND_MAX:
        return sorted(allowed)
    return kept


from typing import Optional
from providers import normalize_provider

_SELF_HOSTED_PROVIDERS = frozenset({
    "lmstudio", "ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp",
})

def _provider_token(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()

def _is_self_hosted_provider(provider: Optional[str]) -> bool:
    """Local servers and the custom endpoint bucket, including aliases that normalize to it."""
    raw = _provider_token(provider)
    if not raw:
        return False
    if raw == "custom" or raw.startswith("custom:"):
        return True

    normalized = normalize_provider(raw)
    if normalized == "custom" or normalized.startswith("custom:"):
        return True
    return raw in _SELF_HOSTED_PROVIDERS or normalized in _SELF_HOSTED_PROVIDERS

def _non_public_host(host: str) -> bool:
    """Loopback, LAN, and single-label hosts are a user's endpoint, never a vendor catalog."""
    host = (host or "").lower().rstrip(".")
    if not host or host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".localhost"):
        return bool(host)
    if host.endswith((".local", ".lan", ".internal", ".home", ".localdomain")):
        return True
    if "." not in host:
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        octets = [int(part) for part in parts]
        if octets[0] in {10, 127} or (octets[0] == 192 and octets[1] == 168):
            return True
        if octets[0] == 172 and 16 <= octets[1] <= 31:
            return True
    return False

def allows_model_whitespace(provider: Optional[str], base_url: Optional[str], stock_host: str) -> bool:
    """True when a spaced id is a real selection, not a cloud-catalog typo.

    Self-hosted providers always qualify. A base_url qualifies when the user
    configured it: a non-public host, or a public host that is not this
    provider's own stock endpoint. A public host with no stock to compare
    against stays rejected, so a cloud URL cannot slip through a cold catalog.
    """
    if _is_self_hosted_provider(provider):
        return True
    url = str(base_url or "").strip()
    if not url:
        return False
    from utils import base_url_hostname

    host = base_url_hostname(url)
    if not host:
        return False
    if _non_public_host(host):
        return True
    raw = _provider_token(provider)

    normalized = normalize_provider(raw) if raw else ""
    stock = stock_host
    return bool(stock) and host != stock
