"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

from typing import Any

from agent.secret_scope import get_secret as _get_secret
from utils import base_url_host_matches, base_url_hostname


_FALLBACK_API_MODES = frozenset({
    "chat_completions",
    "codex_responses",
    "anthropic_messages",
    "bedrock_converse",
})


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def normalize_fallback_api_mode(value: Any) -> str | None:
    """Return a supported in-process fallback transport, or ``None``.

    ``runtime_provider._parse_api_mode`` owns config-level normalization
    (including whitespace/case handling).  Fallback activation narrows that
    set to transports an already-running agent can switch to; the
    ``codex_app_server`` runtime is a whole-process mode rather than an
    in-process provider transport.
    """

    from hermes_cli.runtime_provider import _parse_api_mode

    parsed = _parse_api_mode(value)
    return parsed if parsed in _FALLBACK_API_MODES else None


def fallback_entry_hints(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize resolver inputs shared by every fallback entry consumer."""

    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()
    base_url = _normalized_base_url(entry.get("base_url")) or None

    raw_api_key = entry.get("api_key")
    api_key = raw_api_key.strip() if isinstance(raw_api_key, str) else None
    api_key = api_key or None
    if not api_key:
        key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
        if key_env:
            api_key = (_get_secret(key_env, "") or "").strip() or None

    # Ollama Cloud supports the documented provider-wide key without making
    # users repeat it in every fallback entry.  Keep the hostname check exact
    # so an attacker-controlled lookalike cannot receive the credential.
    if base_url and not api_key and base_url_host_matches(base_url, "ollama.com"):
        api_key = (_get_secret("OLLAMA_API_KEY", "") or "").strip() or None

    return {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "api_mode": normalize_fallback_api_mode(entry.get("api_mode")),
    }


def resolve_fallback_client(
    entry: dict[str, Any],
    *,
    raw_codex: bool = False,
) -> tuple[Any | None, str | None, str | None]:
    """Resolve a fallback client plus its validated explicit transport hint."""

    from agent.auxiliary_client import resolve_provider_client

    hints = fallback_entry_hints(entry)
    kwargs: dict[str, Any] = {
        "model": hints["model"],
        "raw_codex": raw_codex,
        "explicit_base_url": hints["base_url"] or "",
        "explicit_api_key": hints["api_key"] or "",
    }
    if hints["api_mode"]:
        kwargs["api_mode"] = hints["api_mode"]
    client, resolved_model = resolve_provider_client(hints["provider"], **kwargs)
    return client, resolved_model, hints["api_mode"]


def resolve_fallback_runtime(entry: dict[str, Any]) -> dict[str, Any]:
    """Resolve credentials/runtime and apply a validated entry transport.

    Invalid non-empty ``api_mode`` values are treated exactly like an absent
    hint: they are not forwarded and the runtime resolver's provider, URL,
    and target-model heuristics remain authoritative.
    """

    from hermes_cli.runtime_provider import resolve_runtime_provider

    hints = fallback_entry_hints(entry)
    kwargs: dict[str, Any] = {
        "requested": hints["provider"],
        "target_model": hints["model"],
    }
    if hints["base_url"]:
        kwargs["explicit_base_url"] = hints["base_url"]
    if hints["api_key"]:
        kwargs["explicit_api_key"] = hints["api_key"]

    runtime = resolve_runtime_provider(**kwargs)
    if hints["api_mode"]:
        runtime = dict(runtime)
        runtime["api_mode"] = hints["api_mode"]
    return runtime


def resolve_fallback_transport(
    *,
    validated_api_mode: str | None,
    provider: str,
    model_requires_responses: bool,
    base_url: str,
    is_azure: bool,
) -> str:
    """Resolve the effective transport after a fallback client is built."""

    if validated_api_mode:
        return validated_api_mode

    from hermes_cli.providers import determine_api_mode
    from hermes_cli.runtime_provider import _detect_api_mode_for_url

    # Start from the provider's declared transport without passing the URL:
    # determine_api_mode() intentionally accepts legacy substring URL matches
    # for known providers, which would misclassify lookalike hosts such as
    # api.anthropic.com.attacker.test.  The runtime URL detector below parses
    # hostnames and paths exactly, matching the primary-provider path.
    detected = determine_api_mode(provider, "")
    if detected != "chat_completions":
        return detected

    # Azure OpenAI serves GPT-5.x through chat/completions, not Responses.
    if is_azure:
        return "chat_completions"

    url_detected = _detect_api_mode_for_url(base_url)
    if url_detected:
        return url_detected

    hostname = base_url_hostname(base_url)
    if (
        hostname.startswith("bedrock-runtime.")
        and base_url_host_matches(base_url, "amazonaws.com")
    ):
        return "bedrock_converse"

    if model_requires_responses:
        return "codex_responses"
    return "chat_completions"


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain
