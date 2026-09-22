"""OpenCode Zen / Go family routing.

Family membership, flat-namespace model ids, per-model ``api_mode`` and the symmetric base-url
normalization (``/zen`` vs ``/zen/go`` relay paths, ``/v1`` heal). Split out of
``hermes_cli.models``; ``normalize_provider`` still lives there and is imported at call time so
``patch("hermes_cli.models.normalize_provider")`` keeps intercepting.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional


_OPENCODE_FAMILIES = ("opencode-go", "opencode-zen")


def opencode_provider_family(provider_id: Optional[str]) -> Optional[str]:
    """Resolve a provider id (canonical or prefixed) to its OpenCode family, or None.

    Returns ``"opencode-zen"`` or ``"opencode-go"`` for the built-in providers AND for custom providers
    whose name extends a family slug (e.g. ``opencode-go-bridge`` pointing at
    ``https://opencode.ai/zen/go/v1``, issue #85589). Matching is case-insensitive. Custom family providers
    need the same per-model api_mode routing and /v1 base-url normalization as the built-ins — this
    predicate is the single owner of that family-membership question; do not re-implement it inline.
    """
    from hermes_cli.models import normalize_provider
    raw = str(provider_id or "").strip().lower()
    if not raw:
        return None
    canonical = normalize_provider(provider_id)
    if canonical in _OPENCODE_FAMILIES:
        return canonical
    return next((f for f in _OPENCODE_FAMILIES if raw.startswith(f)), None)


def normalize_opencode_model_id(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Normalize OpenCode config IDs to the bare model slug used in API requests."""
    family = opencode_provider_family(provider_id)
    current = str(model_id or "").strip()
    if not current or family is None:
        return current
    for prefix in (f"{provider_id or family}/", f"{family}/"):
        if current.lower().startswith(prefix.lower()):
            return current[len(prefix):]
    return current


# Per-family (model-id prefix → api_mode) routing from OpenCode's published Zen/Go endpoint
# tables, checked in order. GPT/Codex/Grok and Muse Spark use /v1/responses (Muse Spark 503s on
# chat/completions); Claude (Zen), MiniMax (Go), Union Alpha, and Qwen use /v1/messages;
# everything else falls through to /v1/chat/completions.
_OPENCODE_API_MODE_PREFIXES: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {
    "opencode-go": (
        (("gpt-", "grok-", "muse-spark"), "codex_responses"),
        (("minimax-", "qwen", "union-alpha"), "anthropic_messages")),
    "opencode-zen": (
        (("claude-", "union-alpha"), "anthropic_messages"), (("gpt-", "grok-", "muse-spark"), "codex_responses"),
        (("qwen",), "anthropic_messages"))}


def opencode_model_api_mode(provider_id: Optional[str], model_id: Optional[str]) -> str:
    """Determine the API mode for an OpenCode Zen / Go model (see ``_OPENCODE_API_MODE_PREFIXES``)."""
    family = opencode_provider_family(provider_id)
    normalized = normalize_opencode_model_id(provider_id, model_id).lower()
    if normalized:
        for prefixes, mode in _OPENCODE_API_MODE_PREFIXES.get(family or "", ()):
            if normalized.startswith(prefixes):
                return mode
    return "chat_completions"


# Relay path per OpenCode family on opencode.ai hosts.
_OPENCODE_FAMILY_PATHS = {"opencode-zen": "/zen", "opencode-go": "/zen/go"}


def normalize_opencode_base_url(
    provider_id: Optional[str], api_mode: Optional[str], base_url: Optional[str]) -> str:
    """Normalize an OpenCode Zen / Go base URL for the API mode. Must be SYMMETRIC: the anthropic-
    stripped URL gets persisted to ``model.base_url`` after switching into an anthropic-routed model,
    and chat/codex modes heal it by re-adding ``/v1`` — but only on opencode.ai hosts, so custom
    ``OPENCODE_*_BASE_URL`` proxies are left alone. On those hosts the relay path segment follows
    the resolved family too (``/zen`` vs ``/zen/go``): the two relays serve different model sets,
    so a ``model.base_url`` carried over from the other family 401s ("Model ... is not supported").
    The family heal applies to the BUILT-IN providers only: a custom provider merely named after a
    family (``opencode-go-bridge``) declared its relay path explicitly in ``providers:`` and keeps it.
    Only the path is edited, so a port, userinfo, query or fragment round-trips untouched."""
    from hermes_cli.models import normalize_provider
    url = str(base_url or "").strip().rstrip("/")
    family = opencode_provider_family(provider_id)
    if not url or family is None:
        return url
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    official = host == "opencode.ai" or host.endswith(".opencode.ai")
    path = parsed.path.rstrip("/")
    if official and normalize_provider(provider_id) in _OPENCODE_FAMILIES and re.fullmatch(r"/zen(/go)?(/v1)?", path):
        path = _OPENCODE_FAMILY_PATHS[family] + ("/v1" if path.endswith("/v1") else "")
    if api_mode == "anthropic_messages":
        path = re.sub(r"/v1$", "", path)
    elif official and not path.endswith("/v1"):
        path += "/v1"
    return urllib.parse.urlunparse(parsed._replace(path=path))
