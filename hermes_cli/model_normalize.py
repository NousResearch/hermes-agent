"""Per-provider model name normalization."""

from __future__ import annotations

import re
from typing import Optional

# First hyphen-delimited token of a bare model name → vendor slug used by aggregator APIs
# ("claude-sonnet-4.6" → "anthropic/claude-sonnet-4.6").
_VENDOR_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "google",
    "gemma": "google",
    "deepseek": "deepseek",
    "glm": "z-ai",
    "kimi": "moonshotai",
    "minimax": "minimax",
    "grok": "x-ai",
    "qwen": "qwen",
    "mimo": "xiaomi",
    "trinity": "arcee-ai",
    "nemotron": "nvidia",
    "llama": "meta-llama",
    "step": "stepfun"}

# Providers whose APIs consume vendor/model slugs.
_AGGREGATOR_PROVIDERS: frozenset[str] = frozenset({
    "openrouter", "nous", "ai-gateway", "kilocode"})

# Providers that want bare names with dots replaced by hyphens.
_DOT_TO_HYPHEN_PROVIDERS: frozenset[str] = frozenset({
    "anthropic"})

# Providers that want bare names with dots preserved.
_STRIP_VENDOR_ONLY_PROVIDERS: frozenset[str] = frozenset({
    "copilot", "copilot-acp", "openai-codex"})

# Providers whose native naming is authoritative -- pass through unchanged.
_AUTHORITATIVE_NATIVE_PROVIDERS: frozenset[str] = frozenset({
    "huggingface"})

# Direct providers that accept bare native names but should repair a matching
# provider/ prefix when users copy the aggregator form into config.yaml.
_MATCHING_PREFIX_STRIP_PROVIDERS: frozenset[str] = frozenset({
    "zai",
    "kimi-coding",
    # Providers whose endpoint does not accept image input, even though the provider's broader ecosystem has
    # vision models available elsewhere. When `auxiliary.vision.provider: auto` sees one of these as the
    # main provider, it must skip straight to the aggregator chain instead of returning a client that will
    # 404 on every vision request. kimi-coding / kimi-coding-cn: the Kimi Coding Plan routes through
    # api.kimi.com/coding (Anthropic Messages wire) which Kimi's own docs describe as having no image_in
    # capability. Vision lives on the separate Kimi Platform (api.moonshot.ai, OpenAI-wire, pay-as-you-go).
    # See #17076.
    "kimi-coding-cn",
    "minimax",
    "minimax-oauth",
    "minimax-cn",
    "alibaba",
    "qwen-oauth",
    "xiaomi",
    "arcee",
    "ollama-cloud",
    "nebius-token-factory",
    "custom",
    "gemini",
    "xai"})

# Providers whose API serves ``vendor/model`` ids but whose endpoint can also
# front arbitrary self-hosted models, so a bare name cannot be prefixed
# blindly. A bare id is repaired only when the curated catalogue for that
# provider holds exactly one entry ending in ``/<name>`` — a lookup, not a
# guess. NVIDIA NIM is the case in hand: build.nvidia.com serves
# ``nvidia/nemotron-…`` (and third-party ``z-ai/glm-…``), while the same
# provider id also points at local NIM containers with their own naming.
# Without this repair a bare ``nemotron-3-ultra-550b-a55b`` reaches the API
# and returns a bare ``404 page not found`` that never names the model (#78796).
_CATALOGUE_PREFIX_REPAIR_PROVIDERS: frozenset[str] = frozenset({
    "nvidia"})

# Providers whose APIs require lowercase model IDs (Xiaomi rejects ``MiMo-V2.5-Pro`` copied from
# marketing docs; only ``mimo-v2.5-pro`` works). Applied after matching-prefix stripping.
_LOWERCASE_MODEL_PROVIDERS: frozenset[str] = frozenset({
    "xiaomi"})

# DeepSeek's direct API only accepts first-class V-series IDs after the 2026-07-24 cut-off (HTTP 400
# otherwise). Both retired aliases map to deepseek-v4-flash per the official docs (thinking mode is
# controlled by extra_body.thinking on the profile), so saved configs can't keep sending them.
_DEEPSEEK_RETIRED_ALIASES: frozenset[str] = frozenset({
    "deepseek-chat", "deepseek-reasoner"})

_DEEPSEEK_CANONICAL_MODELS: frozenset[str] = frozenset({
    "deepseek-v4-pro", "deepseek-v4-flash"})

# First-class V-series IDs incl. future ``deepseek-v5-*`` and dated variants
# (``deepseek-v4-flash-20260423``): verified real model ids, NOT aliases of ``deepseek-chat``.
_DEEPSEEK_V_SERIES_RE = re.compile(r"^deepseek-v\d+([-.].+)?$")


def _normalize_for_deepseek(model_name: str) -> str:
    """Map a model input to a DeepSeek-accepted id: canonicals and ``deepseek-v<digit>…`` pass
    through (future V-series work without a release); retired aliases and everything else become
    ``deepseek-v4-flash``."""
    bare = _strip_vendor_prefix(model_name).lower()
    if bare in _DEEPSEEK_CANONICAL_MODELS or _DEEPSEEK_V_SERIES_RE.match(bare):
        return bare
    return "deepseek-v4-flash"


def _strip_vendor_prefix(model_name: str) -> str:
    """Remove a ``vendor/`` prefix if present."""
    return model_name.split("/", 1)[1] if "/" in model_name else model_name


def _dots_to_hyphens(model_name: str) -> str:
    return model_name.replace(".", "-")


def _normalize_provider_alias(provider_name: str) -> str:
    """Resolve provider aliases to Hermes' canonical ids."""
    raw = (provider_name or "").strip().lower()
    if not raw:
        return raw
    try:
        from hermes_cli.models import normalize_provider

        return normalize_provider(raw)
    except Exception:
        return raw


def _strip_matching_provider_prefix(model_name: str, target_provider: str) -> str:
    """Strip ``provider/`` only when the prefix matches the target provider, so arbitrary slash-bearing
    ids aren't mangled while ``zai/glm-5.1`` is repaired for ``zai``. ``custom`` is a bucket, not a
    vendor: an alias resolving to it (``ollama``) may be a real LiteLLM-style routing prefix, so only a
    literal ``custom/`` prefix is redundant there."""
    if "/" not in model_name:
        return model_name
    prefix, remainder = model_name.split("/", 1)
    if not prefix.strip() or not remainder.strip():
        return model_name
    normalized_target = _normalize_provider_alias(target_provider)
    if normalized_target == "custom":
        return remainder.strip() if prefix.strip().lower() == "custom" else model_name
    normalized_prefix = _normalize_provider_alias(prefix)
    return remainder.strip() if normalized_prefix and normalized_prefix == normalized_target else model_name


def detect_vendor(model_name: str) -> Optional[str]:
    """Vendor slug from a bare model name: an existing ``vendor/`` prefix, the first hyphen token,
    or a ``_VENDOR_PREFIXES`` key the name starts with (``qwen3.5-plus`` → ``qwen``)."""
    name = model_name.strip()
    if not name:
        return None
    if "/" in name:
        return name.split("/", 1)[0].lower() or None
    name_lower = name.lower()
    first_token = name_lower.split("-")[0]
    if first_token in _VENDOR_PREFIXES:
        return _VENDOR_PREFIXES[first_token]
    return next((vendor for prefix, vendor in _VENDOR_PREFIXES.items() if name_lower.startswith(prefix)), None)


def _prepend_vendor(model_name: str) -> str:
    """Prepend the detected ``vendor/`` for aggregators; names with ``/`` or no detectable vendor
    pass through (the aggregator may still accept them)."""
    if "/" in model_name:
        return model_name
    vendor = detect_vendor(model_name)
    return f"{vendor}/{model_name}" if vendor else model_name


def _repair_prefix_from_catalogue(model_name: str, provider: str) -> str:
    """Restore a dropped ``vendor/`` prefix only when the bare id matches **exactly one** curated
    entry for this provider modulo the prefix — a lookup, never a guess from name shape."""
    if "/" in model_name:
        return model_name
    try:
        from hermes_cli.models import _PROVIDER_MODELS
    except Exception:
        return model_name
    # Compare against the catalogue's own suffix, tag included: a bare ``…:free`` id must resolve to
    # the ``:free`` entry, not its paid sibling.
    needle = model_name.strip().lower()
    catalogue = _PROVIDER_MODELS.get(provider) or []
    matches = {e for e in catalogue if "/" in e and e.split("/", 1)[1].strip().lower() == needle}
    return matches.pop() if len(matches) == 1 else model_name


def suggest_prefixed_model_id(provider: str, model_name: str) -> Optional[str]:
    """Prefixed catalogue id for a bare *model_name* if unambiguous, else ``None`` — the diagnostic
    counterpart to :func:`_repair_prefix_from_catalogue` for explaining a content-free 404."""
    name = (model_name or "").strip()
    if not name or "/" in name:
        return None
    try:
        canonical = _normalize_provider_alias(provider)
    except Exception:
        return None
    repaired = _repair_prefix_from_catalogue(name, canonical)
    return repaired if repaired != name else None


def normalize_model_for_provider(model_input: str, target_provider: str) -> str:
    """Translate a model name (bare, vendor-prefixed or native) into what the target provider's API
    expects. ``target_provider`` should already be canonical. Never raises."""
    name = (model_input or "").strip()
    if not name:
        return name
    provider = _normalize_provider_alias(target_provider)

    if provider in _AGGREGATOR_PROVIDERS:
        return _prepend_vendor(name)

    # OpenCode Zen / Go are flat-namespace resellers: /v1/models returns bare IDs and inference 401s
    # vendor-prefixed names, so strip ANY leading ``vendor/`` (commonly copied from aggregator slugs).
    from hermes_cli.models import opencode_provider_family

    _oc_family = opencode_provider_family(provider)
    if _oc_family is not None:
        if "/" in name:
            name = name.split("/", 1)[1].strip() or name
        if _oc_family == "opencode-zen" and name.lower().startswith("claude-"):
            return _dots_to_hyphens(name)
        return name

    if provider in _DOT_TO_HYPHEN_PROVIDERS:
        bare = _strip_matching_provider_prefix(name, provider)
        return bare if "/" in bare else _dots_to_hyphens(bare)

    # Copilot's own normalizer knows the alias table (vendor stripping, dash-to-dot repair for Claude)
    # and live-catalog lookups; without it dash-notation Claude ids hit HTTP 400 model_not_supported.
    # See issue #6879.
    if provider in {"copilot", "copilot-acp"}:
        try:
            from hermes_cli.models import normalize_copilot_model_id

            normalized = normalize_copilot_model_id(name)
            if normalized:
                return normalized
        except Exception:
            pass  # fall through to the generic strip-vendor behaviour

    if provider in _STRIP_VENDOR_ONLY_PROVIDERS:
        stripped = _strip_matching_provider_prefix(name, provider)
        if stripped == name and name.startswith("openai/"):
            return name.split("/", 1)[1]  # openai-codex maps openai/gpt-5.4 -> gpt-5.4
        return stripped

    if provider == "deepseek":
        bare = _strip_matching_provider_prefix(name, provider)
        return bare if "/" in bare else _normalize_for_deepseek(bare)

    if provider in _MATCHING_PREFIX_STRIP_PROVIDERS:
        result = _strip_matching_provider_prefix(name, provider)
        return result.lower() if provider in _LOWERCASE_MODEL_PROVIDERS else result

    # Unknown names (a local NIM container, a proxied model) pass through untouched.
    if provider in _CATALOGUE_PREFIX_REPAIR_PROVIDERS:
        return _repair_prefix_from_catalogue(name, provider)

    # Authoritative native providers, custom and all others: pass through as-is.
    return name
