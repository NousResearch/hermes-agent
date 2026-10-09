"""Reusable provider-owned model-ID normalization policies."""

from __future__ import annotations

from typing import Iterable

from providers.base import ProviderProfile


_VENDOR_PREFIXES = {
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
    "step": "stepfun",
}

_COPILOT_MODEL_ALIASES = {
    "openai/gpt-5": "gpt-5-mini",
    "openai/gpt-5-chat": "gpt-5-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1",
    "openai/gpt-4.1-nano": "gpt-4.1",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/o1": "gpt-5.2",
    "openai/o1-mini": "gpt-5-mini",
    "openai/o1-preview": "gpt-5.2",
    "openai/o3": "gpt-5.3-codex",
    "openai/o3-mini": "gpt-5-mini",
    "openai/o4-mini": "gpt-5-mini",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4-6": "claude-opus-4.6",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4-0": "claude-sonnet-4",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4-5": "claude-haiku-4.5",
}

_DEEPSEEK_RETIRED_ALIASES = {
    "deepseek-chat": "deepseek-flash",
    "deepseek-reasoner": "deepseek-flash",
}


def _known_ids(values: Iterable[str]) -> set[str]:
    return {str(value or "").strip() for value in values if str(value or "").strip()}


def _prefixes(profile: ProviderProfile) -> set[str]:
    return {profile.name.lower(), *(str(alias).lower() for alias in profile.aliases)}


def strip_matching_prefix(
    profile: ProviderProfile,
    model: str,
    *,
    extra_prefixes: Iterable[str] = (),
    excluded_prefixes: Iterable[str] = (),
) -> str:
    """Strip a leading provider/model or provider:model prefix when owned here."""

    value = str(model or "").strip()
    cut = min((i for i in (value.find("/"), value.find(":")) if i >= 0), default=-1)
    if cut < 0:
        return value
    prefix, remainder = value[:cut].strip().lower(), value[cut + 1 :].strip()
    if not prefix or not remainder:
        return value
    accepted = _prefixes(profile) | {str(item).strip().lower() for item in extra_prefixes}
    accepted -= {str(item).strip().lower() for item in excluded_prefixes}
    if profile.name == "custom":
        accepted = {"custom"}
    return remainder if prefix in accepted else value


def vendor_for_model(model: str) -> str | None:
    value = str(model or "").strip()
    if not value:
        return None
    if "/" in value:
        return value.split("/", 1)[0].lower() or None
    lower = value.lower()
    token = lower.split("-", 1)[0]
    if token in _VENDOR_PREFIXES:
        return _VENDOR_PREFIXES[token]
    return next((vendor for prefix, vendor in _VENDOR_PREFIXES.items() if lower.startswith(prefix)), None)


def prepend_vendor(model: str) -> str:
    value = str(model or "").strip()
    if "/" in value:
        return value
    vendor = vendor_for_model(value)
    return f"{vendor}/{value}" if vendor else value


def repair_prefix_from_known_ids(model: str, known_ids: Iterable[str]) -> str:
    value = str(model or "").strip()
    if not value or "/" in value:
        return value
    needle = value.lower()
    matches = {
        candidate
        for candidate in _known_ids(known_ids)
        if "/" in candidate and candidate.split("/", 1)[1].strip().lower() == needle
    }
    return matches.pop() if len(matches) == 1 else value


def normalize_copilot_id(model: str, known_ids: Iterable[str]) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    aliases = _COPILOT_MODEL_ALIASES
    if raw in aliases:
        return aliases[raw]
    candidates = [raw]
    if "/" in raw:
        candidates.append(raw.split("/", 1)[1].strip())
    if raw.endswith(("-mini", "-nano", "-chat")):
        candidates.append(raw[:-5])
    known = _known_ids(known_ids)
    for candidate in dict.fromkeys(candidates):
        if candidate in aliases:
            return aliases[candidate]
        if candidate in known:
            return candidate
    if "/" in raw:
        stripped = raw.split("/", 1)[1].strip()
        if stripped and "/" not in stripped:
            return stripped
    return raw


def normalize_deepseek_id(profile: ProviderProfile, model: str) -> str:
    bare = strip_matching_prefix(profile, model)
    if "/" in bare:
        return bare
    lowered = bare.lower()
    return _DEEPSEEK_RETIRED_ALIASES.get(lowered, lowered)


class MatchingPrefixModelIdsMixin:
    model_prefix_exclusions: tuple[str, ...] = ()

    def normalize_model_id(self, model: str, *, known_ids: Iterable[str] = ()) -> str:
        return strip_matching_prefix(
            self, model, excluded_prefixes=self.model_prefix_exclusions
        )


class LowercaseMatchingPrefixModelIdsMixin:
    model_prefix_exclusions: tuple[str, ...] = ()

    def normalize_model_id(self, model: str, *, known_ids: Iterable[str] = ()) -> str:
        return strip_matching_prefix(
            self, model, excluded_prefixes=self.model_prefix_exclusions
        ).lower()


class VendorQualifiedModelIdsMixin:
    def normalize_model_id(self, model: str, *, known_ids: Iterable[str] = ()) -> str:
        return prepend_vendor(model)


class MatchingPrefixProviderProfile(MatchingPrefixModelIdsMixin, ProviderProfile): pass


class LowercaseMatchingPrefixProviderProfile(LowercaseMatchingPrefixModelIdsMixin, ProviderProfile): pass


class VendorQualifiedProviderProfile(VendorQualifiedModelIdsMixin, ProviderProfile): pass
