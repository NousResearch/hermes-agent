"""Model/backend sampling defaults inspired by Forge backend registries.

The registry is deliberately conservative: it only supplies request fields when
there is a known local/CLI backend pattern and the caller has not already set
that field. Cloud/provider profiles and explicit request_overrides keep priority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SamplingProfile:
    name: str
    match_providers: tuple[str, ...] = ()
    match_base_url_contains: tuple[str, ...] = ()
    match_model_contains: tuple[str, ...] = ()
    request_defaults: Mapping[str, Any] | None = None
    extra_body_defaults: Mapping[str, Any] | None = None


DEFAULT_SAMPLING_PROFILES: tuple[SamplingProfile, ...] = (
    SamplingProfile(
        name="ollama_tool_stable",
        match_providers=("ollama",),
        match_base_url_contains=("localhost:11434", "127.0.0.1:11434"),
        request_defaults={"temperature": 0.2},
        extra_body_defaults={"top_p": 0.9},
    ),
    SamplingProfile(
        name="llama_cpp_tool_stable",
        match_providers=("llama-cpp", "llamacpp", "llama.cpp"),
        match_base_url_contains=("localhost:8080", "127.0.0.1:8080"),
        request_defaults={"temperature": 0.1},
        extra_body_defaults={"top_p": 0.9},
    ),
    SamplingProfile(
        name="lmstudio_tool_stable",
        match_providers=("lmstudio",),
        match_base_url_contains=("localhost:1234", "127.0.0.1:1234"),
        request_defaults={"temperature": 0.2},
        extra_body_defaults={"top_p": 0.9},
    ),
)


def resolve_sampling_profile(
    *,
    provider: str | None,
    base_url: str | None,
    model: str | None,
    profiles: tuple[SamplingProfile, ...] = DEFAULT_SAMPLING_PROFILES,
) -> SamplingProfile | None:
    provider_l = (provider or "").strip().lower()
    base_l = (base_url or "").strip().lower()
    model_l = (model or "").strip().lower()
    for profile in profiles:
        if profile.match_providers and provider_l in {p.lower() for p in profile.match_providers}:
            return profile
        if profile.match_base_url_contains and any(part.lower() in base_l for part in profile.match_base_url_contains):
            return profile
        if profile.match_model_contains and any(part.lower() in model_l for part in profile.match_model_contains):
            return profile
    return None


def apply_sampling_defaults(api_kwargs: dict[str, Any], profile: SamplingProfile | None) -> dict[str, Any]:
    """Apply sampling defaults without overriding explicit caller fields."""
    if profile is None:
        return api_kwargs
    for key, value in (profile.request_defaults or {}).items():
        api_kwargs.setdefault(key, value)
    extra_defaults = dict(profile.extra_body_defaults or {})
    if extra_defaults:
        existing = api_kwargs.get("extra_body")
        if isinstance(existing, dict):
            merged = dict(extra_defaults)
            merged.update(existing)
            api_kwargs["extra_body"] = merged
        else:
            api_kwargs["extra_body"] = extra_defaults
    return api_kwargs
