"""Delegation model-profile resolver — operator-defined worker tiers for delegate_task.

``delegation.profiles`` maps a profile name to a provider/model pin (plus optional
reasoning_effort, max_iterations, fallback chain). This module is the single place profiles
are parsed, validated, and resolved into an immutable :class:`ProfileRoute` that both
``delegate_task`` and ``SubagentLifecycleService`` consume.

Selection precedence (see docs/plans/2026-09-04-delegation-model-profiles.md):
  per-task profile > top-level profile > ``delegation.default_profile`` > legacy
  ``delegation.provider/model`` (this module returns no profile) > parent inherit.

Design rules honored here:
- Credentials resolve via ``hermes_cli.runtime_provider.resolve_runtime_provider`` — the exact
  path the legacy ``delegation.provider`` branch uses. No duplicated credential logic.
- Invalid ``reasoning_effort`` warns and is ignored (NS-696 clamp-at-transport doctrine), it
  never rejects a profile.
- Profiles never touch toolsets (#25752) — ``_PROFILE_KEYS`` is a closed set.
- Imports of heavyweight collaborators are lazy (house style, avoids import cycles).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PROFILE_KEYS = frozenset({"provider", "model", "reasoning_effort", "max_iterations", "fallback"})
_FALLBACK_ENTRY_KEYS = frozenset({"provider", "model"})


@dataclass(frozen=True)
class FallbackTarget:
    """One provider/model pair in a profile's fallback chain."""
    provider: str
    model: str


@dataclass(frozen=True)
class ProfileSpec:
    """A parsed-and-validated ``delegation.profiles`` entry (no credentials yet)."""
    name: str
    provider: str
    model: str
    reasoning_config: Optional[dict] = None      # parse_reasoning_effort output, None = inherit
    max_iterations: Optional[int] = None
    fallback: Tuple[FallbackTarget, ...] = ()


@dataclass(frozen=True)
class ProfileRoute:
    """An immutable resolved route: profile spec + runtime-provider credentials."""
    requested_profile: str
    provider: Optional[str]
    model: str
    base_url: Optional[str]
    api_key: Optional[str]
    api_mode: Optional[str]
    reasoning_config: Optional[dict] = None
    max_iterations: Optional[int] = None
    fallback: Tuple[FallbackTarget, ...] = ()
    supports_tools: bool = True
    request_overrides: Optional[dict] = None     # provider request personality (runtime provider)
    max_output_tokens: Optional[int] = None


def _profiles_section(cfg: Optional[dict]) -> Any:
    return ((cfg or {}).get("profiles"))


def _parse_fallback(name: str, raw: Any) -> Tuple[FallbackTarget, ...]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, list):
        raise ValueError(
            f"delegation.profiles.{name}.fallback must be a list of {{provider, model}} entries, "
            f"got {type(raw).__name__}")
    targets: List[FallbackTarget] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] must be a dict with provider + model, "
                f"got {type(entry).__name__}")
        unknown = set(entry) - _FALLBACK_ENTRY_KEYS
        if unknown:
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] has unknown keys: {sorted(unknown)} "
                f"(allowed: provider, model)")
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            raise ValueError(
                f"delegation.profiles.{name}.fallback[{i}] needs both 'provider' and 'model'")
        targets.append(FallbackTarget(provider=provider, model=model))
    return tuple(targets)


def _parse_profile(name: str, raw: Any) -> ProfileSpec:
    if not str(name).strip():
        raise ValueError(
            "delegation.profiles contains an empty or whitespace-only profile name; "
            "every profile needs a non-empty name")
    if not isinstance(raw, dict):
        raise ValueError(
            f"delegation.profiles.{name} must be a mapping (provider/model/...), "
            f"got {type(raw).__name__}")
    unknown = set(raw) - _PROFILE_KEYS
    if unknown:
        raise ValueError(
            f"delegation.profiles.{name} has unknown keys: {sorted(unknown)} "
            f"(allowed: {sorted(_PROFILE_KEYS)})")
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError(f"delegation.profiles.{name} is missing required 'model'")
    provider = str(raw.get("provider") or "").strip()

    # Reasoning effort: NS-696 clamp doctrine — invalid values warn and are ignored, never reject.
    reasoning_config = None
    effort = raw.get("reasoning_effort")
    if effort or effort is False:
        from hermes_constants import parse_reasoning_effort
        reasoning_config = parse_reasoning_effort(effort)
        if reasoning_config is None:
            logger.warning(
                "delegation.profiles.%s.reasoning_effort '%s' is not a recognized level; "
                "ignoring it (child inherits the parent's reasoning level)", name, effort)

    max_iterations = raw.get("max_iterations")
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValueError(
                f"delegation.profiles.{name}.max_iterations must be an integer, "
                f"got {type(max_iterations).__name__}")

    return ProfileSpec(
        name=name, provider=provider, model=model, reasoning_config=reasoning_config,
        max_iterations=max_iterations, fallback=_parse_fallback(name, raw.get("fallback")),
    )


def parse_profiles(cfg: Optional[dict]) -> Dict[str, ProfileSpec]:
    """Parse+validate ``delegation.profiles`` from the delegation config section.

    Returns ``{}`` when no profiles are configured. Raises ``ValueError`` with an actionable
    message on the first malformed profile (use :func:`profile_config_errors` to collect all).
    """
    raw = _profiles_section(cfg)
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"delegation.profiles must be a mapping of name -> profile, got {type(raw).__name__}")
    return {str(name): _parse_profile(str(name), spec) for name, spec in raw.items()}


def profile_config_errors(cfg: Optional[dict]) -> List[str]:
    """All profile config problems as messages (for ``hermes config check``); never raises."""
    errors: List[str] = []
    raw = _profiles_section(cfg)
    specs: Dict[str, ProfileSpec] = {}
    if raw and not isinstance(raw, dict):
        errors.append(
            f"delegation.profiles must be a mapping of name -> profile, got {type(raw).__name__}")
    elif isinstance(raw, dict):
        for name, spec in raw.items():
            try:
                specs[str(name)] = _parse_profile(str(name), spec)
            except ValueError as exc:
                errors.append(str(exc))
    default_profile = str((cfg or {}).get("default_profile") or "").strip()
    if default_profile and default_profile not in specs:
        configured = ", ".join(sorted(specs)) or "(none)"
        errors.append(
            f"delegation.default_profile '{default_profile}' is not a configured profile "
            f"(configured: {configured})")
    return errors


def select_profile_name(task_profile: Any, top_profile: Any, cfg: Optional[dict]) -> Optional[str]:
    """The profile name that applies, by precedence: per-task > top-level > default_profile.

    ``None`` = no profile applies (legacy ``delegation.provider/model`` or parent inherit take
    over downstream). Falsy values (None/False/''/0) at any level fall through to the next.
    """
    for candidate in (task_profile, top_profile, (cfg or {}).get("default_profile")):
        if candidate:
            name = str(candidate).strip()
            if name:
                return name
    return None


def resolve_profile_route(name: str, cfg: Optional[dict], parent_agent: Any = None) -> ProfileRoute:
    """Resolve profile *name* into an immutable :class:`ProfileRoute`.

    Raises ``ValueError`` (before any child construction) for an unknown name, listing the
    configured profile names. Credentials come from ``resolve_runtime_provider`` — the same
    ladder the legacy ``delegation.provider`` branch uses.
    """
    specs = parse_profiles(cfg)
    key = str(name or "").strip()
    if key not in specs:
        configured = ", ".join(sorted(specs)) or "(none configured)"
        raise ValueError(
            f"Unknown delegation profile '{key}'. Configured profiles: {configured}. "
            f"Add it under delegation.profiles in config.yaml or pick a configured name.")
    spec = specs[key]

    # Lazy imports (house style — delegate_tool_config.py does the same) to avoid import cycles.
    from hermes_cli.runtime_provider import resolve_runtime_provider
    # Same wrap as the legacy provider branch (tools/delegate_tool_config.py::
    # _runtime_provider_credentials): AuthError subclasses RuntimeError, and the delegate_task /
    # lifecycle seams only catch ValueError — anything else would escape as a raw traceback
    # instead of a clean tool_error.
    try:
        runtime = resolve_runtime_provider(
            requested=spec.provider or None, target_model=spec.model) or {}
    except Exception as exc:
        raise ValueError(
            f"Cannot resolve delegation profile '{key}' "
            f"(provider '{spec.provider or 'auto'}'): {exc}. "
            f"Check that the provider is configured (API key set, valid provider name), "
            f"or fix the entry under delegation.profiles in config.yaml."
        ) from exc

    supports_tools = True
    try:
        from agent.models_dev import get_model_capabilities
        caps = get_model_capabilities(spec.provider or "", spec.model)
        if caps is not None:
            supports_tools = bool(getattr(caps, "supports_tools", True))
    except Exception as exc:  # capability metadata is advisory; never block resolution on it
        logger.debug("Could not load model capabilities for profile '%s': %s", key, exc)

    return ProfileRoute(
        requested_profile=key,
        provider=runtime.get("provider") or spec.provider or None,
        model=spec.model,
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        reasoning_config=spec.reasoning_config,
        max_iterations=spec.max_iterations,
        fallback=spec.fallback,
        supports_tools=supports_tools,
        request_overrides=dict(runtime.get("request_overrides") or {}) or None,
        max_output_tokens=runtime.get("max_output_tokens"),
    )
