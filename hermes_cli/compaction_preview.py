"""Read-only preview of the compaction trigger a session installs for a model route.

Published plugin API (``hermes_cli/data/plugin_api.json``). A plugin that explains or checks
compaction asks here instead of importing the config parser, the compressor's trigger maths and
the context-window lookup, which move between releases.

Nothing here uses a credential: no base URL or API key reaches the window lookup, a Bedrock window
comes from the offline table instead of the AWS probe, and a Mixture of Agents preset is read from
config instead of resolving its aggregator's runtime. The lookup may still fetch a public model
list (OpenRouter's, Novita's), exactly as it does for a session.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class ContextWindow:
    """A route's context window and where it came from.

    ``source`` is ``"config"`` (``model.context_length``), ``"custom_provider"`` (a per-model
    ``context_length`` on a custom provider entry), ``"resolved"`` (a catalog knew the model) or
    ``"default"`` (nothing knew it: a static default, not a detected limit).
    """

    tokens: int
    source: str


@dataclass(frozen=True)
class CompactionPreview:
    """The trigger a session on this route would install, and the factors behind it.

    ``model`` and ``provider`` are the identity the session runs (provider alias resolved, model id
    in the provider's notation). ``context_source`` is a :class:`ContextWindow` source, or
    ``"caller"`` when the window was passed in. ``threshold_tokens`` is the prompt size at which
    automatic compaction fires. ``base_percent`` is ``compression.threshold`` after
    ``model_thresholds`` and the per-model overrides; ``effective_percent`` adds the small-window
    floor. ``matched_rule`` is the ``model_thresholds`` key that won, ``output_reservation`` the
    output tokens carved out of the window, and ``threshold_cap`` is ``compression.threshold_tokens``
    clamped to the window.
    """

    model: str
    provider: str
    context_length: int
    context_source: str
    threshold_tokens: int
    base_percent: float
    effective_percent: float
    matched_rule: Optional[str]
    output_reservation: Optional[int]
    threshold_cap: Optional[int]


def _runtime_identity(model: str, provider: str) -> tuple[str, str]:
    """(model, provider) as a session runs them. An empty or ``auto`` provider is resolved from
    credentials at session start, never from the name, so it is left alone."""
    from hermes_cli.model_normalize import _AGGREGATOR_PROVIDERS, normalize_model_for_provider
    from hermes_cli.models import normalize_provider

    model, raw = (model or "").strip(), (provider or "").strip().lower()
    if raw in ("", "auto"):
        return model, raw
    canonical = normalize_provider(raw)
    if not model or canonical in _AGGREGATOR_PROVIDERS:
        return model, canonical
    return normalize_model_for_provider(model, canonical), canonical


def _section(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    return value if isinstance(value, Mapping) else {}


def _pinned_window(config: Mapping[str, Any], model: str) -> Optional[ContextWindow]:
    """The main route's configured window: ``model.context_length``, else a per-model
    ``context_length`` on the custom provider entry that serves ``model.base_url``."""
    from agent.agent_init import _positive_int
    from hermes_cli.config import get_compatible_custom_providers, get_custom_provider_context_length

    model_cfg = _section(config, "model")
    pin = _positive_int(model_cfg.get("context_length"), reject=(bool,))
    if pin is not None:
        return ContextWindow(pin, "config")
    base_url = str(model_cfg.get("base_url") or "")
    if not base_url:
        return None
    custom = get_custom_provider_context_length(
        model=model, base_url=base_url, custom_providers=get_compatible_custom_providers(dict(config)),
    )
    return ContextWindow(int(custom), "custom_provider") if custom else None


def _bedrock_window(model: str) -> Optional[ContextWindow]:
    """The offline Bedrock table; the full lookup would probe AWS with the machine's credentials."""
    try:
        from agent.bedrock_adapter import BEDROCK_DEFAULT_CONTEXT_LENGTH, bedrock_static_context_length
    except ImportError:  # no boto3: the resolver skips Bedrock too, and never probes
        return None
    known = bedrock_static_context_length(model)
    return ContextWindow(known, "resolved") if known else ContextWindow(BEDROCK_DEFAULT_CONTEXT_LENGTH, "default")


def _moa_window(config: Mapping[str, Any], preset: str) -> ContextWindow:
    """The window of a Mixture of Agents preset's aggregator, read from config alone; the full
    lookup resolves the aggregator's runtime (base URL and API key) and probes with it."""
    from agent.errors import MoAPresetNotFoundError
    from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
    from hermes_cli.moa_config import resolve_moa_preset

    try:
        aggregator = resolve_moa_preset(config.get("moa") or {}, preset).get("aggregator") or {}
    except MoAPresetNotFoundError:
        return ContextWindow(DEFAULT_FALLBACK_CONTEXT, "default")
    model, provider = _runtime_identity(str(aggregator.get("model") or ""), str(aggregator.get("provider") or ""))
    if not model or provider in ("", "moa"):
        return ContextWindow(DEFAULT_FALLBACK_CONTEXT, "default")
    return _window(config, model, provider, main_route=False)


def _window(config: Mapping[str, Any], model: str, provider: str, *, main_route: bool) -> ContextWindow:
    """:func:`resolve_context_window` for an identity that is already the session's."""
    from agent.context_default_record import last_defaulted_context_length
    from agent.model_metadata import get_model_context_length

    pinned = _pinned_window(config, model) if main_route else None
    if pinned is not None:
        return pinned
    if provider == "bedrock" and (bedrock := _bedrock_window(model)) is not None:
        return bedrock
    if provider == "moa":
        return _moa_window(config, model)
    tokens = int(get_model_context_length(model, provider=provider))
    return ContextWindow(tokens, "default" if last_defaulted_context_length() == tokens else "resolved")


def resolve_context_window(
    config: Mapping[str, Any], *, model: str, provider: str = "", main_route: bool = True,
) -> ContextWindow:
    """The context window of ``model`` on ``provider`` under ``config``, without any credential.

    ``main_route`` says the route is the configured default one, so ``model.context_length`` and
    the custom provider entry behind ``model.base_url`` apply; pass ``False`` for a fallback,
    delegation or auxiliary route. Catalog caches and ``model_overrides`` are read from the active
    Hermes home.
    """
    return _window(config, *_runtime_identity(model, provider), main_route=main_route)


def threshold_rule_applies(key: str, *, model: str, provider: str = "") -> bool:
    """Whether a ``compression.model_thresholds`` key matches this route, by the compressor's own
    ranking: a bare key is a substring of the model id, ``"<provider>:<substr>"`` also needs the provider."""
    from agent.context_compressor import _model_threshold_key_rank

    model, provider = _runtime_identity(model, provider)
    return _model_threshold_key_rank(str(key), model, provider) is not None


def preview_compaction(
    config: Mapping[str, Any], *, model: str, provider: str = "", context_length: Optional[int] = None,
    main_route: bool = True, max_tokens: Optional[int] = None,
) -> CompactionPreview:
    """The compaction trigger a session on this route would install under ``config``.

    ``config`` is an effective config mapping (``hermes_cli.config.load_config()``). Pass
    ``context_length`` when the window is already known (the ``post_api_request`` hook reports
    the live one); otherwise it is resolved by :func:`resolve_context_window`. ``max_tokens`` is
    the session's output-token cap when it sets one. Raises ``ValueError`` or ``TypeError`` for a
    ``compression`` value a session start rejects too.
    """
    from agent.agent_init import _compressor_max_tokens, _parse_compression_config
    from agent.context_compressor import ContextCompressor, _model_threshold_key_rank

    model, provider = _runtime_identity(model, provider)
    window = (
        ContextWindow(int(context_length), "caller") if context_length
        else _window(config, model, provider, main_route=main_route)
    )
    agent = SimpleNamespace(model=model, provider=provider, api_mode="", base_url="", max_tokens=max_tokens)
    settings = _parse_compression_config(agent, dict(config))
    compressor = ContextCompressor(
        model=model, threshold_percent=settings.threshold, provider=provider,
        max_tokens=_compressor_max_tokens(agent), model_thresholds=settings.model_thresholds,
        threshold_tokens_cap=settings.threshold_tokens, quiet_mode=True,
    )
    base, effective, threshold = compressor._derive_trigger(model, window.tokens, provider)
    ranked = ((_model_threshold_key_rank(key, model, provider), key) for key in settings.model_thresholds)
    best = max(((rank, key) for rank, key in ranked if rank is not None), default=None)
    return CompactionPreview(
        model=model, provider=provider, context_length=window.tokens, context_source=window.source,
        threshold_tokens=int(threshold), base_percent=float(base), effective_percent=float(effective),
        matched_rule=best[1] if best else None, output_reservation=compressor.max_tokens,
        threshold_cap=compressor._effective_threshold_cap(window.tokens),
    )
