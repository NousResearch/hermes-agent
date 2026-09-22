"""Canonical reasoning-effort vocabulary and wire clamping.

Hermes' internal effort ladder (``VALID_REASONING_EFFORTS`` plus ``none``) is wider than any
single provider wire accepts; hand-rolled per-transport maps leaked new levels (``ultra``) to
wires that 400 and inverted the ladder (unknown → weak default). Single source of truth:
:data:`EFFORT_LADDER` (low→high), :func:`clamp_effort` (verbatim if supported, else the
nearest WEAKER level; only when nothing weaker exists the weakest supported), and named
wire-vocabulary constants so call sites declare data. Rules: wire shape stays local, only the
vocabulary math lives here; unset stays unset (never invent an effort); when a provider
rejects a level fix its declared set, never a predicate.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

#: Matches ``k3`` as a delimited token (``k3``, ``k3-256k``, ``kimi-k3-cot``), never K2-era names (``kimi-k2.6``).
# From #76427 by @ruizanthony.
_KIMI_K3_SLUG_RE = re.compile(r"(?:^|[^a-z0-9])k3(?:[^a-z0-9]|$)")

# Canonical low→high ordering for nearest-level clamping. Includes "none" so an explicit
# disable can be clamped when a provider publishes it as a level. ``ultra`` is Hermes-internal
# (the Codex product tier): no wire accepts it, every declared set stops at ``max``.
EFFORT_LADDER: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

#: Widest OpenAI-compatible wire vocabulary (OpenRouter, Nous Portal).
OPENAI_COMPAT_WIRE_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: OpenAI/Codex Responses per model generation (live-verified): ``minimal`` is rejected by
#: both (clamps to low); ``max`` is gpt-5.6-only.
CODEX_GPT56_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")
CODEX_LEGACY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
# GPT-6 Astra is account-gated and its Responses API accepts no disable/minimal
# wire level; callers normalize those requests to ``low`` at the transport boundary.
CODEX_ASTRA_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")
ASTRA_MODEL_IDS: frozenset[str] = frozenset({"gpt-6-astra", "gpt-6-astra-900k"})
DAYBREAK_MODEL_IDS: frozenset[str] = frozenset(
    {"gpt-daybreak-blue-latest", "gpt-daybreak-blue-latest-900k"}
)

#: xAI Responses — Grok 4.6+ accepts xhigh; older Grok tops out at high.
XAI_GROK46_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
XAI_LEGACY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Actual Computer relays (SGLang/vLLM).
ACTUAL_RELAY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")

#: Moonshot/Kimi K3 (server default high) vs K2-era models. K3 quirks: ``high`` is K3's
#: positional middle AND server default, so ``medium`` rounds to it rather than down to
#: ``low``; ``xhigh`` rounds up to ``max`` (K3's top tier).
KIMI_K3_EFFORTS: tuple[str, ...] = ("low", "high", "max")
KIMI_K2_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
KIMI_K3_OVERRIDES: dict[str, str] = {"medium": "high", "xhigh": "max"}

#: OpenCode "Ox Alpha" (x-preview-f-free): thinking cannot be disabled and the wire accepts
#: exactly low/high/max (medium/none/xhigh 400); xhigh rounds up.
OX_ALPHA_EFFORTS: tuple[str, ...] = ("low", "high", "max")
OX_ALPHA_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Tencent TokenHub / Nebius Token Factory / Upstage Solar: plain three-level knobs.
TOKENHUB_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
NEBIUS_EFFORTS: tuple[str, ...] = ("low", "medium", "high")
SOLAR_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: GLM-5.2 native knob: exactly ``high`` (its minimum thinking level) and ``max``; GLM-5.3
#: widens it to a graded scale (live-verified, monotonic). ``xhigh`` requests the top tier.
GLM52_EFFORTS: tuple[str, ...] = ("high", "max")
GLM52_OVERRIDES: dict[str, str] = {"xhigh": "max"}
# : GLM-5.3 widens the knob to a graded low/medium/high/max scale — verified : live on
# api.z.ai/api/coding/paas/v4 (issue #91789, 2026-08-21): every : level accepted with monotonic
# reasoning-token scaling (low=4, medium=11, : high=98, max=125 on the probe prompt).
GLM53_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
GLM53_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: DeepSeek V4 OpenAI-compat endpoint; ``xhigh`` requests the top tier.
DEEPSEEK_V4_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
DEEPSEEK_V4_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Ollama Cloud /v1/chat/completions: rejects ``minimal`` with HTTP 400.
OLLAMA_CLOUD_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")
OLLAMA_CLOUD_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Meta Model API (Muse): rejects ``none``.
META_AI_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")


def is_astra_model(model: Optional[str]) -> bool:
    """``gpt-6-astra`` or its Hermes-side ``-900k`` picker alias, with or without a ``vendor/`` prefix.
    The single home for the slug set: picker gating, effort vocabulary and the request sanitizer all
    key off it, so a new Astra alias is one edit."""
    return (model or "").strip().lower().rsplit("/", 1)[-1] in ASTRA_MODEL_IDS


def codex_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for an OpenAI/Codex Responses model."""
    if is_astra_model(model):
        return CODEX_ASTRA_EFFORTS
    bare = (model or "").strip().lower().rsplit("/", 1)[-1]
    return (
        CODEX_GPT56_EFFORTS
        if "gpt-5.6" in bare or bare in DAYBREAK_MODEL_IDS
        else CODEX_LEGACY_EFFORTS
    )


def kimi_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for a Moonshot/Kimi slug (bare ``k3``, ``k3-256k``, ``kimi-k3*`` → K3).

    K3 is served as the bare slug ``k3``, plan variants like ``k3-256k``, and the ``kimi-k3*`` aliases; its
    documented set is low/high/max. Everything earlier speaks low/medium/high. Boundary-matched so K2-era
    names (``kimi-k2.6``) never match (detection regex from #76427 by @ruizanthony).
    """
    m = (model or "").strip().lower().split("/")[-1]
    return KIMI_K3_EFFORTS if _KIMI_K3_SLUG_RE.search(m) else KIMI_K2_EFFORTS


def clamp_effort(
    effort: Optional[str], supported: Optional[Sequence[str]], overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested reasoning effort onto a wire's supported levels.

    ``overrides`` (a declared vendor mapping, e.g. Kimi K3 ``medium → high``) is consulted
    first. Otherwise the request passes through unchanged when it is supported, when the
    supported set is unknown/empty, or when it isn't a recognized ladder level (custom
    providers may use bespoke names). Else the **nearest weaker** supported level is returned
    so a clamp never escalates cost; when nothing weaker exists, the weakest supported level
    (the provider's floor is the closest honest match). Monotonic: a stronger request never
    resolves weaker than a weaker request would.
    """
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [lvl for lvl in (str(s).strip().lower() for s in supported) if lvl in EFFORT_LADDER]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides and overrides.get(requested) in supported_norm:
        return overrides[requested]
    if requested not in EFFORT_LADDER:
        return effort
    # "none" disables reasoning — never a degradation target for an enabled ask
    # (clamping "minimal" to "none" would silently switch thinking off).
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [level for level in candidates if EFFORT_LADDER.index(level) < requested_idx]
    return max(below, key=EFFORT_LADDER.index) if below else min(candidates, key=EFFORT_LADDER.index)


def route_supported_efforts(provider: Optional[str], model: Optional[str]) -> tuple[str, ...]:
    """Levels the (provider, model) route's ENTRY clamp accepts: the Codex/OpenAI Responses set per
    model generation, else the widest OpenAI-compatible vocabulary (narrower providers clamp again
    downstream, never upward)."""
    if (provider or "").strip().lower() == "openai-codex":
        return codex_supported_efforts(model)
    return OPENAI_COMPAT_WIRE_EFFORTS


def _config_declared_efforts(provider: Optional[str], model: Optional[str], config: Optional[dict] = None) -> Optional[tuple[str, ...]]:
    """``providers.<provider>.models.<model>.reasoning_efforts`` from config, or None.

    *config* is the already-loaded mapping when the caller has one (the gateway reads its
    config once per command); otherwise the shared cached read-only config is used.
    """
    name, model_id = str(provider or "").strip(), str(model or "").strip()
    if not name or not model_id:
        return None
    try:
        if config is None:
            from hermes_cli.config import load_config_readonly

            config = load_config_readonly()
        providers = (config or {}).get("providers")
        entry = providers.get(name) if isinstance(providers, dict) else None
        models = entry.get("models") if isinstance(entry, dict) else None
        row = models.get(model_id) if isinstance(models, dict) else None
        declared = row.get("reasoning_efforts") if isinstance(row, dict) else None
    except Exception:  # config is user input: never break a command over it
        return None
    return tuple(declared) if isinstance(declared, (list, tuple)) else None


def declared_route_efforts(
    provider: Optional[str], model: Optional[str], config: Optional[dict] = None
) -> Optional[tuple[str, ...]]:
    """Vocabulary the (provider, model) route DECLARES, or ``None`` when nothing declares one.

    Two sources, in order of authority:

    1. ``providers.<provider>.models.<model>.reasoning_efforts`` in config — the explicit
       declaration for an endpoint whose catalog Hermes cannot discover (a relay, a gateway,
       a local server). It wins because the user configured that route on purpose.
    2. :meth:`ProviderProfile.supported_reasoning_efforts` — the profile hook Router feeds
       from its live catalog and Kilo declares statically.

    Levels outside :data:`EFFORT_LADDER` are dropped (``clamp_effort`` ignores them anyway).
    ``None`` means undeclared and keeps every caller fail-open: a picker goes on offering the
    shared ladder instead of hiding levels that may well work. An empty tuple is a declaration
    in its own right — the route takes no reasoning level at all.
    """
    declared = _config_declared_efforts(provider, model, config)
    if declared is None:
        try:
            from providers import get_provider_profile

            profile = get_provider_profile(provider) if provider else None
            declared = profile.supported_reasoning_efforts(model) if profile is not None else None
        except Exception:  # a plugin profile must never break the caller
            declared = None
    if declared is None:
        return None
    levels = dict.fromkeys(str(level).strip().lower() for level in declared)
    return tuple(level for level in levels if level in EFFORT_LADDER)


def effort_display_label(effort: Optional[str], provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Picker / ``/reasoning`` status label for a ladder level: the level itself when the route sends
    it verbatim, else ``"<level> (sends <clamped> on this route)"`` so a Hermes-internal step such as
    ``ultra`` (#61634) is never presented as a distinct wire level the route does not have."""
    requested = str(effort or "").strip().lower()
    clamped = clamp_effort(requested, route_supported_efforts(provider, model))
    return requested if not requested or clamped == requested else f"{requested} (sends {clamped} on this route)"


def requested_effort(reasoning_config: Optional[dict]) -> Optional[str]:
    """The user's explicit effort, or None (absent/malformed config, no effort, or reasoning
    disabled) — callers then omit the wire field."""
    if not isinstance(reasoning_config, dict) or reasoning_config.get("enabled") is False:
        return None
    return str(reasoning_config.get("effort") or "").strip().lower() or None


def clamp_reasoning_config(reasoning_config: Optional[dict], supported: Sequence[str] = OPENAI_COMPAT_WIRE_EFFORTS) -> Optional[dict]:
    """Return ``reasoning_config`` with its ``effort`` clamped onto ``supported`` (non-dicts and
    configs without an effort pass through untouched).

    The entry clamp for an OpenAI-compatible chat-completions request builder: Hermes-internal
    ``ultra`` never reaches a wire (#89503 main transport, #112010 aux/MoA), while provider
    profiles with narrower vocabularies clamp again downstream. Unset stays unset.
    """
    if not isinstance(reasoning_config, dict):
        return reasoning_config
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    clamped = clamp_effort(effort, supported) if effort else effort
    return {**reasoning_config, "effort": clamped} if clamped != effort else reasoning_config


def thinking_toggle_extras(
    reasoning_config: Optional[dict],
    efforts: Sequence[str],
    overrides: Optional[dict[str, str]] = None,
    *,
    always_emit_toggle: bool = False,
) -> tuple[dict, dict]:
    """Translate a reasoning config onto the Moonshot/DeepSeek chat_completions wire:
    ``extra_body.thinking`` toggle and top-level ``reasoning_effort``.

    Moonshot 400s when both are sent, so by default the effort (when it lands in
    ``efforts``) replaces the toggle. DeepSeek instead requires the toggle on every
    request (an omitted toggle defaults thinking on and then demands
    ``reasoning_content`` echoes), hence ``always_emit_toggle``. A requested effort of
    ``none`` is not a level on these wires; it falls back to the plain toggle.
    """
    if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
        return {"thinking": {"type": "disabled"}}, {}
    effort = requested_effort(reasoning_config)
    clamped = clamp_effort(None if effort == "none" else effort, efforts, overrides)
    if clamped in efforts:
        return ({"thinking": {"type": "enabled"}} if always_emit_toggle else {}), {"reasoning_effort": clamped}
    return {"thinking": {"type": "enabled"}}, {}


def ox_alpha_reasoning_extras(reasoning_config: Optional[dict], model: Optional[str]) -> tuple[dict, dict]:
    """Ox Alpha (``x-preview-f-free``) ``reasoning_effort`` translation for the
    opencode-zen profile (low/high/max only; anything else 400s)."""
    if (model or "").strip().rsplit("/", 1)[-1].lower() != "x-preview-f-free":
        return {}, {}
    effort = requested_effort(reasoning_config)
    clamped = clamp_effort(None if effort == "none" else effort, OX_ALPHA_EFFORTS, OX_ALPHA_OVERRIDES)
    return ({}, {"reasoning_effort": clamped}) if clamped in OX_ALPHA_EFFORTS else ({}, {})


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

CODEX_RESPONSES_EFFORTS: tuple[str, ...] = CODEX_GPT56_EFFORTS
# ---- END PLUGIN-COMPAT ----
