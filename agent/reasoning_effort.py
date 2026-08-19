"""Canonical reasoning-effort vocabulary and wire clamping.

Hermes' internal effort ladder (``hermes_constants.VALID_REASONING_EFFORTS``
plus the ``none`` disable level) is wider than what any single provider wire
accepts. Historically every transport and provider profile hand-rolled its own
translation map, and the class of bugs that produced was constant: a new
internal level (``ultra``) leaking to a wire that rejects it with HTTP 400
(#89503, #70058), or an unknown level being dropped to a weak default so the
strongest ask resolved *weaker* than an explicit ``high`` — a ladder
inversion (#74295, #87279).

This module is the single source of truth both kinds of code use instead:

- :data:`EFFORT_LADDER` — canonical low→high ordering.
- :func:`clamp_effort` — the one clamping policy: keep a supported level
  verbatim, otherwise take the **nearest weaker** supported level (never
  silently escalate cost above what was asked), and only when nothing weaker
  exists take the weakest supported level (a provider whose minimum thinking
  level is ``high`` serves ``high`` for a ``low`` ask — GLM-5.2's shape).
- Named wire-vocabulary constants for the common OpenAI-compatible surfaces,
  so call sites declare *data* ("this route accepts these levels") rather
  than logic.

Rules for call sites:

1. **Wire shape stays local.** Whether a route wants ``extra_body.reasoning``,
   a top-level ``reasoning_effort`` string, or a ``thinking`` toggle is the
   caller's business. Only the *vocabulary math* lives here.
2. **Unset stays unset.** ``clamp_effort`` translates an explicit request; it
   does not invent one. When the user expressed no effort, prefer omitting
   the field so the server default applies.
3. **Never patch a predicate.** When a provider rejects a level, fix its
   declared supported set (data), never add another vendor-name special case
   at the call site.
"""

from __future__ import annotations

from typing import Optional, Sequence

# Canonical low→high ordering used for nearest-level clamping. Superset of
# hermes_constants.VALID_REASONING_EFFORTS ("none" included so an explicit
# disable can be clamped too when a provider publishes it as a level).
EFFORT_LADDER: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
)

# ``ultra`` is Hermes-internal ladder vocabulary (the Codex product tier); no
# provider wire accepts it verbatim anywhere. Every declared wire set below
# therefore stops at ``max`` — ``ultra`` always clamps down.

#: The widest OpenAI-compatible wire vocabulary (OpenRouter, Nous Portal):
#: exactly max|xhigh|high|medium|low|minimal|none.
OPENAI_COMPAT_WIRE_EFFORTS: tuple[str, ...] = (
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
)

#: OpenAI/Codex Responses backend (``minimal`` is rejected → clamps to low).
CODEX_RESPONSES_EFFORTS: tuple[str, ...] = (
    "low", "medium", "high", "xhigh", "max",
)

#: xAI Responses — Grok 4.6+ accepts xhigh; older Grok tops out at high.
XAI_GROK46_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
XAI_LEGACY_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Actual Computer relays (SGLang/vLLM): none/low/medium/high/max.
ACTUAL_RELAY_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")

#: Moonshot/Kimi K3: low/high/max (server default high).
KIMI_K3_EFFORTS: tuple[str, ...] = ("low", "high", "max")
#: Moonshot/Kimi K2-era models: low/medium/high.
KIMI_K2_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Tencent TokenHub: low/medium/high.
TOKENHUB_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

#: Kimi K3's vendor-documented translation quirks (platform.kimi.ai
#: thinking-model guide): ``high`` is K3's positional middle AND server
#: default, so ``medium`` rounds to it rather than down to ``low``; ``xhigh``
#: rounds up to ``max`` (K3's top tier), matching the kimi-coding plugin.
KIMI_K3_OVERRIDES: dict[str, str] = {"medium": "high", "xhigh": "max"}

#: GLM-5.2 native reasoning_effort knob: exactly two enabled levels,
#: ``high`` (its minimum thinking level) and ``max`` (per Z.AI/BigModel
#: docs). ``xhigh`` requests the top tier, not the floor.
GLM52_EFFORTS: tuple[str, ...] = ("high", "max")
GLM52_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: DeepSeek V4 OpenAI-compat endpoint: low/medium/high/max; ``xhigh``
#: requests the top tier (matches the shipped profile mapping).
DEEPSEEK_V4_EFFORTS: tuple[str, ...] = ("low", "medium", "high", "max")
DEEPSEEK_V4_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Ollama Cloud /v1/chat/completions: accepts {none, low, medium, high, max};
#: rejects ``minimal`` with HTTP 400. ``xhigh`` requests the top tier.
OLLAMA_CLOUD_EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "max")
OLLAMA_CLOUD_OVERRIDES: dict[str, str] = {"xhigh": "max"}

#: Meta Model API (Muse): minimal..xhigh; rejects ``none``.
META_AI_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh")

#: Upstage Solar Pro/Open: low/medium/high.
SOLAR_EFFORTS: tuple[str, ...] = ("low", "medium", "high")


def kimi_supported_efforts(model: Optional[str]) -> tuple[str, ...]:
    """Supported effort set for a Moonshot/Kimi model slug.

    K3 is served as the bare slug ``k3`` and the ``kimi-k3*`` aliases; its
    documented set is low/high/max. Everything earlier speaks low/medium/high.
    """
    m = (model or "").strip().lower().split("/")[-1]
    if m == "k3" or m.startswith("kimi-k3"):
        return KIMI_K3_EFFORTS
    return KIMI_K2_EFFORTS


def clamp_effort(
    effort: Optional[str],
    supported: Optional[Sequence[str]],
    overrides: Optional[dict[str, str]] = None,
) -> Optional[str]:
    """Clamp a requested reasoning effort onto a wire's supported levels.

    ``overrides`` is an optional declared mapping consulted first, for routes
    whose vendor documents a translation that differs from nearest-weaker
    (Kimi K3 documents ``medium → high``: high is its positional middle and
    server default). Overrides are data, not logic — a call site never adds
    vendor ``if``\\ s around this function.

    Otherwise: returns the requested effort unchanged when it is supported,
    when the supported set is unknown (``None``/empty), or when the effort
    isn't a recognized ladder level (custom providers may use bespoke names —
    pass through rather than guess). Otherwise returns the **nearest weaker**
    supported level, so a clamp never silently escalates cost; when nothing
    weaker exists, the weakest supported level is returned (the caller asked
    for *some* thinking and the provider's floor is the closest honest match).

    The policy is monotonic: a stronger request never resolves to a weaker
    wire level than a weaker request would.
    """
    requested = str(effort or "").strip().lower()
    if not requested or not supported:
        return effort
    supported_norm = [
        str(level).strip().lower()
        for level in supported
        if str(level).strip().lower() in EFFORT_LADDER
    ]
    if not supported_norm or requested in supported_norm:
        return effort
    if overrides:
        mapped = overrides.get(requested)
        if mapped in supported_norm:
            return mapped
    if requested not in EFFORT_LADDER:
        return effort
    # "none" disables reasoning — it is never a *degradation target* for an
    # enabled ask (clamping "minimal" to "none" would silently switch
    # thinking off). It still passes through verbatim when requested.
    candidates = [level for level in supported_norm if level != "none"]
    if not candidates:
        return effort
    requested_idx = EFFORT_LADDER.index(requested)
    below = [
        level for level in candidates
        if EFFORT_LADDER.index(level) < requested_idx
    ]
    if below:
        return max(below, key=EFFORT_LADDER.index)
    return min(candidates, key=EFFORT_LADDER.index)


def requested_effort(reasoning_config: Optional[dict]) -> Optional[str]:
    """Extract the user's explicit effort from a reasoning config, or None.

    Returns ``None`` when the config is absent, malformed, carries no effort,
    or reasoning is explicitly disabled — callers should then omit the wire
    field entirely so the server default applies (rule 2 above).
    """
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return None
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or None
