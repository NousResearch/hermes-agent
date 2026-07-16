"""Mistral AI provider profile.

Mistral's API is standard OpenAI-compatible.  Thinking-enabled models
accept ``reasoning_effort`` as a top-level kwarg.  ``build_api_kwargs_extras``
uses a version-based heuristic to determine which models support reasoning —
versioned models with a date code >= 2603 for mistral-small or >= 2604 for
mistral-medium, and the mistral-medium-3/3.5 family.  Non-reasoning models
(codestral, mistral-large, pixtral, ministral, and older date-stamped variants)
skip the parameter entirely.

Agentic models with tool-calling support:

- ``mistral-large-latest`` — flagship, strong reasoning + tool use (128K ctx)
- ``mistral-medium-latest`` — balanced, vision + reasoning-capable (256K ctx)
- ``mistral-small-latest`` — fast, cost-effective, vision + reasoning (128K ctx)
- ``codestral-latest`` — code generation (256K ctx)
- ``pixtral-12b-latest`` — multimodal vision specialist (128K ctx)
"""

from __future__ import annotations

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


# Regex: extract a 4-digit version/date code (e.g. 2603, 2505) from model names.
# Mistral uses this scheme: ``mistral-small-2603``, ``mistral-medium-2505``, etc.
_VERSION_PATTERN = re.compile(r"-(\d{4})(?:[-.]|$)")


def _model_supports_reasoning(model: str | None) -> bool:
    """Check if ``model`` supports ``reasoning_effort``.

    Uses a combination of exact-name matching, family-based exclusion, and
    version-threshold heuristics to stay future-proof as Mistral releases new
    models.

    Resolution order:
      1. Empty / None → False
      2. Known non-reasoning families → False (codestral, devstral, ministral,
         mistral-large, mistral-code, mistral-ocr, mistral-moderation,
         mistral-tiny, pixtral, voxtral, open-mistral)
      3. Always-reasoning exact names / aliases → True (includes -latest aliases,
         bare names like ``mistral-medium``, known experimental models, and the
         mistral-medium-3/3.5 family)
      4. Versioned models → True if the 4-digit date code >= the per-family
         reasoning threshold (small 2603, medium 2604, magistral 2509)
      5. Unknown naming patterns → False (safe default)
    """
    m = (model or "").strip().lower()
    if not m:
        return False

    # ── Step 2: Non-reasoning families ──────────────────────────────────
    # ``mistral-large`` is deliberately NOT in this list — it doesn't currently
    # have a reasoning-capable version (latest is 2512), but the version
    # threshold (step 4) already covers it at >= 2600, which matches nothing
    # today and will automatically match a future ``mistral-large-26xx`` model
    # with reasoning.  See _THRESHOLDS below.
    _NON_REASONING_FAMILIES = (
        "codestral", "devstral",
        "ministral",
        "mistral-code", "mistral-ocr", "mistral-moderation",
        "mistral-tiny",
        "pixtral",
        "voxtral",
        "open-mistral",
    )
    if m.startswith(_NON_REASONING_FAMILIES):
        return False

    # ── Step 3: Always-reasoning exact names / aliases ──────────────────
    _ALWAYS_REASONING = frozenset({
        # Experimental / labs
        "labs-leanstral-1-5", "labs-leanstral-1-5-1",
        # Generic latest aliases
        "mistral-small-latest",
        "mistral-medium", "mistral-medium-latest",
        # 3.x family — all 3.x variants are reasoning-capable
        # (caught by prefix match in the `startswith` below; listed explicitly
        # for documentation clarity)
        "mistral-medium-3-5", "mistral-medium-3.5",
        # Vibe CLI aliases — real API model IDs that accept reasoning_effort
        "mistral-vibe-cli-fast", "mistral-vibe-cli-latest",
        "mistral-vibe-cli-with-tools",
    })
    if m in _ALWAYS_REASONING:
        return True

    # Also catch mistral-medium-3.x variants with suffixes (e.g. -pro)
    if m.startswith("mistral-medium-3") or m.startswith("mistral-medium-3."):
        return True

    # Magistral models reason natively (always on) — they do NOT accept
    # a reasoning_effort parameter.  Only date-stamped versions >= 2509
    # are treated as reasoning-capable, but this is for the *native*
    # reasoning path, not adjustable reasoning_effort.  Exclude -latest
    # bare aliases so we don't accidentally send reasoning_effort to them.
    _MAGISTRAL_LATEST = frozenset({
        "magistral-small-latest", "magistral-medium-latest",
    })
    if m in _MAGISTRAL_LATEST:
        return False

    # ── Step 4: Version-based heuristic ─────────────────────────────────
    match = _VERSION_PATTERN.search(m)
    if not match:
        # Unknown naming pattern — safe default.
        return False

    version = int(match.group(1))

    # Extract the "family" prefix (everything before the version code).
    family = m[:match.start()].rstrip("-")

    # Per-family reasoning-version thresholds:
    #   mistral-small >= 2603   (2603 introduced reasoning for small)
    #   mistral-medium >= 2604  (2604 introduced reasoning for medium)
    #   mistral-large >= 2600   (first Large model with reasoning — speculative)
    #   magistral-*   >= 2509  (2509 introduced reasoning for magistral)
    #
    # Any date-stamped model >= its family threshold will automatically receive
    # reasoning_effort, even for future versions not yet known.
    #
    # TODO: Update thresholds when new thinking-capable families are released
    #       or when existing families ship their first reasoning version.
    #       Track: https://docs.mistral.ai/getting-started/models/
    _THRESHOLDS = {
        "mistral-small": 2603,
        "mistral-medium": 2604,
        "mistral-large": 2600,  # first large model with reasoning
        "magistral-small": 2509,
        "magistral-medium": 2509,
    }
    threshold = _THRESHOLDS.get(family)
    if threshold is not None:
        return version >= threshold

    return False


class MistralProfile(ProviderProfile):
    """Mistral AI — standard OpenAI-compatible.

    ``reasoning_effort`` is passed through only for models identified as
    thinking-enabled by ``_model_supports_reasoning``.  Non-thinking models
    skip the parameter entirely, avoiding HTTP 400 from Mistral's API.
    """

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_reasoning(model):
            return extra_body, top_level

        if isinstance(reasoning_config, dict):
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort:
                top_level["reasoning_effort"] = effort

        return extra_body, top_level


mistral = MistralProfile(
    name="mistral",
    aliases=("mistral-ai", "mistralai"),
    env_vars=("MISTRAL_API_KEY",),
    display_name="Mistral AI",
    description="Mistral AI — native Mistral API",
    signup_url="https://console.mistral.ai/",
    fallback_models=(
        "mistral-large-latest",
        "mistral-medium-latest",
        "mistral-small-latest",
        "codestral-latest",
        "pixtral-12b-latest",
    ),
    base_url="https://api.mistral.ai/v1",
    supports_vision=True,
    default_aux_model="mistral-small-latest",
)

register_provider(mistral)
