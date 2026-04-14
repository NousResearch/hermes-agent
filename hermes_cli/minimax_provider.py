"""MiniMax-as-chat-provider defaults for TTS / image / vision.

When a user selects MiniMax as their primary chat provider, Hermes can
auto-wire the MiniMax backend for TTS, image generation, and vision
without prompting for a second API key — all three capabilities ride on
the same ``MINIMAX_API_KEY`` the user already configured.

This module mirrors the existing ``nous_subscription.apply_nous_provider_defaults``
pattern: it's called once from the provider-selection flow (both
``hermes setup model`` and ``hermes model``), reads ``model.provider``,
and returns the set of config sections it touched so the caller can
print a friendly summary.

Only sets defaults — never overrides an existing explicit value.  A user
who has deliberately configured ``tts.provider: edge`` (or any non-default
value) keeps their choice; the function only acts when the target section
is empty or at its built-in default.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Set

logger = logging.getLogger(__name__)


#: Provider IDs that trigger MiniMax-native tool defaults.  Both the
#: international (``minimax``) and CN (``minimax-cn``) chat providers
#: share the same multimodal API surface, so both paths enable the
#: same backends.
MINIMAX_PROVIDERS: Set[str] = {"minimax", "minimax-cn"}


def _get_active_provider(config: Dict[str, Any]) -> str:
    """Read ``model.provider`` from a Hermes config dict.

    Returns the lowercased provider id, or an empty string if not set.
    Handles both the new ``model: {provider: ...}`` dict shape and the
    legacy ``model: "provider/name"`` string shape.
    """
    model = config.get("model")
    if isinstance(model, dict):
        provider = model.get("provider")
        return str(provider).strip().lower() if provider else ""
    if isinstance(model, str) and "/" in model:
        return model.split("/", 1)[0].strip().lower()
    return ""


def is_minimax_provider(config: Dict[str, Any]) -> bool:
    """True when the active chat provider is a MiniMax endpoint."""
    return _get_active_provider(config) in MINIMAX_PROVIDERS


def apply_minimax_provider_defaults(config: Dict[str, Any]) -> Set[str]:
    """Wire MiniMax-native defaults for TTS, image, and vision.

    Called from ``hermes setup model`` and ``hermes model`` after the user
    selects a MiniMax chat provider.  For each supported capability, if
    the user hasn't already configured a non-default backend, this
    function sets the config to use MiniMax.  The same
    ``MINIMAX_API_KEY`` the chat path uses is reused — no re-prompt.

    Returns a set describing which config sections were actually changed,
    e.g. ``{"tts", "image_gen", "vision"}``.  Callers use this to print a
    per-section success line.

    Called again on subsequent setup runs, this function is idempotent:
    sections already set to ``minimax`` are left alone and not reported
    as changed.
    """
    if not is_minimax_provider(config):
        return set()

    changed: Set[str] = set()

    # ── TTS ─────────────────────────────────────────────────────────────
    # MiniMax is already a first-class TTS backend in tools/tts_tool.py.
    # Promote it from "user has to discover it" to "auto-selected for
    # MiniMax users".  Only touch when the current value is empty or the
    # built-in Edge default.
    tts_cfg = config.get("tts")
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}
        config["tts"] = tts_cfg
    current_tts = str(tts_cfg.get("provider") or "").strip().lower()
    if current_tts in {"", "edge"}:
        tts_cfg["provider"] = "minimax"
        changed.add("tts")

    # ── Image generation ────────────────────────────────────────────────
    # The image_generate tool grows a ``provider`` key in the companion
    # PR that adds MiniMax as a backend.  Before that PR lands the
    # section simply won't be read; setting it now is harmless forward
    # compatibility.
    image_cfg = config.get("image_gen")
    if not isinstance(image_cfg, dict):
        image_cfg = {}
        config["image_gen"] = image_cfg
    current_image = str(image_cfg.get("provider") or "").strip().lower()
    # "auto" means "pick the best available"; safe to override.  "fal" is
    # the legacy hardcoded default; also safe to override for MiniMax users
    # since MiniMax unlocks image-01 at no additional cost.  Any other
    # explicit value is the user's deliberate choice — don't touch.
    if current_image in {"", "auto", "fal"}:
        image_cfg["provider"] = "minimax"
        changed.add("image_gen")

    # ── Vision ──────────────────────────────────────────────────────────
    # No config write needed.  MiniMax's chat models (M2.7 et al.) do
    # not accept multimodal content — image understanding is served by a
    # dedicated VLM endpoint (/v1/coding_plan/vlm, the same path the
    # official mmx-cli ``mmx vision describe`` command targets).
    # ``tools/vision_tools.py`` dispatches to that endpoint automatically
    # when the active chat provider is MiniMax (see ``_use_minimax_vision``
    # there), so there is nothing to persist in config.yaml.
    #
    # Because vision doesn't write config, it can't independently track
    # "first-time-seen" state.  To keep ``apply_minimax_provider_defaults``
    # idempotent from a UX perspective, we only report vision alongside
    # other changes — once TTS and image_gen are both already wired,
    # subsequent calls stay silent.  This also respects an explicit
    # non-MiniMax vision provider (``openrouter`` etc.) — we don't claim
    # to wire vision in that case because ``_use_minimax_vision`` will
    # correctly step aside.
    aux_cfg = config.get("auxiliary") if isinstance(config.get("auxiliary"), dict) else {}
    vision_cfg = aux_cfg.get("vision") if isinstance(aux_cfg.get("vision"), dict) else {}
    vision_explicit = str(vision_cfg.get("provider") or "").strip().lower()
    # Treat "main" the same as "auto" / "": it routes through the chat
    # provider, which for MiniMax means our VLM dispatcher will step in
    # anyway.  Keep this set in sync with the allowlist in
    # tools/vision_tools.py::_use_minimax_vision().
    if vision_explicit in {"", "auto", "main"} and changed:
        changed.add("vision")

    if changed:
        logger.info(
            "Applied MiniMax provider defaults (changed sections: %s)",
            sorted(changed),
        )
    return changed


def describe_changes(changed: Iterable[str]) -> str:
    """Produce a user-facing summary of what ``apply_minimax_provider_defaults``
    modified.  Used by the setup wizard to print a friendly bullet list.
    """
    labels = {
        "tts": "TTS provider → MiniMax speech-2.6-hd (30+ voices)",
        "image_gen": "Image generation → MiniMax image-01",
        "vision": "Vision analysis → MiniMax VLM (MiniMax-VL-01)",
    }
    changed = list(changed)
    if not changed:
        return "No changes — your existing TTS / image / vision choices were preserved."
    lines = [labels.get(key, key) for key in sorted(changed)]
    return "\n".join(f"  • {line}" for line in lines)
