"""Mixture-of-Agents configuration and slash-command helpers."""

from __future__ import annotations

import base64
import json
import math
from copy import deepcopy
from typing import Any

MOA_MARKER_PREFIX = "__HERMES_MOA_TURN_V1__"
DEFAULT_MOA_PRESET_NAME = "default"

DEFAULT_MOA_REFERENCE_MODELS: list[dict[str, str]] = [
    {"provider": "openai-codex", "model": "gpt-5.5"},
    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}]

DEFAULT_MOA_AGGREGATOR: dict[str, str] = {
    "provider": "openrouter", "model": "anthropic/claude-opus-4.8"}

DEFAULT_MOA_REFERENCE_TIMEOUT: float | None = None


def _default_reference_models() -> list[dict[str, Any]]:
    return [{**slot, "enabled": True} for slot in deepcopy(DEFAULT_MOA_REFERENCE_MODELS)]


def _coerce_number(value: Any, cast, default=None, *, positive: bool = False):
    """Coerce ``value`` with ``cast`` (float/int); ``default`` when unset/blank/invalid.

    ``int`` also accepts float-looking strings ("3.0"). With ``positive`` the result must be > 0
    (and finite for floats) or ``default`` is returned."""
    if value is None or value == "":
        return default
    try:
        number = cast(value)
    except (TypeError, ValueError):
        if cast is not int:
            return default
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return default
    if positive and (number <= 0 or (cast is float and not math.isfinite(number))):
        return default
    return number


def _coerce_reference_timeout(value: Any) -> float | None:
    """Finite positive advisor timeout, or None to inherit ``auxiliary.moa_reference.timeout``.

    No artificial cap: long-thinking advisor models legitimately run far beyond five minutes."""
    if isinstance(value, bool):
        return DEFAULT_MOA_REFERENCE_TIMEOUT
    return _coerce_number(value, float, DEFAULT_MOA_REFERENCE_TIMEOUT, positive=True)


def _coerce_fanout(value: Any) -> str:
    """Normalize the fan-out cadence to ``per_iteration`` | ``user_turn`` | ``every_n:<N>`` (N >= 2);
    the mapping form ``{mode: every_n, n: N}`` becomes the string, ``every_n:1`` collapses to
    ``per_iteration``, anything unparseable falls back to ``user_turn`` (the cheapest cadence).

    The ``every_n`` cadence also accepts the mapping form ``{mode: every_n, n: N}`` from hand-edited YAML
    and normalizes it to the canonical string, so the rest of the pipeline (presets, flattened view,
    runtime) only ever sees one shape. See #67199.
    """
    def _every_n(n: int) -> str:
        return f"every_n:{n}" if n >= 2 else ("per_iteration" if n == 1 else "user_turn")

    if isinstance(value, dict):
        # Non-every_n mapping modes fall through to the string path (e.g. {mode: user_turn}).
        mode = str(value.get("mode") or "").strip().lower()
        if mode == "every_n":
            return _every_n(_coerce_number(value.get("n"), int, 0))
        value = mode
    mode = str(value or "").strip().lower()
    if mode in {"per_iteration", "user_turn"}:
        return mode
    if mode.startswith("every_n"):
        _, sep, rest = mode.partition(":")
        return _every_n(_coerce_number(rest.strip(), int, 0) if sep else 0)
    return "user_turn"


def coerce_privacy_filter(value: Any) -> str:
    """Normalize ``moa.privacy_filter`` to '' (off, the default), 'display', or 'full'.

    ``false``/``None``/unknown values land on '' so a hand-edited config degrades to prior
    behavior. 'display' redacts user-visible surfaces only (reference blocks in the UI and saved
    MoA trace records).

    - ``''`` (empty string): filter off — the default. The aggregator still sees raw advisor text, so answer
    quality is unaffected. - ``'full'``: additionally redact the advisor text injected into the aggregator
    prompt (issue #59959's literal ask). A hand-edited boolean ``true`` maps here because the issue framed
    the toggle as "redact before passing to the aggregator".
    """
    if value is True:
        return "full"
    if value is None or value is False:
        return ""
    mode = str(value).strip().lower()
    return mode if mode in {"display", "full"} else ("full" if mode in {"true", "on", "yes", "1"} else "")


def _clean_reasoning_effort(value: Any) -> str | None:
    """Canonical per-slot reasoning effort, or None when unset/invalid."""
    from hermes_constants import parse_reasoning_effort
    parsed = None if value is None or value is True else parse_reasoning_effort(value)
    if parsed is None:
        return None
    return "none" if parsed.get("enabled") is False else parsed.get("effort")


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        return True if text in {"1", "true", "yes", "on"} else False if text in {"0", "false", "no", "off"} else default
    return bool(value)


def _slot_problem(slot: Any) -> str | None:
    """Human-readable problem for a slot ``_clean_slot`` would drop; None when complete and valid.

    Mirrors ``_clean_slot`` exactly so the write-boundary validator (``validate_moa_payload``) and
    the tolerant runtime normalizer can never disagree about what is acceptable."""
    if not isinstance(slot, dict):
        return "must be an object with 'provider' and 'model'"
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    if not provider and not model:
        return "provider and model are required"
    if not provider:
        return "provider is required"
    if not model:
        return f"model is required (provider '{provider}' has no model selected)"
    # MoA is a virtual provider whose presets are themselves MoA runs; allowing one as a slot
    # would create a recursive MoA tree that the runtime guards only catch mid-turn.
    if provider.lower() == "moa":
        return "the Mixture of Agents provider cannot be used inside a preset (recursive MoA)"
    return None


def _clean_slot(slot: Any, *, include_enabled: bool = False) -> dict[str, Any] | None:
    # Any slot ``_slot_problem`` rejects is dropped, falling back to the preset's defaults.
    if _slot_problem(slot) is not None:
        return None
    clean: dict[str, Any] = {"provider": str(slot["provider"]).strip(), "model": str(slot["model"]).strip()}
    effort = _clean_reasoning_effort(slot.get("reasoning_effort"))
    if effort:
        clean["reasoning_effort"] = effort
    # Optional per-slot max_tokens overrides the preset-level reference_max_tokens for this
    # advisor; None (default) = no cap.
    slot_mt = _coerce_number(slot.get("max_tokens"), int, positive=True)
    if slot_mt is not None:
        clean["max_tokens"] = slot_mt
    if include_enabled:
        clean["enabled"] = _coerce_bool(slot.get("enabled"), True)
    return clean


def _reference_slots(raw_refs: Any) -> list:
    """``reference_models`` as a list: a JSON string (hand-edited config.yaml) is decoded, a single
    mapping is wrapped, and any other scalar / bad type degrades to ``[]`` instead of crashing."""
    if isinstance(raw_refs, str):
        try:
            raw_refs = json.loads(raw_refs)
        except (json.JSONDecodeError, ValueError):
            raw_refs = []
    if not isinstance(raw_refs, list):
        raw_refs = [raw_refs] if isinstance(raw_refs, dict) else []
    return raw_refs


def validate_moa_payload(raw: Any) -> list[str]:
    """Return the problems ``normalize_moa_config`` would silently paper over (empty = safe to save).

    Read-time tolerance (a hand-edited config degrades to defaults instead of crashing) is a
    corruption engine at write time: a half-filled slot would silently replace the whole preset.

    ``normalize_moa_config`` is deliberately tolerant: at *read* time a hand-edited config must degrade to
    defaults rather than crash the agent. API write paths call this first and reject invalid payloads loudly
    instead of saving something the user never chose. See #64156.
    """
    if not isinstance(raw, dict):
        return ["MoA config must be an object"]

    presets_raw = raw.get("presets")
    # Legacy flat payload: the top-level object is the default preset.
    presets: dict[Any, Any] = presets_raw if isinstance(presets_raw, dict) and presets_raw else {DEFAULT_MOA_PRESET_NAME: raw}

    problems: list[str] = []
    for name, preset in presets.items():
        label = str(name or "").strip() or "(unnamed)"
        if not isinstance(preset, dict):
            problems.append(f"preset '{label}': must be an object")
            continue

        refs = preset.get("reference_models")
        if not isinstance(refs, list):
            refs = [refs] if isinstance(refs, dict) else []
        issues = [(index, _slot_problem(slot)) for index, slot in enumerate(refs)]
        problems.extend(f"preset '{label}' reference {index + 1}: {issue}" for index, issue in issues if issue)
        if all(issue for _, issue in issues):
            problems.append(f"preset '{label}': needs at least one complete reference model")

        agg_issue = _slot_problem(preset.get("aggregator"))
        if agg_issue:
            problems.append(f"preset '{label}' aggregator: {agg_issue}")
    return problems


def _normalize_preset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    cleaned = (_clean_slot(item, include_enabled=True) for item in _reference_slots(raw.get("reference_models")))
    refs = [item for item in cleaned if item is not None]
    policy = str(raw.get("degraded_reference_policy") or "loud").strip().lower()
    return {
        "enabled": _coerce_bool(raw.get("enabled"), True),
        "reference_models": refs or _default_reference_models(),
        "aggregator": _clean_slot(raw.get("aggregator")) or deepcopy(DEFAULT_MOA_AGGREGATOR),
        # None means 'don't send it — provider default applies'.
        "reference_temperature": _coerce_number(raw.get("reference_temperature"), float),
        "aggregator_temperature": _coerce_number(raw.get("aggregator_temperature"), float),
        "reference_timeout": _coerce_reference_timeout(raw.get("reference_timeout")),
        # Failed-advisor disclosure policy; unknown values fail loud.
        "degraded_reference_policy": policy if policy in {"loud", "silent"} else "loud",
        "max_tokens": _coerce_number(raw.get("max_tokens"), int, 4096),
        # Per-turn cap on each reference ADVISOR (never the acting aggregator). None = uncapped;
        # advisor generation dominates MoA latency, so e.g. 600 roughly halves wall time.
        "reference_max_tokens": _coerce_number(raw.get("reference_max_tokens"), int, positive=True),
        # "user_turn" (default, cheapest): advisors run ONCE per user turn; "per_iteration": every
        # tool iteration; "every_n:<N>": first iteration of each turn and every Nth after.
        "fanout": _coerce_fanout(raw.get("fanout"))}


_FLAT_PRESET_KEYS = (
    "reference_models", "aggregator", "reference_temperature", "aggregator_temperature",
    "reference_timeout", "degraded_reference_policy", "max_tokens", "reference_max_tokens",
    "fanout", "enabled")


# When the reference fan-out runs. "user_turn" (default) runs the advisors ONCE per user turn (the original
# MoA shape, and the cheapest cadence — #67199): the aggregator gets their upfront plan-level advice, then
# acts alone for the rest of the tool loop. "per_iteration" re-runs the advisors whenever the advisory view
# changes — i.e. every tool iteration, so advice tracks live task state at the cost of multiplying advisor
# spend by tool-loop depth. "every_n:<N>" (N >= 2) is the middle ground: advisors run on the first iteration
# of each user turn and every Nth tool iteration after it; in-between iterations reuse the cached guidance
# from the last advisor run. Also accepts the mapping form {mode: every_n, n: N}, normalized to the
# canonical string.
def normalize_moa_config(raw: Any) -> dict[str, Any]:
    """Return validated MoA config with named presets."""
    if not isinstance(raw, dict):
        raw = {}

    presets_raw = raw.get("presets")
    presets: dict[str, dict[str, Any]] = {}
    if isinstance(presets_raw, dict):
        for name, preset in presets_raw.items():
            clean_name = str(name or "").strip()
            if clean_name:
                presets[clean_name] = _normalize_preset(preset)
    if not presets:  # Legacy flat config becomes the default preset.
        presets[DEFAULT_MOA_PRESET_NAME] = _normalize_preset(raw)

    default_name = str(raw.get("default_preset") or "").strip()
    if not default_name or default_name not in presets:
        default_name = next(iter(presets))  # never empty: legacy flat config seeds the default

    active_name = str(raw.get("active_preset") or "").strip()
    if active_name not in presets:
        active_name = ""
    return {
        "default_preset": default_name,
        "active_preset": active_name,
        "presets": presets,
        # Compatibility/flattened view for existing dashboard/desktop callers.
        **{key: deepcopy(presets[default_name][key]) for key in _FLAT_PRESET_KEYS},
        # MoA-level (not per-preset) toggle; see coerce_privacy_filter for the modes.
        "privacy_filter": coerce_privacy_filter(raw.get("privacy_filter"))}


def resolve_moa_preset(config: Any, name: str | None = None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    preset_name = str(name or cfg.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    preset = cfg["presets"].get(preset_name)
    if preset is None:
        from agent.errors import MoAPresetNotFoundError
        available = ", ".join(cfg["presets"]) or "(none)"
        raise MoAPresetNotFoundError(
            f"MoA preset '{preset_name}' was not found. Available presets: "
            f"{available}. Run `hermes moa list`.")
    return deepcopy(preset)


def exact_moa_preset_name(config: Any, text: str) -> str | None:
    """Return the preset name iff ``text`` exactly matches an *enabled* preset.

    Used by the no-explicit-provider switch path for a bare ``/model <preset>``. Because the match
    is implicit it honors the per-preset ``enabled`` opt-out: a plain model switch that collides
    with a disabled preset's name must not silently pivot onto the MoA provider. Explicit
    ``--provider moa`` / picker selection bypasses this, so disabled presets stay reachable.

    See #55187.
    """
    wanted = str(text or "").strip()
    if not wanted:
        return None
    preset = normalize_moa_config(config)["presets"].get(wanted)
    return None if preset is None or not preset.get("enabled", True) else wanted


def decode_moa_turn(message: Any) -> tuple[str, dict[str, Any] | None]:
    """Decode a hidden /moa one-shot marker."""
    if not isinstance(message, str) or not message.startswith(MOA_MARKER_PREFIX):
        return message, None
    encoded = message[len(MOA_MARKER_PREFIX):].strip()
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except Exception:
        return message, None
    return str(payload.get("prompt") or ""), _normalize_preset(payload.get("config") or {})


def moa_usage() -> str:
    return "Usage: /moa <prompt>  (runs one prompt through the default MoA preset, then restores your model; pick a preset from the model picker to switch for the session)"


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def encode_moa_turn(prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Encode a /moa one-shot turn for frontends that can only send text."""
    payload = {
        "prompt": str(prompt or ""),
        "config": resolve_moa_preset(config or {}, preset),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    return f"{MOA_MARKER_PREFIX}{encoded}"

def build_moa_turn_prompt(user_prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Build the hidden one-shot payload used by TUI/gateway routing."""
    return encode_moa_turn(user_prompt, config, preset=preset)

def list_moa_presets(config: Any) -> list[str]:
    cfg = normalize_moa_config(config)
    return list(cfg["presets"].keys())

def set_active_moa_preset(config: Any, name: str | None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    clean = str(name or "").strip()
    if clean and clean not in cfg["presets"]:
        raise KeyError(clean)
    cfg["active_preset"] = clean
    return cfg
# ---- END PLUGIN-COMPAT ----
