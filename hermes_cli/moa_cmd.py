"""CLI helpers for configuring Mixture of Agents."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import load_config, save_config
from hermes_cli.inventory import build_models_payload, load_picker_context
from hermes_cli.moa_config import DEFAULT_MOA_PRESET_NAME, normalize_moa_config


def _prompt_choice(title: str, rows: list[str], default: int = 0) -> int:
    try:
        from hermes_cli.curses_ui import curses_radiolist

        return curses_radiolist(title, rows, selected=default, cancel_returns=default)
    except Exception:
        for idx, row in enumerate(rows, start=1):
            print(f"{idx}. {row}")
        raw = input(f"{title} [{default + 1}]: ").strip()
        if not raw:
            return default
        try:
            return max(0, min(len(rows) - 1, int(raw) - 1))
        except ValueError:
            return default


def _prompt_optional_max_tokens(current: int | None) -> tuple[bool, int | None]:
    """Prompt for an optional per-slot ``max_tokens`` override.

    Returns ``(changed, value)``. ``changed`` is False when the user kept the
    existing value (empty input); ``value`` is None when the override was
    cleared, matching ``_clean_slot``'s "None = preset default" contract.
    """
    shown = str(current) if current is not None else "preset default"
    prompt = (
        f"Max output tokens for this slot [{shown}] "
        "(blank = keep, 'none' = use preset default): "
    )
    try:
        raw = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False, current
    if not raw:
        return False, current
    if raw.lower() in {"none", "default", "clear", "0"}:
        return True, None
    try:
        value = int(raw)
    except ValueError:
        print(f"  Not a number: {raw!r} — keeping {shown}.")
        return False, current
    if value <= 0:
        print("  max_tokens must be positive — using the preset default.")
        return True, None
    return True, value


def _slot_reasoning_capability(provider: dict[str, Any], model: str) -> dict[str, Any]:
    """Reasoning capability entry for ``model`` on ``provider``, or ``{}``.

    ``build_models_payload(capabilities=True)`` attaches a per-model
    ``{fast, reasoning, can_disable_reasoning?}`` map to every provider row;
    the slot picker reuses it instead of re-probing any catalog.
    """
    caps = provider.get("capabilities")
    if isinstance(caps, dict):
        entry = caps.get(model)
        if isinstance(entry, dict):
            return entry
    return {}


def _prompt_slot_reasoning_effort(
    provider: dict[str, Any],
    model: str,
    current: str | None,
) -> tuple[bool, str | None]:
    """Prompt for a per-slot reasoning effort using the primary-model flow.

    Returns ``(changed, value)``; ``value`` is ``"none"`` when the user disabled
    reasoning for this slot. Skipped entirely for models the picker already
    knows take no reasoning parameter.
    """
    entry = _slot_reasoning_capability(provider, model)
    if entry and not entry.get("reasoning", True):
        return False, current

    from hermes_constants import VALID_REASONING_EFFORTS

    # Local import: hermes_cli.main imports cmd_moa from this module, so a
    # module-level import would be circular.
    from hermes_cli.main import _prompt_reasoning_effort_selection

    selected = _prompt_reasoning_effort_selection(
        list(VALID_REASONING_EFFORTS), current_effort=(current or "")
    )
    if selected is None:
        return False, current
    if selected == "none" and entry.get("can_disable_reasoning") is False:
        print(f"  {model} cannot disable reasoning — keeping the current level.")
        return False, current
    return True, selected


def _model_options() -> list[dict[str, Any]]:
    payload = build_models_payload(
        load_picker_context(),
        # Slot pickers must only offer providers the user can actually call.
        # Including setup-only rows makes an unconfigured canonical provider
        # (usually OpenRouter, due to catalog ordering) become the default.
        include_unconfigured=False,
        picker_hints=True,
        canonical_order=True,
        pricing=True,
        capabilities=True,
        max_models=200,
    )
    providers = payload.get("providers") or []
    return [
        p
        for p in providers
        if p.get("slug")
        and str(p.get("slug")).strip().lower() != "moa"
        and p.get("models")
    ]


def _pick_slot(current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Interactive picker for one MoA slot (a reference model or the aggregator).

    Prompts for provider + model, then the two per-slot tuning knobs the config
    schema already supports but the flow never surfaced: ``reasoning_effort``
    (#102582) and ``max_tokens`` (#102584). When ``current`` already names a
    selectable provider/model, offers an in-place edit of just those knobs so
    retuning a preset doesn't require re-walking both pickers (#102585).
    """
    providers = _model_options()
    if not providers:
        raise RuntimeError("No configured model providers found. Run `hermes model` first.")

    current_provider = str((current or {}).get("provider") or "")
    current_model = str((current or {}).get("model") or "")
    current_effort = str((current or {}).get("reasoning_effort") or "").strip().lower() or None
    current_max_tokens = (current or {}).get("max_tokens")
    if not isinstance(current_max_tokens, int) or isinstance(current_max_tokens, bool):
        current_max_tokens = None

    provider_default = next(
        (idx for idx, p in enumerate(providers) if p.get("slug") == current_provider),
        0,
    )

    # Edit-in-place is only offered when the existing pairing is still
    # selectable, so "keep" can never pin a provider/model the picker dropped.
    keep_existing = False
    if current_provider and current_model:
        existing = providers[provider_default]
        if existing.get("slug") == current_provider and current_model in (existing.get("models") or []):
            keep_existing = (
                _prompt_choice(
                    f"{current_provider}:{current_model}",
                    [
                        "Keep provider/model, edit tuning knobs",
                        "Change provider/model",
                    ],
                    0,
                )
                == 0
            )

    if keep_existing:
        provider = providers[provider_default]
        model = current_model
    else:
        provider_rows = [f"{p.get('name') or p.get('slug')}  ({p.get('slug')})" for p in providers]
        provider = providers[_prompt_choice("Select provider", provider_rows, provider_default)]
        models = list(provider.get("models") or [])
        if not models:
            raise RuntimeError(f"Provider {provider.get('slug')} has no selectable models")
        model_default = models.index(current_model) if current_model in models else 0
        model = models[_prompt_choice(f"Select model for {provider.get('slug')}", models, model_default)]
        # Both knobs are per-model budgets; carrying them onto a DIFFERENT model
        # would silently apply a cap/effort the user chose for something else.
        if str(model) != current_model:
            current_effort = None
            current_max_tokens = None

    slot: dict[str, Any] = {
        "provider": str(provider.get("slug") or ""),
        "model": str(model),
    }

    effort_changed, effort = _prompt_slot_reasoning_effort(provider, str(model), current_effort)
    effort = effort if effort_changed else current_effort
    if effort:
        slot["reasoning_effort"] = effort

    mt_changed, max_tokens = _prompt_optional_max_tokens(current_max_tokens)
    max_tokens = max_tokens if mt_changed else current_max_tokens
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        slot["max_tokens"] = max_tokens

    return slot


def _format_slot(slot: dict[str, Any]) -> str:
    label = f"{slot['provider']}:{slot['model']}"
    parts: list[str] = []
    effort = str(slot.get("reasoning_effort") or "").strip()
    if effort:
        parts.append(f"reasoning={effort}")
    max_tokens = slot.get("max_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        parts.append(f"max_tokens={max_tokens}")
    return f"{label} [{', '.join(parts)}]" if parts else label


def _print_config(config: dict[str, Any]) -> None:
    cfg = normalize_moa_config(config.get("moa") if isinstance(config, dict) else {})
    print("Mixture of Agents presets")
    print(f"Default: {cfg['default_preset']}")
    active = cfg.get("active_preset") or "(off)"
    print(f"Active in config: {active}")
    for name, preset in cfg["presets"].items():
        marker = "*" if name == cfg["default_preset"] else " "
        print(f"\n{marker} {name}")
        print("  Reference models:")
        for idx, slot in enumerate(preset["reference_models"], start=1):
            print(f"    {idx}. {_format_slot(slot)}")
        agg = preset["aggregator"]
        print(f"  Aggregator: {_format_slot(agg)}")


def cmd_moa(args) -> None:
    """Manage Mixture of Agents model presets."""
    cfg = load_config()
    sub = getattr(args, "moa_command", None) or "list"

    if sub in {"list", "ls"}:
        _print_config(cfg)
        return

    if sub in {"config", "configure"}:
        moa = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
        preset_name = (getattr(args, "name", None) or moa.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
        current = moa["presets"].get(preset_name, moa["presets"][moa["default_preset"]])
        print(f"Configure MoA preset: {preset_name}")
        print("Pick at least one reference model; choose Done when finished.")
        refs: list[dict[str, Any]] = []
        existing = list(current.get("reference_models") or [])
        idx = 0
        while True:
            base = existing[idx] if idx < len(existing) else None
            picked = _pick_slot(base)
            picked["enabled"] = bool((base or {}).get("enabled", True))
            refs.append(picked)
            idx += 1
            choice = _prompt_choice("Add another reference model?", ["Add another", "Done"], 1)
            if choice == 1:
                break
        print("Configure aggregator model.")
        current = dict(current)
        current["reference_models"] = refs
        current["aggregator"] = _pick_slot(current.get("aggregator"))
        moa["presets"][preset_name] = current
        moa.setdefault("default_preset", preset_name)
        cfg["moa"] = normalize_moa_config(moa)
        save_config(cfg)
        print(f"Saved MoA preset: {preset_name}")
        _print_config(cfg)
        return

    if sub == "delete":
        moa = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
        preset_name = (getattr(args, "name", None) or "").strip()
        if not preset_name:
            raise SystemExit("Usage: hermes moa delete <name>")
        if preset_name not in moa["presets"]:
            raise SystemExit(f"Unknown MoA preset: {preset_name}")
        if len(moa["presets"]) <= 1:
            raise SystemExit("Cannot delete the only MoA preset")
        del moa["presets"][preset_name]
        if moa["default_preset"] == preset_name:
            moa["default_preset"] = next(iter(moa["presets"]))
        if moa.get("active_preset") == preset_name:
            moa["active_preset"] = ""
        cfg["moa"] = normalize_moa_config(moa)
        save_config(cfg)
        print(f"Deleted MoA preset: {preset_name}")
        return

    raise SystemExit(f"Unknown moa subcommand: {sub}")
