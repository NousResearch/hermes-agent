"""CLI helpers for configuring Mixture of Agents."""

from __future__ import annotations

from typing import Any

from hermes_cli.cli_output import line_input
from hermes_cli.config import load_config, save_config
from hermes_cli.inventory import build_models_payload, load_picker_context
from hermes_cli.moa_config import DEFAULT_MOA_PRESET_NAME, normalize_moa_config

# Fixed canonical scale rather than a per-model "supported efforts" list: the
# latter under-reports (see _apply_capabilities in inventory.py — a route can
# serve a level its catalog doesn't advertise), so the picker always offers
# the full scale on any model whose capabilities mark it reasoning-capable.
_REASONING_EFFORT_SCALE = ("low", "medium", "high")


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


def _model_options() -> list[dict[str, Any]]:
    payload = build_models_payload(
        # Keep the profile override inside the worker thread so the full sync picker build (config load,
        # pricing, refresh probes) runs off the event loop under the requested profile. Use
        # _config_profile_scope (contextvar only, no skill-module lock) — the payload build can block for
        # 15s on a models.dev cache miss, and _profile_scope's RLock held across that block starves
        # concurrent /api/config and freezes the server (#58576).
        load_picker_context(),
        # Slot pickers must only offer providers the user can actually call.
        # Including setup-only rows makes an unconfigured canonical provider
        # (usually OpenRouter, due to catalog ordering) become the default.
        include_unconfigured=False,
        picker_hints=True,
        canonical_order=True,
        pricing=True,
        capabilities=True,
        max_models=200)
    providers = payload.get("providers") or []
    return [p for p in providers if p.get("slug") and str(p.get("slug")).strip().lower() != "moa" and p.get("models")]


def _model_supports_reasoning(provider: dict[str, Any], model: str) -> bool:
    entry = (provider.get("capabilities") or {}).get(model) or {}
    return bool(entry.get("reasoning", True))


def _prompt_slot_tuning(
    provider_slug: str,
    model: str,
    current: dict[str, Any] | None,
    *,
    supports_reasoning: bool,
) -> dict[str, Any]:
    """Prompt for the optional per-slot ``reasoning_effort`` / ``max_tokens`` overrides."""
    current = current or {}
    tuning: dict[str, Any] = {}

    if supports_reasoning:
        from hermes_cli.main import _prompt_reasoning_effort_selection

        current_effort = str(current.get("reasoning_effort") or "")
        selected = _prompt_reasoning_effort_selection(_REASONING_EFFORT_SCALE, current_effort=current_effort)
        effort = current_effort if selected is None else selected
        if effort:
            tuning["reasoning_effort"] = effort

    current_tokens = current.get("max_tokens")
    default = str(current_tokens) if current_tokens else ""
    raw = line_input(
        f"Max tokens for {provider_slug}:{model} (blank = no per-slot cap) [{default}]: "
    ).strip()
    tokens = raw or default
    if tokens:
        tuning["max_tokens"] = tokens

    return tuning


def _pick_slot(current: dict[str, Any] | None = None) -> dict[str, Any]:
    if current and current.get("provider") and current.get("model"):
        choice = _prompt_choice(
            "Edit slot",
            [
                "Keep provider/model — edit max_tokens / reasoning effort only",
                "Change provider/model",
            ],
            0,
        )
        if choice == 0:
            providers = _model_options()
            provider = next((p for p in providers if p.get("slug") == current.get("provider")), {})
            supports_reasoning = _model_supports_reasoning(provider, current["model"])
            slot: dict[str, Any] = {"provider": current["provider"], "model": current["model"]}
            slot.update(
                _prompt_slot_tuning(
                    current["provider"], current["model"], current, supports_reasoning=supports_reasoning
                )
            )
            return slot

    providers = _model_options()
    if not providers:
        raise RuntimeError("No configured model providers found. Run `hermes model` first.")
    current_provider = (current or {}).get("provider", "")
    provider_default = next((idx for idx, p in enumerate(providers) if p.get("slug") == current_provider), 0)
    provider_rows = [f"{p.get('name') or p.get('slug')}  ({p.get('slug')})" for p in providers]
    provider = providers[_prompt_choice("Select provider", provider_rows, provider_default)]
    models = list(provider.get("models") or [])
    if not models:
        raise RuntimeError(f"Provider {provider.get('slug')} has no selectable models")
    current_model = (current or {}).get("model", "")
    model_default = models.index(current_model) if current_model in models else 0
    model = models[_prompt_choice(f"Select model for {provider.get('slug')}", models, model_default)]
    supports_reasoning = _model_supports_reasoning(provider, model)
    slot: dict[str, Any] = {"provider": str(provider.get("slug") or ""), "model": str(model)}
    slot.update(_prompt_slot_tuning(provider.get("slug") or "", model, current, supports_reasoning=supports_reasoning))
    return slot


def _format_slot(slot: dict[str, Any]) -> str:
    label = f"{slot['provider']}:{slot['model']}"
    extras = []
    effort = str(slot.get("reasoning_effort") or "").strip()
    if effort:
        extras.append(f"reasoning={effort}")
    max_tokens = slot.get("max_tokens")
    if max_tokens:
        extras.append(f"max_tokens={max_tokens}")
    return f"{label} [{', '.join(extras)}]" if extras else label


def _print_config(config: dict[str, Any]) -> None:
    cfg = _moa_section(config)
    print("Mixture of Agents presets")
    print(f"Default: {cfg['default_preset']}")
    print(f"Active in config: {cfg.get('active_preset') or '(off)'}")
    for name, preset in cfg["presets"].items():
        print(f"\n{'*' if name == cfg['default_preset'] else ' '} {name}")
        print("  Reference models:")
        for idx, slot in enumerate(preset["reference_models"], start=1):
            print(f"    {idx}. {_format_slot(slot)}")
        print(f"  Aggregator: {_format_slot(preset['aggregator'])}")


def _moa_section(cfg: Any) -> dict[str, Any]:
    return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})


def _save(cfg: dict, moa: dict[str, Any]) -> None:
    cfg["moa"] = normalize_moa_config(moa)
    save_config(cfg)


def _cmd_list(cfg: dict, args) -> None:
    _print_config(cfg)


def _cmd_configure(cfg: dict, args) -> None:
    moa = _moa_section(cfg)
    preset_name = (getattr(args, "name", None) or moa.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    current = moa["presets"].get(preset_name, moa["presets"][moa["default_preset"]])
    print(f"Configure MoA preset: {preset_name}")
    print("Pick at least one reference model; choose Done when finished.")
    refs: list[dict[str, str]] = []
    existing = list(current.get("reference_models") or [])
    while True:
        base = existing[len(refs)] if len(refs) < len(existing) else None
        picked = _pick_slot(base)
        picked["enabled"] = bool((base or {}).get("enabled", True))
        refs.append(picked)
        if _prompt_choice("Add another reference model?", ["Add another", "Done"], 1) == 1:
            break
    print("Configure aggregator model.")
    current = dict(current)
    current["reference_models"] = refs
    current["aggregator"] = _pick_slot(current.get("aggregator"))
    moa["presets"][preset_name] = current
    moa.setdefault("default_preset", preset_name)
    _save(cfg, moa)
    print(f"Saved MoA preset: {preset_name}")
    _print_config(cfg)


def _cmd_delete(cfg: dict, args) -> None:
    moa = _moa_section(cfg)
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
    _save(cfg, moa)
    print(f"Deleted MoA preset: {preset_name}")


_SUBCOMMANDS = {
    "list": _cmd_list,
    "ls": _cmd_list,
    "config": _cmd_configure,
    "configure": _cmd_configure,
    "delete": _cmd_delete}


def cmd_moa(args) -> None:
    """Manage Mixture of Agents model presets."""
    cfg = load_config()
    sub = getattr(args, "moa_command", None) or "list"
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        raise SystemExit(f"Unknown moa subcommand: {sub}")
    handler(cfg, args)
