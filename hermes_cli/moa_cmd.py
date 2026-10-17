"""CLI helpers for configuring Mixture of Agents."""

from __future__ import annotations

import json
from pathlib import Path
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


def _pick_slot(current: dict[str, str] | None = None) -> dict[str, str]:
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
    return {"provider": str(provider.get("slug") or ""), "model": str(model)}


def _format_slot(slot: dict[str, Any]) -> str:
    label = f"{slot['provider']}:{slot['model']}"
    effort = str(slot.get("reasoning_effort") or "").strip()
    return f"{label} [reasoning={effort}]" if effort else label


def _provider_mismatch_notice(
    cfg: dict[str, Any], aggregator: dict[str, Any]
) -> str | None:
    main_provider = ""
    if isinstance(cfg, dict):
        model_section = cfg.get("model")
        if isinstance(model_section, dict):
            main_provider = str(model_section.get("provider") or "").strip().lower()
    agg_provider = str((aggregator or {}).get("provider") or "").strip().lower()
    if (
        not main_provider
        or not agg_provider
        or main_provider in ("moa", "auto")  # "auto" is a routing pseudo-provider, not a billing seat
        or main_provider == agg_provider
    ):
        return None
    return (
        f"Aggregator is on {agg_provider}; the whole tool loop will be billed there, "
        f"not to {main_provider}."
    )


def _print_config(config: dict[str, Any]) -> None:
    cfg = _moa_section(config)
    print("Mixture of Agents presets")
    print(f"Default: {cfg['default_preset']}")
    print(f"Active in config: {cfg.get('active_preset') or '(off)'}")
    for name, preset in cfg["presets"].items():
        print(f"\n{'*' if name == cfg['default_preset'] else ' '} {name}")
        print("  Reference models (advise once per user turn by default):")
        for idx, slot in enumerate(preset["reference_models"], start=1):
            print(f"    {idx}. {_format_slot(slot)}")
        agg_slot = preset["aggregator"]
        print(
            f"  Aggregator: {_format_slot(agg_slot)} (acting model — runs every step and carries almost all of the cost)"
        )
        notice = _provider_mismatch_notice(config, agg_slot)
        if notice:
            print(f"    note: {notice}")


def _moa_section(cfg: Any) -> dict[str, Any]:
    return normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})


def _save(cfg: dict, moa: dict[str, Any]) -> None:
    cfg["moa"] = normalize_moa_config(moa)
    save_config(cfg)


def _cmd_list(cfg: dict, args) -> None:
    _print_config(cfg)


def _slot_from_text(entry: Any, *, where: str) -> dict[str, str]:
    """``provider/model`` text, or a ``{provider, model}`` mapping, as a slot dict.

    Fails closed: ``normalize_moa_config`` deliberately drops a half-filled slot at save time
    (read-time tolerance is not a write-time contract), so a typo in a declared slot would
    otherwise write a preset that is quietly missing that model. Splits on the FIRST ``/`` —
    provider slugs never contain one, while model ids carry their own namespace
    (``openrouter`` + ``deepseek/deepseek-v4-pro``).
    """
    if isinstance(entry, dict):
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
    elif isinstance(entry, str):
        provider, _, model = entry.strip().partition("/")
        provider, model = provider.strip(), model.strip()
    else:
        raise SystemExit(f"{where}: expected 'provider/model', got {entry!r}")
    if not provider or not model:
        raise SystemExit(f"{where}: '{entry}' must be 'provider/model' with both halves filled")
    if provider.lower() == "moa":
        # MoA is a virtual provider: a preset inside a preset is a recursive tree the runtime
        # only catches mid-turn (same rule as moa_config._slot_problem).
        raise SystemExit(f"{where}: '{entry}' — the MoA provider cannot be used inside a preset")
    return {"provider": provider, "model": model}


def _declared_slots(args) -> tuple[list[dict[str, str]] | None, str | None] | None:
    """``(reference_models, aggregator)`` declared on the command line; None to stay interactive.

    Any of ``--slots`` / ``--slots-file`` / ``--aggregator`` switches ``configure`` to the
    declarative path. A file object's ``aggregator`` is a default that ``--aggregator`` overrides;
    when both slot sources are given the file wins (it carries the richer payload).
    """
    slots = getattr(args, "slots", None)
    slots_file = getattr(args, "slots_file", None)
    aggregator = str(getattr(args, "aggregator", None) or "").strip() or None
    if not slots and not slots_file and not aggregator:
        return None

    refs: list[dict[str, str]] | None = None
    if slots_file:
        where = f"--slots-file {slots_file}"
        try:
            payload = json.loads(Path(slots_file).read_text(encoding="utf-8"))
        except OSError as exc:
            raise SystemExit(f"{where}: {exc}")
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{where}: not valid JSON ({exc})")
        if isinstance(payload, dict):
            aggregator = aggregator or (str(payload.get("aggregator") or "").strip() or None)
            payload = payload.get("reference_models")
        if not isinstance(payload, list) or not payload:
            raise SystemExit(f"{where}: needs a non-empty list of 'provider/model' slots")
        refs = [_slot_from_text(item, where=where) for item in payload]
    if slots:
        entries = [part for part in str(slots).split(",") if part.strip()]
        if not entries:
            raise SystemExit("--slots: no slots given")
        refs = [_slot_from_text(part, where="--slots") for part in entries]
    return refs, aggregator


def _configure_declared(
    cfg: dict, moa: dict[str, Any], preset_name: str,
    refs: list[dict[str, Any]] | None, aggregator: str | None,
) -> None:
    """Write a preset from declared slots without ever opening the picker.

    Declared slots REPLACE the preset's reference list — a preset is declarative state, so
    ``--slots`` converges it instead of appending. Whatever the caller did not declare is
    inherited from the preset being updated; creating a preset still needs an aggregator,
    because there is nothing to inherit and choosing one silently would prompt.
    """
    existing = moa["presets"].get(preset_name)
    if refs is None:
        if existing is None:
            raise SystemExit(
                f"--aggregator needs an existing preset to update: no MoA preset named "
                f"'{preset_name}' (pass --slots to create it)")
        refs = [{**slot, "enabled": bool(slot.get("enabled", True))}
                for slot in existing.get("reference_models") or []]
    else:
        refs = [{**slot, "enabled": True} for slot in refs]
    if aggregator is not None:
        aggregator_slot = _slot_from_text(aggregator, where="--aggregator")
    elif existing is not None:
        aggregator_slot = dict(existing.get("aggregator") or {})
    else:
        raise SystemExit(f"--aggregator is required to create MoA preset '{preset_name}'")
    preset = dict(existing or {})
    preset["reference_models"] = refs
    preset["aggregator"] = aggregator_slot
    moa["presets"][preset_name] = preset
    moa.setdefault("default_preset", preset_name)
    notice = _provider_mismatch_notice(cfg, aggregator_slot)
    if notice:
        print(notice)
    _save(cfg, moa)
    print(f"Saved MoA preset: {preset_name}")
    _print_config(cfg)


def _cmd_configure(cfg: dict, args) -> None:
    moa = _moa_section(cfg)
    preset_name = (getattr(args, "name", None) or moa.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    declared = _declared_slots(args)
    if declared is not None:
        _configure_declared(cfg, moa, preset_name, *declared)
        return
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
    print(
        "The aggregator is the acting model: it runs every tool-loop step, and almost all of the run's cost lands on its provider."
    )
    current = dict(current)
    current["reference_models"] = refs
    current["aggregator"] = _pick_slot(current.get("aggregator"))
    moa["presets"][preset_name] = current
    moa.setdefault("default_preset", preset_name)
    notice = _provider_mismatch_notice(cfg, current["aggregator"])
    if notice:
        print(notice)
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
