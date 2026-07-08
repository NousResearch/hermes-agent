"""CLI helpers for configuring the Model Router."""

from __future__ import annotations

from typing import Any

from hermes_cli.config import load_config, save_config
from hermes_cli.moa_cmd import _pick_slot, _prompt_choice
from hermes_cli.router_config import (
    DEFAULT_ROUTER_PRESET_NAME,
    ROUTER_TIERS,
    normalize_router_config,
    resolve_router_preset,
)


def _slot_str(slot: dict[str, str] | None) -> str:
    if not slot:
        return "(none)"
    return f"{slot.get('provider', '')}:{slot.get('model', '')}"


def _context_warning(slot: dict[str, str] | None) -> str | None:
    """Soft warning when a route model's known context window is below the
    64K agent minimum.

    Routed slots bypass the hard primary-model context gate (the router is a
    virtual provider, like MoA), so a small local model IS usable as a tier —
    but long sessions will lean on conversation compression. Surface that as
    guidance, not an error.
    """
    if not slot or not slot.get("model"):
        return None
    try:
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH, get_model_context_length

        ctx = get_model_context_length(slot["model"], provider=slot.get("provider", ""))
        if ctx and 0 < ctx < MINIMUM_CONTEXT_LENGTH:
            return (
                f"    ⚠ {_slot_str(slot)} reports a {ctx:,}-token context window "
                f"(below the {MINIMUM_CONTEXT_LENGTH:,} agent minimum). Routed turns "
                "still work, but long sessions will rely on conversation "
                "compression — load the model at its full window if possible."
            )
    except Exception:
        pass
    return None


def _print_config(config: dict[str, Any]) -> None:
    cfg = normalize_router_config(config.get("router") if isinstance(config, dict) else {})
    print("Model Router presets")
    print(f"Default: {cfg['default_preset']}")
    active = cfg.get("active_preset") or "(off)"
    print(f"Active in config: {active}")
    for name, preset in cfg["presets"].items():
        marker = "*" if name == cfg["default_preset"] else " "
        print(f"\n{marker} {name}" + ("" if preset.get("enabled", True) else "  (disabled)"))
        print(f"  Classifier: {_slot_str(preset['classifier'])}")
        print("  Routes:")
        for tier in ROUTER_TIERS:
            slot = preset["routes"].get(tier)
            default_marker = "  ← default_route" if tier == preset.get("default_route") else ""
            print(f"    {tier:8s} {_slot_str(slot)}{default_marker}")
            warning = _context_warning(slot)
            if warning:
                print(warning)
        fallbacks = preset.get("fallbacks") or []
        if fallbacks:
            print("  Fallbacks (in order):")
            for idx, slot in enumerate(fallbacks, start=1):
                print(f"    {idx}. {_slot_str(slot)}")
                warning = _context_warning(slot)
                if warning:
                    print(warning)
        else:
            print("  Fallbacks: (none)")
        hints = preset.get("channel_hints") or {}
        if hints:
            print("  Channel hints: " + ", ".join(f"{k}→{v}" for k, v in hints.items()))


def cmd_router(args) -> None:
    """Manage Model Router presets."""
    cfg = load_config()
    sub = getattr(args, "router_command", None) or "list"

    if sub in {"list", "ls"}:
        _print_config(cfg)
        return

    if sub in {"config", "configure"}:
        router = normalize_router_config(cfg.get("router") if isinstance(cfg, dict) else {})
        preset_name = (
            getattr(args, "name", None)
            or router.get("default_preset")
            or DEFAULT_ROUTER_PRESET_NAME
        ).strip()
        current = router["presets"].get(preset_name, router["presets"][router["default_preset"]])
        current = dict(current)
        print(f"Configure Router preset: {preset_name}")
        print("The classifier reads each prompt and picks a tier; keep it on a")
        print("strong model — its output is one word, so per-call cost is tiny.")
        print("\nConfigure classifier model.")
        current["classifier"] = _pick_slot(current.get("classifier"))
        routes = dict(current.get("routes") or {})
        print("\nConfigure the 'simple' tier (casual chat, quick questions).")
        routes["simple"] = _pick_slot(routes.get("simple"))
        print("\nConfigure the 'complex' tier (coding, long/multi-step tasks).")
        routes["complex"] = _pick_slot(routes.get("complex"))
        current["routes"] = routes
        print("\nConfigure fallback models (tried in order when the routed")
        print("model fails); choose Done to stop adding.")
        fallbacks: list[dict[str, str]] = []
        existing = list(current.get("fallbacks") or [])
        idx = 0
        while True:
            choice = _prompt_choice(
                "Add a fallback model?" if not fallbacks else "Add another fallback?",
                ["Add", "Done"],
                0 if idx < len(existing) else 1,
            )
            if choice == 1:
                break
            base = existing[idx] if idx < len(existing) else None
            fallbacks.append(_pick_slot(base))
            idx += 1
        current["fallbacks"] = fallbacks
        router["presets"][preset_name] = current
        router.setdefault("default_preset", preset_name)
        cfg["router"] = normalize_router_config(router)
        save_config(cfg)
        print(f"Saved Router preset: {preset_name}")
        _print_config(cfg)
        return

    if sub == "delete":
        router = normalize_router_config(cfg.get("router") if isinstance(cfg, dict) else {})
        preset_name = (getattr(args, "name", None) or "").strip()
        if not preset_name:
            raise SystemExit("Usage: hermes router delete <name>")
        if preset_name not in router["presets"]:
            raise SystemExit(f"Unknown Router preset: {preset_name}")
        if len(router["presets"]) <= 1:
            raise SystemExit("Cannot delete the only Router preset")
        del router["presets"][preset_name]
        if router["default_preset"] == preset_name:
            router["default_preset"] = next(iter(router["presets"]))
        if router.get("active_preset") == preset_name:
            router["active_preset"] = ""
        cfg["router"] = normalize_router_config(router)
        save_config(cfg)
        print(f"Deleted Router preset: {preset_name}")
        return

    if sub == "test":
        prompt = " ".join(getattr(args, "prompt", None) or []).strip()
        if not prompt:
            raise SystemExit('Usage: hermes router test "<prompt>"')
        preset_name = (getattr(args, "name", None) or "").strip() or None
        preset = resolve_router_preset(cfg.get("router") or {}, preset_name)
        platform = (getattr(args, "platform", None) or "").strip() or None

        from agent.router_loop import RouterChatCompletions

        facade = RouterChatCompletions(
            preset_name or "default", platform=platform
        )
        print(f"Classifier: {_slot_str(preset['classifier'])}")
        if platform:
            hint = (preset.get("channel_hints") or {}).get(platform.lower())
            print(f"Channel: {platform}" + (f" (hint: {hint})" if hint else " (no hint)"))
        print("Classifying...")
        tier, record = facade._classify(preset, [{"role": "user", "content": prompt}])
        raw = record.get("raw_output")
        if record.get("failed"):
            print(f"✗ Classifier failed: {record.get('error')}")
            print(f"→ Failing open to default_route: {tier}")
        elif record.get("skipped"):
            print(f"→ Classifier skipped ({record['skipped']}); verdict: {tier}")
        else:
            print(f"Raw output: {raw!r}")
            print(f"Verdict: {tier}")
        slot = (preset.get("routes") or {}).get(tier) or {}
        print(f"Route: {tier} → {_slot_str(slot)}")
        warning = _context_warning(slot)
        if warning:
            print(warning)
        fallbacks = preset.get("fallbacks") or []
        if fallbacks:
            print("Fallback chain: " + " → ".join(_slot_str(s) for s in fallbacks))
        return

    raise SystemExit(f"Unknown router subcommand: {sub}")
