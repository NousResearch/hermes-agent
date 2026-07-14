"""CLI subcommand: ``hermes pets <subcommand>``.

Thin shell around :mod:`agent.pet`.  Browses the public petdex gallery,
installs pets into the profile's ``pets/`` directory, selects the active
mascot (writes ``display.pet.*`` to config.yaml), and runs a doctor check.

No side effects at import time — ``main.py`` wires the argparse subparsers on
demand via :func:`register_cli`.
"""

from __future__ import annotations

import argparse

import logging
logger = logging.getLogger(__name__)

import sys


def _print(msg: str = "") -> None:
    print(msg)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_list(args) -> int:
    """List gallery pets (or only installed ones with ``--installed``)."""
    from agent.pet import store

    if getattr(args, "installed", False):
        pets = store.installed_pets()
        if not pets:
            _print("No pets installed. Try: hermes pets install boba")
            return 0
        _print(f"Installed pets ({len(pets)}):")
        for pet in pets:
            _print(f"  {pet.slug:<24} {pet.display_name}")
        return 0

    from agent.pet.manifest import ManifestError, fetch_manifest

    try:
        entries = fetch_manifest()
    except ManifestError as exc:
        _err(f"✗ {exc}")
        return 1

    query = (getattr(args, "query", "") or "").strip().lower()
    if query:
        entries = [
            e
            for e in entries
            if query in e.slug.lower() or query in e.display_name.lower()
        ]

    limit = getattr(args, "limit", 0) or 0
    shown = entries[:limit] if limit > 0 else entries
    installed = {p.slug for p in store.installed_pets()}

    _print(f"petdex gallery — {len(entries)} pet(s){' matching ' + repr(query) if query else ''}:")
    for entry in shown:
        mark = "✓" if entry.slug in installed else " "
        _print(f"  {mark} {entry.slug:<28} {entry.display_name}  ({entry.kind})")
    if limit and len(entries) > limit:
        _print(f"  … {len(entries) - limit} more (use --limit 0 or --query to filter)")
    _print("\nInstall one with: hermes pets install <slug>")
    return 0


def _cmd_install(args) -> int:
    from agent.pet import store
    from agent.pet.manifest import ManifestError

    slug = args.slug.strip()
    try:
        pet = store.install_pet(slug, force=getattr(args, "force", False))
    except (store.PetStoreError, ManifestError) as exc:
        _err(f"✗ install failed: {exc}")
        return 1

    _print(f"✓ installed {pet.display_name} → {pet.directory}")

    if getattr(args, "select", False) or not _has_active_pet():
        _set_active(slug)
        _print(f"✓ {pet.display_name} is now the active pet (display.pet.slug={slug}, enabled)")
    else:
        _print(f"  Make it active with: hermes pets select {slug}")
    return 0


def _cmd_remove(args) -> int:
    from agent.pet import store

    slug = args.slug.strip()
    if store.remove_pet(slug):
        _print(f"✓ removed {slug}")
        return 0
    _err(f"✗ '{slug}' is not installed")
    return 1


def _cmd_select(args) -> int:
    from agent.pet import store

    slug = (getattr(args, "slug", "") or "").strip()
    if not slug:
        pets = store.installed_pets()
        if not pets:
            _err("✗ no pets installed — run: hermes pets install boba")
            return 1
        slug = _interactive_pick(pets)
        if not slug:
            return 1

    pet = store.load_pet(slug)
    if pet is None or not pet.exists:
        _err(f"✗ '{slug}' is not installed — run: hermes pets install {slug}")
        return 1

    _set_active(slug)
    _print(f"✓ active pet set to {pet.display_name} (display.pet.slug={slug}, enabled)")
    return 0


def _cmd_off(args) -> int:
    _set_enabled(False)
    _print("✓ pet disabled (display.pet.enabled=false)")
    return 0


def _cmd_scale(args) -> int:
    """Persist ``display.pet.scale`` — one knob resizes every surface."""
    scale, err = set_pet_scale(args.factor)
    if err:
        _err(f"✗ {err}")
        return 1
    _print(f"✓ pet scale set to {scale:g} (display.pet.scale)")
    return 0


def _cmd_show(args) -> int:
    """Animate the active (or named) pet in the terminal.

    Uses the shared :class:`~agent.pet.render.PetRenderer` — full graphics
    protocol (kitty/iTerm2/sixel) when the terminal supports it, else a
    truecolor Unicode half-block fallback.  Ctrl+C to stop.
    """
    import time

    from agent.pet import store
    from agent.pet.constants import DEFAULT_SCALE, LOOP_MS, STATE_ROWS, PetState, resolve_cols
    from agent.pet.render import build_renderer

    cfg = _pet_config()
    slug = (getattr(args, "slug", "") or "").strip() or str(cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(slug)
    if pet is None:
        _err("✗ no pet to show — run: hermes pets install boba")
        return 1

    mode_cfg = getattr(args, "mode", None) or str(cfg.get("render_mode", "auto") or "auto")
    scale = float(getattr(args, "scale", 0) or cfg.get("scale", DEFAULT_SCALE) or DEFAULT_SCALE)
    cols = resolve_cols(scale, cfg.get("unicode_cols", 0))

    renderer = build_renderer(
        pet.spritesheet,
        configured_mode=mode_cfg,
        scale=scale,
        unicode_cols=cols,
    )
    if not renderer.available:
        _err(
            "✗ cannot render here (no TTY / graphics disabled). "
            f"Effective mode: {renderer.mode}."
        )
        return 1

    # Which states to play: one named state, or cycle the driveable rows.
    requested = (getattr(args, "state", "") or "").strip().lower()
    if requested:
        states = [requested]
    elif getattr(args, "cycle", False):
        states = [s for s in STATE_ROWS if s in {e.value for e in PetState}]
    else:
        states = [PetState.IDLE.value]

    is_unicode = renderer.mode == "unicode"
    frame_delay = max(0.05, (LOOP_MS / 1000.0) / max(1, renderer.frame_count(states[0]) or 1))

    # Right-align the sprite against the terminal's right edge — half-blocks by
    # indenting each row, graphics protocols by padding the cursor to the right
    # column before the image draws (kitty/iTerm/sixel all render at the cursor).
    import shutil

    term_cols = shutil.get_terminal_size((80, 24)).columns
    indent = ""
    g_indent = ""
    if is_unicode:
        indent = " " * max(0, term_cols - cols - 1)
    else:
        cell_cols = max(1, int(renderer.frame_w * renderer.scale) // 8)
        g_indent = " " * max(0, term_cols - cell_cols - 1)

    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.flush()
    prev_lines = 0
    try:
        _print(f"{pet.display_name} — mode={renderer.mode}  (Ctrl+C to stop)")
        loops = 0
        while True:
            for state in states:
                count = renderer.frame_count(state) or 1
                for i in range(count):
                    encoded = renderer.frame(state, i)
                    if is_unicode:
                        if indent:
                            encoded = "\n".join(indent + ln for ln in encoded.split("\n"))
                        if prev_lines:
                            out.write(f"\x1b[{prev_lines}F")  # cursor up to redraw in place
                        out.write(encoded)
                        out.write("\x1b[0m\n")
                        # Lines drawn = sprite rows + the trailing newline; move
                        # back up exactly that many so the next frame overwrites.
                        prev_lines = encoded.count("\n") + 1
                    else:
                        out.write("\x1b[2J\x1b[3J\x1b[H")  # clear for image protocols
                        out.write(f"{pet.display_name} [{state}]\n")
                        if g_indent:
                            out.write(g_indent)
                        out.write(encoded)
                        out.write("\n")
                    out.flush()
                    time.sleep(frame_delay)
            loops += 1
            if getattr(args, "once", False) and loops >= len(states):
                break
    except KeyboardInterrupt:
        logger.debug("Suppressed exception", exc_info=True)
    finally:
        out.write("\x1b[?25h")  # show cursor
        out.write("\x1b[0m\n")
        out.flush()
    return 0


def _cmd_doctor(args) -> int:
    """Report install state, active pet, config, and terminal capability."""
    from agent.pet import store
    from agent.pet.render import detect_terminal_graphics, resolve_mode

    cfg = _pet_config()
    enabled = bool(cfg.get("enabled"))
    configured_slug = str(cfg.get("slug", "") or "")
    mode_cfg = str(cfg.get("render_mode", "auto") or "auto")

    pets = store.installed_pets()
    active = store.resolve_active_pet(configured_slug)

    _print("petdex doctor")
    _print(f"  pets dir:        {store.pets_dir()}")
    _print(f"  installed:       {len(pets)} ({', '.join(p.slug for p in pets) or 'none'})")
    _print(f"  display.pet.enabled:     {enabled}")
    _print(f"  display.pet.slug:        {configured_slug or '(unset)'}")
    _print(f"  active (resolved):       {active.slug if active else '(none)'}")
    _print(f"  display.pet.render_mode: {mode_cfg}")
    _print(f"  detected graphics:       {detect_terminal_graphics()}")
    _print(f"  effective mode (TTY):    {resolve_mode(mode_cfg)}")

    ok = True
    if not pets:
        _print("  → no pets installed. Run: hermes pets install boba")
        ok = False
    elif active is None:
        _print("  → active pet unresolved. Run: hermes pets select <slug>")
        ok = False
    elif not enabled:
        _print("  → pet display is disabled. Run: hermes pets select " + active.slug)

    try:
        import PIL  # noqa: F401
    except ImportError:
        _print("  ✗ Pillow not importable — sprite decoding will be unavailable")
        ok = False

    _print("  ✓ ready" if ok and enabled else "  (run the suggestions above to finish setup)")
    return 0


# ─────────────────────────────────────────────────────────────────────────
# config helpers
# ─────────────────────────────────────────────────────────────────────────

def _pet_config() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config()
    display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
    pet = display.get("pet", {})
    return pet if isinstance(pet, dict) else {}


def _has_active_pet() -> bool:
    return bool(_pet_config().get("enabled")) and bool(_pet_config().get("slug"))


def _set_active(slug: str) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    display = cfg.setdefault("display", {})
    pet = display.setdefault("pet", {})
    pet["slug"] = slug
    pet["enabled"] = True
    save_config(cfg)


def _set_enabled(enabled: bool) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    display = cfg.setdefault("display", {})
    pet = display.setdefault("pet", {})
    pet["enabled"] = enabled
    save_config(cfg)


def _set_scale(scale: float) -> None:
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    display = cfg.setdefault("display", {})
    pet = display.setdefault("pet", {})
    pet["scale"] = scale
    save_config(cfg)


def set_pet_scale(value: float | str) -> tuple[float, str | None]:
    """Set ``display.pet.scale`` (clamped to bounds). Returns ``(applied, error)``.

    The single write path behind ``/pet scale`` and the desktop slider, so every
    surface that resolves scale from config picks it up identically. *error* is
    set (and nothing written) only when *value* isn't a number.
    """
    from agent.pet.constants import clamp_scale

    try:
        scale = clamp_scale(float(value))
    except (TypeError, ValueError):
        return 0.0, f"not a number: {value!r} — try a value like 0.5"

    _set_scale(scale)
    return scale, None


def toggle_pet_display() -> tuple[bool, str | None, str | None]:
    """Toggle ``display.pet.enabled``.

    Returns ``(enabled, display_name, error_message)``. *error_message* is set
    when turning on but nothing is installed to show.
    """
    from agent.pet import store

    cfg = _pet_config()
    slug = str(cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(slug)

    if bool(cfg.get("enabled")):
        _set_enabled(False)
        return False, pet.display_name if pet else None, None

    if pet is None:
        installed = store.installed_pets()
        if not installed:
            return False, None, "no pets installed — /pet list to browse, or /pet <slug> to adopt"
        pet = installed[0]
        _set_active(pet.slug)
    else:
        _set_enabled(True)
    return True, pet.display_name, None


def print_pet_gallery(*, limit: int = 20) -> None:
    """Print a slice of the public petdex gallery (CLI/TUI text fallback)."""
    from agent.pet import store
    from agent.pet.manifest import ManifestError, fetch_manifest

    try:
        entries = fetch_manifest()
    except ManifestError as exc:
        print(f"(._.) Couldn't reach the petdex gallery: {exc}")
        return

    installed = {p.slug for p in store.installed_pets()}
    shown = entries[:limit] if limit > 0 else entries
    print(f"(^o^)/ petdex gallery — first {len(shown)} of {len(entries)}:")
    for entry in shown:
        mark = "●" if entry.slug in installed else "○"
        print(f"  {mark} {entry.slug:<24} {entry.display_name}")
    print("  /pet <slug> to adopt · /pet to toggle")


def _clear_active_if(slug: str) -> bool:
    """Disable + unset the active pet iff it's ``slug`` (e.g. after removal).

    Returns whether anything changed, so callers don't write config needlessly.
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    pet = cfg.setdefault("display", {}).setdefault("pet", {})
    if not isinstance(pet, dict) or str(pet.get("slug", "") or "") != slug:
        return False
    pet["slug"] = ""
    pet["enabled"] = False
    save_config(cfg)
    return True


def _rename_active_if(old_slug: str, new_slug: str) -> bool:
    """Repoint the active pet from ``old_slug`` to ``new_slug`` iff it's active.

    Used when a rename realigns a pet's slug/dir: if the renamed pet was the
    active one, the config must follow or surfaces point at a now-missing dir.
    Preserves the ``enabled`` flag. Returns whether anything changed.
    """
    if not new_slug or old_slug == new_slug:
        return False
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    pet = cfg.setdefault("display", {}).setdefault("pet", {})
    if not isinstance(pet, dict) or str(pet.get("slug", "") or "") != old_slug:
        return False
    pet["slug"] = new_slug
    save_config(cfg)
    return True


def _interactive_pick(pets) -> str:
    """Minimal numbered picker (avoids curses dep for a tiny list)."""
    _print("Installed pets:")
    for i, pet in enumerate(pets, 1):
        _print(f"  {i}. {pet.slug:<24} {pet.display_name}")
    try:
        choice = input("Select a pet [1]: ").strip() or "1"
        idx = int(choice) - 1
    except (EOFError, KeyboardInterrupt, ValueError):
        _err("✗ cancelled")
        return ""
    if 0 <= idx < len(pets):
        return pets[idx].slug
    _err("✗ invalid selection")
    return ""


# ─────────────────────────────────────────────────────────────────────────
# argparse wiring
# ─────────────────────────────────────────────────────────────────────────

def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach ``pets`` subcommands to *parent* (called by main.py)."""
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="pets_command")

    p_list = subs.add_parser("list", help="Browse the petdex gallery")
    p_list.add_argument("query", nargs="?", default="", help="Filter by slug/name substring")
    p_list.add_argument("--installed", action="store_true", help="Only show installed pets")
    p_list.add_argument("--limit", type=int, default=40, help="Max rows (0 = all)")
    p_list.set_defaults(func=_cmd_list)

    p_install = subs.add_parser("install", help="Install a pet from the gallery")
    p_install.add_argument("slug", help="Pet slug (e.g. boba)")
    p_install.add_argument("--force", action="store_true", help="Re-download even if present")
    p_install.add_argument("--select", action="store_true", help="Make it the active pet")
    p_install.set_defaults(func=_cmd_install)

    p_select = subs.add_parser("select", help="Set the active pet (writes display.pet.*)")
    p_select.add_argument("slug", nargs="?", default="", help="Pet slug (omit for picker)")
    p_select.set_defaults(func=_cmd_select)

    p_show = subs.add_parser("show", help="Animate the active pet in the terminal")
    p_show.add_argument("slug", nargs="?", default="", help="Pet slug (default: active)")
    p_show.add_argument("--state", default="", help="Single state: idle/run/review/failed/wave/jump")
    p_show.add_argument("--cycle", action="store_true", help="Cycle through all states")
    p_show.add_argument("--once", action="store_true", help="Play once instead of looping")
    p_show.add_argument("--mode", default=None, help="Override render mode (kitty/iterm/sixel/unicode/auto)")
    p_show.add_argument("--scale", type=float, default=0, help="Override scale (0 = config)")
    p_show.set_defaults(func=_cmd_show)

    subs.add_parser("off", help="Disable the pet display").set_defaults(func=_cmd_off)

    p_scale = subs.add_parser("scale", help="Resize the pet everywhere (display.pet.scale)")
    p_scale.add_argument("factor", help="Scale factor, e.g. 0.5 (clamped 0.1–3.0)")
    p_scale.set_defaults(func=_cmd_scale)

    p_remove = subs.add_parser("remove", help="Delete an installed pet")
    p_remove.add_argument("slug", help="Pet slug")
    p_remove.set_defaults(func=_cmd_remove)

    subs.add_parser("doctor", help="Check pet setup + terminal graphics support").set_defaults(
        func=_cmd_doctor
    )
