"""CLI subcommand: ``hermes pets <subcommand>``."""

from __future__ import annotations

import argparse
import sys

from utils import is_truthy_value


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_list(args) -> int:
    """List gallery pets (or only installed ones with ``--installed``)."""
    from agent.pet import store

    if args.installed:
        pets = store.installed_pets()
        if not pets:
            print("No pets installed. Try: hermes pets install boba")
            return 0
        print(f"Installed pets ({len(pets)}):")
        for pet in pets:
            print(f"  {pet.slug:<24} {pet.display_name}")
        return 0

    from agent.pet.manifest import ManifestError, fetch_manifest

    try:
        entries = fetch_manifest()
    except ManifestError as exc:
        _err(f"✗ {exc}")
        return 1

    query = (args.query or "").strip().lower()
    if query:
        entries = [e for e in entries if query in e.slug.lower() or query in e.display_name.lower()]

    limit = args.limit or 0
    shown = entries[:limit] if limit > 0 else entries
    installed = {p.slug for p in store.installed_pets()}

    print(f"petdex gallery — {len(entries)} pet(s){' matching ' + repr(query) if query else ''}:")
    for entry in shown:
        mark = "✓" if entry.slug in installed else " "
        print(f"  {mark} {entry.slug:<28} {entry.display_name}  ({entry.kind})")
    if limit and len(entries) > limit:
        print(f"  … {len(entries) - limit} more (use --limit 0 or --query to filter)")
    print("\nInstall one with: hermes pets install <slug>")
    return 0


def _cmd_install(args) -> int:
    from agent.pet import store
    from agent.pet.manifest import ManifestError

    slug = args.slug.strip()
    try:
        pet = store.install_pet(slug, force=args.force)
    except (store.PetStoreError, ManifestError) as exc:
        _err(f"✗ install failed: {exc}")
        return 1

    print(f"✓ installed {pet.display_name} → {pet.directory}")

    if args.select or not _has_active_pet():
        _set_active(slug)
        print(f"✓ {pet.display_name} is now the active pet (display.pet.slug={slug}, enabled)")
    else:
        print(f"  Make it active with: hermes pets select {slug}")
    return 0


def _cmd_remove(args) -> int:
    from agent.pet import store

    slug = args.slug.strip()
    if store.remove_pet(slug):
        print(f"✓ removed {slug}")
        return 0
    _err(f"✗ '{slug}' is not installed")
    return 1


def _cmd_select(args) -> int:
    from agent.pet import store

    slug = (args.slug or "").strip()
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
    print(f"✓ active pet set to {pet.display_name} (display.pet.slug={slug}, enabled)")
    return 0


def _cmd_off(args) -> int:
    _set_enabled(False)
    print("✓ pet disabled (display.pet.enabled=false)")
    return 0


def _cmd_scale(args) -> int:
    """Persist ``display.pet.scale`` — one knob resizes every surface."""
    scale, err = set_pet_scale(args.factor)
    if err:
        _err(f"✗ {err}")
        return 1
    print(f"✓ pet scale set to {scale:g} (display.pet.scale)")
    return 0


def _cmd_show(args) -> int:
    """Animate the active (or named) pet in the terminal via the shared PetRenderer
    (kitty/iTerm2/sixel when supported, else truecolor half-block fallback). Ctrl+C stops."""
    import shutil
    import time

    from agent.pet import store
    from agent.pet.constants import DEFAULT_SCALE, LOOP_MS, STATE_ROWS, PetState, resolve_cols
    from agent.pet.render import build_renderer

    cfg = _pet_config()
    slug = (args.slug or "").strip() or str(cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(slug)
    if pet is None:
        _err("✗ no pet to show — run: hermes pets install boba")
        return 1

    mode_cfg = args.mode or str(cfg.get("render_mode", "auto") or "auto")
    scale = float(args.scale or cfg.get("scale", DEFAULT_SCALE) or DEFAULT_SCALE)
    cols = resolve_cols(scale, cfg.get("unicode_cols", 0))

    renderer = build_renderer(pet.spritesheet, configured_mode=mode_cfg, scale=scale, unicode_cols=cols)
    if not renderer.available:
        _err(f"✗ cannot render here (no TTY / graphics disabled). Effective mode: {renderer.mode}.")
        return 1

    # Which states to play: one named state, or cycle the driveable rows.
    requested = (args.state or "").strip().lower()
    if requested:
        states = [requested]
    elif args.cycle:
        states = [s for s in STATE_ROWS if s in {e.value for e in PetState}]
    else:
        states = [PetState.IDLE.value]

    is_unicode = renderer.mode == "unicode"
    frame_delay = max(0.05, (LOOP_MS / 1000.0) / max(1, renderer.frame_count(states[0]) or 1))

    # Right-align against the terminal's right edge — half-blocks by indenting each row, graphics
    # protocols by padding the cursor (kitty/iTerm/sixel all render at the cursor).
    term_cols = shutil.get_terminal_size((80, 24)).columns
    sprite_cols = cols if is_unicode else max(1, int(renderer.frame_w * renderer.scale) // 8)
    indent = " " * max(0, term_cols - sprite_cols - 1)

    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.flush()
    prev_lines = 0
    try:
        print(f"{pet.display_name} — mode={renderer.mode}  (Ctrl+C to stop)")
        loops = 0
        while True:
            for state in states:
                for i in range(renderer.frame_count(state) or 1):
                    encoded = renderer.frame(state, i)
                    if is_unicode:
                        if indent:
                            encoded = "\n".join(indent + ln for ln in encoded.split("\n"))
                        if prev_lines:
                            out.write(f"\x1b[{prev_lines}F")  # cursor up to redraw in place
                        out.write(encoded)
                        out.write("\x1b[0m\n")
                        # Lines drawn = sprite rows + trailing newline; the next frame overwrites.
                        prev_lines = encoded.count("\n") + 1
                    else:
                        out.write("\x1b[2J\x1b[3J\x1b[H")  # clear for image protocols
                        out.write(f"{pet.display_name} [{state}]\n")
                        out.write(indent + encoded + "\n")
                    out.flush()
                    time.sleep(frame_delay)
            loops += 1
            if args.once and loops >= len(states):
                break
    except KeyboardInterrupt:
        pass
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
    enabled = _pet_enabled(cfg)
    configured_slug = str(cfg.get("slug", "") or "")
    mode_cfg = str(cfg.get("render_mode", "auto") or "auto")

    pets = store.installed_pets()
    active = store.resolve_active_pet(configured_slug)

    print("petdex doctor")
    print(f"  pets dir:        {store.pets_dir()}")
    print(f"  installed:       {len(pets)} ({', '.join(p.slug for p in pets) or 'none'})")
    print(f"  display.pet.enabled:     {enabled}")
    print(f"  display.pet.slug:        {configured_slug or '(unset)'}")
    print(f"  active (resolved):       {active.slug if active else '(none)'}")
    print(f"  display.pet.render_mode: {mode_cfg}")
    print(f"  detected graphics:       {detect_terminal_graphics()}")
    print(f"  effective mode (TTY):    {resolve_mode(mode_cfg)}")

    ok = True
    if not pets:
        print("  → no pets installed. Run: hermes pets install boba")
        ok = False
    elif active is None:
        print("  → active pet unresolved. Run: hermes pets select <slug>")
        ok = False
    elif not enabled:
        print("  → pet display is disabled. Run: hermes pets select " + active.slug)

    try:
        import PIL  # noqa: F401
    except ImportError:
        print("  ✗ Pillow not importable — sprite decoding will be unavailable")
        ok = False

    print("  ✓ ready" if ok and enabled else "  (run the suggestions above to finish setup)")
    return 0


# ── config helpers ────────────────────────────────────────────────────────

def _pet_config() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config()
    display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
    pet = display.get("pet", {})
    return pet if isinstance(pet, dict) else {}


def _has_active_pet() -> bool:
    cfg = _pet_config()
    return _pet_enabled(cfg) and bool(cfg.get("slug"))


def _pet_enabled(cfg: dict) -> bool:
    return is_truthy_value(cfg.get("enabled"), default=False)


def _update_pet_config(when_slug: str | None = None, **values) -> bool:
    """Write ``display.pet.*`` keys and save config; returns whether anything was written.

    With ``when_slug`` the write only happens when the configured slug equals it (remove/rename
    must never disturb a different active pet).
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    pet = cfg.setdefault("display", {}).setdefault("pet", {})
    if when_slug is not None and (not isinstance(pet, dict) or str(pet.get("slug", "") or "") != when_slug):
        return False
    pet.update(values)
    save_config(cfg)
    return True


def _set_active(slug: str) -> None:
    _update_pet_config(slug=slug, enabled=True)


def _set_enabled(enabled: bool) -> None:
    _update_pet_config(enabled=enabled)


def set_pet_scale(value: float | str) -> tuple[float, str | None]:
    """Set ``display.pet.scale`` (clamped). Returns ``(applied, error)``; *error* is set (and
    nothing written) only when *value* isn't a number. Single write path behind ``/pet scale``
    and the desktop slider."""
    from agent.pet.constants import clamp_scale

    try:
        scale = clamp_scale(float(value))
    except (TypeError, ValueError):
        return 0.0, f"not a number: {value!r} — try a value like 0.5"

    _update_pet_config(scale=scale)
    return scale, None


def toggle_pet_display() -> tuple[bool, str | None, str | None]:
    """Toggle ``display.pet.enabled``."""
    from agent.pet import store

    cfg = _pet_config()
    pet = store.resolve_active_pet(str(cfg.get("slug", "") or ""))

    if _pet_enabled(cfg):
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
    """Disable + unset the active pet iff it's ``slug`` (e.g. after removal)."""
    return _update_pet_config(when_slug=slug, slug="", enabled=False)


def _rename_active_if(old_slug: str, new_slug: str) -> bool:
    """Repoint the active pet from ``old_slug`` to ``new_slug`` iff it's active (a rename moves
    the dir; config must follow). Preserves ``enabled``."""
    if not new_slug or old_slug == new_slug:
        return False
    return _update_pet_config(when_slug=old_slug, slug=new_slug)


def _interactive_pick(pets) -> str:
    """Minimal numbered picker (avoids curses dep for a tiny list)."""
    print("Installed pets:")
    for i, pet in enumerate(pets, 1):
        print(f"  {i}. {pet.slug:<24} {pet.display_name}")
    try:
        idx = int(input("Select a pet [1]: ").strip() or "1") - 1
    except (EOFError, KeyboardInterrupt, ValueError):
        _err("✗ cancelled")
        return ""
    if 0 <= idx < len(pets):
        return pets[idx].slug
    _err("✗ invalid selection")
    return ""


# ── argparse wiring ───────────────────────────────────────────────────────

# (name, help, handler, [((flags...), add_argument kwargs), ...]) — registration order is menu order.
_SUBCOMMANDS = (
    ("list", "Browse the petdex gallery", _cmd_list, (
        (("query",), dict(nargs="?", default="", help="Filter by slug/name substring")),
        (("--installed",), dict(action="store_true", help="Only show installed pets")),
        (("--limit",), dict(type=int, default=40, help="Max rows (0 = all)")),
    )),
    ("install", "Install a pet from the gallery", _cmd_install, (
        (("slug",), dict(help="Pet slug (e.g. boba)")),
        (("--force",), dict(action="store_true", help="Re-download even if present")),
        (("--select",), dict(action="store_true", help="Make it the active pet")),
    )),
    ("select", "Set the active pet (writes display.pet.*)", _cmd_select, (
        (("slug",), dict(nargs="?", default="", help="Pet slug (omit for picker)")),
    )),
    ("show", "Animate the active pet in the terminal", _cmd_show, (
        (("slug",), dict(nargs="?", default="", help="Pet slug (default: active)")),
        (("--state",), dict(default="", help="Single state: idle/run/review/failed/wave/jump")),
        (("--cycle",), dict(action="store_true", help="Cycle through all states")),
        (("--once",), dict(action="store_true", help="Play once instead of looping")),
        (("--mode",), dict(default=None, help="Override render mode (kitty/iterm/sixel/unicode/auto)")),
        (("--scale",), dict(type=float, default=0, help="Override scale (0 = config)")),
    )),
    ("off", "Disable the pet display", _cmd_off, ()),
    ("scale", "Resize the pet everywhere (display.pet.scale)", _cmd_scale, (
        (("factor",), dict(help="Scale factor, e.g. 0.5 (clamped 0.1–3.0)")),
    )),
    ("remove", "Delete an installed pet", _cmd_remove, (
        (("slug",), dict(help="Pet slug")),
    )),
    ("doctor", "Check pet setup + terminal graphics support", _cmd_doctor, ()),
)


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach ``pets`` subcommands to *parent* (called by main.py)."""
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="pets_command")

    for name, help_text, func, arguments in _SUBCOMMANDS:
        sub = subs.add_parser(name, help=help_text)
        for flags, kwargs in arguments:
            sub.add_argument(*flags, **kwargs)
        sub.set_defaults(func=func)
