"""CLI subcommand: ``hermes companions <subcommand>``."""

from __future__ import annotations

import argparse
import sys

from utils import is_truthy_value


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _cmd_list(args) -> int:
    """List gallery companions (or only installed ones with ``--installed``)."""
    from agent.companion import store

    if args.installed:
        companions = store.installed_companions()
        if not companions:
            print("No companions installed. Try: hermes companions install boba")
            return 0
        print(f"Installed companions ({len(companions)}):")
        for companion in companions:
            print(f"  {companion.slug:<24} {companion.display_name}")
        return 0

    from agent.companion.manifest import ManifestError, fetch_manifest

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
    installed = {p.slug for p in store.installed_companions()}

    print(f"companiondex gallery — {len(entries)} companion(s){' matching ' + repr(query) if query else ''}:")
    for entry in shown:
        mark = "✓" if entry.slug in installed else " "
        print(f"  {mark} {entry.slug:<28} {entry.display_name}  ({entry.kind})")
    if limit and len(entries) > limit:
        print(f"  … {len(entries) - limit} more (use --limit 0 or --query to filter)")
    print("\nInstall one with: hermes companions install <slug>")
    return 0


def _cmd_install(args) -> int:
    from agent.companion import store
    from agent.companion.manifest import ManifestError

    slug = args.slug.strip()
    try:
        companion = store.install_companion(slug, force=args.force)
    except (store.CompanionStoreError, ManifestError) as exc:
        _err(f"✗ install failed: {exc}")
        return 1

    print(f"✓ installed {companion.display_name} → {companion.directory}")

    if args.select or not _has_active_companion():
        _set_active(slug)
        print(f"✓ {companion.display_name} is now the active companion (display.companion.slug={slug}, enabled)")
    else:
        print(f"  Make it active with: hermes companions select {slug}")
    return 0


def _cmd_remove(args) -> int:
    from agent.companion import store

    slug = args.slug.strip()
    if store.remove_companion(slug):
        print(f"✓ removed {slug}")
        return 0
    _err(f"✗ '{slug}' is not installed")
    return 1


def _cmd_select(args) -> int:
    from agent.companion import store

    slug = (args.slug or "").strip()
    if not slug:
        companions = store.installed_companions()
        if not companions:
            _err("✗ no companions installed — run: hermes companions install boba")
            return 1
        slug = _interactive_pick(companions)
        if not slug:
            return 1

    companion = store.load_companion(slug)
    if companion is None or not companion.exists:
        _err(f"✗ '{slug}' is not installed — run: hermes companions install {slug}")
        return 1

    _set_active(slug)
    print(f"✓ active companion set to {companion.display_name} (display.companion.slug={slug}, enabled)")
    return 0


def _cmd_off(args) -> int:
    _set_enabled(False)
    print("✓ companion disabled (display.companion.enabled=false)")
    return 0


def _cmd_scale(args) -> int:
    """Persist ``display.companion.scale`` — one knob resizes every surface."""
    scale, err = set_companion_scale(args.factor)
    if err:
        _err(f"✗ {err}")
        return 1
    print(f"✓ companion scale set to {scale:g} (display.companion.scale)")
    return 0


def _cmd_show(args) -> int:
    """Animate the active (or named) companion in the terminal via the shared CompanionRenderer
    (kitty/iTerm2/sixel when supported, else truecolor half-block fallback). Ctrl+C stops."""
    import shutil
    import time

    from agent.companion import store
    from agent.companion.constants import DEFAULT_SCALE, LOOP_MS, STATE_ROWS, CompanionState, resolve_cols
    from agent.companion.render import build_renderer

    cfg = _companion_config()
    slug = (args.slug or "").strip() or str(cfg.get("slug", "") or "")
    companion = store.resolve_active_companion(slug)
    if companion is None:
        _err("✗ no companion to show — run: hermes companions install boba")
        return 1

    mode_cfg = args.mode or str(cfg.get("render_mode", "auto") or "auto")
    scale = float(args.scale or cfg.get("scale", DEFAULT_SCALE) or DEFAULT_SCALE)
    cols = resolve_cols(scale, cfg.get("unicode_cols", 0))

    renderer = build_renderer(companion.spritesheet, configured_mode=mode_cfg, scale=scale, unicode_cols=cols)
    if not renderer.available:
        _err(f"✗ cannot render here (no TTY / graphics disabled). Effective mode: {renderer.mode}.")
        return 1

    # Which states to play: one named state, or cycle the driveable rows.
    requested = (args.state or "").strip().lower()
    if requested:
        states = [requested]
    elif args.cycle:
        states = [s for s in STATE_ROWS if s in {e.value for e in CompanionState}]
    else:
        states = [CompanionState.IDLE.value]

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
        print(f"{companion.display_name} — mode={renderer.mode}  (Ctrl+C to stop)")
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
                        out.write(f"{companion.display_name} [{state}]\n")
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
    """Report install state, active companion, config, and terminal capability."""
    from agent.companion import store
    from agent.companion.render import detect_terminal_graphics, resolve_mode

    cfg = _companion_config()
    enabled = _companion_enabled(cfg)
    configured_slug = str(cfg.get("slug", "") or "")
    mode_cfg = str(cfg.get("render_mode", "auto") or "auto")

    companions = store.installed_companions()
    active = store.resolve_active_companion(configured_slug)

    print("companiondex doctor")
    print(f"  companions dir:        {store.companions_dir()}")
    print(f"  installed:       {len(companions)} ({', '.join(p.slug for p in companions) or 'none'})")
    print(f"  display.companion.enabled:     {enabled}")
    print(f"  display.companion.slug:        {configured_slug or '(unset)'}")
    print(f"  active (resolved):       {active.slug if active else '(none)'}")
    print(f"  display.companion.render_mode: {mode_cfg}")
    print(f"  detected graphics:       {detect_terminal_graphics()}")
    print(f"  effective mode (TTY):    {resolve_mode(mode_cfg)}")

    ok = True
    if not companions:
        print("  → no companions installed. Run: hermes companions install boba")
        ok = False
    elif active is None:
        print("  → active companion unresolved. Run: hermes companions select <slug>")
        ok = False
    elif not enabled:
        print("  → companion display is disabled. Run: hermes companions select " + active.slug)

    try:
        import PIL
    except ImportError:
        print("  ✗ Pillow not importable — sprite decoding will be unavailable")
        ok = False

    print("  ✓ ready" if ok and enabled else "  (run the suggestions above to finish setup)")
    return 0


# ── config helpers ────────────────────────────────────────────────────────

def _companion_config() -> dict:
    from hermes_cli.config import load_config

    cfg = load_config()
    display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
    companion = display.get("companion", {})
    return companion if isinstance(companion, dict) else {}


def _has_active_companion() -> bool:
    cfg = _companion_config()
    return _companion_enabled(cfg) and bool(cfg.get("slug"))


def _companion_enabled(cfg: dict) -> bool:
    return is_truthy_value(cfg.get("enabled"), default=False)


def _update_companion_config(when_slug: str | None = None, **values) -> bool:
    """Write ``display.companion.*`` keys and save config; returns whether anything was written.

    With ``when_slug`` the write only happens when the configured slug equals it (remove/rename
    must never disturb a different active companion).
    """
    from hermes_cli.config import load_config, save_config

    cfg = load_config()
    companion = cfg.setdefault("display", {}).setdefault("companion", {})
    if when_slug is not None and (not isinstance(companion, dict) or str(companion.get("slug", "") or "") != when_slug):
        return False
    companion.update(values)
    save_config(cfg)
    return True


def _set_active(slug: str) -> None:
    _update_companion_config(slug=slug, enabled=True)


def _set_enabled(enabled: bool) -> None:
    _update_companion_config(enabled=enabled)


def set_companion_scale(value: float | str) -> tuple[float, str | None]:
    """Set ``display.companion.scale`` (clamped). Returns ``(applied, error)``; *error* is set (and
    nothing written) only when *value* isn't a number. Single write path behind ``/companion scale``
    and the desktop slider."""
    from agent.companion.constants import clamp_scale

    try:
        scale = clamp_scale(float(value))
    except (TypeError, ValueError):
        return 0.0, f"not a number: {value!r} — try a value like 0.5"

    _update_companion_config(scale=scale)
    return scale, None


def toggle_companion_display() -> tuple[bool, str | None, str | None]:
    """Toggle ``display.companion.enabled``."""
    from agent.companion import store

    cfg = _companion_config()
    companion = store.resolve_active_companion(str(cfg.get("slug", "") or ""))

    if _companion_enabled(cfg):
        _set_enabled(False)
        return False, companion.display_name if companion else None, None

    if companion is None:
        installed = store.installed_companions()
        if not installed:
            return False, None, "no companions installed — /companion list to browse, or /companion <slug> to adopt"
        companion = installed[0]
        _set_active(companion.slug)
    else:
        _set_enabled(True)
    return True, companion.display_name, None


def print_companion_gallery(*, limit: int = 20) -> None:
    """Print a slice of the public companiondex gallery (CLI/TUI text fallback)."""
    from agent.companion import store
    from agent.companion.manifest import ManifestError, fetch_manifest

    try:
        entries = fetch_manifest()
    except ManifestError as exc:
        print(f"(._.) Couldn't reach the companiondex gallery: {exc}")
        return

    installed = {p.slug for p in store.installed_companions()}
    shown = entries[:limit] if limit > 0 else entries
    print(f"(^o^)/ companiondex gallery — first {len(shown)} of {len(entries)}:")
    for entry in shown:
        mark = "●" if entry.slug in installed else "○"
        print(f"  {mark} {entry.slug:<24} {entry.display_name}")
    print("  /companion <slug> to adopt · /companion to toggle")


def _clear_active_if(slug: str) -> bool:
    """Disable + unset the active companion iff it's ``slug`` (e.g. after removal)."""
    return _update_companion_config(when_slug=slug, slug="", enabled=False)


def _rename_active_if(old_slug: str, new_slug: str) -> bool:
    """Repoint the active companion from ``old_slug`` to ``new_slug`` iff it's active (a rename moves
    the dir; config must follow). Preserves ``enabled``."""
    if not new_slug or old_slug == new_slug:
        return False
    return _update_companion_config(when_slug=old_slug, slug=new_slug)


def _interactive_pick(companions) -> str:
    """Minimal numbered picker (avoids curses dep for a tiny list)."""
    print("Installed companions:")
    for i, companion in enumerate(companions, 1):
        print(f"  {i}. {companion.slug:<24} {companion.display_name}")
    try:
        idx = int(input("Select a companion [1]: ").strip() or "1") - 1
    except (EOFError, KeyboardInterrupt, ValueError):
        _err("✗ cancelled")
        return ""
    if 0 <= idx < len(companions):
        return companions[idx].slug
    _err("✗ invalid selection")
    return ""


# ── argparse wiring ───────────────────────────────────────────────────────

# (name, help, handler, [((flags...), add_argument kwargs), ...]) — registration order is menu order.
_SUBCOMMANDS = (
    ("list", "Browse the companiondex gallery", _cmd_list, (
        (("query",), dict(nargs="?", default="", help="Filter by slug/name substring")),
        (("--installed",), dict(action="store_true", help="Only show installed companions")),
        (("--limit",), dict(type=int, default=40, help="Max rows (0 = all)")),
    )),
    ("install", "Install a companion from the gallery", _cmd_install, (
        (("slug",), dict(help="Companion slug (e.g. boba)")),
        (("--force",), dict(action="store_true", help="Re-download even if present")),
        (("--select",), dict(action="store_true", help="Make it the active companion")),
    )),
    ("select", "Set the active companion (writes display.companion.*)", _cmd_select, (
        (("slug",), dict(nargs="?", default="", help="Companion slug (omit for picker)")),
    )),
    ("show", "Animate the active companion in the terminal", _cmd_show, (
        (("slug",), dict(nargs="?", default="", help="Companion slug (default: active)")),
        (("--state",), dict(default="", help="Single state: idle/run/review/failed/wave/jump")),
        (("--cycle",), dict(action="store_true", help="Cycle through all states")),
        (("--once",), dict(action="store_true", help="Play once instead of looping")),
        (("--mode",), dict(default=None, help="Override render mode (kitty/iterm/sixel/unicode/auto)")),
        (("--scale",), dict(type=float, default=0, help="Override scale (0 = config)")),
    )),
    ("off", "Disable the companion display", _cmd_off, ()),
    ("scale", "Resize the companion everywhere (display.companion.scale)", _cmd_scale, (
        (("factor",), dict(help="Scale factor, e.g. 0.5 (clamped 0.1–3.0)")),
    )),
    ("remove", "Delete an installed companion", _cmd_remove, (
        (("slug",), dict(help="Companion slug")),
    )),
    ("doctor", "Check companion setup + terminal graphics support", _cmd_doctor, ()),
)


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach ``companions`` subcommands to *parent* (called by main.py)."""
    parent.set_defaults(func=lambda a: (parent.print_help(), 0)[1])
    subs = parent.add_subparsers(dest="companions_command")

    for name, help_text, func, arguments in _SUBCOMMANDS:
        sub = subs.add_parser(name, help=help_text)
        for flags, kwargs in arguments:
            sub.add_argument(*flags, **kwargs)
        sub.set_defaults(func=func)
