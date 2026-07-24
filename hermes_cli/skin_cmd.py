"""``hermes skin`` — list, switch, and tweak skins from the CLI.

``set`` is the load-bearing verb: it changes ONE color of the ACTIVE skin **in
place**, so tweaking (say) the tool marker never disturbs the rest of the look —
background included. Editing the file bumps its mtime; the gateway's skin watcher
repaints every live surface within ~a second. A built-in skin (no file) is forked
into an editable copy that carries its full palette, so the current look is
preserved and only the one key changes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from hermes_constants import display_hermes_home, get_hermes_home

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _skins_dir() -> Path:
    return get_hermes_home() / "skins"


def _active_skin() -> str:
    from hermes_cli.config import load_config

    display = (load_config() or {}).get("display") or {}
    return str(display.get("skin") or "default")


def _use(name: str) -> None:
    """Activate a skin (persists display.skin via the shared config writer)."""
    from hermes_cli.config import config_command

    config_command(argparse.Namespace(config_command="set", key="display.skin", value=name, force=True))


def _skin_set(key: str, value: str, skin: str | None) -> int:
    import yaml

    if not _HEX_RE.match(value):
        print(f"✗ {value!r} is not a #rrggbb hex color", file=sys.stderr)
        return 1

    name = skin or _active_skin()
    path = _skins_dir() / f"{name}.yaml"

    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        target = name
    else:
        # Built-in (or missing): fork into an editable copy that keeps its full
        # palette, under a fresh name so the built-in stays intact for revert.
        from hermes_cli.skin_engine import load_skin

        resolved = load_skin(name)
        target = f"{name}-custom"
        path = _skins_dir() / f"{target}.yaml"
        data = {
            "name": target,
            "description": f"{name} + custom {key}",
            "colors": dict(resolved.colors),
            "branding": dict(resolved.branding),
            "tool_prefix": resolved.tool_prefix,
        }

    if not isinstance(data.get("colors"), dict):
        data["colors"] = {}
    data["colors"][key] = value
    data.setdefault("name", target)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if target != name:
        _use(target)

    print(f"✓ {key} = {value} in {display_hermes_home()}/skins/{target}.yaml (live within ~1s)")
    return 0


def _skin_list() -> int:
    from hermes_cli.skin_engine import list_skins

    active = _active_skin()
    for s in list_skins():
        mark = "*" if s["name"] == active else " "
        print(f"{mark} {s['name']:<16} {s.get('source', ''):<8} {s.get('description', '')}")
    return 0


def skin_command(args) -> None:
    """Dispatch ``hermes skin <verb>``."""
    verb = getattr(args, "skin_command", None)

    if verb == "set":
        sys.exit(_skin_set(args.key, args.value, getattr(args, "skin", None)))
    elif verb == "use":
        _use(args.name)
        print(f"✓ active skin → {args.name} (live within ~1s)")
    else:  # list / default
        sys.exit(_skin_list())
