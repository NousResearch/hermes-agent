"""The desktop's offline slash block-list is a dump of COMMAND_REGISTRY, never hand-authored."""

import json
from pathlib import Path

from hermes_cli.commands import COMMAND_REGISTRY, desktop_surface_registry, resolve_command

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "apps" / "desktop" / "src" / "lib" / "desktop-slash-registry.json"


def test_committed_desktop_dump_matches_registry():
    """Editing a ``desktop=`` value or alias without re-running the dump script is a drift."""
    committed = json.loads(DUMP.read_text(encoding="utf-8"))
    assert committed == desktop_surface_registry(), (
        "apps/desktop/src/lib/desktop-slash-registry.json is stale — run "
        "scripts/dump_desktop_slash_registry.py"
    )


def test_desktop_surface_registry_covers_every_alias_with_its_canonical_value():
    registry = desktop_surface_registry()
    for cmd in COMMAND_REGISTRY:
        for key in (cmd.name, *cmd.aliases):
            assert registry.get(f"/{key}") == cmd.desktop, key
    for key in registry:
        assert resolve_command(key) is not None, key
