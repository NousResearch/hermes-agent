"""The desktop's offline slash block-list is a dump of COMMAND_REGISTRY, never hand-authored."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from hermes_cli.commands import COMMAND_REGISTRY, desktop_surface_registry, resolve_command

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "apps" / "desktop" / "src" / "lib" / "desktop-slash-registry.json"
DUMP_SCRIPT = ROOT / "scripts" / "dump_desktop_slash_registry.py"


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
            assert registry.get(f"/{key}") == (cmd.desktop or None), key
            assert f"/{key}" in registry, key
    for key in registry:
        assert resolve_command(key) is not None, key


def test_offered_built_ins_are_present_as_null_rows():
    # #116159: /context (alias /ctx) has no desktop disposition — offline the desktop
    # still needs its name to keep it out of the Skills group and on the built-in path.
    registry = desktop_surface_registry()
    assert "/context" in registry and registry["/context"] is None
    assert "/ctx" in registry and registry["/ctx"] is None


def test_dump_script_check_mode_is_green_on_the_committed_json_and_red_on_drift(tmp_path):
    """``--check`` is the shell-usable twin of the equality test: 0 on the committed file, 1 on a stale copy."""
    proc = subprocess.run(
        [sys.executable, str(DUMP_SCRIPT), "--check"], cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr

    # Negative case in-process against a hand-edited copy: flipping one value must be caught.
    spec = importlib.util.spec_from_file_location("dump_desktop_slash_registry", DUMP_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stale = tmp_path / "desktop-slash-registry.json"
    payload = json.loads(DUMP.read_text(encoding="utf-8"))
    payload[next(iter(payload))] = {"desktop": "hand-edited"}
    stale.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert mod.check(stale) == 1
    assert mod.check(DUMP) == 0
