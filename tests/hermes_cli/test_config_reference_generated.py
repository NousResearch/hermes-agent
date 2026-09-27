"""The committed config-reference artifact is exactly what ``scripts/gen_config_reference.py``
renders, and its section coverage spans every known config source.

The config docs were hand-maintained prose and rotted (see #125185: three confirmed
numeric-default drifts); this ports the proven gateway-contracts codegen pattern — generate
from ``DEFAULT_CONFIG`` + the known-key lists + the chat-CLI overlay, fail on drift.
Regenerate with ``venv/bin/python scripts/gen_config_reference.py`` when a default changes.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GEN = REPO / "scripts" / "gen_config_reference.py"
ARTIFACT = REPO / "docs" / "reference" / "config-reference.generated.md"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_config_reference", GEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_artifact_is_current(gen):
    """The committed artifact equals an in-memory regeneration (byte-for-byte)."""
    stale = [path.relative_to(REPO) for path, text in gen.render_all().items()
             if (path.read_text(encoding="utf-8") if path.exists() else None) != text]
    assert not stale, f"stale generated config reference {stale}: run scripts/gen_config_reference.py"


def test_artifact_covers_every_known_section(gen):
    """Every section from the three config sources appears as a heading (plumbing contract)."""
    from hermes_cli.cli_config_load import _cli_config_defaults
    from hermes_cli.config import _EXTRA_KNOWN_ROOT_KEYS, _OPEN_SUBKEY_TOP_LEVEL_KEYS
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    headings = set(re.findall(r"^## `([a-z_][a-z0-9_]*)`$", ARTIFACT.read_text(encoding="utf-8"), re.M))
    sources = (
        (set(DEFAULT_CONFIG) - {"_config_version"})
        | set(_EXTRA_KNOWN_ROOT_KEYS)
        | set(_OPEN_SUBKEY_TOP_LEVEL_KEYS)
        | set(_cli_config_defaults())
    )
    missing = sources - headings
    assert not missing, f"generated artifact dropped known config sections: {sorted(missing)}"


@pytest.mark.parametrize(("section", "key"), [("agent", "max_turns"), ("delegation", "max_iterations")])
def test_cli_core_default_splits_are_recorded(gen, section, key):
    """Where the chat-CLI overlay disagrees with DEFAULT_CONFIG, the artifact records both sides.

    The overlay applies before config.yaml, so neither value is wrong — but a generator that
    silently picked one side would codify half the split (#125185).
    """
    from hermes_cli.cli_config_load import _cli_config_defaults
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    block = ARTIFACT.read_text(encoding="utf-8").split(f"## `{section}`\n", 1)[1].split("\n## ", 1)[0]
    m = re.search(rf"^\| `{key}` \| (.+?) \| (.*?) \|$", block, re.M)
    assert m, f"{section}.{key} missing from the generated artifact"
    _, note = m.groups()
    core = DEFAULT_CONFIG.get(section, {}).get(key) if isinstance(DEFAULT_CONFIG.get(section), dict) else None
    cli = _cli_config_defaults().get(section, {}).get(key)
    if core != cli:
        assert note.startswith("cli:"), f"{section}.{key}: core={core!r} vs cli={cli!r} not recorded"
    else:
        assert not note.startswith("cli:")
