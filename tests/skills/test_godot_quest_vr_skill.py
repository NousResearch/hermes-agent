"""Hermetic checks for optional-skills/software-development/godot-quest-vr."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

SKILL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "godot-quest-vr"
)
SKILL_MD = SKILL_ROOT / "SKILL.md"
FIX_SCRIPT = SKILL_ROOT / "scripts" / "fix-godot-mcp-protocol.sh"


def _frontmatter(text: str) -> dict:
    assert text.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"\n---\s*\n", text[3:])
    assert m, "closing frontmatter fence missing"
    fm = yaml.safe_load(text[3 : m.start() + 3])
    assert isinstance(fm, dict)
    return fm


def test_skill_md_exists():
    assert SKILL_MD.is_file()


def test_frontmatter_hardline():
    text = SKILL_MD.read_text(encoding="utf-8")
    fm = _frontmatter(text)
    assert fm.get("name") == "godot-quest-vr"
    desc = fm.get("description") or ""
    assert isinstance(desc, str)
    assert len(desc) <= 60, len(desc)
    assert desc.endswith(".")
    assert "platforms" in fm
    platforms = fm["platforms"]
    assert "linux" in platforms
    author = str(fm.get("author", ""))
    assert "buckster123" in author or "Andre" in author
    assert "Hermes Agent" in author


def test_no_bare_dollar_home_in_mcp_examples():
    text = SKILL_MD.read_text(encoding="utf-8")
    # Allow ${HOME} only — bare $HOME in yaml args is a known Hermes footgun
    for i, line in enumerate(text.splitlines(), 1):
        if "$HOME" in line and "${HOME}" not in line:
            # allow prose like "not bare $HOME"
            if "bare $HOME" in line or "not" in line.lower():
                continue
            if "args:" in line or "godot-mcp" in line:
                pytest.fail(f"bare $HOME at line {i}: {line}")


def test_defers_blender_to_optional_skill():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "blender-mcp" in text
    assert "hermes mcp install blender" in text
    assert "uvx" not in text or "blender-mcp" in text  # no parallel uvx setup block required


def test_references_present():
    refs = SKILL_ROOT / "references"
    expected = {
        "android-toolchain.md",
        "export-pipeline.md",
        "manifest.md",
        "glb-pipeline.md",
        "xr-basics.md",
        "godot-mcp.md",
    }
    found = {p.name for p in refs.glob("*.md")}
    assert expected <= found


def test_fix_protocol_script_idempotent():
    assert FIX_SCRIPT.is_file()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        target = root / "dist" / "utils" / "godot_connection.js"
        target.parent.mkdir(parents=True)
        target.write_text(
            "const x = {\n  protocol: 'json',\n  foo: 1,\n};\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["bash", str(FIX_SCRIPT), str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        out = target.read_text(encoding="utf-8")
        assert "protocol" not in out
        assert "foo: 1" in out
        # second run is no-op success
        r2 = subprocess.run(
            ["bash", str(FIX_SCRIPT), str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "ok:" in r2.stdout or "already" in r2.stdout.lower() or "patched" in r2.stdout.lower()


def test_fix_protocol_script_missing_target_fails():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            ["bash", str(FIX_SCRIPT), td],
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0
