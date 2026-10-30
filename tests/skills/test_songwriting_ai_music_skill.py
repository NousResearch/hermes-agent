"""Invariant tests for the bundled songwriting-and-ai-music skill.

Scope: link resolution for the skill's own reference files. Content quality is a
review question; a reference that is named in SKILL.md but missing on disk is a
broken load for the agent, and that is mechanical.
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "skills" / "creative" / "songwriting-and-ai-music"
SKILL_MD = SKILL_DIR / "SKILL.md"


def test_referenced_reference_files_exist():
    content = SKILL_MD.read_text(encoding="utf-8")
    referenced = set(re.findall(r"references/[A-Za-z0-9._-]+\.md", content))

    assert referenced, "SKILL.md no longer points at any references/ file"
    missing = sorted(ref for ref in referenced if not (SKILL_DIR / ref).is_file())
    assert not missing, f"SKILL.md points at missing reference files: {missing}"


def test_set_arcs_reference_keeps_its_how_to_use_shape():
    reference = SKILL_DIR / "references" / "set-arcs.md"
    text = reference.read_text(encoding="utf-8")

    assert "## 7. Verification" in text, "set-arcs.md lost its verification section"
    assert "[ ]" in text, "set-arcs.md verification list should stay checkable"


def test_skill_has_no_machine_local_paths():
    content = SKILL_MD.read_text(encoding="utf-8")
    assert not re.search(r"/home/[a-z0-9_-]+/", content)
