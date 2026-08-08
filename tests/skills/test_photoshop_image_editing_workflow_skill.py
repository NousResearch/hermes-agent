"""Contract tests for the optional Photoshop image-editing workflow skill."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "optional-skills" / "creative" / "photoshop-image-editing-workflow"
SKILL_PATH = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "open_in_photoshop.py"


def _frontmatter() -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---", SKILL_PATH.read_text(encoding="utf-8"), re.DOTALL)
    assert match, "SKILL.md must have YAML frontmatter"
    return dict(line.split(": ", 1) for line in match.group(1).splitlines() if ": " in line)


def _load_helper():
    spec = importlib.util.spec_from_file_location("photoshop_handoff", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_is_optional_and_frontmatter_is_gated() -> None:
    frontmatter = _frontmatter()
    assert SKILL_PATH.is_file()
    assert "optional-skills" in SKILL_PATH.parts
    assert frontmatter["description"].endswith(".")
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["platforms"] == "[linux]"
    assert frontmatter["author"].startswith("Manseong Lee (@aiebrain)")


def test_only_resolvable_related_skill_is_referenced() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "related_skills: [obsidian]" in text
    assert "reference-image-recreation-qa" not in text


def test_skill_documents_native_tools_and_clean_prerequisite_failure() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8")
    for tool in ("`search_files`", "`vision_analyze`", "`read_file`", "`write_file`", "`terminal`"):
        assert tool in text
    assert "clean prerequisite failure" in text
    assert "wslpath" not in text


def test_helper_rejects_non_wsl_before_handoff(tmp_path: Path) -> None:
    helper = _load_helper()
    input_path = tmp_path / "candidate.png"
    input_path.write_bytes(b"png")
    photoshop = tmp_path / "Photoshop.exe"
    photoshop.write_bytes(b"exe")

    with mock.patch.object(helper, "is_wsl", return_value=False), mock.patch.object(helper.shutil, "which", return_value="/usr/bin/powershell.exe"):
        result = helper.prerequisites(input_path, photoshop)

    assert result["ok"] is False
    assert "WSL is required" in result["missing"]


def test_helper_converts_windows_mounted_path() -> None:
    helper = _load_helper()
    with mock.patch.object(Path, "resolve", return_value=Path("/mnt/c/Users/example/final.png")):
        assert helper.wsl_to_windows_path(Path("ignored")) == r"C:\Users\example\final.png"
