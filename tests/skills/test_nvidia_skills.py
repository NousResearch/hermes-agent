from __future__ import annotations

import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = ROOT / "optional-skills" / "mlops" / "nvidia"


def _load_frontmatter(skill_path: str) -> dict:
    skill_file = SKILLS_DIR / skill_path / "SKILL.md"
    assert skill_file.exists(), f"Missing SKILL.md for {skill_path}"
    text = skill_file.read_text(encoding="utf-8")
    assert text.startswith("---"), f"SKILL.md for {skill_path} must start with ---"
    parts = text.split("---", 2)
    assert len(parts) >= 2, f"SKILL.md for {skill_path} has malformed frontmatter"
    return yaml.safe_load(parts[1])


def test_cudaq_guide_frontmatter_is_valid():
    fm = _load_frontmatter("cudaq-guide")
    assert fm["name"] == "nvidia-cudaq-guide"
    assert fm["description"].endswith(".")
    assert len(fm["description"]) <= 60
    assert "metadata" in fm
    assert "hermes" in fm["metadata"]
    assert "tags" in fm["metadata"]["hermes"]
    assert "category" in fm["metadata"]["hermes"]
    assert fm["metadata"]["hermes"]["category"] == "mlops"


def test_cudaq_guide_no_stray_yaml_after_frontmatter():
    skill_file = SKILLS_DIR / "cudaq-guide" / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) >= 2
    body = parts[2] if len(parts) > 2 else ""
    lines = body.strip().splitlines()
    assert lines, "Body must not be empty"
    assert not lines[0].startswith("- "), "No stray YAML list items after frontmatter"
    assert "domain:" not in body.split("---")[0] or "---" in body.split("domain:")[0], \
        "No stray 'domain:' YAML key after frontmatter close"


def test_cudaq_guide_command_matches_name():
    fm = _load_frontmatter("cudaq-guide")
    name = fm["name"]
    skill_file = SKILLS_DIR / "cudaq-guide" / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert f"/{name}" in text, \
        f"Skill name '{name}' must be used in command references (not /cudaq-guide)"
    assert "/cudaq-guide " not in text, \
        "Old command /cudaq-guide should not appear; use /nvidia-cudaq-guide"
    assert "$ARGUMENTS" not in text, \
        "$ARGUMENTS is not substituted by Hermes; remove it"


def test_cuopt_name_matches_directory():
    fm = _load_frontmatter("cuopt-numerical-optimization-api-python")
    assert fm["name"] == "nvidia-cuopt-numerical-optimization-api-python", \
        "Skill name must match directory slug for consistent slash-command registration"


def test_rag_blueprint_uses_native_tools():
    skill_file = SKILLS_DIR / "rag-blueprint" / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert "grep -" not in text or "terminal" in text, \
        "Skill guidance must reference native Hermes tools (terminal, read_file, search_files) instead of shell utilities"
    assert "docker ps" not in text or "terminal" in text, \
        "Use terminal tool for docker commands"
    assert "ps aux" not in text or "terminal" in text, \
        "Use terminal tool for process listing"


def test_cudaq_guide_platforms_declared():
    fm = _load_frontmatter("cudaq-guide")
    assert "platforms" in fm
    assert isinstance(fm["platforms"], list)
    assert len(fm["platforms"]) > 0


def test_cuopt_platforms_declared():
    fm = _load_frontmatter("cuopt-numerical-optimization-api-python")
    assert "platforms" in fm
    assert isinstance(fm["platforms"], list)


def test_rag_platforms_declared():
    fm = _load_frontmatter("rag-blueprint")
    assert "platforms" in fm
    assert isinstance(fm["platforms"], list)
