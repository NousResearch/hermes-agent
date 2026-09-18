"""Contract tests for the optional obliteratus skill.

Asserts authoring shape and the AGPL/CLI/local-path invariants that keep
Hermes (MIT) from importing the AGPL package. No live network, no GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "mlops" / "obliteratus"
SKILL_MD = SKILL_DIR / "SKILL.md"
REQUIRED_HEADINGS = (
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
)
CLI_METHODS = (
    "basic",
    "advanced",
    "aggressive",
    "spectral_cascade",
    "informed",
    "surgical",
    "optimized",
    "som",
    "inverted",
    "nuclear",
)


def _frontmatter_and_body() -> tuple[dict, str]:
    src = SKILL_MD.read_text(encoding="utf-8")
    assert src.startswith("---"), "SKILL.md must start with ---"
    m = re.search(r"^---\n(.*)\n---\n", src, re.S)
    assert m, "unclosed frontmatter"
    fm = yaml.safe_load(m.group(1))
    assert isinstance(fm, dict)
    return fm, src[m.end() :]


def test_skill_files_present() -> None:
    assert SKILL_MD.is_file()
    assert (SKILL_DIR / "references" / "local-models.md").is_file()
    assert (SKILL_DIR / "references" / "methods-guide.md").is_file()
    assert (SKILL_DIR / "references" / "analysis-modules.md").is_file()
    assert (SKILL_DIR / "templates" / "abliteration-config.yaml").is_file()


def test_description_hardline() -> None:
    fm, _ = _frontmatter_and_body()
    desc = str(fm["description"])
    assert len(desc) <= 60, f"description is {len(desc)} chars: {desc!r}"
    assert desc.endswith(".")
    assert re.search(
        r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
        desc,
        re.I,
    ) is None


def test_required_frontmatter_fields() -> None:
    fm, _ = _frontmatter_and_body()
    assert fm["name"] == "obliteratus"
    for field in ("version", "author", "license", "platforms"):
        assert fm.get(field), f"missing frontmatter field: {field}"
    author = str(fm["author"])
    assert "Hermes Agent" in author
    assert author.strip() != "Hermes Agent"
    hermes = (fm.get("metadata") or {}).get("hermes") or {}
    assert hermes.get("tags")
    related = hermes.get("related_skills") or []
    names = {p.parent.name for p in REPO.glob("skills/**/SKILL.md")} | {
        p.parent.name for p in REPO.glob("optional-skills/**/SKILL.md")
    }
    dangling = [r for r in related if r not in names]
    assert dangling == [], f"dangling related_skills: {dangling}"


def test_modern_sections_present() -> None:
    _, body = _frontmatter_and_body()
    for heading in REQUIRED_HEADINGS:
        assert heading in body, f"missing {heading}"


def test_cli_not_python_import_as_howto() -> None:
    _, body = _frontmatter_and_body()
    how = body.split("## How to Run", 1)[1].split("## Quick Reference", 1)[0]
    assert "obliteratus obliterate" in how
    assert "from obliteratus" not in how
    assert "import obliteratus" not in how
    assert "`terminal`" in body
    assert "AGPL" in body


def test_local_path_and_quantization_flags() -> None:
    src = SKILL_MD.read_text(encoding="utf-8")
    assert "local directory" in src.lower() or "local directories" in src.lower()
    assert "--quantization 4bit" in src
    assert "--output-dir" in src
    for method in CLI_METHODS:
        assert method in src, f"missing CLI method {method!r}"
    assert "--quantization bitsandbytes-4bit" not in src
    assert "qwen-hybrid" in src
    assert "self-improve" in src


def test_no_machine_local_paths() -> None:
    src = SKILL_MD.read_text(encoding="utf-8")
    for ref in (SKILL_DIR / "references").glob("*.md"):
        src += "\n" + ref.read_text(encoding="utf-8")
    m = re.search(r"/home/(?!runner\b)[a-z0-9_-]+/", src)
    assert m is None, f"machine-local path {m.group(0)!r}"
