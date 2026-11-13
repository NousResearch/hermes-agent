"""Validate the porkbun optional skill's SKILL.md.

stdlib + pytest only, no live network. Mirrors the checks a reviewer applies
to a new optional skill: valid frontmatter, a <=60-char description, no
machine-local paths, no house-specific wiring, and the hard-won operational
lessons preserved.
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / "optional-skills" / "devops" / "porkbun" / "SKILL.md"


def _load():
    text = SKILL.read_text()
    assert text.startswith("---"), "frontmatter must start at byte 0 with ---"
    m = re.search(r"\n---\s*\n", text[3:])
    assert m, "frontmatter must close with \\n---\\n"
    fm = text[3:m.start() + 3]
    body = text[3 + m.end():]
    return text, fm, body


def _field(fm, key):
    m = re.search(rf"^{key}:\s*(.+)$", fm, re.MULTILINE)
    return m.group(1).strip() if m else None


def test_file_exists_and_frontmatter_valid():
    _, _, body = _load()
    assert body.strip(), "body must be non-empty after frontmatter"


def test_required_fields_present():
    _, fm, _ = _load()
    for key in ("name", "description", "version", "author", "license", "platforms"):
        assert _field(fm, key), f"missing frontmatter field: {key}"
    assert _field(fm, "name") == "porkbun"


def test_description_within_limits():
    _, fm, _ = _load()
    desc = _field(fm, "description")
    assert desc is not None
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline 60)"
    assert desc.endswith("."), "description must end with a period"
    assert ":" not in desc, "description must not contain an unquoted colon"


def test_no_machine_local_paths_or_house_wiring():
    text, _, _ = _load()
    for bad in ("/home/", "/Users/", "/root/", "~/.hermes", "bitshed", "thebitshed"):
        assert bad not in text, f"house-specific/path token leaked: {bad!r}"


def test_operational_gotchas_preserved():
    text, _, _ = _load()
    for needle in (
        "api.porkbun.com/api/json/v3",
        "LABEL ONLY",
        "credentialsValid",
        "curitiba.porkbun.com",
        "request/second",
        "`status`",
    ):
        assert needle in text, f"operational lesson missing: {needle!r}"
