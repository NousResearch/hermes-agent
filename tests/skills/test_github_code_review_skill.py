"""Runtime loading tests for the bundled github-code-review skill.

These exercise the production skill-loading/validation code paths in
``tools.skill_manager_tool`` against the shipped skill bundle — the same gates
a skill passes when Hermes creates, edits, and loads it — plus bundle
consistency (every ``references/`` file the SKILL.md points at must ship).
No source-shape assertions, no network; stdlib + pytest only.
"""

import re
import shutil
from pathlib import Path

from hermes_constants import get_hermes_home
from tools.skill_manager_tool import (
    _find_skill,
    _security_scan_skill,
    _validate_category,
    _validate_content_size,
    _validate_frontmatter,
    _validate_name,
)

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "github" / "github-code-review"
)
SKILL_MD = SKILL_DIR / "SKILL.md"


def _skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_loads_through_production_validation():
    """The skill must pass the same gates the skill manager applies on write/load."""
    content = _skill_text()
    assert _validate_name("github-code-review") is None
    assert _validate_category("github") is None
    assert _validate_frontmatter(content) is None
    assert _validate_content_size(content) is None


def test_description_fits_new_skill_prompt_budget():
    """Create-path gate: description must fit the system-prompt char budget."""
    assert _validate_frontmatter(_skill_text(), new_skill=True) is None


def test_skill_dir_passes_security_scan():
    assert _security_scan_skill(SKILL_DIR) is None


def test_referenced_bundle_files_exist():
    """Every references/ file the SKILL.md points at must ship in the bundle."""
    refs = set(re.findall(r"references/[A-Za-z0-9._-]+\.md", _skill_text()))
    assert refs, "expected the skill to reference at least one bundle file"
    for ref in sorted(refs):
        assert (SKILL_DIR / ref).is_file(), f"missing referenced file: {ref}"


def test_skill_discoverable_and_loadable_after_install():
    """Copied into a fresh HERMES_HOME, the skill is found by name through the
    production discovery path and its content loads clean."""
    import tools.skill_manager_tool as smt

    # conftest points HERMES_HOME at a per-test tempdir; install the bundle there.
    dest = get_hermes_home() / "skills" / "github" / "github-code-review"
    shutil.copytree(SKILL_DIR, dest)

    found = smt._find_skill("github-code-review")
    assert found is not None, "skill not discoverable after install"
    content = (found["path"] / "SKILL.md").read_text(encoding="utf-8")
    assert smt._validate_frontmatter(content) is None
    assert smt._validate_content_size(content) is None
