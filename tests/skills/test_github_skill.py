"""Security scan contracts for the bundled GitHub skill."""
from pathlib import Path

import pytest

from tools.threat_patterns import scan_for_threats

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "software-development"
    / "github"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"


# Other bundled skills have existing findings; keep this gate scoped to GitHub.
@pytest.mark.parametrize(
    "path",
    [SKILL_PATH]
    + sorted(path for path in (SKILL_DIR / "references").rglob("*") if path.is_file()),
    ids=lambda path: str(path.relative_to(SKILL_DIR)),
)
def test_skill_documents_pass_context_threat_scan(path):
    assert scan_for_threats(path.read_text(encoding="utf-8"), scope="context") == []

