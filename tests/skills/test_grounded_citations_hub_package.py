"""Hub packaging regressions for the grounded-citations bundled skill."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "research" / "grounded-citations"
)


def _referenced_paths() -> set[str]:
    from tools.skills_hub import _referenced_support_paths

    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    referenced = _referenced_support_paths(body)
    assert referenced is not None
    return referenced


def test_skill_md_references_all_hub_required_support_files() -> None:
    """Hub installs only download support files referenced from SKILL.md."""
    assert {
        "scripts/sources.py",
        "scripts/_hermes_home.py",
        "references/citation-formats.md",
        "references/grounding-rationale.md",
    }.issubset(_referenced_paths())


def test_hub_referenced_package_imports_from_profile_install(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A Hub-installed file subset includes every runtime dependency."""
    referenced = _referenced_paths()

    installed = tmp_path / "profile" / "skills" / "research" / "grounded-citations"
    installed.mkdir(parents=True)
    (installed / "SKILL.md").write_text(
        (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for rel_path in referenced:
        src = SKILL_DIR / rel_path
        dst = installed / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    home = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.syspath_prepend(str(installed / "scripts"))
    sys.modules.pop("_hermes_home", None)

    spec = importlib.util.spec_from_file_location(
        "installed_grounded_citations_sources",
        installed / "scripts" / "sources.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    ledger_path = module.resolve_ledger_path()
    assert ledger_path == home / "cache" / "citations" / "ledger.json"
    entries = module.add_sources(ledger_path, ["https://example.com"])
    assert entries[0]["id"] == 1
    assert (
        json.loads(ledger_path.read_text(encoding="utf-8"))["sources"][0]["url"]
        == "https://example.com"
    )
