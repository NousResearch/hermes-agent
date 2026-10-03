"""Named adoption previews must share real adoption's refusal policy."""
import json

import pytest

from hermes_cli.curator import cli_main
from tools import skill_usage


@pytest.fixture
def skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    root = home / "skills"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("local", "hub", "bundled", "protected"):
        target = root / name
        target.mkdir()
        (target / "SKILL.md").write_text(f"---\nname: {name}\n---\nFixture.\n", encoding="utf-8")
    (root / ".hub").mkdir()
    (root / ".hub" / "lock.json").write_text(json.dumps({"installed": {"hub": {}}}), encoding="utf-8")
    (root / ".bundled_manifest").write_text("bundled\n", encoding="utf-8")
    monkeypatch.setattr(skill_usage, "PROTECTED_BUILTIN_SKILLS", {"protected"})
    external = home / "external"
    (external / "ext").mkdir(parents=True)
    (external / "ext" / "SKILL.md").write_text("---\nname: ext\n---\nFixture.\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "skills:\n  external_dirs:\n    - " + json.dumps(str(external)) + "\n", encoding="utf-8"
    )
    return root


@pytest.mark.parametrize("name", ["hub", "bundled", "protected", "missing", "ext"])
def test_preview_matches_real_refusal_without_writes(skills, capsys, name):
    before = {str(p.relative_to(skills)): p.read_bytes() for p in skills.rglob("*") if p.is_file()}
    assert cli_main(["adopt", name]) == 1
    real = capsys.readouterr().out
    assert cli_main(["adopt", name, "--dry-run"]) == 1
    preview = capsys.readouterr().out
    assert real.strip() in preview
    assert f"  + {name}" not in preview
    assert {str(p.relative_to(skills)): p.read_bytes() for p in skills.rglob("*") if p.is_file()} == before


def test_mixed_preview_only_lists_eligible_skills(skills, capsys):
    assert cli_main(["adopt", "hub", "local", "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "would adopt 1 skill(s)" in out
    assert "  + local" in out
    assert "  + hub" not in out
    assert not skill_usage.is_curator_managed("local")


def test_local_preview_preserves_real_adoption(skills, capsys):
    assert cli_main(["adopt", "local", "--dry-run"]) == 0
    assert "  + local" in capsys.readouterr().out
    assert not skill_usage.is_curator_managed("local")
    assert cli_main(["adopt", "local"]) == 0
    assert skill_usage.is_curator_managed("local")


def test_empty_bulk_preview(skills, capsys):
    assert cli_main(["adopt", "local"]) == 0
    capsys.readouterr()
    assert cli_main(["adopt", "--all-unmanaged", "--dry-run"]) == 0
    assert "no unmanaged skills" in capsys.readouterr().out
