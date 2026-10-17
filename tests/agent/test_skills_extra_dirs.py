"""``skills.extra_dirs`` (additional writable roots) and ``skills.excluded_dirs``."""

from agent import skill_utils
from agent.skill_utils import (
    get_all_skills_dirs,
    is_external_skill_path,
    iter_skill_index_files,
    normalize_skill_lookup_name,
)


def _skill(root, rel, name):
    d = root / rel
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name} skill\n---\nbody\n", encoding="utf-8")
    return d


def test_extra_dir_skill_is_found_writable_and_excluded_names_skipped(tmp_path, monkeypatch):
    home, extra = tmp_path / ".hermes", tmp_path / "shared-skills"
    home.mkdir()
    extra.mkdir()
    (home / "config.yaml").write_text(
        f"skills:\n  extra_dirs:\n    - {extra.as_posix()}\n  excluded_dirs:\n    - .inbox\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    skill_utils._external_dirs_cache_clear()
    demo = _skill(extra, "tools/demo", "demo")
    _skill(extra, ".inbox/pending", "pending")

    assert get_all_skills_dirs()[1] == extra.resolve()
    assert not is_external_skill_path(demo / "SKILL.md")
    assert [p.parent.name for p in iter_skill_index_files(extra.resolve(), "SKILL.md")] == ["demo"]
    assert normalize_skill_lookup_name(str(demo)) == "tools/demo"

    from tools.skills_tool import _find_all_skills
    names = {s["name"] for s in _find_all_skills(skip_disabled=True)}
    assert "demo" in names and "pending" not in names
