"""``disable-model-invocation`` (Claude Code / Agent Skills frontmatter key).

``disable-model-invocation: true`` removes a skill from what the model is offered (prompt index, skills_list,
the not-found hint) while every explicit load keeps working. It does not change name resolution.
"""

import json
from unittest.mock import patch

import pytest

from agent.skill_commands import scan_skill_commands
from agent.skill_utils import TIER_EXTERNAL, skill_model_invocable
from tools.skills_tool import _find_all_skills, skill_view, skills_list


def _write_skill(root, name, extra=""):
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n{extra}---\n\n# {name}\n\nStep 1.\n",
        encoding="utf-8",
    )
    return skill_dir


@pytest.mark.parametrize("frontmatter, expected", [
    ({}, True),
    ({"disable-model-invocation": True}, False),
    ({"disable-model-invocation": False}, True),
    ({"disable-model-invocation": "true"}, False),
    ({"disable-model-invocation": " False "}, True),
    ({"disable-model-invocation": "yes"}, True),  # not a boolean: the default applies
])
def test_model_invocable_parsing(frontmatter, expected):
    assert skill_model_invocable(frontmatter) is expected


@pytest.fixture
def prompt_builder(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(pb, "get_disabled_skill_names", lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield pb
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_hidden_skill_stays_out_of_the_prompt_index_on_scan_and_snapshot(tmp_path, prompt_builder):
    pb = prompt_builder
    skills = tmp_path / "skills"
    _write_skill(skills, "wrapper")
    _write_skill(skills, "helper", "disable-model-invocation: true\n")
    _write_skill(skills, "quoted", 'disable-model-invocation: "true"\n')

    def build():
        return pb._build_skills_system_prompt_inner(skills, [], None, None, None)

    cold = build()
    assert "wrapper: Description for wrapper." in cold
    assert "helper" not in cold and "quoted" not in cold
    snapshot = pb._load_skills_snapshot(skills)
    assert snapshot is not None
    assert {e["skill_name"]: e["model_invocable"] for e in snapshot["skills"]} == {
        "wrapper": True, "helper": False, "quoted": False}
    pb.clear_skills_system_prompt_cache()  # in-process cache only: the next build uses the disk snapshot
    assert build() == cold


def test_snapshot_from_an_older_version_is_rebuilt(tmp_path, prompt_builder):
    pb = prompt_builder
    skills = tmp_path / "skills"
    _write_skill(skills, "helper", "disable-model-invocation: true\n")
    pb._build_skills_system_prompt_inner(skills, [], None, None, None)
    snapshot_path = pb._skills_prompt_snapshot_path()
    stale = json.loads(snapshot_path.read_text(encoding="utf-8"))
    stale["version"] = 4
    for entry in stale["skills"]:
        entry.pop("model_invocable")
    snapshot_path.write_text(json.dumps(stale), encoding="utf-8")
    pb.clear_skills_system_prompt_cache()

    assert "helper" not in pb._build_skills_system_prompt_inner(skills, [], None, None, None)


def test_hidden_external_skill_is_not_indexed(tmp_path, prompt_builder):
    pb = prompt_builder
    skills, external = tmp_path / "skills", tmp_path / "external"
    skills.mkdir()
    _write_skill(external, "ext-helper", "disable-model-invocation: true\n")
    _write_skill(external, "ext-visible")

    index = pb._build_skills_system_prompt_inner(skills, [(TIER_EXTERNAL, external)], None, None, None)
    assert "ext-visible" in index
    assert "ext-helper" not in index


def test_hidden_skill_still_owns_its_name(tmp_path, prompt_builder):
    """Precedence is unchanged: a hidden local skill shadows a same-named external one."""
    pb = prompt_builder
    skills, external = tmp_path / "skills", tmp_path / "external"
    _write_skill(skills, "deploy", "disable-model-invocation: true\n")
    _write_skill(external, "deploy")

    assert "deploy" not in pb._build_skills_system_prompt_inner(
        skills, [(TIER_EXTERNAL, external)], None, None, None)


def test_skills_list_and_not_found_hint_omit_hidden_skills_but_skill_view_loads_them(tmp_path):
    _write_skill(tmp_path, "wrapper")
    _write_skill(tmp_path, "helper", "disable-model-invocation: true\n")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        listed = json.loads(skills_list())
        missing = json.loads(skill_view("no-such-skill"))
        loaded = json.loads(skill_view("helper"))
        everything = {s["name"] for s in _find_all_skills()}

    assert [s["name"] for s in listed["skills"]] == ["wrapper"]
    assert missing["available_skills"] == ["wrapper"]
    assert loaded["success"] is True and "Step 1." in loaded["content"]
    # User-facing listings (banner, /skills, API) keep showing it.
    assert everything == {"wrapper", "helper"}


def test_hidden_skill_keeps_its_slash_command(tmp_path):
    _write_skill(tmp_path, "user-only", "disable-model-invocation: true\n")

    with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
        commands = scan_skill_commands()

    assert "/user-only" in commands
