"""Contract tests for the generated GBrain runtime skill artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_gbrain_skill_contract import (
    ContractError,
    build_contract,
    main,
    render_manifest_json,
)


def write_skill(
    path: Path,
    *,
    name: object = "example",
    description: object = "Example capability.",
    triggers: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---", f"name: {name}"]
    if isinstance(description, str) and "\n" in description:
        lines.extend([
            "description: |",
            *[f"  {line}" for line in description.splitlines()],
        ])
    else:
        lines.append(f"description: {description}")
    if triggers is not None:
        lines.append("triggers:")
        if isinstance(triggers, list):
            lines.extend(f"  - {trigger}" for trigger in triggers)
        else:
            lines.append(f"  value: {triggers}")
    lines.extend(["---", "", "# Example", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def test_contract_lists_nested_skills_deterministically(tmp_path):
    skills = tmp_path / "skills"
    write_skill(
        skills / "research" / "deep-research" / "SKILL.md",
        name="deep-research",
        description="Research a complex question.",
    )
    write_skill(
        skills / "creative" / "pixel-art" / "SKILL.md",
        name="pixel-art",
        description="Create pixel art.",
    )

    result = build_contract(skills)

    assert [entry["name"] for entry in result.manifest["skills"]] == [
        "creative/pixel-art",
        "research/deep-research",
    ]
    assert "`skills/creative/pixel-art/SKILL.md`" in result.resolver
    assert "`skills/research/deep-research/SKILL.md`" in result.resolver
    assert result.resolver.index("Create pixel art.") < result.resolver.index(
        "Research a complex question."
    )


def test_contract_rejects_missing_description(tmp_path):
    skills = tmp_path / "skills"
    write_skill(skills / "a" / "one" / "SKILL.md", name="one", description="")

    with pytest.raises(ContractError, match="skill description is required"):
        build_contract(skills)


def test_contract_rejects_outside_skill_symlink_before_reading(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    outside = tmp_path / "outside" / "SKILL.md"
    write_skill(outside, name="outside")
    linked_skill = skills / "linked" / "SKILL.md"
    linked_skill.parent.mkdir(parents=True)
    linked_skill.symlink_to(outside)

    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path == linked_skill:
            pytest.fail("symlinked SKILL.md was read before validation")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    with pytest.raises(ContractError, match="symbolic links are not allowed"):
        build_contract(skills)


def test_contract_rejects_symlinked_path_component(tmp_path):
    skills = tmp_path / "skills"
    outside_dir = tmp_path / "outside"
    write_skill(outside_dir / "SKILL.md", name="outside")
    skills.mkdir()
    (skills / "linked").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ContractError, match="symbolic links are not allowed"):
        build_contract(skills)


@pytest.mark.parametrize(
    "unsafe_segment",
    ["bad|segment", "bad segment", "..skill", "_hidden", "UPPER"],
)
@pytest.mark.parametrize("segment_position", ["category", "skill"])
def test_contract_rejects_unsafe_manifest_path_segments(
    tmp_path, unsafe_segment, segment_position
):
    skills = tmp_path / "skills"
    if segment_position == "category":
        skill = skills / unsafe_segment / "good-skill" / "SKILL.md"
    else:
        skill = skills / "good-category" / unsafe_segment / "SKILL.md"
    write_skill(skill, name="good-skill")

    with pytest.raises(ContractError, match="invalid skill path segment"):
        build_contract(skills)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# No frontmatter\n", "YAML frontmatter is required"),
        ("---\n- one\n- two\n---\n", "frontmatter must be a mapping"),
        ("---\nname: [\n---\n", "invalid YAML frontmatter"),
        ("---\ndescription: Does work.\n---\n", "skill name is required"),
        ("---\nname: 42\ndescription: Does work.\n---\n", "skill name is required"),
        ("---\nname: Bad Name\ndescription: Does work.\n---\n", "invalid skill name"),
        (
            "---\nname: one\ndescription: [does, work]\n---\n",
            "skill description is required",
        ),
    ],
)
def test_contract_rejects_invalid_frontmatter(tmp_path, content, message):
    skill = tmp_path / "skills" / "one" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(content, encoding="utf-8")

    with pytest.raises(ContractError, match=message):
        build_contract(tmp_path / "skills")


def test_multiline_description_is_one_normalized_fallback_trigger(tmp_path):
    skills = tmp_path / "skills"
    skill = skills / "one" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\n"
        "name: one\n"
        "description: |\n"
        "  Research\ta complex\n"
        "  question carefully.\n"
        "---\n",
        encoding="utf-8",
    )

    result = build_contract(skills)

    assert "| Research a complex question carefully. |" in result.resolver
    assert "Research\t" not in result.resolver


def test_explicit_triggers_preserve_declared_order_and_manifest_is_unique(tmp_path):
    skills = tmp_path / "skills"
    write_skill(
        skills / "one" / "SKILL.md",
        name="one",
        description="Fallback trigger.",
        triggers=["Second intent", "First intent"],
    )

    result = build_contract(skills)

    assert result.resolver.index("Second intent") < result.resolver.index(
        "First intent"
    )
    assert "Fallback trigger." not in result.resolver
    assert result.manifest["skills"] == [{"name": "one", "path": "one/SKILL.md"}]


@pytest.mark.parametrize(
    ("triggers", "message"),
    [
        ([], "non-empty list"),
        ("not-a-list", "non-empty list"),
        (["valid", "   "], "trigger must not be empty"),
        (["valid", 7], "trigger must be a string"),
        (["same intent", "same\tintent"], "duplicate trigger"),
        (["left | right"], r"trigger must not contain \|"),
    ],
)
def test_contract_rejects_invalid_explicit_triggers(tmp_path, triggers, message):
    skills = tmp_path / "skills"
    skill = skills / "one" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    frontmatter = {"name": "one", "description": "Does work.", "triggers": triggers}
    import yaml

    skill.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n",
        encoding="utf-8",
    )

    with pytest.raises(ContractError, match=message):
        build_contract(skills)


def test_contract_rejects_pipe_in_description_fallback(tmp_path):
    skills = tmp_path / "skills"
    write_skill(
        skills / "one" / "SKILL.md",
        name="one",
        description="Do left | right.",
    )

    with pytest.raises(ContractError, match=r"trigger must not contain \|"):
        build_contract(skills)


def test_generated_files_are_not_contract_inputs(tmp_path):
    skills = tmp_path / "skills"
    write_skill(skills / "one" / "SKILL.md", name="one")
    (skills / "RESOLVER.md").write_text(
        "| stale | `skills/not-real/SKILL.md` |\n", encoding="utf-8"
    )
    (skills / "manifest.json").write_text(
        '{"skills":[{"name":"not-real","path":"not-real/SKILL.md"}]}',
        encoding="utf-8",
    )

    result = build_contract(skills)

    assert result.manifest["skills"] == [{"name": "one", "path": "one/SKILL.md"}]
    assert "not-real" not in result.resolver


def test_manifest_json_is_sorted_and_has_trailing_newline(tmp_path):
    skills = tmp_path / "skills"
    write_skill(skills / "zeta" / "SKILL.md", name="zeta")
    contract = build_contract(skills)

    rendered = render_manifest_json(contract.manifest)

    assert rendered.endswith("\n")
    assert (
        rendered
        == json.dumps(
            contract.manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_write_then_check_succeeds(tmp_path):
    skills = tmp_path / "skills"
    write_skill(skills / "one" / "SKILL.md", name="one")

    assert main(["--skills-dir", str(skills)]) == 0
    assert main(["--skills-dir", str(skills), "--check"]) == 0
    assert (skills / "RESOLVER.md").read_text(encoding="utf-8").endswith("\n")
    assert (skills / "manifest.json").read_text(encoding="utf-8").endswith("\n")


def test_check_detects_drift_without_writing(tmp_path, capsys):
    skills = tmp_path / "skills"
    write_skill(skills / "one" / "SKILL.md", name="one")
    resolver = skills / "RESOLVER.md"
    manifest = skills / "manifest.json"
    resolver.write_text("stale resolver\n", encoding="utf-8")
    manifest.write_text("stale manifest\n", encoding="utf-8")

    assert main(["--skills-dir", str(skills), "--check"]) == 1

    assert resolver.read_text(encoding="utf-8") == "stale resolver\n"
    assert manifest.read_text(encoding="utf-8") == "stale manifest\n"
    assert "generated GBrain skill contract is out of date" in capsys.readouterr().err
