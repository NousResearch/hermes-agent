from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli.federation import (
    _discover_skill_names,
    _find_bundled_skill,
    audit_federation,
    load_manifest,
    seed_federation,
    seed_federation_groups,
)


MANIFEST = Path(__file__).parents[2] / "configs" / "federation" / "roles.json"


def test_manifest_covers_the_federation_departments() -> None:
    manifest = load_manifest(MANIFEST)

    assert {department.id for department in manifest.departments} >= {
        "federal_core",
        "engineering",
        "research_intelligence",
        "knowledge_commons",
        "arts_media",
        "content_studio",
        "memory",
    }
    role_ids = {role.id for role in manifest.roles}
    assert {
        "arts-director",
        "librarian",
        "nerdy-content-scout",
        "memory-curator",
        "capability-planner",
        "scientific-validator",
    } <= role_ids
    assert len(role_ids) == sum(len(department.roles) for department in manifest.departments)
    grouped_roles = {role_id for group in manifest.groups for role_id in group.roles}
    assert grouped_roles == role_ids
    assert manifest.model_policy_provenance["evidence_class"] == (
        "configured_routes_and_partial_local_benchmark_evidence"
    )


def test_default_skill_discovery_uses_checkout_roots() -> None:
    skills = _discover_skill_names()

    assert {"arxiv", "llm-wiki", "youtube-content"} <= skills


def test_skill_discovery_honors_packaged_skill_roots(tmp_path: Path, monkeypatch) -> None:
    bundled = tmp_path / "bundled"
    optional = tmp_path / "optional"
    (bundled / "research" / "packaged-skill").mkdir(parents=True)
    (optional / "creative" / "optional-skill").mkdir(parents=True)
    (bundled / "research" / "packaged-skill" / "SKILL.md").write_text("# bundled\n")
    (optional / "creative" / "optional-skill" / "SKILL.md").write_text("# optional\n")
    monkeypatch.setenv("HERMES_BUNDLED_SKILLS", str(bundled))
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional))

    assert {"packaged-skill", "optional-skill"} <= _discover_skill_names()
    assert _find_bundled_skill("packaged-skill") == bundled / "research" / "packaged-skill"


def test_audit_distinguishes_exact_profiles_from_equivalent_coverage() -> None:
    manifest = load_manifest(MANIFEST)

    report = audit_federation(
        manifest,
        existing_profiles={"task-orchestrator", "writer", "research-scout"},
        available_skills={"llm-wiki", "youtube-content"},
    )

    by_id = {row["role_id"]: row for row in report["roles"]}
    assert by_id["federation-steward"]["status"] == "covered_by_existing"
    assert by_id["writer"]["status"] == "installed"
    assert by_id["arts-director"]["status"] == "missing"
    assert by_id["arts-director"]["skill_ready"] is False
    assert report["summary"]["missing"] > 0
    assert report["summary"]["covered_by_existing"] > 0


def test_seed_is_dry_run_by_default_and_selects_a_department() -> None:
    manifest = load_manifest(MANIFEST)
    calls: list[str] = []

    result = seed_federation(
        manifest,
        department="arts_media",
        existing_profiles=set(),
        apply=False,
        create_profile=lambda **kwargs: calls.append(kwargs["name"]),
    )

    assert result["applied"] is False
    assert result["created"] == []
    assert result["planned"]
    assert calls == []
    assert all(row["department"] == "arts_media" for row in result["planned"])


def test_seed_apply_writes_role_identity_metadata(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "librarian"
    profile_dir.mkdir(parents=True)

    def create_profile(**kwargs):
        assert kwargs["name"] == "librarian"
        return profile_dir

    result = seed_federation(
        manifest,
        role_ids=["librarian"],
        existing_profiles=set(),
        apply=True,
        create_profile=create_profile,
    )

    assert result["created"] == ["librarian"]
    metadata = json.loads((profile_dir / "federation_role.json").read_text())
    assert metadata["role_id"] == "librarian"
    assert metadata["department"] == "knowledge_commons"
    assert metadata["authority"] == "advisory"
    assert (profile_dir / "SOUL.md").read_text().startswith("# Federation Librarian")
    assert (profile_dir / "skills" / "research" / "llm-wiki" / "SKILL.md").is_file()
    config = (profile_dir / "config.yaml").read_text()
    assert "gpt-5.5" in config
    assert "agent:" in config
    assert "reasoning_effort: low" in config
    assert "auxiliary:" in config
    assert "qwen3.5:4b" in config
    assert "Your working style:" in (profile_dir / "SOUL.md").read_text()


def test_seed_refresh_existing_preserves_soul_and_adopts_route(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "writer"
    profile_dir.mkdir(parents=True)
    original_soul = "# My existing writer\n\nKeep the author's voice.\n"
    (profile_dir / "SOUL.md").write_text(original_soul)
    (profile_dir / "config.yaml").write_text(
        "model:\n"
        "  provider: old\n"
        "  default: old-model\n"
        "  base_url: https://old.example\n"
        "  api_key: stale-key\n"
        "  api_mode: old-mode\n"
    )

    result = seed_federation(
        manifest,
        role_ids=["writer"],
        existing_profiles={"writer"},
        apply=True,
        refresh_existing=True,
        create_profile=lambda **kwargs: profile_dir,
        profile_dir_for=lambda name: profile_dir,
    )

    assert result["refreshed_existing"] == ["writer"]
    soul = (profile_dir / "SOUL.md").read_text()
    assert soul.startswith(original_soul.rstrip())
    assert "Federation Role Profile" in soul
    assert "Federation working style:" in soul
    assert json.loads((profile_dir / "federation_role.json").read_text())["role_id"] == "writer"
    refreshed_config = (profile_dir / "config.yaml").read_text()
    assert "gpt-5.5" in refreshed_config
    assert "old.example" not in refreshed_config
    assert "stale-key" not in refreshed_config
    assert "old-mode" not in refreshed_config


def test_seed_refresh_existing_with_identity_preserves_custom_soul(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "writer"
    profile_dir.mkdir(parents=True)
    original_soul = "# Customized writer\n\nKeep this voice exactly.\n"
    (profile_dir / "SOUL.md").write_text(original_soul)
    (profile_dir / "federation_role.json").write_text('{"role_id":"writer"}\n')

    result = seed_federation(
        manifest,
        role_ids=["writer"],
        existing_profiles={"writer"},
        apply=True,
        refresh_existing=True,
        profile_dir_for=lambda name: profile_dir,
    )

    assert not result["failed"]
    assert (profile_dir / "SOUL.md").read_text().startswith(original_soul.rstrip())


def test_seed_cleans_up_partial_new_profile_after_failure(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "writer"
    profile_dir.parent.mkdir()

    def create_profile(**kwargs):
        profile_dir.mkdir()
        raise RuntimeError("simulated profile setup failure")

    result = seed_federation(
        manifest,
        role_ids=["writer"],
        existing_profiles=set(),
        apply=True,
        create_profile=create_profile,
        profile_dir_for=lambda name: profile_dir,
    )

    assert result["failed"]
    assert not profile_dir.exists()


def test_seed_does_not_refresh_owned_profile_without_explicit_opt_in(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "writer"
    profile_dir.mkdir(parents=True)
    soul = "# Existing\n"
    config = "model:\n  provider: old\n  default: old-model\n"
    (profile_dir / "SOUL.md").write_text(soul)
    (profile_dir / "config.yaml").write_text(config)
    (profile_dir / "federation_role.json").write_text('{"role_id":"writer"}\n')

    result = seed_federation(
        manifest,
        role_ids=["writer"],
        existing_profiles={"writer"},
        apply=True,
        profile_dir_for=lambda name: profile_dir,
    )

    assert result["refreshed_existing"] == []
    assert (profile_dir / "SOUL.md").read_text() == soul
    assert (profile_dir / "config.yaml").read_text() == config


def test_seed_fails_closed_on_invalid_config_yaml(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profile_dir = tmp_path / "profiles" / "writer"
    profile_dir.mkdir(parents=True)
    original_soul = "# Existing\n"
    (profile_dir / "SOUL.md").write_text(original_soul)
    (profile_dir / "config.yaml").write_text("model: [not: valid\n")

    result = seed_federation(
        manifest,
        role_ids=["writer"],
        existing_profiles={"writer"},
        apply=True,
        refresh_existing=True,
        profile_dir_for=lambda name: profile_dir,
    )

    assert result["failed"]
    assert "invalid YAML" in result["failed"][0]["error"]
    assert not (profile_dir / "federation_role.json").exists()
    assert (profile_dir / "SOUL.md").read_text() == original_soul


def test_seed_groups_writes_stable_projection_and_memberships(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for role in manifest.roles:
        (profiles / role.id).mkdir()
    default = tmp_path / "default"
    default.mkdir()

    result = seed_federation_groups(
        manifest,
        apply=True,
        profile_dir_for=lambda name: default if name == "default" else profiles / name,
    )

    assert len(result["seeded"]) == len(manifest.groups)
    assert not result["failed"]
    import yaml

    default_doc = yaml.safe_load((default / "profile.yaml").read_text())
    snapshot = default_doc["ui_meta"]["hermes-bots-groups"]
    assert snapshot["version"] == 3
    assert len(snapshot["rooms"]) == len(manifest.groups)
    assert all(2 <= len(room["members"]) <= 6 for room in snapshot["rooms"].values())
    librarian_doc = yaml.safe_load((profiles / "librarian" / "profile.yaml").read_text())
    assert "Knowledge Commons" in librarian_doc["ui_meta"]["hermes-bots"]["groups"]

    second = seed_federation_groups(
        manifest,
        apply=True,
        profile_dir_for=lambda name: default if name == "default" else profiles / name,
    )
    assert second["metadata_updated"] == []
    assert yaml.safe_load((default / "profile.yaml").read_text()) == default_doc


def test_seed_groups_preserves_existing_room_history(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for role in manifest.roles:
        (profiles / role.id).mkdir()
    default = tmp_path / "default"
    default.mkdir()
    seed_federation_groups(
        manifest,
        apply=True,
        profile_dir_for=lambda name: default if name == "default" else profiles / name,
    )
    import yaml

    path = default / "profile.yaml"
    doc = yaml.safe_load(path.read_text())
    room_key = next(iter(doc["ui_meta"]["hermes-bots-groups"]["rooms"]))
    room = doc["ui_meta"]["hermes-bots-groups"]["rooms"][room_key]
    room["log"] = [{"role": "user", "content": "keep me"}]
    room["revision"] = 7
    room["custom"] = "preserve me"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))

    result = seed_federation_groups(
        manifest,
        apply=True,
        profile_dir_for=lambda name: default if name == "default" else profiles / name,
    )

    assert not result["failed"]
    after = yaml.safe_load(path.read_text())
    after_room = after["ui_meta"]["hermes-bots-groups"]["rooms"][room_key]
    assert after_room["log"] == [{"role": "user", "content": "keep me"}]
    assert after_room["revision"] == 7
    assert after_room["custom"] == "preserve me"


def test_seed_rejects_unknown_role_or_department() -> None:
    manifest = load_manifest(MANIFEST)

    with pytest.raises(ValueError, match="unknown federation role"):
        seed_federation(manifest, role_ids=["does-not-exist"], apply=False)
    with pytest.raises(ValueError, match="unknown federation department"):
        seed_federation(manifest, department="does-not-exist", apply=False)
