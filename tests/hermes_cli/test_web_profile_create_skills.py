"""Route-level contract for Profile Builder seeded-skill selection."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


SEEDED_SKILLS = ("alpha-skill", "beta-skill")


def _seed_profile_skills(
    profile_dir: Path, quiet: bool = False
) -> dict[str, list[str]]:
    del quiet
    for skill_name in SEEDED_SKILLS:
        skill_dir = profile_dir / "skills" / "custom" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\n---\n\n# {skill_name}\n",
            encoding="utf-8",
        )
    return {"copied": list(SEEDED_SKILLS)}


@pytest.fixture()
def profile_create_client(tmp_path, monkeypatch, _isolate_hermes_home):
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles as profiles_mod
    from hermes_cli import web_server

    home = get_hermes_home()
    profiles_root = home / "profiles"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(profiles_mod, "seed_profile_skills", _seed_profile_skills)
    monkeypatch.setattr(profiles_mod, "check_alias_collision", lambda _name: "test")

    with TestClient(web_server.app) as client:
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        yield client, profiles_root


def _disabled_skills(profile_dir: Path) -> set[str]:
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import load_config
    from hermes_cli.skills_config import get_disabled_skills

    token = set_hermes_home_override(str(profile_dir))
    try:
        return get_disabled_skills(load_config())
    finally:
        reset_hermes_home_override(token)


def test_omitted_keep_skills_preserves_seeded_bundle(profile_create_client):
    client, profiles_root = profile_create_client

    response = client.post("/api/profiles", json={"name": "defaults-kept"})

    assert response.status_code == 200, response.text
    assert response.json()["skills_disabled"] == 0
    profile_dir = profiles_root / "defaults-kept"
    assert _disabled_skills(profile_dir).isdisjoint(SEEDED_SKILLS)
    for skill_name in SEEDED_SKILLS:
        assert (profile_dir / "skills" / "custom" / skill_name / "SKILL.md").is_file()


def test_explicit_empty_keep_skills_disables_seeded_bundle(profile_create_client):
    client, profiles_root = profile_create_client

    response = client.post(
        "/api/profiles",
        json={"name": "defaults-disabled", "keep_skills": []},
    )

    assert response.status_code == 200, response.text
    assert response.json()["skills_disabled"] == len(SEEDED_SKILLS)
    assert _disabled_skills(profiles_root / "defaults-disabled") == set(SEEDED_SKILLS)
