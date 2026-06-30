from pathlib import Path

import pytest


SKILL_MD = """---
name: {name}
description: {description}
version: 1.2.3
---

# {name}
"""


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app

    monkeypatch.setenv("CONTAINER_INTERNAL_TOKEN", "test-token")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers["X-Container-Token"] = "test-token"
    c.headers["Authorization"] = "Bearer test-token"
    return c


def _write_skill(root: Path, slug: str, *, description: str = "test skill") -> Path:
    skill_dir = root / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        SKILL_MD.format(name=slug, description=description),
        encoding="utf-8",
    )
    return skill_dir


def test_container_skills_requires_both_headers(client):
    client.headers.pop("Authorization", None)
    response = client.get("/skills")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_container_token"


def test_container_skills_list_marks_bundled_and_workspace(client):
    from hermes_constants import get_hermes_home

    skills_dir = get_hermes_home() / "skills"
    _write_skill(skills_dir, "static-site", description="Publish static sites")
    _write_skill(skills_dir, "my-skill", description="User installed")
    (skills_dir / ".bundled_manifest").write_text("static-site:abc123\n", encoding="utf-8")

    response = client.get("/skills")
    assert response.status_code == 200
    skills = {skill["slug"]: skill for skill in response.json()["skills"]}

    assert skills["static-site"]["source"] == "bundled"
    assert skills["static-site"]["bundled"] is True
    assert skills["static-site"]["description"] == "Publish static sites"
    assert skills["my-skill"]["source"] == "workspace"
    assert skills["my-skill"]["bundled"] is False


def test_container_skill_upload_writes_and_can_overwrite_workspace(client):
    from hermes_constants import get_hermes_home

    response = client.post(
        "/skills/upload",
        json={
            "slug": "my-skill",
            "files": [
                {"path": "SKILL.md", "content": SKILL_MD.format(name="my-skill", description="First")},
                {"path": "references/api.md", "content": "# API"},
            ],
        },
    )
    assert response.status_code == 201
    data = response.json()["skill"]
    assert data["slug"] == "my-skill"
    assert data["source"] == "workspace"
    assert data["totalBytes"] > 0
    assert (get_hermes_home() / "skills" / "my-skill" / "references" / "api.md").exists()

    response = client.post(
        "/skills/upload",
        json={
            "slug": "my-skill",
            "files": [
                {"path": "SKILL.md", "content": SKILL_MD.format(name="my-skill", description="Second")},
            ],
        },
    )
    assert response.status_code == 201
    assert not (get_hermes_home() / "skills" / "my-skill" / "references" / "api.md").exists()


@pytest.mark.parametrize(
    "payload,code",
    [
        ({"slug": "../bad", "files": [{"path": "SKILL.md", "content": "x"}]}, "invalid_slug"),
        ({"slug": "my-skill", "files": [{"path": "nested/SKILL.md", "content": "x"}]}, "missing_skill_md"),
        ({"slug": "my-skill", "files": [{"path": "../SKILL.md", "content": "x"}]}, "unsafe_path"),
        ({"slug": "my-skill", "files": [{"path": "SKILL.md", "content": "x"}] * 81}, "too_many_files"),
        ({"slug": "my-skill", "files": [{"path": "SKILL.md", "content": "x" * (512 * 1024 + 1)}]}, "payload_too_large"),
    ],
)
def test_container_skill_upload_validation_errors(client, payload, code):
    response = client.post("/skills/upload", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


def test_container_skill_upload_rejects_bundled_overwrite(client):
    from hermes_constants import get_hermes_home

    skills_dir = get_hermes_home() / "skills"
    _write_skill(skills_dir, "static-site")
    (skills_dir / ".bundled_manifest").write_text("static-site:abc123\n", encoding="utf-8")

    response = client.post(
        "/skills/upload",
        json={
            "slug": "static-site",
            "files": [{"path": "SKILL.md", "content": SKILL_MD.format(name="static-site", description="New")}],
        },
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "bundled_skill_immutable"


def test_container_skill_read_file_rejects_traversal_and_symlinks(client, tmp_path):
    from hermes_constants import get_hermes_home

    skills_dir = get_hermes_home() / "skills"
    skill_dir = _write_skill(skills_dir, "my-skill")
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "api.md").write_text("# API", encoding="utf-8")
    (skill_dir / "leak").symlink_to(tmp_path / "outside")

    response = client.get("/skills/my-skill/files", params={"path": "references/api.md"})
    assert response.status_code == 200
    assert response.json()["content"] == "# API"
    assert response.json()["sizeBytes"] == 5

    response = client.get("/skills/my-skill/files", params={"path": "../secret"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsafe_path"

    response = client.get("/skills/my-skill/files", params={"path": "leak/file.txt"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsafe_path"


def test_container_skill_delete_only_workspace(client):
    from hermes_constants import get_hermes_home

    skills_dir = get_hermes_home() / "skills"
    _write_skill(skills_dir, "workspace-skill")
    _write_skill(skills_dir, "bundled-skill")
    (skills_dir / ".bundled_manifest").write_text("bundled-skill:abc123\n", encoding="utf-8")

    response = client.delete("/skills/bundled-skill")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "bundled_skill_immutable"

    response = client.delete("/skills/workspace-skill")
    assert response.status_code == 200
    assert response.json() == {
        "slug": "workspace-skill",
        "source": "workspace",
        "deleted": True,
    }
    assert not (skills_dir / "workspace-skill").exists()
