"""Behavior tests for the trusted Desktop Project session binding."""

from __future__ import annotations

from hermes_cli import projects_db
from hermes_cli.desktop_project_authority import (
    prepare_desktop_project_request,
    validate_desktop_project,
)


def test_validated_project_id_resolves_its_host_owned_paths(tmp_path):
    primary = tmp_path / "primary"
    member = tmp_path / "member"
    primary.mkdir()
    member.mkdir()
    database = tmp_path / "projects.db"

    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(
            connection,
            name="Research",
            slug="research",
            primary_path=str(primary),
            folders=[str(member)],
        )

    project = validate_desktop_project(project_id, db_path=database)

    assert project is not None
    assert project.project_id == project_id
    assert project.primary_path == str(primary.resolve())
    assert set(project.folders) == {str(primary.resolve()), str(member.resolve())}


def test_project_slug_is_not_accepted_as_desktop_identity(tmp_path):
    database = tmp_path / "projects.db"
    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", slug="research", primary_path=str(tmp_path)
        )

    assert validate_desktop_project("research", db_path=database) is None
    assert validate_desktop_project(project_id, db_path=database) is not None


def test_archived_or_unknown_project_is_rejected(tmp_path):
    database = tmp_path / "projects.db"
    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(connection, name="Research", primary_path=str(tmp_path))
        connection.execute("UPDATE projects SET archived = 1 WHERE id = ?", (project_id,))
        connection.commit()

    assert validate_desktop_project("p_unknown", db_path=database) is None
    assert validate_desktop_project(project_id, db_path=database) is None


def test_missing_project_store_is_not_created(tmp_path):
    database = tmp_path / "missing.db"

    assert validate_desktop_project("p_unknown", db_path=database) is None
    assert not database.exists()


def test_structured_project_id_replaces_untrusted_inherited_cwd(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "projects.db"
    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", primary_path=str(root)
        )

    params, project = prepare_desktop_project_request(
        {
            "source": "gui",
            "desktop_project_id": project_id,
            "cwd": str(tmp_path / "untrusted"),
            "cwd_explicit": False,
        },
        db_path=database,
    )

    assert project is not None and project.project_id == project_id
    assert params["cwd"] == str(root.resolve())
    assert params["cwd_explicit"] is True
    assert params["desktop_project_id"] == project_id


def test_structured_project_and_conflicting_explicit_cwd_fail_closed(tmp_path):
    project_root = tmp_path / "project"
    other_root = tmp_path / "other"
    project_root.mkdir()
    other_root.mkdir()
    database = tmp_path / "projects.db"
    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", primary_path=str(project_root)
        )

    try:
        prepare_desktop_project_request(
            {
                "source": "gui",
                "desktop_project_id": project_id,
                "cwd": str(other_root),
                "cwd_explicit": True,
            },
            db_path=database,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting explicit workspace must be refused")


def test_host_project_id_is_only_accepted_for_gui_sessions(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "projects.db"
    with projects_db.connect_closing(db_path=database) as connection:
        project_id = projects_db.create_project(
            connection, name="Research", primary_path=str(root)
        )

    try:
        prepare_desktop_project_request(
            {"source": "cli", "desktop_project_id": project_id},
            db_path=database,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-GUI session must not claim a Desktop Project")
