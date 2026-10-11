"""Profile rename must rebase absolute paths stored under the old profile directory (#136430).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/``, but ``projects.db``
(``projects.primary_path``, ``project_folders.path``, ``discovered_repos.root``) and
``state.db`` (``sessions.cwd``, ``sessions.git_repo_root``) store absolute paths. Left alone they
point at a directory that no longer exists, and the Desktop app refuses to move a session into
the project with ``working directory does not exist: .../profiles/<old>/projects/<slug>``.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import projects_db
from hermes_cli.profile_identity import migrate_profile_identity
from hermes_cli.profiles import create_profile, rename_profile
from hermes_state_registry import acquire, release_or_close


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """Isolate profile paths and the process-level Hermes root."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def _rename(old: str, new: str, *, live_mux: bool = False) -> Path:
    with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=live_mux):
        return rename_profile(old, new)


def _seed_project(profile_dir: Path, folder: Path, outside: Path) -> str:
    """A project rooted under the profile, plus an outside folder and discovered repos."""
    with projects_db.connect_closing(profile_dir / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name="Active Trading", primary_path=str(folder))
        projects_db.add_folder(conn, project_id, str(outside))
        projects_db.record_discovered_repos(conn, [(str(folder), "inside"), (str(outside), "outside")])
    return project_id


def _project_paths(profile_dir: Path, project_id: str):
    with projects_db.connect_closing(profile_dir / "projects.db") as conn:
        project = projects_db.get_project(conn, project_id)
        assert project is not None
        roots = {repo["root"] for repo in projects_db.list_discovered_repos(conn)}
        return project.primary_path, {f.path for f in project.folders}, roots


def _seed_sessions(db_path: Path, rows: dict[str, tuple[str, str | None]]) -> None:
    db = acquire(db_path)
    try:
        for session_id, (cwd, repo_root) in rows.items():
            db.create_session(session_id, "desktop", cwd=cwd, git_repo_root=repo_root)
    finally:
        release_or_close(db)


def _session_paths(db_path: Path) -> dict[str, tuple[str, str | None]]:
    db = acquire(db_path)
    try:
        rows = {sid: db.get_session(sid) for sid in ("inside", "nested", "sibling", "outside")}
        return {sid: (row["cwd"], row["git_repo_root"]) for sid, row in rows.items() if row}
    finally:
        release_or_close(db)


@pytest.fixture()
def seeded(profile_env, tmp_path):
    old_dir = create_profile("wealthfront", no_alias=True)
    sibling = create_profile("wealthfront2", no_alias=True)  # shares the old name as a prefix
    folder = old_dir / "projects" / "active-trading"
    folder.mkdir(parents=True)
    outside = tmp_path / "outside-repo"
    outside.mkdir()
    project_id = _seed_project(old_dir, folder, outside)
    rows = {
        "inside": (str(folder), str(folder)),
        "nested": (str(folder / "data"), None),
        "sibling": (str(sibling / "projects" / "x"), None),
        "outside": (str(outside), str(outside)),
    }
    _seed_sessions(old_dir / "state.db", rows)
    _seed_sessions(profile_env / "state.db", rows)
    return {"project_id": project_id, "outside": outside, "sibling": sibling, "rows": rows}


def _assert_rebased(new_dir: Path, profile_env: Path, seeded: dict) -> None:
    new_folder = new_dir / "projects" / "active-trading"
    outside = seeded["outside"]
    primary, folders, roots = _project_paths(new_dir, seeded["project_id"])
    assert primary == str(new_folder)
    assert folders == {str(new_folder), str(outside)}
    assert roots == {str(new_folder), str(outside)}

    expected = {
        "inside": (str(new_folder), str(new_folder)),
        "nested": (str(new_folder / "data"), None),
        "sibling": seeded["rows"]["sibling"],  # profiles/wealthfront2 is not under profiles/wealthfront
        "outside": seeded["rows"]["outside"],
    }
    for db_path in (new_dir / "state.db", profile_env / "state.db"):
        assert _session_paths(db_path) == expected


def test_rename_rebases_project_and_session_paths(profile_env, seeded):
    new_dir = _rename("wealthfront", "finance")
    _assert_rebased(new_dir, profile_env, seeded)


def test_rename_rebases_paths_when_a_live_gateway_owns_routing(profile_env, seeded):
    """Path rebasing is durable-only; it must not be skipped when the gateway migrates routing."""
    with patch("gateway.control_socket.migrate_gateway_profile_identity", return_value={"ok": True}), \
         patch("hermes_cli.profiles._notify_multiplexer"), \
         patch("hermes_cli.profiles.mark_named_profile_deleted"), \
         patch("hermes_cli.profiles.clear_named_profile_deleted"):
        new_dir = _rename("wealthfront", "finance", live_mux=True)
    _assert_rebased(new_dir, profile_env, seeded)


def test_migrate_identity_repairs_a_rename_that_left_stale_paths(profile_env, seeded):
    """``hermes profile migrate-identity <old> <new>`` repairs profiles renamed before the fix."""
    with patch("hermes_cli.profile_identity._migrate_profile_paths", return_value=True):
        new_dir = _rename("wealthfront", "finance")
    primary, _, _ = _project_paths(new_dir, seeded["project_id"])
    assert primary is not None and "/wealthfront/" in primary  # the pre-fix state the user is left in

    with patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        assert migrate_profile_identity("wealthfront", "finance") is True
        _assert_rebased(new_dir, profile_env, seeded)
        assert migrate_profile_identity("wealthfront", "finance") is True  # idempotent
    _assert_rebased(new_dir, profile_env, seeded)


def test_rebase_merges_a_folder_that_already_exists_under_the_new_path(profile_env, tmp_path):
    """project_folders is keyed (project_id, path): a pre-existing new-path row must not abort."""
    old_dir = create_profile("wealthfront", no_alias=True)
    folder = old_dir / "projects" / "p"
    folder.mkdir(parents=True)
    new_folder = profile_env / "profiles" / "finance" / "projects" / "p"
    with projects_db.connect_closing(old_dir / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name="P", primary_path=str(folder))
        projects_db.add_folder(conn, project_id, str(new_folder))
    new_dir = _rename("wealthfront", "finance")
    primary, folders, _ = _project_paths(new_dir, project_id)
    assert primary == str(new_folder)
    assert folders == {str(new_folder)}
