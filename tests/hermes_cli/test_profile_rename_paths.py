"""Profile rename must rebase absolute paths stored under the old profile directory (#136430).

``rename_profile`` moves ``profiles/<old>/`` to ``profiles/<new>/``, but ``projects.db``
(``projects.primary_path``, ``project_folders.path``, ``discovered_repos.root``) and ``state.db``
(``sessions.cwd``, ``sessions.git_repo_root``, ACP's ``model_config.cwd``) store absolute paths.
Left alone they point at a directory that no longer exists, and the Desktop app refuses to move a
session into the project with ``working directory does not exist: .../profiles/<old>/projects/<slug>``.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import projects_db
from hermes_cli.profile_identity import migrate_profile_identity
from hermes_cli.profiles import create_profile, rename_profile
from hermes_state import SessionDB
from hermes_state_errors import StateDbReplacedError
from hermes_state_registry import acquire, release_or_close

SESSION_IDS = ("inside", "nested", "sibling", "outside", "acp")


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


def _seed_project(db_path: Path, name: str, folder: Path, outside: Path | None = None) -> str:
    """A project rooted at *folder*, optionally with an outside folder and discovered repos."""
    with projects_db.connect_closing(db_path) as conn:
        project_id = projects_db.create_project(conn, name=name, primary_path=str(folder))
        if outside is not None:
            projects_db.add_folder(conn, project_id, str(outside))
            projects_db.record_discovered_repos(conn, [(str(folder), "inside"), (str(outside), "outside")])
    return project_id


def _project_state(db_path: Path, project_id: str):
    """``(primary_path, {folder: is_primary}, discovered roots)``."""
    with projects_db.connect_closing(db_path) as conn:
        project = projects_db.get_project(conn, project_id)
        assert project is not None
        roots = {repo["root"] for repo in projects_db.list_discovered_repos(conn)}
        return project.primary_path, {f.path: f.is_primary for f in project.folders}, roots


def _seed_sessions(db_path: Path, rows: dict) -> None:
    db = acquire(db_path)
    try:
        for session_id, (cwd, repo_root) in rows.items():
            source = "acp" if session_id == "acp" else "desktop"
            model_config = {"cwd": cwd, "model": "m"} if source == "acp" else None
            db.create_session(session_id, source, cwd=cwd, git_repo_root=repo_root, model_config=model_config)
    finally:
        release_or_close(db)


def _session_state(db_path: Path) -> dict:
    """``{id: (cwd, git_repo_root)}`` plus ``acp_model_config`` for the ACP row."""
    db = acquire(db_path)
    try:
        rows = {sid: db.get_session(sid) for sid in SESSION_IDS}
        state = {sid: (row["cwd"], row["git_repo_root"]) for sid, row in rows.items() if row}
        acp = rows["acp"]
        assert acp is not None
        state["acp_model_config"] = json.loads(acp["model_config"])
        return state
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
    project_id = _seed_project(old_dir / "projects.db", "Active Trading", folder, outside)
    # The default profile's projects.db can also hold a project rooted inside a named profile.
    root_project_id = _seed_project(profile_env / "projects.db", "From Root", folder / "sub")
    rows = {
        "inside": (str(folder), str(folder)),
        "nested": (str(folder / "data"), None),
        "sibling": (str(sibling / "projects" / "x"), None),
        "outside": (str(outside), str(outside)),
        "acp": (str(folder), None),
    }
    _seed_sessions(old_dir / "state.db", rows)
    _seed_sessions(profile_env / "state.db", rows)
    return {"project_id": project_id, "root_project_id": root_project_id, "outside": outside, "rows": rows}


def _assert_rebased(new_dir: Path, profile_env: Path, seeded: dict) -> None:
    new_folder = new_dir / "projects" / "active-trading"
    outside = seeded["outside"]
    primary, folders, roots = _project_state(new_dir / "projects.db", seeded["project_id"])
    assert primary == str(new_folder)
    assert folders == {str(new_folder): True, str(outside): False}
    assert roots == {str(new_folder), str(outside)}

    root_primary, root_folders, _ = _project_state(profile_env / "projects.db", seeded["root_project_id"])
    assert root_primary == str(new_folder / "sub")
    assert root_folders == {str(new_folder / "sub"): True}

    expected = {
        "inside": (str(new_folder), str(new_folder)),
        "nested": (str(new_folder / "data"), None),
        "sibling": seeded["rows"]["sibling"],  # profiles/wealthfront2 is not under profiles/wealthfront
        "outside": seeded["rows"]["outside"],
        "acp": (str(new_folder), None),
        # ACP list/resume read the workspace from model_config, not the column.
        "acp_model_config": {"cwd": str(new_folder), "model": "m"},
    }
    for db_path in (new_dir / "state.db", profile_env / "state.db"):
        assert _session_state(db_path) == expected


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
    primary, _, _ = _project_state(new_dir / "projects.db", seeded["project_id"])
    assert primary is not None and "/wealthfront/" in primary  # the pre-fix state the user is left in

    with patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        assert migrate_profile_identity("wealthfront", "finance") is True
        _assert_rebased(new_dir, profile_env, seeded)
        assert migrate_profile_identity("wealthfront", "finance") is True  # idempotent
    _assert_rebased(new_dir, profile_env, seeded)


def test_a_state_store_failure_never_aborts_the_rest_of_the_rename(profile_env, seeded, capsys):
    """The directory has already moved: any rebase error must leave the remaining steps running."""
    with patch.object(SessionDB, "rebase_session_paths", side_effect=StateDbReplacedError("replaced")), \
         patch("hermes_cli.profiles._maybe_register_gateway_service") as register:
        new_dir = _rename("wealthfront", "finance")

    assert new_dir.is_dir()
    register.assert_called_with("finance")  # rename_profile step 7 still ran
    err = capsys.readouterr().err
    assert "StateDbReplacedError" in err
    assert "hermes profile migrate-identity wealthfront finance" in err
    # projects.db is independent of the failing store and is still rebased.
    primary, _, _ = _project_state(new_dir / "projects.db", seeded["project_id"])
    assert primary == str(new_dir / "projects" / "active-trading")


def test_migrate_identity_leaves_paths_of_a_profile_that_reused_the_old_name(profile_env, capsys):
    """After the old name is reused, ``profiles/<old>/`` paths belong to the new profile."""
    create_profile("wealthfront", no_alias=True)
    with patch("hermes_cli.profile_identity._migrate_profile_paths", return_value=True):
        _rename("wealthfront", "finance")
    reborn = create_profile("wealthfront", no_alias=True)
    (reborn / "w").mkdir()
    project_id = _seed_project(profile_env / "projects.db", "Reborn", reborn / "w")

    with patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        assert migrate_profile_identity("wealthfront", "finance") is False
    primary, _, _ = _project_state(profile_env / "projects.db", project_id)
    assert primary == str(reborn / "w")
    assert "profile named 'wealthfront' exists again" in capsys.readouterr().err


def test_rebase_merges_a_folder_that_already_exists_under_the_new_path(profile_env):
    """project_folders is keyed (project_id, path): a pre-existing new-path row is merged."""
    old_dir = create_profile("wealthfront", no_alias=True)
    folder = old_dir / "projects" / "p"
    folder.mkdir(parents=True)
    new_folder = profile_env / "profiles" / "finance" / "projects" / "p"
    with projects_db.connect_closing(old_dir / "projects.db") as conn:
        project_id = projects_db.create_project(conn, name="P", primary_path=str(folder))
        projects_db.add_folder(conn, project_id, str(new_folder))
    new_dir = _rename("wealthfront", "finance")
    primary, folders, _ = _project_state(new_dir / "projects.db", project_id)
    assert primary == str(new_folder)
    assert folders == {str(new_folder): True}  # the merged row inherits the primary flag


def test_rebase_leaves_primary_flags_of_untouched_projects_alone(profile_env, tmp_path):
    """Only projects whose rows the rename merged get their primary flag re-derived."""
    old_dir = create_profile("wealthfront", no_alias=True)
    (old_dir / "p").mkdir()
    with projects_db.connect_closing(old_dir / "projects.db") as conn:
        projects_db.create_project(conn, name="Inside", primary_path=str(old_dir / "p"))
        drifted = projects_db.create_project(conn, name="Drifted", primary_path=str(tmp_path / "Repo"))
        legacy = projects_db.create_project(conn, name="Legacy", primary_path=str(tmp_path / "a"))
        # An older row whose folder spelling drifted from primary_path (normalize-equal, not string-equal).
        conn.execute("UPDATE project_folders SET path = ? WHERE project_id = ?", (str(tmp_path / "Repo") + "/", drifted))
        # Pre-existing state that disagrees with primary_path: not this rename's business to "fix".
        conn.execute("UPDATE projects SET primary_path = ? WHERE id = ?", (str(tmp_path / "b"), legacy))
        conn.commit()
    new_dir = _rename("wealthfront", "finance")
    _, drifted_folders, _ = _project_state(new_dir / "projects.db", drifted)
    _, legacy_folders, _ = _project_state(new_dir / "projects.db", legacy)
    assert drifted_folders == {str(tmp_path / "Repo") + "/": True}
    assert legacy_folders == {str(tmp_path / "a"): True}


def test_rename_rebases_paths_stored_through_a_symlinked_hermes_root(tmp_path, monkeypatch):
    """Paths may be stored in the symlink-resolved spelling of the Hermes root (e.g. /var vs /private/var)."""
    real_home = tmp_path / "real-hermes"
    real_home.mkdir()
    link_home = tmp_path / ".hermes"
    link_home.symlink_to(real_home, target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(link_home))

    old_dir = create_profile("wealthfront", no_alias=True)
    resolved_folder = real_home / "profiles" / "wealthfront" / "projects" / "p"
    resolved_folder.mkdir(parents=True)
    assert str(old_dir).startswith(str(link_home))  # built through the symlink
    project_id = _seed_project(old_dir / "projects.db", "P", resolved_folder)

    new_dir = _rename("wealthfront", "finance")
    primary, _, _ = _project_state(new_dir / "projects.db", project_id)
    assert primary == str(real_home / "profiles" / "finance" / "projects" / "p")
