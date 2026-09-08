"""Tests for _resolve_agent_workspace: the memory-provider workspace name
must reflect the registered project the agent is working in, and fall back
to "" (no workspace) outside any project so {workspace} bank templates
resolve to the configured bank_id.

All fixture data is generic placeholders (a project named "acme", a role
named "lead", tmp_path directories) — no real projects, orgs, agents or
user paths.
"""

import pytest


def _acme_folder(tmp_path):
    d = tmp_path / "acme"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture()
def fake_projects_db(tmp_path, monkeypatch):
    """Build a real project row (acme) in an isolated projects.db and point
    the backend helpers at it."""
    from hermes_cli import projects_db as pdb

    db_file = tmp_path / "projects.db"
    monkeypatch.setattr(pdb, "projects_db_path", lambda: db_file)
    with pdb.connect_closing() as conn:
        pid = pdb.create_project(
            conn,
            name="acme",
            folders=[_acme_folder(tmp_path)],
        )
    return pid


def _resolve(cwd, monkeypatch):
    """Run _resolve_agent_workspace with a forced TERMINAL_CWD (which
    resolve_agent_cwd honors over os.getcwd)."""
    monkeypatch.setenv("TERMINAL_CWD", cwd)
    from agent.agent_init import _resolve_agent_workspace
    return _resolve_agent_workspace()


class TestResolveAgentWorkspace:
    def test_exact_project_folder(self, tmp_path, monkeypatch, fake_projects_db):
        assert _resolve(_acme_folder(tmp_path), monkeypatch) == "acme"

    def test_nested_folder_inside_project(self, tmp_path, monkeypatch, fake_projects_db):
        nested = tmp_path / "acme" / "src" / "views"
        nested.mkdir(parents=True)
        assert _resolve(str(nested), monkeypatch) == "acme"

    def test_outside_any_project_is_empty(self, tmp_path, monkeypatch, fake_projects_db):
        other = tmp_path / "other-project"
        other.mkdir(parents=True, exist_ok=True)
        assert _resolve(str(tmp_path), monkeypatch) == ""
        assert _resolve(str(other), monkeypatch) == ""

    def test_unresolvable_cwd_is_empty(self, tmp_path, monkeypatch, fake_projects_db):
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        monkeypatch.chdir(tmp_path)
        from agent.agent_init import _resolve_agent_workspace
        assert _resolve_agent_workspace() == ""

    def test_missing_projects_db_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TERMINAL_CWD", _acme_folder(tmp_path))
        from hermes_cli import projects_db as pdb
        monkeypatch.setattr(pdb, "projects_db_path", lambda: tmp_path / "no-such.db")
        from agent.agent_init import _resolve_agent_workspace
        assert _resolve_agent_workspace() == ""

    def test_archived_project_ignored(self, tmp_path, monkeypatch):
        from hermes_cli import projects_db as pdb
        db_file = tmp_path / "projects.db"
        monkeypatch.setattr(pdb, "projects_db_path", lambda: db_file)
        with pdb.connect_closing() as conn:
            pid = pdb.create_project(
                conn, name="acme", folders=[_acme_folder(tmp_path)]
            )
            pdb.archive_project(conn, pid)
        monkeypatch.setenv("TERMINAL_CWD", _acme_folder(tmp_path))
        from agent.agent_init import _resolve_agent_workspace
        assert _resolve_agent_workspace() == ""

    def test_profile_db_falls_back_to_default_db(self, tmp_path, monkeypatch):
        """Sub-agent profiles have their own (often empty) projects.db; the
        project is registered in the default profile's projects.db. Workspace
        must still resolve from the default profile's registry."""
        from hermes_cli import projects_db as pdb

        # Sub-agent profile's own projects.db: exists but EMPTY (no projects).
        profile_db = tmp_path / "profile-projects.db"
        with pdb.connect_closing(db_path=profile_db) as conn:
            pass  # create schema only
        monkeypatch.setattr(pdb, "projects_db_path", lambda: profile_db)

        # Default profile's projects.db: has the acme project. The function
        # resolves it as <default_home>/projects.db; default_home is patched
        # to tmp_path so the file must be tmp_path/projects.db.
        default_db = tmp_path / "projects.db"
        with pdb.connect_closing(db_path=default_db) as conn:
            pdb.create_project(
                conn,
                name="acme",
                folders=[_acme_folder(tmp_path)],
            )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home",
            lambda: str(tmp_path),
        )
        assert _resolve(_acme_folder(tmp_path), monkeypatch) == "acme"

    def test_profile_db_wins_ties_with_default_db(self, tmp_path, monkeypatch):
        """When both registries know the cwd, the active profile's project
        wins (scoped, more specific)."""
        from hermes_cli import projects_db as pdb

        profile_db = tmp_path / "profile-projects.db"
        with pdb.connect_closing(db_path=profile_db) as conn:
            pdb.create_project(
                conn,
                name="profile-acme",
                folders=[_acme_folder(tmp_path)],
            )
        monkeypatch.setattr(pdb, "projects_db_path", lambda: profile_db)

        default_db = tmp_path / "projects.db"
        with pdb.connect_closing(db_path=default_db) as conn:
            pdb.create_project(
                conn,
                name="default-acme",
                folders=[_acme_folder(tmp_path)],
            )
        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home",
            lambda: str(tmp_path),
        )
        assert _resolve(_acme_folder(tmp_path), monkeypatch) == "profile-acme"


class TestWorkspaceFeedsBankTemplate:
    def test_workspace_placeholder_resolves_when_in_project(
        self, tmp_path, monkeypatch, fake_projects_db
    ):
        """End-to-end: in the acme project, bank_id_template {workspace}
        resolves to acme instead of the old hardcoded fallback."""
        import json
        from plugins.memory.hindsight import HindsightMemoryProvider

        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "lead",
            "bank_id_template": "{workspace}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )
        monkeypatch.setenv("TERMINAL_CWD", _acme_folder(tmp_path))

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="lead",
            agent_workspace=_resolve(_acme_folder(tmp_path), monkeypatch),
        )
        assert p._bank_id == "acme"

    def test_workspace_placeholder_falls_back_outside_project(
        self, tmp_path, monkeypatch, fake_projects_db
    ):
        import json
        from plugins.memory.hindsight import HindsightMemoryProvider

        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "lead",
            "bank_id_template": "{workspace}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )
        outside = tmp_path / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("TERMINAL_CWD", str(outside))

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="lead",
            agent_workspace=_resolve(str(outside), monkeypatch),
        )
        # rendered template empty → falls back to configured bank_id
        assert p._bank_id == "lead"