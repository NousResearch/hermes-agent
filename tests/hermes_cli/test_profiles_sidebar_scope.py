"""The sidebar's profile scope, across both endpoints that serve it.

Two behaviors that only show up with more than one profile on disk:

* ``/api/profiles/sessions/sidebar`` must answer to one scope for all three of
  its slices. Cron and messaging ignoring it is what made a concrete profile
  show another profile's Telegram threads and cronjobs (#65710, #42651,
  #70629).
* ``/api/profiles/projects/tree`` must build each profile's tree from that
  profile's own state.db AND its own projects.db, and hand back ids that can
  coexist in one list.
"""

import pytest


@pytest.fixture
def profiles_on_disk(tmp_path, monkeypatch, _isolate_hermes_home):
    """An isolated default home plus one named profile, each with a state.db."""
    from hermes_cli import profiles
    from hermes_constants import get_hermes_home

    default_home = get_hermes_home()
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker"

    for home in (default_home, worker_home):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)

    return {"default": default_home, "worker": worker_home}


@pytest.fixture
def client(monkeypatch, profiles_on_disk):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    return c


def _seed_session(home, session_id, *, source, cwd=None, tokens=None, cost=None):
    """One session with a message, so it clears the sidebar's min_messages=1.

    ``cwd`` is what attaches it to a project — without one it lands in Home.
    ``tokens`` is an (input, output) pair; both it and ``cost`` are written
    straight to the row, the shape a finished turn leaves behind.
    """
    import sqlite3

    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session(session_id, source=source, cwd=str(cwd) if cwd else None)
        db.append_message(session_id=session_id, role="user", content="hi")
    finally:
        db.close()

    if tokens is None and cost is None:
        return

    conn = sqlite3.connect(home / "state.db")
    try:
        conn.execute(
            "UPDATE sessions SET input_tokens = ?, output_tokens = ?, estimated_cost_usd = ? WHERE id = ?",
            (*(tokens or (0, 0)), cost or 0.0, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_project(home, name, folder):
    from hermes_cli import projects_db

    with projects_db.connect_closing(db_path=home / "projects.db") as conn:
        return projects_db.create_project(conn, name=name, folders=[str(folder)])


def _slice_ids(payload, slice_name):
    return {row["id"] for row in payload[slice_name]["sessions"]}


class TestSidebarScope:

    def test_concrete_profile_sees_only_its_own_slices(self, client, profiles_on_disk):
        _seed_session(profiles_on_disk["default"], "default-chat", source="cli")
        _seed_session(profiles_on_disk["default"], "default-cron", source="cron")
        _seed_session(profiles_on_disk["default"], "default-telegram", source="telegram")
        _seed_session(profiles_on_disk["worker"], "worker-chat", source="cli")
        _seed_session(profiles_on_disk["worker"], "worker-cron", source="cron")
        _seed_session(profiles_on_disk["worker"], "worker-telegram", source="telegram")

        payload = client.get(
            "/api/profiles/sessions/sidebar",
            params={"recents_profile": "worker", "recents_exclude": "cron,telegram", "messaging_exclude": "cli,cron"},
        ).json()

        assert payload["errors"] == []
        assert _slice_ids(payload, "recents") == {"worker-chat"}
        # The bug: these two used to come back with the default profile's rows
        # folded in, whatever scope the sidebar asked for.
        assert _slice_ids(payload, "cron") == {"worker-cron"}
        assert _slice_ids(payload, "messaging") == {"worker-telegram"}

    def test_all_scope_still_spans_every_profile(self, client, profiles_on_disk):
        _seed_session(profiles_on_disk["default"], "default-telegram", source="telegram")
        _seed_session(profiles_on_disk["worker"], "worker-telegram", source="telegram")

        payload = client.get(
            "/api/profiles/sessions/sidebar",
            params={"recents_profile": "all", "messaging_exclude": "cli,cron"},
        ).json()

        assert _slice_ids(payload, "messaging") == {"default-telegram", "worker-telegram"}
        assert {row["profile"] for row in payload["messaging"]["sessions"]} == {"default", "worker"}


class TestCrossProfileProjectTree:

    def test_one_folder_worked_in_by_two_profiles_is_one_project(self, client, profiles_on_disk, tmp_path):
        # A folder is a folder no matter who opened it. Two profiles working the
        # same checkout is the normal case (that's the point of profiles), so it
        # heads ONE group carrying both their sessions — not one group each.
        shared = tmp_path / "repos" / "shared"
        shared.mkdir(parents=True)

        for name, home in profiles_on_disk.items():
            _seed_session(home, f"{name}-chat", source="cli", cwd=shared)
            _seed_project(home, "Shared", shared)

        payload = client.get("/api/profiles/projects/tree").json()

        assert payload["errors"] == []

        declared = [project for project in payload["projects"] if not project["isNoProject"]]
        assert [project["path"] for project in declared] == [str(shared)]
        assert declared[0]["sessionCount"] == 2

    def test_group_totals_add_up_the_sessions_the_group_counts(self, client, profiles_on_disk, tmp_path):
        # A header total is only meaningful if it covers exactly what the header
        # says it counts — a project's totals span every profile working it, the
        # same set `sessionCount` reports.
        shared = tmp_path / "repos" / "shared"
        shared.mkdir(parents=True)

        for name, home in profiles_on_disk.items():
            _seed_session(home, f"{name}-chat", source="cli", cwd=shared, tokens=(100, 20), cost=0.25)
            _seed_project(home, "Shared", shared)

        payload = client.get("/api/profiles/projects/tree").json()
        project = next(p for p in payload["projects"] if not p["isNoProject"])

        assert project["sessionCount"] == 2
        assert project["totalTokens"] == 240
        assert project["totalCostUsd"] == pytest.approx(0.5)

    def test_profile_usage_covers_sessions_past_the_window(self, client, profiles_on_disk):
        # The whole point of aggregating in SQL: the total must not be a sum of
        # whichever page the sidebar happens to have asked for.
        for index in range(3):
            _seed_session(
                profiles_on_disk["worker"], f"worker-{index}", source="cli", tokens=(10, 5), cost=1.5
            )

        payload = client.get(
            "/api/profiles/sessions/sidebar", params={"recents_profile": "all", "recents_limit": 1}
        ).json()

        assert payload["recents"]["profiles_usage"]["worker"] == {
            "cost_usd": pytest.approx(4.5),
            "tokens": 45,
        }

    def test_home_is_one_bucket_across_profiles(self, client, profiles_on_disk):
        # Every profile builds its own unowned-sessions bucket. Merging by id is
        # what keeps the sidebar from stacking N identical "Home" rows.
        for name, home in profiles_on_disk.items():
            _seed_session(home, f"{name}-chat", source="cli")

        payload = client.get("/api/profiles/projects/tree").json()

        homes = [project for project in payload["projects"] if project["isNoProject"]]
        assert len(homes) == 1
        assert homes[0]["sessionCount"] == 2

    def test_each_profile_contributes_its_own_projects_db(self, client, profiles_on_disk, tmp_path):
        """Proves the per-profile scoping, not just that two trees got merged.

        The builder reads projects.db, the repo-scan policy and the junk
        filters through ``get_hermes_home()``. If the fan-out failed to rebind
        it per profile, every tree would come back describing whichever home
        the process happens to be running as.
        """
        for name in profiles_on_disk:
            (tmp_path / "repos" / f"only-{name}").mkdir(parents=True)
            _seed_session(profiles_on_disk[name], f"{name}-chat", source="cli")
            _seed_project(profiles_on_disk[name], f"Only {name}", tmp_path / "repos" / f"only-{name}")

        payload = client.get("/api/profiles/projects/tree").json()

        labels = {project["label"] for project in payload["projects"] if not project["isNoProject"]}

        assert labels == {"Only default", "Only worker"}

    def test_a_profile_that_cannot_be_read_does_not_sink_the_rest(
        self, client, profiles_on_disk, tmp_path, monkeypatch
    ):
        (tmp_path / "repos" / "healthy").mkdir(parents=True)
        # A state.db has to exist for a profile to be visited at all.
        for name, home in profiles_on_disk.items():
            _seed_session(home, f"{name}-chat", source="cli")
        _seed_project(profiles_on_disk["default"], "Healthy", tmp_path / "repos" / "healthy")

        from tui_gateway import server as gateway_server

        real_build = gateway_server._build_project_tree

        def explode_for_worker(db, **kwargs):
            from hermes_constants import get_hermes_home

            if get_hermes_home().name == "worker":
                raise RuntimeError("worker store is unreadable")

            return real_build(db, **kwargs)

        monkeypatch.setattr(gateway_server, "_build_project_tree", explode_for_worker)

        payload = client.get("/api/profiles/projects/tree").json()

        assert [error["profile"] for error in payload["errors"]] == ["worker"]
        # The healthy profile's tree still lands; only the broken one drops out.
        assert "Healthy" in [project["label"] for project in payload["projects"]]
        assert [project["sessionCount"] for project in payload["projects"] if project["isNoProject"]] == [1]
