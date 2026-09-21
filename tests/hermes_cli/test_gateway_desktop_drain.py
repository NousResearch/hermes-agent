from pathlib import Path


def test_wait_for_desktop_drain_marks_every_profile_and_waits_for_two_idle_samples():
    from hermes_cli.gateway_desktop_drain import wait_for_desktop_drain

    events = []
    snapshots = iter([(1, 2), (0, 1), (0, 0), (0, 0)])
    homes = [Path("/profiles/default"), Path("/profiles/research")]

    result = wait_for_desktop_drain(
        homes=homes,
        snapshot=lambda: next(snapshots),
        write_marker=lambda home: events.append(("mark", home)),
        sleep=lambda seconds: events.append(("sleep", seconds)),
        poll_interval=0.25,
        idle_samples_required=2,
    )

    assert result.gateway_agents == 0
    assert result.kanban_workers == 0
    assert events[:2] == [("mark", homes[0]), ("mark", homes[1])]
    assert events.count(("sleep", 0.25)) == 3


def test_wait_for_desktop_drain_refreshes_markers_during_long_work():
    from hermes_cli.gateway_desktop_drain import wait_for_desktop_drain

    marks = []
    now = iter([0.0, 0.0, 31.0, 31.0])
    snapshots = iter([(0, 1), (0, 1), (0, 0)])

    wait_for_desktop_drain(
        homes=[Path("/profiles/default")],
        snapshot=lambda: next(snapshots),
        write_marker=marks.append,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(now),
        refresh_interval=30.0,
        idle_samples_required=1,
    )

    assert marks == [Path("/profiles/default"), Path("/profiles/default")]


def test_drain_all_desktop_work_scopes_single_profile_to_current_home(monkeypatch, tmp_path):
    import hermes_constants
    import hermes_cli.gateway_desktop_drain as drain

    other_home = tmp_path / "other"
    current_home = tmp_path / "current"
    seen = []
    monkeypatch.setattr(
        drain,
        "desktop_profile_homes",
        lambda: (current_home, other_home),
    )
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: current_home)
    monkeypatch.setattr(
        drain,
        "wait_for_desktop_drain",
        lambda **kwargs: seen.append(tuple(kwargs["homes"])) or drain.DesktopDrainSnapshot(0, 0),
    )

    drain.drain_all_desktop_work(all_profiles=False)

    assert seen == [(current_home,)]


def test_read_desktop_drain_snapshot_filters_shared_board_workers_by_profile(
    monkeypatch, tmp_path
):
    import hermes_constants
    import hermes_cli.gateway_desktop_drain as drain
    from hermes_cli import kanban_db as kb
    import hermes_cli.kanban_db_connect as kbc

    current_home = tmp_path / "current"
    other_home = tmp_path / "other"
    db_path = tmp_path / "kanban.db"
    db_path.touch()
    queries = []

    class Connection:
        def execute(self, query):
            queries.append(query)
            return self

        def fetchall(self):
            return [
                {"assignee": "current", "worker_pid": 101},
                {"assignee": "other", "worker_pid": 202},
            ]

        def close(self):
            pass

    monkeypatch.setattr(
        hermes_constants,
        "profile_name_for_home",
        lambda home: "current" if Path(home) == current_home else "other",
    )
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "gateway.status.runtime_status_pid_is_live",
        lambda _runtime: False,
    )
    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{"slug": "default"}])
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: db_path)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(kbc, "connect", lambda board=None: Connection())

    result = drain.read_desktop_drain_snapshot((current_home,))

    assert result == (0, 1)
    assert any("assignee" in query for query in queries)


def test_all_profile_snapshot_keeps_live_workers_for_deleted_profiles(
    monkeypatch, tmp_path
):
    import hermes_cli.gateway_desktop_drain as drain
    from hermes_cli import kanban_db as kb
    import hermes_cli.kanban_db_connect as kbc

    current_home = tmp_path / "current"
    db_path = tmp_path / "kanban.db"
    db_path.touch()

    class Connection:
        def execute(self, _query):
            return self

        def fetchall(self):
            return [{"assignee": "deleted-profile", "worker_pid": 303}]

        def close(self):
            pass

    monkeypatch.setattr("gateway.status.read_runtime_status", lambda **_kwargs: None)
    monkeypatch.setattr("gateway.status.runtime_status_pid_is_live", lambda _runtime: False)
    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{"slug": "default"}])
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: db_path)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(kbc, "connect", lambda board=None: Connection())

    assert drain.read_desktop_drain_snapshot(
        (current_home,), filter_workers_by_profile=False
    ) == (0, 1)
