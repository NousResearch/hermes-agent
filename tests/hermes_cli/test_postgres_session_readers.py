"""Dashboard reads must use the configured remote store even without a local state.db."""

from hermes_cli.web_routers import profiles, status
from hermes_cli.web_server_sessions import _open_session_db_at_path


def test_sqlite_repair_refuses_postgres_without_touching_retained_file(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    import hermes_state
    from hermes_cli.sessions_cmd import _cmd_repair

    path = tmp_path / "state.db"
    path.write_bytes(b"retained SQLite data")
    (tmp_path / "config.yaml").write_text("database:\n  backend: postgres\n")
    (tmp_path / ".env").write_text("HERMES_DATABASE_URL=postgresql://unreachable@127.0.0.1:1/unreachable\n")
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", path)

    assert _cmd_repair(Namespace(check_only=False, no_backup=False)) == 1
    assert "Use PostgreSQL administration tools" in capsys.readouterr().out
    assert path.read_bytes() == b"retained SQLite data"


def test_postgres_cli_stats_and_optimize_report_native_storage(postgres_db, capsys):
    from argparse import Namespace
    from hermes_cli.sessions_cmd import _cmd_optimize, _cmd_stats

    postgres_db.create_session("maintenance-reader", source="cli")
    postgres_db.append_message("maintenance-reader", "user", "Searchable maintenance example")
    _cmd_stats(postgres_db, Namespace())
    stats = capsys.readouterr().out
    assert "Total sessions: 1" in stats
    assert f"Database size: {postgres_db.logical_size_bytes() / (1024 * 1024):.1f} MB" in stats

    assert not _cmd_optimize(postgres_db, Namespace())
    optimized = capsys.readouterr().out
    assert "PostgreSQL search index optimized." in optimized
    assert "VACUUM" not in optimized
    assert f"Database size: {postgres_db.logical_size_bytes() / (1024 * 1024):.1f} MB" in optimized
    assert postgres_db.get_messages("maintenance-reader")[0]["content"] == "Searchable maintenance example"


def test_doctor_reads_postgres_and_does_not_repair_a_local_file(postgres_db, postgres_home, monkeypatch, capsys):
    from hermes_cli import doctor
    from hermes_cli.doctor_state import _check_state_db

    postgres_db.create_session("doctor-reader", source="cli")
    monkeypatch.setattr(doctor, "HERMES_HOME", postgres_home)
    finding = _check_state_db(should_fix=True)

    assert finding.issues == finding.manual_issues == []
    assert finding.fixed == 0
    assert "PostgreSQL session database is reachable (1 sessions)" in capsys.readouterr().out
    assert not (postgres_home / "state.db").exists()


def test_doctor_reports_uninitialized_postgres_without_sqlite_repair(postgres_home, monkeypatch, capsys):
    from hermes_cli import doctor
    from hermes_cli.doctor_state import _check_state_db

    monkeypatch.setattr(doctor, "HERMES_HOME", postgres_home)
    finding = _check_state_db(should_fix=True)

    assert finding.manual_issues
    assert finding.fixed == 0
    assert "PostgreSQL session database is unavailable" in capsys.readouterr().out
    assert not (postgres_home / "state.db").exists()


def test_approval_suggestions_read_postgres_and_exclude_denied_calls(postgres_db, postgres_home, capsys):
    import json
    from argparse import Namespace
    from hermes_cli.approvals_suggest import scan_approval_history, suggest_command

    postgres_db.create_session("approval-reader", source="cli")
    for call_id, command, result in [
        ("approved", "git push --force origin feature", "ok: done"),
        ("denied", "git push --force origin other", "BLOCKED: User denied"),
    ]:
        postgres_db.append_message("approval-reader", "assistant", "", tool_calls=[{
            "id": call_id, "type": "function", "function": {
                "name": "terminal", "arguments": json.dumps({"command": command}),
            },
        }])
        postgres_db.append_message("approval-reader", "tool", result, tool_call_id=call_id)

    records = scan_approval_history(postgres_home / "state.db")
    assert [command for command, _ in records] == ["git push --force origin feature"]
    assert suggest_command(Namespace(days=90, min_count=1, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["proposals"][0]["pattern"] == "git push *"
    assert not (postgres_home / "state.db").exists()


def test_usage_analytics_reads_postgres_daily_totals_and_auxiliary_usage(postgres_db, postgres_home):
    from datetime import datetime, timezone
    from hermes_cli.web_routers.analytics import _get_usage_analytics

    postgres_db.create_session("analytics-reader", source="cli", model="main-model")
    postgres_db.append_message("analytics-reader", "user", "Usage report")
    postgres_db.record_auxiliary_usage("analytics-reader", model="vision-model", task="vision",
                                       input_tokens=500, output_tokens=50)

    report = _get_usage_analytics(days=1)

    assert report["totals"]["total_sessions"] == 1
    assert report["daily"][0]["day"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert {row["model"] for row in report["by_model"]} == {"main-model", "vision-model"}
    assert report["by_task"][0]["task"] == "vision"
    assert report["by_task"][0]["input_tokens"] == 500


def _conversation(db):
    db.create_session("postgres-reader", source="cli")
    db.append_message("postgres-reader", "user", "A conversation in PostgreSQL")


def test_dashboard_resolves_postgres_model_switch_descendants(postgres_db, postgres_home):
    from hermes_cli.web_server_sessions import _session_latest_descendant

    postgres_db.create_session("before-model-switch", source="cli")
    postgres_db.create_session("after-model-switch", source="cli", parent_session_id="before-model-switch")
    with _open_session_db_at_path(postgres_home / "state.db", read_only=True) as reader:
        latest, lineage = _session_latest_descendant("before-model-switch", reader)

    assert latest == "after-model-switch"
    assert lineage == ["before-model-switch", "after-model-switch"]


def test_dashboard_read_only_open_reads_postgres_without_bootstrapping_sqlite(postgres_db, postgres_home):
    _conversation(postgres_db)
    db_path = postgres_home / "state.db"
    assert not db_path.exists()

    with _open_session_db_at_path(db_path, read_only=True) as reader:
        assert reader.backend == "postgres"
        assert reader.read_only
        assert reader.get_session("postgres-reader")["message_count"] == 1

    assert not db_path.exists()


def test_profile_reader_uses_target_credentials_and_preserves_remote_history(
    postgres_db, postgres_home, tmp_path, monkeypatch,
):
    _conversation(postgres_db)
    caller_home = tmp_path / "caller"
    caller_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller_home))
    monkeypatch.setenv("HERMES_DATABASE_URL", "postgresql://wrong-profile@127.0.0.1:1/wrong")
    errors = []

    session = profiles._read_profile_db(
        "target", postgres_home, errors, lambda db: db.get_session("postgres-reader"),
    )

    assert session["message_count"] == 1
    assert errors == []
    assert not (postgres_home / "state.db").exists()


def test_sidebar_cache_invalidates_on_postgres_changes(postgres_db, postgres_home, monkeypatch):
    _conversation(postgres_db)
    monkeypatch.setattr(profiles, "_profile_targets", lambda *args, **kwargs: [("default", postgres_home)])
    profiles._sidebar_profile_cache_clear()
    try:
        sidebar = profiles.get_profiles_sessions_sidebar.__wrapped__
        before = profiles._sidebar_db_fingerprint(postgres_home / "state.db")
        first = sidebar()
        assert [s["id"] for s in first["recents"]["sessions"]] == ["postgres-reader"]

        postgres_db.set_session_title("postgres-reader", "Changed remotely")
        after = profiles._sidebar_db_fingerprint(postgres_home / "state.db")
        assert after != before
        second = sidebar()
        assert second["recents"]["sessions"][0]["title"] == "Changed remotely"
        assert second["errors"] == []
    finally:
        profiles._sidebar_profile_cache_clear()


def test_status_counts_postgres_sessions_without_a_local_file(postgres_db, postgres_home):
    _conversation(postgres_db)
    assert not (postgres_home / "state.db").exists()
    assert status._count_status_active_sessions() == 1
