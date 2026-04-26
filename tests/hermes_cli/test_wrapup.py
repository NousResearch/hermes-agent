import json
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_cli import wrapup


@pytest.fixture()
def session_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    yield db
    db.close()


def test_runtime_record_round_trip(tmp_path):
    hermes_home = tmp_path / ".hermes"

    record = wrapup.write_runtime_record(
        session_id="20260426_010203_abcdef",
        status="idle",
        hermes_home=hermes_home,
        cwd=Path("/tmp/project"),
        pid=12345,
        tty="/dev/pts/7",
    )

    records = wrapup.load_runtime_records(hermes_home, pid_alive=lambda pid: True)

    assert len(records) == 1
    assert records[0].session_id == "20260426_010203_abcdef"
    assert records[0].status == "idle"
    assert records[0].pid == 12345
    assert record.path == records[0].path


def test_wrap_up_skips_runtime_records_without_confident_session_id(tmp_path, session_db):
    hermes_home = tmp_path / ".hermes"
    run_dir = hermes_home / "wrap-up" / "runs" / "test-run"
    records = [
        wrapup.RuntimeRecord(
            pid=111,
            session_id=None,
            status="idle",
            cwd="/tmp/project",
            tty="/dev/pts/1",
            cmdline=["hermes"],
            path=hermes_home / "runtime" / "cli_sessions" / "111.json",
        )
    ]

    manifest = wrapup.wrap_up_records(
        records,
        session_db=session_db,
        hermes_home=hermes_home,
        run_id="test-run",
        run_dir=run_dir,
        close_session=lambda record: "closed",
        wait_for_idle=lambda record, timeout_seconds: record,
    )

    assert manifest["sessions"][0]["close_status"] == "skipped_no_confident_session_id"
    assert not (run_dir / "exports").exists()


def test_wrap_up_exports_and_closes_confident_idle_session(tmp_path, session_db):
    hermes_home = tmp_path / ".hermes"
    session_db.create_session("s1", source="cli", model="test")
    session_db.append_message("s1", "user", "hello")
    closed = []
    records = [
        wrapup.RuntimeRecord(
            pid=222,
            session_id="s1",
            status="idle",
            cwd="/tmp/project",
            tty="/dev/pts/2",
            cmdline=["hermes"],
            path=hermes_home / "runtime" / "cli_sessions" / "222.json",
        )
    ]

    manifest = wrapup.wrap_up_records(
        records,
        session_db=session_db,
        hermes_home=hermes_home,
        run_id="test-run",
        run_dir=hermes_home / "wrap-up" / "runs" / "test-run",
        close_session=lambda record: closed.append(record.session_id) or "closed_gracefully",
        wait_for_idle=lambda record, timeout_seconds: record,
    )

    entry = manifest["sessions"][0]
    assert entry["session_id"] == "s1"
    assert entry["close_status"] == "closed_gracefully"
    assert closed == ["s1"]
    export_path = Path(entry["export_path"])
    assert export_path.exists()
    exported = json.loads(export_path.read_text())
    assert exported["id"] == "s1"
    assert exported["messages"][0]["content"] == "hello"
    assert json.loads((hermes_home / "wrap-up" / "latest.json").read_text())["run_id"] == "test-run"


def test_wrap_up_does_not_wait_on_controller_session(tmp_path, session_db):
    hermes_home = tmp_path / ".hermes"
    session_db.create_session("controller", source="cli", model="test")
    waited = []
    closed = []
    records = [
        wrapup.RuntimeRecord(
            pid=333,
            session_id="controller",
            status="active",
            cwd="/tmp/project",
            tty="/dev/pts/3",
            cmdline=["hermes"],
            path=hermes_home / "runtime" / "cli_sessions" / "333.json",
        )
    ]

    manifest = wrapup.wrap_up_records(
        records,
        session_db=session_db,
        hermes_home=hermes_home,
        run_id="test-run",
        run_dir=hermes_home / "wrap-up" / "runs" / "test-run",
        close_session=lambda record: closed.append(record.session_id) or "closed_gracefully",
        wait_for_idle=lambda record, timeout_seconds: waited.append(record.session_id) or record,
        controller_pids={333},
    )

    assert waited == []
    assert closed == ["controller"]
    assert manifest["sessions"][0]["controller_session"] is True


def test_continue_sessions_launches_at_most_five_and_prints_remaining(tmp_path):
    hermes_home = tmp_path / ".hermes"
    manifest = {
        "run_id": "test-run",
        "sessions": [
            {"session_id": f"s{i}", "close_status": "closed_gracefully", "resume_command": f"hermes --resume s{i}"}
            for i in range(7)
        ],
    }
    latest = hermes_home / "wrap-up" / "latest.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(json.dumps(manifest))
    launched = []

    result = wrapup.continue_sessions(
        hermes_home=hermes_home,
        launcher=lambda cmd: launched.append(cmd) or True,
        max_auto_open=5,
    )

    assert launched == [f"hermes --resume s{i}" for i in range(5)]
    assert result["launched"] == [f"hermes --resume s{i}" for i in range(5)]
    assert result["manual"] == [f"hermes --resume s{i}" for i in range(5, 7)]


def test_resume_candidates_build_safe_commands_from_session_id(tmp_path):
    manifest = {
        "sessions": [
            {
                "session_id": "s3;touch /tmp/pwned",
                "close_status": "closed_after_timeout",
                "resume_command": "malicious command",
            },
        ]
    }

    assert wrapup.resume_commands_from_manifest(manifest) == ["hermes --resume 's3;touch /tmp/pwned'"]


def test_resume_candidates_ignore_skipped_sessions(tmp_path):
    manifest = {
        "sessions": [
            {"session_id": "s1", "close_status": "skipped_no_confident_session_id"},
            {"session_id": "s2", "close_status": "failed"},
            {"session_id": "s3", "close_status": "closed_after_timeout"},
        ]
    }

    assert wrapup.resume_commands_from_manifest(manifest) == ["hermes --resume s3"]
