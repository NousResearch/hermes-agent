import json

import pytest


class _IntegrityConnection:
    def __init__(self, messages: list[str], statements: list[str]) -> None:
        self._messages = messages
        self._statements = statements

    def execute(self, statement: str):
        self._statements.append(statement)
        return self

    def fetchall(self):
        return [(message,) for message in self._messages]

    def close(self) -> None:
        return None


def test_post_restart_sweep_checks_root_and_profiles_with_full_integrity(monkeypatch, tmp_path, capsys):
    from hermes_cli import update_cmd, update_cmd_maint, update_receipt

    root = tmp_path / "root"
    work = tmp_path / "profiles" / "work"
    for home in (root, work):
        home.mkdir(parents=True)
        (home / "state.db").write_bytes(b"SQLite format 3\0fixture")
        (home / "gateway_state.json").write_text(json.dumps({"pid": 123}), encoding="utf-8")

    statements: list[str] = []

    def connect(target, **kwargs):
        messages = ["ok"] if str(root) in target else ["max rootpage (91) disagrees with header (89)"]
        return _IntegrityConnection(messages, statements)

    recorded = []
    monkeypatch.setattr(update_cmd_maint.sqlite3, "connect", connect)
    monkeypatch.setattr(update_receipt, "_profile_homes", lambda: [("default", root), ("work", work)])
    monkeypatch.setattr(update_receipt, "record_state_db_integrity", recorded.append)

    results = update_cmd._verify_state_dbs_after_restart()

    assert [(result["profile"], result["status"]) for result in results] == [
        ("default", "ok"),
        ("work", "failed"),
    ]
    assert statements == ["PRAGMA integrity_check", "PRAGMA integrity_check"]
    assert recorded == [results]
    persisted = json.loads((work / "gateway_state.json").read_text(encoding="utf-8"))
    assert persisted["state_db_integrity"]["status"] == "failed"
    assert persisted["state_db_integrity"]["source"] == "post_update_restart"
    assert "Post-restart state.db integrity verification FAILED" in capsys.readouterr().out


def test_fleet_verifier_runs_integrity_after_restart_probe_and_fails_closed(monkeypatch):
    from hermes_cli import main, update_cmd, update_cmd_fleet, update_receipt

    restart = update_cmd_fleet._GatewayRestartOutcome(
        incomplete=False,
        phase_errors=[],
        pre_restart_gateway_pids=[],
        restarted_services=[],
        failed_or_stale_units=[],
        relaunched_profiles=[],
        externally_supervised_profiles=[],
        killed_pids=set(),
    )
    events: list[str] = []
    finalized: list[str] = []
    cleared: list[bool] = []
    monkeypatch.setattr(update_cmd_fleet, "_print_legacy_units_warning", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *args, **kwargs: None)
    monkeypatch.setattr(update_cmd, "_surviving_pre_update_serve_runtimes", lambda plan: [])
    monkeypatch.setattr(update_cmd, "_warn_stale_serve_runtimes", lambda rows: None)
    monkeypatch.setattr(main, "_fleet_probe_expected_runtimes", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        update_cmd_fleet,
        "_collect_fleet_snapshot",
        lambda *args, **kwargs: events.append("fleet") or [],
    )
    monkeypatch.setattr(
        update_cmd,
        "_verify_state_dbs_after_restart",
        lambda: events.append("integrity") or [
            {"profile": "default", "status": "failed", "detail": "damaged", "checked_at": "now"}
        ],
    )
    monkeypatch.setattr(update_cmd, "_record_update_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(update_receipt, "print_fleet_version_matrix", lambda rows: False)
    monkeypatch.setattr(
        update_receipt,
        "finalize_update_receipt",
        lambda outcome, fleet=None: finalized.append(outcome),
    )
    monkeypatch.setattr(update_cmd_fleet, "_clear_fleet_restart_pending_marker", lambda: cleared.append(True))

    with pytest.raises(SystemExit) as exc_info:
        update_cmd_fleet._verify_fleet_after_update(
            restart,
            _pre_update_plan=None,
            _windows_gateway_resume=None,
            node_failures=[],
            update_complete=True,
        )

    assert exc_info.value.code == 1
    assert events == ["fleet", "integrity"]
    assert finalized == ["partial"]
    assert cleared == []