"""Deep diagnostics distinguish the current gateway from its previous life."""

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import psutil
import pytest

from gateway import lifecycle_ledger, status
from hermes_cli import gateway, gateway_windows


@pytest.fixture
def lifecycle_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    identity = (os.getpid(), psutil.Process().create_time())

    def running_identity(pid_path):
        assert pid_path == tmp_path / "gateway.pid"
        return identity

    monkeypatch.setattr(status, "get_running_pid_identity_strict", running_identity)
    return tmp_path, identity


@pytest.mark.windows_only
@pytest.mark.parametrize(
    ("previous_unclean", "store", "passes"),
    [(False, "absent", True), (True, "absent", True),
     (True, "ok", True), (True, "corrupt", False)],
)
def test_deep_probe_preserves_previous_exit_and_integrity(
    lifecycle_home, capsys, previous_unclean, store, passes
):
    home, identity = lifecycle_home
    prior_pid = 0  # Invalid PID: the previous owner cannot still be alive.
    prior_started = "2026-01-01T00:00:00+00:00"
    if previous_unclean:
        sentinel = lifecycle_ledger.get_lifecycle_sentinel_path(home)
        sentinel.parent.mkdir()
        sentinel.write_text(json.dumps({
            "phase": "running", "pid": prior_pid,
            "start_time": 1.0, "started_at": prior_started,
        }), encoding="utf-8")
    if store == "ok":
        with sqlite3.connect(home / "state.db") as db:
            db.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY)")
    elif store == "corrupt":
        (home / "state.db").write_bytes(b"not a SQLite database")

    # The CLI emits start before GatewayRunner records the previous unclean life.
    gateway._make_exit_diag()("gateway.start")
    evidence = lifecycle_ledger.record_startup()
    diag_path = home / "logs" / "gateway-exit-diag.log"
    original_log = diag_path.read_bytes()
    if previous_unclean:
        assert evidence is not None
        assert (evidence["state_db_integrity"] in ("ok", "absent")) == passes
    else:
        assert evidence is None

    gateway_windows._print_deep_probes()

    probe = next(line for line in capsys.readouterr().out.splitlines() if "[6]" in line)
    assert ("PASS" in probe) == passes, probe
    assert f"pid={identity[0]}" in probe
    if previous_unclean:
        assert "WARNING" in probe
        assert "previous_unclean_exit" in probe
        assert f"prior_pid={prior_pid}" in probe
        assert prior_started in probe
        assert f"state_db_integrity={evidence['state_db_integrity']}" in probe
    assert diag_path.read_bytes() == original_log


@pytest.mark.windows_only
@pytest.mark.parametrize("case", [
    "missing-log", "empty-log", "malformed-json", "non-object", "missing-pid",
    "string-pid", "other-pid", "missing-ts", "malformed-ts", "naive-ts",
    "future-ts", "recycled-pid", "no-live-identity", "ambiguous-identity",
    "terminal", "unknown-tag", "missing-integrity", "failed-integrity",
])
def test_deep_probe_rejects_unverified_or_failed_lifecycle(
    lifecycle_home, monkeypatch, capsys, case
):
    home, identity = lifecycle_home
    gateway._make_exit_diag()("gateway.start")
    lifecycle_ledger.record_startup()
    diag_path = home / "logs" / "gateway-exit-diag.log"
    event = json.loads(diag_path.read_text(encoding="utf-8").splitlines()[-1])
    changes = {
        "string-pid": {"pid": str(identity[0])},
        "other-pid": {"pid": identity[0] + 1},
        "malformed-ts": {"ts": "not-a-time"},
        "naive-ts": {"ts": "2026-01-01T00:00:00"},
        "future-ts": {"ts": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()},
        "recycled-pid": {"ts": datetime.fromtimestamp(identity[1] - 10, timezone.utc).isoformat()},
        "terminal": {"tag": "gateway.finally"},
        "unknown-tag": {"tag": "gateway.unrecognized"},
        "missing-integrity": {"tag": "gateway.previous_unclean_exit"},
        "failed-integrity": {"tag": "gateway.previous_unclean_exit", "state_db_integrity": "check-failed"},
    }
    event.update(changes.get(case, {}))
    if case.startswith("missing-"):
        event.pop(case.removeprefix("missing-"), None)
    contents = {"empty-log": "", "malformed-json": "{", "non-object": "[]"}.get(
        case, json.dumps(event) + "\n"
    )
    diag_path.write_text(contents, encoding="utf-8")
    if case == "missing-log":
        diag_path.unlink()
    if case == "no-live-identity":
        monkeypatch.setattr(status, "get_running_pid_identity_strict", lambda path: None)
    elif case == "ambiguous-identity":
        def ambiguous(path):
            raise RuntimeError("gateway creation time is unavailable")
        monkeypatch.setattr(status, "get_running_pid_identity_strict", ambiguous)

    gateway_windows._print_deep_probes()

    probe = next(line for line in capsys.readouterr().out.splitlines() if "[6]" in line)
    assert "FAIL" in probe, probe
