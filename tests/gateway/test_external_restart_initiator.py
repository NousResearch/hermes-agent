import logging
import os

from gateway import run as gateway_run
from gateway import status


def test_external_restart_initiator_round_trips_into_shutdown_log(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)

    status.write_external_restart_request(
        os.getpid(),
        argv=["hermes", "gateway", "restart", "--token", "super-secret"],
    )
    marker = tmp_path / ".gateway-restart-request.json"
    persisted = marker.read_text(encoding="utf-8")
    assert "super-secret" not in persisted
    assert "--token" not in persisted

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        gateway_run._log_external_restart_initiator()

    assert not marker.exists()
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert f"caller_pid={os.getpid()}" in message
    assert f"caller_ppid={os.getppid()}" in message
    assert "argv_classification=gateway_restart" in message
    assert "request_id=" in message
    assert "requested_at=" in message
    assert "super-secret" not in message


def test_external_restart_consume_preserves_marker_written_after_claim(
    tmp_path, monkeypatch
):
    # Given: a restart marker and a newer writer that runs immediately after
    # the consumer reads the claimed marker.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 42)
    status.write_external_restart_request(
        os.getpid(), argv=["hermes", "gateway", "restart"]
    )
    marker = tmp_path / ".gateway-restart-request.json"
    real_read_json_file = status._read_json_file
    wrote_new_marker = False
    claimed_paths = []

    def read_then_write_new_marker(path):
        nonlocal wrote_new_marker
        claimed_paths.append(path)
        record = real_read_json_file(path)
        if not wrote_new_marker:
            wrote_new_marker = True
            status.write_external_restart_request(
                os.getpid(), argv=["hermes", "update"]
            )
        return record

    monkeypatch.setattr(status, "_read_json_file", read_then_write_new_marker)

    # When: the original marker is consumed.
    original = status.consume_external_restart_request_for_self()

    # Then: the newer canonical marker survives and remains consumable.
    assert original is not None
    assert original.argv_classification == "gateway_restart"
    assert marker.exists()
    newer = status.consume_external_restart_request_for_self()
    assert newer is not None
    assert newer.argv_classification == "update"
    assert not marker.exists()
    assert len(claimed_paths) == 2
    assert claimed_paths[0].parent == marker.parent
    assert claimed_paths[1].parent == marker.parent
    assert claimed_paths[0] != claimed_paths[1]
