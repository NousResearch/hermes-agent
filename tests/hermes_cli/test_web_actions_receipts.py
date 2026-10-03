"""Durable dashboard update results must belong to a completed update run."""

import json
import os
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import web_server_gateway
from hermes_cli.web_routers import actions


ACTION_ID = "c" * 32
FINISHED_AT = "2026-08-17T11:20:00+00:00"


@pytest.fixture
def status_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    logs = home / "logs"
    receipts = logs / "update_receipts"
    receipts.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", logs)
    for name in ("_ACTION_PROCS", "_ACTION_RESULTS", "_ACTION_COMMANDS", "_ACTION_IDS"):
        monkeypatch.setattr(web_server_gateway, name, {})
    (logs / "update.log").write_text(
        "=== hermes update started 2026-08-17T11:19:35 ===\n"
        f"=== hermes-update completed {ACTION_ID} ===\n",
        encoding="utf-8",
    )
    app = FastAPI()
    app.include_router(actions.status_router)
    with TestClient(app) as client:
        yield client, receipts


def write_receipt(directory, receipt, filename="latest.json"):
    (directory / filename).write_text(json.dumps(receipt), encoding="utf-8")


@pytest.fixture
def owned_route_worker(status_env, tmp_path, monkeypatch):
    """Only replace the update payload; admission, spawn, streams and status are real."""
    import socket
    from pathlib import Path
    from hermes_cli import _launchers, image_provenance, web_server

    client, receipts = status_env
    client.app.include_router(actions.router)
    root = tmp_path / "checkout"
    root.mkdir()
    (root / ".git").mkdir()
    monkeypatch.setattr(web_server, "PROJECT_ROOT", root)
    monkeypatch.setattr(image_provenance, "IMAGE_PROVENANCE_PATH", tmp_path / "image-provenance.json")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(10)
    script = (
        "import json, os, socket, sys; "
        "s = socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=10); "
        "print('owned stdout ready', flush=True); "
        "print('owned stderr ready', file=sys.stderr, flush=True); "
        "s.sendall((json.dumps({'pid': os.getpid(), 'action_id': os.environ['HERMES_ACTION_ID']}) + '\\n').encode()); "
        "s.settimeout(30); assert s.recv(1) == b'x'; "
        "print('owned stdout released', flush=True); "
        "print('owned stderr released', file=sys.stderr, flush=True)"
    )
    launches = []

    def inert_command(project_root, subcommand):
        assert project_root == root and subcommand == ["update"]
        launches.append(tuple(subcommand))
        return [sys.executable, "-u", "-c", script, str(listener.getsockname()[1])]

    monkeypatch.setattr(_launchers, "runtime_command", inert_command)
    proc = connection = None
    try:
        first = client.post("/api/hermes/update")
        assert first.status_code == 200 and first.json()["ok"] is True
        proc = web_server_gateway._ACTION_PROCS["hermes-update"]
        connection, _ = listener.accept()
        connection.settimeout(10)
        ready = json.loads(connection.makefile("rb").readline())
        assert ready == {"pid": proc.pid, "action_id": first.json()["action_id"]}
        assert proc.poll() is None
        log = receipts.parent / "hermes-update.log"
        assert "owned stdout ready" in log.read_text()
        assert "owned stderr ready" in log.read_text()
        print(f"owned worker ready pid={proc.pid} action_id={ready['action_id']}")

        def release_worker():
            nonlocal connection
            connection.sendall(b"x")
            connection.close()
            connection = None
            assert proc.wait(timeout=10) == 0

        yield client, receipts, root, proc, ready["action_id"], launches, release_worker
    finally:
        if proc is None:
            proc = web_server_gateway._ACTION_PROCS.get("hermes-update")
        try:
            if connection is not None:
                connection.sendall(b"x")
        finally:
            if connection is not None:
                connection.close()
            listener.close()
            if proc is not None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
                assert proc.returncode == 0, "owned worker did not exit on release"
                assert not Path(f"/proc/{proc.pid}").exists(), "owned worker was not reaped"
                log = receipts.parent / "hermes-update.log"
                assert "owned stdout released" in log.read_text()
                assert "owned stderr released" in log.read_text()
                print(f"owned worker reaped pid={proc.pid} exit={proc.returncode}")


def set_route_refusal(gate, root, receipts, monkeypatch):
    """Exercise each real route gate using owned filesystem facts, not canned decisions."""
    from hermes_cli import image_provenance, update_contract, web_server_files

    if gate == "commit-build":
        (root / "install-stamp.json").write_text(json.dumps({"source": "commit-build"}))
        assert actions.is_commit_build(root) is True
        return "commit-build", False
    if gate == "managed-root":
        monkeypatch.setattr(web_server_files, "_HOSTED_MANAGED_FILES_ROOT", receipts.parent.parent)
        assert web_server_files._dashboard_local_update_managed_externally() is True
        return "dashboard_update_managed_externally", False
    marker = image_provenance.IMAGE_PROVENANCE_PATH
    marker.write_text(json.dumps({"schema": 1, "deployment_kind": "image", "manager": "docker"})
                      if gate == "image-marker" else "{")
    refusal = update_contract.evaluate_update_admission(root)
    assert refusal is not None and refusal.code == gate
    return "docker_update_unsupported", True


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("gate", ["commit-build", "managed-root", "image-marker", "image-marker-invalid"])
def test_second_update_reuses_live_worker_before_commit_build_refusal(owned_route_worker, monkeypatch, gate):
    from pathlib import Path

    client, receipts, root, proc, action_id, launches, _ = owned_route_worker
    admission = receipts.parent / "hermes-update-action.json"
    log = receipts.parent / "hermes-update.log"
    original_record, original_log = admission.read_bytes(), log.read_bytes()
    original_command = web_server_gateway._ACTION_COMMANDS["hermes-update"]
    fds = [Path(f"/proc/{proc.pid}/fd/{fd}") for fd in (1, 2)]
    original_streams = [path.readlink() for path in fds]
    set_route_refusal(gate, root, receipts, monkeypatch)

    second = client.post("/api/hermes/update")
    status = client.get("/api/actions/hermes-update/status", params={"action_id": action_id}).json()
    assert proc.poll() is None
    assert status["state"] == "running", (second.json(), status)
    assert status["action_id"] == action_id and status["pid"] == proc.pid
    assert status["running"] is True and status["exit_code"] is None and "receipt" not in status
    assert second.json() == {
        "ok": True, "pid": proc.pid, "name": "hermes-update", "already_running": True, "action_id": action_id,
    }
    assert web_server_gateway._ACTION_PROCS["hermes-update"] is proc
    assert web_server_gateway._ACTION_IDS["hermes-update"] == action_id
    assert web_server_gateway._ACTION_COMMANDS["hermes-update"] is original_command
    assert "hermes-update" not in web_server_gateway._ACTION_RESULTS
    assert admission.read_bytes() == original_record and log.read_bytes() == original_log
    assert [path.readlink() for path in fds] == original_streams
    assert launches == [("update",)] and list(receipts.iterdir()) == []


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("gate", ["commit-build", "managed-root", "image-marker", "image-marker-invalid"])
@pytest.mark.parametrize("old_attempt", ["idle", "exited-process", "cached-result"])
def test_nonrunning_update_still_records_real_refusal(owned_route_worker, monkeypatch, gate, old_attempt):
    client, receipts, root, proc, action_id, launches, release = owned_route_worker
    release()
    if old_attempt == "cached-result":
        finished = client.get("/api/actions/hermes-update/status", params={"action_id": action_id}).json()
        assert finished["state"] == "finished" and finished["exit_code"] == 0
    elif old_attempt == "idle":
        for registry in (web_server_gateway._ACTION_PROCS, web_server_gateway._ACTION_RESULTS,
                         web_server_gateway._ACTION_COMMANDS, web_server_gateway._ACTION_IDS):
            registry.clear()
        (receipts.parent / "hermes-update-action.json").unlink()
    admission = receipts.parent / "hermes-update-action.json"
    original_record = admission.read_bytes() if admission.exists() else None
    error, has_receipt = set_route_refusal(gate, root, receipts, monkeypatch)

    response = client.post("/api/hermes/update")
    assert response.status_code == 200
    assert response.json()["ok"] is False and response.json()["error"] == error
    assert response.json()["pid"] is None and "already_running" not in response.json()
    assert "action_id" not in response.json()
    assert web_server_gateway._ACTION_RESULTS["hermes-update"] == {"exit_code": 1, "pid": None}
    assert all("hermes-update" not in registry for registry in (
        web_server_gateway._ACTION_PROCS, web_server_gateway._ACTION_IDS, web_server_gateway._ACTION_COMMANDS,
    ))
    assert (admission.read_bytes() if admission.exists() else None) == original_record
    assert response.json()["message"] in (receipts.parent / "hermes-update.log").read_text()
    assert launches == [("update",)]  # A refusal never starts a replacement worker.
    if has_receipt:
        receipt = json.loads((receipts / "latest.json").read_text())
        assert receipt["outcome"] == "refused" and receipt["finished_at"]
        assert receipt["stop_reason"] == gate
    else:
        assert list(receipts.iterdir()) == []


def test_requested_attempt_recovers_archive_not_newer_process_failure(status_env):
    client, receipts = status_env
    newer_id = "b" * 32
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps({
        "version": 1, "action_id": newer_id, "admitted_at": actions.time.time(),
    }))
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    }, f"update_20260817_112000_123_{ACTION_ID}.json")
    write_receipt(receipts, {
        "update_id": newer_id, "outcome": "failed", "finished_at": FINISHED_AT,
    })
    with subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"]) as proc:
        assert proc.wait(timeout=10) == 7
    web_server_gateway._ACTION_IDS["hermes-update"] = newer_id
    web_server_gateway._ACTION_PROCS["hermes-update"] = proc

    # Exact recovery must not poll/reap B or use its exit code, pid or output.
    for _ in range(2):
        data = client.get("/api/actions/hermes-update/status", params={"action_id": ACTION_ID}).json()
        assert data["action_id"] == ACTION_ID
        assert data["state"] == "finished" and data["exit_code"] == 0
        assert data["receipt"]["update_id"] == ACTION_ID
        assert data["pid"] is None and data["lines"] == []
        assert web_server_gateway._ACTION_PROCS["hermes-update"] is proc
    # No-query clients still see B's explicit process failure, including its cached result.
    for _ in range(2):
        data = client.get("/api/actions/hermes-update/status").json()
        assert data["action_id"] == newer_id and data["exit_code"] == 7
        assert data["pid"] == proc.pid


@pytest.mark.parametrize("newer_state", ["finished", "abandoned", "pending"])
@pytest.mark.parametrize("archived", [False, True])
def test_requested_unfinished_attempt_is_superseded_not_newer_outcome(status_env, monkeypatch, newer_state, archived):
    client, receipts = status_env
    newer_id = "b" * 32
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps({
        "version": 1, "action_id": newer_id,
        "admitted_at": actions.time.time() - (1201 if newer_state == "abandoned" else 0),
    }))
    (receipts.parent / "hermes-update.log").write_text("newer attempt output\n")
    monkeypatch.setattr(actions, "_update_activity", lambda *_: False)
    if archived:
        write_receipt(receipts, {
            "update_id": ACTION_ID, "outcome": "running", "finished_at": None,
        }, f"update_20260817_112000_123_{ACTION_ID}.json")
    if newer_state == "finished":
        web_server_gateway._ACTION_RESULTS["hermes-update"] = {"pid": 12345, "exit_code": 7}
        write_receipt(receipts, {"update_id": newer_id, "outcome": "failed", "finished_at": FINISHED_AT})
    assert client.get("/api/actions/hermes-update/status").json()["state"] == newer_state

    data = client.get("/api/actions/hermes-update/status", params={"action_id": ACTION_ID}).json()
    assert data["state"] == "superseded"
    assert data["action_id"] == ACTION_ID
    assert data["running"] is False and data["exit_code"] is None and data["pid"] is None
    assert data["lines"] == [] and "receipt" not in data


def test_requested_attempt_without_current_identity_keeps_unknown_deadline(status_env, monkeypatch):
    client, receipts = status_env
    record = receipts.parent / "hermes-update-action.json"
    record.write_text("{}")
    os.utime(record, (100, 100))
    os.utime(receipts.parent / "update.log", (100, 100))
    monkeypatch.setattr(actions.time, "time", lambda: 200)
    monkeypatch.setattr(actions, "_update_activity", lambda *_: False)
    params = {"action_id": ACTION_ID}
    data = client.get("/api/actions/hermes-update/status", params=params).json()
    assert data["state"] == "unknown" and data["exit_code"] is None
    assert "action_id" not in data and "receipt" not in data
    monkeypatch.setattr(actions.time, "time", lambda: 2000)
    data = client.get("/api/actions/hermes-update/status", params=params).json()
    assert data["state"] == "abandoned" and data["exit_code"] is None


@pytest.mark.parametrize("invalid_id", ["", "../bad", "a" * 31, "a" * 33, "A" * 32, "*", "a" * 32 + "\n"])
def test_status_rejects_malformed_requested_id_before_receipt_lookup(status_env, monkeypatch, invalid_id):
    client, _ = status_env
    looked_up = []
    monkeypatch.setattr(actions, "_latest_update_receipt_summary", lambda action_id: looked_up.append(action_id))
    response = client.get("/api/actions/hermes-update/status", params={"action_id": invalid_id})
    assert response.status_code == 400
    assert looked_up == []


@pytest.mark.parametrize("old_admission", [False, True])
def test_new_unfinished_start_does_not_reuse_old_success_after_registry_loss(status_env, old_admission):
    client, receipts = status_env
    if old_admission:
        (receipts.parent / "hermes-update.log").write_text(
            f"=== hermes-update started {ACTION_ID} ===\n", encoding="utf-8",
        )
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    })
    with (receipts.parent / "update.log").open("a", encoding="utf-8") as log:
        log.write("=== hermes update started 2026-08-17T12:00:00 ===\n")

    data = client.get("/api/actions/hermes-update/status").json()

    assert data["exit_code"] is None
    assert "receipt" not in data
    # Historical timestamp banners are not action identities.
    assert "action_id" not in data


@pytest.mark.parametrize("outcome, expected", [(None, None), ("failed", 1), ("partial", 1), ("refused", 1), ("success", 0)])
def test_durable_admission_survives_crash_before_updater_banner(status_env, monkeypatch, outcome, expected):
    client, receipts = status_env
    new_id = "e" * 32
    web_server_gateway._ACTION_RESULTS["hermes-update"] = {"exit_code": 0, "pid": 12345}
    web_server_gateway._ACTION_IDS["hermes-update"] = ACTION_ID
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    })
    from hermes_cli import _launchers
    monkeypatch.setattr(_launchers, "runtime_command", lambda *_: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(web_server_gateway, "_action_targets_system_gateway", lambda *_: False)
    monkeypatch.setattr(web_server_gateway, "_profile_action_environment", lambda *args: {})
    synced_files = []
    real_fsync = os.fsync

    def observe_fsync(fd):
        real_fsync(fd)
        synced_files.append(os.fstat(fd))

    monkeypatch.setattr(web_server_gateway.os, "fsync", observe_fsync)

    def crash_before_spawn(*args, **kwargs):
        assert os.fstat(kwargs["stdout"].fileno()) in synced_files
        record_path = receipts.parent / "hermes-update-action.json"
        record = json.loads(record_path.read_text())
        assert record["version"] == 1 and record["action_id"] == new_id
        synced_inodes = {(stat.st_dev, stat.st_ino) for stat in synced_files}
        stat = record_path.stat()
        assert (stat.st_dev, stat.st_ino) in synced_inodes
        if os.name != "nt":
            stat = receipts.parent.stat()
            assert (stat.st_dev, stat.st_ino) in synced_inodes
        # Inspect the real route at the spawn boundary, while no updater has run.
        data = client.get("/api/actions/hermes-update/status").json()
        assert data.get("action_id") == new_id
        assert data["exit_code"] is None
        assert "receipt" not in data
        raise RuntimeError("owned test crash before updater banner")

    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", crash_before_spawn)
    with pytest.raises(RuntimeError, match="owned test crash"):
        web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": new_id})
    for registry in (web_server_gateway._ACTION_IDS, web_server_gateway._ACTION_PROCS,
                     web_server_gateway._ACTION_RESULTS, web_server_gateway._ACTION_COMMANDS):
        registry.clear()
    if outcome:
        write_receipt(receipts, {"update_id": new_id, "outcome": outcome, "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["action_id"] == new_id
    assert data["exit_code"] == expected


@pytest.mark.parametrize("erase_log", [False, True])
def test_admitted_identity_survives_output_tail_loss(status_env, monkeypatch, erase_log):
    client, receipts = status_env
    new_id = "e" * 32
    write_receipt(receipts, {"update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT})
    from hermes_cli import _launchers
    monkeypatch.setattr(_launchers, "runtime_command", lambda *_: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(web_server_gateway, "_action_targets_system_gateway", lambda *_: False)
    monkeypatch.setattr(web_server_gateway, "_profile_action_environment", lambda *args: {})

    def crash_before_spawn(*args, **kwargs):
        log = receipts.parent / "hermes-update.log"
        if erase_log:
            log.unlink()
        else:
            with log.open("a") as handle:
                handle.write("noisy child output\n" * 2001)
        data = client.get("/api/actions/hermes-update/status").json()
        assert data.get("action_id") == new_id
        assert data["exit_code"] is None
        assert "receipt" not in data
        raise RuntimeError("owned crash")

    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", crash_before_spawn)
    with pytest.raises(RuntimeError, match="owned crash"):
        web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": new_id})
    # Actual durable read after registry loss, then the correlated terminal control.
    for registry in (web_server_gateway._ACTION_IDS, web_server_gateway._ACTION_PROCS,
                     web_server_gateway._ACTION_RESULTS, web_server_gateway._ACTION_COMMANDS):
        registry.clear()
    assert client.get("/api/actions/hermes-update/status").json()["action_id"] == new_id
    write_receipt(receipts, {"update_id": new_id, "outcome": "success", "finished_at": FINISHED_AT})
    assert client.get("/api/actions/hermes-update/status").json()["exit_code"] == 0


def test_unfinished_admission_keeps_polling_until_correlated_terminal(status_env):
    client, receipts = status_env
    new_id = "e" * 32
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps({
        "version": 1, "action_id": new_id, "admitted_at": actions.time.time(),
    }))
    write_receipt(receipts, {"update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["action_id"] == new_id
    assert data["exit_code"] is None
    assert data.get("state") == "pending"
    write_receipt(receipts, {"update_id": new_id, "outcome": "running", "finished_at": None})
    assert client.get("/api/actions/hermes-update/status").json()["state"] == "pending"
    write_receipt(receipts, {"update_id": new_id, "outcome": "success", "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["state"] == "finished" and data["exit_code"] == 0


@pytest.mark.parametrize("age, live, expected", [
    (1199, False, "pending"), (1201, False, "abandoned"),
    (1201, True, "pending"), (1201, None, "pending"),
])
def test_abandonment_requires_expired_admission_and_no_live_evidence(status_env, monkeypatch, age, live, expected):
    client, receipts = status_env
    monkeypatch.setattr(actions.time, "time", lambda: 10000)
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps({
        "version": 1, "action_id": "e" * 32, "admitted_at": 10000 - age,
    }))
    # Liveness is independent from elapsed time; inaccessible evidence is not absence.
    monkeypatch.setattr(actions, "_update_activity", lambda *_: live, raising=False)
    data = client.get("/api/actions/hermes-update/status").json()
    assert data.get("state") == expected
    assert data["exit_code"] is None  # Abandoned means unknown outcome, never invented failure.


def test_unknown_abandoned_admission_is_bounded_without_inventing_failure(status_env, monkeypatch):
    client, receipts = status_env
    record = receipts.parent / "hermes-update-action.json"
    record.write_text("{}")
    os.utime(record, (100, 100))
    os.utime(receipts.parent / "update.log", (100, 100))
    monkeypatch.setattr(actions.time, "time", lambda: 2000)
    monkeypatch.setattr(actions, "_update_activity", lambda *_: False)
    data = client.get("/api/actions/hermes-update/status").json()
    assert data.get("state") == "abandoned"
    assert data["exit_code"] is None and "action_id" not in data


def test_unknown_activity_checks_actual_process_absence(status_env, monkeypatch):
    import psutil
    _, receipts = status_env
    monkeypatch.setattr(psutil, "process_iter", lambda: iter(()))
    assert actions._update_activity(receipts.parent, None) is False


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("sync_target", ["file", "directory"])
@pytest.mark.parametrize("old_registry", ["result", "process"])
def test_failed_admission_never_attaches_old_process_result(status_env, monkeypatch, sync_target, old_registry):
    import stat
    from hermes_cli.web_update_action import read_update_admission, write_update_admission

    client, receipts = status_env
    new_id = "e" * 32
    write_update_admission(receipts.parent, ACTION_ID)
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    }, f"update_20260817_112000_123_{ACTION_ID}.json")
    with subprocess.Popen([sys.executable, "-c", "pass"]) as old_proc:
        assert old_proc.wait(timeout=10) == 0
    if old_registry == "result":
        web_server_gateway._ACTION_RESULTS["hermes-update"] = {"exit_code": 0, "pid": old_proc.pid}
    else:
        web_server_gateway._ACTION_PROCS["hermes-update"] = old_proc
    spawned = []
    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", lambda *a, **kw: spawned.append(True))
    real_fsync = os.fsync

    def fail_selected_sync(fd):
        is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
        if is_directory == (sync_target == "directory"):
            raise OSError("owned selected admission fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_selected_sync)
    with pytest.raises(OSError, match="owned selected admission fsync"):
        web_server_gateway._spawn_hermes_action(
            ["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": new_id},
        )
    assert spawned == []
    assert not list(receipts.parent.glob(".hermes-update-action-*"))
    record, exists = read_update_admission(receipts.parent)
    assert exists and record["action_id"] == (new_id if sync_target == "directory" else ACTION_ID)
    for _ in range(2):
        data = client.get("/api/actions/hermes-update/status", params={"action_id": new_id}).json()
        assert data["exit_code"] is None
        assert data["pid"] is None
        assert data["state"] == ("pending" if sync_target == "directory" else "superseded")
        assert "receipt" not in data
    old_data = client.get("/api/actions/hermes-update/status", params={"action_id": ACTION_ID}).json()
    assert old_data["action_id"] == ACTION_ID
    assert old_data["state"] == "finished" and old_data["exit_code"] == 0
    assert old_data["receipt"]["update_id"] == ACTION_ID


@pytest.mark.platforms("posix")
def test_admission_preserves_live_update_then_retires_old_registries_on_spawn(status_env, monkeypatch):
    import select
    from hermes_cli import _launchers, update_contract
    from hermes_cli.web_update_action import read_update_admission, write_update_admission

    client, receipts = status_env
    client.app.include_router(actions.router)
    monkeypatch.setattr(actions, "is_commit_build", lambda *_: False)
    monkeypatch.setattr(actions, "_dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr(update_contract, "evaluate_update_admission", lambda *_: None)
    monkeypatch.setattr(_launchers, "runtime_command", lambda *_: [sys.executable, "-c", "pass"])
    monkeypatch.setattr(web_server_gateway, "_action_targets_system_gateway", lambda *_: False)
    monkeypatch.setattr(web_server_gateway, "_profile_action_environment", lambda *args: {})
    write_update_admission(receipts.parent, ACTION_ID)
    original_record = (receipts.parent / "hermes-update-action.json").read_bytes()
    with subprocess.Popen(
        [sys.executable, "-u", "-c", "import sys; print('ready', flush=True); sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    ) as old_proc:
        try:
            assert select.select([old_proc.stdout], [], [], 10)[0], "owned worker readiness"
            assert old_proc.stdout.readline() == b"ready\n"
            web_server_gateway._ACTION_PROCS["hermes-update"] = old_proc
            web_server_gateway._ACTION_IDS["hermes-update"] = ACTION_ID
            web_server_gateway._ACTION_COMMANDS["hermes-update"] = ("update",)
            response = client.post("/api/hermes/update")
            assert response.status_code == 200
            assert response.json()["already_running"] is True
            assert response.json()["action_id"] == ACTION_ID
            assert response.json()["pid"] == old_proc.pid
            assert web_server_gateway._ACTION_PROCS["hermes-update"] is old_proc
            assert (receipts.parent / "hermes-update-action.json").read_bytes() == original_record
        finally:
            old_proc.communicate(timeout=10)
    assert old_proc.returncode == 0
    web_server_gateway._ACTION_RESULTS["hermes-update"] = {"exit_code": 0, "pid": old_proc.pid}
    response = client.post("/api/hermes/update")
    assert response.status_code == 200
    data = response.json()
    proc = web_server_gateway._ACTION_PROCS["hermes-update"]
    try:
        assert proc is not old_proc and data["pid"] == proc.pid
        assert data["action_id"] != ACTION_ID
        assert web_server_gateway._ACTION_IDS["hermes-update"] == data["action_id"]
        assert web_server_gateway._ACTION_COMMANDS["hermes-update"] == ("update",)
        assert "hermes-update" not in web_server_gateway._ACTION_RESULTS
        record, exists = read_update_admission(receipts.parent)
        assert exists and record["action_id"] == data["action_id"]
    finally:
        assert proc.wait(timeout=10) == 0


@pytest.mark.parametrize("fault", ["fsync", "replace"])
def test_admission_storage_failure_never_spawns(status_env, monkeypatch, fault):
    from hermes_cli import web_update_action
    _, receipts = status_env
    spawned = []
    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", lambda *a, **kw: spawned.append(True))
    def fail(*args):
        raise OSError("owned admission storage failure")
    monkeypatch.setattr(web_update_action.os, fault, fail)
    with pytest.raises(OSError, match="owned admission"):
        web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": "e" * 32})
    assert not spawned
    assert not list(receipts.parent.glob(".hermes-update-action-*"))


@pytest.mark.parametrize("invalid_id", [None, "../bad", "a" * 31, "A" * 32, "a" * 33, 32])
def test_malformed_admission_id_never_spawns(status_env, monkeypatch, invalid_id):
    spawned = []
    monkeypatch.setattr(web_server_gateway.subprocess, "Popen", lambda *a, **kw: spawned.append(True))
    with pytest.raises(ValueError, match="32-hex"):
        web_server_gateway._spawn_hermes_action(["update"], "hermes-update", env_overrides={"HERMES_ACTION_ID": invalid_id})
    assert not spawned


@pytest.mark.parametrize("admitted_at, valid", [
    (10**400, False), (-10**400, False), (True, False), ("100", False),
    (float("nan"), False), (float("inf"), False), (100, True),
])
def test_admission_timestamp_validation_keeps_stale_receipt_barrier(status_env, monkeypatch, admitted_at, valid):
    from hermes_cli.web_update_action import read_update_admission

    client, receipts = status_env
    monkeypatch.setattr(actions.time, "time", lambda: 101)
    record = {"version": 1, "action_id": "e" * 32, "admitted_at": admitted_at}
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps(record))
    write_receipt(receipts, {"update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT})
    response = client.get("/api/actions/hermes-update/status")
    assert response.status_code == 200
    data = response.json()
    assert data["state"] == ("pending" if valid else "unknown")
    assert data["exit_code"] is None and data["pid"] is None
    assert "receipt" not in data
    if valid:
        assert data["action_id"] == record["action_id"]
    else:
        assert "action_id" not in data
    assert read_update_admission(receipts.parent) == (record if valid else None, True)


@pytest.mark.parametrize("field, invalid", [("version", True), ("version", 2),
    ("action_id", "../bad"), ("action_id", "A" * 32), ("action_id", 32),
    ("admitted_at", True), ("admitted_at", "100"), ("admitted_at", -1),
    ("admitted_at", float("nan")), ("admitted_at", float("inf")), ("extra", 1)])
def test_invalid_admission_record_is_barrier_to_stale_success(status_env, field, invalid):
    client, receipts = status_env
    record = {"version": 1, "action_id": "e" * 32, "admitted_at": actions.time.time()}
    record[field] = invalid
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps(record))
    write_receipt(receipts, {"update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["state"] == "unknown" and data["exit_code"] is None
    assert "action_id" not in data and "receipt" not in data


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("marker", [False, True])
def test_expired_admission_does_not_abandon_live_owned_worker(status_env, monkeypatch, marker):
    import select
    client, receipts = status_env
    new_id = "e" * 32
    (receipts.parent / "hermes-update-action.json").write_text(json.dumps({
        "version": 1, "action_id": new_id, "admitted_at": actions.time.time() - 1201,
    }))
    env = {**os.environ, "HERMES_ACTION_ID": new_id}
    script = "import sys; print('ready', flush=True); sys.stdin.read()"
    with subprocess.Popen([sys.executable, "-u", "-c", script], env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
        try:
            assert select.select([proc.stdout], [], [], 10)[0], "owned worker readiness"
            assert proc.stdout.readline() == b"ready\n"
            if marker:
                (receipts.parent.parent / ".hermes-update-in-progress").write_text(f"{proc.pid}\n1\n")
            assert actions._update_activity(receipts.parent, new_id) is True
            assert client.get("/api/actions/hermes-update/status").json()["state"] == "pending"
        finally:
            proc.communicate(timeout=10)
    write_receipt(receipts, {"update_id": new_id, "outcome": "success", "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["state"] == "finished" and data["exit_code"] == 0


@pytest.mark.parametrize("command, expected", [
    ([sys.executable, "-m", "hermes_cli.main", "update"], True),
    ([sys.executable, "-m", "hermes_cli.main", "kanban", "--preserve-cache", "update"], False),
])
def test_unknown_live_activity_uses_canonical_updater_matcher(status_env, monkeypatch, command, expected):
    import psutil
    from types import SimpleNamespace
    _, receipts = status_env
    process = SimpleNamespace(status=lambda: psutil.STATUS_RUNNING, environ=lambda: {}, cmdline=lambda: command)
    monkeypatch.setattr(psutil, "process_iter", lambda: iter([process]))
    assert actions._update_activity(receipts.parent, None) is expected


def test_admission_records_and_status_remain_home_local_a_b_a(status_env, monkeypatch, tmp_path):
    from hermes_cli.web_update_action import write_update_admission
    client, _ = status_env
    homes = [tmp_path / "A", tmp_path / "B"]
    ids = ["a" * 32, "b" * 32]
    for home, action_id in zip(homes, ids):
        logs = home / "logs"
        logs.mkdir(parents=True)
        write_update_admission(logs, action_id)
    for index in (0, 1, 0):
        monkeypatch.setenv("HERMES_HOME", str(homes[index]))
        monkeypatch.setattr(web_server_gateway, "_ACTION_LOG_DIR", homes[index] / "logs")
        data = client.get("/api/actions/hermes-update/status").json()
        assert data["action_id"] == ids[index] and data["state"] == "pending"


@pytest.mark.parametrize("kind", ["update", "sync"])
@pytest.mark.parametrize("admitted", [False, True])
def test_pm_pointer_is_not_an_update_receipt(status_env, kind, admitted):
    client, receipts = status_env
    (receipts.parent / "update.log").unlink()
    if admitted:
        (receipts.parent / "hermes-update.log").write_text(
            f"=== hermes-update started {ACTION_ID} ===\n", encoding="utf-8",
        )
        write_receipt(receipts, {
            "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
        }, f"update_20260817_112000_123_{ACTION_ID}.json")
    write_receipt(receipts, {
        "kind": kind, "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    })

    response = client.get("/api/hermes/update/receipt")
    assert response.status_code == 404
    assert actions._latest_update_receipt_summary() is None
    data = client.get("/api/actions/hermes-update/status").json()
    if admitted:
        assert data["action_id"] == data["receipt"]["update_id"] == ACTION_ID
        assert data["exit_code"] == 0
    else:
        assert data["exit_code"] is None
        assert "receipt" not in data


@pytest.mark.parametrize("legacy_start", ["2026-08-17 12:00:00", "not-a-hex-action-id"])
def test_legacy_dashboard_start_does_not_recover_old_completed_id(status_env, legacy_start):
    client, receipts = status_env
    (receipts.parent / "hermes-update.log").write_text(
        f"=== hermes-update started {legacy_start} ===\n", encoding="utf-8",
    )
    write_receipt(receipts, {"update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT})
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["exit_code"] is None
    assert "action_id" not in data
    assert "receipt" not in data


@pytest.mark.parametrize("finished_at", [None, "", FINISHED_AT])
def test_marker_success_requires_terminal_receipt(status_env, finished_at):
    client, receipts = status_env
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": finished_at,
    })

    response = client.get("/api/actions/hermes-update/status")

    assert response.status_code == 200
    data = response.json()
    assert data["running"] is False
    assert data["exit_code"] == (0 if finished_at else None)


@pytest.mark.parametrize("receipt_id", [None, "d" * 32, ACTION_ID])
def test_terminal_receipt_must_match_action_id(status_env, receipt_id):
    client, receipts = status_env
    receipt = {
        "update_id": receipt_id, "outcome": "success", "finished_at": FINISHED_AT,
    }
    write_receipt(receipts, receipt)

    data = client.get("/api/actions/hermes-update/status").json()

    expected = 0 if receipt_id == ACTION_ID else None
    assert data["exit_code"] == expected
    # Reader correlation must not mask the independent exit-code validation.
    assert actions._completed_exit_code(None, ACTION_ID, receipt) == expected
    if receipt_id == ACTION_ID:
        assert data["receipt"]["update_id"] == data["action_id"]


@pytest.mark.parametrize("outcome, exit_code", [
    ("success", 0), ("partial", 1), ("failed", 1), ("refused", 1), ("running", None),
])
def test_terminal_receipt_reports_outcome_not_marker(status_env, outcome, exit_code):
    client, receipts = status_env
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": outcome, "finished_at": FINISHED_AT,
    })

    data = client.get("/api/actions/hermes-update/status").json()

    assert data["exit_code"] == exit_code


@pytest.mark.parametrize("latest", [
    {"update_id": "d" * 32, "outcome": "success", "finished_at": FINISHED_AT},
    {"kind": "update", "update_id": ACTION_ID, "outcome": "ok", "finished_at": FINISHED_AT},
    {"kind": "sync", "update_id": None, "outcome": "ok", "finished_at": FINISHED_AT},
])
@pytest.mark.parametrize("outcome, exit_code", [("success", 0), ("failed", 1)])
def test_status_looks_up_exact_update_when_latest_is_unrelated(status_env, latest, outcome, exit_code):
    client, receipts = status_env
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": outcome, "finished_at": FINISHED_AT,
    }, f"update_20260817_112000_123_{ACTION_ID}.json")
    write_receipt(receipts, latest)

    data = client.get("/api/actions/hermes-update/status").json()

    assert data["exit_code"] == exit_code
    assert data["receipt"]["update_id"] == data["action_id"] == ACTION_ID
    assert data["receipt"]["outcome"] == outcome


@pytest.mark.parametrize("old_marker", [None, "d" * 32])
def test_known_action_id_takes_precedence_over_old_completion(status_env, old_marker):
    client, receipts = status_env
    log = receipts.parent / "update.log"
    log.write_text(
        f"=== hermes-update completed {old_marker} ===\n" if old_marker else "",
        encoding="utf-8",
    )
    web_server_gateway._ACTION_IDS["hermes-update"] = ACTION_ID
    write_receipt(receipts, {
        "update_id": "d" * 32, "outcome": "success", "finished_at": FINISHED_AT,
    })

    data = client.get("/api/actions/hermes-update/status").json()

    assert data["exit_code"] is None
    assert "receipt" not in data
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    }, f"update_20260817_112000_123_{ACTION_ID}.json")
    data = client.get("/api/actions/hermes-update/status").json()
    assert data["exit_code"] == 0
    assert data["receipt"]["update_id"] == data["action_id"] == ACTION_ID


@pytest.mark.parametrize("receipt_text", [
    None, "{", "[]",
    json.dumps({"update_id": "d" * 32, "outcome": "success", "finished_at": FINISHED_AT}),
    json.dumps({"kind": "sync", "update_id": ACTION_ID, "outcome": "ok", "finished_at": FINISHED_AT}),
])
def test_marker_without_valid_receipt_does_not_claim_success(status_env, receipt_text):
    client, receipts = status_env
    if receipt_text is not None:
        (receipts / "latest.json").write_text(receipt_text, encoding="utf-8")
        # A matching filename alone is not proof of the archived receipt's identity.
        (receipts / f"update_20260817_112000_123_{ACTION_ID}.json").write_text(receipt_text, encoding="utf-8")
    assert client.get("/api/actions/hermes-update/status").json()["exit_code"] is None


@pytest.mark.parametrize("registered_process", [False, True])
def test_explicit_process_failure_overrides_success_receipt(status_env, registered_process):
    client, receipts = status_env
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
    })
    if registered_process:
        with subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"]) as proc:
            assert proc.wait(timeout=10) == 7
        pid = proc.pid
        web_server_gateway._ACTION_PROCS["hermes-update"] = proc
    else:
        pid = 12345
        web_server_gateway._ACTION_RESULTS["hermes-update"] = {"pid": pid, "exit_code": 7}

    # Both the initial poll and cached result must preserve the explicit process failure.
    for _ in range(2):
        data = client.get("/api/actions/hermes-update/status").json()
        assert data["exit_code"] == 7
        assert data["running"] is False
        assert data["pid"] == pid


@pytest.mark.parametrize("marker_present", [True, False])
def test_success_without_gateway_keeps_deferred_verification(status_env, marker_present):
    client, receipts = status_env
    if not marker_present:
        (receipts.parent / "update.log").unlink()
        # Rotation may lose completion banners; durable admission still correlates the receipt.
        (receipts.parent / "hermes-update.log").write_text(
            f"=== hermes-update started {ACTION_ID} ===\n", encoding="utf-8",
        )
    write_receipt(receipts, {
        "update_id": ACTION_ID, "outcome": "success", "finished_at": FINISHED_AT,
        "fleet": [], "skips": [{"name": "fleet-verify", "reason": "no gateways running; deferred"}],
    })

    data = client.get("/api/actions/hermes-update/status").json()

    assert data["exit_code"] == 0
    assert data["receipt"]["fleet_states"] == []
    assert data["receipt"]["update_id"] == ACTION_ID
