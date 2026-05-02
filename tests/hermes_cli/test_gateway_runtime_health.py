from hermes_cli.gateway import _runtime_health_lines


def test_runtime_health_lines_include_fatal_platform_and_startup_reason(monkeypatch):
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda: {
            "gateway_state": "startup_failed",
            "exit_reason": "telegram conflict",
            "platforms": {
                "telegram": {
                    "state": "fatal",
                    "error_message": "another poller is active",
                }
            },
        },
    )

    lines = _runtime_health_lines()

    assert "⚠ telegram: another poller is active" in lines
    assert "⚠ Last startup issue: telegram conflict" in lines


def test_gateway_owner_status_reports_shared_lock_without_local_pid(monkeypatch, tmp_path):
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    lock_path = tmp_path / "gateway.lock"
    pid_path.write_text('{"pid": 424242, "kind": "hermes-gateway", "argv": ["hermes", "gateway"], "start_time": 1}')
    lock_path.write_text('{"pid": 424242, "kind": "hermes-gateway", "argv": ["hermes", "gateway"], "start_time": 1}')

    monkeypatch.setattr(status, "_get_pid_path", lambda: pid_path)
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda pid_path_arg=None: lock_path)
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path_arg=None: True)
    monkeypatch.setattr(status.os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))

    owner_status = status.get_gateway_owner_status()

    assert owner_status["state"] == "shared_lock_active"
    assert owner_status["pid"] == 424242
    assert "another namespace/container" in owner_status["message"]
    assert pid_path.exists()
    assert lock_path.exists()


def test_gateway_owner_status_reports_local_pid_when_visible(monkeypatch, tmp_path):
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    lock_path = tmp_path / "gateway.lock"
    record = '{"pid": 123, "kind": "hermes-gateway", "argv": ["hermes", "gateway"], "start_time": 77}'
    pid_path.write_text(record)
    lock_path.write_text(record)

    monkeypatch.setattr(status, "_get_pid_path", lambda: pid_path)
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda pid_path_arg=None: lock_path)
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path_arg=None: True)
    monkeypatch.setattr(status.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: 77)
    monkeypatch.setattr(status, "_looks_like_gateway_process", lambda pid: True)

    owner_status = status.get_gateway_owner_status()

    assert owner_status["state"] == "local_pid_running"
    assert owner_status["pid"] == 123


def test_gateway_owner_status_reports_not_running_without_runtime_lock(monkeypatch, tmp_path):
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    lock_path = tmp_path / "gateway.lock"

    monkeypatch.setattr(status, "_get_pid_path", lambda: pid_path)
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda pid_path_arg=None: lock_path)
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path_arg=None: False)

    owner_status = status.get_gateway_owner_status()

    assert owner_status["state"] == "not_running"
    assert owner_status["pid"] is None


def test_gateway_owner_status_reports_shared_lock_when_pid_not_inspectable(monkeypatch, tmp_path):
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    lock_path = tmp_path / "gateway.lock"
    record = '{"pid": 123, "kind": "hermes-gateway", "argv": ["hermes", "gateway"], "start_time": 77}'
    pid_path.write_text(record)
    lock_path.write_text(record)

    monkeypatch.setattr(status, "_get_pid_path", lambda: pid_path)
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda pid_path_arg=None: lock_path)
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path_arg=None: True)
    monkeypatch.setattr(status.os, "kill", lambda pid, sig: (_ for _ in ()).throw(PermissionError()))

    owner_status = status.get_gateway_owner_status()

    assert owner_status["state"] == "shared_lock_active"
    assert owner_status["pid"] == 123
    assert "not inspectable" in owner_status["message"]


def test_get_running_pid_preserves_metadata_when_shared_lock_is_active(monkeypatch, tmp_path):
    from gateway import status

    pid_path = tmp_path / "gateway.pid"
    lock_path = tmp_path / "gateway.lock"
    record = '{"pid": 424242, "kind": "hermes-gateway", "argv": ["hermes", "gateway"], "start_time": 1}'
    pid_path.write_text(record)
    lock_path.write_text(record)

    cleanup_calls = []
    monkeypatch.setattr(status, "_get_pid_path", lambda: pid_path)
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda pid_path_arg=None: lock_path)
    monkeypatch.setattr(status, "is_gateway_runtime_lock_active", lambda lock_path_arg=None: True)
    monkeypatch.setattr(status.os, "kill", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(status, "_cleanup_invalid_pid_path", lambda *args, **kwargs: cleanup_calls.append((args, kwargs)))

    assert status.get_running_pid() is None
    assert cleanup_calls == []
    assert pid_path.exists()
    assert lock_path.exists()
