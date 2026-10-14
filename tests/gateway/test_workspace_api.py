"""Real HTTP + session database + local native terminal workspace contracts."""
import base64
import shlex
import sys

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms import api_server_workspace as workspace
from hermes_state import SessionDB


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from tools import terminal_tool

    root = tmp_path / "workspace"
    root.mkdir()
    (get_hermes_home() / "config.yaml").write_text(
        f"terminal:\n  env: local\n  cwd: {root}\ngateway:\n  api_server:\n    workspace:\n      enabled: true\n")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    db = SessionDB(tmp_path / "state.db")
    db.create_session("one", source="api_server")
    db.create_session("two", source="api_server")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "workspace-key"}))
    adapter._session_db = db
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in workspace.http_routes(adapter):
        app.router.add_route(method, path, handler)
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/health/detailed", adapter._handle_health_detailed)
    async with TestClient(TestServer(app), headers={"Authorization": "Bearer workspace-key"}) as client:
        client.workspace_keys = {}
        yield client, adapter, root
    await workspace.close(adapter)
    for env in terminal_tool._active_environments.values():
        env.cleanup()
    db.close()


async def post(client, path, body, key="key"):
    if key not in client.workspace_keys:
        caps = await (await client.get("/v1/capabilities")).json()
        client.workspace_keys[key] = caps["features"]["workspace_io"]["admission_epoch"] + ":" + key
    response = await client.post("/api/workspace" + path, json={"session_id": "one", **body},
                                 headers={"Idempotency-Key": client.workspace_keys[key]})
    data = await response.json()
    assert response.status < 300, data
    return data


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_file_owner_bounds_and_original_key_replay(api):
    client, _adapter, root = api
    caps = await (await client.get("/v1/capabilities")).json()
    assert caps["features"]["workspace_io"]["root"] == str(root)
    data = b"\x00\xffbinary"
    body = {"path": "nested/file", "data_base64": base64.b64encode(data).decode()}
    first = await post(client, "/files/write", body)
    assert (root / "nested/file").read_bytes() == data
    (root / "nested/file").write_bytes(b"newer")
    assert await post(client, "/files/write", body) == first
    assert (root / "nested/file").read_bytes() == b"newer"
    assert await post(client, "/files/read", {"path": "absent"}) == {"kind": "missing"}
    response = await client.post("/api/workspace/files/read", json={"session_id": "one", "path": "../secret"})
    assert response.status == 400
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"secret")
    (root / "link").symlink_to(outside, target_is_directory=True)
    response = await client.post("/api/workspace/files/read", json={"session_id": "one", "path": "link/secret"})
    assert response.status == 400
    await post(client, "/files/remove", {"path": "link", "recursive": True}, key="remove")
    assert (outside / "secret").read_bytes() == b"secret"
    response = await client.post("/api/workspace/processes/" + first["operation_id"] + "/status", json={"session_id": "two"})
    assert response.status == 404


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_native_streams_env_replay_cancel_and_drain(api):
    client, adapter, root = api
    code = "import os; os.write(1, b'out\\x00\\xff' + os.environ['CHILD_VALUE'].encode()); os.write(2, b'err\\xfe')"
    body = {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", "env": {"CHILD_VALUE": "value"}}
    first = await post(client, "/processes", body, "start")
    operation = first["operation_id"]
    service = next(iter(adapter._workspace_services.values()))
    await service.tasks[operation]
    status = await post(client, f"/processes/{operation}/status", {})
    assert status["status"] == "completed", status
    assert status["exit_code"] == 0, status
    chunks = await post(client, f"/processes/{operation}/output", {})
    assert base64.b64decode(chunks["stdout"]["data_base64"]) == b"out\x00\xffvalue"
    assert base64.b64decode(chunks["stderr"]["data_base64"]) == b"err\xfe"
    assert (await post(client, "/processes", body, "start"))["operation_id"] == operation
    cancelled = await post(client, "/processes/cancel", {}, "cancel-before")
    assert cancelled["status"] == "cancelled"
    assert (await post(client, "/processes", {"command": "touch must-not-exist"}, "cancel-before"))["status"] == "cancelled"
    assert not (root / "must-not-exist").exists()
    adapter._shutdown_requested = True
    # Drain state is native; completed-key reads and replay remain permitted.
    assert (await post(client, "/processes", body, "start"))["operation_id"] == operation


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_completed_output_survives_restart_without_reexecution(api):
    client, adapter, root = api
    code = "from pathlib import Path; import os; p=Path('count'); p.write_bytes(p.read_bytes()+b'x' if p.exists() else b'x'); os.write(1,b'\\xffout'); os.write(2,b'\\xfeerr')"
    body = {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"}
    first = await post(client, "/processes", body, "durable")
    service = next(iter(adapter._workspace_services.values()))
    await service.tasks[first["operation_id"]]
    expected = await post(client, f"/processes/{first['operation_id']}/output", {})
    await workspace.close(adapter)
    replay = await post(client, "/processes", body, "durable")
    assert replay["status"] == "completed"
    assert replay["termination_confirmed"]
    actual = await post(client, f"/processes/{first['operation_id']}/output", {})
    assert actual == expected
    assert (root / "count").read_bytes() == b"x"


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_active_cancel_and_drain_deny_new_side_effects(api, monkeypatch):
    from tests.waiting import async_wait_until

    client, adapter, root = api
    code = "from pathlib import Path; import threading; Path('started').touch(); threading.Event().wait()"
    first = await post(client, "/processes", {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"}, "active")
    service = next(iter(adapter._workspace_services.values()))
    await async_wait_until(lambda: (root / "started").exists(), timeout=15)
    assert adapter.active_agent_work_count() > 0
    health = await (await client.get("/health/detailed")).json()
    assert health["api_server"]["active_runs"] > 0
    assert health["api_server"]["last_request_at"] is not None
    monkeypatch.setattr(adapter, "_gateway_is_draining", lambda: True)
    denied = await client.post("/api/workspace/processes", json={"session_id": "one", "command": "touch denied"},
                               headers={"Idempotency-Key": "new"})
    assert denied.status == 503
    await post(client, "/processes/cancel", {}, "active")
    await service.tasks[first["operation_id"]]
    status = await post(client, f"/processes/{first['operation_id']}/status", {})
    assert status["status"] == "cancelled", status
    assert status["termination_confirmed"]
    assert adapter.active_agent_work_count() == 0
    assert not (root / "denied").exists()


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_config_opt_in_backend_gate_and_exact_file_limit(api, monkeypatch):
    from hermes_cli.config import set_config_value

    client, adapter, root = api
    set_config_value("gateway.api_server.workspace.enabled", "false")
    assert not workspace.capability(adapter)["enabled"]
    set_config_value("gateway.api_server.workspace.enabled", "true")
    assert workspace.capability(adapter)["enabled"]
    data = b"x" * 262_144
    await post(client, "/files/write", {"path": "max", "data_base64": base64.b64encode(data).decode()}, "max")
    assert (root / "max").read_bytes() == data
    response = await client.post("/api/workspace/files/write", json={
        "session_id": "one", "path": "too-large", "data_base64": base64.b64encode(data + b"x").decode()},
        headers={"Idempotency-Key": workspace.capability(adapter)["admission_epoch"] + ":large"})
    assert response.status == 400
    assert not (root / "too-large").exists()
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    assert workspace.capability(adapter) == {
        "enabled": False, "protocol_version": 1, "reason": "unsupported_backend", "backend": "ssh"}
    response = await client.post("/api/workspace/processes", json={"session_id": "one", "command": "touch denied"},
                                 headers={"Idempotency-Key": "ssh"})
    assert response.status == 503
    assert not (root / "denied").exists()


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
@pytest.mark.live_system_guard_bypass  # Own producer deliberately exits; its checkpoint identifies the orphan.
async def test_interrupted_process_retains_partial_bytes_and_requires_confirmed_stop(api):
    import asyncio
    import hashlib
    import subprocess
    import textwrap

    from hermes_constants import get_hermes_home
    from tools.process_registry import process_registry

    client, adapter, root = api
    scope = hashlib.sha256(b"default\0workspace-key").hexdigest()
    owner = hashlib.sha256(f"{scope}\0one".encode()).hexdigest()
    code = "import os,threading; os.write(1,b'partial\\xff'); threading.Event().wait()"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    helper = textwrap.dedent('''
        import os, sys
        from pathlib import Path
        from gateway.platforms.api_server_workspace_models import StartRequest
        from gateway.platforms.api_server_workspace_store import WorkspaceStore
        from tools.process_registry import ProcessRegistry
        from tests.waiting import wait_until
        home, root, owner, command = sys.argv[1:]
        store = WorkspaceStore(Path(home) / 'workspace_operations.db')
        body = StartRequest(session_id='one', command=command)
        key = store.admission_epoch + ':crash'
        receipt, _ = store.reserve(owner, key, 'process', body)
        registry = ProcessRegistry()
        session = registry.spawn_local(command, cwd=root, owner_task_id='workspace:' + owner,
            session_key=receipt.operation_id, shell_interactive=False)
        receipt.process_id, receipt.status = session.id, 'running'
        wait_until(lambda: session.byte_capture.read('stdout', 0, 100) == b'partial\\xff', timeout=15)
        store.capture(owner, receipt, session.byte_capture)
        print(key, flush=True)
        os._exit(0)
    ''')
    result = await asyncio.to_thread(subprocess.run, [
        sys.executable, "-c", helper, str(get_hermes_home()), str(root), owner, command],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    client.workspace_keys["crash"] = result.stdout.strip()
    try:
        receipt = await post(client, "/processes", {"command": command}, "crash")
        assert receipt["status"] == "interrupted"
        assert not receipt["termination_confirmed"]
        assert adapter.active_agent_work_count() > 0
        health = await (await client.get("/health/detailed")).json()
        assert health["api_server"]["active_runs"] > 0
        chunk = await post(client, f"/processes/{receipt['operation_id']}/output", {})
        assert chunk["partial"] and not chunk["complete"]
        assert base64.b64decode(chunk["stdout"]["data_base64"]) == b"partial\xff"
        cancelled = await post(client, "/processes/cancel", {}, "crash")
        assert cancelled["termination_confirmed"], cancelled
        assert cancelled["status"] == "interrupted"
        assert adapter.active_agent_work_count() == 0
    finally:
        for session in process_registry.list_sessions(include_retained=True):
            if session["owner_task_id"] == f"workspace:{owner}":
                process_registry.kill_process(session["session_id"])


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_acknowledgement_is_scoped_and_removes_only_consumed_output(api):
    client, adapter, _root = api
    first = await post(client, "/processes", {"command": "printf result"}, "ack")
    operation_id = first["operation_id"]
    await next(iter(adapter._workspace_services.values())).tasks[operation_id]
    path = f"/api/workspace/operations/{operation_id}/acknowledge"
    for body, key in [({"session_id": "two"}, client.workspace_keys["ack"]),
                      ({"session_id": "one"}, client.workspace_keys["ack"] + "wrong")]:
        denied = await client.post(path, json=body, headers={"Idempotency-Key": key})
        assert denied.status == 409
    before = await post(client, f"/processes/{operation_id}/output", {})
    assert base64.b64decode(before["stdout"]["data_base64"]) == b"result"
    acknowledged = await post(client, f"/operations/{operation_id}/acknowledge", {}, "ack")
    assert acknowledged["acknowledged"] and not acknowledged["output_available"]
    assert await post(client, f"/operations/{operation_id}/acknowledge", {}, "ack") == acknowledged
    response = await client.post(f"/api/workspace/processes/{operation_id}/output", json={"session_id": "one"})
    assert response.status == 410
    assert (await response.json())["error"]["code"] == "output_acknowledged"


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_disconnected_write_remains_visible_and_replay_does_not_repeat(api, monkeypatch):
    import asyncio
    import threading
    from tests.waiting import async_wait_until

    client, adapter, root = api
    workspace.capability(adapter)
    service = next(iter(adapter._workspace_services.values()))
    original_write = service.files.write
    started, release = threading.Event(), threading.Event()
    writes = []

    def delayed_write(path, data):
        started.set()
        assert release.wait(15)
        writes.append(path)
        original_write(path, data)

    monkeypatch.setattr(service.files, "write", delayed_write)
    body = {"path": "disconnected", "data_base64": "Ynl0ZXM="}
    request = asyncio.create_task(post(client, "/files/write", body, "disconnect"))
    try:
        await async_wait_until(started.is_set, timeout=15)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert adapter.active_agent_work_count() > 0
    finally:
        release.set()
    await async_wait_until(lambda: all(task.done() for task in service.tasks.values()), timeout=15)
    assert (await post(client, "/files/write", body, "disconnect"))["status"] == "completed"
    assert writes == ["disconnected"]
    assert (root / "disconnected").read_bytes() == b"bytes"


@pytest.mark.asyncio
@pytest.mark.platforms("posix")
async def test_process_timeout_retains_bounded_partial_output(api):
    client, adapter, _root = api
    code = "import os,threading; os.write(1,b'abcdefghijk'); threading.Event().wait()"
    receipt = await post(client, "/processes", {
        "command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}",
        "max_output_bytes": 8, "timeout_seconds": 1}, "timeout")
    operation_id = receipt["operation_id"]
    await next(iter(adapter._workspace_services.values())).tasks[operation_id]
    final = await post(client, f"/processes/{operation_id}/status", {})
    assert final["status"] == "failed" and final["reason"] == "timeout", final
    assert final["termination_confirmed"]
    chunk = await post(client, f"/processes/{operation_id}/output", {"max_bytes": 3})
    assert base64.b64decode(chunk["stdout"]["data_base64"]) == b"abc"
    assert chunk["stdout"]["next_cursor"] == 3 and chunk["stdout"]["truncated"]
    assert chunk["complete"]
