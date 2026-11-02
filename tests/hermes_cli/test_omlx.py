"""Managed oMLX contracts, with process/network/install boundaries substituted."""

from pathlib import Path
from unittest.mock import Mock, MagicMock

import psutil
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.local_runtime import omlx
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(omlx, "root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(omlx, "models_dir", lambda: tmp_path / "models")
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


@pytest.mark.platforms("macos", arch="arm64")
def test_select_is_only_a_profile_preference(client):
    from hermes_cli import config
    config.save_config({"model": {"provider": "custom", "default": "mine", "base_url": "http://localhost:9999/v1"}})
    before = config.load_config()["model"]
    response = client.post("/api/local-models/omlx/select", json={"engine": "omlx"})
    assert response.status_code == 200
    assert config.load_config()["model"] == before
    assert config.load_config()["local_runtime"]["engine"] == "omlx"
    assert client.get("/api/local-models/omlx/status").json()["selected"] is True
    assert client.post("/api/local-models/omlx/select", json={"engine": "invalid"}).status_code == 422


def test_select_rejects_unsupported_capability_without_changing_config(client, monkeypatch):
    from hermes_cli import config
    from hermes_cli.local_runtime import omlx
    monkeypatch.setattr(omlx, "supported", lambda: False)
    config.save_config({"local_runtime": {"engine": "llamacpp"}})
    before = config.load_config()
    response = client.post("/api/local-models/omlx/select", json={"engine": "omlx"})
    assert response.status_code == 400
    assert config.load_config() == before
    assert client.post("/api/local-models/omlx/select", json={"engine": "llamacpp"}).status_code == 200


def test_routes_require_auth(client):
    client.headers.clear()
    assert client.get("/api/local-models/omlx/status").status_code == 401
    assert client.post("/api/local-models/omlx/install").status_code == 401
    assert client.post("/api/local-models/download/cancel", json={"job_id": "unowned"}).status_code == 401


def test_discovery_excludes_incomplete_models(client):
    from hermes_cli.local_runtime import omlx
    for name in ("ready", ".partial", "no-weights"):
        path = omlx.models_dir() / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        if name != "no-weights":
            (path / "model.safetensors").touch()
    assert omlx.local_models() == ["ready"]
    data = client.get("/api/local-models/omlx/status").json()
    assert data["models"] == [{"id": "ready", "active": False, "size_bytes": 0}]
    assert data["installed"] is False and data["running"] is False


def test_omlx_browse_does_not_boot_llama(monkeypatch):
    from hermes_cli.local_runtime import bootstrap
    def fail():
        raise AssertionError("llama discovery must not run")
    monkeypatch.setattr(bootstrap, "adopt_legacy_models", fail)
    assert bootstrap.ensure_local_runtime({"local_runtime": {"enabled": True, "engine": "omlx"}}) is None


def test_active_omlx_boot_ignores_browse_preference(monkeypatch):
    from hermes_cli.local_runtime import bootstrap, omlx
    start = Mock()
    monkeypatch.setattr(omlx, "start", start)
    config = {"model": {"provider": "custom", "base_url": omlx.BASE_URL},
              "local_runtime": {"engine": "llamacpp", "omlx_enabled": True, "enabled": True}}
    assert bootstrap.ensure_local_runtime(config) is None
    start.assert_called_once_with()
    config["local_runtime"]["omlx_enabled"] = False
    bootstrap.ensure_local_runtime(config)
    start.assert_called_once_with()


@pytest.mark.platforms("macos", arch="arm64")
def test_failed_generation_does_not_change_provider(client, monkeypatch):
    from hermes_cli import config
    from hermes_cli.local_runtime import omlx
    config.save_config({"model": {"provider": "custom", "default": "old", "base_url": "http://localhost:9999/v1"}})
    before = config.load_config()
    def fail(model):
        raise RuntimeError("generation failed")
    monkeypatch.setattr(omlx, "validate_model", fail)
    response = client.post("/api/local-models/omlx/activate", json={"model_id": "broken"})
    assert response.status_code == 409
    assert config.load_config() == before


@pytest.mark.platforms("macos", arch="arm64")
def test_activation_uses_custom_assignment_and_retains_llama_settings(client, monkeypatch):
    from hermes_cli import config, web_server_config
    from hermes_cli.model_switch import ModelSwitchResult
    from hermes_cli.local_runtime import omlx
    config.save_config({"local_runtime": {"enabled": True, "backend": "metal"}})
    checked = []
    monkeypatch.setattr(omlx, "validate_model", lambda mid: checked.append(mid))
    monkeypatch.setattr(omlx, "read_state", lambda: {"api_key": "local-test-key"})
    monkeypatch.setattr(web_server_config, "_validated_main_model_selection", lambda *a, **kw:
                        ModelSwitchResult(success=True, new_model="ready", target_provider="custom", base_url=omlx.BASE_URL))
    response = client.post("/api/local-models/omlx/activate", json={"model_id": "ready"})
    assert response.status_code == 200, response.text
    saved = config.load_config()
    assert checked == ["ready"]
    assert saved["model"]["provider"] == "custom"
    assert saved["model"]["base_url"] == omlx.BASE_URL
    assert saved["model"]["default"] == "ready"
    assert saved["local_runtime"]["enabled"] is True
    assert saved["local_runtime"]["backend"] == "metal"
    assert saved["local_runtime"]["omlx_enabled"] is True


@pytest.mark.platforms("macos", arch="arm64")
def test_stop_refuses_live_other_owner(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    proc = Mock()
    monkeypatch.setattr(omlx, "read_state", lambda: {"owner_pid": 999999, "owner_create_time": 1})
    monkeypatch.setattr(omlx, "_process", lambda state: proc)
    monkeypatch.setattr(omlx, "_owner_is_dead", lambda state: False)
    response = client.post("/api/local-models/omlx/server", json={"action": "stop"})
    assert response.status_code == 409
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_external_executable_cannot_be_adopted(client):
    from hermes_cli.local_runtime import omlx
    proc = psutil.Process()
    state = {"pid": proc.pid, "create_time": proc.create_time(), "executable": proc.exe(),
             "owner_pid": proc.pid, "owner_create_time": proc.create_time(), "base_url": omlx.BASE_URL}
    assert omlx._process(state) is None


@pytest.mark.platforms("macos", arch="arm64")
def test_delete_removes_only_the_requested_managed_model(client):
    from hermes_cli import config
    from hermes_cli.local_runtime import omlx

    for name in ("remove-me", "keep-me"):
        path = omlx.models_dir() / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"model")
    before = config.load_config_readonly()
    response = client.post("/api/local-models/omlx/delete", json={"model_id": "remove-me"})
    assert response.status_code == 200, response.text
    assert omlx.local_models() == ["keep-me"]
    assert config.load_config_readonly() == before


@pytest.mark.platforms("macos", arch="arm64")
@pytest.mark.parametrize("protected", ["active", "running", "symlink", "traversal"])
def test_delete_refuses_unsafe_or_in_use_models(client, monkeypatch, tmp_path, protected):
    from hermes_cli import config
    from hermes_cli.local_runtime import omlx

    path = tmp_path / "outside" if protected in {"symlink", "traversal"} else omlx.models_dir() / "ready"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"model")
    mid = "ready"
    if protected == "active":
        config.save_config({"model": {"provider": "custom", "default": mid, "base_url": omlx.BASE_URL}})
    elif protected == "running":
        monkeypatch.setattr(omlx, "_process", lambda state: Mock())
    elif protected == "symlink":
        omlx.models_dir().mkdir(parents=True)
        (omlx.models_dir() / mid).symlink_to(path, target_is_directory=True)
    else:
        mid = str(path)
    response = client.post("/api/local-models/omlx/delete", json={"model_id": mid})
    assert response.status_code == 409, response.text
    assert (path / "model.safetensors").read_bytes() == b"model"


@pytest.mark.platforms("posix")
def test_operation_lock_refuses_racing_mutation(client):
    from hermes_cli.local_runtime import omlx
    with omlx.operation_lock():
        with pytest.raises(RuntimeError, match="in progress"):
            with omlx.operation_lock():
                pytest.fail("two operations acquired the lock")
    with omlx.operation_lock():
        pass


@pytest.mark.platforms("macos", arch="arm64")
def test_download_validation_and_job_contract(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    from tests.waiting import wait_until
    omlx.cli().parent.mkdir(parents=True)
    omlx.cli().touch()
    assert client.post("/api/local-models/omlx/download", json={"repo": "../bad"}).status_code == 422
    received = []
    monkeypatch.setattr(omlx, "download", lambda repo, **kwargs: received.append(repo))
    response = client.post("/api/local-models/omlx/download", json={"repo": "mlx-community/tiny"})
    assert response.status_code == 200
    jid = response.json()["job_id"]
    assert wait_until(lambda: client.get(f"/api/local-models/jobs/{jid}").json()["status"] == "done")
    job = client.get(f"/api/local-models/jobs/{jid}").json()
    assert received == ["mlx-community/tiny"]
    assert job["kind"] == "model-download" and job["target"].startswith("oMLX")
    assert job["model_id"] == "mlx-community--tiny"


@pytest.mark.platforms("macos", arch="arm64")
def test_download_labels_cached_bytes_and_uses_only_network_samples_for_speed(client, monkeypatch):
    from threading import Event
    from types import SimpleNamespace

    from hermes_cli.local_runtime import omlx
    from hermes_cli.web_routers import local_models as jobs
    from hermes_cli.web_routers import omlx as omlx_router
    from tests.waiting import wait_until

    omlx.cli().parent.mkdir(parents=True)
    omlx.cli().touch()
    entered, release = Event(), Event()
    clock = iter([0.0, 2.0])
    monkeypatch.setattr(omlx_router, "time", SimpleNamespace(monotonic=lambda: next(clock)))

    def fetch(repo, *, progress=None, transfer_progress=None, cancel_event=None, before_publish=None):
        if transfer_progress is not None:
            transfer_progress(16)
        if progress is not None:
            progress(32, 128, {})
            progress(96, 128, {})
        if transfer_progress is not None:
            transfer_progress(64)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(omlx, "download", fetch)
    response = client.post("/api/local-models/omlx/download", json={"repo": "mlx-community/tiny"})
    jid = response.json()["job_id"]
    try:
        assert entered.wait(5)
        job = client.get(f"/api/local-models/jobs/{jid}").json()
        assert job["status"] == "running"
        assert job["done_bytes"] == 96 and job["total_bytes"] == 128
        assert job["percent"] == 75
        assert job["cached_bytes"] == 32
        assert job["received_bytes"] == 64
        assert job["bytes_per_sec"] == 24
        assert "eta_seconds" not in job
        assert job["can_pause"] is False
    finally:
        release.set()
    assert wait_until(lambda: client.get(f"/api/local-models/jobs/{jid}").json()["status"] == "done")
    assert jid not in jobs._RUNNING


@pytest.mark.parametrize("cached,total", [(64, 192), (9064196870, 19760308230)])
def test_snapshot_progress_counts_cached_bytes_and_ignores_network_bar(monkeypatch, capsys, cached, total):
    import json
    import sys
    from types import SimpleNamespace

    from hermes_cli.local_runtime import omlx_download

    def snapshot(**kwargs):
        if kwargs.get("dry_run"):
            return [SimpleNamespace(file_size=cached, will_download=False, commit_hash="pinned"),
                    SimpleNamespace(file_size=total-cached, will_download=True, commit_hash="pinned")]
        assert kwargs["revision"] == "pinned"
        progress = kwargs["tqdm_class"]
        with progress(desc="Fetching files", total=2) as files:
            files.update(1)
        remaining = total - cached
        chunk = remaining // 4
        with progress(desc="Downloading bytes", unit="B", total=remaining) as network:
            network.update(remaining)
            network.refresh()
        with progress(desc="Reconstructing (incomplete total...)", unit="B", total=remaining) as model:
            model.update(chunk)
            model.refresh()
            model.update(remaining - chunk)
            model.refresh()

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    omlx_download.download("owner/model", "isolated-model")
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line.removeprefix("HERMES_OMLX_PROGRESS "))
              for line in lines if line.startswith("HERMES_OMLX_PROGRESS ")]
    transfers = [json.loads(line.removeprefix("HERMES_OMLX_TRANSFER "))
                 for line in lines if line.startswith("HERMES_OMLX_TRANSFER ")]
    assert events[0] == [cached, total] and events[-1] == [total, total]
    assert [cached + (total - cached) // 4, total] in events
    assert all(size == total and cached <= done <= size for done, size in events)
    assert transfers and transfers[-1] == total - cached


@pytest.mark.platforms("macos", arch="arm64")
@pytest.mark.parametrize("fail", [False, True])
def test_download_streams_progress_before_publishing(client, monkeypatch, fail):
    import subprocess
    import sys

    from hermes_cli.local_runtime import omlx

    python = omlx.cli().with_name("omlx-cluster-python")
    python.parent.mkdir(parents=True)
    python.touch()
    destination = omlx.models_dir() / "mlx-community--tiny"
    stage = omlx.models_dir() / ".mlx-community--tiny"
    real_popen = subprocess.Popen
    processes, events, received, published = [], [], [], []
    script = '''
import json, sys, time
from pathlib import Path
from threading import Event
stage = Path(sys.argv[1])
(stage / 'config.json').write_text('{}')
(stage / 'model.safetensors').write_bytes(b'x' * 32)
print('HERMES_OMLX_TRANSFER 16', flush=True)
print('HERMES_OMLX_PROGRESS ' + json.dumps([32, 128]), flush=True)
deadline = time.monotonic() + 5
while not (stage / 'observed').exists():
    assert time.monotonic() < deadline
    Event().wait(0.01)
if sys.argv[2] == 'True':
    print('fixture transfer failed', file=sys.stderr)
    sys.exit(2)
(stage / 'model.safetensors').write_bytes(b'x' * 128)
print('HERMES_OMLX_TRANSFER 64', flush=True)
print('HERMES_OMLX_PROGRESS ' + json.dumps([128, 128]), flush=True)
'''

    def fetch(args, **kwargs):
        process = real_popen([sys.executable, "-c", script, args[-1], str(fail)], **kwargs)
        processes.append(process)
        return process

    def progress(done, total, ranges):
        assert not destination.exists()
        if done == 32:
            assert processes[0].poll() is None
            (stage / "observed").touch()
        events.append((done, total, ranges))

    def before_publish():
        assert processes[0].poll() is not None
        assert not destination.exists()
        published.append(True)

    monkeypatch.setattr(omlx.subprocess, "Popen", fetch)
    if fail:
        with pytest.raises(RuntimeError, match="exit 2") as error:
            omlx.download("mlx-community/tiny", progress=progress, transfer_progress=received.append)
        assert "fixture transfer failed" in str(error.value)
        assert omlx.local_models() == []
        assert stage.is_dir()
        assert events == [(32, 128, {})]
        assert received == [16]
    else:
        assert omlx.download("mlx-community/tiny", progress=progress, transfer_progress=received.append,
                             before_publish=before_publish) == "mlx-community--tiny"
        assert published == [True]
        assert omlx.local_models() == ["mlx-community--tiny"]
        assert events == [(32, 128, {}), (128, 128, {})]
        assert received == [16, 64]
        assert (destination / "model.safetensors").stat().st_size == 128
    assert processes[0].poll() is not None


@pytest.mark.platforms("macos", arch="arm64")
def test_cancel_download_reaps_worker_and_keeps_partial_snapshot(client, monkeypatch):
    import subprocess
    import sys

    from hermes_cli.local_runtime import omlx
    from hermes_cli.web_routers import local_models as jobs
    from tests.waiting import wait_until

    omlx.cli().parent.mkdir(parents=True)
    omlx.cli().touch()
    omlx.cli().with_name("omlx-cluster-python").touch()
    real_popen, children = subprocess.Popen, []
    script = '''
from pathlib import Path
from threading import Event
import sys
stage = Path(sys.argv[1])
(stage / 'config.json').write_text('{}')
(stage / 'model.safetensors').write_bytes(b'x' * 32)
print('HERMES_OMLX_PROGRESS [32, 128]', flush=True)
Event().wait(30)
'''

    def spawn(args, **kwargs):
        if str(args[1]).endswith("omlx_download.py"):
            child = real_popen([sys.executable, "-c", script, args[-1]], **kwargs)
            children.append(child)
            return child
        return real_popen(args, **kwargs)

    monkeypatch.setattr(omlx.subprocess, "Popen", spawn)
    response = client.post("/api/local-models/omlx/download", json={"repo": "owner/cancel-test"})
    jid = response.json()["job_id"]
    try:
        assert wait_until(lambda: jobs._JOBS[jid]["done_bytes"] == 32, timeout=5)
        assert client.get(f"/api/local-models/jobs/{jid}").json()["can_cancel"]
        cancelled = client.post("/api/local-models/download/cancel", json={"job_id": jid})
        assert cancelled.status_code == 200 and cancelled.json()["cancel_requested"]
        assert wait_until(lambda: jobs._JOBS[jid]["status"] == "cancelled", timeout=5)
        job = client.get(f"/api/local-models/jobs/{jid}").json()
        assert job["error"] is None and not job["can_cancel"]
        assert children[0].poll() is not None
        assert jid not in jobs._RUNNING
        assert (omlx.models_dir() / ".owner--cancel-test/model.safetensors").read_bytes() == b"x" * 32
        assert not (omlx.models_dir() / "owner--cancel-test").exists()
        with omlx.operation_lock("download"):
            pass
        assert jid in {j["job_id"] for j in client.get("/api/local-models/jobs").json()["jobs"]}
        assert not client.post("/api/local-models/download/cancel", json={"job_id": jid}).json()["cancel_requested"]
        assert client.post("/api/local-models/download/cancel", json={"job_id": "missing"}).status_code == 404
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait()


@pytest.mark.platforms("macos", arch="arm64")
def test_cancel_is_refused_once_download_publication_starts(client, monkeypatch):
    from threading import Event

    from hermes_cli.local_runtime import omlx
    from hermes_cli.web_routers import local_models as jobs
    from tests.waiting import wait_until

    omlx.cli().parent.mkdir(parents=True)
    omlx.cli().touch()
    entered, release = Event(), Event()
    replies = []

    def fetch(repo, *, before_publish=None, **kwargs):
        entered.set()
        assert release.wait(5)
        if before_publish is not None:
            before_publish()
        replies.append(client.post("/api/local-models/download/cancel", json={"job_id": jid}).json())

    monkeypatch.setattr(omlx, "download", fetch)
    jid = client.post("/api/local-models/omlx/download", json={"repo": "owner/completed"}).json()["job_id"]
    try:
        assert entered.wait(5)
    finally:
        release.set()
    assert wait_until(lambda: jobs._JOBS[jid]["status"] != "running", timeout=5)
    assert replies == [{"ok": True, "cancel_requested": False}]
    assert jobs._JOBS[jid]["status"] == "done"


@pytest.mark.platforms("macos", arch="arm64")
def test_model_validation_requires_generated_text(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    monkeypatch.setattr(omlx, "local_models", lambda: ["ready"])
    monkeypatch.setattr(omlx, "start", lambda: None)
    monkeypatch.setattr(omlx, "request", lambda path, *a, **kw:
                        {"data": [{"id": "ready"}]} if path == "/models" else {"choices": [{"message": {"content": ""}}]})
    with pytest.raises(RuntimeError, match="no text"):
        omlx.validate_model("ready")


def test_custom_runtime_autostarts_only_managed_selected_endpoint(client, monkeypatch):
    from hermes_cli import config, runtime_provider
    from hermes_cli.local_runtime import omlx
    config.save_config({"model": {"provider": "custom", "default": "ready", "base_url": omlx.BASE_URL},
                        "local_runtime": {"omlx_enabled": True}})
    start = Mock()
    monkeypatch.setattr(omlx, "start", start)
    monkeypatch.setattr(runtime_provider, "_ladder_rungs", lambda *args: iter([
        {"provider": "custom", "api_key": "local-key", "base_url": omlx.BASE_URL, "api_mode": "chat_completions"}]))
    result = runtime_provider.resolve_runtime_provider(requested="custom")
    assert result["base_url"] == omlx.BASE_URL
    start.assert_called_once_with()
    cfg = config.load_config()
    cfg["local_runtime"]["omlx_enabled"] = False
    config.save_config(cfg)
    runtime_provider.resolve_runtime_provider(requested="custom")
    start.assert_called_once_with()


@pytest.mark.platforms("macos", arch="arm64")
def test_install_uses_verified_bundle_without_python_environment_mutation(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    import pm.downloader
    sources = []
    def fetch(self, progress=None):
        source = sources[0]
        source.dest.write_bytes(b"test image")
    def downloader(items):
        sources.extend(items)
        return Mock(run=lambda progress=None: fetch(None, progress))
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        if args[0] == "/usr/bin/ditto":
            path = Path(args[-1]) / "Contents/MacOS/omlx-cli"
            path.parent.mkdir(parents=True)
            path.touch()
        return Mock(returncode=0)
    monkeypatch.setattr(pm.downloader, "Download", downloader)
    monkeypatch.setattr(omlx.subprocess, "run", run)
    omlx.install()
    assert sources[0].sha256 == omlx.SHA256
    assert sources[0].url == omlx.ARCHIVE
    assert omlx.cli().is_file()
    assert commands[0][0:2] == ["/usr/bin/hdiutil", "attach"]
    assert commands[-1][0:2] == ["/usr/bin/hdiutil", "detach"]
    assert len(commands) == 4  # attach, copy, verify entrypoint, detach; no installer executes.


@pytest.mark.platforms("macos", arch="arm64")
def test_start_refreshes_executable_after_shell_exec(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    omlx.cli().parent.mkdir(parents=True)
    omlx.cli().touch()
    shell = Mock()
    shell.exe.return_value = "/bin/sh"
    python = Mock()
    python.exe.return_value = str(omlx.cli().parent / "python3")
    python.create_time.return_value = 2
    owner = Mock(pid=99)
    owner.create_time.return_value = 1
    instances = iter([shell, python])
    monkeypatch.setattr(omlx.psutil, "Process", lambda pid=None: owner if pid is None else next(instances, python))
    monkeypatch.setattr(omlx.subprocess, "Popen", Mock(return_value=Mock(pid=100, poll=lambda: None)))
    monkeypatch.setattr(omlx.socket, "socket", MagicMock())
    monkeypatch.setattr(omlx.time, "monotonic", Mock(side_effect=[0, 1, 2, 121]))
    monkeypatch.setattr(omlx.time, "sleep", Mock())
    monkeypatch.setattr(omlx.atexit, "register", Mock())
    request = Mock(return_value={"data": []})
    monkeypatch.setattr(omlx, "request", request)
    omlx.start()
    request.assert_called_once_with("/models")
    assert omlx.read_state()["executable"] == python.exe()
