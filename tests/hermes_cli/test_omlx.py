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
    assert data["models"] == [{"id": "ready", "active": False}]
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
    monkeypatch.setattr(omlx, "download", lambda repo: received.append(repo))
    response = client.post("/api/local-models/omlx/download", json={"repo": "mlx-community/tiny"})
    assert response.status_code == 200
    jid = response.json()["job_id"]
    assert wait_until(lambda: client.get(f"/api/local-models/jobs/{jid}").json()["status"] == "done")
    job = client.get(f"/api/local-models/jobs/{jid}").json()
    assert received == ["mlx-community/tiny"]
    assert job["kind"] == "model-download" and job["target"].startswith("oMLX")
    assert job["model_id"] == "mlx-community--tiny"


@pytest.mark.platforms("macos", arch="arm64")
def test_download_publishes_only_complete_snapshot(client, monkeypatch):
    from hermes_cli.local_runtime import omlx
    python = omlx.cli().with_name("omlx-cluster-python")
    python.parent.mkdir(parents=True)
    python.touch()
    def fetch(args, **kwargs):
        stage = Path(args[-1])
        assert stage.name.startswith(".")
        (stage / "config.json").write_text("{}")
        (stage / "model.safetensors").touch()
        return Mock(returncode=0)
    monkeypatch.setattr(omlx.subprocess, "run", fetch)
    assert omlx.download("mlx-community/tiny") == "mlx-community--tiny"
    assert omlx.local_models() == ["mlx-community--tiny"]


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
