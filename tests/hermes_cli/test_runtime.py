"""Invariant tests for named per-bot execution runtimes."""

from argparse import Namespace

import pytest
import yaml

from hermes_cli.config import _LOAD_CONFIG_CACHE, _RAW_CONFIG_CACHE, load_config
from hermes_cli.runtime import (
    RuntimeError_,
    apply_spec_to_terminal,
    assigned_runtime_block,
    assign_runtime,
    build_runtime_spec,
    catalog,
    exportable_spec,
    register_runtime,
    runtime_command,
    runtime_status_payload,
    stop_profile_runtime,
)


def _isolate(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(home / "missing-managed"))
    from hermes_cli import managed_scope

    _LOAD_CONFIG_CACHE.clear()
    _RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    home.mkdir(parents=True, exist_ok=True)


def _args(**kwargs):
    defaults = {
        "runtime_action": None,
        "name": "",
        "kind": "docker",
        "workspace": "",
        "image": "",
        "volume": [],
        "cpu": None,
        "memory": None,
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": None,
        "json": False,
        "yes": False,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def test_assign_applies_to_this_profile_home_only(tmp_path, monkeypatch):
    home_a = tmp_path / "bot-a"
    home_b = tmp_path / "bot-b"
    _isolate(home_a, monkeypatch)
    register_runtime("box", build_runtime_spec(
        "docker", workspace="/work/a", image="img:a", container_cpu=2,
    ))
    assign_runtime("box")
    cfg_a = load_config()["terminal"]
    assert cfg_a["runtime"] == "box"
    assert cfg_a["backend"] == "docker"
    assert cfg_a["cwd"] == "/work/a"
    assert cfg_a["docker_image"] == "img:a"
    assert "box" in catalog()

    _isolate(home_b, monkeypatch)
    cfg_b = load_config()["terminal"]
    assert not cfg_b.get("runtime")
    assert catalog() == {}
    assert cfg_b.get("backend") != "docker" or cfg_b.get("cwd") != "/work/a"


def test_catalog_refuses_secrets_and_cloud_provisioning(tmp_path, monkeypatch):
    _isolate(tmp_path / "home", monkeypatch)
    dirty = exportable_spec({
        "kind": "ssh",
        "ssh_host": "10.0.0.5",
        "ssh_user": "ubuntu",
        "ssh_key": "/secret/id_ed25519",
        "password": "nopenope",
    })
    assert "ssh_key" not in dirty
    assert "password" not in dirty
    assert dirty["ssh_host"] == "10.0.0.5"

    with pytest.raises(RuntimeError_, match="does not provision"):
        build_runtime_spec("modal")
    with pytest.raises(RuntimeError_, match="does not provision"):
        build_runtime_spec("daytona")

    register_runtime("sshbox", build_runtime_spec(
        "ssh", workspace="~/proj", ssh_host="10.0.0.5", ssh_user="ubuntu",
    ))
    raw = yaml.safe_load((tmp_path / "home" / "config.yaml").read_text())
    spec = raw["terminal"]["runtimes"]["sshbox"]
    assert "ssh_key" not in spec
    assert spec["ssh_host"] == "10.0.0.5"


def test_test_and_stop_stay_on_this_profile(tmp_path, monkeypatch):
    _isolate(tmp_path / "home", monkeypatch)
    register_runtime("localbox", build_runtime_spec("local", workspace="/tmp/ws"))
    assign_runtime("localbox")
    assert runtime_command(_args(runtime_action="test", name="localbox")) == 0
    assert runtime_command(_args(runtime_action="ls", json=True)) == 0

    calls = []

    def _fake_cleanup(task_id, *, force_remove=False):
        calls.append((task_id, force_remove))

    monkeypatch.setattr(
        "hermes_cli.runtime.list_profile_containers",
        lambda: [{"id": "abc123deadbeef", "status": "Up", "image": "img", "name": "h", "profile": "default"}],
    )
    monkeypatch.setattr("hermes_cli.runtime._find_docker", lambda: None)
    monkeypatch.setattr("hermes_cli.runtime._cleanup_vm", _fake_cleanup)

    assert runtime_command(_args(runtime_action="stop")) == 1
    result = stop_profile_runtime(force_remove=True)
    assert calls == [("default", True)]
    assert result["stopped"] == ["abc123deadbeef"]
    assert result["profile"]


def test_unready_assigned_runtime_blocks_bot_start(tmp_path, monkeypatch):
    _isolate(tmp_path / "home", monkeypatch)
    register_runtime("missing-docker", build_runtime_spec("docker", workspace="/w"))
    assign_runtime("missing-docker")
    monkeypatch.setattr(
        "hermes_cli.runtime.probe_spec",
        lambda spec: ("needs_setup", "Docker daemon not reachable — start Docker and retry."),
    )
    block = assigned_runtime_block()
    assert block is not None
    assert "missing-docker" in block["reason"]
    assert "Desktop host" in block["reason"]
    assert runtime_command(_args(runtime_action="test", name="missing-docker")) == 1

    payload = runtime_status_payload()
    assert payload["assigned"] == "missing-docker"
    assert payload["backend"] == "docker"


def test_apply_spec_does_not_copy_secret_keys():
    terminal = {"backend": "local"}
    apply_spec_to_terminal(terminal, {
        "kind": "ssh",
        "ssh_host": "box",
        "ssh_user": "u",
        "ssh_key": "should-not-land",
    })
    assert terminal["backend"] == "ssh"
    assert terminal["ssh_host"] == "box"
    assert "ssh_key" not in terminal
