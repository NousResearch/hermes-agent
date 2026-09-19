"""CLI, authenticated REST and WebSocket RPC preserve source plugin packages.

This is package parity, not a promise of native memory identity isolation. The
CLI retains its historical Honcho post-create behavior.
"""
from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import subprocess

import pytest
import yaml

from agent import secret_scope
from hermes_cli import profiles
from hermes_constants import (
    get_hermes_home, reset_hermes_home_override, set_hermes_home_override,
)


def _package(home, label):
    package = home / "plugins" / "parity_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'from pathlib import Path\n'
        'Path(__file__).with_name("runtime-imported").touch()\n'
        '# MemoryProvider\nraise AssertionError("runtime activation forbidden")\n')
    (package / "clone.py").write_text('raise AssertionError("no native clone protocol")\n')
    (package / "payload.txt").write_text(label)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
               GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
    def git(*args):
        return subprocess.run(["git", "-C", str(package), *args], env=env, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init")
    git("add", ".")
    git("commit", "-m", "fixture")
    origin = "https://example.invalid/parity-probe.git"
    git("remote", "add", "origin", origin)
    from hermes_cli.plugin_inventory import capture_install_inventory
    (package / ".hermes-catalog.json").write_text(json.dumps({
        "catalog_name": "parity-probe", "repo": origin, "sha": git("rev-parse", "HEAD")}))
    (home / "plugins" / ".install-metadata.json").write_text(json.dumps({
        "parity_probe": {"source": origin, "revision": git("rev-parse", "HEAD"), "pinned": True,
                         "files": capture_install_inventory(package, package)}}))


@pytest.fixture
def clone_surface_env(tmp_path, monkeypatch, surface):
    from hermes_cli import gateway_multiplex_served
    from hermes_cli.config import DEFAULT_CONFIG

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".hermes"
    homes = {"default": root, **{
        name: root / "profiles" / name for name in ("source-a", "source-b", "caller")}}
    voice = {
        "default": {"tts": {"provider": "edge", "edge": {"voice": "default-voice"}}},
        "source-a": {"stt": {"provider": "local", "local": {"model": "source-a-model"}}},
        "source-b": {"tts": {"provider": "edge", "edge": {"voice": "source-b-voice"}}},
        "caller": {"stt": {"provider": "openai"}, "tts": {"provider": "openai"},
                   "voice": {"auto_tts": True}},
    }
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump({
            "_config_version": DEFAULT_CONFIG["_config_version"],
            "memory": {"provider": "parity_probe"},
            "plugins": {"enabled": [], "parity_probe": {"enabled": False}},
            "model": {"provider": "custom", "default": f"{name}-model",
                      "base_url": f"https://{name}.example.invalid/v1"},
            **voice[name]}))
        (home / ".env").write_text(
            f"PARITY_SHARED_KEY={name}-key\n"
            f"PARITY_{name.upper().replace('-', '_')}_ONLY_KEY={name}-only-key\n")
        (home / "auth.json").write_text(json.dumps({
            "providers": {}, "credential_pool": {"openrouter": [
                {"id": name, "auth_type": "api_key", "api_key": f"{name}-auth-key"}]}}))
        (home / "SOUL.md").write_text(f"{name} personality\n")
        (home / "parity_probe.json").write_text(json.dumps({"peer": name}))
        skill = home / "skills" / "curated" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(f"{name} curated skill\n")
        _package(home, name)
    caller = homes["caller"]
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", lambda: None)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: None)
    seeded = []
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, **kw: seeded.append(path))
    notifications = []
    def notified(name):
        destination = profiles.get_profile_dir(name)
        assert (destination / "plugins" / "parity_probe" / "payload.txt").is_file()
        assert json.loads((destination / ".clone-report.json").read_text())["plugins"]["copied"] == ["parity_probe"]
        notifications.append(name)
    monkeypatch.setattr(profiles, "_notify_multiplexer", notified)
    from hermes_cli import web_server
    from tui_gateway import server
    home_token = set_hermes_home_override(caller)
    secret_token = secret_scope.set_secret_scope({"PARITY_SHARED_KEY": "caller-key"})
    monkeypatch.setenv("PARITY_SHARED_KEY", "ambient-key")
    monkeypatch.setenv("PARITY_AMBIENT_ONLY_KEY", "ambient-only-key")
    # ASGI transport uses no TCP. Fail even if a best-effort hook swallows the error.
    import socket
    connections = []
    connect = socket.socket.connect
    def offline_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            connections.append(address)
            raise AssertionError("clone must not contact a provider")
        return connect(sock, address)
    monkeypatch.setattr(socket.socket, "connect", offline_connect)
    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    from starlette.testclient import TestClient
    try:
        if surface == "rpc":
            with TestClient(web_server.app).websocket_connect(f"/api/ws?token={web_server._SESSION_TOKEN}") as connection:
                assert connection.receive_json()["params"]["type"] == "gateway.ready"
                yield homes, notifications, seeded, connection
        else:
            yield homes, notifications, seeded, None
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        assert connections == []
        assert not list(root.rglob("runtime-imported"))


def _create(surface, name, full, capsys, connection, *, source, owner):
    payload = {"name": name, "clone_all": full}
    if source is not None:
        payload["clone_from"] = source
    if surface == "cli":
        from hermes_cli.profile_cmd import _profile_create
        from hermes_cli.web_server_profiles import _config_profile_scope
        with _config_profile_scope(owner):
            _profile_create(Namespace(profile_name=name, clone_from=source, clone=not full,
                                      clone_all=full, no_alias=True))
        return capsys.readouterr().out
    if surface == "rest":
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        # No app lifespan: this route needs neither gateway services nor background tickers.
        client = TestClient(app)
        try:
            response = client.post("/api/profiles", headers={_SESSION_HEADER_NAME: _SESSION_TOKEN},
                                   json=payload)
        finally:
            client.close()
        assert response.status_code == 200, response.text
        assert response.json()["ok"] is True
        assert response.json()["model_set"] is False
        return response.text
    connection.send_json({"jsonrpc": "2.0", "id": name, "method": "profiles.create",
                          "params": {**payload, "profile": owner}})
    while True:
        response = connection.receive_json()
        if response.get("id") == name:
            break
    assert "error" not in response, response
    assert response["result"]["ok"] is True
    assert response["result"]["model_set"] is False
    assert response["result"]["mirrored"] == {
        "env": False, "auth": False, "model_inherited": False, "voice": False}
    return json.dumps(response)


def _assert_clone(destination, source, full, output):
    from dotenv import dotenv_values

    source_cfg = yaml.safe_load((source / "config.yaml").read_text())
    destination_cfg = yaml.safe_load((destination / "config.yaml").read_text())
    for section in ("model", "memory", "plugins", "stt", "tts", "voice"):
        assert destination_cfg.get(section) == source_cfg.get(section), section
    source_env = dotenv_values(source / ".env")
    destination_env = dotenv_values(destination / ".env")
    # Discovery may add baseline scaffolding; fixture credentials must still be source-only.
    assert {k: v for k, v in destination_env.items() if k.startswith("PARITY_")} == source_env
    assert "PARITY_AMBIENT_ONLY_KEY" not in destination_env
    if full:
        assert json.loads((destination / "auth.json").read_text()) == json.loads((source / "auth.json").read_text())
        assert (destination / "parity_probe.json").read_bytes() == (source / "parity_probe.json").read_bytes()
    else:
        assert not (destination / "auth.json").exists()
        assert not (destination / "parity_probe.json").exists()
    package_path = Path("plugins/parity_probe/payload.txt")
    assert (destination / package_path).read_bytes() == (source / package_path).read_bytes()
    assert not (destination / "plugins/parity_probe/.git").exists()
    for relative in ("SOUL.md", "skills/curated/SKILL.md"):
        assert (destination / relative).read_bytes() == (source / relative).read_bytes()
    report = json.loads((destination / ".clone-report.json").read_text())
    assert report["plugins"]["copied"] == ["parity_probe"]
    assert report["plugins"]["warnings"]
    assert "needs_auth" not in report
    assert "clone_needs_auth" not in output
    assert not list(destination.parent.glob(f".{destination.name}.staging-*"))
    original = (source / package_path).read_bytes()
    (destination / package_path).write_text("independent")
    assert (source / package_path).read_bytes() == original


@pytest.mark.parametrize("surface", ["cli", "rest", "rpc"])
@pytest.mark.parametrize("full", [False, True])
def test_explicit_clone_source_is_not_replaced_by_caller(clone_surface_env, surface, full, capsys):
    homes, notifications, seeded, connection = clone_surface_env
    preserved = {home / relative: (home / relative).read_bytes()
                 for home in homes.values()
                 for relative in ("config.yaml", ".env", "auth.json", "parity_probe.json")}
    # Distinct caller/owner/source: RPC requests owners A→B→A on one connection.
    # HTTP's create route is machine-level; clone_from is its source selector.
    for index, (owner, source_name) in enumerate((
        ("source-a", "source-b"), ("source-b", "source-a"), ("source-a", "source-b"),
    )):
        name = f"destination-{index}"
        output = _create(surface, name, full, capsys, connection, source=source_name, owner=owner)
        _assert_clone(profiles.get_profile_dir(name), homes[source_name], full, output)
        assert get_hermes_home() == homes["caller"]
        assert secret_scope.get_secret("PARITY_SHARED_KEY") == "caller-key"
        assert os.environ["PARITY_SHARED_KEY"] == "ambient-key"
        assert {path: path.read_bytes() for path in preserved} == preserved
    assert notifications == ["destination-0", "destination-1", "destination-2"]
    assert seeded == []


@pytest.mark.parametrize("surface", ["cli", "rest", "rpc"])
def test_clone_all_without_source_retains_surface_source_selection(clone_surface_env, surface, capsys):
    homes, notifications, seeded, connection = clone_surface_env
    for index, owner in enumerate(("source-a", "source-b", "source-a")):
        name = f"implicit-{index}"
        output = _create(surface, name, True, capsys, connection, source=None, owner=owner)
        # REST historically chooses default; CLI/RPC choose their active owner.
        source = homes["default" if surface == "rest" else owner]
        _assert_clone(profiles.get_profile_dir(name), source, True, output)
        assert get_hermes_home() == homes["caller"]
        assert secret_scope.get_secret("PARITY_SHARED_KEY") == "caller-key"
    assert notifications == ["implicit-0", "implicit-1", "implicit-2"]
    assert seeded == []
