"""Offline clone invariants through native resolution and relocated activation."""
from contextlib import contextmanager
from dataclasses import asdict
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import sys
import subprocess
import tempfile

import pytest

from agent.secret_scope import load_env_file, reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.profile_clone import clone_source_scope
from plugins.memory.surfaces import load_provider_companion


@contextmanager
def scope(home):
    home_token = set_hermes_home_override(home)
    token = set_secret_scope(load_env_file(home / ".env"))
    try:
        yield
    finally:
        reset_secret_scope(token)
        reset_hermes_home_override(home_token)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def deny_network(*args, **kwargs):
    raise AssertionError("clone must not contact a provider")


@pytest.mark.parametrize("case", ["host", "legacy", "root", "default-host", "override",
    "default-file", "global-file", "env", "empty", "disabled", "oauth-host",
    "oauth-root", "oauth-env", "host-static", "lower-oauth", "symlink", "invalid-url", "null-enabled",
    "ambient-base-url", "ambient-sdk-url", "env-sdk-url", "observation-null-peer", "observation-unknown",
    "refresh-root", "refresh-env", "opaque-root-oauth", "missing-root-oauth", "missing-host-oauth",
    "null-key", "nested-falsy"])
@pytest.mark.parametrize("clone_all", [False, True])
def test_clone_preserves_effective_settings_without_sharing_grants(tmp_path, monkeypatch, case, clone_all):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    source_name = "default" if case == "default-host" else "alpha"
    source = root if source_name == "default" else root / "profiles" / source_name
    destination = root / "profiles" / "beta"
    stage = root / "profiles" / ".staging-beta"
    source.mkdir(parents=True)
    stage.mkdir(parents=True)
    caller = root / "profiles" / "caller"
    caller.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("HONCHO_API_KEY", "unrelated-process-key")
    monkeypatch.setenv("HERMES_HONCHO_HOST", "unrelated-process-host")
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("HONCHO_BASE_URL", "https://caller-base.invalid")
    monkeypatch.setenv("HONCHO_URL", "https://caller-sdk.invalid")
    monkeypatch.setenv("HONCHO_ENVIRONMENT", "caller-environment")
    if case == "ambient-sdk-url":
        monkeypatch.delenv("HONCHO_BASE_URL")
    monkeypatch.delenv("HONCHO_TIMEOUT", raising=False)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    companion = load_provider_companion("honcho", "clone")
    assert companion is not None, "Honcho must expose offline clone preparation"
    client = importlib.import_module(companion.__package__ + ".client")
    host = client.profile_host_key(source_name)
    block = {
        "peerName": "owner", "runtimePeerPrefix": "", "userPeerAliases": {},
        "pinPeerName": False, "saveMessages": False, "contextTokens": 0,
        "contextCadence": 0, "dialecticCadence": 0, "dialecticDynamic": False,
        "dialecticDepth": 3, "dialecticDepthLevels": [], "dialecticMaxChars": 0,
        "dialecticMaxInputChars": 0, "messageMaxChars": 0, "reasoningHeuristic": False,
        "reasoningLevelCap": "max", "dialecticReasoningLevel": "medium", "queryRewrite": True,
        "recallSync": True, "a2aSessions": False, "initOnSessionStart": False,
        "injectionFrequency": "first-turn", "firstTurnBaseWait": 0, "firstTurnDialecticWait": 0,
        "requestTimeout": 7, "timeout": "", "writeFrequency": "session",
        "sessionStrategy": "per-session", "sessionAiPeerPrefix": False, "sessionPeerPrefix": False,
        "observation": {"ai": {"observeMe": False}, "user": {"observeOthers": False}},
        "injection": {"sessionStart": []}, "logging": False, "recallMode": "tools",
        "aiPeer": "source-ai", "unknownSecret": "discard-me",
    }
    raw = {"apiKey": "root-static", "workspace": "shared-workspace", "peerName": "root-owner",
           "pinUserPeer": True, "runtimePeerPrefix": "root-prefix", "userPeerAliases": {"42": "alias"},
           "contextTokens": 400, "timeout": 20, "endpoint": {"baseUrl": "https://native.invalid", "oauth": "drop"},
           "injection": {"sessionStart": ["summary"]}, "logging": True,
           "sessions": {str(tmp_path): "source-session"}, "defaultHost": "elsewhere",
           "hosts": {host: block, "unrelated": {"apiKey": "hch-at-other", "oauth": {"refreshToken": "hch-rt-other"}}},
           "unknown": "discard-me"}
    env = "UNRELATED_TOKEN=keep-me\n"
    path = source / "honcho.json"
    if case in {"ambient-base-url", "ambient-sdk-url", "env-sdk-url"}:
        raw.pop("endpoint")
    if case == "env-sdk-url":
        env += "HONCHO_URL=https://source-sdk.invalid\nHONCHO_ENVIRONMENT=local\n"
    if case in {"observation-null-peer", "observation-unknown"}:
        raw["observation"] = {"ai": {"observeMe": True}, "user": {"observeOthers": True}}
        block["observation"] = {"ai": None} if case == "observation-null-peer" else {"extra": True}
    if case == "legacy":
        raw["hosts"]["hermes.alpha"] = raw["hosts"].pop(host)
    if case in {"default-host", "override"}:
        raw["hosts"]["chosen"] = raw["hosts"].pop(host)
        host = "chosen"
        if case == "default-host":
            raw["defaultHost"] = host
        else:
            env += f"HERMES_HONCHO_HOST={host}\n"
    if case == "root":
        raw.pop("hosts")
        raw.pop("workspace")  # source's implicit workspace must survive the new host
    if case == "default-file":
        path = root / "honcho.json"
    if case == "global-file":
        path = tmp_path / ".honcho" / "config.json"
    if case == "disabled":
        block["enabled"] = False
    if case == "null-enabled":
        raw["enabled"] = None
    if case == "oauth-host":
        block.update(apiKey="hch-at-source", oauth={"refreshToken": "hch-rt-source"})
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "host-static":
        block["apiKey"] = "host-only-static"
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "oauth-root":
        raw.update(apiKey="hch-at-source", oauth={"refreshToken": "hch-rt-source"})
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "lower-oauth":
        env += "HONCHO_API_KEY=hch-at-lower\n"
    if case in {"opaque-root-oauth", "missing-root-oauth"}:
        raw.update(apiKey="opaque-access" if case == "opaque-root-oauth" else None,
                   oauth={"refreshToken": "opaque-refresh"})
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "missing-host-oauth":
        block.update(apiKey=None, oauth={"refreshToken": "opaque-refresh"})
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "refresh-root":
        raw["apiKey"] = " hch-rt-source "
        env += "HONCHO_API_KEY=lower-account-key\n"
    if case == "null-key":
        raw["apiKey"] = None
        block["apiKey"] = None
        env += "HONCHO_API_KEY=env-static\n"
    if case == "nested-falsy":
        block.update(injection={"sessionStart": None}, userPeerAliases={"ignored": None},
                     observation={"user": {"observeMe": None, "observeOthers": 0},
                                  "ai": {"observeMe": False}})
    if case in {"env", "empty", "oauth-env", "refresh-env"}:
        raw = None
        if case != "empty":
            env += "HONCHO_BASE_URL=https://env.invalid\nHONCHO_ENVIRONMENT=local\n"
            env += "HONCHO_API_KEY=" + ({"oauth-env": "hch-at-source", "refresh-env": "hch-rt-source"}.get(case, "env-static")) + "\n"
    if case == "empty":
        raw = {}  # Existing inactive native config still receives hygiene.
    if case == "invalid-url":
        raw = {"baseUrl": "https://invalid.example\u001b"}
    if raw is not None:
        write_json(path, raw)
    (source / ".env").write_text(env)
    (stage / ".env").write_text(env + "HERMES_HONCHO_HOST=stale-inherited\n"
        "HONCHO_BASE_URL=https://stale-base.invalid\nHONCHO_URL=https://stale-sdk.invalid\n"
        "HONCHO_ENVIRONMENT=stale-environment\n")
    if case == "symlink":
        (stage / "honcho.json").symlink_to(path)
    else:
        write_json(stage / "honcho.json", {"apiKey": "hch-at-stale", "oauth": {"refreshToken": "stale"}})
    before = {p: p.read_bytes() for p in (source / ".env", path) if p.exists()}
    environ = dict(os.environ)
    with clone_source_scope(source):
        original = client.HonchoClientConfig.from_global_config()
        report = companion.prepare_clone(source_home=source, source_name=source_name,
            staging_home=stage, destination_home=destination, destination_name="beta", clone_all=clone_all)
    assert dict(os.environ) == environ
    assert not destination.exists()
    assert all(p.read_bytes() == content for p, content in before.items())
    stage.rename(destination)
    result = json.loads((destination / "honcho.json").read_text())
    assert set(result["hosts"]) == {client.profile_host_key("beta")}
    new_block = result["hosts"][client.profile_host_key("beta")]
    assert new_block["aiPeer"] == "beta"
    assert "apiKey" not in new_block
    assert "sessions" not in result and "defaultHost" not in result
    assert "oauth" not in result and "oauth" not in new_block
    assert "hch-at-" not in json.dumps(result) and "hch-rt-" not in json.dumps(result)
    assert "discard-me" not in json.dumps(result)
    assert set(new_block["observation"]) == {"user", "ai"}
    assert all(set(fields) == {"observeMe", "observeOthers"}
               for fields in new_block["observation"].values())
    assert load_env_file(destination / ".env").get("UNRELATED_TOKEN") == "keep-me"
    assert "hch-at-" not in (destination / ".env").read_text()
    if os.name != "nt":
        assert stat.S_IMODE((destination / "honcho.json").stat().st_mode) == 0o600
    needs_auth = case in {"oauth-host", "oauth-root", "oauth-env", "host-static",
                         "refresh-root", "refresh-env", "opaque-root-oauth", "missing-root-oauth", "missing-host-oauth"}
    assert report["needs_auth"] is needs_auth
    assert report["enabled"] is (False if needs_auth else bool(original.enabled))
    snapshots = []
    for home in (source, destination, source):
        with (scope(home) if home == destination else clone_source_scope(home)):
            snapshots.append(client.HonchoClientConfig.from_global_config())
    assert asdict(snapshots[0]) == asdict(snapshots[2])
    cloned = snapshots[1]
    assert cloned.host == client.profile_host_key("beta")
    ignored = {"host", "ai_peer", "raw", "sessions", "explicitly_configured", "config_path", "hermes_home", "session_ai_peer_prefix"}
    if needs_auth:
        ignored |= {"api_key", "enabled"}
        assert not cloned.enabled
        assert not cloned.api_key
    assert {k: v for k, v in asdict(cloned).items() if k not in ignored} == {
        k: v for k, v in asdict(original).items() if k not in ignored}
    assert cloned.session_ai_peer_prefix is True
    if case in {"ambient-base-url", "ambient-sdk-url"}:
        assert cloned.api_key == "root-static"
        assert cloned.base_url is None
    for name in ("HONCHO_BASE_URL", "HONCHO_URL", "HONCHO_ENVIRONMENT"):
        assert load_env_file(destination / ".env")[name] == load_env_file(source / ".env").get(name, "")
    source_look = client._HostLookup(client._host_block(original.raw, original.host), original.raw)
    clone_look = client._HostLookup(client._host_block(cloned.raw, cloned.host), cloned.raw)
    assert clone_look.present("injection") == source_look.present("injection")
    assert clone_look.pick_set("logging") == source_look.pick_set("logging")
    assert cloned.resolve_session_name(gateway_session_key="same-chat") != original.resolve_session_name(gateway_session_key="same-chat")
    assert cloned.resolve_session_name(session_id="same-run").startswith("beta-")


@pytest.mark.parametrize("layer", ["root", "host"])
@pytest.mark.parametrize("fields", [
    {"apiKey": {"accessToken": "do-not-display", "refreshToken": "do-not-display"}},
    {"apiKey": []}, {"apiKey": False}, {"apiKey": 0},
    {"oauth": ["do-not-display"]},
    {"injection": {"sessionStart": {"oauth": {"refreshToken": "do-not-display"}}}},
    {"injection": {"sessionStart": [{"refreshToken": "do-not-display"}]}},
    {"injection": ["do-not-display"]},
    {"observation": {"ai": {"observeMe": {"refreshToken": "do-not-display"}}}},
    {"observation": {"user": ["do-not-display"]}},
    {"userPeerAliases": {"owner": {"refreshToken": "do-not-display"}}},
    {"dialecticDepthLevels": [{"refreshToken": "do-not-display"}]},
    {"logging": {"refreshToken": "do-not-display"}},
    {"workspace": {"refreshToken": "do-not-display"}},
    {"contextTokens": ["do-not-display"]},
    {"endpoint": {"baseUrl": {"refreshToken": "do-not-display"}}},
])
def test_clone_refuses_malformed_supported_fields_before_writing(tmp_path, monkeypatch, layer, fields):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    source, stage, destination = (tmp_path / name for name in ("source", "stage", "destination"))
    source.mkdir()
    stage.mkdir()
    raw = {"apiKey": "root-static", "hosts": {"hermes_alpha": {}}}
    (raw if layer == "root" else raw["hosts"]["hermes_alpha"]).update(fields)
    write_json(source / "honcho.json", raw)
    (source / ".env").write_text("HONCHO_API_KEY=lower-account\n")
    (stage / ".env").write_text("UNCHANGED=yes\n")
    before = {p: p.read_bytes() for home in (source, stage) for p in home.iterdir()}
    companion = load_provider_companion("honcho", "clone")
    assert companion is not None
    with pytest.raises(ValueError, match="safe local settings") as error:
        companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
            destination_home=destination, destination_name="beta", clone_all=True)
    assert "do-not-display" not in str(error.value)
    assert all(p.read_bytes() == value for p, value in before.items())
    assert not (stage / "honcho.json").exists()
    assert not destination.exists()


@pytest.mark.parametrize("case", ["external", "unconfigured", "ambient-miss", "malformed", "malformed-host", "env-link", "stage-link", "replace-failure",
    "source-alias", "destination-alias", "stage-traversal"])
def test_clone_is_local_fail_closed_and_relocatable(tmp_path, monkeypatch, case):
    import plugins.memory as memory
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    source = root / "profiles" / "alpha"
    stage = root / "profiles" / ".stage"
    destination = root / "profiles" / "beta"
    source.mkdir(parents=True)
    stage.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    write_json(source / "honcho.json", {"apiKey": "root-static", "peerName": "owner", "recallMode": "tools"})
    (source / ".env").write_text("HONCHO_API_KEY=env-static\n")
    (stage / ".env").write_text("HONCHO_API_KEY=hch-at-inherited\n")
    if case == "external":
        plugin_dir = source / "plugins" / "honcho"
        shutil.copytree(Path(memory.__file__).parent / "honcho", plugin_dir, ignore=shutil.ignore_patterns("__pycache__"))
        monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", tmp_path / "no-bundled")
        monkeypatch.setattr(memory, "_get_project_plugins_dir", lambda: None)
        class BlockBundled(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "plugins.memory.honcho" or fullname.startswith("plugins.memory.honcho."):
                    raise AssertionError("relocated Honcho imported its bundled copy")
        for name in tuple(sys.modules):
            if name == "plugins.memory.honcho" or name.startswith("plugins.memory.honcho."):
                monkeypatch.delitem(sys.modules, name)
        guard = BlockBundled()
        monkeypatch.setattr(sys, "meta_path", [guard, *sys.meta_path])
    with scope(source):
        companion = load_provider_companion("honcho", "clone")
    assert companion is not None
    if case == "unconfigured":
        (source / "honcho.json").unlink()
        (source / ".env").write_text("OTHER_KEY=source\n")
        (stage / ".env").write_text("OTHER_KEY=source\n")
        monkeypatch.setenv("HONCHO_API_KEY", "unrelated-process-key")
        before = {p.name: p.read_bytes() for p in stage.iterdir()}
        with scope(source):
            report = companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
                destination_home=destination, destination_name="beta", clone_all=False)
        assert report is None
        assert {p.name: p.read_bytes() for p in stage.iterdir()} == before
        return
    if case == "ambient-miss":
        from agent import secret_scope
        monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
        monkeypatch.setenv("HONCHO_API_KEY", "wrong-account")
        monkeypatch.setenv("HERMES_HONCHO_HOST", "wrong-host")
        write_json(source / "honcho.json", {"peerName": "owner"})
        (source / ".env").write_text("")
        token = set_secret_scope({})
        try:
            report = companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
                destination_home=destination, destination_name="beta", clone_all=False)
        finally:
            reset_secret_scope(token)
        result = json.loads((stage / "honcho.json").read_text())
        assert not report["enabled"]
        assert "wrong-account" not in json.dumps(result)
        assert result["hosts"]["hermes_beta"]["workspace"] == "hermes_alpha"
        assert load_env_file(stage / ".env")["HONCHO_API_KEY"] == ""
        return
    if case == "malformed":
        (source / "honcho.json").write_text('{"secret":"do-not-display",')
    if case == "malformed-host":
        write_json(source / "honcho.json", {"hosts": {"hermes_alpha": ["do-not-display"]}})
    if case == "env-link":
        (stage / ".env").unlink()
        (stage / ".env").symlink_to(source / ".env")
    if case == "stage-link":
        real_stage = stage.with_name("real-stage")
        stage.rename(real_stage)
        stage.symlink_to(real_stage, target_is_directory=True)
    if case in {"source-alias", "destination-alias"}:
        alias = root / "alias"
        alias.symlink_to(source.parent, target_is_directory=True)
        if case == "source-alias":
            stage = alias / source.name
        else:
            destination = alias / stage.name
    if case == "stage-traversal":
        nested = stage / "nested"
        nested.mkdir()
        stage = nested / ".."
    if case == "replace-failure":
        def fail_replace(*args, **kwargs):
            raise OSError("do-not-display")
        monkeypatch.setattr(os, "replace", fail_replace)
    before = {p: p.read_bytes() for p in source.iterdir() if p.is_file()}
    if case != "external":
        with pytest.raises(ValueError) as error, scope(source):
            companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
                destination_home=destination, destination_name="beta", clone_all=True)
        assert "do-not-display" not in str(error.value)
        if case != "destination-alias":
            assert not destination.exists()
    else:
        assert not hasattr(sys.modules[companion.__package__], "HonchoMemoryProvider")
        with scope(source):
            companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
                destination_home=destination, destination_name="beta", clone_all=True)
        shutil.copytree(plugin_dir, stage / "plugins" / "honcho")
        stage.rename(destination)
        with scope(destination):
            provider = memory.load_memory_provider("honcho", register_skills=False)
            assert provider is not None
            provider.initialize("run-id", platform="cli", gateway_session_key="same-chat")
            assert provider._config is not None
            assert provider._config.host == "hermes_beta"
            assert provider._config.ai_peer == "beta"
            assert provider._config.peer_name == "owner"
            assert provider._config.workspace_id == "hermes_alpha"
            assert provider._session_key == "beta-same-chat"
            session = importlib.import_module(provider.__module__ + ".session")
            manager = session.HonchoSessionManager(config=provider._config, runtime_user_peer_name="runtime")
            assert manager.assistant_peer_id() == "beta"
            assert manager._resolve_user_peer_id("same-chat") == "runtime"
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.macos_only
@pytest.mark.parametrize("alias", ["/tmp", "/var/tmp"])
def test_clone_trusts_canonical_parent_above_private_staging(tmp_path, monkeypatch, alias):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    companion = load_provider_companion("honcho", "clone")
    assert companion is not None
    with tempfile.TemporaryDirectory(dir=alias) as directory:
        parent = Path(directory)
        assert parent != parent.resolve(), "exercise the native macOS directory alias"
        source, stage, destination = (parent / name for name in ("source", "stage", "destination"))
        source.mkdir()
        stage.mkdir()
        write_json(source / "honcho.json", {"apiKey": "root-static"})
        before = (source / "honcho.json").read_bytes()
        # A replaceable JSON leaf is safe even when its target is the source.
        (stage / "honcho.json").symlink_to(source / "honcho.json")
        companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
            destination_home=destination, destination_name="beta", clone_all=True)
        assert (source / "honcho.json").read_bytes() == before
        assert not (stage / "honcho.json").is_symlink()
        stage.rename(destination)
        with scope(destination):
            assert companion.__package__ is not None
            client = importlib.import_module(companion.__package__ + ".client")
            assert client.HonchoClientConfig.from_global_config().api_key == "root-static"


@pytest.mark.windows_only
@pytest.mark.parametrize("leaf", [None, ".env", "honcho.json"])
def test_clone_rejects_native_staging_junctions(tmp_path, monkeypatch, leaf):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    source, stage, destination = (tmp_path / name for name in ("source", "stage", "destination"))
    source.mkdir()
    write_json(source / "honcho.json", {"apiKey": "root-static"})
    if leaf is not None:
        stage.mkdir()
    link = stage / leaf if leaf else stage
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(source)], check=True, capture_output=True)
    companion = load_provider_companion("honcho", "clone")
    assert companion is not None
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    try:
        with pytest.raises(ValueError):
            companion.prepare_clone(source_home=source, source_name="alpha", staging_home=stage,
                destination_home=destination, destination_name="beta", clone_all=True)
        assert {p.name: p.read_bytes() for p in source.iterdir()} == before
        assert not destination.exists()
    finally:
        link.rmdir()
