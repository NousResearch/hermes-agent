"""CLI dispatch, authenticated REST and RPC share clone publication semantics.

The invariant covers implicit clone preparation, not explicitly requested
post-create SOUL/model customization. Running backends are booted before the
snapshot; the CLI subprocess separately covers bootstrap with an explicit source.
"""
from __future__ import annotations

from argparse import Namespace
import json
import os
from pathlib import Path
import sys

import pytest
import yaml

from agent import secret_scope
from hermes_cli import profiles
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override


COMPANION = '''import json
from agent.secret_scope import get_secret
from hermes_constants import get_hermes_home

def prepare_clone(*, source_home, source_name, staging_home, destination_home, destination_name, clone_all):
    assert get_hermes_home() == source_home
    assert get_secret("PARITY_SOURCE_KEY") == source_name
    assert not get_secret("PARITY_SHELL_ONLY_KEY")
    assert not destination_home.exists()
    assert staging_home != destination_home
    assert destination_home.name == destination_name
    state = json.loads((source_home / "parity_probe.json").read_text())
    if state.get("fail"):
        raise RuntimeError("private-provider-sentinel")
    state.update(peer=destination_name, source=source_name, full=clone_all)
    (staging_home / "parity_probe.json").write_text(json.dumps(state))
    # Simulate an offline companion intentionally leaving authentication unconfigured.
    (staging_home / ".env").write_text("# authenticate this clone independently\\n")
    (staging_home / "auth.json").unlink(missing_ok=True)
    (staging_home / "config.yaml").write_text("memory:\\n  provider: parity_probe\\n")
    return {"needs_auth": state["needs_auth"], "secret": "private-provider-sentinel"}
'''

CHANNEL_PLUGIN = '''from pathlib import Path
raise AssertionError("channel plugin activation forbidden")

def register(ctx):
    ctx.register_platform(name="parity_channel", label="Parity", adapter_factory=lambda cfg: None,
                          check_fn=lambda: True, required_env=["UNRELATED_BOT_SECRET"],
                          allowed_users_env="UNRELATED_ALLOWED_USERS")
'''


CASES = [(surface, full, explicit) for surface in ("cli", "rest", "rpc")
         for full, explicit in ((False, True), (True, True), (True, False))]
CASES += [(surface, False, False) for surface in ("cli", "rest")]


def _snapshot(home):
    return {p.relative_to(home): ("symlink", os.readlink(p)) if p.is_symlink()
            else ("directory",) if p.is_dir() else ("file", p.read_bytes())
            for p in home.rglob("*") if p != home / ".plugin-installation.lock"}


@pytest.fixture
def clone_surface_env(tmp_path, monkeypatch):
    from plugins import memory
    from hermes_cli import gateway_multiplex_served

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".hermes"
    homes = {"default": root, "source": root / "profiles" / "source",
             "caller": root / "profiles" / "caller"}
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(yaml.safe_dump({
            "memory": {"provider": "parity_probe"},
            "model": {"provider": "custom", "default": f"{name}-model"},
            "tts": {"provider": f"{name}-voice"}}))
        (home / ".env").write_text(f"PARITY_SOURCE_KEY={name}\n")
        (home / "auth.json").write_text(json.dumps({"providers": {}, "owner": name}))
        (home / "SOUL.md").write_text(f"{name} personality\n")
        (home / "parity_probe.json").write_text(json.dumps({"peer": name, "needs_auth": True}))
        skill = home / "skills" / "curated" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        skill.write_text(f"{name} curated skill\n")
    caller = homes["caller"]
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("PARITY_SHELL_ONLY_KEY", "caller-shell-secret")
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    # A served backend binds requests without publishing dotenv into process env.
    # Also prevents server import's development .env fallback from reading the checkout.
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    bundle = tmp_path / "bundled-memory"
    package = bundle / "parity_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('# MemoryProvider\nraise AssertionError("runtime activation forbidden")\n')
    (package / "clone.py").write_text(COMPANION)
    monkeypatch.setattr(memory, "_MEMORY_PLUGINS_DIR", bundle)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", lambda: None)
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: None)
    seeded = []
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda path, **kw: seeded.append(path))
    notifications = []

    def notified(name):
        destination = profiles.get_profile_dir(name)
        state = json.loads((destination / "parity_probe.json").read_text())
        assert state["peer"] == name
        notifications.append((name, _snapshot(destination)))

    monkeypatch.setattr(profiles, "_notify_multiplexer", notified)
    # Boot the transports before snapshotting: these are already-running backends,
    # not assertions about their normal startup scaffolding.
    from hermes_cli import web_server  # noqa: F401
    from tui_gateway import server  # noqa: F401
    home_token = set_hermes_home_override(caller)
    secret_token = secret_scope.set_secret_scope({"PARITY_SOURCE_KEY": "caller", "PARITY_SHELL_ONLY_KEY": "caller-secret"})
    # Do not start background host liveness/orphan services in a transport test.
    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    from starlette.testclient import TestClient
    try:
        with TestClient(web_server.app).websocket_connect(f"/api/ws?token={web_server._SESSION_TOKEN}") as connection:
            assert connection.receive_json()["params"]["type"] == "gateway.ready"
            # Bootstrap (including attachment heartbeat) precedes the invariant;
            # actual RPC requests go through the authenticated live connection.
            yield homes, notifications, seeded, connection
    finally:
        secret_scope.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)
        for name in list(sys.modules):
            if name == "plugins.memory.parity_probe" or name.startswith("plugins.memory.parity_probe."):
                sys.modules.pop(name, None)


def _runtime_snapshot():
    from hermes_cli import plugins
    from tools.registry import registry
    from gateway.platform_registry import platform_registry
    from agent.secret_sources import registry as sources
    return (dict(plugins._plugin_managers_by_home), plugins._plugin_manager,
            dict(registry._tools), {k: dict(v) for k, v in registry._scoped_tools.items()},
            dict(platform_registry._entries),
            {k: dict(v) for k, v in platform_registry._scoped_entries.items()},
            dict(platform_registry._deferred),
            {k: dict(v) for k, v in platform_registry._scoped_deferred.items()},
            dict(sources._SOURCES), {k: dict(v) for k, v in sources._SCOPED_SOURCES.items()})


def _create(surface, name, full, explicit, capsys, connection):
    source = "source" if explicit else None
    if surface == "cli":
        from hermes_cli.profile_cmd import _profile_create
        code = 0
        try:
            _profile_create(Namespace(profile_name=name, clone_from=source, clone=not full,
                                      clone_all=full, no_alias=True))
        except SystemExit as exc:
            code = exc.code
        output = capsys.readouterr()
        return code == 0, {}, output.out + output.err
    if surface == "rest":
        from starlette.testclient import TestClient
        from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
        client = TestClient(app)
        client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
        response = client.post("/api/profiles", json={"name": name, "clone_from": source,
                              "clone_all": full, "clone_from_default": not explicit and not full})
        client.close()
        return response.status_code == 200, response.json(), response.text
    connection.send_json({"jsonrpc": "2.0", "id": name, "method": "profiles.create",
                          "params": {"name": name, "clone_from": source, "clone_all": full}})
    while True:
        response = connection.receive_json()
        if response.get("id") == name:
            break
    return "error" not in response, response.get("result", {}), json.dumps(response)


@pytest.mark.parametrize("surface,full,explicit", CASES)
@pytest.mark.parametrize("needs_auth", [False, True])
def test_clone_surface_publishes_prepared_source_only(clone_surface_env, surface, full, explicit, needs_auth, capsys):
    homes, notifications, seeded, connection = clone_surface_env
    source_name = "source" if explicit else ("default" if surface == "rest" else "caller")
    source = homes[source_name]
    (source / "parity_probe.json").write_text(json.dumps({"peer": source_name, "needs_auth": needs_auth}))
    before = {home: {p: value for p, value in _snapshot(home).items()
                     if p not in {Path("profiles") / name / ".plugin-installation.lock" for name in homes}}
              for home in homes.values()}
    env_before = dict(os.environ)
    runtime_before = _runtime_snapshot()
    ok, result, output = _create(surface, "destination", full, explicit, capsys, connection)
    assert ok, output
    destination = profiles.get_profile_dir("destination")
    state = json.loads((destination / "parity_probe.json").read_text())
    assert state == {"peer": "destination", "source": source_name, "needs_auth": needs_auth, "full": full}
    assert (destination / "SOUL.md").read_bytes() == (source / "SOUL.md").read_bytes()
    assert (destination / "skills" / "curated" / "SKILL.md").read_bytes() == (source / "skills" / "curated" / "SKILL.md").read_bytes()
    assert notifications == [("destination", _snapshot(destination))], "surface changed prepared clone after publication"
    assert seeded == [], "clones must preserve source-curated skills without reseeding"
    assert not (destination / "auth.json").exists()
    cfg = yaml.safe_load((destination / "config.yaml").read_text())
    assert cfg["memory"] == {"provider": "parity_probe"}
    assert not any(cfg.get(key) for key in ("model", "stt", "tts", "voice"))
    assert "PARITY_SOURCE_KEY" not in (destination / ".env").read_text()
    if surface == "cli":
        assert ("needs authentication" in output) is needs_auth
    else:
        assert Path(result["path"]) == destination
        assert result["clone_needs_auth"] == (["parity_probe"] if needs_auth else [])
    if surface == "rpc":
        assert not any(result["mirrored"].values())
    assert "private-provider-sentinel" not in output
    assert "private-provider-sentinel" not in (destination / ".clone-report.json").read_text()
    for home, snapshot in before.items():
        current = _snapshot(home)
        if home == homes["default"]:
            current = {p: value for p, value in current.items()
                       if p.parts[:2] != ("profiles", "destination")
                       and p not in {Path("profiles") / name / ".plugin-installation.lock" for name in homes}}
        assert current == snapshot
    assert _runtime_snapshot() == runtime_before
    assert dict(os.environ) == env_before
    assert get_hermes_home() == homes["caller"]
    assert secret_scope.get_secret("PARITY_SHELL_ONLY_KEY") == "caller-secret"
    assert not list(destination.parent.glob(".destination.staging-*"))


@pytest.mark.parametrize("multiplex", [False, True])
def test_channel_inventory_is_offline_source_owned(tmp_path, monkeypatch, multiplex):
    from hermes_cli.profile_channels import strip_channel_settings
    from agent.secret_sources.base import SecretSource
    from agent.secret_sources import registry as sources
    import shutil

    source, caller, destination = (tmp_path / name for name in ("source", "caller", "destination"))
    for home in (source, caller):
        package = home / "plugins" / "custom-channel"
        package.mkdir(parents=True)
        (package / "plugin.yaml").write_text("name: custom-channel\nkind: platform\n")
        (package / "__init__.py").write_text(CHANNEL_PLUGIN.replace("parity_channel", home.name + "_channel"))
        (home / "config.yaml").write_text(yaml.safe_dump({
            "plugins": {"enabled": ["custom-channel"]},
            home.name + "_channel": {"token": "private"},
            "secrets": {"offline_probe": {"enabled": False}},
            "model": {"default": "keep-model"}}))
        (home / ".env").write_text("UNRELATED_BOT_SECRET=private\nUNRELATED_ALLOWED_USERS=owner\n"
                                  f"{home.name.upper()}_CHANNEL_EXTRA=private\nOPENAI_API_KEY=keep\n")
        (home / (home.name + "_channel_state")).mkdir()
    shutil.copytree(source, destination)
    monkeypatch.setenv("HERMES_HOME", str(caller))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    calls = []
    class OfflineProbe(SecretSource):
        name = "offline_probe"
        def is_enabled(self, cfg):
            calls.append("enabled")
            return True
        def fetch(self, cfg, home_path):
            calls.append("fetch")
            raise AssertionError("external secret source called")
    probe = OfflineProbe()
    monkeypatch.setitem(sources._SOURCES, probe.name, probe)
    monkeypatch.setitem(sources._SOURCE_ORIGINS, probe.name, "plugin")
    before = _snapshot(source)
    runtime_before = _runtime_snapshot()
    env_before = dict(os.environ)
    removed = strip_channel_settings(destination, include_state=True, source_dir=source)
    assert removed["source_channel"] == ["UNRELATED_BOT_SECRET", "UNRELATED_ALLOWED_USERS", "SOURCE_CHANNEL_EXTRA"]
    assert (destination / ".env").read_text() == "OPENAI_API_KEY=keep\n"
    assert "source_channel" not in yaml.safe_load((destination / "config.yaml").read_text())
    assert not (destination / "source_channel_state").exists()
    assert _snapshot(source) == before
    assert _runtime_snapshot() == runtime_before
    assert dict(os.environ) == env_before
    assert calls == []


@pytest.mark.parametrize("declaration", [
    'ctx.register_platform(name="hidden", **metadata())',
    'ctx.register_platform(name="hidden", **{"required_env": ["NONCANONICAL_SECRET"]})',
    'ctx.register_platform(**metadata)',
    'register = ctx.register_platform; register(name="hidden", required_env=["NONCANONICAL_SECRET"])',
    "exec('ctx.register_platform(name=\"hidden\", required_env=[\"NONCANONICAL_SECRET\"])')",
])
def test_channel_inventory_never_silently_omits_indirect_credentials(tmp_path, monkeypatch, declaration):
    from hermes_cli.profile_channels import strip_channel_settings

    source = tmp_path / "source"
    package = source / "plugins" / "hidden"
    package.mkdir(parents=True)
    (package / "plugin.yaml").write_text("name: hidden\nkind: platform\n")
    (package / "__init__.py").write_text(
        'raise AssertionError("must not import")\n'
        'metadata = {"name": "hidden", "required_env": ["NONCANONICAL_SECRET"]}\n'
        f'def register(ctx):\n    {declaration}\n')
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / ".env").write_text("NONCANONICAL_SECRET=private\nOPENAI_API_KEY=keep\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "caller"))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    before = _snapshot(source)
    if "metadata()" in declaration or "register =" in declaration or declaration.startswith("exec"):
        with pytest.raises(ValueError, match="safely inventory"):
            strip_channel_settings(destination, include_state=False, source_dir=source)
        assert (destination / ".env").read_text().startswith("NONCANONICAL_SECRET=private\n")
    else:
        strip_channel_settings(destination, include_state=False, source_dir=source)
        assert (destination / ".env").read_text() == "OPENAI_API_KEY=keep\n"
    assert _snapshot(source) == before


def _inventory_clone_source(tmp_path, monkeypatch, body):
    root = tmp_path / ".hermes"
    source = root / "profiles" / "source"
    package = tmp_path / "bundled" / "inventory-probe"
    package.mkdir(parents=True)
    source.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: package.parent)
    (package / "plugin.yaml").write_text("name: inventory-probe\nkind: platform\n")
    (package / "__init__.py").write_text(body)
    (source / "config.yaml").write_text("memory:\n  provider: builtin\n")
    (source / ".env").write_text(
        "FIRST_SECRET=first\nSECOND_SECRET=second\nNONCANONICAL_SECRET=hidden\nOPENAI_API_KEY=keep\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.setattr(profiles, "check_alias_collision", lambda name: None)
    monkeypatch.setattr(profiles, "create_wrapper_script", lambda name: None)
    effects = []
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: effects.append("service"))
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: effects.append("publish"))
    monkeypatch.setattr(profiles, "seed_profile_skills", lambda *a, **kw: effects.append("seed"))
    return source, package, effects


@pytest.mark.macos_only
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("unreadable", ["manifest", "directory"])
def test_channel_inventory_refuses_unreadable_installation(tmp_path, monkeypatch, full, unreadable):
    from hermes_cli.profile_channels import ChannelKeyIndex
    from hermes_cli.plugins_discovery import collect_directory_manifests

    source, package, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'raise AssertionError("must not activate")\n'
        'def register(ctx):\n'
        '    ctx.register_platform(name="probe", required_env=["NONCANONICAL_SECRET"])\n')
    before, installed_before = _snapshot(source), _snapshot(package)
    runtime_before, env_before = _runtime_snapshot(), dict(os.environ)
    target = package / "plugin.yaml" if unreadable == "manifest" else package
    mode = target.stat().st_mode
    target.chmod(0)
    try:
        # Real host permissions, not a mocked read failure; runtime stays best effort.
        with pytest.raises(PermissionError):
            (package / "plugin.yaml").read_bytes()
        assert not any(m.name == "inventory-probe" for m in collect_directory_manifests())
        for operation in (lambda: ChannelKeyIndex(source),
                          lambda: profiles.create_profile("refused", clone_from="source", clone_all=full)):
            with pytest.raises(ValueError, match="safely inventory") as error:
                operation()
            assert "inventory-probe" in str(error.value)
            assert "repair" in str(error.value) and "retry" in str(error.value)
        assert not profiles.get_profile_dir("refused").exists()
        assert not list(source.parent.glob(".refused.staging-*"))
        assert _snapshot(source) == before
        assert effects == []
        assert _runtime_snapshot() == runtime_before and dict(os.environ) == env_before
    finally:
        target.chmod(mode)
    assert _snapshot(package) == installed_before


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("manifest_data,declaration", [
    *[({"requires_env": value}, 'name="probe"') for value in (
        None, "PRIVATE_SENTINEL", {"PRIVATE_SENTINEL": True}, 42,
        [{}], [{"description": "PRIVATE_SENTINEL"}], [{"name": ["PRIVATE_SENTINEL"]}],
        [{"name": None}], [{"name": ""}], [{"name": "  "}], [None], [42], [""], ["  "],
    )],
    *[({}, f'name="probe", required_env={value!r}') for value in (None, "PRIVATE_SENTINEL", [None], [""], ["  "])],
    ({}, 'name="  "'),
    ({"name": ["PRIVATE_SENTINEL"]}, 'name="probe"'),
    ({"name": None}, 'name="probe"'),
    ({"name": ""}, 'name="probe"'),
    ({"name": "  "}, 'name="probe"'),
    ("name: [PRIVATE_SENTINEL", 'name="probe"'),
])
def test_channel_inventory_refuses_malformed_metadata_before_publication(
    tmp_path, monkeypatch, full, manifest_data, declaration,
):
    from hermes_cli.profile_channels import ChannelKeyIndex

    source, package, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'raise AssertionError("must not activate")\n'
        f'def register(ctx):\n    ctx.register_platform({declaration})\n')
    (package / "plugin.yaml").write_text(
        manifest_data if isinstance(manifest_data, str) else
        yaml.safe_dump({"name": "inventory-probe", "kind": "platform", **manifest_data}))
    before, installed_before = _snapshot(source), _snapshot(package)
    runtime_before, env_before = _runtime_snapshot(), dict(os.environ)
    for operation in (lambda: ChannelKeyIndex(source),
                      lambda: profiles.create_profile("refused", clone_from="source", clone_all=full)):
        with pytest.raises(ValueError, match="safely inventory") as error:
            operation()
        assert "inventory-probe" in str(error.value)
        assert "repair" in str(error.value) and "retry" in str(error.value)
        assert "PRIVATE_SENTINEL" not in str(error.value)
    assert not profiles.get_profile_dir("refused").exists()
    assert not list(source.parent.glob(".refused.staging-*"))
    assert _snapshot(source) == before and _snapshot(package) == installed_before
    assert effects == []
    assert _runtime_snapshot() == runtime_before and dict(os.environ) == env_before


@pytest.mark.parametrize("full", [False, True])
def test_channel_inventory_refuses_broken_entrypoint_metadata(tmp_path, monkeypatch, full):
    from hermes_cli.profile_channels import ChannelKeyIndex

    source, package, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'def register(ctx):\n    ctx.register_platform(name="probe")\n')
    site = tmp_path / "site-packages"
    dist = site / "inventory_probe-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: inventory-probe\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text("[hermes_agent.plugins]\nbroken-entry-without-value\n")
    monkeypatch.syspath_prepend(str(site))
    before, installed_before = _snapshot(source), _snapshot(site)
    runtime_before, env_before = _runtime_snapshot(), dict(os.environ)
    for operation in (lambda: ChannelKeyIndex(source),
                      lambda: profiles.create_profile("refused", clone_from="source", clone_all=full)):
        with pytest.raises(ValueError, match="safely inventory.*entry point") as error:
            operation()
        assert "repair" in str(error.value) and "retry" in str(error.value)
    assert not profiles.get_profile_dir("refused").exists()
    assert not list(source.parent.glob(".refused.staging-*"))
    assert _snapshot(source) == before and _snapshot(site) == installed_before
    assert effects == []
    assert _runtime_snapshot() == runtime_before and dict(os.environ) == env_before


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("extra", [
    "exec('ctx.register_platform(name=\"hidden\", required_env=[\"NONCANONICAL_SECRET\"])')",
    "exec('ctx.' + 'register_' + 'platform(name=\"hidden\", required_env=[\"NONCANONICAL_SECRET\"])')",
    'getattr(ctx, "register_platform")(name="hidden", required_env=["NONCANONICAL_SECRET"])',
    'register = ctx.register_platform; register(name="hidden", required_env=["NONCANONICAL_SECRET"])',
    'from gateway.platform_registry import PlatformEntry as Entry\n    factory = Entry\n    factory(name="hidden", required_env=["NONCANONICAL_SECRET"])',
    'ctx.register_platform(name="probe", required_env=["SECOND_SECRET"], allowed_users_env="OTHER_POLICY")',
    'hidden_registration(ctx)',
    'hidden_registration()',
    'result = hidden_registration()',
    pytest.param('if hidden_registration():\n        pass', id="if-condition"),
    pytest.param('while hidden_registration():\n        break', id="while-condition"),
    pytest.param('[hidden_registration() for item in (1,)]', id="comprehension-element"),
    pytest.param('[item for item in (1,) if hidden_registration()]', id="comprehension-filter"),
    pytest.param('True and hidden_registration()', id="boolop"),
    pytest.param('assert hidden_registration()', id="assert"),
    pytest.param('value = (hidden_registration(),)', id="nested-container"),
    pytest.param('value = f"{hidden_registration()}"', id="formatted-value"),
    pytest.param('def callback(value=hidden_registration()):\n        pass', id="default-argument"),
    pytest.param('@hidden_registration()\n    def callback():\n        pass', id="decorator"),
    pytest.param('callback = lambda value=hidden_registration(): None', id="lambda-default"),
    pytest.param('with hidden_registration():\n        pass', id="with-context"),
    pytest.param('dict = hidden_registration\n    if dict():\n        pass', id="shadowed-dict"),
    pytest.param('from hidden import hidden_registration as dict\n    dict()', id="imported-dict"),
    pytest.param('logger = hidden_registration\n    logger.info()', id="fake-logger"),
    pytest.param('from hidden import hidden_registration as PlatformEntry\n    PlatformEntry(name="other")', id="fake-entry"),
    pytest.param('from .helper import register_tools\n    register_tools(ctx)', id="opaque-helper-condition"),
    pytest.param('value = (unknown or "").strip()', id="opaque-strip-receiver"),
    pytest.param('logger.info = hidden_registration\n    logger.info()', id="mutated-logger-method"),
    pytest.param('@hidden_registration\n    def callback():\n        pass', id="bare-decorator"),
    pytest.param('def helper():\n        dict = hidden_registration\n        dict()\n    helper()', id="nested-shadowed-dict"),
    'import builtins as b\n    b.exec("hidden registration")',
    'from builtins import exec as run\n    run("hidden registration")',
    'getattr(ctx, "register_" + "platform")(name="hidden", required_env=["NONCANONICAL_SECRET"])',
])
def test_channel_inventory_rejects_mixed_uninspectable_registration(tmp_path, monkeypatch, full, extra):
    from hermes_cli.profile_channels import ChannelKeyIndex

    source, package, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'raise AssertionError("must not activate")\n'
        'import logging\n'
        'logger = logging.getLogger(__name__)\n'
        'def register(ctx):\n'
        '    ctx.register_platform(name="probe", required_env=["FIRST_SECRET"], allowed_users_env="POLICY")\n'
        f'    {extra}\n')
    (package / "helper.py").write_text(
        'def register_tools(ctx):\n'
        '    if hidden_registration():\n'
        '        ctx.register_tool(name="probe")\n')
    before = _snapshot(source)
    runtime_before, env_before = _runtime_snapshot(), dict(os.environ)
    for operation in (lambda: ChannelKeyIndex(source),
                      lambda: profiles.create_profile("refused", clone_from="source", clone_all=full)):
        with pytest.raises(ValueError, match="safely inventory") as error:
            operation()
        assert "inventory-probe" in str(error.value)
        assert "literal" in str(error.value) and "retry" in str(error.value)
    assert not profiles.get_profile_dir("refused").exists()
    assert not list(source.parent.glob(".refused.staging-*"))
    assert _snapshot(source) == before
    assert effects == []
    assert _runtime_snapshot() == runtime_before and dict(os.environ) == env_before


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("body", [
    'from gateway.platform_registry import PlatformEntry as Entry\n'
    'def register(ctx):\n'
    '    ctx.register_platform(name="probe", required_env=["FIRST_SECRET"])\n'
    '    Entry(name="probe", required_env=["SECOND_SECRET", "NONCANONICAL_SECRET"])\n',
    'def register(ctx):\n'
    '    if runtime_condition:\n'
    '        ctx.register_platform(name="probe", required_env=["FIRST_SECRET"])\n'
    '    else:\n'
    '        ctx.register_platform(name="probe", required_env=["SECOND_SECRET", "NONCANONICAL_SECRET"])\n',
    'from gateway.platform_registry import PlatformEntry as Entry\n'
    'def register(ctx):\n'
    '    Entry("probe", required_env=["FIRST_SECRET", "SECOND_SECRET", "NONCANONICAL_SECRET"])\n',
])
def test_channel_inventory_unions_all_literal_ownership(tmp_path, monkeypatch, full, body):
    from hermes_cli.profile_channels import ChannelKeyIndex

    source, package, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'raise AssertionError("must not activate")\n' + body)
    before = _snapshot(source)
    index = ChannelKeyIndex(source)
    for key in ("FIRST_SECRET", "SECOND_SECRET", "NONCANONICAL_SECRET"):
        assert index.platform_for(key) == "probe"
    destination = profiles.create_profile("destination", clone_from="source", clone_all=full)
    assert (destination / ".env").read_text() == "OPENAI_API_KEY=keep\n"
    assert _snapshot(source) == before
    assert "publish" in effects


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("aliased", [False, True])
def test_module_entrypoint_inventory_collects_relative_helper_ownership(tmp_path, monkeypatch, full, aliased):
    from hermes_cli.profile_channels import ChannelKeyIndex
    from hermes_cli.profile_channel_inventory import _declarations
    from hermes_cli.plugins_discovery import discover_entrypoint_manifests

    source, bundled, effects = _inventory_clone_source(tmp_path, monkeypatch,
        'def register(ctx):\n    ctx.register_platform(name="probe", required_env=["FIRST_SECRET"])\n')
    site = tmp_path / "site-packages"
    package = site / "relative_inventory_probe"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('raise AssertionError("must not activate package")\n')
    helper_name = "extra" if aliased else "register_extra"
    entry_name = "Entry" if aliased else "PlatformEntry"
    (package / "plugin.py").write_text(
        'raise AssertionError("must not activate entrypoint")\n'
        f'from .helper import register_extra as {helper_name}\n'
        'def register(ctx):\n'
        '    ctx.register_platform(name="visible", required_env=["SECOND_SECRET"])\n'
        f'    {helper_name}(ctx)\n'
        f'    {helper_name}(ctx)\n')
    (package / "helper.py").write_text(
        'raise AssertionError("must not activate helper")\n'
        'from .leaf import register_leaf as leaf\n'
        'def register_extra(ctx):\n'
        '    ctx.register_platform(name="extra", required_env=["NONCANONICAL_SECRET"])\n'
        '    leaf()\n')
    (package / "leaf.py").write_text(
        f'from gateway.platform_registry import PlatformEntry as {entry_name}\n'
        'def register_leaf():\n'
        f'    {entry_name}(name="extra", required_env=["LEAF_SECRET"])\n')
    # A module entry point must not expand into an unrelated package-wide scan.
    (package / "unrelated.py").write_text('not valid Python!\n')
    dist = site / "relative_inventory_probe-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: relative-inventory-probe\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[hermes_agent.plugins]\nrelative-inventory-probe = relative_inventory_probe.plugin:register\n")
    monkeypatch.syspath_prepend(str(site))
    assert any(m.path == "relative_inventory_probe.plugin:register" for m in discover_entrypoint_manifests())
    (source / ".env").write_text(
        "FIRST_SECRET=first\nSECOND_SECRET=visible\nNONCANONICAL_SECRET=extra\nLEAF_SECRET=leaf\nOPENAI_API_KEY=keep\n")
    (source / "config.yaml").write_text(yaml.safe_dump({
        "memory": {"provider": "builtin"}, "visible": {"token": "visible"}, "extra": {"token": "extra"}}))
    before, installed_before = _snapshot(source), _snapshot(site)
    runtime_before, env_before = _runtime_snapshot(), dict(os.environ)
    index = ChannelKeyIndex(source)
    assert index.platform_for("SECOND_SECRET") == "visible"
    for key in ("NONCANONICAL_SECRET", "LEAF_SECRET"):
        assert index.platform_for(key) == "extra"
    declarations = _declarations(package / "plugin.py")
    assert [(d.name, d.required_env) for d in declarations] == [
        ("visible", ["SECOND_SECRET"]), ("extra", ["NONCANONICAL_SECRET"]), ("extra", ["LEAF_SECRET"])]
    assert _runtime_snapshot() == runtime_before and dict(os.environ) == env_before
    destination = profiles.create_profile("destination", clone_from="source", clone_all=full)
    assert (destination / ".env").read_text() == "OPENAI_API_KEY=keep\n"
    cloned = yaml.safe_load((destination / "config.yaml").read_text())
    assert not {"visible", "extra"}.intersection(cloned)
    assert _snapshot(source) == before and _snapshot(site) == installed_before
    assert dict(os.environ) == env_before
    assert not any(name.startswith("relative_inventory_probe") for name in sys.modules)
    # Repeated edges above are deduplicated; a back-edge still fails closed.
    (package / "leaf.py").write_text(
        'from .helper import register_extra\n'
        'def register_leaf():\n    register_extra(None)\n')
    with pytest.raises(ValueError, match="recursive registration helper"):
        _declarations(package / "plugin.py")


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("source_mode", ["absent", "disabled", "enabled"])
def test_cli_bootstrap_clone_uses_explicit_source_without_activation(tmp_path, full, source_mode):
    import subprocess
    root = tmp_path / ".hermes"
    source = root / "profiles" / "source"
    source.mkdir(parents=True)
    (root / ".env").write_text("# isolated launch profile\n")
    (root / "config.yaml").write_text("memory:\n  provider: builtin\n")
    source_config: dict = {
        "memory": {"provider": "builtin"}, "model": {"default": "keep-model"},
    }
    marker = tmp_path / "external-command-ran"
    if source_mode != "absent":
        source_config["secrets"] = {"command": {
            "enabled": source_mode == "enabled", "command": f"touch '{marker}'"}}
    (source / "config.yaml").write_text(yaml.safe_dump(source_config))
    (source / ".env").write_text("OPENAI_API_KEY=keep\nTELEGRAM_BOT_TOKEN=drop\n")
    (source / "SOUL.md").write_text("source personality\n")
    bundled = tmp_path / "bundled" / "offline-channel"
    bundled.mkdir(parents=True)
    (bundled / "plugin.yaml").write_text("name: offline-channel\nkind: platform\n")
    (bundled / "__init__.py").write_text(CHANNEL_PLUGIN)
    with (source / ".env").open("a") as stream:
        stream.write("UNRELATED_BOT_SECRET=drop\n")
    before = _snapshot(source)
    # Exercise real main bootstrap/parser/dispatch, with only host integration and
    # the checkout's development dotenv location isolated. No clone call is mocked.
    script = '''import runpy, sys
from pathlib import Path
from hermes_cli import _startup_fast, profiles, plugins
from agent.secret_sources import registry as sources
from agent.secret_sources.base import SecretSource
source_calls = []
class SourceProbe(SecretSource):
    name = "offline_probe"
    def is_enabled(self, cfg):
        source_calls.append("enabled")
        return True
    def fetch(self, cfg, home_path):
        source_calls.append("fetch")
        raise AssertionError("external source fetched")
sources._SCOPED_SOURCES[str((Path.home() / ".hermes/profiles/source").resolve())] = {"offline_probe": SourceProbe()}
plugins.get_bundled_plugins_dir = lambda: Path.home() / "bundled"
_startup_fast.project_root_str = lambda: str(Path.home())
profiles._maybe_register_gateway_service = lambda name: None
profiles._notify_multiplexer = lambda name: None
sys.argv = ["hermes", "profile", "create", "destination", "--clone-from", "source", "--no-alias"] + sys.argv[1:]
try:
    runpy.run_module("hermes_cli.main", run_name="__main__")
except SystemExit as exc:
    expected = 1 if __import__('os').environ['EXPECT_REFUSAL'] == '1' else 0
    assert (exc.code or 0) == expected, exc.code
assert source_calls == [], source_calls
assert not plugins._plugin_managers_by_home
'''
    env = {"HOME": str(tmp_path), "HERMES_HOME": str(root), "PATH": os.environ["PATH"],
           "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PYTHONDONTWRITEBYTECODE": "1",
           "HERMES_ENABLE_PROJECT_PLUGINS": "0", "HERMES_SKIP_VENV_REEXEC": "1",
           "EXPECT_REFUSAL": str(int(source_mode == "enabled"))}
    result = subprocess.run([sys.executable, "-c", script, *( ["--clone-all"] if full else [])],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    destination = root / "profiles" / "destination"
    assert not marker.exists()
    assert _snapshot(source) == before
    assert not list(destination.parent.glob(".destination.staging-*"))
    if source_mode == "enabled":
        assert not destination.exists()
        assert "external secret source" in result.stdout + result.stderr
        return
    assert destination.is_dir()
    assert (destination / "SOUL.md").read_bytes() == (source / "SOUL.md").read_bytes()
    assert "TELEGRAM_BOT_TOKEN" not in (destination / ".env").read_text()
    assert "UNRELATED_BOT_SECRET" not in (destination / ".env").read_text()
    assert "OPENAI_API_KEY=keep" in (destination / ".env").read_text()
    assert _snapshot(source) == before


@pytest.mark.parametrize("surface,full,explicit", CASES)
@pytest.mark.parametrize("source_kind", ["command", "bitwarden", "offline_probe"])
def test_clone_surface_refuses_unbounded_secret_sources(clone_surface_env, surface, full, explicit, source_kind, capsys, monkeypatch):
    from agent.secret_sources import command, bitwarden, registry as sources
    from agent.secret_sources.base import SecretSource

    homes, notifications, seeded, connection = clone_surface_env
    source_name = "source" if explicit else ("default" if surface == "rest" else "caller")
    source = homes[source_name]
    cfg = yaml.safe_load((source / "config.yaml").read_text())
    cfg["secrets"] = {"sources": [], "preserve_existing": ["TELEGRAM_BOT_TOKEN"],
                      source_kind: {"enabled": True, "command": "private-command-sentinel",
                                    "project_id": "private-project-sentinel",
                                    "env": {"OPENAI_API_KEY": "not-a-supported-filter"}}}
    (source / "config.yaml").write_text(yaml.safe_dump(cfg))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("activated")
        raise AssertionError("must not resolve external sources")
    class OfflineProbe(SecretSource):
        name = "offline_probe"
        is_enabled = forbidden
        fetch = forbidden
    monkeypatch.setitem(sources._SOURCES, "offline_probe", OfflineProbe())
    monkeypatch.setattr(command, "_run_helper", forbidden)
    monkeypatch.setattr(bitwarden, "find_bws", forbidden)
    before = _snapshot(source)
    env_before, runtime_before = dict(os.environ), _runtime_snapshot()
    ok, result, output = _create(surface, "refused", full, explicit, capsys, connection)
    assert not ok, output
    assert "external secret source" in output and "fresh profile" in output
    assert "disabled" in output and "retry" in output.lower()
    assert "private-command-sentinel" not in output and "private-project-sentinel" not in output
    assert not profiles.get_profile_dir("refused").exists()
    assert not list((homes["default"] / "profiles").glob(".refused.staging-*"))
    assert _snapshot(source) == before
    assert notifications == seeded == calls == []
    assert dict(os.environ) == env_before and _runtime_snapshot() == runtime_before


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("source_mode", ["defaults", "disabled", "api_only", "mapped_channels"])
def test_clone_filters_only_declared_external_channel_bindings(clone_surface_env, full, source_mode, monkeypatch):
    from copy import deepcopy
    from agent.secret_sources import onepassword
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    homes, notifications, seeded, connection = clone_surface_env
    source = homes["source"]
    cfg: dict = {"memory": {"provider": "builtin"}}
    refs = {"OPENAI_API_KEY": "op://vault/model/key"}
    if source_mode == "defaults":
        secrets = deepcopy(DEFAULT_CONFIG["secrets"])
    elif source_mode == "disabled":
        secrets = {"command": {"enabled": False, "command": "never-run"},
                   "bitwarden": {"enabled": False, "project_id": "never-read"},
                   "offline_probe": {"enabled": False}, "sources": ["command", "offline_probe"]}
    else:
        if source_mode == "mapped_channels":
            refs.update(TELEGRAM_BOT_TOKEN="op://vault/bot/token",
                        TELEGRAM_BOT_TOKEN_DESTINATION="op://vault/bot/alias",
                        HASS_TOKEN="op://vault/home/token", HASS_URL="op://vault/home/url",
                        GATEWAY_ALLOWED_USERS="op://vault/bot/users")
        secrets = {"onepassword": {"enabled": True, "env": refs}}
    cfg["secrets"] = secrets
    (source / "config.yaml").write_text(yaml.safe_dump(cfg))
    # No companion rewrites the config in this test.
    monkeypatch.setattr("plugins.memory._MEMORY_PLUGINS_DIR", source / "absent-memory")
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: None)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("fetch")
        raise AssertionError("must not fetch mapped refs")
    monkeypatch.setattr(onepassword, "find_op", forbidden)
    before = _snapshot(source)
    destination = profiles.create_profile("destination", clone_from="source", clone_all=full)
    cloned = yaml.safe_load((destination / "config.yaml").read_text())["secrets"]
    if source_mode == "mapped_channels":
        assert cloned["onepassword"]["env"] == {"OPENAI_API_KEY": refs["OPENAI_API_KEY"]}
    elif source_mode == "defaults":
        # Lightweight cloning prunes empty mappings; it must preserve disablement.
        assert all(cloned[name]["enabled"] is False for name in secrets)
    else:
        assert cloned == secrets
    assert _snapshot(source) == before
    assert calls == []


@pytest.mark.parametrize("secret_config", [
    {"command": {"enabled": True, "command": "never-run"}},
    {"onepassword": {"enabled": True, "env": {"TELEGRAM_BOT_TOKEN": "op://vault/bot/token"}}},
])
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("live,served,source_name,refused", [
    pytest.param(False, [], "default", False, id="no-live-gateway"),
    pytest.param(False, ["default"], "default", False, id="stale-singleton-record"),
    pytest.param(True, [], "default", False, id="live-standalone"),
    pytest.param(True, ["default"], "default", True, id="live-singleton-multiplexer"),
    pytest.param(True, ["default"], "source", False, id="source-not-served"),
    pytest.param(True, ["default", "source"], "source", True, id="live-multiple-profiles"),
])
def test_clone_channels_opt_in_keeps_live_multiplexer_refusal_offline(
    clone_surface_env, secret_config, full, live, served, source_name, refused, monkeypatch,
):
    from hermes_cli import gateway_multiplex_served
    from agent.secret_sources import command, onepassword
    homes, notifications, seeded, connection = clone_surface_env
    source = homes[source_name]
    (source / "config.yaml").write_text(yaml.safe_dump({"secrets": secret_config}))
    default_config = yaml.safe_load((homes["default"] / "config.yaml").read_text())
    default_config["gateway"] = {"multiplex_profiles": True}
    (homes["default"] / "config.yaml").write_text(yaml.safe_dump(default_config))
    (homes["default"] / "gateway_state.json").write_text(json.dumps({"served_profiles": served}))
    # Keep the real record reader: standalone writes [], while even a singleton
    # multiplexer records its active profile. A stale record is not authoritative.
    monkeypatch.setattr(gateway_multiplex_served, "live_default_gateway_pid", lambda: 12345 if live else None)
    monkeypatch.setattr("plugins.memory._MEMORY_PLUGINS_DIR", source / "absent-memory")
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: notifications.append(name))
    service_calls = []
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: service_calls.append(name))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("fetch")
        raise AssertionError("opt-in does not authorize source hydration")
    monkeypatch.setattr(command, "_run_helper", forbidden)
    monkeypatch.setattr(onepassword, "find_op", forbidden)
    before = {name: {p: value for p, value in _snapshot(home).items()
                     if p.name != ".plugin-installation.lock"} for name, home in homes.items()}
    env_before, runtime_before = dict(os.environ), _runtime_snapshot()
    if refused:
        with pytest.raises(ValueError, match="multiplexed gateway"):
            profiles.create_profile("destination", clone_from=source_name, clone_all=full, clone_channels=True)
        assert not profiles.get_profile_dir("destination").exists()
        assert notifications == service_calls == []
    else:
        destination = profiles.create_profile("destination", clone_from=source_name, clone_all=full, clone_channels=True)
        assert yaml.safe_load((destination / "config.yaml").read_text())["secrets"] == secret_config
        assert notifications == ["destination"]
    for name, home in homes.items():
        current = {p: value for p, value in _snapshot(home).items() if p.name != ".plugin-installation.lock"}
        if name == "default":
            current = {p: value for p, value in current.items() if p.parts[:2] != ("profiles", "destination")}
        assert current == before[name]
    assert not list((homes["default"] / "profiles").glob(".destination.staging-*"))
    assert seeded == calls == []
    assert dict(os.environ) == env_before and _runtime_snapshot() == runtime_before
    assert get_hermes_home() == homes["caller"]


@pytest.mark.parametrize("surface,full,explicit", CASES)
def test_clone_surface_failure_never_publishes_or_leaks(clone_surface_env, surface, full, explicit, capsys):
    homes, notifications, seeded, connection = clone_surface_env
    source_name = "source" if explicit else ("default" if surface == "rest" else "caller")
    (homes[source_name] / "parity_probe.json").write_text(json.dumps({"peer": source_name, "fail": True}))
    before = {home: {p: value for p, value in _snapshot(home).items()
                     if p not in {Path("profiles") / name / ".plugin-installation.lock" for name in homes}}
              for home in homes.values()}
    env_before = dict(os.environ)
    runtime_before = _runtime_snapshot()
    ok, result, output = _create(surface, "refused", full, explicit, capsys, connection)
    assert not ok
    assert "clone" in output.lower() and "retry" in output.lower()
    assert "private-provider-sentinel" not in output
    assert not profiles.get_profile_dir("refused").exists()
    assert not list((homes["default"] / "profiles").glob(".refused.staging-*"))
    assert notifications == []
    assert seeded == []
    for home, snapshot in before.items():
        current = _snapshot(home)
        if home == homes["default"]:
            current = {p: value for p, value in current.items()
                       if p.parts[:2] != ("profiles", "destination")
                       and p not in {Path("profiles") / name / ".plugin-installation.lock" for name in homes}}
        assert current == snapshot
    assert _runtime_snapshot() == runtime_before
    assert dict(os.environ) == env_before
    assert get_hermes_home() == homes["caller"]
    assert secret_scope.get_secret("PARITY_SHELL_ONLY_KEY") == "caller-secret"
