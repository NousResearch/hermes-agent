"""plugins.isolation: host — third-party plugins run in a per-profile host process, reached only via ctx."""

import json
import os
import sys
import time
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

PROBE_PLUGIN = '''
import json, os
from agent.image_gen_provider import ImageGenProvider

class Painter(ImageGenProvider):
    @property
    def name(self):
        return "hostpainter"
    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": f"{prompt}@{os.getpid()}"}

SEEN = []

def register(ctx):
    ctx.register_tool(name="hostprobe_pid", toolset="hostprobe", schema={"name": "hostprobe_pid",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: json.dumps({"pid": os.getpid(), "seen": SEEN, "x": args.get("x")}))
    ctx.register_tool(name="hostprobe_nested", toolset="hostprobe", schema={"name": "hostprobe_nested",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: ctx.dispatch_tool("hostprobe_pid", {"x": "nested"}))
    ctx.register_tool(name="hostprobe_crash", toolset="hostprobe", schema={"name": "hostprobe_crash",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: os._exit(3))
    ctx.register_hook("post_tool_call", lambda tool_name=None, **kw: SEEN.append(tool_name))
    ctx.register_image_gen_provider(Painter())
'''

PLATFORM_PLUGIN = '''
def register(ctx):
    ctx.register_platform("x", "X", adapter_factory=lambda cfg: None, check_fn=lambda: True)
'''


def _home_with_plugins(tmp_path, monkeypatch, plugins: dict, isolation="host"):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    for name, body in plugins.items():
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": list(plugins), "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


@pytest.mark.platforms("posix")  # os._exit / SIGKILL process semantics
def test_plugin_runs_out_of_process_with_ctx_round_trips_and_survives_host_crash(tmp_path, monkeypatch):
    from agent import image_gen_registry
    from agent.image_gen_provider import ImageGenProvider
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN, "platformprobe": PLATFORM_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        # The plugin's code never entered this interpreter, yet its registrations are ordinary.
        assert not any("hostprobe" in name for name in sys.modules)
        assert manager._plugins["hostprobe"].error is None
        first = json.loads(registry.dispatch("hostprobe_pid", {"x": 1}, scope=manager.scope_key))
        host_pid = first["pid"]
        assert host_pid != os.getpid()
        # A handler calling back into Hermes while Hermes waits on it (nested ctx.dispatch_tool).
        nested = json.loads(registry.dispatch("hostprobe_nested", {}, scope=manager.scope_key))
        assert nested["pid"] == host_pid and nested["x"] == "nested"
        # Hooks fire in the host; provider objects arrive as instances of the ABC Hermes checks.
        manager.invoke_hook("post_tool_call", tool_name="read_file", args={}, result="ok")
        assert "read_file" in json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))["seen"]
        painter = image_gen_registry.get_provider("hostpainter")
        assert isinstance(painter, ImageGenProvider)
        assert painter.generate("cat") == {"success": True, "image": f"cat@{host_pid}"}
        # Live-object ctx surfaces fail the plugin with the boundary's reason, not a crash.
        assert "register_platform" in str(manager._plugins["platformprobe"].error)

        # A plugin killing its host costs one tool error; Hermes keeps running and the host restarts.
        crashed = registry.dispatch("hostprobe_crash", {}, scope=manager.scope_key)
        assert "plugin host" in crashed
        deadline, result = time.monotonic() + 15, ""
        while time.monotonic() < deadline:
            result = registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key)
            if '"pid"' in result:
                break
            time.sleep(0.2)
        assert json.loads(result)["pid"] not in {host_pid, os.getpid()}
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_isolation_host_keeps_every_user_import_path_out_of_process(tmp_path, monkeypatch):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir
    from plugins import plugin_loader

    home = _home_with_plugins(tmp_path, monkeypatch, {"platformprobe": PLATFORM_PLUGIN, "fine": PROBE_PLUGIN})
    # Category loaders' in-process import refuses user code under host isolation (the backstop
    # behind the memory / context-engine / cron host routes).
    import logging
    assert plugin_loader.load_plugin_module(
        "_hermes_user_x.fine", home / "plugins" / "fine", parents=("_hermes_user_x",),
        logger=logging.getLogger("t"), synthetic_namespace="_hermes_user_x") is None
    assert "_hermes_user_x.fine" not in sys.modules
    # The static audit reads the same boundary table the host enforces.
    assert audit_plugin_dir(home / "plugins" / "fine").verdict == "host"
    blocked = audit_plugin_dir(home / "plugins" / "platformprobe")
    assert blocked.verdict == "in_process" and "register_platform" in blocked.reasons[0]



def test_the_host_never_imports_a_package_planted_in_the_profile_home(tmp_path, monkeypatch):
    """The host process starts in the profile home. A home a plugin's backend writes (a profile it runs
    elsewhere) must not put a ``hermes_cli`` package of its own ahead of Hermes's."""
    home = _home_with_plugins(tmp_path, monkeypatch, {"quietprobe": "def register(ctx):\n    pass\n"})
    planted = tmp_path / "planted.ran"
    (home / "hermes_cli").mkdir()
    (home / "hermes_cli" / "__init__.py").write_text(f"open({str(planted)!r}, 'a').write('ran')\n", encoding="utf-8")
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert not planted.exists()
        assert manager._plugins["quietprobe"].error is None
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_an_empty_pythonpath_entry_does_not_put_the_profile_home_back_on_the_hosts_path(tmp_path, monkeypatch):
    """``-P`` keeps the home the host starts in off its import path, but an empty ``PYTHONPATH`` entry means
    the current directory again (``export PYTHONPATH=$PYTHONPATH:/x`` with it unset writes one)."""
    home = _home_with_plugins(tmp_path, monkeypatch, {"quietprobe": "def register(ctx):\n    pass\n"})
    planted = tmp_path / "planted.ran"
    (home / "json.py").write_text(f"open({str(planted)!r}, 'a').write('ran')\n", encoding="utf-8")
    userlib = tmp_path / "userlib"
    userlib.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep + str(userlib))
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert not planted.exists()
        assert manager._plugins["quietprobe"].error is None
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


def test_a_marked_profiles_own_launcher_is_not_used(tmp_path, monkeypatch):
    """``plugins.host.launcher`` is an argv prefix the host starts under. The backend of a profile a plugin
    runs elsewhere writes that profile's config.yaml, so only the managed overlay's launcher counts there."""
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_cli.plugin_isolation import host_launcher

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"isolation": "host", "host": {"launcher": ["profile-runner"]}}}), encoding="utf-8")
    assert host_launcher() == ["profile-runner"]
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    assert host_launcher() == []
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["operator-runner"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    assert host_launcher() == ["operator-runner"]


def test_a_managed_launcher_expands_its_env_refs_for_a_marked_profile_too(tmp_path, monkeypatch):
    """``${VAR}`` refs in the managed overlay expand against the process env. A profile a plugin runs
    elsewhere takes its launcher from that overlay alone, and gets the same expanded value as any profile."""
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_cli.plugin_isolation import host_launcher

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${PLUGIN_RUNNER}", "--quiet"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PLUGIN_RUNNER", "operator-runner")
    assert host_launcher() == ["operator-runner", "--quiet"]
    pre.mark_profile_run_elsewhere("default", "sandbox-plugin", root=home)
    assert host_launcher() == ["operator-runner", "--quiet"]


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_marked_profile_linked_out_of_the_root_does_not_get_its_own_launcher(tmp_path, monkeypatch):
    """``hermes -p ops`` names the home ``<root>/profiles/ops``, a link to a directory outside the root;
    discovery binds the directory it leads to, where no marker folder is. The marker is asked by the name."""
    from hermes_cli import profiles_run_elsewhere as pre

    outside = _home_with_plugins(tmp_path / "outside", monkeypatch, {"quietprobe": "def register(ctx):\n    pass\n"})
    ran = tmp_path / "launcher.ran"
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    (outside / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["quietprobe"], "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "ops").symlink_to(outside, target_is_directory=True)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=root)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert not ran.exists()
        assert manager._plugins["quietprobe"].error is None
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


def _marked_profile_linked_out_of_the_root(tmp_path, monkeypatch, plugins: dict, settings: dict):
    """``<root>/profiles/ops`` links to a directory outside the root and is marked; HERMES_HOME names it."""
    from hermes_cli import profiles_run_elsewhere as pre

    outside = _home_with_plugins(tmp_path / "outside", monkeypatch, plugins)
    (outside / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": list(plugins), "isolation": "host", **settings}}), encoding="utf-8")
    root = tmp_path / ".hermes"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "ops").symlink_to(outside, target_is_directory=True)
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=root)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    return outside


@pytest.mark.platforms("posix")  # a directory symlink
def test_plugin_discovery_for_a_marked_linked_profile_runs_under_the_name_its_marker_is_found_by(
        tmp_path, monkeypatch):
    """Code that runs while a manager discovers (a plugin's register, a module it imports that lists the
    model providers and so extracts a hosted one) asks the marker about the home it is bound to."""
    seen = tmp_path / "seen.txt"
    _marked_profile_linked_out_of_the_root(tmp_path, monkeypatch, {"markprobe": (
        "from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal\n"
        f"def register(ctx):\n    open({str(seen)!r}, 'w').write(str(run_elsewhere_refusal() is not None))\n")},
        {"isolation": "in_process"})
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert seen.read_text(encoding="utf-8") == "True"
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_marked_linked_profiles_provider_extraction_from_the_host_does_not_get_its_own_launcher(
        tmp_path, monkeypatch):
    """The host's start asks for hosted providers' ambient credentials, in the host's own context; with no
    extraction cached yet that is where the model-provider profiles are extracted, under that home's launcher."""
    ran = tmp_path / "launcher.ran"
    runner = tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    outside = _marked_profile_linked_out_of_the_root(
        tmp_path, monkeypatch, {"quietprobe": "def register(ctx):\n    pass\n"},
        {"host": {"launcher": [str(runner)]}})
    plugin_dir = outside / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert not ran.exists()
        assert manager._plugins["quietprobe"].error is None
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_the_host_child_takes_no_environment_from_a_marked_linked_profiles_env_file(tmp_path, monkeypatch):
    """A marked home's ``.env`` reads as empty; the host child's environment (a hosted provider's ambient
    credential included) is built for the home as named."""
    planted, seen = tmp_path / "planted.ran", tmp_path / "seen.txt"
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "json.py").write_text(f"open({str(planted)!r}, 'a').write('ran')\n", encoding="utf-8")
    outside = _marked_profile_linked_out_of_the_root(tmp_path, monkeypatch, {"envprobe": (
        f"import os\ndef register(ctx):\n    open({str(seen)!r}, 'w').write(os.environ.get('HOSTMODEL_KEY', ''))\n")}, {})
    plugin_dir = outside / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN.replace(
        'env_vars=("HOSTMODEL_KEY",)', 'env_vars=("HOSTMODEL_KEY",), ambient_credentials=True'), encoding="utf-8")
    (outside / ".env").write_text(f"PYTHONPATH={evil}\nHOSTMODEL_KEY=from-the-backend\n", encoding="utf-8")
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert not planted.exists()
        assert manager._plugins["envprobe"].error is None and seen.read_text(encoding="utf-8") == ""
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_marked_linked_profiles_provider_extraction_child_runs_under_the_name_its_marker_is_found_by(
        tmp_path, monkeypatch):
    """The one-shot extraction imports a model-provider plugin in a child process; code there asks the marker
    about the home it was given, as code in the plugin host does."""
    import providers

    seen = tmp_path / "seen.txt"
    outside = _marked_profile_linked_out_of_the_root(
        tmp_path, monkeypatch, {"quietprobe": "def register(ctx):\n    pass\n"}, {})
    plugin_dir = outside / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal as _refusal\n"
        f"open({str(seen)!r}, 'w').write(str(_refusal() is not None))\n" + MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    try:
        assert providers.get_provider_profile("hostmodel") is not None
        assert seen.read_text(encoding="utf-8") == "True"
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


def test_a_manager_opened_by_a_relative_home_runs_plugins_there_after_the_working_directory_moves(
        tmp_path, monkeypatch):
    """A symbolic profile id bound as a home is a relative path. The manager is keyed by the directory that
    name meant when it opened, so its plugins run there too, not wherever the name points from a later cwd."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    seen = tmp_path / "seen.txt"
    home = _home_with_plugins(tmp_path / "opened-in", monkeypatch, {"homeprobe": (
        "from hermes_constants import get_hermes_home\n"
        f"def register(ctx):\n    open({str(seen)!r}, 'w').write(str(get_hermes_home().resolve()))\n")},
        isolation="in_process")
    (tmp_path / "later").mkdir()
    monkeypatch.chdir(home.parent)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(home.name)
    try:
        manager = plugins_mod.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    monkeypatch.chdir(tmp_path / "later")
    manager.discover_and_load()
    try:
        assert seen.exists() and seen.read_text(encoding="utf-8") == str(home.resolve())
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_marked_linked_profile_opened_by_a_relative_name_finds_its_marker_after_the_working_directory_moves(
        tmp_path, monkeypatch):
    """The name a manager opened by is kept absolute: once the working directory moves, a relative ``ops`` would
    lead nowhere, and the directory outside the root it links to has no marker to find."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    seen = tmp_path / "seen.txt"
    _marked_profile_linked_out_of_the_root(tmp_path, monkeypatch, {"markprobe": (
        "from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal\n"
        f"def register(ctx):\n    open({str(seen)!r}, 'w').write(str(run_elsewhere_refusal() is not None))\n")},
        {"isolation": "in_process"})
    (tmp_path / "later").mkdir()
    monkeypatch.chdir(tmp_path / ".hermes" / "profiles")
    token = set_hermes_home_override("ops")
    try:
        manager = plugins_mod.get_plugin_manager()
    finally:
        reset_hermes_home_override(token)
    monkeypatch.chdir(tmp_path / "later")
    manager.discover_and_load()
    try:
        assert seen.read_text(encoding="utf-8") == "True"
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script; SIGKILL
@pytest.mark.parametrize("beside", [None, "_backup", "default", "deleted", "relinked", "uppercase"])
def test_a_linked_profile_marked_under_its_new_name_after_a_rename_is_found_by_the_restarted_host(
        beside, tmp_path, monkeypatch):
    """A rename leaves the kept manager holding a name that leads nowhere. The host restarted after the profile
    was marked under its new name must find that marker, so the profile's own launcher stays unused. Another
    entry leading to the same directory that is not a live profile id (sorted ahead of the new name) is not
    taken for it, a name that appears after a lookup found none is still found, and a new name spelled in
    capitals is read as the marker lookup reads it."""
    import signal

    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_constants import mark_named_profile_deleted
    from tests.waiting import wait_until
    from tools.registry import registry

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {"whereprobe": (
        "import json, os\ndef register(ctx):\n"
        "    ctx.register_tool(name='whereprobe_pid', toolset='whereprobe', schema={'name': 'whereprobe_pid',\n"
        "        'description': 'd', 'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kw: json.dumps({'pid': os.getpid()}))\n")})
    (target / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["whereprobe"], "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
    root = tmp_path / "root"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles" / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()

    def served():
        out = registry.dispatch("whereprobe_pid", {}, scope=manager.scope_key)
        return out if '"pid"' in out else None

    try:
        first = json.loads(served())["pid"]
        assert ran.read_text(encoding="utf-8").count("ran") == 1  # unmarked: its launcher is the profile's own
        if beside == "relinked":  # the link is gone while a lookup runs, then comes back under the new name
            (root / "profiles" / "ops").unlink()
            assert manager.owner_home == manager.home_path
            (root / "profiles" / "ops").symlink_to(target, target_is_directory=True)
        elif beside in ("_backup", "default", "deleted"):
            other = root / "profiles" / ("aaa" if beside == "deleted" else beside)
            other.symlink_to(target, target_is_directory=True)
            if beside == "deleted":
                mark_named_profile_deleted(other)
        new_name = "Ops2" if beside == "uppercase" else "ops2"
        (root / "profiles" / "ops").rename(root / "profiles" / new_name)
        pre.mark_profile_run_elsewhere(new_name, "sandbox-plugin", root=root)
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / new_name))
        os.kill(first, signal.SIGKILL)
        assert json.loads(wait_until(served, timeout=15, what="the restarted host"))["pid"] != first
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_host_whose_new_home_failed_to_bind_binds_it_on_the_next_start(tmp_path, monkeypatch):
    """A start that moves the host to the manager's current home and fails to build that home's context leaves
    the next start to build it again, instead of starting under the context of the home it left."""
    from hermes_cli.plugin_host import PluginHost
    from hermes_constants import get_hermes_home

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    (profiles / "ops").rename(profiles / "ops2")
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops2"))
    build, failures = PluginHost._build_base_context, [RuntimeError("profile secrets unreadable")]

    def flaky(self):
        if failures:
            raise failures.pop()
        return build(self)

    monkeypatch.setattr(PluginHost, "_build_base_context", flaky)
    try:
        with pytest.raises(RuntimeError):
            host.ensure_started()
        host.ensure_started()
        assert host._base_context.run(get_hermes_home) == host.home == manager.owner_home
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_profile_link_repointed_at_another_directory_loads_plugins_under_the_new_directorys_key(
        tmp_path, monkeypatch):
    """A process remembers the key each home spelling resolved to. After a profile's link is repointed at another
    directory, plugin code loaded for the new directory's manager must not run under that spelling, whose
    remembered key is still the old directory's."""
    from hermes_constants import hermes_home_key

    out = tmp_path / "key.out"
    monkeypatch.setenv("KEYPROBE_OUT", str(out))
    old = _home_with_plugins(tmp_path / "old", monkeypatch, {}, isolation="in_process")
    new = _home_with_plugins(tmp_path / "new", monkeypatch, {"keyprobe": (
        "import os\nfrom pathlib import Path\nfrom hermes_constants import hermes_home_key\n"
        "def register(ctx):\n    Path(os.environ['KEYPROBE_OUT']).write_text(hermes_home_key())\n")},
        isolation="in_process")
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(old, target_is_directory=True)
    hermes_home_key(link)  # the process served the old directory under this name
    link.unlink()
    link.symlink_to(new, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        assert out.read_text() == manager.scope_key == hermes_home_key(new)
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.fixture
def repointed_profile_link(tmp_path, monkeypatch):
    """Two profile directories, each with a ``${PLUGIN_RUNNER}`` launcher its own ``.env`` names, the launch home
    with the same launcher and its value only in the process env, and a profile link ``ops`` that can be pointed at
    any of them. Yields ``point(dir)``, ``host_for(home)`` and which runners ran, multiplexing for the test's sake
    (a caller can then bind any of them)."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho "ran ${{OPENAI_API_KEY:-unset}}" >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    runs = {name: runner(f"{name}-runner") for name in ("old", "new", "launch")}
    dirs = {name: _home_with_plugins(tmp_path / name, monkeypatch, {}) for name in runs}  # launch last: HERMES_HOME
    launcher = yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}})
    for name, home in dirs.items():
        (home / "config.yaml").write_text(launcher, encoding="utf-8")
        if name != "launch":
            (home / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / f'{name}-runner'}\n", encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))  # the launch profile's, env-only
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    hosts = []

    def point(name):
        if link.is_symlink():
            link.unlink()
        link.symlink_to(dirs[name], target_is_directory=True)

    def host_for(home):
        token = set_hermes_home_override(home)
        try:
            hosts.append(plugins_mod.get_plugin_manager()._plugin_host())
        finally:
            reset_hermes_home_override(token)
        return hosts[-1]

    plugins_mod._reset_plugin_managers_for_tests()
    context_token = ss.set_multiplex_context(True)
    try:
        yield SimpleNamespace(link=link, dirs=dirs, point=point, host_for=host_for,
                              ran=lambda: sorted(name for name, path in runs.items() if path.exists()))
    finally:
        for host in hosts:
            host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
@pytest.mark.parametrize("moves", [
    # (the host's manager, opened while the link points at old; where the link points when the caller binds through
    # it; where it points when the host starts; whose runner that host runs)
    ("link", "new", "new", "old"),  # bound after the repoint: holds new's values, the host is old's
    ("link", "new", "old", "old"),  # bound while repointed, link restored before the start
    ("new", "old", "new", "new"),   # bound before the repoint: holds old's values, the host is new's
    ("launch", "old", "launch", "launch"),  # ... and the link now leads to the launch profile's directory
], ids=["bound_after_the_repoint", "restored_before_the_start", "bound_before_the_repoint",
        "repointed_at_the_launch_directory"])
def test_a_caller_bound_through_a_moving_profile_link_lends_its_values_to_no_host(repointed_profile_link, moves):
    """A caller bound through a profile's link holds what the link's directory held when it bound. However the
    link moved around that bind, the host that starts expands its launcher from its own profile's sources."""
    from agent import secret_scope as ss
    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    manager, bound, started, runs = moves
    env.point("old")
    host = env.host_for(env.link if manager == "link" else env.dirs[manager])
    env.point(bound)
    hermes_home_key(env.link)  # the process serves that profile under this name
    with server._session_profile_runtime_scope({"profile_home": str(env.link)}, hydrate_secrets=False):
        assert ss.get_secret("PLUGIN_RUNNER").endswith(f"{bound}-runner")
        env.point(started)
        host.ensure_started()
    assert env.ran() == [runs]


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
def test_a_launch_body_bound_before_its_own_link_was_repointed_lends_nothing_to_the_new_directorys_host(
        repointed_profile_link, monkeypatch):
    """The launch profile's body holds what its directory held when it bound. After the process's own link is
    repointed at another profile's directory, that profile's host expands from its own files, not the body's."""
    from agent import secret_scope as ss
    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process served its own profile, the old directory, under this name
    host = env.host_for(env.dirs["new"])
    with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
        assert ss.get_secret("PLUGIN_RUNNER").endswith("old-runner")
        env.point("new")
        host.ensure_started()
    assert env.ran() == ["new"]


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
@pytest.mark.parametrize("caller", ["bound_to_the_secondary", "none"])
def test_a_secondary_host_takes_no_launch_env_value_after_the_launch_link_is_repointed_at_its_directory(
        repointed_profile_link, monkeypatch, caller):
    """The process names its own profile by the key it remembered for its own path. After that link is repointed at
    a secondary's directory, the secondary's host is still another profile's: a launcher name its files lack stays
    unresolved, never the launch profile's env-only value, whether a body bound to the secondary starts it or not."""
    import contextlib

    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    (env.dirs["new"] / ".env").write_text("", encoding="utf-8")  # the launch runner is then in the process env only
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process served its own profile, the old directory, under this name
    host = env.host_for(env.dirs["new"])
    body = (server._session_profile_runtime_scope({"profile_home": str(env.dirs["new"])}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    with body:
        env.point("new")
        with pytest.raises(FileNotFoundError, match=r"\$\{PLUGIN_RUNNER\}"):
            host.ensure_started()
    assert env.ran() == []


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
@pytest.mark.parametrize("caller", ["bound_to_the_secondary", "none"])
def test_a_secondary_hosts_child_takes_no_launch_only_credential_after_the_launch_link_is_repointed_at_it(
        repointed_profile_link, tmp_path, monkeypatch, caller):
    """The host child of another profile never inherits a credential only the launch profile's service env holds.
    After the launch profile's own link is repointed at a secondary's directory, the secondary's host child is
    still another profile's, whether a body bound to the secondary starts it or not."""
    import contextlib

    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    monkeypatch.setenv("OPENAI_API_KEY", "the-launch-profiles-key")
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process served its own profile, the old directory, under this name
    host = env.host_for(env.dirs["new"])
    body = (server._session_profile_runtime_scope({"profile_home": str(env.dirs["new"])}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    with body:
        env.point("new")
        host.ensure_started()
    assert (tmp_path / "new-runner.ran").read_text(encoding="utf-8").split() == ["ran", "unset"]


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
@pytest.mark.parametrize("caller", ["bound_to_the_secondary", "none"])
def test_a_secondary_hosts_env_templates_resolve_as_another_profiles_after_the_launch_link_is_repointed_at_it(
        repointed_profile_link, tmp_path, monkeypatch, caller):
    """Another profile's ``.env`` templates resolve against process-global values only, never the launch env. After
    the launch profile's own link is repointed at a secondary's directory, the secondary's host still resolves them
    that way: a ``${NAME:-default}`` whose name only the launch env holds takes its default."""
    import contextlib

    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    (env.dirs["new"] / ".env").write_text(
        f"PLUGIN_RUNNER=${{LAUNCH_ONLY_RUNNER:-{tmp_path / 'new-runner'}}}\n", encoding="utf-8")
    monkeypatch.setenv("LAUNCH_ONLY_RUNNER", str(tmp_path / "launch-runner"))
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process served its own profile, the old directory, under this name
    host = env.host_for(env.dirs["new"])
    body = (server._session_profile_runtime_scope({"profile_home": str(env.dirs["new"])}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    with body:
        env.point("new")
        host.ensure_started()
    assert env.ran() == ["new"]


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launchers are shell scripts
@pytest.mark.parametrize("surface", ["tui_gateway", "dashboard"])
def test_a_launch_body_bound_while_the_process_link_pointed_elsewhere_gives_the_launch_host_its_own_files(
        repointed_profile_link, monkeypatch, surface):
    """A launch-profile body that bound while the process's own link pointed at another directory holds that
    directory's ``.env``. If the link is back before the launch host starts, the host reads the launch profile's own
    files, not the body's values."""
    from hermes_cli.web_server_profiles import _config_profile_scope
    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process serves its own profile, the old directory, under this name
    host = env.host_for(env.link)
    env.point("new")
    body = (server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False)
            if surface == "tui_gateway" else _config_profile_scope(None))
    with body:
        env.point("old")
        host.ensure_started()
    assert env.ran() == ["old"]


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
def test_a_single_profile_launcher_expands_against_the_process_env_not_a_later_env_file(tmp_path, monkeypatch):
    """Outside multiplexing a launcher's ``${VAR}`` is the running process's value, as on main: an ``.env`` edited
    after the process loaded it does not pick a different launcher for the next host start."""
    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    loaded_ran, edited_ran = runner("loaded-runner"), runner("edited-runner")
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "loaded-runner"))
    (home / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'edited-runner'}\n", encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert loaded_ran.exists() and not edited_ran.exists()
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("surface", ["tui_gateway", "dashboard"])
def test_a_host_started_inside_the_launch_profiles_bound_scope_takes_its_launcher_value_from_it(
        tmp_path, monkeypatch, surface):
    """The TUI gateway and the dashboard bind the launch profile's secrets around their bodies even single-profile
    (``.env`` over the live env; the dashboard stamps that scope with the launch home). A host started inside such a
    body expands its launcher from that scope, as on main, so a value only the ``.env`` holds (set after the process
    loaded it) still reaches the launcher."""
    from hermes_cli.web_server_profiles import _config_profile_scope
    from tui_gateway import server

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.delenv("PLUGIN_RUNNER", raising=False)
    (home / ".env").write_text(f"PLUGIN_RUNNER={runner}\n", encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        body = (server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False)
                if surface == "tui_gateway" else _config_profile_scope(None))
        with body:
            host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


def _tagging_launch_host(tmp_path, monkeypatch, dotenv: str):
    """The launch profile's host with a ``[runner, ${RUNNER_TAG}]`` launcher whose runner records its tag, the home's
    ``.env`` set to *dotenv*. Returns the host, the tags file and the home."""
    tags, runner = tmp_path / "tags", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1" >> "{tags}"\nshift\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": [str(runner), "${RUNNER_TAG}"]}}}), encoding="utf-8")
    (home / ".env").write_text(dotenv, encoding="utf-8")
    plugins_mod._reset_plugin_managers_for_tests()
    return plugins_mod.get_plugin_manager()._plugin_host(), tags, home


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("dotenv, expected", [
    ("", "${RUNNER_TAG}"),  # a name the launch scope lacks stays unresolved
    ("RUNNER_TAG=${RUNNER_OVERRIDE:-from-the-env-file}\n", "from-the-env-file"),  # a template keeps its default
], ids=["missing_name", "env_file_template"])
def test_a_launch_body_under_task_local_multiplexing_gives_its_launcher_no_value_written_to_the_live_env_since(
        tmp_path, monkeypatch, dotenv, expected):
    """A launch-profile body under task-local multiplexing (the desktop cron ticker) never reads a value from the
    live process env, which another profile's task may have written since: a name its scope lacks stays
    unresolved, and a ``.env`` template resolves against the launch env frozen when multiplexing began. The
    launcher of a host it starts reads its values the same way."""
    from agent import secret_scope as ss
    from tui_gateway import server

    for name in ("RUNNER_TAG", "RUNNER_OVERRIDE"):
        monkeypatch.delenv(name, raising=False)
    host, tags, _ = _tagging_launch_host(tmp_path, monkeypatch, dotenv)
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            for name in ("RUNNER_TAG", "RUNNER_OVERRIDE"):
                monkeypatch.setenv(name, "written-after-the-scope-was-bound")
            assert ss.get_secret("RUNNER_TAG") == (expected if dotenv else None)
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == [expected]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("reloaded", [False, True], ids=["loaded_once", "reloaded_since_the_freeze"])
def test_a_launch_launchers_env_file_template_resolves_as_the_process_loaded_the_file(
        tmp_path, monkeypatch, reloaded):
    """Under multiplexing the launch profile's launcher resolves its ``.env`` templates against the frozen launch
    env with the process's own dotenv output peeled back, as the process loaded the file and the body reads it: a
    line naming a key the file only sets later takes its default, not that later value. A reload since the freeze
    that changed that value does not change it."""
    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import server

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    for name in ("RUNNER_TAG", "RUNNER_BASE"):
        monkeypatch.setenv(name, "registered-for-restore")
        monkeypatch.delenv(name)
    host, tags, home = _tagging_launch_host(tmp_path, monkeypatch, "RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/old\n")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    assert os.environ["RUNNER_TAG"] == "unset/tag"
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            assert ss.get_secret("RUNNER_TAG") == "unset/tag"
            if reloaded:
                (home / ".env").write_text("RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/new\n", encoding="utf-8")
                env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
                assert os.environ["RUNNER_BASE"] == "/new"
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == ["unset/tag"]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_reload_racing_the_launch_env_freeze_cannot_split_the_frozen_env_from_its_dotenv_record(
        tmp_path, monkeypatch):
    """The launch env freezes once, with the dotenv publication record that peels its dotenv output back for the
    launcher's ``.env`` templates. A ``.env`` reload racing that freeze waits for it, so the frozen env is never
    paired with a later reload's record (which no longer recognises its values) until the process restarts."""
    import threading

    from agent import secret_scope as ss
    from hermes_cli import env_loader
    from tui_gateway import launch_profile_policy, server

    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    for name in ("RUNNER_TAG", "RUNNER_BASE"):
        monkeypatch.setenv(name, "registered-for-restore")
        monkeypatch.delenv(name)
    host, tags, home = _tagging_launch_host(tmp_path, monkeypatch, "RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/old\n")
    env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    (home / ".env").write_text("RUNNER_TAG=${RUNNER_BASE:-unset}/tag\nRUNNER_BASE=/new\n", encoding="utf-8")
    env_copied, reloaded = threading.Event(), threading.Event()
    record = env_loader.dotenv_publication_record

    def record_after_the_env_copy():
        env_copied.set()
        reloaded.wait(2)  # a reload not held off by the freeze lands here, between the two copies
        return record()

    def reload():
        env_copied.wait(10)
        env_loader.load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
        reloaded.set()

    monkeypatch.setattr(env_loader, "dotenv_publication_record", record_after_the_env_copy)
    racer = threading.Thread(target=reload, daemon=True)
    racer.start()
    launch_profile_policy.capture_launch_env()
    racer.join(10)
    assert reloaded.is_set() and os.environ["RUNNER_BASE"] == "/new"
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            host.ensure_started()
        assert tags.read_text(encoding="utf-8").split() == ["unset/tag"]
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
@pytest.mark.parametrize("caller", ["launch", "secondary"])
def test_a_caller_bound_to_another_profile_lends_its_scope_to_no_other_profiles_launcher(
        tmp_path, monkeypatch, caller):
    """A caller bound to one profile gives another profile's launcher none of its values. A body bound to the
    launch profile that starts a secondary's host leaves the secondary on its own ``.env``, and a body bound to a
    secondary that starts the launch profile's host leaves it on the env the process loaded, not on a launch
    ``.env`` edited since."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    ops_ran, launch_ran, edited_ran = runner("ops-runner"), runner("launch-runner"), runner("edited-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {})
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    launcher = yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}})
    for home in (ops, launch):
        (home / "config.yaml").write_text(launcher, encoding="utf-8")
    (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'ops-runner'}\n", encoding="utf-8")
    (launch / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'edited-runner'}\n", encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    plugins_mod._reset_plugin_managers_for_tests()
    started, bound_to = (ops, None) if caller == "launch" else (launch, str(ops))
    home_token = set_hermes_home_override(started)
    try:
        host = plugins_mod.get_plugin_manager()._plugin_host()
    finally:
        reset_hermes_home_override(home_token)
    context_token = ss.set_multiplex_context(True)
    try:
        with server._session_profile_runtime_scope({"profile_home": bound_to}, hydrate_secrets=False):
            host.ensure_started()
        ran = [path.exists() for path in (ops_ran, launch_ran, edited_ran)]
        assert ran == ([True, False, False] if started == ops else [False, True, False])
    finally:
        host.shutdown()
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts
def test_a_secondary_profiles_launcher_under_task_local_multiplexing_expands_from_its_own_env_file(
        tmp_path, monkeypatch):
    """A process that is not a multiplexer still serves another profile in a task (the desktop backend's cron
    ticker firing a sibling's job): that profile's launcher expands from its own ``.env``, never the launch
    profile's value for the name, whichever context the host start runs in."""
    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway.launch_profile_policy import owner_secret_scope

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    own_ran, launch_ran = runner("own-runner"), runner("launch-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {})
    _home_with_plugins(tmp_path, monkeypatch, {})
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'own-runner'}\n", encoding="utf-8")
    monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    plugins_mod._reset_plugin_managers_for_tests()
    context_token = ss.set_multiplex_context(True)
    home_token = set_hermes_home_override(ops)
    scope_token = ss.set_secret_scope(owner_secret_scope(ops), profile_home=str(ops))
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert own_ran.exists() and not launch_ran.exists()
    finally:
        host.shutdown()
        ss.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        ss.reset_multiplex_context(context_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launchers are shell scripts; SIGKILL
@pytest.mark.parametrize("crashed", ["secondary", "launch"])
def test_a_crash_restart_under_multiplexing_runs_the_profiles_own_launcher(tmp_path, monkeypatch, crashed):
    """A crash restart starts the host from a thread with no binding. A ``${VAR}`` launcher still expands from
    the crashed profile's own secrets: a secondary's ``.env`` (never the launch profile's value for the name),
    or the launch profile's env-only keys (systemd ``Environment=``), which no file holds."""
    import signal

    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tests.waiting import wait_until
    from tools.registry import registry
    from tui_gateway.launch_profile_policy import owner_secret_scope

    def runner(name):
        ran, script = tmp_path / f"{name}.ran", tmp_path / name
        script.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
        script.chmod(0o755)
        return ran

    def runs(ran):
        return ran.read_text(encoding="utf-8").count("ran") if ran.exists() else 0

    own_ran, launch_ran = runner("own-runner"), runner("launch-runner")
    ops = _home_with_plugins(tmp_path / "ops", monkeypatch, {"hostprobe": PROBE_PLUGIN})
    launch = _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN})
    home = ops if crashed == "secondary" else launch
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["hostprobe"], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    if crashed == "secondary":
        (ops / ".env").write_text(f"PLUGIN_RUNNER={tmp_path / 'own-runner'}\n", encoding="utf-8")
        monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "launch-runner"))
    else:
        monkeypatch.setenv("PLUGIN_RUNNER", str(tmp_path / "own-runner"))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    # The first start runs inside the profile's binding, as a gateway's load of that profile's plugins does.
    home_token = set_hermes_home_override(home)
    scope_token = ss.set_secret_scope(owner_secret_scope(home), profile_home=str(home))
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        first = manager._plugins["hostprobe"]
        assert first.error is None
        assert runs(own_ran) == 1
        crashed_pid = manager._plugin_host().pid
        os.kill(crashed_pid, signal.SIGKILL)

        def reloaded():
            loaded = manager._plugins.get("hostprobe")
            return loaded if loaded is not first else None

        # The restarted host answered hello and loaded the plugin again, so both launcher counts are final.
        after = wait_until(reloaded, timeout=15, what="the crashed host's restart and reload")
        assert after is not None and after.error is None
        answer = json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))
        assert answer["pid"] not in {crashed_pid, os.getpid()}
        assert (runs(own_ran), runs(launch_ran)) == (2, 0)
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        ss.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script; SIGKILL
@pytest.mark.parametrize("env_file, tag", [
    ("RUNNER_TAG=from-env-file\n", "from-env-file"),
    ("RUNNER_TAG=${LAUNCH_ONLY_TAG:-not-the-launch-env}\n", "from-launch-env"),  # a template of an env-only key
], ids=["plain_value", "template_of_an_env_only_key"])
def test_a_crash_restart_after_the_launch_profiles_link_is_renamed_keeps_its_env_only_launcher_keys(
        tmp_path, monkeypatch, env_file, tag):
    """The process keeps naming its own profile by the path it started with. After that link is renamed the
    manager answers to the new name, and a crash restart under multiplexing still expands the launcher from both
    of the launch profile's sources: its env-only keys (systemd ``Environment=``) and its ``.env``, read through
    the name that now leads to it, its templates resolved as the process's own against those env-only keys. A
    start inside a launch-profile body bound since (its scope read through the old name, so without the ``.env``)
    builds that scope again through the new name rather than borrowing the body's. The host child's own env is
    still the launch profile's too: a credential only the service env holds, and its ``.env`` resolved the same
    way."""
    import signal

    from agent import secret_scope as ss
    from tests.waiting import wait_until
    from tui_gateway import server

    ran, seen, runner = tmp_path / "launcher.ran", tmp_path / "child-env.seen", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho "$1" >> "{ran}"\n'
                      f'echo "${{RUNNER_TAG:-unset}}|${{OPENAI_API_KEY:-unset}}" >> "{seen}"\nshift\nexec "$@"\n',
                      encoding="utf-8")
    runner.chmod(0o755)
    child = f"{tag}|from-the-service"
    outside = _home_with_plugins(tmp_path / "outside", monkeypatch, {"hostprobe": PROBE_PLUGIN})
    (outside / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["hostprobe"], "isolation": "host", "host": {"launcher": ["${PLUGIN_RUNNER}", "${RUNNER_TAG}"]}}}),
        encoding="utf-8")
    (outside / ".env").write_text(env_file, encoding="utf-8")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(outside, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    monkeypatch.setenv("PLUGIN_RUNNER", str(runner))
    monkeypatch.setenv("LAUNCH_ONLY_TAG", "from-launch-env")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-service")  # e.g. systemd Environment=, no .env line
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        first = manager._plugins["hostprobe"]
        assert first.error is None and ran.read_text(encoding="utf-8").split() == [tag]
        assert seen.read_text(encoding="utf-8").split() == [child]
        (profiles / "ops").rename(profiles / "ops2")
        os.kill(manager._plugin_host().pid, signal.SIGKILL)
        def reloaded():
            loaded = manager._plugins.get("hostprobe")
            return loaded if loaded is not first else None

        after = wait_until(reloaded, timeout=15, what="the crashed host's restart")
        assert after is not None and after.error is None
        assert ran.read_text(encoding="utf-8").split() == [tag, tag]
        assert seen.read_text(encoding="utf-8").split() == [child, child]
        manager._plugin_host().shutdown()
        with server._session_profile_runtime_scope({"profile_home": None}, hydrate_secrets=False):
            assert ss.get_secret("RUNNER_TAG") is None
            manager._plugin_host().ensure_started()
        assert ran.read_text(encoding="utf-8").split() == [tag] * 3
        assert seen.read_text(encoding="utf-8").split() == [child] * 3
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
def test_a_marked_profiles_managed_launcher_expands_against_the_process_env_under_multiplexing(tmp_path, monkeypatch):
    """Under multiplexing too, a profile a plugin runs elsewhere takes its launcher from the managed overlay alone,
    its ``${VAR}``s expanded against the process env as for any profile, not through a scope built for its home."""
    from agent import secret_scope as ss
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host"}}), encoding="utf-8")
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=launch)
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${PLUGIN_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("PLUGIN_RUNNER", str(runner))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # the stand-in launcher is a shell script
@pytest.mark.parametrize("caller", ["none", "bound_to_the_secondary"])
def test_a_secondary_profiles_managed_launcher_expands_against_the_process_env_under_multiplexing(
        tmp_path, monkeypatch, caller):
    """The managed overlay's ``${VAR}``s expand against the process env, so a user's ``.env`` cannot stand in for
    an administrator's value. A secondary profile's launcher is read inside that profile's own secret scope, but
    a managed launcher still takes a name only the process env holds (systemd ``Environment=``)."""
    import contextlib

    from agent import secret_scope as ss
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tui_gateway import server

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    launch = _home_with_plugins(tmp_path, monkeypatch, {})
    ops = launch / "profiles" / "ops"
    ops.mkdir(parents=True)
    (ops / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": [], "isolation": "host"}}), encoding="utf-8")
    (ops / ".env").write_text("PLUGIN_RUNNER=/not/the/operators/runner\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"host": {"launcher": ["${SANDBOX_RUNNER}"]}}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("SANDBOX_RUNNER", str(runner))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    token = set_hermes_home_override(ops)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    body = (server._session_profile_runtime_scope({"profile_home": str(ops)}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    try:
        with body:
            host.ensure_started()
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        reset_hermes_home_override(token)
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script
def test_a_host_started_by_a_caller_bound_to_a_since_renamed_link_runs_the_profiles_launcher(tmp_path, monkeypatch):
    """The launcher comes from the config of the home the host starts in, not from the caller's binding: a load
    binds the name it read, which a rename can retire before that load starts the host."""
    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    (target / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": [], "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    try:
        with plugins_mod._plugin_home_scope(manager.owner_home):  # a load binds the name it read
            (profiles / "ops").rename(profiles / "ops2")
            host.ensure_started()
        assert host.home == profiles / "ops2"
        assert ran.read_text(encoding="utf-8").count("ran") == 1
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_host_start_asks_about_its_launcher_for_the_home_it_starts_in(tmp_path, monkeypatch):
    """One start, one home: a profile name that appears while the host starts does not give the launcher check a
    different home from the one the host's environment and working directory were built for."""
    from hermes_cli import plugin_isolation
    from hermes_cli.plugin_host import PluginHost

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    (profiles / "ops").unlink()  # no name leads to the directory: this start runs under the directory itself
    monkeypatch.setenv("HERMES_HOME", str(target))
    build, launcher, asked = PluginHost._build_base_context, plugin_isolation.host_launcher, []

    def linked_while_building(self):
        context = build(self)
        (profiles / "ops2").symlink_to(target, target_is_directory=True)
        return context

    def recording(*args, **kwargs):
        asked.append(kwargs.get("home"))
        return launcher(*args, **kwargs)

    monkeypatch.setattr(PluginHost, "_build_base_context", linked_while_building)
    monkeypatch.setattr(plugin_isolation, "host_launcher", recording)
    try:
        host.ensure_started()
        assert host.home == manager.home_path
        assert asked == [host.home]
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; SIGKILL
def test_a_hosted_model_provider_of_a_renamed_linked_profile_still_runs_after_a_host_crash(tmp_path, monkeypatch):
    """A hosted provider's overrides run in the host, addressed by the plugin's directory. That address must
    outlive a rename of the link the profile was opened by, as the restarted host loads the plugin from it."""
    import signal

    import providers
    from tests.waiting import wait_until

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    plugin_dir = target / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()

    def body():
        try:
            return profile.build_extra_body(session_id="s1")
        except Exception:
            return None

    try:
        profile = providers.get_provider_profile("hostmodel")
        assert profile is not None
        first = profile.build_extra_body(session_id="s1")["pid"]
        (profiles / "ops").rename(profiles / "ops2")
        monkeypatch.setenv("HERMES_HOME", str(profiles / "ops2"))
        os.kill(first, signal.SIGKILL)
        after = wait_until(body, timeout=15, what="the restarted host to run the provider's override")
        assert after["session_id"] == "s1" and after["pid"] not in {first, os.getpid()}
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_hosted_model_providers_address_names_the_installation_its_profiles_were_extracted_from(
        tmp_path, monkeypatch):
    """A hosted provider's profiles come from a one-shot extraction and its overrides run in the host, addressed
    by the plugin's directory. Both name one installation: a profile link repointed while the extraction runs,
    then restored, leaves the provider runnable, never addressed at the other directory's copy of the path."""
    import providers
    from hermes_cli import plugin_host_profiles

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    plugin_dir = target / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    elsewhere = _home_with_plugins(tmp_path / "elsewhere", monkeypatch, {})
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    extract = plugin_host_profiles._extract

    def extract_then_repoint(*args, **kwargs):
        payload = extract(*args, **kwargs)
        link.unlink()
        link.symlink_to(elsewhere, target_is_directory=True)
        return payload

    monkeypatch.setattr(plugin_host_profiles, "_extract", extract_then_repoint)
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        profile = providers.get_provider_profile("hostmodel")
        assert profile is not None
        link.unlink()
        link.symlink_to(target, target_is_directory=True)
        assert profile.build_extra_body(session_id="s1")["session_id"] == "s1"
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; SIGKILL
def test_a_hosted_model_provider_whose_install_link_was_removed_stays_removed_after_a_host_crash(
        tmp_path, monkeypatch):
    """Only the profile home's spelling is resolved in a hosted provider's address: a plugin installed as a link
    and then removed is not loaded again from the link's old target by the restarted host."""
    import signal

    import providers
    from tests.waiting import wait_until

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    installed = tmp_path / "installed" / "hostmodel"
    installed.mkdir(parents=True)
    (installed / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (installed / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    (home / "plugins" / "model-providers").mkdir(parents=True)
    (home / "plugins" / "model-providers" / "hostmodel").symlink_to(installed, target_is_directory=True)
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        profile = providers.get_provider_profile("hostmodel")
        assert profile is not None
        first = profile.build_extra_body(session_id="s1")["pid"]
        killed = plugins_mod.get_plugin_manager()._plugin_host()
        (home / "plugins" / "model-providers" / "hostmodel").unlink()
        os.kill(first, signal.SIGKILL)
        wait_until(lambda: not killed.alive, timeout=15, what="the killed host to be noticed")
        with pytest.raises(Exception, match="__init__.py"):
            profile.build_extra_body(session_id="s1")
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_recall_matched_by_the_recorded_repo_refuses_a_linked_profiles_plugin(tmp_path, monkeypatch):
    """Discovery records a linked profile's plugins under its directory while code runs under its name; the
    installer's record is still found, so a kill-list entry matching only the recorded repo refuses the load."""
    from hermes_cli import plugins_cmd_catalog
    from hermes_cli.plugin_catalog import RemovedEntry

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {"recalled": "def register(ctx):\n    pass\n"},
                                isolation="in_process")
    (target / "plugins" / ".install-metadata.json").write_text(
        json.dumps({"recalled": {"source": "https://github.com/example/recalled"}}), encoding="utf-8")
    monkeypatch.setattr(plugins_cmd_catalog, "cached_removed_entries", lambda: [RemovedEntry(
        name="another-name", repo="https://github.com/example/recalled", reason="backdoor")])
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        assert "backdoor" in (manager._plugins["recalled"].error or "")
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; SIGKILL
@pytest.mark.parametrize("change", ["renamed", "removed"])
def test_a_linked_profile_whose_link_moves_while_its_manager_is_kept_gets_its_host_plugins_back_after_a_crash(
        change, tmp_path, monkeypatch):
    """A process keeps one manager per directory, so a linked profile renamed (or its link removed) while it
    runs keeps the manager it was opened by under the old name. The host restarted after a crash runs under the
    profile's new name, or the directory when no name leads there, never the link that is gone; a reloaded
    plugin is told the profile of the home that host runs under."""
    import signal

    from hermes_cli.plugins_loader import _plugin_home_scope
    from hermes_cli.profiles import get_active_profile_name
    from tests.waiting import wait_until
    from tools.registry import registry

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {"whereprobe": (
        "import json, os\ndef register(ctx):\n"
        "    ctx.register_tool(name='whereprobe_pid', toolset='whereprobe', schema={'name': 'whereprobe_pid',\n"
        "        'description': 'd', 'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kw: json.dumps({'pid': os.getpid(), 'profile': ctx.profile_name}))\n")})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        first = json.loads(registry.dispatch("whereprobe_pid", {}, scope=manager.scope_key))["pid"]
        if change == "renamed":
            (profiles / "ops").rename(profiles / "ops2")
            monkeypatch.setenv("HERMES_HOME", str(profiles / "ops2"))
        else:
            (profiles / "ops").unlink()
            monkeypatch.setenv("HERMES_HOME", str(target))
        assert plugins_mod.get_plugin_manager() is manager
        assert manager.owner_home == {"renamed": profiles / "ops2", "removed": manager.home_path}[change]
        os.kill(first, signal.SIGKILL)

        def served():
            out = registry.dispatch("whereprobe_pid", {}, scope=manager.scope_key)
            return out if '"pid"' in out else None

        result = wait_until(served, timeout=15, what="the restarted host to serve the plugin's tool")
        assert json.loads(result)["pid"] not in {first, os.getpid()}
        with _plugin_home_scope(manager.owner_home):
            assert json.loads(result)["profile"] == get_active_profile_name()
    finally:
        manager.unload()
        manager._plugin_host().shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


def test_managed_scope_pins_host_isolation_over_the_profiles_own_config(tmp_path, monkeypatch):
    """The isolated party must not be able to opt out: an operator pin in the managed scope
    (/etc/hermes/config.yaml) wins over a profile config that says in_process."""
    from hermes_cli import managed_scope
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN}, isolation="in_process")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert not any("hostprobe" in name for name in sys.modules)
        result = json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))
        assert result["pid"] != os.getpid()
    finally:
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        managed_scope.invalidate_managed_cache()

MODEL_PROVIDER_PLUGIN = '''
import os
from providers import register_provider
from providers.base import ProviderProfile

class HostModel(ProviderProfile):
    def build_extra_body(self, *, session_id=None, **context):
        return {"pid": os.getpid(), "session_id": session_id}

register_provider(HostModel(name="hostmodel", base_url="https://hostmodel.example/v1", env_vars=("HOSTMODEL_KEY",)))
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_model_provider_profile_data_is_local_and_overrides_run_in_the_host(tmp_path, monkeypatch):
    import providers

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    plugin_dir = home / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    profile = providers.get_provider_profile("hostmodel")
    try:
        assert isinstance(profile, providers.ProviderProfile)
        # Discovery never started a host: data came from the cached, credential-free extraction.
        assert profile.base_url == "https://hostmodel.example/v1" and tuple(profile.env_vars) == ("HOSTMODEL_KEY",)
        assert list((home / "cache" / "plugin_host" / "model-providers").glob("hostmodel-*.json"))
        assert not any("hostmodel" in name for name in sys.modules)
        body = profile.build_extra_body(session_id="s1")
        assert body["session_id"] == "s1" and body["pid"] != os.getpid()
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


ASYNC_PLUGIN = '''
import asyncio, json
S = {"type": "object", "properties": {}}

def register(ctx):
    async def whoami(args, **kw):  # async plugin code calling back into Hermes
        return json.dumps({"inner": json.loads(ctx.dispatch_tool("session_probe", {})),
                           "acomplete_awaitable": asyncio.iscoroutinefunction(ctx.llm.acomplete)})
    ctx.register_tool(name="async_whoami", toolset="asyncprobe", schema={"name": "async_whoami",
        "description": "d", "parameters": S}, handler=whoami, is_async=True)
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_async_plugin_code_calls_back_in_the_callers_session(tmp_path, monkeypatch):
    import contextvars
    from tools.registry import registry

    session = contextvars.ContextVar("session", default="<unset>")
    session.set("host-creator")  # whoever first touches the host must not leak into later calls
    registry.register(name="session_probe", toolset="asyncprobe", schema={"name": "session_probe",
                      "description": "d", "parameters": {"type": "object", "properties": {}}},
                      handler=lambda args, **kw: json.dumps({"session": session.get()}))
    _home_with_plugins(tmp_path, monkeypatch, {"asyncprobe": ASYNC_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        def call_as(name):
            def run():
                session.set(name)
                return json.loads(registry.dispatch("async_whoami", {}, scope=manager.scope_key))
            return contextvars.copy_context().run(run)

        assert call_as("session-B") == {"inner": {"session": "session-B"}, "acomplete_awaitable": True}
        assert call_as("session-C")["inner"] == {"session": "session-C"}
    finally:
        registry.deregister("session_probe")
        manager.unload()
        manager._plugin_host().shutdown()


MEMORY_PLUGIN = '''
import os
from agent.memory_provider import MemoryProvider

class Probe(MemoryProvider):
    level = 0  # a class-level default the provider changes later
    @property
    def name(self): return "memprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kw): pass
    def get_tool_schemas(self): return []
    def whoami(self):
        self.level += 1
        return os.getpid()

def register(ctx):
    ctx.register_memory_provider(Probe())
'''


@pytest.mark.platforms("posix")  # SIGKILL
def test_hosted_memory_provider_stays_live_across_a_host_crash(tmp_path, monkeypatch):
    import signal
    from plugins.memory import load_memory_provider
    from plugins.memory.config_schema import get_provider_config_schema

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text(MEMORY_PLUGIN, encoding="utf-8")
    marker = tmp_path / "schema_pid"
    (provider_dir / "config_schema.py").write_text(
        f"import os, pathlib\npathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\nCONFIG_SCHEMA = None\n",
        encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        provider = load_memory_provider("memprobe")
        first_pid = provider.whoami()
        assert first_pid == host.pid != os.getpid()
        assert provider.level == 1  # read live from the plugin's object, not the class default
        # The provider's schema file is user code too: it runs in the host, never here.
        get_provider_config_schema("memprobe")
        assert int(marker.read_text(encoding="utf-8-sig")) == host.pid

        os.kill(first_pid, signal.SIGKILL)  # windows-footgun: ok — posix-only test (platforms marker)
        deadline = time.monotonic() + 10
        while host.alive and time.monotonic() < deadline:
            time.sleep(0.1)
        # The proxy Hermes already holds reloads the provider into the new host on next use.
        assert provider.whoami() not in {first_pid, os.getpid()}
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_host_plugin_can_launch_status_and_steer_subagent(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from agent.subagent_lifecycle import bind_subagent_parent, SubagentLifecycleService
    from tools.delegate_tool_registry import _register_subagent, _unregister_subagent
    from tools.registry import registry

    body = '''
import json, os
from agent.subagent_lifecycle import SubagentLaunchRequest
def register(ctx):
    def run(args, **kw):
        service = ctx.subagent_lifecycle
        handle = service.launch(SubagentLaunchRequest(goal="review from host"))
        return json.dumps({"pid": os.getpid(), "state": service.status(handle).state,
                           "accepted": service.steer(handle, "check correctness"), "handle": handle})
    ctx.register_tool(name="host_lifecycle", toolset="hostlife",
        schema={"name": "host_lifecycle", "description": "d",
                "parameters": {"type": "object", "properties": {}}}, handler=run)
'''
    finish, steering = threading.Event(), []
    child = SimpleNamespace(_subagent_id="sa-host-lifecycle", provider="test", model="test-model",
                            steer=lambda text: steering.append(text) is None)
    parent = SimpleNamespace(session_id="parent-host-lifecycle")

    def build(**kwargs):
        assert kwargs["parent_agent"] is parent
        assert kwargs["goal"] == "review from host"
        _register_subagent({"subagent_id": child._subagent_id, "agent": child})
        return child

    def run(*_args):
        assert finish.wait(5)
        return {"status": "completed", "summary": "host child done"}

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", build)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run)
    _home_with_plugins(tmp_path, monkeypatch, {"hostlife": body})
    manager = PluginManager()
    manager.discover_and_load()
    service = SubagentLifecycleService(lambda: parent)
    result = None
    try:
        with bind_subagent_parent(parent):
            result = json.loads(registry.dispatch("host_lifecycle", {}, scope=manager.scope_key))
        assert result["pid"] != os.getpid()
        assert result["state"] in {"PENDING", "RUNNING"}
        assert result["accepted"] is True
        assert steering == ["check correctness"]
    finally:
        finish.set()
        from agent.subagent_lifecycle import _REGISTRY
        record = _REGISTRY.records.get(child._subagent_id)
        if record is not None:
            record.future.result(timeout=5)
        _unregister_subagent(child._subagent_id, agent=child)
        manager.unload()
        manager._plugin_host().shutdown()
    assert service.result(result["handle"]).summary == "host child done"


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["host", "in_process"])
def test_failed_register_runs_its_on_unload_cleanup(tmp_path, monkeypatch, isolation):
    # A register() that queued cleanup and then raised must have that cleanup run, in either mode.
    marker = tmp_path / "cleaned"
    body = f'''
from pathlib import Path
def register(ctx):
    ctx.on_unload(lambda: Path({str(marker)!r}).write_text("cleaned"))
    raise RuntimeError("register failed")
'''
    _home_with_plugins(tmp_path, monkeypatch, {"halfload": body}, isolation=isolation)
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert "register failed" in str(manager._plugins["halfload"].error)
        assert marker.is_file(), "the failed load's on_unload cleanup never ran"
    finally:
        manager.unload()
        if isolation == "host":
            manager._plugin_host().shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("cleanup", ['raise SystemExit("cleanup exit")', "import os; os._exit(7)"])
def test_failed_hosted_load_keeps_its_error_when_cleanup_fails(tmp_path, monkeypatch, cleanup):
    # Cleanup that raises or kills the host must not replace the register() error that is recorded.
    body = f'''
def register(ctx):
    def cleanup():
        {cleanup}
    ctx.on_unload(cleanup)
    raise RuntimeError("register failed")
'''
    _home_with_plugins(tmp_path, monkeypatch, {"halfload": body})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert "register failed" in str(manager._plugins["halfload"].error)
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("held_in", ["register", "host_start"])
def test_late_failure_of_an_abandoned_load_leaves_the_reloaded_plugin_alone(tmp_path, monkeypatch, held_in):
    # A hosted load abandoned at its deadline (inside register(), or still waiting for the host to start)
    # can fail after a reload of the same plugin succeeded. Host state is keyed by plugin, not by load,
    # so that late failure must not unload the live load.
    from hermes_cli import plugin_host, plugins_loader
    from tests.waiting import wait_until
    from tools.registry import registry

    state = tmp_path / "state"
    state.mkdir()
    # The abandoned load's register() is the first to run when it is held inside register(), the second
    # when it is held before the host start returns (the reload reaches the host first).
    failing_attempt = 1 if held_in == "register" else 2
    body = f'''
import json, time
from pathlib import Path
STATE = Path({str(state)!r})
def register(ctx):
    attempt = len(list(STATE.glob("attempt-*"))) + 1
    (STATE / f"attempt-{{attempt}}").write_text("x")
    if attempt == {failing_attempt}:
        while not (STATE / "go").exists():
            time.sleep(0.05)
        raise RuntimeError("abandoned load failed late")
    ctx.register_tool(name="lateprobe_ping", toolset="lateprobe", schema={{"name": "lateprobe_ping",
        "description": "d", "parameters": {{"type": "object", "properties": {{}}}}}},
        handler=lambda args, **kw: json.dumps({{"pong": True}}))
'''
    home = _home_with_plugins(tmp_path, monkeypatch, {"lateprobe": body})
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["lateprobe"], "isolation": "host", "load_timeout_seconds": 2}}), encoding="utf-8")
    manager = PluginManager()
    manager._plugin_host().ensure_started()  # outside any load deadline: a slow host start can't time out the reload
    if held_in == "host_start":
        start, callers = plugin_host.PluginHost.ensure_started, []

        def slow_first_start(self):
            channel = start(self)
            callers.append(self)
            if len(callers) == 1:
                wait_until(lambda: (state / "go").exists(), timeout=30, what="the test to release the host start")
            return channel

        monkeypatch.setattr(plugin_host.PluginHost, "ensure_started", slow_first_start)
    manager.discover_and_load()
    try:
        assert "timed out" in str(manager._plugins["lateprobe"].error)
        manager.discover_and_load(force=True)
        assert manager._plugins["lateprobe"].enabled
        (state / "go").write_text("go")
        loader = f"{plugins_loader._LOADER_THREAD_PREFIX}lateprobe"
        wait_until(lambda: not any(t.is_alive() and t.name == loader for t in plugins_loader._ABANDONED_LOADERS),
                   timeout=15, what="the abandoned first load to finish")
        assert len(list(state.glob("attempt-*"))) == 2
        assert json.loads(registry.dispatch("lateprobe_ping", {}, scope=manager.scope_key)) == {"pong": True}
    finally:
        (state / "go").write_text("go")
        manager.unload()
        manager._plugin_host().shutdown()
