"""plugins.isolation: host — linked, relative, renamed and repointed profile homes."""

import json
import os
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from tests.hermes_cli.plugin_host_support import (
    MODEL_PROVIDER_PLUGIN,
    _home_with_plugins,
    _hosted_model_provider,
)


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
    _hosted_model_provider(outside)
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
    from hermes_cli.plugins_loader import PluginHomeMoved
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
            pytest.raises(PluginHomeMoved, lambda: manager.owner_home)
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
        script.write_text(f'#!/bin/sh\necho "ran ${{OPENAI_API_KEY:-unset}} ${{LAUNCH_SETTING:-unset}}" >> "{ran}"\n'
                          f'exec "$@"\n', encoding="utf-8")
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
    ("link", "new", "new", None),  # bound after the repoint: holds new's values; no name leads to old's host
    ("link", "new", "old", "old"),  # bound while repointed, link restored before the start
    ("new", "old", "new", "new"),   # bound before the repoint: holds old's values, the host is new's
    ("launch", "old", "launch", "launch"),  # ... and the link now leads to the launch profile's directory
], ids=["bound_after_the_repoint", "restored_before_the_start", "bound_before_the_repoint",
        "repointed_at_the_launch_directory"])
def test_a_caller_bound_through_a_moving_profile_link_lends_its_values_to_no_host(repointed_profile_link, moves):
    """A caller bound through a profile's link holds what the link's directory held when it bound. However the
    link moved around that bind, the host that starts expands its launcher from its own profile's sources, and a
    host no name leads to any more starts none."""
    import contextlib

    from agent import secret_scope as ss
    from hermes_cli.plugins_loader import PluginHomeMoved
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
        with pytest.raises(PluginHomeMoved) if runs is None else contextlib.nullcontext():
            host.ensure_started()
    assert env.ran() == ([runs] if runs else [])


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
def test_a_secondary_hosts_child_takes_no_launch_only_value_after_the_launch_link_is_repointed_at_it(
        repointed_profile_link, tmp_path, monkeypatch, caller):
    """The host child of another profile never inherits the launch profile's values: neither a credential only its
    service env holds nor a setting its ``.env`` loaded into the process env. After the launch profile's own link is
    repointed at a secondary's directory, the secondary's host child is still another profile's, whether a body
    bound to the secondary starts it or not."""
    import contextlib

    from hermes_cli import env_loader
    from hermes_constants import hermes_home_key
    from tui_gateway import server

    env = repointed_profile_link
    monkeypatch.setattr(env_loader, "_DOTENV_PUBLISHED", {})
    monkeypatch.setattr(env_loader, "_LOADED_DOTENV_KEYS", set())
    monkeypatch.setenv("LAUNCH_SETTING", "registered-for-restore")
    monkeypatch.delenv("LAUNCH_SETTING")
    monkeypatch.setenv("OPENAI_API_KEY", "the-launch-profiles-key")
    launch_env_file = env.dirs["old"] / ".env"
    launch_env_file.write_text(launch_env_file.read_text(encoding="utf-8") + "LAUNCH_SETTING=from-the-launch-env-file\n",
                               encoding="utf-8")
    env.point("old")
    monkeypatch.setenv("HERMES_HOME", str(env.link))
    hermes_home_key(env.link)  # the process served its own profile, the old directory, under this name
    env_loader.load_hermes_dotenv(hermes_home=env.link, load_external_secrets=False)
    host = env.host_for(env.dirs["new"])
    body = (server._session_profile_runtime_scope({"profile_home": str(env.dirs["new"])}, hydrate_secrets=False)
            if caller == "bound_to_the_secondary" else contextlib.nullcontext())
    with body:
        env.point("new")
        host.ensure_started()
    assert (tmp_path / "new-runner.ran").read_text(encoding="utf-8").split() == ["ran", "unset", "unset"]


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
    """One start, one home: a profile name that moves while the host starts does not give the launcher check a
    different home from the one the host's environment and working directory were built for, and the start is
    refused once that name leads elsewhere (Greptile on #159450: it read the other directory's files)."""
    from hermes_cli import plugin_isolation
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugins_loader import PluginHomeMoved

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    (profiles / "ops").rename(profiles / "ops2")  # this start runs under the profile's new name
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops2"))
    build, launcher, asked = PluginHost._build_base_context, plugin_isolation.host_launcher, []

    def linked_while_building(self):  # then ops3 is the name that leads to the directory
        context = build(self)
        (profiles / "ops3").symlink_to(target, target_is_directory=True)
        (profiles / "ops2").unlink()
        (profiles / "ops2").symlink_to(tmp_path, target_is_directory=True)
        return context

    def recording(*args, **kwargs):
        asked.append(kwargs.get("home"))
        return launcher(*args, **kwargs)

    monkeypatch.setattr(PluginHost, "_build_base_context", linked_while_building)
    monkeypatch.setattr(plugin_isolation, "host_launcher", recording)
    try:
        with pytest.raises(PluginHomeMoved):
            host.ensure_started()
        assert host.home == profiles / "ops2" and host.pid is None
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
    _hosted_model_provider(target)
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
    _hosted_model_provider(target)
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
        change, tmp_path, monkeypatch, caplog):
    """A process keeps one manager per directory, so a linked profile renamed (or its link removed) while it
    runs keeps the manager it was opened by under the old name. The host restarted after a crash runs under the
    profile's new name, never the link that is gone; a reloaded plugin is told the profile of the home that host
    runs under. With no name leading to the directory, the crashed host's plugin is unloaded and stays down."""
    import signal

    from hermes_cli.plugins_loader import PluginHomeMoved, _plugin_home_scope
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
        if change == "removed":
            pytest.raises(PluginHomeMoved, lambda: manager.owner_home)
            os.kill(first, signal.SIGKILL)
            wait_until(lambda: "Not reloading plugin 'whereprobe'" in caplog.text, timeout=15, what="the refusal")
            assert "whereprobe" not in manager._plugins and '"pid"' not in registry.dispatch(
                "whereprobe_pid", {}, scope=manager.scope_key)
            return
        assert manager.owner_home == profiles / "ops2"
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
