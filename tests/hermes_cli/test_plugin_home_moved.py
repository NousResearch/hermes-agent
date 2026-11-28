"""A plugin manager whose directory no profile name leads to any more refuses plugin work (Greptile review on
#159450): the directory alone reads as an unmarked profile, so a marked profile's run-elsewhere marker would go
unseen and its own launcher would run."""

import contextlib
from pathlib import Path

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from tests.hermes_cli.test_plugin_host import _home_with_plugins

UNLOAD_PROBE = (
    "import os\nfrom pathlib import Path\nfrom hermes_constants import get_hermes_home\n"
    "def register(ctx):\n"
    "    ctx.on_unload(lambda: Path(os.environ['UNLOADPROBE_OUT']).write_text(str(get_hermes_home())))\n")


def _hosted_provider_behind_a_launcher(tmp_path, home):
    """Install the hosted model provider ``hostmodel`` under *home*, whose config runs the plugin host and
    extractions under a stand-in launcher; return the file that launcher appends to each time it runs."""
    from tests.hermes_cli.test_plugin_host import _hosted_model_provider

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    _hosted_model_provider(home)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
    return ran


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script
def test_a_marked_profile_link_repointed_after_its_key_was_remembered_runs_no_plugin_code(tmp_path, monkeypatch):
    """A process remembers the key each home spelling resolved to. After a marked profile's link is repointed at
    another directory, that name still keys the old one, and the new directory alone is outside the root its marker
    is read in. Its manager refuses plugin work rather than run there as an unmarked profile: no plugin loads and the
    profile's own launcher never runs, until a restart opens the profile again."""
    from hermes_cli import profiles_run_elsewhere as pre
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import hermes_home_key

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    old = _home_with_plugins(tmp_path / "old", monkeypatch, {})
    new = _home_with_plugins(tmp_path / "new", monkeypatch, {"tiny": "def register(ctx):\n    pass\n"})
    (new / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "enabled": ["tiny"], "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(old, target_is_directory=True)
    hermes_home_key(link)  # the process served the old directory under this name
    pre.mark_profile_run_elsewhere("ops", "sandbox-plugin", root=tmp_path / "root")
    link.unlink()
    link.symlink_to(new, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        with pytest.raises(PluginHomeMoved):
            manager.discover_and_load()
        with pytest.raises(PluginHomeMoved):
            manager._plugin_host().ensure_started()
        assert not ran.exists() and not manager._plugins
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
@pytest.mark.parametrize("target", ["opened_before", "opened_after"])
def test_a_link_repointed_at_another_homes_directory_is_refused_and_leaves_that_homes_plugins_alone(
        target, tmp_path, monkeypatch):
    """A request through a profile link repointed while the process runs is refused, whether the directory it now
    leads to was opened as its own home before the repoint or after it, and that home's plugins still load."""
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import hermes_home_key

    old = _home_with_plugins(tmp_path / "old", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {"tiny": "def register(ctx):\n    pass\n"},
                               isolation="in_process")
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(old, target_is_directory=True)
    plugins_mod._reset_plugin_managers_for_tests()

    def manager_for(home):
        monkeypatch.setenv("HERMES_HOME", str(home))
        return plugins_mod.get_plugin_manager()

    try:
        manager_for(link)  # the process opens ops at the old directory
        hermes_home_key(link)  # and serves it under this name
        if target == "opened_before":
            manager_for(other).discover_and_load()
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        with pytest.raises(PluginHomeMoved):
            manager_for(link).discover_and_load()
        others = manager_for(other)
        others.discover_and_load()
        assert "tiny" in others._plugins and others.owner_home == other
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_profile_directory_replaced_by_a_link_while_the_process_runs_is_refused(tmp_path, monkeypatch):
    """A profile opened as a plain directory and then replaced by a link to another home refuses plugin work,
    rather than read the other home's plugins under the key of the directory it replaced."""
    from hermes_cli.plugins_loader import PluginHomeMoved

    other = _home_with_plugins(tmp_path / "other", monkeypatch, {"tiny": "def register(ctx):\n    pass\n"},
                               isolation="in_process")
    ops = _home_with_plugins(tmp_path / "root" / "profiles" / "ops", monkeypatch, {}, isolation="in_process")
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        plugins_mod.get_plugin_manager()  # HERMES_HOME is the plain directory
        ops.rename(tmp_path / "ops-was-here")
        ops.symlink_to(other, target_is_directory=True)
        manager = plugins_mod.get_plugin_manager()
        with pytest.raises(PluginHomeMoved):
            manager.discover_and_load()
        assert "tiny" not in manager._plugins
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink in one case
@pytest.mark.parametrize("gone", ["removed", "renamed", "link_target_removed"])
def test_a_profile_directory_removed_or_renamed_after_its_plugins_loaded_refuses_plugin_work(gone, tmp_path, monkeypatch):
    """A plain profile directory's name still resolves to itself once the directory is removed or renamed away, and
    a link to a removed directory still resolves to it, so the name read as leading to the manager's directory and
    discovery ran there with no profile left. Once the manager has seen its directory, a name that leads to no
    directory is refused like a moved link."""
    import shutil

    from hermes_cli.plugin_isolation import ISOLATION_IN_PROCESS
    from hermes_cli.plugins_loader import PluginHomeMoved

    seeded = _home_with_plugins(tmp_path, monkeypatch, {"tiny": "def register(ctx):\n    pass\n"},
                                isolation=ISOLATION_IN_PROCESS)
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    if gone == "link_target_removed":
        home, name = seeded.rename(tmp_path / "outside"), profiles / "ops"
        name.symlink_to(home, target_is_directory=True)
    else:
        home = name = seeded.rename(profiles / "ops")
    monkeypatch.setenv("HERMES_HOME", str(name))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        assert manager._plugins
        if gone == "renamed":
            home.rename(profiles / "ops2")
        else:
            shutil.rmtree(home)
        with pytest.raises(PluginHomeMoved):
            manager.discover_and_load(force=True)
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


def test_a_profile_directory_removed_before_its_first_discovery_refuses_plugin_work(tmp_path, monkeypatch):
    """A directory that existed when its manager opened and is removed before discovery first looks it up (the
    background discovery thread) is a removed home, not one that was never created."""
    import shutil

    from hermes_cli.plugins_loader import PluginHomeMoved

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        shutil.rmtree(home)
        with pytest.raises(PluginHomeMoved):
            manager.discover_and_load()
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


def test_a_home_not_created_yet_is_not_refused(tmp_path, monkeypatch):
    """A home that has never existed (a first run, or a profile about to be created) still discovers plugins."""
    _home_with_plugins(tmp_path, monkeypatch, {})
    home = tmp_path / "root" / "profiles" / "fresh"
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        assert manager.owner_home == home
    finally:
        manager.unload()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # directory symlinks
def test_a_reused_profile_name_reaches_no_host_of_the_renamed_profile(tmp_path, monkeypatch):
    """Managers are found by the key the process remembered for a name. After a linked profile is renamed and its
    old name is reused for another directory, a request through the old name finds the renamed profile's manager,
    which runs under the new name with that profile's environment. A caller hands such a host plugin paths it read
    through its own name, the other directory's, so the request gets no host; the renamed profile still does."""
    from hermes_cli.plugin_isolation import transient_plugin_host, user_plugin_host
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import hermes_home_key

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(home, target_is_directory=True)
    hermes_home_key(profiles / "ops")  # the process opened the profile under this name
    (profiles / "ops").rename(profiles / "ops2")
    (profiles / "ops").symlink_to(other, target_is_directory=True)  # the old name, reused
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops2"))
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        assert user_plugin_host().home == profiles / "ops2"
        monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
        with pytest.raises(PluginHomeMoved):
            user_plugin_host()
        with pytest.raises(PluginHomeMoved), transient_plugin_host():
            pass
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # directory symlinks
def test_a_profiles_dashboard_plugins_scanned_through_a_name_since_reused_are_scanned_again(tmp_path, monkeypatch):
    """The dashboard caches a secondary profile's plugins by the key the process remembered, with paths read
    through the name it scanned. After ops is renamed to ops2 and the name ops is reused for another directory,
    a request through ops2 got those paths, which now lead to the other directory's plugin code."""
    from hermes_cli import web_server_plugin_profiles as profiles_mod
    from tests.hermes_cli.test_plugin_api_consent_gate import _install

    monkeypatch.setattr(profiles_mod, "_profile_plugins", {})
    monkeypatch.setattr(profiles_mod, "_profile_rescans", {})
    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {})
    _install(home, "dash")
    _install(other, "dash")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(home, target_is_directory=True)
    dash = lambda name: next(p for p in profiles_mod.profile_dashboard_plugins(profiles / name) if p["name"] == "dash")
    assert Path(dash("ops")["_dir"]).resolve() == (home / "plugins" / "dash" / "dashboard").resolve()
    (profiles / "ops").rename(profiles / "ops2")
    (profiles / "ops").symlink_to(other, target_is_directory=True)  # the old name, reused
    assert Path(dash("ops2")["_dir"]).resolve() == (home / "plugins" / "dash" / "dashboard").resolve()


@contextlib.contextmanager
def _launch_dashboard_profiles(tmp_path, monkeypatch, *names, plugin_in_root=False):
    """A launch home and another home, each enabling the hosted dashboard plugin ``dash``, installed in each home or
    once in the Hermes root (*plugin_in_root*), with ``<root>/profiles/<name>`` linked to the launch home for each
    of *names*; the first is the process's home. Yields ``(home, other, profiles)``."""
    from hermes_cli import web_server
    from tests.hermes_cli.test_plugin_api_consent_gate import _install

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {})
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    for each in (home, other):
        (each / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": ["dash"], "isolation": "host"}}),
                                          encoding="utf-8")
    for each in (profiles.parent,) if plugin_in_root else (home, other):
        _install(each, "dash")
    for name in names:
        (profiles / name).symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(profiles / names[0]))
    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        yield home, other, profiles
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # directory symlinks
@pytest.mark.parametrize("moved", ["before_the_first_lookup", "before_the_lookup", "during_the_lookup"])
def test_a_launch_home_name_that_moves_forwards_no_dashboard_plugin(moved, tmp_path, monkeypatch):
    """The dashboard's hosted routes forward the launch profile's copy of a plugin, discovered through the process's
    own home name, ops. A request through another name for the launch home (ops2 after a rename that reused ops, or
    a second live link repointed while the lookup runs) still counts as the launch profile, the same remembered key,
    and passes its own name's check, while discovery reads the other directory through ops: that directory's plugin
    code would run in the launch profile's host."""
    from hermes_cli import web_server, web_server_dashboard
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    with _launch_dashboard_profiles(tmp_path, monkeypatch, "ops") as (home, other, profiles):
        if moved == "before_the_first_lookup":
            hermes_home_key(profiles / "ops")  # the process opened its home under this name, then no lookup ran
        else:
            _, dashboard_dir, _ = web_server_dashboard._hosted_plugin_for_request("dash")
            assert Path(dashboard_dir).resolve() == (home / "plugins" / "dash" / "dashboard").resolve()

        def repoint_ops():
            (profiles / "ops").unlink(missing_ok=True)
            (profiles / "ops").symlink_to(other, target_is_directory=True)

        if moved != "during_the_lookup":
            (profiles / "ops").rename(profiles / "ops2")
            repoint_ops()  # the old name, reused
        else:
            (profiles / "ops2").symlink_to(home, target_is_directory=True)
            listed = web_server._get_dashboard_plugins
            monkeypatch.setattr(web_server, "_get_dashboard_plugins",
                                lambda *a, **k: (repoint_ops(), listed(*a, **k))[1])
        token = set_hermes_home_override(profiles / "ops2")
        try:
            with pytest.raises(PluginHomeMoved):
                web_server_dashboard._hosted_plugin_for_request("dash")
        finally:
            reset_hermes_home_override(token)


@pytest.mark.platforms("posix")  # directory symlinks
@pytest.mark.parametrize("moved", ["in_the_queue", "while_the_host_starts", "while_releases_flush"])
def test_a_dashboard_plugin_whose_home_name_moves_after_its_lookup_is_not_loaded_by_the_host(moved, tmp_path,
                                                                                            monkeypatch):
    """A forwarded request waits in its profile's queue, and the host may start cold, before the host loads the
    plugin's path. The lookup ran through a second live name for the launch home (ops2) and found the copy read
    through ops; ops repointed meanwhile, or while the host is sent releases queued ahead of the request, would hand
    the host another directory's plugin code, while ops2 still passes. The path's own name is checked right before
    the host is sent it."""
    from hermes_cli import web_server_dashboard
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    loaded, starts = [], []
    with _launch_dashboard_profiles(tmp_path, monkeypatch, "ops", "ops2") as (_, other, profiles):
        def repoint_ops():
            (profiles / "ops").unlink()
            (profiles / "ops").symlink_to(other, target_is_directory=True)

        # A started host is a no-op start; the second request's start is a cold one during which ops moves.
        monkeypatch.setattr(PluginHost, "ensure_started",
                            lambda self: starts.append(1) if len(starts) < 1 or moved != "while_the_host_starts"
                            else (starts.append(1), repoint_ops()))

        class Channel:  # the host's end: queued releases go first, then the request that hands it the path
            def call(self, method, params, timeout=None):
                if method == "release" and moved == "while_releases_flush":
                    repoint_ops()
                elif method == "asgi":
                    loaded.append(params["dashboard_dir"])
                return {}

        token = set_hermes_home_override(profiles / "ops2")
        try:
            host, dashboard_dir, api_file = web_server_dashboard._hosted_plugin_for_request("dash")
            host._channel = Channel()
            host.asgi_request("dash", dashboard_dir, api_file, "GET", "/", "", [], b"")
            assert loaded == [dashboard_dir]
            if moved == "in_the_queue":
                repoint_ops()  # while the request waits for its profile's threads
            host._release(host._generation, 7)  # an object proxy collected meanwhile: its release goes out first
            with pytest.raises(PluginHomeMoved):
                host.asgi_request("dash", dashboard_dir, api_file, "GET", "/", "", [], b"")
            assert loaded == [dashboard_dir]
        finally:
            reset_hermes_home_override(token)


@pytest.mark.platforms("posix")  # directory symlinks
def test_a_plugin_in_the_hermes_root_still_serves_the_renamed_launch_profile(tmp_path, monkeypatch):
    """After the launch link ops is renamed to ops2, a request through ops2 still finds a plugin installed in the
    Hermes root. Its path never went through ops, so the moved launch name doesn't refuse it; a copy read through
    ops is refused (test_a_launch_home_name_that_moves_forwards_no_dashboard_plugin)."""
    from hermes_cli import web_server_dashboard
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    with _launch_dashboard_profiles(tmp_path, monkeypatch, "ops", plugin_in_root=True) as (_, _, profiles):
        shared = (profiles.parent / "plugins" / "dash" / "dashboard").resolve()
        assert Path(web_server_dashboard._hosted_plugin_for_request("dash")[1]).resolve() == shared
        (profiles / "ops").rename(profiles / "ops2")
        token = set_hermes_home_override(profiles / "ops2")
        try:
            assert Path(web_server_dashboard._hosted_plugin_for_request("dash")[1]).resolve() == shared
        finally:
            reset_hermes_home_override(token)


@pytest.mark.platforms("posix")  # directory symlinks
def test_a_sibling_name_the_process_served_elsewhere_is_not_taken_over_when_repointed_here(tmp_path, monkeypatch):
    """A renamed profile's manager takes the name beside its old one that leads to its directory, but not a name the
    process already serves another directory under: that name keys the other directory, so plugin code bound to it
    would register there. With no other name, the manager refuses."""
    from hermes_cli.plugins_loader import PluginHomeMoved
    from hermes_constants import hermes_home_key

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {}, isolation="in_process")
    elsewhere = _home_with_plugins(tmp_path / "elsewhere", monkeypatch, {}, isolation="in_process")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "ops").symlink_to(home, target_is_directory=True)
    (profiles / "ops2").symlink_to(elsewhere, target_is_directory=True)
    hermes_home_key(profiles / "ops2")  # the process serves elsewhere under ops2
    monkeypatch.setenv("HERMES_HOME", str(profiles / "ops"))
    plugins_mod._reset_plugin_managers_for_tests()
    try:
        manager = plugins_mod.get_plugin_manager()
        (profiles / "ops").unlink()
        (profiles / "ops2").unlink()
        (profiles / "ops2").symlink_to(home, target_is_directory=True)
        with pytest.raises(PluginHomeMoved):
            manager.owner_home
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
def test_a_host_start_whose_link_is_repointed_while_it_builds_the_childs_env_starts_no_host(tmp_path, monkeypatch):
    """The child's env is the start's last read through the profile's name: a link repointed during it refuses the
    start, and no host process runs on the other directory's files (Greptile on #159450)."""
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugins_loader import PluginHomeMoved

    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {})
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    plugins_mod._reset_plugin_managers_for_tests()
    build_env = PluginHost._env

    def repointed_while_building(self, *args):
        env = build_env(self, *args)
        link.unlink()
        link.symlink_to(tmp_path, target_is_directory=True)
        return env

    monkeypatch.setattr(PluginHost, "_env", repointed_while_building)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        with pytest.raises(PluginHomeMoved):
            host.ensure_started()
        assert host.pid is None
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script
def test_a_provider_extraction_for_a_home_whose_link_was_repointed_runs_no_launcher(tmp_path, monkeypatch):
    """A model-provider extraction outside a host start, for a home whose name the process remembered and that now
    leads to another directory, is refused (Greptile on #159450): the launcher that directory's config names never
    runs with the remembered profile's values."""
    from hermes_cli.plugin_host_profiles import load_hosted_profiles
    from hermes_cli.plugin_host_wire import PluginHostUnavailable
    from hermes_constants import hermes_home_key

    old = _home_with_plugins(tmp_path / "old", monkeypatch, {})
    new = _home_with_plugins(tmp_path / "new", monkeypatch, {})
    ran = _hosted_provider_behind_a_launcher(tmp_path, new)
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(old, target_is_directory=True)
    hermes_home_key(link)  # the process served the old directory under this name
    link.unlink()
    link.symlink_to(new, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    with pytest.raises(PluginHostUnavailable, match="restart Hermes"):
        load_hosted_profiles(link / "plugins" / "model-providers" / "hostmodel", "hostmodel")
    assert not ran.exists()


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script
def test_a_provider_extraction_whose_link_is_repointed_while_it_picks_its_launcher_runs_none(tmp_path, monkeypatch):
    """The launcher is the extraction's last read through the home's name: a link repointed while it is picked
    refuses the extraction, and that launcher never runs on the other directory's files."""
    from hermes_cli import plugin_isolation
    from hermes_cli.plugin_host_profiles import load_hosted_profiles
    from hermes_cli.plugin_host_wire import PluginHostUnavailable
    from hermes_constants import hermes_home_key

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {})
    ran = _hosted_provider_behind_a_launcher(tmp_path, home)
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(home, target_is_directory=True)
    hermes_home_key(link)
    monkeypatch.setenv("HERMES_HOME", str(link))
    pick = plugin_isolation.extraction_launcher

    def repointed_while_picking(at):
        launcher = pick(at)
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        return launcher

    monkeypatch.setattr(plugin_isolation, "extraction_launcher", repointed_while_picking)
    with pytest.raises(PluginHostUnavailable, match="restart Hermes"):
        load_hosted_profiles(link / "plugins" / "model-providers" / "hostmodel", "hostmodel")
    assert not ran.exists()


@pytest.mark.platforms("posix")  # a directory symlink
@pytest.mark.parametrize("moved", ["after_the_launcher", "after_the_launcher_cold", "while_the_env_builds"])
def test_a_host_start_whose_link_is_repointed_before_it_reads_ambient_credentials_imports_no_provider(
        moved, tmp_path, monkeypatch):
    """A start reads which providers take ambient credentials by scanning the home's provider plugins, and an
    in-process home imports them into Hermes; so does the first import of the child-env builder, whose blocklist
    reads the provider registry (the cold case). A link repointed before either, or while the child's env is
    built, refuses the start, and the other directory's provider code never runs."""
    import sys

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugin_isolation import ISOLATION_IN_PROCESS
    from hermes_cli.plugins_loader import PluginHomeMoved

    if moved == "after_the_launcher_cold":  # the start is the process's first importer
        for module in ("tools.environments.local", "tools.environments.local_env_policy"):
            monkeypatch.delitem(sys.modules, module, raising=False)
        PROVIDER_REGISTRY.sync_before_first_read()

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {}, isolation=ISOLATION_IN_PROCESS)
    imported = tmp_path / "provider.imported"
    provider = other / "plugins" / "model-providers" / "otherprov"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text(f"from pathlib import Path\nPath({str(imported)!r}).write_text('ran')\n",
                                          encoding="utf-8")
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()
    launcher = PluginHost._launcher

    def repoint():
        link.unlink()
        link.symlink_to(other, target_is_directory=True)

    if moved == "while_the_env_builds":
        import providers
        from tools.environments import local
        # The env build outlasts the provider layer's stamp check (1 s): the scan after it re-reads the plugin dirs.
        monkeypatch.setattr(providers, "_PLUGIN_DIR_STAMP_TTL_SECONDS", 0.0)
        build = local.served_profile_child_env
        monkeypatch.setattr(local, "served_profile_child_env", lambda *a, **k: (build(*a, **k), repoint())[0])
    else:
        monkeypatch.setattr(PluginHost, "_launcher", lambda self, own: (launcher(self, own), repoint())[0])
    try:
        with pytest.raises(PluginHomeMoved):
            host.ensure_started()
        assert not imported.exists()
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # directory symlinks
def test_a_host_start_from_a_caller_bound_to_a_moved_name_reads_providers_through_the_hosts_own_home(
        tmp_path, monkeypatch):
    """The first import of the child-env builder reads the provider registry, which imports an in-process home's
    provider plugins. A start that is that first import, run from a caller bound to another profile whose link
    has since been repointed, reads the registry for the host's own checked home, not through the caller's
    moved name."""
    import sys

    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugin_isolation import ISOLATION_IN_PROCESS
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

    for module in ("tools.environments.local", "tools.environments.local_env_policy"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    PROVIDER_REGISTRY.sync_before_first_read()
    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    caller = _home_with_plugins(tmp_path / "caller", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {}, isolation=ISOLATION_IN_PROCESS)
    imported = tmp_path / "provider.imported"
    provider = other / "plugins" / "model-providers" / "otherprov"
    provider.mkdir(parents=True)
    (provider / "__init__.py").write_text(f"from pathlib import Path\nPath({str(imported)!r}).write_text('ran')\n",
                                          encoding="utf-8")
    link = tmp_path / "root" / "profiles" / "ops"
    link.parent.mkdir(parents=True)
    link.symlink_to(caller, target_is_directory=True)
    hermes_home_key(link)  # the process opened ops as the caller's directory
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    plugins_mod._reset_plugin_managers_for_tests()
    host = plugins_mod.get_plugin_manager()._plugin_host()

    class EnvBuilt(Exception):
        pass

    env = PluginHost._env
    monkeypatch.setattr(PluginHost, "_env", lambda self, *a: (env(self, *a), (_ for _ in ()).throw(EnvBuilt()))[0])
    token = set_hermes_home_override(link)
    try:
        with pytest.raises(EnvBuilt):  # the start got past its env build, its own home unmoved
            host.ensure_started()
        assert not imported.exists()
    finally:
        reset_hermes_home_override(token)
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # directory symlinks
@pytest.mark.parametrize("failure", ["refused", "launcher_unreadable", "built_with_the_host"])
def test_a_host_context_built_while_its_link_led_elsewhere_reads_the_profiles_own_secrets_once_it_leads_back(
        failure, tmp_path, monkeypatch):
    """A host context built through the profile's link while it led to another directory holds that directory's
    secrets, and is not kept: when the start then refuses, when it fails before its last check (that directory's
    launcher does not parse), and when the context was built with the host. Once the link leads back, the next
    start's context holds the profile's own secrets."""
    from agent import secret_scope as ss
    from hermes_cli.plugin_host import PluginHost
    from hermes_cli.plugins_loader import PluginHomeMoved

    home = _home_with_plugins(tmp_path / "home", monkeypatch, {})
    other = _home_with_plugins(tmp_path / "other", monkeypatch, {})
    (home / ".env").write_text("MOVED_PROBE=own\n", encoding="utf-8")
    (other / ".env").write_text("MOVED_PROBE=other\n", encoding="utf-8")
    if failure == "launcher_unreadable":
        (other / "config.yaml").write_text(yaml.safe_dump({"plugins": {
            "isolation": "host", "host": {"launcher": "runner 'unclosed"}}}), encoding="utf-8")
    profiles = tmp_path / "root" / "profiles"
    profiles.mkdir(parents=True)
    link = profiles / "ops"
    link.symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    ss.set_multiplex_active(True)
    plugins_mod._reset_plugin_managers_for_tests()
    if failure != "built_with_the_host":
        host = plugins_mod.get_plugin_manager()._plugin_host()
        link = link.rename(profiles / "ops2")  # the next start rebuilds the context for the new name
    build = PluginHost._build_base_context

    def repointed_while_building(self):
        link.unlink()
        link.symlink_to(other, target_is_directory=True)
        return build(self)

    monkeypatch.setattr(PluginHost, "_build_base_context", repointed_while_building)
    if failure == "built_with_the_host":
        host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        with pytest.raises(ValueError if failure == "launcher_unreadable" else PluginHomeMoved):
            host.ensure_started()
        monkeypatch.setattr(PluginHost, "_build_base_context", build)
        link.unlink()
        link.symlink_to(home, target_is_directory=True)
        host.ensure_started()
        assert host._base_context.run(ss.get_secret, "MOVED_PROBE") == "own"
    finally:
        host.shutdown()
        plugins_mod._reset_plugin_managers_for_tests()


def test_a_relative_home_looked_up_from_another_working_directory_gets_the_directory_it_names_there(
        tmp_path, monkeypatch):
    """A symbolic profile id bound as a home is a relative path. A lookup from another working directory gets the
    manager of the directory the name means there, not the one opened from the first."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    first = _home_with_plugins(tmp_path / "first", monkeypatch, {"one": "def register(ctx):\n    pass\n"},
                               isolation="in_process")
    second = _home_with_plugins(tmp_path / "second", monkeypatch, {"two": "def register(ctx):\n    pass\n"},
                                isolation="in_process")
    plugins_mod._reset_plugin_managers_for_tests()

    def manager_from(cwd):
        monkeypatch.chdir(cwd)
        token = set_hermes_home_override(".hermes")
        try:
            manager = plugins_mod.get_plugin_manager()
            manager.discover_and_load()
            return manager
        finally:
            reset_hermes_home_override(token)

    try:
        assert "one" in manager_from(first.parent)._plugins
        assert "two" in manager_from(second.parent)._plugins
    finally:
        plugins_mod._reset_plugin_managers_for_tests()


@pytest.mark.platforms("posix")  # a directory symlink
@pytest.mark.parametrize("link", ["kept", "removed"])
def test_unloading_runs_under_the_profile_name_or_under_the_directory_once_no_name_leads_there(
        link, tmp_path, monkeypatch):
    """Unloading stops plugin work, so it still runs once no profile name leads to the manager's directory: a
    plugin's unload callback runs under the profile's name while one does, and under the directory after."""
    out = tmp_path / "home.out"
    monkeypatch.setenv("UNLOADPROBE_OUT", str(out))
    target = _home_with_plugins(tmp_path / "outside", monkeypatch, {"unloadprobe": UNLOAD_PROBE},
                                isolation="in_process")
    ops = tmp_path / "root" / "profiles" / "ops"
    ops.parent.mkdir(parents=True)
    ops.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(ops))
    plugins_mod._reset_plugin_managers_for_tests()
    manager = plugins_mod.get_plugin_manager()
    manager.discover_and_load()
    try:
        if link == "removed":
            ops.unlink()
        manager.unload()
        assert out.read_text(encoding="utf-8") == str(ops if link == "kept" else manager.home_path)
    finally:
        plugins_mod._reset_plugin_managers_for_tests()
