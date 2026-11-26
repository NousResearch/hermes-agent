"""A plugin manager whose directory no profile name leads to any more refuses plugin work (Greptile review on
#159450): the directory alone reads as an unmarked profile, so a marked profile's run-elsewhere marker would go
unseen and its own launcher would run."""

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from tests.hermes_cli.test_plugin_host import _home_with_plugins

UNLOAD_PROBE = (
    "import os\nfrom pathlib import Path\nfrom hermes_constants import get_hermes_home\n"
    "def register(ctx):\n"
    "    ctx.on_unload(lambda: Path(os.environ['UNLOADPROBE_OUT']).write_text(str(get_hermes_home())))\n")


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


@pytest.mark.platforms("posix")  # a directory symlink; the stand-in launcher is a shell script
def test_a_provider_extraction_for_a_home_whose_link_was_repointed_runs_no_launcher(tmp_path, monkeypatch):
    """A model-provider extraction outside a host start, for a home whose name the process remembered and that now
    leads to another directory, is refused (Greptile on #159450): the launcher that directory's config names never
    runs with the remembered profile's values."""
    from hermes_cli.plugin_host_profiles import load_hosted_profiles
    from hermes_cli.plugin_host_wire import PluginHostUnavailable
    from hermes_constants import hermes_home_key
    from tests.hermes_cli.test_plugin_host import MODEL_PROVIDER_PLUGIN

    ran, runner = tmp_path / "launcher.ran", tmp_path / "runner"
    runner.write_text(f'#!/bin/sh\necho ran >> "{ran}"\nexec "$@"\n', encoding="utf-8")
    runner.chmod(0o755)
    old = _home_with_plugins(tmp_path / "old", monkeypatch, {})
    new = _home_with_plugins(tmp_path / "new", monkeypatch, {})
    plugin = new / "plugins" / "model-providers" / "hostmodel"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    (new / "config.yaml").write_text(yaml.safe_dump({"plugins": {
        "isolation": "host", "host": {"launcher": [str(runner)]}}}), encoding="utf-8")
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
