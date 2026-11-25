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
