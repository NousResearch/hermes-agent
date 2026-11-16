"""Bare plugin commands are installer-owned, not plugin import side effects."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    assert importlib.util.find_spec('hermes_cli.plugin_shell_commands'), 'Missing installer support for bare plugin commands'
    return importlib.import_module('hermes_cli.plugin_shell_commands')


@pytest.fixture
def layout(tmp_path):
    home = tmp_path / 'profile'
    plugin = home / 'plugins' / 'example-warden'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: example-warden\nshell_commands: [warden]\n')
    (plugin / 'pyproject.toml').write_text('[project]\nname="example-warden"\nversion="1.0"\n[project.scripts]\nwarden="example.cli:main"\n')
    out = tmp_path / 'bin'
    out.mkdir(mode=0o700)
    return home, plugin, out


def test_builtin_dispatch_preserves_arguments_and_exit_code(monkeypatch):
    import argparse
    from hermes_cli.subcommands.plugins import build_plugins_parser
    from hermes_cli.plugins_cmd import plugins_command
    parser = argparse.ArgumentParser()
    build_plugins_parser(parser.add_subparsers(dest='command'), cmd_plugins=plugins_command)
    args = parser.parse_args(['plugins', 'run-shell', 'example-warden', 'warden', '--', 'setup', 'a&b'])
    monkeypatch.setattr(api(), 'run_command', lambda incoming: 7 if incoming.shell_arguments == ['setup', 'a&b'] else 9)
    assert plugins_command(args) == 7


def test_opt_in_commands_must_match_package_entrypoints(layout):
    _, plugin, _ = layout
    assert api().declared_commands(plugin) == {'warden': 'example.cli:main'}
    (plugin / 'plugin.yaml').write_text('name: example-warden\nshell_commands: [missing]\n')
    with pytest.raises(ValueError, match='project.scripts'):
        api().declared_commands(plugin)


@pytest.mark.parametrize('name', ['hermes', '../bad', 'bad/name', 'python', 'sh', 'UPPER', 'warden.exe'])
def test_rejects_unsafe_or_reserved_names(layout, name):
    _, plugin, _ = layout
    (plugin / 'plugin.yaml').write_text('name: example-warden\nshell_commands: [' + json.dumps(name) + ']\n')
    with pytest.raises(ValueError): api().declared_commands(plugin)


@pytest.mark.platforms('posix')
def test_private_primary_group_directory_is_accepted(layout, monkeypatch):
    import grp
    import pwd
    from types import SimpleNamespace
    home, plugin, out = layout
    out.chmod(0o770)
    me = pwd.getpwuid(os.getuid()).pw_name
    monkeypatch.setattr(grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=[me]))
    monkeypatch.setattr(pwd, 'getpwall', lambda: [SimpleNamespace(pw_name=me, pw_gid=out.stat().st_gid)])
    assert api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    api().remove_commands(plugin, home, out)
    monkeypatch.setattr(grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=[me, 'another-account']))
    with pytest.raises(OSError): api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))


def test_new_shell_command_requires_update_consent(layout):
    from hermes_cli.plugins_cmd_catalog import plugin_surface, surface_delta
    _, plugin, _ = layout
    old = plugin_surface({}, plugin)
    new = plugin_surface({'shell_commands': ['warden']}, plugin)
    assert surface_delta(old, new)['shell_commands'] == ['warden']


def test_no_export_without_explicit_manifest_opt_in(layout):
    _, plugin, _ = layout
    (plugin / 'plugin.yaml').write_text('name: example-warden\n')
    assert api().declared_commands(plugin) == {}


def test_foreign_command_is_never_overwritten(layout):
    home, plugin, out = layout
    target = out / ('warden.exe' if os.name == 'nt' else 'warden')
    target.write_bytes(b'foreign application')
    with pytest.raises(FileExistsError):
        api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    assert target.read_bytes() == b'foreign application'


@pytest.mark.platforms('posix')
def test_fresh_shell_command_forwards_arguments_and_binds_profile(layout):
    home, plugin, out = layout
    hermes = out / ('hermes.exe' if os.name == 'nt' else 'hermes')
    hermes.write_text('#!' + sys.executable + '\nimport os,sys,json\nprint(json.dumps([sys.argv[1:],os.environ.get("HERMES_HOME")]))\n')
    hermes.chmod(0o755)
    paths = api().export_commands(plugin, home, out, hermes)
    assert paths == [out / 'warden']
    env = dict(os.environ, PATH=str(out) + os.pathsep + os.environ.get('PATH', ''), HERMES_HOME='/wrong')
    args = ['setup', 'space value', 'a&b', ';not-a-shell-command', 'quote\"value']
    result = subprocess.run(['warden', *args], env=env, cwd=out.parent, text=True, capture_output=True, check=True)
    forwarded, selected_home = json.loads(result.stdout)
    assert forwarded == ['plugins', 'run-shell', 'example-warden', 'warden', '--', *args]
    assert selected_home == str(home)
    # Repeat install/update is idempotent; removal only removes owned files.
    before = paths[0].stat().st_mtime_ns
    assert api().export_commands(plugin, home, out, hermes) == paths
    assert paths[0].stat().st_mtime_ns == before, 'Unchanged launchers must not be replaced'
    api().remove_commands(plugin, home, out)
    assert not paths[0].exists()
    assert hermes.exists()


@pytest.mark.platforms('posix')
def test_symlink_destination_is_never_followed(layout):
    home, plugin, out = layout
    victim = out / 'victim'
    victim.write_text('keep')
    (out / 'warden').symlink_to(victim)
    with pytest.raises(FileExistsError): api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    assert victim.read_text() == 'keep'


@pytest.mark.platforms('posix')
def test_tampered_owned_command_is_not_overwritten_or_deleted(layout):
    home, plugin, out = layout
    target, = api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    target.write_text('user replacement')
    with pytest.raises(FileExistsError): api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    api().remove_commands(plugin, home, out)
    assert target.read_text() == 'user replacement'


@pytest.mark.platforms('windows')
def test_windows_native_launcher_preserves_argv_without_batch_interpretation(layout):
    home, plugin, out = layout
    hermes = api()._windows_launcher(
        out, 'hermes', 'import json,os,sys\nprint(json.dumps([sys.argv[1:],os.environ.get("HERMES_HOME")]))\n', Path(sys.executable))
    paths = api().export_commands(plugin, home, out, hermes)
    assert paths == [out / 'warden.exe']
    args = ['setup', 'space value', 'a&b', 'x|y', 'quote\"value', '%USERNAME%', '日本語']
    result = subprocess.run([str(paths[0]), *args], cwd=out.parent, text=True, capture_output=True, check=True)
    forwarded, selected_home = json.loads(result.stdout)
    assert forwarded == ['plugins', 'run-shell', plugin.name, 'warden', '--', *args]
    assert selected_home == str(home)
    before = paths[0].stat().st_mtime_ns
    api().export_commands(plugin, home, out, hermes)
    assert paths[0].stat().st_mtime_ns == before
    api().remove_commands(plugin, home, out)
    assert not paths[0].exists()


def test_core_refresh_does_not_publish_new_commands_without_consent(layout, monkeypatch):
    home, plugin, out = layout
    hermes = out / ('hermes.exe' if os.name == 'nt' else 'hermes')
    api().export_commands(plugin, home, out, hermes)
    with (plugin / 'pyproject.toml').open('a') as handle:
        handle.write('extra="example.cli:main"\n')
    (plugin / 'plugin.yaml').write_text('name: example-warden\nshell_commands: [warden, extra]\n')
    import pm.plugins_state
    monkeypatch.setattr(pm.plugins_state, 'read_home_selection', lambda home: {'plugins': {'enabled': [plugin.name]}})
    api().refresh_published_commands(out, hermes)
    assert not (out / ('extra.exe' if os.name == 'nt' else 'extra')).exists()
    assert (out / ('warden.exe' if os.name == 'nt' else 'warden')).exists()


def test_other_profile_cannot_take_over_command(layout, tmp_path):
    home, plugin, out = layout
    api().export_commands(plugin, home, out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
    with pytest.raises(FileExistsError):
        api().export_commands(plugin, tmp_path / 'other-profile', out, out / ('hermes.exe' if os.name == 'nt' else 'hermes'))
