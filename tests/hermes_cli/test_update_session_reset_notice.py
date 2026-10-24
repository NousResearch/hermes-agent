"""Upgrade-time visibility for the retired timed reset policy (related to #136012)."""
from pathlib import Path

import pytest

from hermes_cli import update_cmd_maint as maint


@pytest.fixture
def notice_profiles(tmp_path, monkeypatch):
    from hermes_cli import profiles

    root = tmp_path / 'hermes'
    root.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(profiles, '_get_default_hermes_home', lambda: root)
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: root / 'profiles')
    # Exercise the real post-update notice pipeline, without unrelated maintenance.
    for name in (
        '_print_fts_optimize_available_notice', '_print_curator_first_run_notice',
        '_print_curator_recent_run_notice', '_ensure_fhs_path_guard',
        '_refresh_cua_driver_after_update', '_install_default_tools_after_update',
        '_print_checkpoint_footprint_notice', '_migrate_relay_exporter_env',
    ):
        monkeypatch.setattr(maint, name, lambda: None)
        monkeypatch.setattr('hermes_cli.update_cmd.' + name, lambda: None, raising=False)
    monkeypatch.setattr('hermes_cli._launchers.expose_cli', lambda root: None)
    monkeypatch.setattr('hermes_cli._install_repair.migrate_windows_bin_path', lambda root: None)
    return root


def _configure(home, mode='both', *, plugin_enabled=False, nested=False):
    import hermes_yaml as yaml

    home.mkdir(parents=True, exist_ok=True)
    block = {'mode': mode, 'idle_minutes': 1440, 'at_hour': 4}
    config = {'gateway': {'session_reset': block}} if nested else {'session_reset': block}
    if plugin_enabled:
        config['plugins'] = {'enabled': ['hermes-session-reset-policy']}
        plugin = home / 'plugins' / 'hermes-session-reset-policy'
        plugin.mkdir(parents=True)
        (plugin / 'plugin.yaml').write_text(
            'name: hermes-session-reset-policy\nversion: 0.3.0\nkind: standalone\n', encoding='utf-8')
        (plugin / '__init__.py').write_text('def register(ctx):\n    pass\n', encoding='utf-8')
    (home / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')


@pytest.mark.parametrize('invoking', ['default', 'work'])
@pytest.mark.parametrize('enabled', [set(), {'default'}, {'work'}, {'default', 'work'}])
def test_update_notices_report_only_profiles_without_reset_plugin(
    notice_profiles, monkeypatch, capsys, invoking, enabled,
):
    from hermes_constants import get_hermes_home

    root = notice_profiles
    homes = {'default': root, 'work': root / 'profiles' / 'work'}
    for name, home in homes.items():
        _configure(home, plugin_enabled=name in enabled, nested=name == 'work')
    original_configs = {name: (home / 'config.yaml').read_text() for name, home in homes.items()}
    monkeypatch.setenv('HERMES_HOME', str(homes[invoking]))

    maint._print_post_update_notices_and_self_heals()

    output = capsys.readouterr().out
    for name in homes:
        if name in enabled:
            assert f"Session reset policy needs attention (profile '{name}')" not in output
        else:
            assert f"Session reset policy needs attention (profile '{name}')" in output
            assert f'hermes -p {name} plugins install hermes-session-reset-policy' in output
    if enabled != set(homes):
        assert 'idle_minutes: 1440' in output
        assert 'at_hour: 4' in output
        assert 'no longer applies' in output
        assert 'third-party' in output
    if 'work' not in enabled:
        assert 'top-level session_reset' in output
    assert get_hermes_home() == homes[invoking]
    for name, home in homes.items():
        assert (home / 'config.yaml').read_text() == original_configs[name]
        if name not in enabled:
            assert not (home / 'plugins').exists()


@pytest.mark.parametrize('mode', [None, 'none'])
def test_update_notices_are_silent_without_timed_policy(notice_profiles, capsys, mode):
    if mode is not None:
        _configure(notice_profiles, mode)
    maint._print_post_update_notices_and_self_heals()
    assert 'Session reset policy needs attention' not in capsys.readouterr().out
