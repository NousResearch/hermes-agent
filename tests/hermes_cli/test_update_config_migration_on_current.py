"""Migration UX policy; selected-child and real profile writes live in completion tests."""
import sys

import pytest

from hermes_agent.hermes_cli import config, update_cmd


@pytest.mark.parametrize('case,expected', [
    ('format', [(False, True)]), ('warnings', [(False, True)]),
    ('migration-error', [(False, True)]), ('current', []), ('ahead', []), ('read-error', []),
    ('yes', [(False, False)]), ('tty-yes', [(True, False)]), ('tty-decline', []),
    ('noninteractive', [(False, False)]), ('gateway', [(False, False)]), ('eof', []), ('unicode', []),
])
def test_migration_policy(monkeypatch, capsys, case, expected):
    prompts, calls = [], []
    named = case in {'yes', 'tty-yes', 'tty-decline', 'noninteractive', 'gateway', 'eof', 'unicode'}
    monkeypatch.setattr(config, 'get_missing_env_vars', lambda **_: [
        {'name': 'NEW_TOKEN', 'description': 'new credential'}] if named else [])
    monkeypatch.setattr(config, 'get_missing_config_fields', lambda: [
        {'key': 'new.option', 'description': 'new option'}] if named else [])

    def version(**kwargs):
        assert kwargs == {'raise_on_parse_error': True}
        if case == 'read-error':
            raise RuntimeError('cannot read config')
        return (4 if case == 'ahead' else 3 if case == 'current' else 2, 3)

    def migrate(*, interactive, quiet):
        calls.append((interactive, quiet))
        if case == 'migration-error':
            raise RuntimeError('cannot write config')
        return {'env_added': [], 'config_added': ['setting reset'] if case == 'warnings' else [],
                'warnings': ['personality reset'] if case == 'warnings' else []}

    def prompt(text):
        prompts.append(text)
        if case == 'eof':
            raise EOFError()
        if case == 'unicode':
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid')
        return 'n' if case == 'tty-decline' else 'y'

    monkeypatch.setattr(config, 'check_config_version', version)
    monkeypatch.setattr(config, 'migrate_config', migrate)
    monkeypatch.setattr(update_cmd, '_migrate_sibling_profile_configs', lambda *_: [])
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr('builtins.input', prompt)
    gateway_prompts = []
    monkeypatch.setattr(update_cmd, '_gateway_prompt', lambda text, default: gateway_prompts.append((text, default)) or 'y')
    if case == 'migration-error':
        # A failed write is owed, not silently completed: the owed-step guard needs the raise.
        with pytest.raises(RuntimeError, match=r'v2 → v3 was not written: cannot write config'):
            update_cmd._check_and_apply_config_migration(assume_yes=False, gateway_mode=False)
    else:
        update_cmd._check_and_apply_config_migration(assume_yes=case == 'yes', gateway_mode=case == 'gateway')
    assert calls == expected
    assert bool(prompts) is (case in {'tty-yes', 'tty-decline', 'eof', 'unicode'})
    assert bool(gateway_prompts) is (case == 'gateway')
    output = capsys.readouterr().out
    if named:
        assert 'NEW_TOKEN' in output and 'new.option' in output
        assert 'new credential' in output and 'new option' in output
        if case in {'yes', 'noninteractive', 'gateway'}:
            assert 'API keys require manual entry' in output
        elif not expected:
            assert 'hermes config migrate' in output
    elif case in {'current', 'ahead'}:
        assert 'Configuration is up to date' in output
    elif case == 'read-error':
        assert 'Could not check config version' in output
    elif case == 'migration-error':
        assert 'Config format update failed: cannot write config' in output
    else:
        assert 'v2 → v3' in output and 'no new settings to configure' in output
        if case == 'warnings':
            assert 'setting reset' in output and 'personality reset' in output


@pytest.mark.parametrize('named', [False, True])
def test_update_copies_bundled_skill_bytes_to_default_active_and_sibling(tmp_path, monkeypatch, named):
    from pathlib import Path
    from hermes_agent.hermes_cli import update_cmd_maint

    home = tmp_path / '.hermes'
    homes = [home, home / 'profiles/active', home / 'profiles/sibling'] if named else [home]
    for profile in homes:
        profile.mkdir(parents=True, exist_ok=True)
        (profile / 'config.yaml').write_text('{}\n', encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(homes[1] if named else home))
    update_cmd_maint._sync_profiles_after_update()
    bundled = Path(__file__).resolve().parents[2] / 'skills'
    witness = next(bundled.rglob('SKILL.md'))
    for profile in homes:
        assert (profile / 'skills' / witness.relative_to(bundled)).read_bytes() == witness.read_bytes()


def test_active_config_write_failure_still_migrates_siblings_and_restores_snapshots(tmp_path, monkeypatch):
    """A failed ACTIVE config write is owed debt, but the independent sibling migration and the
    pre-update snapshot safety nets (the only holders of this run's snapshot ids) still run."""
    import json
    from pathlib import Path

    from hermes_agent.hermes_cli import backup, update_cmd_config

    root = tmp_path / '.hermes'
    sibling = root / 'profiles' / 'sibling'
    jobs = {'jobs': [{'id': 'nightly', 'name': 'nightly', 'schedule': '0 3 * * *', 'prompt': 'report'}]}
    for home in (root, sibling):
        (home / 'cron').mkdir(parents=True)
        (home / 'config.yaml').write_text('_config_version: 48\n', encoding='utf-8')
        (home / 'cron' / 'jobs.json').write_text(json.dumps(jobs), encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    active_snapshot = backup.create_quick_snapshot(label='pre-update', hermes_home=root)
    monkeypatch.setattr(update_cmd_config, '_LAST_SIBLING_SNAPSHOTS',
                        backup.create_pre_update_snapshots_all_profiles(root))
    for home in (root, sibling):  # the documented mutation-window loss the safety nets repair
        (home / 'cron' / 'jobs.json').write_text('{"jobs": []}', encoding='utf-8')
    real_replace = config.atomic_config_replace

    def replace(path, *args, **kwargs):
        if Path(path) == root / 'config.yaml':
            raise OSError('disk full')
        return real_replace(path, *args, **kwargs)

    monkeypatch.setattr(config, 'atomic_config_replace', replace)
    with pytest.raises(RuntimeError, match='was not written: disk full'):
        update_cmd_config._check_and_apply_config_migration(pre_update_snapshot_id=active_snapshot)
    for home in (root, sibling):
        assert json.loads((home / 'cron' / 'jobs.json').read_text(encoding='utf-8')) == jobs
    assert config.check_config_version()[0] == 48  # the active write stays owed
    monkeypatch.setenv('HERMES_HOME', str(sibling))
    sibling_version, latest = config.check_config_version()
    assert sibling_version == latest > 48

def test_owed_reasons_from_the_active_write_and_a_sibling_are_all_kept(monkeypatch, capsys):
    """A failed active write and a sibling that advanced then stopped both reach the receipt:
    one RuntimeError names each reason, chained to the write error, and the sibling that only
    got part of the way is not reported with a success mark."""
    from hermes_agent.hermes_cli import update_cmd_config

    disk_full = OSError("disk full")

    def failing_write(**_):
        raise disk_full

    def siblings(notes=None, stopped=None):
        stopped["work"] = 21
        return [("work", 16, 20)]

    monkeypatch.setattr(config, "check_config_version", lambda **_: (20, 53))
    monkeypatch.setattr(config, "get_missing_env_vars", lambda **_: [])
    monkeypatch.setattr(config, "get_missing_config_fields", list)
    monkeypatch.setattr(config, "migrate_config", failing_write)
    monkeypatch.setattr(update_cmd, "_migrate_sibling_profile_configs", siblings)
    monkeypatch.setattr(update_cmd_config, "_restore_snapshot_safety_nets", lambda _snapshot: None)

    with pytest.raises(RuntimeError) as raised:
        update_cmd_config._check_and_apply_config_migration(pre_update_snapshot_id="snap")
    assert "format v20 → v53 was not written: disk full" in str(raised.value)
    assert "profile 'work' stopped at v21" in str(raised.value)
    assert raised.value.__cause__ is disk_full
    out = capsys.readouterr().out
    assert "Profile 'work': config migration stopped at v21 (v16 → v20 applied)" in out
    assert "✓ Profile 'work'" not in out


@pytest.mark.parametrize("new_options", [False, True], ids=("format-only", "configure-new-options"))
def test_partial_config_migration_stays_owed_after_update(monkeypatch, capsys, new_options):
    """A step failure returns failed_step rather than raising. Both update paths must retain a
    config_migration follow-up and must not say the configuration was fully updated.
    The post-commit caller catches the raise and records the owed step in the update receipt.
    """
    from hermes_agent.hermes_cli import update_cmd_config

    monkeypatch.setattr(config, "check_config_version", lambda **_: (20, 53))
    monkeypatch.setattr(config, "get_missing_env_vars", lambda **_: [])
    monkeypatch.setattr(config, "get_missing_config_fields",
                        lambda: [{"key": "display.new_setting"}] if new_options else [])
    migrate_calls = []

    def stopped_migration(**kwargs):
        migrate_calls.append(kwargs)
        return {
            "env_added": [], "config_added": ["v17: compression model moved"],
            "warnings": ["config migration to v21 failed: malformed plugins.disabled"],
            "failed_step": 21,
            "failed_warning": "config migration to v21 failed: malformed plugins.disabled",
        }

    monkeypatch.setattr(config, "migrate_config", stopped_migration)
    monkeypatch.setattr(update_cmd, "_migrate_sibling_profile_configs", lambda *_: [])
    snapshots = []
    monkeypatch.setattr(update_cmd_config, "_restore_snapshot_safety_nets",
                        lambda snapshot: snapshots.append(snapshot))

    with pytest.raises(RuntimeError, match="config migration stopped at v21"):
        update_cmd._check_and_apply_config_migration(
            assume_yes=True, pre_update_snapshot_id="pre-update-snapshot")

    assert len(migrate_calls) == 1
    assert snapshots == ["pre-update-snapshot"], "independent snapshot recovery must still run"
    output = capsys.readouterr().out
    assert "✓ Config format updated" not in output
    assert "✓ Configuration updated!" not in output


def test_a_sibling_migration_that_stopped_stays_owed_after_update(tmp_path, monkeypatch):
    """A sibling profile whose pass stopped at a failing step is owed like the active one: the
    sweep still finishes and the snapshot nets still run, then the update records a follow-up."""
    from pathlib import Path

    import hermes_agent.hermes_yaml as yaml
    from hermes_agent.hermes_cli import update_cmd_config

    root = tmp_path / '.hermes'
    sibling = root / 'profiles' / 'sibling'
    sibling.mkdir(parents=True)
    latest = config.DEFAULT_CONFIG['_config_version']
    (root / 'config.yaml').write_text(yaml.safe_dump({'_config_version': latest}), encoding='utf-8')
    (sibling / 'config.yaml').write_text(
        yaml.safe_dump({'_config_version': 20, 'plugins': {'disabled': [['x']]}}), encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.setattr(config, 'get_missing_env_vars', lambda **_: [])
    monkeypatch.setattr(config, 'get_missing_config_fields', list)
    snapshots = []
    monkeypatch.setattr(update_cmd_config, '_restore_snapshot_safety_nets', snapshots.append)

    with pytest.raises(RuntimeError, match="profile 'sibling' stopped at v21"):
        update_cmd_config._check_and_apply_config_migration(pre_update_snapshot_id='snap')
    assert snapshots == ['snap']
    monkeypatch.setenv('HERMES_HOME', str(sibling))
    assert config.check_config_version()[0] == 20, 'the steps before v21 are kept, v21 stays owed'
