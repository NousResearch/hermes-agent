"""Migration UX policy; selected-child and real profile writes live in completion tests."""
import sys

import pytest

from hermes_cli import config, update_cmd, update_cmd_config


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
        return (4 if case == 'ahead' else 3 if case == 'current' or calls else 2, 3)

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
    monkeypatch.setattr(update_cmd, '_migrate_sibling_profile_configs', lambda: [])
    monkeypatch.setattr('hermes_cli.profiles.get_active_profile_name', lambda: 'default')
    monkeypatch.setattr(update_cmd_config, '_validate_profile_configs', lambda: [('default', [])])
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr('builtins.input', prompt)
    gateway_prompts = []
    monkeypatch.setattr(update_cmd, '_gateway_prompt', lambda text, default: gateway_prompts.append((text, default)) or 'y')
    update_cmd._check_and_apply_config_migration(assume_yes=case == 'yes', gateway_mode=case == 'gateway')
    assert calls == expected
    assert bool(prompts) is (case in {'tty-yes', 'tty-decline', 'eof', 'unicode'})
    assert bool(gateway_prompts) is (case == 'gateway')
    output = capsys.readouterr().out
    if case != 'read-error':
        assert '→ Validating configuration...' in output
        assert 'default: valid' in output
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
        assert 'default: migrated v2 → v3' in output
        assert 'no new settings' not in output.lower()
        if case == 'warnings':
            assert 'setting reset' in output and 'personality reset' in output


@pytest.mark.parametrize('named', [False, True])
def test_update_copies_bundled_skill_bytes_to_default_active_and_sibling(tmp_path, monkeypatch, named):
    from pathlib import Path
    from hermes_cli import update_cmd_maint

    home = tmp_path / '.hermes'
    homes = [home, home / 'profiles/active', home / 'profiles/sibling'] if named else [home]
    for profile in homes:
        profile.mkdir(parents=True, exist_ok=True)
        (profile / 'config.yaml').write_text('{}\n', encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(homes[1] if named else home))
    monkeypatch.setattr('plugins.memory.honcho.cli.sync_honcho_profiles_quiet', lambda: [])
    update_cmd_maint._sync_profiles_after_update()
    bundled = Path(__file__).resolve().parents[2] / 'skills'
    witness = next(bundled.rglob('SKILL.md'))
    for profile in homes:
        assert (profile / 'skills' / witness.relative_to(bundled)).read_bytes() == witness.read_bytes()


def test_validation_status_is_explicit_per_profile(capsys):
    """Migration and validation are separate, positively stated statuses."""

    class Issue:
        def __init__(self, severity):
            self.severity = severity

    update_cmd_config._print_config_validation_status([
            ("default", []),
            ("coder", [Issue("warning")]),
            ("author", [Issue("error"), Issue("warning")]),
    ])
    out = capsys.readouterr().out

    assert "→ Validating configuration..." in out
    assert "default: valid" in out
    assert "coder: valid with 1 warning(s)" in out
    assert "author: invalid (1 error(s), 1 warning(s))" in out
