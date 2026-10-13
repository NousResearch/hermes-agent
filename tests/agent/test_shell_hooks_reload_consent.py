"""Force-reload consent regression adapted from the independent V4 review probe."""
import shlex
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from agent import shell_hooks
from agent.secret_scope import is_multiplex_active, set_multiplex_active, set_secret_scope, reset_secret_scope
from hermes_cli import plugins
from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from tools.terminal_scope import set_terminal_scope, reset_terminal_scope


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'launch'))
    monkeypatch.setenv('HOME', str(tmp_path / 'os-home'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'os-home')
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: False)
    monkeypatch.delenv('HERMES_ACCEPT_HOOKS', raising=False)
    active = is_multiplex_active()
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    yield
    plugins._reset_plugin_managers_for_tests()
    shell_hooks.reset_for_tests()
    set_multiplex_active(active)


@contextmanager
def home_scope(home):
    home.mkdir(parents=True, exist_ok=True)
    ht = set_hermes_home_override(home)
    st = set_secret_scope({})
    tt = set_terminal_scope({})
    try:
        yield
    finally:
        reset_terminal_scope(tt)
        reset_secret_scope(st)
        reset_hermes_home_override(ht)


def entry(tmp_path, name):
    script = tmp_path / f'{name}.py'
    script.write_text('import json, sys, os\np=json.load(sys.stdin)\n'
                      f'print(json.dumps({{"decision":"block", "reason":{name!r}}}))\n')
    return {'command': shlex.join([sys.executable, str(script)]), 'matcher': 'write_file', 'fail_closed': True}


def config(*entries):
    return {'hooks': {'pre_tool_call': list(entries)}}


def block():
    return plugins.get_pre_tool_call_block_message('write_file', {}, tool_call_id='review-current')


@pytest.mark.parametrize('new_consent', ['declined', 'approved', 'error'])
def test_force_reload_preserves_approved_policy_with_unapproved_addition(tmp_path, monkeypatch, new_consent):
    """Real force reload; old approved gate must not disappear because another command is denied."""
    a = entry(tmp_path, 'approved-old')
    b = entry(tmp_path, 'additional-new')
    set_multiplex_active(True)
    other = tmp_path / 'beta'
    with home_scope(other):
        shell_hooks.register_from_config(config(entry(tmp_path, 'beta-policy')), accept_hooks=True)
        other_manager = plugins.get_plugin_manager()
        other_callbacks = other_manager._hooks['pre_tool_call']
        assert block() == 'beta-policy'
    marker = tmp_path / 'new-command-ran'
    script = tmp_path / 'additional-new.py'
    script.write_text(script.read_text() + f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
    home = tmp_path / 'alpha'
    with home_scope(home):
        shell_hooks.register_from_config(config(a), accept_hooks=True)
        manager = plugins.get_plugin_manager()
        assert block() == 'approved-old'
        if new_consent == 'approved':
            shell_hooks._record_approval('pre_tool_call', b['command'])
        (home / 'config.yaml').write_text(yaml.safe_dump(config(a, b)))
        if new_consent == 'error':
            def consent_error(*args, **kwargs):
                raise OSError('consent store unavailable')
            monkeypatch.setattr(shell_hooks, '_prompt_and_record', consent_error)
        # Real loader/config/dispatch; only the explicit consent I/O error is injected.
        manager.discover_and_load(force=True)
        actual = block()
        assert actual == 'approved-old', (
            'force reload disarmed the approved policy', actual,
            [c.__name__ for c in manager._hooks.get('pre_tool_call', [])])
        all_results = plugins.invoke_hook('pre_tool_call', tool_name='write_file', args={}, tool_call_id='review-all')
        assert [r['message'] for r in all_results] == (
            ['approved-old', 'additional-new'] if new_consent == 'approved' else ['approved-old'])
        assert marker.exists() == (new_consent == 'approved')
        assert other_manager._hooks['pre_tool_call'] is other_callbacks
    with home_scope(other):
        assert block() == 'beta-policy'
