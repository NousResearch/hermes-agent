"""``type: prompt`` quick commands on the gateway-owned local session path (``slash.exec`` /
``command.dispatch`` in ``gateway/session_commands.py``): the saved prompt becomes a ``send``
directive, never ``unsupported_command``."""
import sys
from types import SimpleNamespace

import pytest

from gateway import session_commands as sc

_CFG = {'quick_commands': {
    'rn': {'type': 'prompt', 'prompt': 'Notes for {args}', 'review_button': 'Release notes'},
    'rn-alias': {'type': 'alias', 'target': '/rn'},
    'empty': {'type': 'prompt', 'prompt': '  '},
}}


@pytest.fixture
def cfg(monkeypatch):
    # Only the config loader is read; a stub module keeps the heavy gateway.run import out.
    monkeypatch.setitem(sys.modules, 'gateway.run', SimpleNamespace(_load_gateway_config=lambda: _CFG))


def test_resolve_alias_admits_prompt_and_follows_an_alias_to_it(cfg):
    assert sc._resolve_alias('rn', 'v2') == ('rn', 'v2')
    assert sc._resolve_alias('rn-alias', 'v2') == ('rn', 'v2')


def test_prompt_quick_command_renders_args_and_flags_an_empty_prompt(cfg, monkeypatch):
    assert sc._prompt_quick_command('rn', 'v2.3') == 'Notes for v2.3'
    assert sc._prompt_quick_command('empty', '') == ''
    assert sc._prompt_quick_command('rn-alias', '') is None
    assert sc._prompt_quick_command('missing', '') is None
    # CLI and messaging precedence: a saved prompt beats a same-named plugin command.
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, 'get_plugin_command_handler', lambda name: (lambda _arg: 'plugin ran'))
    assert sc._runner_command('rn') is None
    assert sc._runner_command('other') == ('plugin', 'other')


def _runner(denied=None):
    return SimpleNamespace(_check_slash_access=lambda _source, _name: denied)


def test_unregistered_directive_sends_a_prompt_quick_command(cfg):
    out = sc._unregistered_directive(_runner(), SimpleNamespace(), 'rn', 'v2', 'route')
    assert out == {'type': 'send', 'message': 'Notes for v2', 'display': '/rn v2'}


def test_prompt_directive_obeys_slash_access_and_reports_an_empty_prompt(cfg):
    assert sc._unregistered_directive(_runner('admins only'), SimpleNamespace(), 'rn', '', 'route') == {
        'type': 'exec', 'output': 'admins only'}
    out = sc._unregistered_directive(_runner(), SimpleNamespace(), 'empty', '', 'route')
    assert out['type'] == 'exec' and 'no prompt' in out['output']
