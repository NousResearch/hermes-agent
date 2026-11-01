"""Classic recap line counts and CLI/TUI command persistence (PR #61537)."""

from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
from hermes_cli.cli_commands_mixin import CLICommandsMixin


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    import cli
    from tui_gateway import server

    monkeypatch.setattr(cli, '_hermes_home', tmp_path)
    monkeypatch.setattr(server, '_hermes_home', tmp_path)
    for key in ['_cfg_cache', '_cfg_sig', '_cfg_path']:
        monkeypatch.setattr(server, key, None)
    return tmp_path


def _carrier():
    return SimpleNamespace(reasoning_config=None, show_reasoning=True, reasoning_full=False,
                           reasoning_clamp_lines=10, agent=None)


def _command(surface, carrier, value):
    if surface == 'cli':
        CLICommandsMixin._handle_reasoning_command(carrier, '/reasoning ' + value)
    else:
        from tui_gateway import server
        result = server._methods['config.set']('test', {'key': 'reasoning', 'value': value})
        assert 'error' not in result, result


@pytest.mark.parametrize('surface', ['cli', 'gateway'])
def test_custom_clamp_full_and_default_clamp_persist_consistently(home, surface):
    carrier = _carrier()
    for command, expected_count, expected_full in [('clamp 3', 3, False), ('full', 3, True), ('clamp', 10, False)]:
        _command(surface, carrier, command)
        saved = yaml.safe_load((home / 'config.yaml').read_text())['display']
        assert saved['reasoning_clamp_lines'] == expected_count
        assert saved['reasoning_full'] is expected_full
        if surface == 'cli':
            assert carrier.reasoning_clamp_lines == expected_count
            assert carrier.reasoning_full is expected_full


@pytest.mark.parametrize('surface', ['cli', 'gateway'])
@pytest.mark.parametrize('value', ['clamp 0', 'clamp -2', 'clamp bad', 'clamp 3 extra'])
def test_invalid_clamp_does_not_change_persisted_settings(home, surface, value):
    path = home / 'config.yaml'
    original = 'display:\n  reasoning_full: true\n  reasoning_clamp_lines: 7\n'
    path.write_text(original)
    carrier = _carrier()
    carrier.reasoning_full = True
    carrier.reasoning_clamp_lines = 7
    if surface == 'cli':
        CLICommandsMixin._handle_reasoning_command(carrier, '/reasoning ' + value)
        assert carrier.reasoning_full is True
        assert carrier.reasoning_clamp_lines == 7
    else:
        from tui_gateway import server
        result = server._methods['config.set']('test', {'key': 'reasoning', 'value': value})
        assert result['error']['code'] == 4002
    assert path.read_text() == original


@pytest.mark.parametrize('modern', [False, True])
@pytest.mark.parametrize('value,expected', [(3, 3), ('4', 4), ('bad', 10), (3.7, 10), (3.0, 10), (0, 10), (None, 10), (False, 10)])
def test_recap_obeys_loaded_config_and_safely_defaults_malformed_values(home, monkeypatch, modern, value, expected):
    import cli
    from hermes_cli.cli_config_load import load_cli_config

    (home / 'config.yaml').write_text(yaml.safe_dump({'display': {'reasoning_clamp_lines': value}}))
    loaded = load_cli_config()
    output = []
    monkeypatch.setattr(cli, '_cprint', lambda *args, **kw: output.append(' '.join(map(str, args))))
    carrier = _carrier()
    carrier.reasoning_clamp_lines = loaded['display']['reasoning_clamp_lines']
    carrier._modern_layout = modern
    carrier._reasoning_shown_this_turn = False
    carrier._scrollback_box_width = lambda: 80
    lines = [f'reasoning-line-{index:02}' for index in range(30)]
    turn = SimpleNamespace(result={'last_reasoning': '\n'.join(lines)})

    CLIChatTurnMixin._chat_print_reasoning_box(carrier, turn)

    rendered = '\n'.join(output)
    assert [line for line in lines if line in rendered] == lines[:expected]
    carrier.reasoning_full = True
    output.clear()
    CLIChatTurnMixin._chat_print_reasoning_box(carrier, turn)
    assert all(line in '\n'.join(output) for line in lines)
