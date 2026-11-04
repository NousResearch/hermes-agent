"""Public TUI capability discovery is independent of a running chat or gateway."""
import json
import sys

from hermes_cli import main
from hermes_cli.tui_capabilities import print_requested_capabilities, tui_capabilities


def test_probe_returns_only_versioned_feature_metadata(capsys):
    assert print_requested_capabilities(['--tui-capabilities'])
    result = json.loads(capsys.readouterr().out)
    assert result['schema'] == 1
    assert result['ui'] == 'hermes-tui'
    assert result['features']['session_tabs'] == 1
    assert result['features']['tab_persistence'] == 1
    assert set(result) == {'schema', 'ui', 'features'}
    first = tui_capabilities()
    first['features'].clear()
    assert tui_capabilities()['features']['session_tabs'] == 1


def test_other_commands_do_not_enter_the_standalone_probe(capsys):
    for args in ([], ['--help'], ['chat'], ['--', '--tui-capabilities']):
        assert not print_requested_capabilities(args)
    assert not capsys.readouterr().out


def test_real_entrypoint_does_not_start_recovery_plugins_or_chat(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('The capability probe entered normal startup')

    monkeypatch.setattr(sys, 'argv', ['hermes', '--tui-capabilities'])
    for name in ('_set_process_title', '_recover_update_debts_on_startup', '_prepare_agent_startup', 'cmd_chat'):
        monkeypatch.setattr(main, name, forbidden)
    main.main()
    assert json.loads(capsys.readouterr().out) == tui_capabilities()


def test_parser_exits_with_json_instead_of_launching_for_combined_flags(capsys):
    import pytest
    from hermes_cli._parser import build_top_level_parser

    parser = build_top_level_parser()[0]
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(['--tui', '--tui-capabilities'])
    assert stopped.value.code == 0
    assert json.loads(capsys.readouterr().out) == tui_capabilities()
