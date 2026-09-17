"""Send diagnostics must identify the active configuration home."""
import argparse
from pathlib import Path

import pytest

from hermes_cli.send_cmd import cmd_send, register_send_subparser


@pytest.mark.parametrize('surface', ['help', 'list', 'send'])
def test_send_diagnostics_identify_active_home(tmp_path, monkeypatch, capsys, surface):
    home = tmp_path / 'custom-home'
    # Pin the display relationship, independent of where the host puts temp files.
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'user-home')
    home.mkdir()
    (home / 'config.yaml').write_text('{}\n', encoding='utf-8')
    (home / '.env').write_text('', encoding='utf-8')
    monkeypatch.setenv('HERMES_HOME', str(home))
    parser = argparse.ArgumentParser(prog='hermes')
    register_send_subparser(parser.add_subparsers())
    argv = {'help': ['send', '--help'], 'list': ['send', '--list'],
            'send': ['send', '--to', 'discord', 'probe']}[surface]
    with pytest.raises(SystemExit) as stopped:
        args = parser.parse_args(argv)
        cmd_send(args)
    assert stopped.value.code == (1 if surface == 'send' else 0)
    output = capsys.readouterr()
    text = output.out + output.err
    assert ''.join(str(home).split()) in ''.join(text.split())
    assert '~/.hermes/' not in text
