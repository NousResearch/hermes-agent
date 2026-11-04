"""The TUI exit epilogue names a canonical session as compactly as main did (PR #106742 row 36).

Main printed ``hermes --tui --resume 20261012_185208_a67483`` (22 chars). A canonical local session is
keyed ``local-`` + 64 hex (70 chars); the epilogue prints its unique 22-char prefix, and ``--resume``
accepts that prefix back.
"""
import pytest


def _seed(home, sid, tokens=(300, 60)):
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(sid, source='tui')
    db.append_message(sid, 'user', 'q')
    db.append_message(sid, 'assistant', 'a')
    db.update_token_counts(sid, input_tokens=tokens[0], output_tokens=tokens[1])
    db.close()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    return home


def test_epilogue_prints_short_resume_id_for_canonical_session(home, capsys):
    from hermes_cli.main_tui_launch import _print_tui_exit_summary
    sid = 'local-' + 'ab12cd34ef56a7b8' + '0' * 48
    _seed(home, sid)
    _print_tui_exit_summary(sid)
    out = capsys.readouterr().out
    short = sid[:22]
    assert f'hermes --tui --resume {short}\n' in out
    assert sid not in out
    assert 'Tokens:         360 (in 300, out 60' in out


def test_short_id_resumes_the_session(home):
    from hermes_cli.main import _resolve_session_by_name_or_id
    sid = 'local-' + 'ab12cd34ef56a7b8' + '0' * 48
    _seed(home, sid)
    assert _resolve_session_by_name_or_id(sid[:22]) == sid


def test_ambiguous_prefix_falls_back_to_full_id(home, capsys):
    from hermes_cli.main import _resolve_session_by_name_or_id
    from hermes_cli.main_tui_launch import _print_tui_exit_summary
    a = 'local-' + 'ab12cd34ef56a7b8' + '0' * 48
    b = 'local-' + 'ab12cd34ef56a7b8' + '1' * 48
    _seed(home, a)
    _seed(home, b)
    _print_tui_exit_summary(a)
    assert f'hermes --tui --resume {a}\n' in capsys.readouterr().out
    assert _resolve_session_by_name_or_id(a[:22]) is None


def test_stored_ids_print_unchanged(home, capsys):
    from hermes_cli.main_tui_launch import _print_tui_exit_summary
    _seed(home, '20261012_185208_a67483')
    _print_tui_exit_summary('20261012_185208_a67483')
    assert 'hermes --tui --resume 20261012_185208_a67483\n' in capsys.readouterr().out


def test_branch_child_epilogue_counts_the_branched_conversation(home, capsys):
    """Main's epilogue after /branch reported the conversation's tokens (it still named the parent);
    the canonical child is named, and its total includes the branch it was cut from."""
    import json
    from hermes_state import SessionDB
    from hermes_cli.main_tui_launch import _print_tui_exit_summary
    parent = 'local-' + '1' * 64
    child = 'local-' + '2' * 64
    _seed(home, parent, tokens=(300, 60))
    db = SessionDB(db_path=home / 'state.db')
    db.create_session(child, source='tui', model_config={'_branched_from': parent}, parent_session_id=parent)
    db.append_message(child, 'user', 'q')
    db.append_message(child, 'assistant', 'a')
    db.close()
    _print_tui_exit_summary(child)
    out = capsys.readouterr().out
    assert f'--resume {child[:22]}' in out
    assert 'Tokens:         360 (in 300, out 60' in out
