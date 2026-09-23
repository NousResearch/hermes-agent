"""Launch metadata reads cannot create or migrate canonical storage."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("populated", [False, True])
def test_resume_and_exit_read_without_writing(tmp_path, populated):
    repo = Path(__file__).resolve().parents[2]
    home = tmp_path / "profile"
    home.mkdir()
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), PYTHONPATH=str(repo))
    code = '''
from pathlib import Path
from hermes_state import SessionDB
from hermes_cli.main import _resolve_last_session, _resolve_session_by_name_or_id
from hermes_cli.main_tui_launch import _print_tui_exit_summary
import os
path = Path(os.environ['HERMES_HOME']) / 'state.db'
populated = POPULATED
if populated:
    with SessionDB(db_path=path) as db:
        db.create_session('reader-fixture', source='tui')
        db.append_message('reader-fixture', role='user', content='hello')
        db.set_session_title('reader-fixture', 'Reader fixture')
before = path.read_bytes() if path.exists() else None
from hermes_cli.terminal_breadcrumbs import write_breadcrumb, resolve_breadcrumb_session
os.environ['TMUX_PANE'] = 'reader-fixture'
write_breadcrumb('reader-fixture')
assert resolve_breadcrumb_session() == ('reader-fixture' if populated else None)
assert _resolve_last_session(source='tui') == ('reader-fixture' if populated else None)
assert _resolve_session_by_name_or_id('reader-fixture') == ('reader-fixture' if populated else None)
_print_tui_exit_summary('reader-fixture')
from hermes_cli.main_agent_cmds import cmd_insights
from types import SimpleNamespace
cmd_insights(SimpleNamespace(days=30, source='tui'))
from hermes_cli.cli_info_mixin import CLIInfoMixin
CLIInfoMixin._show_insights(object(), '/insights --days 30 --source tui')
after = path.read_bytes() if path.exists() else None
assert after == before, 'launch reader created or mutated state.db'
'''.replace('POPULATED', repr(populated))
    result = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Error generating insights:' not in result.stdout
    if populated:
        assert 'Hermes Insights' in result.stdout
        assert 'Resume this session with:' in result.stdout
        assert 'Reader fixture' in result.stdout
    else:
        assert 'No sessions found in the last 30 days (source: tui).' in result.stdout
