"""Seed Inbox automation only in a disposable Electron test sandbox."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time


def main():
    home, key, title, goal_text = sys.argv[1:]
    home = Path(home).resolve()
    allowed = (home.name == 'hermes-home'
               and home.parent.name.startswith('hermes-e2e-')
               and home.parent.parent == Path(tempfile.gettempdir()).resolve())
    if not allowed:
        raise RuntimeError('Refusing to seed outside a disposable Inbox test sandbox')
    # Must precede all Hermes imports; goal storage resolves its own home.
    os.environ['HERMES_HOME'] = str(home)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from hermes_state import SessionDB
    from hermes_cli.goals import GoalState, save_goal
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session(session_id=key, source='cli', model='mock-model',
                          display_name=title, cwd=str(home))
        db.set_session_title(key, title)
        save_goal(key, GoalState(goal=goal_text, status='active', turns_used=2,
                                max_turns=10, created_at=time.time(), last_turn_at=time.time()))
    finally:
        db.close()
    # Verify raw sandbox storage, not another helper resolving the same wrong home.
    with sqlite3.connect((home / 'state.db').as_uri() + '?mode=ro', uri=True) as conn:
        assert conn.execute('SELECT id FROM sessions WHERE id=?', (key,)).fetchone()
        assert conn.execute('SELECT key FROM state_meta WHERE key=?', ('goal:' + key,)).fetchone()
    print('SEED_OK')


if __name__ == '__main__':
    main()
