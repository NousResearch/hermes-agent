import sqlite3

from hermes_cli.kanban_generation import acquire_fence, activate_first_write, build_ready, initialize_generation_db, publish_prepared


def test_gateway_style_connection_selects_published_generation_without_migration(tmp_path, monkeypatch):
    home = tmp_path / "home"
    board = home / "kanban" / "boards" / "default"
    generation = board / "generations" / "g1"
    generation.mkdir(parents=True)
    db = generation / "kanban.db"
    initialize_generation_db(db, gid="g1", nonce="n", control_root=board)
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO tasks VALUES ('gateway', 'selected')")
        conn.commit()
    ready = build_ready(generation, gid="g1", nonce="n")
    fence = acquire_fence(board, generation="g1", owner="test")
    publish_prepared(generation, expected_manifest=ready["manifest_digest"], fence_token=fence["token"])
    activate_first_write(generation, fence_epoch=fence["epoch"])
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli.kanban_db_connect import connect
    conn = connect(board="default")
    try:
        assert conn.execute("SELECT title FROM tasks WHERE id='gateway'").fetchone()[0] == "selected"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='tasks'").fetchone()
    finally:
        conn.close()
