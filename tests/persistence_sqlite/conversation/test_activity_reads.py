"""The activity reader must bound BLOB I/O before decoding and exclude private columns."""
from contextlib import contextmanager
import sqlite3


from persistence.sqlite.conversation._store.stored_reads import activity_messages


def test_activity_blob_io_is_bounded_before_reading():
    sizes = []
    class Blob:
        def __init__(self, blob):
            self.blob = blob
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.blob.close()
        def __len__(self):
            return len(self.blob)
        def read(self, size):
            assert len(self.blob) <= 8192, "oversized BLOB read"
            sizes.append(len(self.blob))
            return self.blob.read(size)
    class Connection(sqlite3.Connection):
        def blobopen(self, *args, **kwargs):
            assert kwargs["readonly"] is True
            return Blob(super().blobopen(*args, **kwargs))
    with sqlite3.connect(":memory:", factory=Connection) as db:
        db.row_factory = sqlite3.Row
        db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, "
                   "content TEXT, active INTEGER DEFAULT 1, display_kind TEXT, _compressed_summary INTEGER DEFAULT 0, "
                   "reasoning TEXT, tool_calls TEXT)")
        for role, content in (("user", "a" * 8192), ("assistant", "b" * 8193),
                              ("assistant", "é" * 4096), ("tool", "PRIVATE TOOL")):
            db.execute("INSERT INTO messages (session_id,role,timestamp,content) VALUES ('s',?,1,?)", (role, content))
        def authorize(action, table, column, *args):
            if action == sqlite3.SQLITE_READ and column in {"reasoning", "tool_calls"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        db.set_authorizer(authorize)
        class Holder:
            def _read_retrying_ioerr(self, fn):
                return fn(db)
        rows = activity_messages(Holder(), "s")
    assert len(rows) == 3
    assert sizes == [8192, 8192]
    assert rows[1].content is None
    assert rows[1].content_oversized is True



def test_activity_role_and_blob_share_one_sqlite_snapshot(tmp_path):
    path = tmp_path / "snapshot.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, "
                   "content TEXT, active INTEGER DEFAULT 1, display_kind TEXT, _compressed_summary INTEGER DEFAULT 0)")
    writer.execute("INSERT INTO messages VALUES (1,'s','assistant',1,'public answer',1,NULL,0)")
    writer.commit()
    class Connection(sqlite3.Connection):
        def execute(self, sql, *args):
            cursor = super().execute(sql, *args)
            if sql.startswith("SELECT id, role"):
                class Cursor:
                    def fetchall(self):
                        rows = cursor.fetchall()
                        writer.execute("UPDATE messages SET role='tool', content='PRIVATE TOOL'")
                        writer.commit()
                        return rows
                return Cursor()
            return cursor
    reader = sqlite3.connect(path, factory=Connection)
    reader.row_factory = sqlite3.Row
    class Holder:
        @contextmanager
        def _read_ctx(self):
            yield reader
        def _read_retrying_ioerr(self, fn):
            return fn(reader)
    try:
        rows = activity_messages(Holder(), "s")
        assert rows[0].content == "public answer"
        assert reader.in_transaction is False
    finally:
        reader.close()
        writer.close()
