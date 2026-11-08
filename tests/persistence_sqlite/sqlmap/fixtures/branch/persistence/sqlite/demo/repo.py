"""Backend fixture: the demo family's repository."""

from ._sql import ITEM_COLUMNS, _placeholders

_COUNT_SQL = "SELECT COUNT(*) FROM items WHERE owner = ?"
_TOUCH_SQL = "UPDATE items SET touched_at = ? WHERE id = ?"


class DemoRepository:
    def __init__(self, holder):
        self._holder = holder

    def add_item(self, owner, name, size, now):
        def _do(conn):
            cur = conn.execute(
                f"INSERT INTO items ({ITEM_COLUMNS}, owner) "
                "VALUES (?, ?, ?, ?)",
                (None, name, size, owner),
            )
            conn.execute(_TOUCH_SQL, (now, cur.lastrowid))
            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))
            return cur.lastrowid

        return self._holder.execute_write(_do)

    def count_items(self, owner):
        with self._holder.read_ctx() as conn:
            return conn.execute(_COUNT_SQL, (owner,)).fetchone()[0]

    def find_items(self, owner, names=None, min_size=None):
        clauses = ["owner = ?"]
        params = [owner]
        if names:
            clauses.append(f"name IN ({_placeholders(len(names))})")
            params.extend(names)
        if min_size is not None:
            clauses.append("size >= ?")
            params.append(min_size)
        where = " AND ".join(clauses)
        with self._holder.read_ctx() as conn:
            return conn.execute(f"SELECT {ITEM_COLUMNS} FROM items WHERE {where} ORDER BY id", params).fetchall()

    def rename_many(self, pairs):
        self._holder.execute_write(lambda conn: conn.executemany("UPDATE items SET name = ? WHERE id = ?", pairs))

    def reset(self):
        self._holder.execute_write(lambda conn: conn.executescript("DELETE FROM items; DELETE FROM owners;"))

    def drain_ledger(self):
        def _do(conn):
            conn.execute("DELETE FROM ledger WHERE done = 1")

        self._holder.execute_write(_do)
