"""Legacy store fixture: the statements the demo family moves."""

ITEM_COLUMNS = "id, name, size"
_COUNT_SQL = "SELECT COUNT(*) FROM items WHERE owner = ?"


def _placeholders(count):
    return ", ".join("?" * count)


class DemoMixin:
    _TOUCH_SQL = "UPDATE items SET touched_at = ? WHERE id = {marker}".format(marker="?")

    def add_item(self, owner, name, size, now):
        def _do(conn):
            cur = conn.execute(
                f"INSERT INTO items ({ITEM_COLUMNS}, owner) VALUES (?, ?, ?, ?)",
                (None, name, size, owner),
            )
            conn.execute(self._TOUCH_SQL, (now, cur.lastrowid))
            conn.execute("UPDATE owners SET item_count = item_count + 1 WHERE id = ?", (owner,))
            return cur.lastrowid

        return self._execute_write(_do)

    def count_items(self, owner):
        return self._read_one(_COUNT_SQL, (owner,))[0]

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
        with self._read_ctx() as conn:
            return conn.execute(f"SELECT {ITEM_COLUMNS} FROM items WHERE {where} ORDER BY id", params).fetchall()

    def rename_many(self, pairs):
        self._execute_write(lambda conn: conn.executemany("UPDATE items SET name = ? WHERE id = ?", pairs))

    def reset(self):
        self._execute_write(lambda conn: conn.executescript("DELETE FROM items; DELETE FROM owners;"))

    def drain_ledger(self, conn):
        conn.execute("BEGIN")
        conn.execute("DELETE FROM ledger WHERE done = 1")
        conn.execute("COMMIT")

    def _read_one(self, sql, params=()):
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchone()
