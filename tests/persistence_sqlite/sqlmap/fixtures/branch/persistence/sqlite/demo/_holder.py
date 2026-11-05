"""Backend fixture: the writer holder."""


class Holder:
    def execute_write(self, fn):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(self._conn)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return result
