"""Recovery must not let a corrupt rowid hide readable tail rows (#158969)."""

import sqlite3
import struct

from persistence.sqlite.admin._recovery import _copy_table_salvage


def test_salvage_continues_after_out_of_order_leaf_rowid(tmp_path):
    path = tmp_path / "damaged.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA page_size=512")
        db.execute("CREATE TABLE records (value TEXT NOT NULL)")
        db.executemany("INSERT INTO records VALUES (?)", [(str(i) + "x" * 100,) for i in range(1, 481)])
        db.commit()
        root = db.execute("SELECT rootpage FROM sqlite_master WHERE name='records'").fetchone()[0]
    db.close()
    raw = bytearray(path.read_bytes())
    page_size = 512

    def leaves(page):
        start = (page - 1) * page_size
        header = start + (100 if page == 1 else 0)
        if raw[header] == 13:
            return [page]
        count = struct.unpack_from(">H", raw, header + 3)[0]
        result = []
        for i in range(count):
            cell = struct.unpack_from(">H", raw, header + 12 + 2 * i)[0]
            result.extend(leaves(struct.unpack_from(">I", raw, start + cell)[0]))
        result.extend(leaves(struct.unpack_from(">I", raw, header + 8)[0]))
        return result

    leaf = leaves(root)[len(leaves(root)) // 2]
    start = (leaf - 1) * page_size
    cell = struct.unpack_from(">H", raw, start + 8)[0]
    # These records have a one-byte payload length and a two-byte rowid.
    assert raw[start + cell] < 128
    rowid_at = start + cell + 1
    assert raw[rowid_at] >= 128 and raw[rowid_at + 1] < 128
    damaged_id = ((raw[rowid_at] & 127) << 7) | raw[rowid_at + 1]
    raw[rowid_at:rowid_at + 2] = bytes((255, 127))  # 16383, outside the table's real bounds
    path.write_bytes(raw)

    source = sqlite3.connect(path)
    destination = sqlite3.connect(":memory:")
    try:
        destination.execute("CREATE TABLE records (value TEXT NOT NULL)")
        report = _copy_table_salvage(source, destination, "records", chunk_size=20,
                                     progress_cb=None, source_rows=480)
        values = {row[0] for row in destination.execute("SELECT value FROM records")}
        assert "480" + "x" * 100 in values
        assert report["copied_rows"] == 479
        assert len(values) == 479
        assert report["status"] == "partial"
        assert any(span["low"] <= damaged_id <= span["high"] for span in report["skipped_rowid_ranges"])
    finally:
        source.close()
        destination.close()


def test_healthy_sparse_rowids_do_not_trigger_reconciliation():
    source = sqlite3.connect(":memory:")
    destination = sqlite3.connect(":memory:")
    try:
        source.execute("CREATE TABLE records (value TEXT NOT NULL)")
        destination.execute("CREATE TABLE records (value TEXT NOT NULL)")
        source.executemany("INSERT INTO records (rowid, value) VALUES (?, ?)",
                           [(-1000, "first"), (1000000, "last")])
        report = _copy_table_salvage(source, destination, "records", chunk_size=1,
                                     progress_cb=None, source_rows=2)
        assert report["status"] == "complete"
        assert report["copied_rows"] == 2
        assert report["range_queries"] == 1
        assert report["skipped_rowid_ranges"] == []
        assert destination.execute("SELECT value FROM records").fetchall() == [("first",), ("last",)]
    finally:
        source.close()
        destination.close()
