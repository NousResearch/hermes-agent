"""The reference backend's text spelling of a float, held against SQLite's: the reference converts a REAL in Python, and the text must be what a SQLite TEXT column stores for it."""

from __future__ import annotations

import random
import sqlite3
import struct
from contextlib import closing

from tests.contracts._reference._real_text import real_as_text

_EDGES = (
    0.0, -0.0, 1.0, -1.0, 1.25, 0.1, 0.1 + 0.2, 1 / 3, 2 / 3, 49.47, 100.0, 1e15, 1e16, 2e16, 1e17, 1e20, 1e-4, 1e-5, 0.000123,
    999999999999999.9, 525695.039609469, 9007199254740993.0, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
    float("inf"), float("-inf"), float("nan"),
)


def _doubles() -> list[float]:
    """The edge values, then seeded doubles: every bit pattern, everyday magnitudes, short decimals, and wide decimal exponents."""
    rng = random.Random(20261107)
    values = list(_EDGES)
    values += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0] for _ in range(20000)]
    values += [struct.unpack("<d", struct.pack("<Q", rng.getrandbits(52)))[0] for _ in range(2000)]
    values += [rng.uniform(-1e6, 1e6) for _ in range(10000)]
    values += [round(rng.uniform(-1000, 1000), rng.randint(0, 6)) for _ in range(5000)]
    values += [rng.uniform(1, 10) * 10.0 ** rng.randint(-30, 30) for _ in range(5000)]
    return values


def test_the_reference_spells_a_float_as_a_sqlite_text_column_stores_it():
    values = _doubles()
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("CREATE TABLE spelled (value TEXT)")
        conn.executemany("INSERT INTO spelled (value) VALUES (?)", [(value,) for value in values])
        stored = [row[0] for row in conn.execute("SELECT value FROM spelled ORDER BY rowid")]
    mismatched = [(value, text) for value, text in zip(values, stored) if real_as_text(value) != text]
    assert len(stored) == len(values)
    assert mismatched == [], f"{len(mismatched)} of {len(values)} differ, first {mismatched[:5]} on SQLite {sqlite3.sqlite_version}"
