"""SQL-evolution fitness template.

Evaluates a candidate SQL query against an in-memory SQLite fixture.
The default rewards queries that return exactly the expected number of
rows; replace the fixture and expectation with your schema.

Candidates that raise a ``sqlite3.Error`` score 0.0.
"""

from __future__ import annotations

import sqlite3

from evolver_sdk import fitness_spec


_FIXTURE = """
    CREATE TABLE users(id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
    INSERT INTO users(name, age) VALUES
        ('alice',   31),
        ('bob',     42),
        ('carol',   28),
        ('dave',    55),
        ('erin',    19);
"""

_EXPECTED_ROWS = 3     # e.g. users older than 25


@fitness_spec(held_out_frac=0.2, timeout_s=10)
def fitness(candidate: str, context: dict) -> float:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(_FIXTURE)
        rows = conn.execute(candidate).fetchall()
    except sqlite3.Error:
        return 0.0
    finally:
        conn.close()
    distance = abs(len(rows) - _EXPECTED_ROWS)
    return 1.0 - min(1.0, distance / max(_EXPECTED_ROWS, 1))
