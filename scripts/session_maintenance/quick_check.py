"""Standalone cold-snapshot integrity check (parent enforces an OS process timeout)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hermes_state_fts import load_fts5_cjk_extension
from verification import open_snapshot


def check_snapshot(path):
    conn = open_snapshot(Path(path))
    try:
        load_fts5_cjk_extension(conn)
        return [row[0] for row in conn.execute('PRAGMA quick_check')]
    finally:
        conn.close()


if __name__ == '__main__':
    output = check_snapshot(sys.argv[1])
    print(json.dumps(output))
    sys.exit(0 if output == ['ok'] else 1)
