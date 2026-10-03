"""E2E helper for `project-picker.spec.ts`.

Two jobs, both against the sandbox HERMES_HOME the fixture hands us:

* ``seed <home> <json>`` — create real project rows through
  ``hermes_cli.projects_db`` (the same per-profile store the sidebar and the
  project tree read), with one folder each.
* ``sessions <home>`` — read back ``state.db`` session rows so the spec can
  prove a picked project anchors a new chat at that folder.

Run with the *installed* runtime (`HERMES_E2E_PYTHON`, default the shipped
venv) because the sandbox home has no Python environment of its own.
"""

import json
import os
import sqlite3
import sys


def seed(home: str, specs: list) -> list:
    from hermes_cli import projects_db as pdb

    with pdb.connect_closing() as conn:
        for spec in specs:
            pdb.create_project(
                conn,
                name=spec["name"],
                folders=[spec["folder"]],
                primary_path=spec["folder"],
            )

        return [{"id": p.id, "name": p.name} for p in pdb.list_projects(conn)]


def session_rows(home: str) -> dict:
    db = os.path.join(home, "state.db")

    if not os.path.exists(db):
        return {"columns": [], "rows": []}

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(sessions)")]

    if "cwd" not in columns:
        return {"columns": columns, "rows": []}

    rows = conn.execute("SELECT id, cwd FROM sessions ORDER BY rowid DESC LIMIT 40").fetchall()

    return {"columns": columns, "rows": [{"id": r["id"], "cwd": r["cwd"]} for r in rows]}


def main() -> None:
    mode, home = sys.argv[1], sys.argv[2]

    if mode == "seed":
        print(json.dumps(seed(home, json.loads(sys.argv[3]))))

        return

    print(json.dumps(session_rows(home)))


main()
