"""Single-writer invariant for task_events.

CONTINUATION WORK, not part of the 8 September candidate. Added by Claude Code
Windows session 82b5e4d1-0800-4482-baf7-160b11f66249 on 2026-09-11 under the
adopt-and-continue authorisation; the surrounding candidate is the work of the
earlier Hermes CLI worker and is unmodified by this file.

WHY. Durable history captures inside ``kanban_db._append_event``: it inserts the
source row and, on the SAME connection inside the SAME transaction, records the
durable entry. That design is only lossless if ``_append_event`` is the ONLY way a
task_events row is ever created. It was not. The installed source carried five raw
``INSERT INTO task_events`` statements in the dashboard plugin; the candidate routed
two of them and left three, so dashboard-originated 'reprioritized' and 'edited'
events were captured by nothing while every test still passed.

A per-call-site test would not have caught that, because the sites it did not know
about are exactly the ones that bypass the writer. So this asserts the INVARIANT over
the whole product tree: outside kanban_db._append_event itself, no product module may
insert into task_events. Tests are excluded deliberately: fixtures legitimately seed
rows directly, and forbidding that would make the guard unmaintainable.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSERT = re.compile(r"INSERT\s+INTO\s+task_events", re.IGNORECASE)
# The single permitted writer.
ALLOWED = {Path("hermes_cli/kanban_db.py")}


def _product_sources():
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        if parts[0] in {"tests", ".git", ".phase3-evidence", "node_modules"}:
            continue
        if any(p.startswith(".") for p in parts[:-1]):
            continue
        yield rel, path


def test_append_event_is_the_only_task_events_writer():
    offenders = []
    for rel, path in _product_sources():
        if rel in ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if INSERT.search(line):
                offenders.append(f"{rel.as_posix()}:{number}")
    assert not offenders, (
        "task_events must only be written through kanban_db._append_event, or durable "
        "history silently misses these events. Offending raw inserts: " + ", ".join(offenders)
    )


def test_the_permitted_writer_still_exists():
    """Non-vacuity: if the allowed file or its insert disappears, this guard is empty."""
    allowed = ROOT / "hermes_cli" / "kanban_db.py"
    assert allowed.is_file(), "the permitted writer file is missing; the guard above is vacuous"
    text = allowed.read_text(encoding="utf-8", errors="replace")
    assert len(INSERT.findall(text)) == 1, (
        "expected exactly one INSERT INTO task_events inside kanban_db.py (_append_event); "
        "found " + str(len(INSERT.findall(text)))
    )
