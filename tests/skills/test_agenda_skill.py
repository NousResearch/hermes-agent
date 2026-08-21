"""
Tests for the Hermes Agenda skill (skills/productivity/agenda).

Covers:
- Database schema auto-creation
- Adding agenda items with domains, priorities, and details
- Priority ordering (P1 before P3, FIFO for same priority)
- Next item retrieval and state transition (pending -> active)
- Completing items and recording outcomes in the log table
- Recurring items cooldown retention
- Idea sparks creation and listing
- Status reporting aggregation
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENDA_SCRIPT = REPO_ROOT / "skills" / "productivity" / "agenda" / "scripts" / "agenda.py"

# Import functions directly from agenda.py
sys.path.insert(0, str(AGENDA_SCRIPT.parent))
import agenda


@pytest.fixture
def temp_db(tmp_path):
    return tmp_path / "test_agenda.db"


class TestAgendaDirectAPI:
    def test_auto_initializes_db_and_tables(self, temp_db):
        conn = agenda.get_conn(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        assert "agenda" in tables
        assert "log" in tables
        assert "sparks" in tables

    def test_add_and_list_items(self, temp_db):
        item = agenda.add_item(
            "Read AI benchmark paper",
            detail="Check section 4 results",
            domain="research",
            kind="paper",
            priority=1,
            db_path=temp_db,
        )
        assert item["id"] == 1
        assert item["title"] == "Read AI benchmark paper"
        assert item["priority"] == 1
        assert item["status"] == "pending"

        items = agenda.list_items(db_path=temp_db)
        assert len(items) == 1
        assert items[0]["title"] == "Read AI benchmark paper"

    def test_priority_ordering_in_next_items(self, temp_db):
        agenda.add_item("Low priority chore", priority=4, db_path=temp_db)
        agenda.add_item("Urgent bug fix", priority=1, db_path=temp_db)
        agenda.add_item("Medium priority task", priority=2, db_path=temp_db)

        next_items = agenda.next_items(n=2, db_path=temp_db)
        assert len(next_items) == 2
        assert next_items[0]["title"] == "Urgent bug fix"
        assert next_items[0]["status"] == "pending"  # returned row was pending, now active in DB
        assert next_items[1]["title"] == "Medium priority task"

        # Verify DB updated to active
        active_items = agenda.list_items(status="active", db_path=temp_db)
        assert len(active_items) == 2

    def test_done_item_and_logging(self, temp_db):
        item = agenda.add_item("Write docs", priority=2, db_path=temp_db)
        done = agenda.done_item(item["id"], outcome="Completed all sections", db_path=temp_db)
        assert done is not None
        assert done["status"] == "done"
        assert done["times_done"] == 1
        assert done["last_done"] is not None

        conn = agenda.get_conn(temp_db)
        cur = conn.cursor()
        cur.execute("SELECT * FROM log WHERE agenda_id = ?", (item["id"],))
        log_entry = cur.fetchone()
        conn.close()
        assert log_entry is not None
        assert log_entry["outcome"] == "Completed all sections"

    def test_recurring_item_cooldown(self, temp_db):
        item = agenda.add_item("Weekly status review", priority=2, cooldown_days=7, db_path=temp_db)
        assert item["status"] == "recurring"

        done = agenda.done_item(item["id"], outcome="Sprint 1 review done", db_path=temp_db)
        assert done["status"] == "recurring"
        assert done["times_done"] == 1

    def test_sparks_recording(self, temp_db):
        spark = agenda.add_spark(
            "Train dynamic memory retrieval agent",
            observation="Observed high recall in test runs",
            domain="ml",
            score=0.9,
            db_path=temp_db,
        )
        assert spark["id"] == 1
        assert spark["idea"] == "Train dynamic memory retrieval agent"
        assert spark["score"] == 0.9

        sparks = agenda.list_sparks(db_path=temp_db)
        assert len(sparks) == 1

    def test_status_reporting(self, temp_db):
        agenda.add_item("Task 1", domain="research", db_path=temp_db)
        agenda.add_item("Task 2", domain="dev", db_path=temp_db)
        agenda.add_spark("Idea 1", domain="research", db_path=temp_db)

        st = agenda.get_status(db_path=temp_db)
        assert st["status_counts"]["pending"] == 2
        assert st["domain_counts"]["research"] == 1
        assert st["domain_counts"]["dev"] == 1
        assert st["open_sparks"] == 1


class TestAgendaCLI:
    def test_cli_add_and_next_json(self, temp_db):
        # Add item via CLI
        res = subprocess.run(
            [
                sys.executable,
                str(AGENDA_SCRIPT),
                "--db",
                str(temp_db),
                "--json",
                "add",
                "CLI Task Test",
                "--priority",
                "1",
                "--domain",
                "testing",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        assert data["title"] == "CLI Task Test"
        assert data["priority"] == 1

        # Next item via CLI
        res = subprocess.run(
            [
                sys.executable,
                str(AGENDA_SCRIPT),
                "--db",
                str(temp_db),
                "--json",
                "next",
                "--n",
                "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        items = json.loads(res.stdout)
        assert len(items) == 1
        assert items[0]["title"] == "CLI Task Test"
