"""Tests for optional-skills/productivity/memento-flashcards/scripts/memento_cards.py"""

import csv
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

# Add the scripts dir so we can import the module directly
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "memento-flashcards" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import memento_cards


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Redirect card storage to a temp directory for every test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(memento_cards, "DATA_DIR", data_dir)
    monkeypatch.setattr(memento_cards, "CARDS_FILE", data_dir / "cards.json")
    return data_dir


def _run(capsys, argv: list[str]) -> dict:
    """Run main() with given argv and return parsed JSON output."""
    with mock.patch("sys.argv", ["memento_cards"] + argv):
        memento_cards.main()
    captured = capsys.readouterr()
    return json.loads(captured.out)


# ── Add / List / Delete ──────────────────────────────────────────────────────

class TestCardCRUD:
    def test_add_creates_card(self, capsys):
        result = _run(capsys, ["add", "--question", "What is 2+2?", "--answer", "4", "--collection", "Math"])
        assert result["ok"] is True
        card = result["card"]
        assert card["question"] == "What is 2+2?"
        assert card["answer"] == "4"
        assert card["collection"] == "Math"
        assert card["status"] == "learning"
        assert card["ease_streak"] == 0
        uuid.UUID(card["id"])  # validates it's a real UUID


    def test_list_all(self, capsys):
        _run(capsys, ["add", "--question", "Q1", "--answer", "A1", "--collection", "C1"])
        _run(capsys, ["add", "--question", "Q2", "--answer", "A2", "--collection", "C2"])
        result = _run(capsys, ["list"])
        assert result["count"] == 2



    def test_delete_card(self, capsys):
        result = _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = result["card"]["id"]
        del_result = _run(capsys, ["delete", "--id", card_id])
        assert del_result["ok"] is True
        assert del_result["deleted"] == card_id
        # Verify gone
        list_result = _run(capsys, ["list"])
        assert list_result["count"] == 0




# ── Due Filtering ────────────────────────────────────────────────────────────

class TestDueFiltering:
    def test_new_card_is_due(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        result = _run(capsys, ["due"])
        assert result["count"] == 1

    def test_future_card_not_due(self, capsys, monkeypatch):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        # Rate it good (pushes next_review_at to +3 days)
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        _run(capsys, ["rate", "--id", card_id, "--rating", "good"])
        result = _run(capsys, ["due"])
        assert result["count"] == 0

    def test_retired_card_not_due(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        _run(capsys, ["rate", "--id", card_id, "--rating", "retire"])
        result = _run(capsys, ["due"])
        assert result["count"] == 0

    def test_due_with_collection_filter(self, capsys):
        _run(capsys, ["add", "--question", "Q1", "--answer", "A1", "--collection", "C1"])
        _run(capsys, ["add", "--question", "Q2", "--answer", "A2", "--collection", "C2"])
        result = _run(capsys, ["due", "--collection", "C1"])
        assert result["count"] == 1
        assert result["cards"][0]["collection"] == "C1"


# ── Rating and Rescheduling ──────────────────────────────────────────────────

class TestRating:
    def test_hard_adds_1_day(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        before = datetime.now(timezone.utc)
        result = _run(capsys, ["rate", "--id", card_id, "--rating", "hard"])
        after = datetime.now(timezone.utc)
        next_review = datetime.fromisoformat(result["card"]["next_review_at"])
        assert before + timedelta(days=1) <= next_review <= after + timedelta(days=1)
        assert result["card"]["ease_streak"] == 0

    def test_good_adds_3_days(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        before = datetime.now(timezone.utc)
        result = _run(capsys, ["rate", "--id", card_id, "--rating", "good"])
        next_review = datetime.fromisoformat(result["card"]["next_review_at"])
        assert next_review >= before + timedelta(days=3)
        assert result["card"]["ease_streak"] == 0


    def test_retire_sets_retired(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        result = _run(capsys, ["rate", "--id", card_id, "--rating", "retire"])
        assert result["card"]["status"] == "retired"
        assert result["card"]["ease_streak"] == 0

    def test_auto_retire_after_3_easys(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]

        # Force card to be due by manipulating next_review_at through rate
        for i in range(3):
            # Load and directly set next_review_at to now so it's ratable
            data = memento_cards._load()
            for c in data["cards"]:
                if c["id"] == card_id:
                    c["next_review_at"] = memento_cards._iso(memento_cards._now())
            memento_cards._save(data)

            result = _run(capsys, ["rate", "--id", card_id, "--rating", "easy"])

        assert result["card"]["ease_streak"] == 3
        assert result["card"]["status"] == "retired"




# ── CSV Export/Import ────────────────────────────────────────────────────────

class TestCSV:
    def test_export_import_roundtrip(self, capsys, tmp_path):
        _run(capsys, ["add", "--question", "Q1", "--answer", "A1", "--collection", "C1"])
        _run(capsys, ["add", "--question", "Q2", "--answer", "A2", "--collection", "C2"])

        csv_path = str(tmp_path / "export.csv")
        result = _run(capsys, ["export", "--output", csv_path])
        assert result["ok"] is True
        assert result["exported"] == 2

        # Verify CSV content
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == ["Q1", "A1", "C1"]
        assert rows[1] == ["Q2", "A2", "C2"]

        # Delete all and reimport
        data = memento_cards._load()
        data["cards"] = []
        memento_cards._save(data)

        result = _run(capsys, ["import", "--file", csv_path, "--collection", "Fallback"])
        assert result["ok"] is True
        assert result["imported"] == 2

        # Verify imported cards use CSV collection column
        list_result = _run(capsys, ["list"])
        collections = {c["collection"] for c in list_result["cards"]}
        assert collections == {"C1", "C2"}





# ── Quiz Batch Add ───────────────────────────────────────────────────────────



# ── Statistics ───────────────────────────────────────────────────────────────



# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_corrupt_json_recovery(self, capsys):
        """Corrupt JSON file should be treated as empty."""
        memento_cards.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(memento_cards.CARDS_FILE, "w") as f:
            f.write("{corrupted json...")
        result = _run(capsys, ["list"])
        assert result["count"] == 0
        # Can still add
        result = _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        assert result["ok"] is True


    def test_atomic_write_creates_dir(self, capsys):
        """Data dir is created automatically if missing."""
        import shutil
        if memento_cards.DATA_DIR.exists():
            shutil.rmtree(memento_cards.DATA_DIR)
        result = _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        assert result["ok"] is True
        assert memento_cards.CARDS_FILE.exists()



# ── User Answer Tracking ────────────────────────────────────────────────────

class TestUserAnswer:



    def test_user_answer_persists_in_list(self, capsys):
        _run(capsys, ["add", "--question", "Q", "--answer", "A"])
        card_id = _run(capsys, ["list"])["cards"][0]["id"]
        _run(capsys, ["rate", "--id", card_id, "--rating", "easy",
                      "--user-answer", "my answer"])
        result = _run(capsys, ["list"])
        assert result["cards"][0]["last_user_answer"] == "my answer"

