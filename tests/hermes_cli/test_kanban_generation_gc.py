import sqlite3
import pytest

from hermes_cli.kanban_generation import GenerationError, gc_generations


def test_gc_retains_invalid_ready_for_recovery(tmp_path):
    board = tmp_path / "board"
    root = board / "generations"
    stale = root / "stale"
    stale.mkdir(parents=True)
    (stale / "READY.json").write_text("{}")
    with pytest.raises(GenerationError, match="READY_INVALID"):
        gc_generations(board)
    assert stale.exists()


def test_gc_retains_journaled_generation_even_without_pointer(tmp_path):
    board = tmp_path / "board"
    generation = board / "generations" / "journaled"
    generation.mkdir(parents=True)
    (generation / "import-journal.json").write_text('{"state":"prepared"}')
    with pytest.raises(GenerationError, match="READY_INVALID"):
        gc_generations(board)
    assert generation.exists()
