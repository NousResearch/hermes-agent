import json
import sqlite3

import pytest

from hermes_cli.kanban_generation import GenerationError, acquire_fence, adopt_fence, board_write_guard, release_fence


def test_fence_is_exclusive_persistent_and_adoptable(tmp_path):
    first = acquire_fence(tmp_path, generation="g1", owner="builder")
    with pytest.raises(GenerationError, match="WRITER_FENCE_HELD"):
        acquire_fence(tmp_path, generation="g1", owner="other")
    adopted = adopt_fence(tmp_path, token=first["token"], owner="recovered")
    assert adopted["token"] == first["token"]
    assert adopted["epoch"] > first["epoch"]
    release_fence(tmp_path, token=adopted["token"])
    assert not (tmp_path / "generation.writer-fence").exists()


def test_shared_mutation_guard_requires_current_fence_token(tmp_path):
    fence = acquire_fence(tmp_path, generation="g1", owner="builder")
    with pytest.raises(GenerationError, match="WRITER_FENCE_TOKEN_INVALID"):
        with board_write_guard(tmp_path, owner="writer"):
            pass
    with board_write_guard(tmp_path, owner="writer", token=fence["token"]):
        (tmp_path / "mutation.marker").write_text("ok", encoding="utf-8")


def test_guard_is_legacy_compatible_without_a_generation_fence(tmp_path):
    with board_write_guard(tmp_path, owner="legacy"):
        (tmp_path / "legacy.marker").write_text("ok", encoding="utf-8")
