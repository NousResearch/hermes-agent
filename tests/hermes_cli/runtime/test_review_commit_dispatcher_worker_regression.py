from __future__ import annotations


def test_gate10j_real_worker_review_commit_harness_is_importable_and_opt_in():
    from scripts.runtime_smokes import gate10j_review_commit_runtime as smoke

    assert callable(smoke.run_gate10j_smoke)
    assert smoke.CODER_TOOLS == [
        "kanban_show",
        "write_file",
        "kanban_comment",
        "kanban_complete",
        "kanban_block",
    ]
    assert smoke.REVIEWER_TOOLS == [
        "kanban_show",
        "read_file",
        "kanban_comment",
        "kanban_complete",
        "kanban_block",
    ]
    assert smoke.DEFAULT_BOARD.name == "kanban.db"
