from agent.prompt_builder import KANBAN_GUIDANCE


def test_kanban_guidance_splits_governed_reviewer_loop_from_legacy_review_required():
    assert "admission_snapshot.ready_contract" in KANBAN_GUIDANCE
    assert "reviewer_loop.required=true" in KANBAN_GUIDANCE
    assert "do NOT use legacy `review-required`" in KANBAN_GUIDANCE
    assert "missing-lifecycle-contract" in KANBAN_GUIDANCE
    assert "changed_files, read_only" in KANBAN_GUIDANCE
    assert "Legacy exception" in KANBAN_GUIDANCE
