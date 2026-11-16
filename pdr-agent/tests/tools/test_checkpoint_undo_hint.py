"""A whole-tree /rollback undoes the turn that made (or followed) the checkpoint, never a turn kept before it."""
from tools.checkpoint_undo_hint import rollback_rewinds_newest_turn


def test_the_exact_take_time_separates_the_turn_that_made_the_checkpoint_from_one_kept_before_it():
    """All within one git second: the kept turn ended at T+0.2, the snapshot was taken at T+0.5, and the
    turn that took it wrote its tool rows at T+0.7."""
    kept = [{"role": "user", "timestamp": 1700000000.1}, {"role": "assistant", "timestamp": 1700000000.2}]
    maker = [{"role": "user", "timestamp": 1700000000.3}, {"role": "tool", "timestamp": 1700000000.7}]
    taken = (1700000000.5, True)
    assert rollback_rewinds_newest_turn(kept, [0], taken) is False
    assert rollback_rewinds_newest_turn(maker, [0], taken) is True
    # An older build's snapshot records only its second: inside it, the turn is ambiguous and kept.
    assert rollback_rewinds_newest_turn(kept, [0], (1700000000.0, False)) is False
    assert rollback_rewinds_newest_turn(kept, [0], (1699999999.0, False)) is True
