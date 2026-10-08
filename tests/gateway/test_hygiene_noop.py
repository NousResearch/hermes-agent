"""Hygiene no-op cooldown predicate (#21301)."""
from gateway.run import hygiene_noop_should_cool


def test_noop_cools_when_neither_aborted_nor_recovered():
    assert hygiene_noop_should_cool(aborted=False, recovered=False) is True


def test_abort_and_recovery_are_not_noops():
    assert hygiene_noop_should_cool(aborted=True, recovered=False) is False
    assert hygiene_noop_should_cool(aborted=False, recovered=True) is False
    assert hygiene_noop_should_cool(aborted=True, recovered=True) is False
