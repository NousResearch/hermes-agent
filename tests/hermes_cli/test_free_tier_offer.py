"""The install-wide record behind the free tier's sign-in offer: first due a fixed delay after the first
finished task, then again only after another finished task and never sooner than the back-off since
the previous offer; each due offer is claimable exactly once, and only while the user is on the free
tier."""

from __future__ import annotations

import threading

import pytest

from hermes_cli import anon_auth, free_tier_offer

DELAY = free_tier_offer.OFFER_DELAY_S
BACKOFF = free_tier_offer.REOFFER_AFTER_S


class _Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root" / "profiles" / "setup"))
    monkeypatch.setattr(anon_auth, "has_guest", lambda: True)
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: True)
    fake = _Clock()
    monkeypatch.setattr(free_tier_offer, "_clock", fake)
    return fake


def _take_due_offer(clock: _Clock) -> None:
    clock.now += free_tier_offer.offer_due_in()
    assert free_tier_offer.claim_offer() is True


def test_first_offer_is_due_after_the_delay_from_the_first_task_and_later_tasks_do_not_move_it(clock):
    assert free_tier_offer.offer_due_in() is None
    free_tier_offer.record_task_done()
    assert free_tier_offer.offer_due_in() == DELAY
    clock.now += 60
    free_tier_offer.record_task_done()
    assert free_tier_offer.offer_due_in() == DELAY - 60
    clock.now += DELAY
    assert free_tier_offer.offer_due_in() == 0


def test_claim_is_refused_before_due_then_granted_once(clock):
    free_tier_offer.record_task_done()
    assert free_tier_offer.claim_offer() is False
    clock.now += DELAY
    assert free_tier_offer.claim_offer() is True
    assert free_tier_offer.claim_offer() is False


def test_no_reoffer_without_a_task_finished_after_the_offer(clock):
    free_tier_offer.record_task_done()
    _take_due_offer(clock)
    clock.now += 10 * BACKOFF[-1]  # time alone never brings it back
    assert free_tier_offer.offer_due_in() is None
    assert free_tier_offer.claim_offer() is False


@pytest.mark.parametrize("task_after_offer_s,due_in", [
    (60, BACKOFF[0] - 60),  # a task soon after the offer: the back-off decides
    (BACKOFF[0], DELAY),    # a task after the back-off ran out: the delay after that task decides
], ids=["back-off-bound", "task-bound"])
def test_reoffer_is_the_later_of_task_plus_delay_and_offer_plus_back_off(clock, task_after_offer_s, due_in):
    free_tier_offer.record_task_done()
    _take_due_offer(clock)
    clock.now += task_after_offer_s
    free_tier_offer.record_task_done()
    assert free_tier_offer.offer_due_in() == due_in


def test_back_off_climbs_the_ladder_then_holds_at_its_last_step(clock):
    free_tier_offer.record_task_done()
    _take_due_offer(clock)
    for backoff in (*BACKOFF, BACKOFF[-1]):
        clock.now += 1  # a task right after the offer: the back-off is the bound
        free_tier_offer.record_task_done()
        assert free_tier_offer.offer_due_in() == backoff - 1
        _take_due_offer(clock)


def test_concurrent_claims_grant_exactly_one(clock):
    free_tier_offer.record_task_done()
    clock.now += DELAY
    results: list[bool] = []
    threads = [threading.Thread(target=lambda: results.append(free_tier_offer.claim_offer())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 7 + [True]


def test_record_is_install_wide_across_profiles(clock, tmp_path, monkeypatch):
    free_tier_offer.record_task_done()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "root" / "profiles" / "work"))
    assert free_tier_offer.offer_due_in() == DELAY


@pytest.mark.parametrize("signed_in,tier_on", [(True, True), (False, False)], ids=["signed-in", "tier-off"])
def test_never_offered_off_the_free_tier(clock, monkeypatch, signed_in, tier_on):
    free_tier_offer.record_task_done()
    clock.now += DELAY
    monkeypatch.setattr(anon_auth, "has_guest", lambda: not signed_in)
    monkeypatch.setattr(anon_auth, "guest_enabled", lambda: tier_on)
    assert free_tier_offer.offer_due_in() is None
    assert free_tier_offer.claim_offer() is False
