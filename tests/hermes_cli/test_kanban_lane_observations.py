"""Storage/lifecycle contracts; these are not live runtime capture evidence."""
import pytest

from hermes_cli.kanban_provider_lanes import Candidate, LaneLedger


@pytest.fixture
def ledger(tmp_path):
    instance = LaneLedger(tmp_path / "lanes.db")
    yield instance
    instance.close()


def reserve(ledger, task="build", alive=lambda row: True):
    return ledger.reserve(board="board", task=task, run=1, owner="owner",
                          candidates=(Candidate("anthropic", "requested"),), alive=alive)


def test_worker_binding_cannot_be_overwritten(ledger):
    admitted = reserve(ledger)
    assert admitted is not None
    token, _ = admitted
    assert not ledger.bind_worker(token, "another-owner", "worker")
    assert ledger.bind_worker(token, "owner", "worker")
    assert ledger.bind_worker(token, "owner", "worker")
    assert not ledger.bind_worker(token, "owner", "replacement-worker")
    assert ledger.snapshot()[0]["worker"] == "worker"


def test_provider_mismatch_quarantine_survives_later_matching_observation(ledger):
    admitted = reserve(ledger)
    assert admitted is not None
    token, _ = admitted
    assert ledger.bind_worker(token, "owner", "worker")
    assert ledger.observe(token, "worker", provider="openai-codex", model="actual")
    assert reserve(ledger, "next") is None
    assert ledger.observe(token, "worker", provider="anthropic", model="requested")
    assert reserve(ledger, "next") is None
    assert reserve(ledger, "next", alive=lambda row: False) is not None
    assert len(ledger.observations(board="board", task="build", run=1)) == 2


def test_model_mismatch_is_retained_after_worker_exit(ledger):
    admitted = reserve(ledger)
    assert admitted is not None
    token, _ = admitted
    assert ledger.bind_worker(token, "owner", "worker")
    assert not ledger.observe(token, "impostor", provider="anthropic", model="actual")
    assert ledger.observe(token, "worker", provider="anthropic", model="actual")
    assert not ledger.release(token, "owner")
    assert not ledger.release(token, "owner", alive=lambda row: True)
    assert not ledger.release(token, "owner", alive=lambda row: None)
    assert ledger.release(token, "owner", alive=lambda row: False)
    assert ledger.snapshot() == []
    evidence = ledger.observations(board="board", task="build", run=1)
    assert len(evidence) == 1
    assert evidence[0]["requested_model"] == "requested"
    assert evidence[0]["observed_model"] == "actual"
    assert evidence[0]["observed_at"] > 0
    assert ledger.observations(board="another-board", task="build", run=1) == []


def test_release_resolver_error_preserves_reservation(ledger):
    admitted = reserve(ledger)
    assert admitted is not None
    token, _ = admitted
    assert ledger.bind_worker(token, "owner", "worker")

    def unavailable(row):
        raise OSError("unreadable process table")

    with pytest.raises(OSError):
        ledger.release(token, "owner", alive=unavailable)
    assert len(ledger.snapshot()) == 1
    assert reserve(ledger, "next") is not None
