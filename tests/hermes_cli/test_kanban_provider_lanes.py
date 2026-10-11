"""Provider admission contracts, including independent real-process races."""
import multiprocessing
from pathlib import Path

import pytest

from hermes_cli.kanban_provider_lanes import (
    Allowance, Candidate, LaneLedger, aged_priority, load_router,
    memory_admits, order_candidates,
)

CLAUDE = Candidate("anthropic", "claude-fixture")
CODEX = Candidate("openai-codex", "codex-fixture")
ROUTES = (CLAUDE, CODEX)


def _reserve(ledger, task, candidates=ROUTES, board="a", alive=lambda row: True):
    return ledger.reserve(board=board, task=task, run=1, candidates=candidates,
                          owner="dispatcher-fingerprint", alive=alive)


def test_lane_caps_total_and_cross_board(tmp_path):
    ledger = LaneLedger(tmp_path / "lanes.db")
    try:
        assert _reserve(ledger, "a", (CLAUDE,))
        assert _reserve(ledger, "b", (CLAUDE,), board="second-board")
        assert _reserve(ledger, "c", (CLAUDE,)) is None
        assert _reserve(ledger, "c")[1] == CODEX
        assert _reserve(ledger, "d", (CODEX,))
        assert _reserve(ledger, "e") is None
        assert len(ledger.snapshot()) == 4
    finally:
        ledger.close()


def _race_worker(path, barrier, results, board):
    ledger = LaneLedger(Path(path))
    try:
        barrier.wait(timeout=30)
        for number in range(10):
            result = _reserve(ledger, f"{board}-{number}", board=board)
            results.put(result[1].provider if result else None)
    finally:
        ledger.close()


def test_two_processes_cannot_overbook_two_boards(tmp_path):
    path = tmp_path / "lanes.db"
    ledger = LaneLedger(path)
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    children = [ctx.Process(target=_race_worker, args=(str(path), barrier, results, board))
                for board in ("one", "two")]
    try:
        for child in children:
            child.start()
        observations = [results.get(timeout=40) for _ in range(20)]
        for child in children:
            child.join(timeout=30)
            assert child.exitcode == 0
        assert observations.count("anthropic") == 2
        assert observations.count("openai-codex") == 2
        assert observations.count(None) == 16
        assert len(ledger.snapshot()) == 4
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(timeout=30)
        results.close()
        results.join_thread()
        ledger.close()


def test_release_crash_and_unknown_liveness(tmp_path):
    ledger = LaneLedger(tmp_path / "lanes.db")
    try:
        first = _reserve(ledger, "a", (CLAUDE,))
        second = _reserve(ledger, "b", (CLAUDE,))
        assert not ledger.release(first[0], "different-owner")
        assert _reserve(ledger, "c", (CLAUDE,), alive=lambda row: None) is None
        assert ledger.release(first[0], "dispatcher-fingerprint")
        assert _reserve(ledger, "c", (CLAUDE,))
        assert _reserve(ledger, "d", (CLAUDE,), alive=lambda row: row["token"] != second[0])
        assert len(ledger.snapshot()) == 2
    finally:
        ledger.close()


def test_duplicate_run_and_failed_resolver_do_not_mutate_reservations(tmp_path):
    ledger = LaneLedger(tmp_path / "lanes.db")
    try:
        assert _reserve(ledger, "a")
        assert _reserve(ledger, "a") is None
        before = ledger.snapshot()

        def unavailable(row):
            raise OSError("cannot inspect process")

        with pytest.raises(OSError):
            _reserve(ledger, "b", alive=unavailable)
        assert ledger.snapshot() == before
        assert _reserve(ledger, "b")
    finally:
        ledger.close()


@pytest.mark.parametrize("stage", ["plan", "release", "protected", "ruling", "unknown"])
def test_protected_stages_never_cross_provider(stage):
    assert order_candidates(ROUTES, stage=stage, pinned_provider="anthropic") == (CLAUDE,)
    assert order_candidates(ROUTES, stage=stage, pinned_provider="openai") == (CODEX,)
    assert order_candidates(ROUTES, stage=stage, pinned_provider="unknown") == ()


@pytest.mark.parametrize("stage", ["research", "build", "audit"])
def test_only_ordinary_stages_can_fail_over(stage):
    assert order_candidates(ROUTES, stage=stage, pinned_provider="anthropic") == ROUTES


def test_audit_is_independent_and_unknown_builder_fails_closed():
    assert order_candidates(ROUTES, stage="audit", pinned_provider="openai", built_by="openai") == (CLAUDE,)
    assert order_candidates(ROUTES, stage="audit", pinned_provider="anthropic", built_by="anthropic") == (CODEX,)
    assert order_candidates(ROUTES, stage="audit", pinned_provider="anthropic", built_by="unresolved") == ()


def test_allowance_requires_fresh_trusted_comparable_observations():
    readings = {"anthropic": Allowance(10, 990, "quota-service", "requests"),
                "openai-codex": Allowance(90, 995, "quota-service", "requests")}
    kwargs = dict(stage="build", pinned_provider="anthropic", now=1000,
                  trusted_sources=frozenset({"quota-service"}))
    assert order_candidates(ROUTES, allowances=readings, **kwargs) == (CODEX, CLAUDE)
    assert order_candidates(ROUTES, allowances=None, **kwargs) == ROUTES
    assert order_candidates(ROUTES, allowances={"anthropic": readings["anthropic"]}, **kwargs) == ROUTES
    for observation in [Allowance(90, 600, "quota-service", "requests"),
                        Allowance(90, 1001, "quota-service", "requests"),
                        Allowance(90, 995, "config", "requests"),
                        Allowance(90, 995, "quota-service", "tokens"),
                        Allowance(float("nan"), 995, "quota-service", "requests")]:
        assert order_candidates(ROUTES, allowances={**readings, "openai-codex": observation}, **kwargs) == ROUTES


@pytest.mark.parametrize("available", [None, -1, 0, 99, True, float("inf")])
def test_memory_unknown_or_insufficient_fails_closed(available):
    assert not memory_admits(available, 100)


def test_memory_threshold_and_bounded_aging():
    assert memory_admits(100, 100)
    assert not memory_admits(100, 0)
    assert aged_priority(0, 0, 900) > aged_priority(0, 899, 900)
    assert aged_priority(0, 0, 999999) == 20
    assert aged_priority(3, 100, 50) == 3


def test_real_runtime_mismatch_is_separate_from_requested(tmp_path):
    ledger = LaneLedger(tmp_path / "lanes.db")
    try:
        token, _ = _reserve(ledger, "a", (CLAUDE,))
        assert ledger.bind_worker(token, "dispatcher-fingerprint", "worker-fingerprint")
        assert not ledger.observe(token, "impostor", provider="openai-codex", model="actual-model")
        assert ledger.observe(token, "worker-fingerprint", provider="openai-codex", model="actual-model")
        row = ledger.snapshot()[0]
        assert (row["requested_provider"], row["requested_model"]) == ("anthropic", "claude-fixture")
        assert (row["observed_provider"], row["observed_model"]) == ("openai-codex", "actual-model")
    finally:
        ledger.close()


def test_router_maps_openai_to_subscription_not_api_billing(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[stages.build]\ncandidates = [{provider="openai", model="fixture"}]\n')
    assert load_router(path)["build"] == (Candidate("openai-codex", "fixture"),)
    path.write_text('[stages.build]\ncandidates = [{provider="other", model="fixture"}]\n')
    with pytest.raises(ValueError, match="unsupported route"):
        load_router(path)
