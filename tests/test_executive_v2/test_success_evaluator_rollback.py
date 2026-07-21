"""Phase 6 Success Evaluator rollback tests.

Rollback is intentionally narrow: it only deletes the Phase 6 state_meta
artifacts and leaves all Phase 1+5 source artifacts intact.
"""

from __future__ import annotations

from agent.executive.success_evaluator import (
    SuccessEvaluatorEngine,
    success_evaluator_rollback,
)
from agent.executive.types import SuccessStatus

from .test_success_evaluator import _seed_full


def test_rollback_deletes_only_phase6_artifacts(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)

    report = eng.evaluate("obj-1")
    assert report.status == SuccessStatus.SUCCESS
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    assert in_memory_storage.get_objective_success_report("obj-1") is not None

    assert eng.rollback("obj-1") is True

    assert in_memory_storage.get_objective_evaluation("obj-1") is None
    assert in_memory_storage.get_objective_success_report("obj-1") is None
    # Phase 1+5 state is preserved.
    assert in_memory_storage.get_objective_plan("obj-1") is not None
    assert in_memory_storage.get_objective_policy_decision("obj-1") is not None
    assert in_memory_storage.get_objective_approval_request("obj-1") is not None
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is not None
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is not None


def test_rollback_is_idempotent_when_no_phase6_artifacts_exist(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)

    assert eng.rollback("obj-1") is False
    assert eng.rollback("obj-1") is False

    # Source artifacts remain untouched even when rollback has nothing to do.
    assert in_memory_storage.get_objective_worker_dispatch("obj-1") is not None
    assert in_memory_storage.get_objective_kanban_apply("obj-1") is not None


def test_module_level_rollback_wrapper(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)
    eng.evaluate("obj-1")

    assert success_evaluator_rollback("obj-1", storage=in_memory_storage) is True
    assert in_memory_storage.get_objective_evaluation("obj-1") is None
    assert in_memory_storage.get_objective_success_report("obj-1") is None


def test_rollback_after_persist_then_re_evaluate(in_memory_storage):
    _seed_full(in_memory_storage)
    eng = SuccessEvaluatorEngine(state_storage=in_memory_storage)

    first = eng.evaluate("obj-1")
    assert eng.rollback("obj-1") is True
    second = eng.evaluate("obj-1")

    assert first.status == second.status
    assert second.successful_tasks == 2
    assert in_memory_storage.get_objective_evaluation("obj-1") is not None
    assert in_memory_storage.get_objective_success_report("obj-1") is not None
