"""Validation helpers for autonomous contract packages."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from pydantic import ValidationError

from .models import Contract


class ContractValidationError(ValueError):
    """Raised when a contract is syntactically valid data but semantically unsafe."""


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)



def _assert_no_dependency_cycles(sprint_dependencies: dict[str, list[str]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(sprint_id: str, stack: list[str]) -> None:
        if sprint_id in visited:
            return
        if sprint_id in visiting:
            cycle = stack[stack.index(sprint_id):] + [sprint_id]
            raise ContractValidationError(f"sprint dependency cycle: {' -> '.join(cycle)}")
        visiting.add(sprint_id)
        for dep in sprint_dependencies.get(sprint_id, []):
            visit(dep, stack + [dep])
        visiting.remove(sprint_id)
        visited.add(sprint_id)

    for sprint_id in sprint_dependencies:
        visit(sprint_id, [sprint_id])


def _raise_if_duplicates(label: str, values: Iterable[str]) -> None:
    dupes = _duplicates(values)
    if dupes:
        raise ContractValidationError(f"duplicate {label}: {', '.join(dupes)}")


def validate_contract(data: dict[str, Any] | Contract) -> Contract:
    """Return a validated contract or raise a deterministic validation error.

    Pydantic enforces field-level shape; this function enforces graph integrity:
    unique ids, sprint dependencies, gate references, section references, and the
    chain-of-command invariants required before autonomous execution.
    """

    try:
        contract = data if isinstance(data, Contract) else Contract.model_validate(data)
    except ValidationError as exc:
        raise ContractValidationError(str(exc)) from exc

    _raise_if_duplicates("section ids", (section.id for section in contract.sections))
    _raise_if_duplicates("sprint ids", (sprint.id for sprint in contract.sprints))
    _raise_if_duplicates("gate ids", (gate.id for gate in contract.gates))

    section_ids = {section.id for section in contract.sections}
    sprint_ids = {sprint.id for sprint in contract.sprints}
    gate_ids = {gate.id for gate in contract.gates}

    for sprint in contract.sprints:
        if sprint.section not in section_ids:
            raise ContractValidationError(f"sprint {sprint.id} references missing section {sprint.section}")
        for dep in sprint.dependsOn:
            if dep not in sprint_ids:
                raise ContractValidationError(f"sprint {sprint.id} depends on missing sprint {dep}")
            if dep == sprint.id:
                raise ContractValidationError(f"sprint {sprint.id} depends on itself")
        for gate_id in sprint.gates:
            if gate_id not in gate_ids:
                raise ContractValidationError(f"sprint {sprint.id} references missing gate {gate_id}")
        _raise_if_duplicates(f"acceptance ids in sprint {sprint.id}", (ac.id for ac in sprint.acceptance))

    _assert_no_dependency_cycles({sprint.id: list(sprint.dependsOn) for sprint in contract.sprints})

    for gate in contract.gates:
        for blocked in gate.blocksSprintIds:
            if blocked not in sprint_ids:
                raise ContractValidationError(f"gate {gate.id} blocks missing sprint {blocked}")
        if gate.type == "human_approval" and gate.owner == "benjamin":
            if contract.chainOfCommand.benjaminEscalationAllowedOnlyBy != "galt":
                raise ContractValidationError("Benjamin-owned human approval gates must route through Galt")

    if contract.chainOfCommand.escalationPolicy != "galt_first":
        raise ContractValidationError("escalationPolicy must be galt_first")
    if contract.chainOfCommand.benjaminEscalationAllowedOnlyBy != "galt":
        raise ContractValidationError("Benjamin escalation must be allowed only by Galt")
    if "direct_benjamin_escalation" not in contract.authority.forbidden:
        raise ContractValidationError("authority.forbidden must include direct_benjamin_escalation")
    if contract.kanban.enabled and contract.kanban.sourceOfTruth != "ledger":
        raise ContractValidationError("Kanban may only be a projection; sourceOfTruth must be ledger")

    return contract
