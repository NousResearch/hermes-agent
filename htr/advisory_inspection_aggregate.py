"""Task 29 — aggregate budget and completeness helpers (R5-10)."""

from __future__ import annotations

from typing import Iterable

_BUDGET_RANKS: tuple[tuple[str, str], ...] = (
    ("budget_control_json_exceeded", "budget_control_json_exceeded"),
    ("budget_path_exceeded", "budget_path_exceeded"),
    ("budget_url_exceeded", "budget_url_exceeded"),
    ("budget_references_exceeded", "budget_references_exceeded"),
    ("budget_links_exceeded", "budget_links_exceeded"),
    ("budget_artifact_exceeded", "budget_artifact_exceeded"),
    ("budget_manifest_exceeded", "budget_manifest_exceeded"),
)


def compute_aggregate_budget_status(
    unit_budgets: Iterable[str],
    *,
    hash_exceeded: bool = False,
    dir_exceeded: bool = False,
) -> str:
    """Aggregate ``budget_status`` from unit budgets (R5 ten-row table)."""
    for unit_status in unit_budgets:
        for token, aggregate in _BUDGET_RANKS:
            if unit_status == token:
                return aggregate
    if hash_exceeded:
        return "budget_aggregate_hash_exceeded"
    if dir_exceeded:
        return "budget_directory_entries_exceeded"
    return "budget_within_limits"


def compute_aggregate_completeness(
    *,
    blocked_untrusted_scope: bool = False,
    indeterminate_selector_unbound: bool = False,
    indeterminate_race: bool = False,
    partial_budget_exhausted: bool = False,
    partial_malformed: bool = False,
    partial_scope_missing: bool = False,
    partial_unreferenced_capped: bool = False,
    applicable_unit_count: int = 0,
    fully_complete_unit_count: int = 0,
    single_selector: bool = False,
) -> str:
    """Aggregate ``aggregate_completeness`` (R5 ten-row table)."""
    if blocked_untrusted_scope:
        return "aggregate_blocked_untrusted_scope"
    if indeterminate_selector_unbound:
        return "aggregate_indeterminate_selector_unbound"
    if indeterminate_race:
        return "aggregate_indeterminate_race"
    if partial_budget_exhausted:
        return "aggregate_partial_budget_exhausted"
    if partial_malformed:
        return "aggregate_partial_malformed"
    if partial_scope_missing:
        return "aggregate_partial_scope_missing"
    if partial_unreferenced_capped:
        return "aggregate_partial_unreferenced_capped"
    if single_selector:
        return "aggregate_not_applicable"
    if applicable_unit_count == 0:
        return "aggregate_empty"
    if fully_complete_unit_count >= 1:
        return "aggregate_complete"
    return "aggregate_empty"


def artifact_item_sort_key(
    run_id: str,
    task_id: str,
    attempt_id: str,
    entry_index: int,
) -> tuple[str, str, str, int]:
    """Deterministic aggregate item sort key (R5-11)."""
    return (run_id, task_id, attempt_id, entry_index)
