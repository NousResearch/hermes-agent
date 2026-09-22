"""Deterministic Kanban handoff ownership contracts."""

from hermes_cli.kanban_repair_routing import repair_profile_for_task


def test_known_repair_scopes_bypass_router_profiles():
    assert repair_profile_for_task(
        "Verify market data authority freshness",
        "Check stale market-data provenance and freshness.",
    ) == "market-data-authority-auditor"
    assert repair_profile_for_task(
        "Upstream PR audit",
        "Compare NousResearch upstream/main and the current PR head.",
    ) == "hermes-upstream-auditor"
    assert repair_profile_for_task(
        "Local PR CI audit",
        "Run the exact local CI audit for this pull request.",
    ) == "pr-local-ci-auditor"


def test_unknown_scope_stays_unassigned_for_explicit_triage():
    assert repair_profile_for_task("Investigate an unclear issue", "Needs more scope.") is None
