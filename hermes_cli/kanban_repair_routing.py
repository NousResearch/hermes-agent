"""Deterministic handoff from Task Orchestrator to repair profiles."""

from __future__ import annotations

from typing import Optional


# Ordered from narrowest scope to broadest so a PR audit is not swallowed by
# the generic cron or maintenance rules.
_REPAIR_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("market data", "authority freshness", "stale market"), "market-data-authority-auditor"),
    (("upstream pr", "upstream issue", "nousresearch", "upstream/main"), "hermes-upstream-auditor"),
    (("local pr ci", "audit pr", "audit this pull request"), "pr-local-ci-auditor"),
    (("live-trading", "paper-safety", "paper-safety", "broker safety"), "paper-safety-guardian"),
    (("alpaca", "broker credential", "broker validation"), "coding-expert"),
    (("dashboard.secret", "state.db", "retired-wal", "gateway restart"), "hermes-maintenance-steward"),
    (("cron", "scheduled job", "cron job"), "hermes-maintenance-steward"),
    (("rnd-", "dependency skew", "fuzz test", "permutation"), "rnd-adversarial-tester"),
)


def repair_profile_for_task(title: Optional[str], body: Optional[str]) -> Optional[str]:
    """Return the specialist profile for known repair scopes.

    Matching is intentionally conservative: an unknown task stays in triage
    for explicit scope rather than letting the orchestrator perform work.
    """
    text = f"{title or ''}\n{body or ''}".casefold()
    for needles, profile in _REPAIR_RULES:
        if any(needle in text for needle in needles):
            return profile
    return None
