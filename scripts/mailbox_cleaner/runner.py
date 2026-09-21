from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .classify import classify
from .clients import MetadataClient, ReadAdapter
from .policy import AccountPolicy
from .render import render_report
from .state import StateStore


@dataclass(frozen=True, slots=True)
class ScheduledReport:
    text: str
    observed_count: int
    urgent_count: int


def run_scheduled(policies: Sequence[AccountPolicy], adapter: ReadAdapter, state: StateStore) -> ScheduledReport:
    """Read accounts sequentially. This path has no credential or transport setup."""
    classifications = []
    fresh_urgent = 0
    for policy in policies:
        client = MetadataClient(policy, adapter)
        for item in client.list_metadata(account=policy.account):
            result = classify(item)
            classifications.append(result)
            if result.urgent and state.record_urgent(item):
                fresh_urgent += 1
    return ScheduledReport(render_report(classifications), len(classifications), fresh_urgent)
