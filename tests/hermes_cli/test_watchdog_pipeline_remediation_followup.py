"""Regression tests for the ready_breakdown/provider-meter follow-up fixes."""
from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone
from pathlib import Path

from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_provider_meter as pm
from hermes_cli.kanban_provider_meter import ProviderMeterCandidate, remediate_candidate


def test_ready_breakdown_declares_reserved_review_and_dependency_buckets() -> None:
    result = kb.DispatchResult()

    assert result.ready_breakdown["review"] == []
    assert result.ready_breakdown["dependency"] == []
    assert "reserved" in inspect.getsource(kb.DispatchResult)


def test_provider_meter_fallback_does_not_self_assign_except_binding() -> None:
    source = Path(__file__).parents[2] / "hermes_cli" / "kanban_provider_meter.py"
    text = source.read_text()
    assert "_account_usage_import_err = _account_usage_import_err" not in text
    tree = ast.parse(text)
    target = "_account_usage_import_err"

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for assigned in node.targets:
                if isinstance(assigned, ast.Name) and assigned.id == target:
                    assert not (
                        isinstance(node.value, ast.Name) and node.value.id == target
                    ), "exception binding must not be self-assigned after the except block"


def test_remediate_candidate_uses_live_usage_lookup(monkeypatch) -> None:
    snapshot = AccountUsageSnapshot(
        provider="anthropic",
        source="test",
        fetched_at=datetime.now(timezone.utc),
        windows=(AccountUsageWindow(label="window", used_percent=20),),
    )
    calls: list[str] = []

    def fetch(provider: str, *, api_key=None, base_url=None):
        calls.append(provider)
        return snapshot

    monkeypatch.setattr(pm.account_usage, "fetch_account_usage", fetch)
    candidates = [ProviderMeterCandidate("anthropic", "claude-sonnet-4-6")]

    assert remediate_candidate(candidates) == candidates[0]
    assert calls == ["anthropic"]
