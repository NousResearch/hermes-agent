"""Shared benchmark fixtures.

`compressor_pair` returns two identically-configured compressors except
for the qwen_aware flag block. Every benchmark that A/B-compares uses
this fixture so the only variable is the flag set.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


@pytest.fixture
def compressor_pair():
    """Return ``(baseline, with_flags)`` compressor pair.

    Both have:
      - identical context_length (256K, mocked)
      - identical threshold_percent (0.50)
      - identical protect_first_n / protect_last_n / target_ratio
      - quiet_mode=True

    Difference: ``with_flags`` has every qwen_aware flag enabled.
    """
    def _make(**kw):
        defaults = dict(
            model="bench/qwen-instruct",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            summary_target_ratio=0.20,
            quiet_mode=True,
            base_url="",
            api_key="",
            config_context_length=262_144,
            provider="bench",
            api_mode="chat_completions",
        )
        defaults.update(kw)
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=262_144,
        ):
            return ContextCompressor(**defaults)

    baseline = _make()
    with_flags = _make(
        qwen_aware_enabled=True,
        dedup_operations=True,
        anchor_first_assistant=True,
        threshold_absolute_max=80_000,
        message_threshold=200,
        turn_threshold=30,
    )
    return baseline, with_flags


@pytest.fixture
def stub_summarizer(monkeypatch):
    """Patch _generate_summary to return a deterministic short string.

    Tier-1 benchmarks should never fire a real LLM call. This fixture
    makes the summarizer deterministic so token-delta math is reproducible.
    """
    def _stub(self, turns, focus_topic=None):
        return f"## Summary\n{len(turns)} turns compressed."
    monkeypatch.setattr(ContextCompressor, "_generate_summary", _stub)


def pytest_sessionfinish(session, exitstatus):
    from tests.agent.benchmarks._report import _RESULTS, emit_report
    if not _RESULTS:
        return
    out = Path(__file__).resolve().parents[3] / (
        "docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md"
    )
    emit_report(out)
    print(f"\n[bench] report written to {out}")
