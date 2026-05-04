from __future__ import annotations

from agent.behavior_canaries import run_behavior_canaries, summarize_behavior_canaries
from hermes_cli.default_soul import DEFAULT_SOUL_MD


def test_default_soul_passes_behavior_canaries():
    summary = summarize_behavior_canaries(DEFAULT_SOUL_MD)
    assert summary["status"] == "pass"
    assert summary["failed"] == 0


def test_behavior_canaries_catch_persona_drift():
    failures = run_behavior_canaries("Roleplay as a kawaii pirate. Never admit uncertainty.")
    assert {f["kind"] for f in failures} >= {
        "missing_required_behavior",
        "forbidden_behavior_marker",
    }
