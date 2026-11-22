"""Docs drift ratchet: every DEFAULT_CONFIG key and slash command is documented, or baselined.

The checker lives in scripts/ci/check_docs_drift.py; this test is how CI runs it (the Python lane
already carries the repo's dependencies). Only NEW gaps fail."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "check_docs_drift.py"


@pytest.fixture(scope="module")
def drift():
    spec = importlib.util.spec_from_file_location("check_docs_drift", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_docs_drift"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gaps(drift):
    return drift.find_gaps()


def test_no_new_undocumented_config_keys_or_slash_commands(drift, gaps):
    new, _ = drift.compare(gaps, drift._load_baseline())
    assert not any(new.values()), drift.failure_message(new)
    # Documenting a baselined key never fails a PR (that would punish the fix); the script
    # prints a note and `--update` prunes the baseline whenever someone gets to it.


def test_detects_each_documentation_form(drift, monkeypatch):
    docs = (
        "Set `agent.max_turns` to cap a run.\n"
        "Every `auxiliary.<task>.timeout` is in seconds; `display.pet.*` tunes the pet.\n"
        "Per model: `providers.<id>.models.<model>.timeout_seconds`.\n"
        "```yaml\ncompression:\n  threshold: 0.5\n```\n"
        "This sentence has emphasis on *agent.* things.\n"
    )
    monkeypatch.setattr(drift, "_docs_text", lambda: docs)
    monkeypatch.setattr(drift, "config_leaves", lambda: [
        "agent.max_turns", "auxiliary.vision.timeout", "display.pet.size",
        "compression.threshold", "agent.secret_knob", "auxiliary.vision.api_key",
        "providers.nous.models.hermes.timeout_seconds"])
    monkeypatch.setattr(drift, "slash_commands", list)
    assert drift.find_gaps()["config"] == ["agent.secret_knob", "auxiliary.vision.api_key"]
