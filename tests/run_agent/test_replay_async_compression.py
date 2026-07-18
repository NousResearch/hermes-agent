"""CI wrapper for the offline replay harness (task 9).

Runs every builtin scenario of ``scripts/replay_async_compression.py`` and
enforces the structural gate: 100% of the invariants must hold for every
session shape (simple, tools, subagent, concurrent arrivals, already
compressed, in-place, legacy rotation, summariser failure, timeout, reset
during preparation) plus any sanitized fixture exports present in
``tests/fixtures/async_compression/``.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_async_compression",
        REPO_ROOT / "scripts" / "replay_async_compression.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("replay_async_compression", module)
    spec.loader.exec_module(module)
    return module


def test_replay_structural_gate_passes_for_every_scenario():
    replay = _load_replay_module()
    results = replay.run_all()
    names = {r.name for r in results}
    # The full plan matrix must actually run.
    expected = {
        "simple_operational",
        "tools_closed_groups",
        "subagent_delegation",
        "messages_during_execution",
        "already_compressed",
        "in_place_session",
        "legacy_rotation_session",
        "summariser_failure",
        "summariser_timeout",
        "reset_during_preparation",
    }
    assert expected.issubset(names)

    failures = {
        r.name: {check: ok for check, ok in r.checks.items() if not ok}
        for r in results if not r.ok
    }
    assert not failures, f"structural gate failed: {failures}"
