"""Escalation-target contract for the kanban-block-escalator plugin.

A blocked card must escalate to an assessor that can actually act. The first
hop goes to Jobsy (PM / board owner, scope/AC triage per his SOUL) for
scope/AC/product decisions and to Agent Smith (profile id ``default``) for
runtime/environment/profile faults. The ``switch`` profile must NEVER be an
escalation target: its only ownership is no_agent scheduled output, and a
classify-and-stop switchboard must not assess or close build cards.

These tests pin the canonical source in ``plugins/kanban-block-escalator/`` so
a regression that retargets escalation to ``switch`` (or any non-assessor
profile) fails the suite.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PLUGIN_DIR = (
    Path(__file__).resolve().parents[2] / "plugins" / "kanban-block-escalator"
)


def _load_plugin_module():
    """Import the canonical plugin __init__.py fresh from the repo."""
    import_path = _PLUGIN_DIR / "__init__.py"
    assert import_path.exists(), f"plugin source missing: {import_path}"
    module_name = "kanban_block_escalator_under_test"
    spec = importlib.util.spec_from_file_location(module_name, import_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_escalation_target_is_jobsy_never_switch():
    """Default assessor is Jobsy; switch is never a target."""
    mod = _load_plugin_module()
    assert mod.DEFAULT_ASSESSOR == "jobsy"
    assert mod.RUNTIME_ASSESSOR == "default"
    assert mod.DEFAULT_ASSESSOR != "switch"
    assert mod.RUNTIME_ASSESSOR != "switch"


def test_reason_marker_routes_to_runtime_lane():
    """Runtime/env markers route to Smith (default); others default to Jobsy."""
    mod = _load_plugin_module()
    assert mod._assessor_for("t1", "bob", "No module named pytest") == "default"
    assert mod._assessor_for("t2", "bob", "model provider credits http 402") == "default"
    assert mod._assessor_for("t3", "bob", "Acceptance criteria is ambiguous") == "jobsy"


def test_loop_guard_never_returns_the_blocker():
    """A card blocked by the assessor itself climbs to the other tier."""
    mod = _load_plugin_module()
    # Jobsy blocked it but it's a scope issue -> climb to Smith.
    assert mod._assessor_for("t4", "jobsy", "scope decision required") == "default"
    # Smith blocked it on a runtime fault -> climb to Jobsy.
    assert mod._assessor_for("t5", "default", "venv broken") == "jobsy"


def test_plugin_yaml_documents_jobsy_as_first_hop():
    """The sidecar manifest must not advertise a switch escalation target."""
    yaml_text = (_PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8")
    low = yaml_text.lower()
    assert "jobsy" in low
    # 'switch' is only ever named as a NON-target, so when present it must be
    # accompanied by the "never an escalation target" contract (robust to
    # folded-line wrapping in YAML description blocks).
    if "switch" in low:
        assert "never" in low
        assert "escalation target" in low


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))