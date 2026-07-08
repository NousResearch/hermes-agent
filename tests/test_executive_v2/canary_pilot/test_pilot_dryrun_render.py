"""Pilot G7 — render_dry_run produces a deterministic snapshot.

The renderer is a pure function. Pilot confirms it is byte-stable
given the same state, and includes the key fields the operator uses
to gate ACK.
"""

from __future__ import annotations

import pytest

from agent.executive.dryrun import render_dry_run


class _StubState:
    """Minimal ObjectiveStateData stand-in for hermetic unit tests.

    Mirrors the attribute names render_dry_run reads via getattr/state.X.
    """

    def __init__(self, **kw):
        self.objective_id = kw.get("objective_id", "obj-pilot-render-001")
        self.fingerprint = kw.get("fingerprint", "pilot-fp")
        self.state_value = kw.get("state_value", "draft")
        self.normalized = kw.get("normalized")
        self.discovered = kw.get("discovered")
        self.contract = kw.get("contract")

    # render_dry_run reads state.state.value (Enum). Provide a shim.
    @property
    def state(self):
        class _S:
            value = self.state_value
        return _S()


def test_g7_render_minimal_state():
    state = _StubState()
    out = render_dry_run(state)
    assert "obj-pilot-render-001" in out
    assert "pilot-fp" in out
    assert "Executive v2 Dry-Run" in out


def test_g7_render_with_normalized_shows_success_criteria():
    state = _StubState(
        normalized={
            "goal_class": "READONLY",
            "risk_profile": "low",
            "estimated_complexity": "XS",
            "success_criteria": ["alpha", "beta"],
        }
    )
    out = render_dry_run(state)
    assert "READONLY" in out
    assert "Success Criteria" in out
    assert "1. alpha" in out
    assert "2. beta" in out


def test_g7_render_is_deterministic():
    a = render_dry_run(_StubState(objective_id="obj-X", fingerprint="fp-Y"))
    b = render_dry_run(_StubState(objective_id="obj-X", fingerprint="fp-Y"))
    assert a == b


def test_g7_render_changes_with_fingerprint():
    a = render_dry_run(_StubState(fingerprint="fp-A"))
    b = render_dry_run(_StubState(fingerprint="fp-B"))
    assert a != b