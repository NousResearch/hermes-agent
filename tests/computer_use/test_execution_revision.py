"""Phase 0B exit-gate cases for #112734: the internal ExecutionRevision contract.

The lease-epoch machinery itself lives in #108914 (still open), so the takeover/hand-back cases are
proven against the contract with an injected epoch source; the wiring tests prove the capture/input
boundaries admit and validate without changing observable behavior on main.
"""
import json
from types import SimpleNamespace

import pytest

from tools.computer_use import tool as tool_mod
from tools.computer_use.backend import CaptureResult
from tools.computer_use.execution_revision import (
    CAPTURE_DEPS,
    INPUT_DEPS,
    ExecutionRevision,
    ExecutionState,
    target_mismatch,
)


@pytest.fixture
def facts():
    store = {"display_identity": ":1", "backend_generation": 7, "control_epoch": 18}
    return store


@pytest.fixture
def state(facts):
    return ExecutionState("profile-a", lambda: facts)


@pytest.fixture
def clean_tool_state():
    tool_mod.reset_backend_for_tests()
    yield
    tool_mod.reset_backend_for_tests()


# ── contract: stale work fails closed ──────────────────────────────────────

def test_stale_revision_rejected_at_publish_boundary(state, facts):
    rev = state.admit()
    facts["backend_generation"] = 8  # daemon rebound mid-operation
    verdict = state.validate(rev, CAPTURE_DEPS)
    assert not verdict.ok
    assert verdict.reason == "backend_replaced"


def test_takeover_handback_epoch_mismatch_fails_closed(state, facts):
    rev = state.admit()  # admitted at epoch 18
    facts["control_epoch"] = 20  # human took over and handed back
    verdict = state.validate(rev, INPUT_DEPS)
    assert not verdict.ok
    assert verdict.reason == "control_epoch_changed"


def test_display_rebind_invalidates_capture(state, facts):
    rev = state.admit()
    facts["display_identity"] = ":2"
    verdict = state.validate(rev, CAPTURE_DEPS)
    assert not verdict.ok
    assert verdict.reason == "display_changed"


def test_target_changed_invalidates(state):
    rev = state.admit(app="kate", pid=100, window_id="w1")
    moved = ExecutionState("profile-a", lambda: {"pid": 200})
    verdict = moved.validate(rev, CAPTURE_DEPS)
    assert not verdict.ok
    assert verdict.reason == "target_changed"


def test_snapshot_stale_invalidates(state):
    rev = state.admit(snapshot_id="s00000183")
    fresh = ExecutionState("profile-a", lambda: {"snapshot_id": "s00000184"})
    verdict = fresh.validate(rev)
    assert not verdict.ok
    assert verdict.reason == "snapshot_stale"


def test_profile_mismatch_fails_closed(state):
    other = ExecutionState("profile-b", lambda: {})
    assert not other.validate(state.admit()).ok


# ── contract: unprovable facts never invent staleness ──────────────────────

def test_unprovable_facts_never_invalidate():
    state = ExecutionState("p", lambda: {"backend_generation": 3})
    rev = state.admit()  # display/control/snapshot unprovable -> None
    assert rev.display_identity is None and rev.control_epoch is None
    later = ExecutionState("p", lambda: {"display_identity": ":9", "backend_generation": 3,
                                         "control_epoch": 42, "snapshot_id": "s1"})
    assert later.validate(rev).ok  # nothing provable changed


def test_fresh_admit_validates(state):
    assert state.validate(state.admit()).ok


# ── target rule ────────────────────────────────────────────────────────────

def test_target_mismatch_rule():
    assert target_mismatch("kcalc", "kate") == "kcalc"
    assert target_mismatch("kate", "kate") is None
    assert target_mismatch("Google-chrome", "chrome") is None  # substring either way is fine
    assert target_mismatch("chrome", "Google-chrome") is None
    assert target_mismatch(None, "kate") is None  # unknown fails open
    assert target_mismatch("", "kate") is None


# ── wiring: capture fence ──────────────────────────────────────────────────

def _stub_backend(**kw):
    return SimpleNamespace(_last_app="kate", _last_target={"pid": 1, "window_id": "w1"},
                           capture=lambda **ckw: CaptureResult(mode="ax", width=800, height=600,
                                                               elements=[], **kw), **kw)


def test_capture_fence_passes_when_facts_stable(clean_tool_state):
    out = tool_mod._guarded_capture(_stub_backend(), None, mode="ax")
    assert json.loads(out)["mode"] == "ax"


def test_capture_fence_rejects_mid_capture_rebind(clean_tool_state, monkeypatch):
    calls = {"n": 0}
    real = tool_mod._current_revision_facts

    def flipping(sid):
        calls["n"] += 1
        facts = dict(real(sid))
        facts["backend_generation"] = 1 if calls["n"] == 1 else 2  # rebound between admit and validate
        return facts

    monkeypatch.setattr(tool_mod, "_current_revision_facts", flipping)
    out = json.loads(tool_mod._guarded_capture(_stub_backend(), None, mode="ax"))
    assert out["code"] == "revision_invalidated"
    assert out["invalidation_reason"] == "backend_replaced"


def test_backend_generation_bumps_on_reinstall(clean_tool_state):
    backend = _stub_backend()
    with tool_mod._backend_lock:
        tool_mod._install_backend("sid-x", backend, "standard")
        tool_mod._install_backend("sid-x", backend, "standard")
    assert tool_mod._current_revision_facts("sid-x")["backend_generation"] == 2


# ── wiring: input guard behavior unchanged ─────────────────────────────────

def test_input_mismatch_still_refused_via_revision(clean_tool_state):
    out = json.loads(tool_mod._dispatch(SimpleNamespace(_last_app="kcalc"), "type",
                                        {"text": "777", "app": "kate"}))
    assert out["code"] == "input_target_mismatch"
    assert "kcalc" in out["error"] and "kate" in out["error"]


def test_input_matching_app_reaches_backend(clean_tool_state):
    from tools.computer_use.backend import ActionResult
    backend = SimpleNamespace(_last_app="kate")
    backend.type_text = lambda text, **kw: ActionResult(ok=True, action="type_text")
    out = json.loads(tool_mod._dispatch(backend, "type", {"text": "hi", "app": "kate"}))
    assert out.get("ok") is True
