"""Unit tests for Caduceus session state, prompt stack, and tiering.

Pure/offline — no model calls. Covers the behaviour a maintainer most needs to
trust: the mode is OFF by default and a strict no-op when off, the reminder
lifecycle fires exactly once on enter/exit + on cadence, and role-aware tiering
degrades to inherit when unset.
"""
from types import SimpleNamespace

import agent.caduceus as cad


# ---------------------------------------------------------------------------
# State + config
# ---------------------------------------------------------------------------

def test_default_state_is_off():
    st = cad.CaduceusState()
    assert st.enabled is False
    assert st.effort == "high"          # high, not xhigh (faster startup)
    assert st.budget_tokens is None
    assert st.is_split() is False


def test_state_from_config_defaults():
    st = cad.state_from_config({})
    assert st.enabled is False
    assert st.effort == "high"
    assert (st.router or {}).get("enabled") in (None, False)


def test_state_from_config_reads_section():
    st = cad.state_from_config({"caduceus": {
        "enabled": True, "effort": "xhigh",
        "worker": {"provider": "openrouter", "model": "x"},
        "router": {"enabled": True, "threshold": 0.8,
                   "candidates": [{"model": "m", "cost": 1}]},
    }})
    assert st.enabled is True
    assert st.effort == "xhigh"
    assert st.is_split() is True
    assert st.router["enabled"] is True and st.router["threshold"] == 0.8


def test_activate_deactivate_arms_reminders():
    st = cad.CaduceusState()
    st.activate()
    assert st.enabled and st._enter_pending and not st._exit_pending
    st.deactivate()
    assert not st.enabled and st._exit_pending and not st._enter_pending


def test_activate_is_idempotent():
    st = cad.CaduceusState()
    st.activate()
    st._enter_pending = False  # simulate the enter reminder already fired
    st.activate()              # second activate while already on
    assert st._enter_pending is False  # no re-arm


def test_summary_shape():
    st = cad.state_from_config({"caduceus": {"enabled": True}})
    s = st.summary()
    assert set(s) >= {"enabled", "effort", "orchestrator", "worker", "budget", "split"}


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------

def test_tier_for_role_off_returns_none():
    st = cad.CaduceusState(enabled=False)
    assert cad.tier_for_role(st, "leaf") is None
    assert cad.tier_for_role(None, "leaf") is None


def test_tier_for_role_worker_and_orchestrator():
    st = cad.CaduceusState(enabled=True)
    st.orchestrator = {"provider": "p1", "model": "big"}
    st.worker = {"provider": "p2", "model": "fast"}
    assert cad.tier_for_role(st, "orchestrator") == {"provider": "p1", "model": "big"}
    assert cad.tier_for_role(st, "leaf") == {"provider": "p2", "model": "fast"}


def test_tier_for_role_solo_falls_back_to_orchestrator():
    st = cad.CaduceusState(enabled=True)
    st.orchestrator = {"provider": "p1", "model": "big"}
    st.worker = {"provider": "", "model": ""}  # solo
    assert cad.tier_for_role(st, "leaf") == {"provider": "p1", "model": "big"}


def test_tier_for_role_unset_inherits():
    st = cad.CaduceusState(enabled=True)  # both tiers empty
    assert cad.tier_for_role(st, "leaf") is None
    assert cad.tier_for_role(st, "orchestrator") is None


def test_resolve_effort_config_maps_known_levels():
    assert cad.resolve_effort_config("high") is not None
    assert cad.resolve_effort_config("xhigh") is not None


# ---------------------------------------------------------------------------
# Reminder lifecycle
# ---------------------------------------------------------------------------

def test_enter_reminder_fires_once_then_cadence():
    st = cad.CaduceusState()   # off, like a fresh agent
    st.activate()              # off -> on arms the enter reminder
    st.turns_between_maintenance = 8
    # Turn 1: enter reminder.
    assert cad.compute_turn_reminder(st, 1) == cad.ENTER_REMINDER_FULL
    # Turns 2..8: nothing.
    for t in range(2, 9):
        assert cad.compute_turn_reminder(st, t) is None
    # Turn 9 (>= cadence since last): sparse maintenance.
    assert cad.compute_turn_reminder(st, 9) == cad.SPARSE_REMINDER


def test_exit_reminder_fires_once():
    st = cad.CaduceusState()
    st.activate()
    cad.compute_turn_reminder(st, 1)   # consume enter
    st.deactivate()
    assert cad.compute_turn_reminder(st, 2) == cad.EXIT_REMINDER
    assert cad.compute_turn_reminder(st, 3) is None  # off -> silent


def test_no_reminder_when_never_enabled():
    st = cad.CaduceusState()
    assert cad.compute_turn_reminder(st, 1) is None


def test_message_has_workflow_keyword():
    assert cad.message_has_workflow_keyword("please run a workflow")
    assert cad.message_has_workflow_keyword("use workflows here")
    assert not cad.message_has_workflow_keyword("just chat")
    assert not cad.message_has_workflow_keyword(None)


# ---------------------------------------------------------------------------
# Standing reminder injection
# ---------------------------------------------------------------------------

def _agent(enabled, tools=("todo",)):
    st = cad.CaduceusState(enabled=enabled)
    return SimpleNamespace(caduceus=st, valid_tool_names=tools)


def test_standing_reminder_off_returns_none():
    assert cad.standing_reminder_for_prompt(_agent(False)) is None


def test_standing_reminder_on_returns_text():
    r = cad.standing_reminder_for_prompt(_agent(True))
    assert r == cad.STANDING_REMINDER


def test_standing_reminder_skipped_when_todo_disabled():
    assert cad.standing_reminder_for_prompt(_agent(True, tools=("terminal",))) is None


def test_standing_reminder_injected_when_tool_list_unknown():
    # Empty/None tool list -> assume todo present, inject.
    assert cad.standing_reminder_for_prompt(_agent(True, tools=())) == cad.STANDING_REMINDER


def test_standing_reminder_is_devin_style_not_always_workflow():
    r = cad.STANDING_REMINDER
    for needle in ("RIGHT-SIZE THE PLAN", "ONE STEP AT A TIME", "PARALLELIZE",
                   "COMPLETION HONESTY", "VERIFY", "ESCALATE"):
        assert needle in r
    # The pivot: it must NOT push "run a workflow on every task".
    assert "every substantive task" not in r
