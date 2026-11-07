"""Contract cases for ``SessionRepository.advance_onboarding_run``: one onboarding event applied to the run stored at the compression tip and written back in one write.

Every expected step is ``session_rules.onboarding_flow.apply`` on the stored record.
"""

from __future__ import annotations

import copy

import pytest

from persistence.conversation.sessions import ONBOARDING_RUN_KEY, NewSession, OnboardingAdvance
from session_rules.onboarding_flow import COMPLETE, PHASE_SKIPPED, Answered, Done, End, Facts, Shown, apply, new_record

FACTS = Facts(os="linux", plugins=("p1",))
RECORD = new_record(FACTS, now=1.0, run_id="run-1")


def _rotated(store, root: str, tip: str, **fields) -> None:
    """``root`` ended by compression with ``tip`` as its continuation."""
    store.sessions.create(NewSession(root, "cli", **fields))
    store.sessions.end(root, "compression")
    store.sessions.create(NewSession(tip, "cli", parent_session_id=root))


def _stored(store, sid: str):
    return store.sessions.get_model_config_value(sid, ONBOARDING_RUN_KEY)


def test_an_event_is_applied_and_stored_at_the_compression_tip(backend, scope_a):
    """CONTRACT: the run at the tip takes ``apply`` with its stored facts at ``now``; the outcome names the tip, the run was active, and the tip stores the step's record while the root keeps its copy."""
    store = backend.open(scope_a)
    _rotated(store, "ob-root", "ob-tip", model_config={ONBOARDING_RUN_KEY: RECORD, "keep": 1})
    store.sessions.patch_model_config("ob-tip", {ONBOARDING_RUN_KEY: RECORD})
    event = Shown("name", tool_call_id="call-1")
    expected = apply(RECORD, event, Facts.from_stored(RECORD["facts"]), now=5.0)
    outcome = store.sessions.advance_onboarding_run("ob-root", event, now=5.0)
    assert outcome == OnboardingAdvance(session_id="ob-tip", was_active=True, step=expected)
    assert outcome.step.record["waiting"] == {"beat": "name", "tool_call_id": "call-1", "since": 5.0}
    assert _stored(store, "ob-tip") == expected.record
    assert store.sessions.get("ob-root").model_config == {ONBOARDING_RUN_KEY: RECORD, "keep": 1}


def test_successive_events_build_on_the_stored_record(backend, scope_a):
    """CONTRACT: each event reads what the previous one stored; an answer settles its beat and ``updated_at`` takes ``now``."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("ob-seq", "cli", model_config={ONBOARDING_RUN_KEY: RECORD}))
    store.sessions.advance_onboarding_run("ob-seq", Shown("name"), now=2.0)
    outcome = store.sessions.advance_onboarding_run("ob-seq", Answered("name", {"name": "Ada"}, said="Ada"), now=3.0)
    assert outcome.was_active is True
    assert outcome.step.result == {"beat": "name", "answer": {"name": "Ada"}, "said": "Ada", "next": "theme"}
    assert outcome.step.reminder == "after_name"
    stored = _stored(store, "ob-seq")
    assert stored == outcome.step.record
    assert (stored["beats"]["name"], stored["picks"]["name"], stored["updated_at"]) == ("done", "Ada", 3.0)


def test_ending_reports_the_run_was_active_once(backend, scope_a):
    """CONTRACT: the event that ends the run reports ``was_active=True`` and stores the ended record; a repeat end on the ended run reports ``was_active=False`` with the ended run's answer."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("ob-end", "cli", model_config={ONBOARDING_RUN_KEY: RECORD}))
    first = store.sessions.advance_onboarding_run("ob-end", End("skipped"), now=4.0)
    assert (first.was_active, first.step.record["phase"], first.step.record["ended_by"]) == (True, PHASE_SKIPPED, "skipped")
    assert first.step.reminder == "skip_all"
    again = store.sessions.advance_onboarding_run("ob-end", End("skipped"), now=6.0)
    assert again == OnboardingAdvance(session_id="ob-end", was_active=False, step=apply(first.step.record, End("skipped"), FACTS, now=6.0))
    assert again.step.result == {"skipped_all": True}
    assert _stored(store, "ob-end") == first.step.record


def test_done_completes_a_run_whose_build_was_shown(backend, scope_a):
    """CONTRACT: ``Done`` after the build card completes the run, as ``apply`` decides."""
    store = backend.open(scope_a)
    record = copy.deepcopy(RECORD)
    record["beats"].update({beat: "done" for beat in record["beats"]}, build="shown")
    store.sessions.create(NewSession("ob-done", "cli", model_config={ONBOARDING_RUN_KEY: record}))
    outcome = store.sessions.advance_onboarding_run("ob-done", Done(), now=7.0)
    assert (outcome.was_active, outcome.step.record["phase"], outcome.step.result) == (True, COMPLETE, {"completed": True})
    assert _stored(store, "ob-done")["phase"] == COMPLETE


def test_a_tip_without_a_run_record_writes_nothing(backend, scope_a):
    """CONTRACT: an absent key or a value that is not a record gives no step and leaves the value as it is."""
    store = backend.open(scope_a)
    store.sessions.create(NewSession("ob-none", "cli", model_config={"keep": 1}))
    assert store.sessions.advance_onboarding_run("ob-none", End(), now=1.0) == OnboardingAdvance(session_id="ob-none", was_active=False, step=None)
    assert store.sessions.get("ob-none").model_config == {"keep": 1}
    store.sessions.create(NewSession("ob-text", "cli", model_config={ONBOARDING_RUN_KEY: "not a record"}))
    assert store.sessions.advance_onboarding_run("ob-text", End(), now=1.0) == OnboardingAdvance(session_id="ob-text", was_active=False, step=None)
    assert _stored(store, "ob-text") == "not a record"


def test_a_missing_session_writes_nothing(backend, scope_a):
    """CONTRACT: without a row the outcome has no tip and no step, and no row appears."""
    store = backend.open(scope_a)
    assert store.sessions.advance_onboarding_run("ob-missing", End(), now=1.0) == OnboardingAdvance(session_id=None, was_active=False, step=None)
    assert store.sessions.get("ob-missing") is None


def test_an_error_in_apply_leaves_the_row_as_it_was(backend, scope_a):
    """CONTRACT: a record ``apply`` cannot read (no ``facts``) raises its error and the stored record stays."""
    store = backend.open(scope_a)
    broken = {key: value for key, value in RECORD.items() if key != "facts"}
    store.sessions.create(NewSession("ob-broken", "cli", model_config={ONBOARDING_RUN_KEY: broken}))
    with pytest.raises(KeyError):
        store.sessions.advance_onboarding_run("ob-broken", End(), now=1.0)
    assert _stored(store, "ob-broken") == broken
