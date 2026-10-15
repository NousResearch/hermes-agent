"""The adapter's follow-up fence cancels the review of the turn it was registered for.

``TurnRunner._wire_turn_agent_callbacks`` registers a per-turn, non-blocking review fence on
the adapter (``register_followup_review_cancel``), and the adapter fires it on the gateway
loop inside every accepted follow-up's queue mutation. That thread is outside the turn's
profile scope -- under ``multiplex_profiles`` its ambient profile is the launch profile's --
while the turn's review is published under the turn's own ``(profile, session)``
(``_GatewayReviewAdmission``), a session id another profile may hold a review under
(``agent/review_admission.py``). The fence names the turn's owner instead of resolving the
registry key from the loop thread's profile, so it fences this turn's review, never the launch
profile's review on the same session id.
"""

from __future__ import annotations

import threading
import types
from typing import cast

import pytest

from agent import review_admission
from agent.background_review import prepare_background_review_run
from gateway.platforms.base import BasePlatformAdapter
from gateway.run_turn_review import _GatewayReviewAdmission
from gateway.run_turn_runner import TurnRunner

SESSION_ID = "shared-session"
SESSION_KEY = "session-key"
LAUNCH_PROFILE = "profile-launch"
ROUTED_PROFILE = "profile-routed"


@pytest.fixture(autouse=True)
def _clean_registry():
    with review_admission._lock:
        review_admission._review_runs.clear()
    yield
    with review_admission._lock:
        review_admission._review_runs.clear()


def _agent():
    return types.SimpleNamespace(
        session_id=SESSION_ID, _background_review_lock=threading.Lock()
    )


def _adapter(monkeypatch):
    monkeypatch.setattr(BasePlatformAdapter, "__abstractmethods__", frozenset())
    adapter = object.__new__(BasePlatformAdapter)
    adapter._post_delivery_callbacks = {}
    return adapter


def _wire(adapter, agent, admission):
    """``TurnRunner._wire_turn_agent_callbacks`` for one turn whose context carries
    ``admission`` (None: a turn wired without gateway ownership)."""
    ctx = types.SimpleNamespace(
        progress_callback=None,
        native_tool_start_callback=None,
        voice_ack_callback=None,
        _voice_ack_guild=[None],
        _native_slack_task_cards=False,
        native_tool_complete_callback=None,
        _step_callback_sync=None,
        _hooks_ref=types.SimpleNamespace(loaded_hooks=[]),
        _status_callback_sync=None,
        _event_callback_sync=None,
        _status_adapter=adapter,
        _status_thread_metadata={},
        session_key=SESSION_KEY,
        user_config={},
        source=types.SimpleNamespace(platform="telegram"),
        mute_notification_reply=False,
        _thinking_enabled=False,
        agent_holder=[None],
        tools_holder=[None],
        process_task_id=None,
        process_baseline=None,
        run_generation=1,
        voice_turn=False,
        gateway_review_admission=admission,
    )
    holder = types.SimpleNamespace(
        _ctx=ctx,
        _approval_owner=None,
        _runner=types.SimpleNamespace(
            _service_tier=None,
            _consume_pending_turn_sidecar_notes=lambda key: [],
            _overflow_queue=lambda _key: [],
        ),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None,
        _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    TurnRunner._wire_turn_agent_callbacks(
        cast(TurnRunner, holder), agent, {}, None, None, None, False
    )


def _two_profiles_hold_a_review_on_the_same_session():
    """The launch profile's review and this turn's, both on ``SESSION_ID``."""
    launch_agent, agent = _agent(), _agent()
    launch_run = prepare_background_review_run(
        launch_agent, session_id=SESSION_ID, profile_key=LAUNCH_PROFILE
    )
    own_run = prepare_background_review_run(
        agent, session_id=SESSION_ID, profile_key=ROUTED_PROFILE
    )
    assert launch_run is not None and own_run is not None
    return agent, own_run, launch_run


def test_followup_fence_cancels_only_the_turns_own_review(monkeypatch):
    adapter = _adapter(monkeypatch)
    agent, own_run, launch_run = _two_profiles_hold_a_review_on_the_same_session()
    _wire(adapter, agent, _GatewayReviewAdmission(SESSION_ID, ROUTED_PROFILE, 0))
    # The fence fires on the gateway loop, whose ambient profile is the launch profile's.
    monkeypatch.setattr(review_admission, "current_profile_key", lambda: LAUNCH_PROFILE)

    assert adapter.apply_followup_queue_mutation(SESSION_KEY, lambda: True) is True

    assert own_run.cancel_requested.is_set(), "the turn's own review was left un-fenced"
    assert not launch_run.cancel_requested.is_set(), (
        "the launch profile's review was fenced by another profile's follow-up"
    )


def test_followup_fence_without_gateway_ownership_keys_by_the_active_turns_profile(
    monkeypatch,
):
    """A turn wired without a gateway admission (an unowned in-band follow-up) still names
    the profile its foreground registration carries on the agent."""
    adapter = _adapter(monkeypatch)
    agent, own_run, launch_run = _two_profiles_hold_a_review_on_the_same_session()
    agent._active_turn_profile_key = ROUTED_PROFILE
    _wire(adapter, agent, None)
    monkeypatch.setattr(review_admission, "current_profile_key", lambda: LAUNCH_PROFILE)

    assert adapter.apply_followup_queue_mutation(SESSION_KEY, lambda: True) is True

    assert own_run.cancel_requested.is_set()
    assert not launch_run.cancel_requested.is_set()
