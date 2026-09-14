"""A profile config with `display: null` must not crash turn wiring.

`user_config.get("display", {})` returns None when the key is present but null
(bare `display:` in YAML), so the chained `.get("memory_notifications")` raised
AttributeError on every real gateway turn (#105674). Oneshot turns bypass
`_wire_turn_agent_callbacks`, which masked the regression during smoke tests.
"""

from __future__ import annotations

import types

import pytest

from gateway.run_turn_runner import TurnRunner


def _wire(user_config):
    """Run `_wire_turn_agent_callbacks` over minimal fakes; return the agent."""
    agent = types.SimpleNamespace()
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
        _status_adapter=None,
        session_key="",
        user_config=user_config,
        _thinking_enabled=False,
        agent_holder=[None],
        tools_holder=[None],
        process_task_id=None,
        process_baseline=None,
        run_generation=0,
    )
    holder = types.SimpleNamespace(
        _ctx=ctx,
        _runner=types.SimpleNamespace(
            _service_tier=None,
            _consume_pending_turn_sidecar_notes=lambda key: [],
        ),
        _make_bg_review_callbacks=lambda: (lambda message: None, lambda: None),
        _merge_turn_request_overrides=TurnRunner._merge_turn_request_overrides,
        _clarify_callback_sync=lambda *a, **k: None,
        _notice_callback_sync=lambda *a, **k: None,
        _attach_session_title_callback=lambda agent, ctx: None,
    )
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
    return agent


@pytest.mark.parametrize("user_config", [{"display": None}, {}])
def test_null_or_missing_display_falls_back_to_on(user_config):
    agent = _wire(user_config)
    assert agent.memory_notifications == "on"


def test_memory_notifications_setting_still_applies():
    agent = _wire({"display": {"memory_notifications": "verbose"}})
    assert agent.memory_notifications == "verbose"
