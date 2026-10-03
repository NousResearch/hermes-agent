"""The self-improvement notice is delivered by the session's own surface.

``display.memory_notifications`` is documented per platform
(``display.platforms.<platform>.memory_notifications``), but the gateway's publication point used to
read the profile-wide key once, so a profile that serves the operator and a client could silence the
notice for all of its surfaces or for none. These tests pin the per-surface resolution, the refusal
of a process environment variable, and the wiring of the single publication point.
"""

from __future__ import annotations

import inspect

import pytest

from gateway import run_turn_runner
from gateway.display_config import OVERRIDEABLE_KEYS, resolve_memory_notifications


@pytest.mark.parametrize(
    "user_config, platform_key, expected",
    [
        # the per-surface declaration wins over the profile-wide value (the fix)
        (
            {"display": {"memory_notifications": "on", "platforms": {"discord": {"memory_notifications": "off"}}}},
            "discord",
            "off",
        ),
        # the same profile keeps reporting on its other surface
        (
            {
                "display": {
                    "memory_notifications": "on",
                    "platforms": {"discord": {"memory_notifications": "off"}},
                }
            },
            "telegram",
            "on",
        ),
        # the profile-wide value applies when the surface declares nothing about this key
        (
            {"display": {"memory_notifications": "off", "platforms": {"discord": {"tool_progress": "off"}}}},
            "discord",
            "off",
        ),
        # the platform default
        ({}, "discord", "on"),
        ({"display": None}, "discord", "on"),
        # verbose passes through, normalised
        ({"display": {"platforms": {"discord": {"memory_notifications": " VERBOSE "}}}}, "discord", "verbose"),
        # booleans are accepted the way the rest of the display settings accept them
        ({"display": {"memory_notifications": False}}, "discord", "off"),
        ({"display": {"memory_notifications": True}}, "discord", "on"),
        # another surface's declaration does not leak into this one
        ({"display": {"platforms": {"telegram": {"memory_notifications": "off"}}}}, "discord", "on"),
        # a CLI surface resolves too
        ({"display": {"platforms": {"cli": {"memory_notifications": "off"}}}}, "cli", "off"),
    ],
)
def test_resolution_is_per_surface(user_config, platform_key, expected):
    assert resolve_memory_notifications(user_config, platform_key) == expected


def test_environment_variable_takes_no_part(monkeypatch):
    """A process environment variable must not silence (or unsilence) the notice."""
    silenced = {"display": {"platforms": {"discord": {"memory_notifications": "off"}}}}
    for name in (
        "HERMES_MEMORY_NOTIFICATIONS",
        "MEMORY_NOTIFICATIONS",
        "HERMES_DISPLAY_MEMORY_NOTIFICATIONS",
        "HERMES_NOTIFICATIONS",
    ):
        monkeypatch.setenv(name, "on")
    assert resolve_memory_notifications(silenced, "discord") == "off"

    talkative = {
        "display": {"memory_notifications": "on", "platforms": {"discord": {"memory_notifications": "on"}}}
    }
    monkeypatch.setenv("HERMES_MEMORY_NOTIFICATIONS", "off")
    assert resolve_memory_notifications(talkative, "discord") == "on"


def test_memory_notifications_is_an_overrideable_display_setting():
    """The documented per-platform form must be a real display setting, not prose."""
    assert "memory_notifications" in OVERRIDEABLE_KEYS


def _wire_agent(user_config, source_platform=None):
    """Run `_wire_turn_agent_callbacks` over minimal fakes; return the agent (upstream's pattern)."""
    import types

    from gateway.run_turn_runner import TurnRunner

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
    if source_platform is not None:
        ctx.source = types.SimpleNamespace(platform=source_platform)
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
    agent = types.SimpleNamespace()
    TurnRunner._wire_turn_agent_callbacks(holder, agent, {}, None, None, None, False)
    return agent


def test_the_publication_point_honours_the_surfaces_own_declaration():
    """The publication point resolves the surface the turn came from, not the profile as a whole."""
    from gateway.config import Platform

    user_config = {
        "display": {"memory_notifications": "on", "platforms": {"discord": {"memory_notifications": "off"}}}
    }
    assert _wire_agent(user_config, Platform.DISCORD).memory_notifications == "off"
    assert _wire_agent(user_config, Platform.TELEGRAM).memory_notifications == "on"
    # a surface that carries no source resolves the profile-wide value (minimal fakes, oneshot turns)
    assert _wire_agent(user_config).memory_notifications == "on"
    assert _wire_agent({"display": {"memory_notifications": "verbose"}}).memory_notifications == "verbose"
