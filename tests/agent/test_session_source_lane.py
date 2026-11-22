"""A delegate child and a cron run keep their own session source.

Both lanes are intrinsic to the agent: the ambient gateway context (and the
``HERMES_SESSION_SOURCE`` export) belongs to the turn that spawned them, not to them.
"""

from __future__ import annotations

import pytest

from gateway.session_context import _UNSET, _VAR_MAP, clear_session_vars, set_session_vars
from agent.session_source import _SELF_SOURCED_PLATFORMS, session_source_for


@pytest.fixture(autouse=True)
def _reset_contextvars():
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    yield
    for var in _VAR_MAP.values():
        var.set(_UNSET)


def test_self_sourced_lanes_are_the_delegate_and_cron_lanes():
    """The set is the whole contract of the early return."""
    assert _SELF_SOURCED_PLATFORMS == frozenset({"subagent", "cron"})


# A delegate child and a cron agent carry their OWN lane, whatever surface the spawning turn is on:
# the ambient gateway context is live for the duration of a parent's turn, so inheriting it stamped
# children as 'desktop' and made them indistinguishable from real desktop conversations.
@pytest.mark.parametrize("lane", ["subagent", "cron"])
@pytest.mark.parametrize("ambient", ["desktop", "discord", "cli", "tui"])
def test_intrinsic_lane_wins_over_ambient_gateway_context(monkeypatch, lane, ambient):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source=ambient)
    try:
        assert session_source_for(lane) == lane
    finally:
        clear_session_vars(tokens)


def test_intrinsic_lane_wins_over_the_environment_fallback(monkeypatch):
    """The gateway exports HERMES_SESSION_SOURCE for the whole turn; the lane is not that surface."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    try:
        assert session_source_for("subagent") == "subagent"
        assert session_source_for("cron") == "cron"
    finally:
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)


def test_intrinsic_lane_survives_the_oneshot_transport_drop(monkeypatch):
    """The #112550 one-shot drop clears an inherited UI transport source. An intrinsic lane is the
    agent's own identity rather than an inherited label, so it must not be cleared along with it."""
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "desktop")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    try:
        assert session_source_for("subagent") == "subagent"
        assert session_source_for("cron") == "cron"
    finally:
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
        monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)


def test_ambient_ui_surface_still_defers_to_the_ambient_context(monkeypatch):
    """cli/tui/desktop/acp rows must keep tracking the session they were launched in."""
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="discord")
    try:
        assert session_source_for("desktop") == "discord"
    finally:
        clear_session_vars(tokens)


def test_ambient_ui_surface_defers_to_the_environment_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "discord")
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    try:
        assert session_source_for("tui") == "discord"
    finally:
        monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)


def test_platform_fallback_still_reaches_every_lane(monkeypatch):
    """With no ambient source at all the platform label is still the answer (unchanged behaviour)."""
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    for lane in ("subagent", "cron", "desktop", "cli"):
        assert session_source_for(lane) == lane
