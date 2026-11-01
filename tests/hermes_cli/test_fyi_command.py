"""Tests for the /fyi command — context-only steering (registry, framing, delivery)."""

from agent.prompt_builder import (
    FYI_CONTEXT_LEAD,
    STEER_MARKER_CLOSE,
    STEER_MARKER_OPEN,
    format_fyi_context,
    steer_user_row,
)
from hermes_cli.commands import COMMAND_REGISTRY, GATEWAY_KNOWN_COMMANDS, resolve_command


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestFyiRegistry:

    def test_registered_with_steer_family_busy_fields(self):
        cmd = resolve_command("fyi")
        assert cmd is not None and cmd.name == "fyi"
        assert cmd.busy_policy == "dispatch"
        assert cmd.busy_handler == "fyi"
        assert cmd.args_hint == "<context>"

    def test_dispatchable_on_gateway(self):
        assert "fyi" in GATEWAY_KNOWN_COMMANDS

    def test_no_alias_collision(self):
        names = [c.name for c in COMMAND_REGISTRY]
        assert names.count("fyi") == 1


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


class TestFyiFraming:

    def test_framing_wraps_payload_and_declares_context_only(self):
        framed = format_fyi_context("the API base URL moved to staging.example.com")
        assert "the API base URL moved to staging.example.com" in framed
        assert "NOT a new task" in framed

    def test_framing_rides_inside_the_steer_marker(self):
        """A mid-run /fyi is delivered through the steer channel: the framed text becomes the
        marker body, so the model sees provenance (marker) + contract (FYI lead) together."""
        row = steer_user_row(format_fyi_context("deployment windows are 02:00 UTC"))
        assert row["role"] == "user"
        assert row["content"].startswith(STEER_MARKER_OPEN)
        assert row["content"].rstrip().endswith(STEER_MARKER_CLOSE)
        assert FYI_CONTEXT_LEAD in row["content"]

    def test_empty_framing_still_carries_the_lead(self):
        assert FYI_CONTEXT_LEAD in format_fyi_context("")
