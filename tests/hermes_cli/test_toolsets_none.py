"""Behaviour contract for ``--toolsets none``, the explicit no-tools selection.

A ranking or batch worker hands untrusted text to the model with no human in the loop, so "no
tools" has to end up as a selection the caller made and the agent honours. ``None`` (the flag was
not given) must keep meaning "the caller's default", and an empty list must mean "nothing".

Regression for the career-ops integration: it refuses to run its ranking workers on Hermes because
there was no documented way to ask the CLI for a session without tools.
"""

import pytest

from hermes_cli.oneshot import _normalize_toolsets
from model_tools import get_tool_definitions
from toolsets import parse_toolset_selection


@pytest.mark.parametrize("raw", ["none", "NONE", " None ", ["none"], ("none",), "none,none"])
def test_none_token_selects_no_tools(raw):
    assert parse_toolset_selection(raw) == []


@pytest.mark.parametrize("raw", [None, "", [], ()])
def test_absent_flag_leaves_the_caller_default(raw):
    assert parse_toolset_selection(raw) is None


def test_none_token_is_dropped_from_a_mixed_selection():
    assert parse_toolset_selection("none,web") == ["web"]
    assert parse_toolset_selection("web, none") == ["web"]


def test_oneshot_normalization_keeps_the_empty_selection():
    assert _normalize_toolsets("none") == []
    assert _normalize_toolsets(None) is None


def test_the_empty_selection_registers_no_tools():
    assert get_tool_definitions(enabled_toolsets=[], quiet_mode=True) == []


def test_an_absent_selection_still_registers_tools():
    assert get_tool_definitions(enabled_toolsets=None, quiet_mode=True)
