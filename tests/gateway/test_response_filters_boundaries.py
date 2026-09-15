"""Boundary invariants for gateway response filtering."""

import pytest

from gateway.response_filters import is_autonomous_silence_response


@pytest.mark.parametrize(
    "response",
    (
        "**[SILENT]** Nothing new this tick.",
        "“[静默]” 没有新消息。",
    ),
)
def test_autonomous_marker_prefix_survives_edge_formatting(response):
    """Formatting around a leading control marker must not make a silent tick visible."""
    assert is_autonomous_silence_response(response)


@pytest.mark.parametrize(
    "response",
    (
        "**Status:** [SILENT] is the configured control token.",
        "“Silent retry succeeded after the transient error.”",
    ),
)
def test_edge_formatting_does_not_hide_substantive_prose(response):
    """Only a bracketed marker at the opening edge activates the autonomous rule."""
    assert not is_autonomous_silence_response(response)
