"""Regression tests for the whitespace-padded silence marker.

What: a cron job asked to answer ``[SILENT]`` sometimes emits the token with padding *inside*
the brackets (``[ SILENT ]``, ``[\tSILENT\t]``). The canonical form collapsed whitespace runs but
kept those single spaces, so the padded marker went unrecognized: the autonomous lane delivered
the control token to the user verbatim, and ``context_from`` recorded it as the job's last
"non-silent" answer and re-injected it into every following run, where the model copied the
malformed form back. Proven RED on the base commit.
"""

from cron.scheduler_prompt import _archive_answer
from gateway.response_filters import (
    is_autonomous_silence_response,
    is_intentional_silence_response,
)

_PADDED_VARIANTS = (
    "[ SILENT ]",
    "[\tSILENT\t]",
    "[  SILENT  ]",
    "[ SILENT ]\n",
    "**[ SILENT ]**",
    "[ 静默 ]",
)


def test_padded_bracketed_marker_is_still_intentional_silence():
    """Whitespace inside the brackets carries no meaning: the marker still suppresses."""
    for variant in _PADDED_VARIANTS:
        assert is_intentional_silence_response(variant), variant


def test_autonomous_lane_suppresses_the_padded_marker_like_the_tight_one():
    """The cron/webhook lanes must agree with the interactive lane, or a padded token reaches
    the chat as content exactly like the leaked ``[ SILENT ]`` did."""
    for variant in _PADDED_VARIANTS:
        assert is_autonomous_silence_response(variant), variant
    assert is_autonomous_silence_response("2 deals filtered\n\n[ SILENT ]")
    assert is_autonomous_silence_response("2 deals filtered\n\n[ SILENT ]\n")


def test_padded_marker_opening_a_note_is_delivered_on_purpose():
    """Deliberate asymmetry with the tight form: the prefix rule stays byte-exact so a report
    that merely starts with a padded token is never swallowed. Losing an alert costs more than
    one stray line, and ``[SILENT] <note>`` already suppresses today."""
    assert is_autonomous_silence_response("[SILENT] No changes detected")
    assert not is_autonomous_silence_response("[ SILENT ] No changes detected")


def _framed_archive(response: str) -> str:
    """A run archive written with the writer's real length frames."""
    prompt = "revisa mi inbox"
    return (
        "# Cron Job: avisos-urgentes\n\n"
        f"**Prompt Characters:** {len(prompt)}\n## Prompt\n\n"
        f"{prompt}\n\n"
        f"**Response Characters:** {len(response)}\n## Response\n\n"
        f"{response}\n"
    )


def test_padded_marker_alone_is_not_a_reusable_archive_answer():
    """The leak's engine: an unrecognized marker was stored as a real answer and re-injected as
    "Your previous run's output", where the next run copied the malformed form back."""
    for response in ("[ SILENT ]", "[\tSILENT\t]", "[  SILENT  ]"):
        assert _archive_answer(_framed_archive(response)) is None, response
    # A real report keeps being reusable continuity.
    assert _archive_answer(_framed_archive("1 aviso alto: notaría, lunes 10:00")) == (
        "1 aviso alto: notaría, lunes 10:00"
    )


def test_prose_around_the_padded_marker_is_still_delivered():
    """Widening the matcher must not swallow a real report that merely quotes the token."""
    assert not is_intentional_silence_response("el cron respondió [ SILENT ] y hay 3 avisos")
    assert not is_autonomous_silence_response("el cron respondió [ SILENT ] y hay 3 avisos")
