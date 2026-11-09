"""A completed run whose product build failed must not print the success headline.

The code is committed either way, so the update is not a failure — but the products the user
actually runs were not rebuilt, and ``✓ Update complete!`` (exit 0) is exactly what made a
broken desktop build pass as a good update: the next launch opened the previous bundle while the
summary claimed everything was fine.
"""

from __future__ import annotations

import pytest

from hermes_cli.update_cmd_maint import _unbuilt_completion_message, _update_complete_message


def test_unbuilt_headline_withholds_the_success_marker_and_keeps_the_version_span():
    message = _unbuilt_completion_message("✓ Update complete! (v0.23.0 → v0.23.1)")

    # _print_verified_update_completion only claims success for a ✓ line.
    assert not message.startswith("✓")
    assert message.startswith("⚠ Update partially complete")
    # What the user got is still worth showing, and the retry is named.
    assert "(v0.23.0 → v0.23.1)" in message
    assert "hermes update" in message
    assert "previous desktop app is still what launches" in message


def test_unbuilt_headline_without_a_version_span_stays_one_readable_line():
    message = _unbuilt_completion_message("✓ Update complete!")

    assert message.startswith("⚠ Update partially complete —")
    assert "\n" not in message


@pytest.mark.parametrize("completion", ["✓ Install complete!", "✓ Update complete! (v1 → v2)"])
def test_only_a_check_marked_headline_is_rewritten(completion: str):
    """The rewrite is the caller's decision; this pins what it must never keep."""
    assert _unbuilt_completion_message(completion).startswith("⚠")
