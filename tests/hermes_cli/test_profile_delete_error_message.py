"""Regression for #141911: a profile delete that dies on foreign-owned files (root-owned Docker
sandbox mount points) names the one command that recovers, instead of surfacing a bare Errno 13
into the Desktop's delete dialog."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.profiles_remove import removal_error_message


def test_permission_errors_name_the_recovery_command():
    profile_dir = Path("/srv/hermes/profiles/x")
    message = removal_error_message(profile_dir, PermissionError(13, "Permission denied"))
    assert "Could not remove profile directory" in message
    assert "sudo chown -R" in message and str(profile_dir) in message


def test_other_errors_stay_plain():
    message = removal_error_message(Path("/srv/x"), OSError(39, "Directory not empty"))
    assert "Directory not empty" in message and "chown" not in message
