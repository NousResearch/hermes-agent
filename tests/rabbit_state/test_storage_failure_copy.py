"""One copy table for "session storage unavailable / not written" across CLI, gateway and RPC.

Contract: the user gets a plain cause + the exact repair command, never the raw sqlite text as the
lead, plus a stable machine-readable `code` a GUI can key a "Run doctor" button on.
"""

import sqlite3

import pytest

from rabbit_state_user_copy import describe_storage_failure, storage_failure_details


@pytest.mark.parametrize(
    ("exc", "code", "command"),
    [
        (sqlite3.OperationalError("database is locked"), "storage_locked", "try again"),
        (sqlite3.OperationalError("attempt to write a readonly database"), "storage_readonly", "rabbit doctor --fix"),
        (sqlite3.DatabaseError("database disk image is malformed"), "storage_corrupt", "rabbit doctor --fix"),
        (OSError(28, "No space left on device"), "disk_full", "Free some disk space"),
        (None, "storage_unavailable", "rabbit doctor --fix"),
    ],
)
def test_each_cause_has_a_stable_code_and_an_action(exc, code, command):
    failure = describe_storage_failure(exc)
    assert failure.code == code
    assert command in failure.action
    assert failure.gloss
    # The raw sqlite wording never leaks into the user-facing gloss.
    assert "sqlite" not in failure.gloss.lower() and "OperationalError" not in failure.gloss


def test_details_line_is_flattened_and_bounded():
    details = storage_failure_details("line one\n   line two " + "x" * 400, limit=60)
    assert "\n" not in details and len(details) == 60 and details.endswith("...")


def test_action_command_is_pinned_to_the_failing_profile(monkeypatch, tmp_path):
    """The action names the profile whose store failed (multi-profile backends serve sessions
    whose state.db is not the process default; a bare ``rabbit`` follows active_profile)."""
    from rabbit_constants import profile_cli_selector

    monkeypatch.setenv("RABBIT_HOME", str(tmp_path / ".rabbit" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    for exc in (sqlite3.OperationalError("database is locked"), sqlite3.DatabaseError("malformed"), None):
        action = describe_storage_failure(exc).action
        assert "{profile_arg}" not in action
        assert f"`rabbit {selector}" in action and "`rabbit doctor" not in action and "`rabbit gateway" not in action
