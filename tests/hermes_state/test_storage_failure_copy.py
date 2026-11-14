"""One copy table for "session storage unavailable / not written" across CLI, gateway and RPC.

Contract: the user gets a plain cause + the exact repair command, never the raw sqlite text as the
lead, plus a stable machine-readable `code` a GUI can key a "Run doctor" button on.
"""

import sqlite3

import pytest

from hermes_state_user_copy import describe_storage_failure, storage_failure_details


@pytest.mark.parametrize(
    ("exc", "code", "command"),
    [
        (sqlite3.OperationalError("database is locked"), "storage_locked", "try again"),
        (sqlite3.OperationalError("attempt to write a readonly database"), "storage_readonly", "hermes doctor --fix"),
        (sqlite3.DatabaseError("database disk image is malformed"), "storage_corrupt", "hermes doctor --fix"),
        (OSError(28, "No space left on device"), "disk_full", "Free some disk space"),
        (None, "storage_unavailable", "hermes doctor --fix"),
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


def test_every_classifier_cause_has_a_copy_row():
    """Enumeration guard (gate t_0b006df6 RED-2): the table is keyed by the classifier's
    cause buckets and feeds every surface, so a new cause with no row silently renders the
    "could not be opened" fallback on CLI/TUI/Desktop/gateway. "unknown" IS the fallback and
    "disk" is split to disk_full at lookup; every other cause needs its own row."""
    from hermes_state_errors import PERSISTENCE_ERROR_CAUSES
    from hermes_state_user_copy import _STORAGE_FAILURES

    uncovered = (set(PERSISTENCE_ERROR_CAUSES) - {"unknown", "disk"}) - set(_STORAGE_FAILURES)
    assert not uncovered, f"causes without a copy row: {sorted(uncovered)}"


@pytest.mark.parametrize(
    "token",
    ["worker_receipt_timeout", "pending_receipt", "outbox_full", "owner_unavailable"],
)
def test_receipt_transport_cause_gets_the_receipt_row(token):
    """The four client-side receipt strings must land on the receipt row (storage_busy,
    outcome-neutral copy), never on the storage_unavailable "could not be opened" fallback
    that sent operators hunting disk on a healthy store (fleet incident 2026-11-13)."""
    failure = describe_storage_failure(token)
    assert failure.cause == "receipt_transport"
    assert failure.code == "storage_busy"
    assert "confirm" in failure.gloss.lower()
    # Outcome unknown (a lost ACK can have committed): the copy claims nothing either way.
    assert "not saved" not in failure.gloss.lower()
    # No disk/lock blame: the store may be perfectly healthy.
    for wrong in ("disk", "drive", "space", "locked", "holding the database"):
        assert wrong not in failure.gloss.lower(), wrong
    assert "doctor" in failure.action


def test_action_command_is_pinned_to_the_failing_profile(monkeypatch, tmp_path):
    """The action names the profile whose store failed (multi-profile backends serve sessions
    whose state.db is not the process default; a bare ``hermes`` follows active_profile)."""
    from hermes_constants import profile_cli_selector

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    for exc in (sqlite3.OperationalError("database is locked"), sqlite3.DatabaseError("malformed"), None):
        action = describe_storage_failure(exc).action
        assert "{profile_arg}" not in action
        assert f"`hermes {selector}" in action and "`hermes doctor" not in action and "`hermes gateway" not in action
