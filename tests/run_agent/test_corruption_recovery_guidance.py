"""Regression tests for #88235 — state.db corruption must surface a warning
to the user's messaging platform, not stay silently in the logs.

When SessionDB init fails at gateway startup (corruption, NFS/SMB locks,
disk errors), the gateway sets _session_db = None and logs a warning — but
the user never sees it.  Messages may flow but nothing is persisted, and the
user only discovers the breakage when /resume or session_search comes back
empty.

The fix adds:
1. _session_db_init_error attribute on GatewayRunner, set when init fails
2. _send_session_db_warning_notifications() — broadcasts a recovery-guidance
   message to all home channels after the gateway connects
3. Improved "corrupt" cause wording in _format_turn_completion_explanation
   with the full recovery path (hermes doctor, sqlite3 .recover, backups)
"""

from pytest import fixture


def test_format_turn_completion_corrupt_includes_recovery_options():
    """The 'corrupt' persistence cause must list all recovery options."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "corrupt"
    )
    assert "hermes doctor" in explanation
    assert ".recover" in explanation
    assert "backups" in explanation
    assert "Freeing disk space will not help" in explanation


def test_format_turn_completion_corrupt_never_names_the_live_db():
    """The 'corrupt' cause must not direct a raw sqlite3 shell at the live DB.

    #100368 forensics: the system sqlite3 CLI on Debian/Ubuntu (3.45.1/
    3.46.1, below the 3.51.x WAL-reset fix) unlinks the live WAL/SHM pair
    when pointed at a live state.db, splitting the store into two
    generations whose acknowledged writes vanish. The guidance that ships
    in the corruption banner must be the snapshot-copying
    `hermes sessions recover` lane.
    """
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "corrupt"
    )
    assert "sessions recover" in explanation
    assert 'sqlite3 ~/.hermes/state.db ".recover"' not in explanation
    # The replacement guidance names the safe command.
    assert "hermes sessions recover --source" in explanation


def test_format_turn_completion_disk_still_advises_space():
    """The 'disk' cause still gives disk-space advice (unchanged)."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "disk"
    )
    assert "free some space" in explanation


def test_format_turn_completion_locked_still_advises_retry():
    """The 'locked' cause still advises retrying (unchanged)."""
    from run_agent import AIAgent

    explanation = AIAgent._format_turn_completion_explanation(
        "session_persistence_failed", "locked"
    )
    assert "busy" in explanation
    assert "send it again" in explanation
