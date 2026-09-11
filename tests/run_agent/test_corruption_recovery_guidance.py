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
4. _send_session_db_warning_notifications() re-probes the active store (15 x 1s,
   off the event loop) before broadcasting, so a startup lock that clears while the
   messaging adapters connect never produces a warning about a failure that is
   already gone
"""

from pytest import fixture


def test_session_db_warning_rechecks_recovery_before_notifying():
    """A startup lock that has already cleared must not produce a stale user warning."""
    import asyncio

    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db_init_error = "OperationalError: database is locked"
    probes = []

    def recover(*, raise_on_error=False):
        probes.append(raise_on_error)
        runner._session_db_init_error = None
        return object()

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("a cleared lock must not be broadcast as a live failure")

    runner._open_session_db_for_active_scope = recover
    runner._send_home_channel_message = fail_if_called

    asyncio.run(runner._send_session_db_warning_notifications())

    assert probes == [False]
    assert runner._session_db_init_error is None


def test_session_db_warning_waits_for_transient_lock_recovery(monkeypatch):
    """A lock hidden behind the shared-store wrapper gets a recovery grace period."""
    import asyncio

    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db_init_error = "SessionStore SQLite handle unavailable"
    outcomes = iter((None, None, object()))

    def recover(*, raise_on_error=False):
        outcome = next(outcomes)
        if outcome is not None:
            runner._session_db_init_error = None
        return outcome

    async def no_sleep(_seconds):
        return None

    runner._open_session_db_for_active_scope = recover
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    asyncio.run(runner._send_session_db_warning_notifications())

    assert runner._session_db_init_error is None


def test_session_db_warning_broadcasts_when_the_lock_persists(monkeypatch):
    """A lock that never clears still warns the user — the probe must not swallow real failures."""
    import asyncio

    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db_init_error = "OperationalError: database is locked"
    runner._open_session_db_for_active_scope = lambda *, raise_on_error=False: None
    sent = []

    monkeypatch.setattr(
        runner, "_home_channel_transports", lambda: [("telegram", {}, "home-chat", object())]
    )

    async def capture_send(_platform, _home, _transport, message, _log_fmt):
        sent.append(message)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(runner, "_send_home_channel_message", capture_send)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    asyncio.run(runner._send_session_db_warning_notifications())

    assert sent, "a lock that survived the grace period must still be broadcast"
    assert "locked" in sent[0].lower() or "may not be persisted" in sent[0]


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


def test_gateway_corruption_banner_backups_dir_follows_hermes_home(monkeypatch, tmp_path):
    """The gateway broadcast's step 3 must name the live backups dir, not ~/.hermes (#104250).

    Pre-update backups live at ``<hermes_root>/backups`` (``hermes_cli/backup.py``); a
    custom-HERMES_HOME gateway must not be told to restore from a directory that never
    held its backups.
    """
    import asyncio

    import gateway.run as gateway_run

    custom_home = tmp_path / "custom-hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(custom_home / "profiles" / "research"))

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._session_db_init_error = "database disk image is malformed"
    sent = []
    monkeypatch.setattr(
        runner, "_home_channel_transports", lambda: [("telegram", {}, "home-chat", object())]
    )

    async def _capture_send(_platform, _home, _transport, message, _log_fmt):
        sent.append(message)

    monkeypatch.setattr(runner, "_send_home_channel_message", _capture_send)
    asyncio.run(runner._send_session_db_warning_notifications())

    assert sent, "warning must be broadcast to home channels"
    assert f"{custom_home / 'backups'}" in sent[0]
    assert "~/.hermes/backups" not in sent[0]


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
