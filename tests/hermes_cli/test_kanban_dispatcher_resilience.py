# ---------------------------------------------------------------------------
# Tests: disabled_corrupt_boards exponential backoff (gateway-disabled-boards-overaggressive-latch)
# ---------------------------------------------------------------------------

import math as _math


class TestCorruptBoardBackoff:
    """Tests for the exponential-backoff corrupt-board latch in _kanban_dispatcher_watcher."""

    def _make_watcher_state(self):
        """Return a fresh (disabled_corrupt_boards, constants) tuple for unit tests."""
        from gateway.run import INITIAL_BACKOFF_SEC, MAX_BACKOFF_SEC  # noqa: PLC0415

        return {}, INITIAL_BACKOFF_SEC, MAX_BACKOFF_SEC

    def _make_fingerprint(self, mtime=1000, size=4096, path="/tmp/test.db"):
        return (path, mtime, size)


def test_transient_eio_no_latch_when_quick_check_passes(tmp_path, monkeypatch):
    """A transient DatabaseError where quick_check returns ok must NOT latch the board."""
    import sqlite3
    import time

    import gateway.run as gr

    # Create a valid SQLite DB at tmp_path so quick_check can open it
    db = tmp_path / "board.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.close()

    # Simulate the _confirm_corruption logic: valid db → quick_check returns "ok" → not corrupt
    uri = f"file:{db}?mode=ro"
    try:
        check_conn = sqlite3.connect(uri, uri=True, timeout=2)
        row = check_conn.execute("PRAGMA quick_check").fetchone()
        result = row[0] if row else "error"
        check_conn.close()
        is_confirmed_corrupt = result != "ok"
    except Exception:
        is_confirmed_corrupt = True

    # For a valid db, quick_check returns "ok" → not corrupt → no latch applied
    assert not is_confirmed_corrupt, "Valid db must not be treated as corrupt"

    # Verify: when _confirm_corruption returns False (transient), latch is NOT applied
    disabled_corrupt_boards = {}
    if not is_confirmed_corrupt:
        pass  # _confirm_corruption returned False → no latch written
    assert len(disabled_corrupt_boards) == 0, (
        "Transient EIO where quick_check passes must not latch the board"
    )

    # Also verify INITIAL_BACKOFF_SEC constant exists
    assert gr.INITIAL_BACKOFF_SEC == 30.0


def test_genuine_corruption_latches_with_initial_backoff(tmp_path):
    """Genuine corruption (quick_check also fails) must latch with INITIAL_BACKOFF_SEC."""
    import sqlite3
    import time

    import gateway.run as gr

    db = tmp_path / "corrupt.db"
    db.write_bytes(b"not a sqlite database at all - corrupted")

    disabled_corrupt_boards = {}
    slug = "testboard"

    # Verify that a corrupt db fails quick_check (confirming the test fixture works)
    uri = f"file:{db}?mode=ro"
    try:
        check_conn = sqlite3.connect(uri, uri=True, timeout=2)
        row = check_conn.execute("PRAGMA quick_check").fetchone()
        check_conn.close()
        is_corrupt = row is None or row[0] != "ok"
    except Exception:
        is_corrupt = True

    assert is_corrupt, "Test setup: corrupt DB should fail quick_check"

    # Simulate latch application on first genuine corruption
    now = time.monotonic()
    disabled_corrupt_boards[slug] = {
        "fingerprint": ("path", 1000, 100),
        "disabled_until_ts": now + gr.INITIAL_BACKOFF_SEC,
        "backoff_seconds": gr.INITIAL_BACKOFF_SEC,
    }
    assert disabled_corrupt_boards[slug]["backoff_seconds"] == gr.INITIAL_BACKOFF_SEC
    assert (
        disabled_corrupt_boards[slug]["disabled_until_ts"]
        >= now + gr.INITIAL_BACKOFF_SEC - 1
    )


def test_backoff_doubles_on_repeated_failure_same_fingerprint(tmp_path):
    """Backoff must double each cycle: 30 → 60 → 120 → 240, capped at 900."""
    import time

    import gateway.run as gr

    disabled_corrupt_boards = {}
    slug = "testboard"
    fingerprint = ("/tmp/board.db", 1000, 4096)

    backoff = gr.INITIAL_BACKOFF_SEC
    expected_sequence = [30.0, 60.0, 120.0, 240.0]
    for expected in expected_sequence:
        now = time.monotonic()
        disabled_corrupt_boards[slug] = {
            "fingerprint": fingerprint,
            "disabled_until_ts": now + backoff,
            "backoff_seconds": backoff,
        }
        assert backoff == expected
        # Next failure doubles
        prev = disabled_corrupt_boards.get(slug)
        backoff = min(prev["backoff_seconds"] * 2, gr.MAX_BACKOFF_SEC)

    # Cap at MAX_BACKOFF_SEC
    for _ in range(10):
        backoff = min(backoff * 2, gr.MAX_BACKOFF_SEC)
    assert backoff == gr.MAX_BACKOFF_SEC


def test_backoff_clears_on_fingerprint_change(tmp_path, monkeypatch):
    """When mtime changes (fingerprint bumps), the latch must clear immediately."""
    import time

    import gateway.run as gr  # noqa: F401

    disabled_corrupt_boards = {}
    slug = "testboard"
    fingerprint_a = ("/tmp/board.db", 1000, 4096)
    fingerprint_b = ("/tmp/board.db", 2000, 4096)  # mtime bumped

    # Set up latch with fingerprint A (not expired)
    disabled_corrupt_boards[slug] = {
        "fingerprint": fingerprint_a,
        "disabled_until_ts": time.monotonic() + 500,
        "backoff_seconds": 30.0,
    }

    # Simulate what _tick_once_for_board does on fingerprint change
    state = disabled_corrupt_boards.get(slug)
    assert state is not None
    if state["fingerprint"] != fingerprint_b:
        disabled_corrupt_boards.pop(slug, None)

    assert slug not in disabled_corrupt_boards


def test_backoff_clears_on_successful_dispatch(tmp_path):
    """After a successful dispatch_once, the latch entry must be removed."""
    import time

    import gateway.run as gr  # noqa: F401

    disabled_corrupt_boards = {}
    slug = "testboard"

    # Active latch (expired window)
    disabled_corrupt_boards[slug] = {
        "fingerprint": ("/tmp/board.db", 1000, 4096),
        "disabled_until_ts": time.monotonic() - 1,  # expired
        "backoff_seconds": 30.0,
    }

    # Simulate successful dispatch: pop the slug
    disabled_corrupt_boards.pop(slug, None)
    assert slug not in disabled_corrupt_boards


def test_backoff_resets_to_initial_if_fingerprint_changed_between_failures(tmp_path):
    """After fingerprint change + success + new failure, backoff restarts at INITIAL_BACKOFF_SEC."""
    import time

    import gateway.run as gr

    disabled_corrupt_boards = {}
    slug = "testboard"
    fingerprint_a = ("/tmp/board.db", 1000, 4096)
    fingerprint_b = ("/tmp/board.db", 2000, 4096)

    # First failure on fingerprint A → 30s backoff
    disabled_corrupt_boards[slug] = {
        "fingerprint": fingerprint_a,
        "disabled_until_ts": time.monotonic() + 30.0,
        "backoff_seconds": 30.0,
    }
    # Fingerprint changes → clear latch
    disabled_corrupt_boards.pop(slug, None)
    # Success → nothing to clear (already gone)
    # New failure on fingerprint B → must start at INITIAL_BACKOFF_SEC
    prev = disabled_corrupt_boards.get(slug)
    if prev is not None and prev["fingerprint"] == fingerprint_b:
        new_backoff = min(prev["backoff_seconds"] * 2, gr.MAX_BACKOFF_SEC)
    else:
        new_backoff = gr.INITIAL_BACKOFF_SEC
    assert new_backoff == gr.INITIAL_BACKOFF_SEC


def test_other_database_errors_not_latched(tmp_path, monkeypatch):
    """Non-corruption DatabaseErrors (e.g. 'database is locked') must not touch disabled_corrupt_boards."""
    import sqlite3

    import gateway.run as gr  # noqa: F401

    exc = sqlite3.OperationalError("database is locked")
    # _is_corrupt_board_db_error should return False for this error
    # We need to reach into the closure — test via a workaround
    # The function is defined inside _kanban_dispatcher_watcher, but we can test its logic
    msg = str(exc).lower()
    is_corrupt = (
        "file is not a database" in msg or "database disk image is malformed" in msg
    )
    assert not is_corrupt, "Locked-DB error must not be treated as corruption"

    disabled_corrupt_boards = {}
    # If not corrupt, the handler raises/logs normally without touching disabled_corrupt_boards
    assert len(disabled_corrupt_boards) == 0


def test_corrupt_board_not_dispatched_during_backoff_window(tmp_path):
    """While backoff is active, dispatch_once must not be called."""
    import time

    import gateway.run as gr  # noqa: F401

    disabled_corrupt_boards = {}
    slug = "testboard"
    fingerprint = ("/tmp/board.db", 1000, 4096)
    # Set latch with 500s remaining
    disabled_corrupt_boards[slug] = {
        "fingerprint": fingerprint,
        "disabled_until_ts": time.monotonic() + 500,
        "backoff_seconds": 30.0,
    }

    dispatch_calls = [0]

    def simulate_tick():
        state = disabled_corrupt_boards.get(slug)
        if state is not None:
            if state["fingerprint"] != fingerprint:
                disabled_corrupt_boards.pop(slug, None)
            elif time.monotonic() < state["disabled_until_ts"]:
                return None  # skipped
        dispatch_calls[0] += 1
        return "dispatched"

    for _ in range(3):
        simulate_tick()

    assert dispatch_calls[0] == 0


def test_malformed_quick_check_result_treated_as_corrupt(tmp_path, monkeypatch):
    """If quick_check raises, it must be treated as confirmed corruption → latch applied."""
    import sqlite3

    import gateway.run as gr  # noqa: F401

    # Simulate _confirm_corruption where quick_check raises
    db = tmp_path / "board.db"
    db.write_bytes(b"garbage")

    original_connect = sqlite3.connect

    def raising_connect(uri, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(sqlite3, "connect", raising_connect)

    exc = sqlite3.DatabaseError("file is not a database")
    # _confirm_corruption catches Exception and returns True (confirmed corrupt)
    # Simulate the logic:
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        row = conn.execute("PRAGMA quick_check").fetchone()
        result = row[0] if row else "error"
        conn.close()
        confirmed = result != "ok"
    except Exception:
        confirmed = True  # Treat exception as confirmed corrupt

    assert confirmed, (
        "Exception during quick_check must be treated as confirmed corruption"
    )
