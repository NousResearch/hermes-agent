import sqlite3


class _FakeBackupConnection:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def backup(self, _destination, **kwargs):
        progress = kwargs["progress"]
        for status, remaining, total in self.events:
            progress(status, remaining, total)

    def close(self):
        self.closed = True


class _FakeDestination:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _run_fake_backup(monkeypatch, tmp_path, events, clock):
    import hermes_cli.backup as backup_mod

    src = tmp_path / "source.db"
    dst = tmp_path / "dest.db"
    src.touch()
    dst.write_bytes(b"partial")

    source = _FakeBackupConnection(events)
    destination = _FakeDestination()
    connections = iter((source, destination))

    monkeypatch.setattr(
        backup_mod.sqlite3,
        "connect",
        lambda *_args, **_kwargs: next(connections),
    )
    times = iter(clock)
    monkeypatch.setattr(backup_mod.time, "monotonic", lambda: next(times))

    result = backup_mod._safe_copy_db(src, dst, timeout_seconds=0.05)
    return result, dst, source, destination


def test_nonadvancing_sqlite_ok_callbacks_timeout(monkeypatch, tmp_path):
    events = [
        (sqlite3.SQLITE_OK, 10, 10),
        (sqlite3.SQLITE_OK, 10, 10),
        (sqlite3.SQLITE_OK, 10, 10),
    ]
    result, dst, source, destination = _run_fake_backup(
        monkeypatch, tmp_path, events, [0.00, 0.00, 0.03, 0.06]
    )
    assert result is False
    assert not dst.exists()
    assert source.closed
    assert destination.closed


def test_restart_oscillation_does_not_count_as_forward_progress(monkeypatch, tmp_path):
    events = [
        (sqlite3.SQLITE_OK, 10, 20),
        (sqlite3.SQLITE_OK, 6, 20),
        (sqlite3.SQLITE_OK, 12, 20),
        (sqlite3.SQLITE_OK, 11, 20),
        (sqlite3.SQLITE_OK, 12, 20),
        (sqlite3.SQLITE_OK, 11, 20),
    ]
    result, dst, source, destination = _run_fake_backup(
        monkeypatch, tmp_path, events, [0.00, 0.00, 0.01, 0.02, 0.03, 0.05, 0.07]
    )
    assert result is False
    assert not dst.exists()
    assert source.closed
    assert destination.closed


def test_restart_that_reaches_new_low_can_complete(monkeypatch, tmp_path):
    events = [
        (sqlite3.SQLITE_OK, 10, 20),
        (sqlite3.SQLITE_OK, 6, 20),
        (sqlite3.SQLITE_OK, 12, 20),
        (sqlite3.SQLITE_OK, 5, 20),
        (sqlite3.SQLITE_BUSY, 5, 20),
        (sqlite3.SQLITE_DONE, 0, 20),
    ]
    result, _dst, source, destination = _run_fake_backup(
        monkeypatch, tmp_path, events, [0.00, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
    )
    assert result is True
    assert source.closed
    assert destination.closed
