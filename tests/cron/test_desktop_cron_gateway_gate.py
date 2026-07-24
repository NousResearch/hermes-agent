"""#66629 — the desktop cron ticker must defer to a live gateway.

When ``hermes gateway run`` and ``hermes serve`` (desktop) share a HERMES_HOME,
the desktop ticker's standalone delivery path has no live platform adapters, so
Feishu interactive cards silently degrade to plain text. The fix gates the
desktop ticker on the gateway runtime lock: it dispatches only when no live
gateway owns this HERMES_HOME, and resumes on its own if the gateway stops.

The headline test is behavioral, through the public ticker entry point
(``hermes_cli.web_server._start_desktop_cron_ticker``): it fails before the fix
(the ungated ticker fires regardless of the gateway lock) and passes after. The
probe unit tests pin the read-only contract of the lock probe.
"""
import errno
import threading
import time
from unittest.mock import patch


def _wait_until(predicate, timeout=10.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_desktop_ticker_defers_to_a_live_gateway_then_resumes(monkeypatch, tmp_path):
    """RED before the gate, GREEN after. While a live gateway owns the runtime
    lock on this HERMES_HOME the desktop ticker must not drive cron.scheduler.tick
    (its standalone path degrades interactive cards). It resumes once the gateway
    releases the lock."""
    import gateway.status as status
    from hermes_cli.web_server import _start_desktop_cron_ticker

    # Keep everything inside tmp: the lock the probe reads and the HERMES_HOME
    # the ticker heartbeats into.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    # A second real OS lock stands in for a live gateway holding the lock.
    holder = open(lock_path, "a+", encoding="utf-8")
    assert status._try_acquire_file_lock(holder) is True

    ticks = []
    stop = threading.Event()

    def fake_tick(*args, **kwargs):
        ticks.append(kwargs)
        return 0

    with patch("cron.scheduler.tick", side_effect=fake_tick):
        t = threading.Thread(
            target=_start_desktop_cron_ticker,
            args=(stop,),
            kwargs={"interval": 0.01},
            daemon=True,
        )
        t.start()
        try:
            # A live gateway owns the lock -> the desktop ticker must stay quiet.
            time.sleep(0.2)
            assert ticks == [], (
                f"desktop ticker fired {len(ticks)} tick(s) while a live gateway "
                "owned the runtime lock (interactive cards would degrade to text)"
            )
            # Gateway stops -> lock released -> desktop cron resumes on its own.
            status._release_file_lock(holder)
            holder.close()
            assert _wait_until(lambda: len(ticks) >= 1), (
                "desktop ticker did not resume after the gateway released the lock"
            )
        finally:
            stop.set()
            t.join(timeout=5)
    assert not t.is_alive()


def test_probe_reports_free_without_creating_the_lock(tmp_path, monkeypatch):
    """No lock file -> "free", and the read-only probe must not create one."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    assert status.probe_gateway_runtime_lock() == "free"
    assert not lock_path.exists(), "probe must not create the lock file (opens r+, not a+)"


def test_probe_reports_held_then_free_across_a_real_lock(tmp_path, monkeypatch):
    """A held OS lock reads as "held"; once released it reads as "free"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    holder = open(lock_path, "a+", encoding="utf-8")
    try:
        assert status._try_acquire_file_lock(holder) is True
        assert status.probe_gateway_runtime_lock() == "held"
    finally:
        status._release_file_lock(holder)
        holder.close()
    assert status.probe_gateway_runtime_lock() == "free"


def test_probe_reports_unknown_on_non_contention_lock_error(tmp_path, monkeypatch):
    """A non-contention lock error (ENOTSUP / ENOLCK / EIO) must report
    "unknown", NOT "held" (Sol REQUEST-CHANGES finding 1). Collapsing every
    OSError to "held" would stall a desktop-only cron indefinitely on an
    unsupported filesystem — a fail-open contract violation."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    lock_path.write_text("owner-record", encoding="utf-8")
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    def raise_enotsup(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "operation not supported by this filesystem")

    if status._IS_WINDOWS:
        monkeypatch.setattr(status.msvcrt, "locking", raise_enotsup)
    else:
        monkeypatch.setattr(status.fcntl, "flock", raise_enotsup)

    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_reports_unknown_when_path_inspection_raises(tmp_path, monkeypatch):
    """A stat / permission error from Path.exists() must be caught as "unknown",
    not escape the documented three-valued API (Sol REQUEST-CHANGES finding 2)."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    def deny(self, *args, **kwargs):
        raise PermissionError("simulated stat denied")

    monkeypatch.setattr("pathlib.Path.exists", deny)
    assert status.probe_gateway_runtime_lock() == "unknown"


def test_probe_stays_read_only_on_empty_windows_lock(tmp_path, monkeypatch):
    """On Windows an empty existing lock must report "unknown" WITHOUT writing
    a byte to it (Sol REQUEST-CHANGES finding 3 — strict read-only contract).
    On POSIX an empty file locks fine, so it reports "free"."""
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    lock_path.touch()
    assert lock_path.stat().st_size == 0
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    result = status.probe_gateway_runtime_lock()
    if status._IS_WINDOWS:
        assert result == "unknown"
        assert lock_path.stat().st_size == 0, "probe wrote to an empty lock file"
    else:
        assert result == "free"


def test_probe_reports_unknown_without_unlinking_on_permission_error(tmp_path, monkeypatch):
    """A permission error reading an existing lock -> "unknown", and the probe
    must NOT unlink the lock (is_gateway_runtime_lock_active does; the probe must
    not, or a desktop cron could delete a live gateway's lock — #66629)."""
    import builtins
    import gateway.status as status

    monkeypatch.setattr(status, "_gateway_lock_handle", None)
    lock_path = tmp_path / "gateway.lock"
    lock_path.write_text("owner-record", encoding="utf-8")
    monkeypatch.setattr(status, "_get_gateway_lock_path", lambda *a, **k: lock_path)

    real_open = builtins.open

    def denying_open(path, *args, **kwargs):
        if str(path) == str(lock_path):
            raise PermissionError("simulated denied read")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denying_open)
    try:
        assert status.probe_gateway_runtime_lock() == "unknown"
    finally:
        monkeypatch.setattr(builtins, "open", real_open)
    assert lock_path.exists(), "probe must not unlink the lock on a permission error"
