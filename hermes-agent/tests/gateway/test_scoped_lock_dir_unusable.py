"""#134889 W2: a token-lock dir this user may not write gives one named warning, never a crash loop.

The Windows reports were ``Permission denied`` on ``gateway-locks\\telegram-bot-token-*.lock``; the
leading cause is a ``gateway-locks`` dir created by an elevated install. The POSIX analogue used
here is real, not patched: a lock dir without write permission for this uid. The Windows ACL shape
itself is proved by ``test_scoped_lock_dir_windows_acl.py`` on Windows CI.
"""
import asyncio
import logging
import os
import stat
from typing import Any
from unittest.mock import MagicMock

import pytest

from gateway import status
from gateway import status_lock_dir as scoped_locks
from gateway.status_scoped_locks import release_scoped_lock
from gateway.platforms.base import BasePlatformAdapter
from gateway.restart import is_global_startup_conflict
from gateway.run_startup import GatewayStartupMixin

pytestmark = [pytest.mark.platforms("posix"),
              pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0, reason="root ignores mode bits")]


class _StubAdapter(BasePlatformAdapter):
    platform = MagicMock(value="telegram")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {}


def _adapter():
    obj = _StubAdapter.__new__(_StubAdapter)
    obj._running = True
    obj._fatal_error_code = obj._fatal_error_message = obj._fatal_error_handler = None
    obj._fatal_error_retryable = True
    obj._platform_lock_scope = obj._platform_lock_identity = None
    obj._platform_lock_takeover_allowed = obj._platform_lock_takeover_attempted = False
    obj._status_write_logged = None
    obj._write_runtime_status_safe = lambda *a, **k: None
    return obj


@pytest.fixture()
def locked_out_dir(tmp_path, monkeypatch):
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    lock_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # r-x: lists, cannot create the token lock
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    scoped_locks.reset_unusable_lock_dir_reports()
    yield lock_dir
    lock_dir.chmod(stat.S_IRWXU)


def test_unusable_lock_dir_parks_the_adapter_with_the_repair(locked_out_dir, caplog):
    adapter = _adapter()
    with caplog.at_level(logging.DEBUG, logger="gateway.status"):
        # On main this raised PermissionError out of connect() and the adapter retried forever.
        assert adapter._acquire_platform_lock("telegram-bot-token", "tok", "Telegram bot token") is False
        assert _adapter()._acquire_platform_lock("telegram-bot-token", "tok", "Telegram bot token") is False

    assert adapter._fatal_error_code == "lock_dir_unusable"
    # Retryable: the shared lock dir parks EVERY token-locked adapter at once, and a non-retryable
    # park on a Telegram-only gateway exits 78 (taking cron and the Desktop attach endpoint down).
    assert adapter._fatal_error_retryable is True
    assert not is_global_startup_conflict(adapter._fatal_error_code)
    assert str(locked_out_dir) in adapter._fatal_error_message
    assert "chown" in adapter._fatal_error_message
    assert adapter._platform_lock_identity is None  # nothing to release
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "not usable" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in caplog.records]
    assert str(locked_out_dir) in warnings[0].getMessage()


def test_lock_dir_repair_is_picked_up_without_restart(locked_out_dir):
    with pytest.raises(scoped_locks.ScopedLockDirUnusable):
        status.acquire_scoped_lock("telegram-bot-token", "tok")
    locked_out_dir.chmod(stat.S_IRWXU)
    assert status.acquire_scoped_lock("telegram-bot-token", "tok") == (True, None)
    assert not scoped_locks.unusable_lock_dir_reported(locked_out_dir)
    release_scoped_lock("telegram-bot-token", "tok")


def test_unreadable_stale_token_lock_is_named_not_reported_as_a_holderless_conflict(tmp_path, monkeypatch):
    lock_dir = tmp_path / "gateway-locks"
    lock_dir.mkdir()
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(lock_dir))
    stale = status._get_scope_lock_path("telegram-bot-token", "tok")
    stale.write_text('{"pid": 1}', encoding="utf-8")
    stale.chmod(0)
    lock_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # the stale file cannot be read or unlinked either
    try:
        with pytest.raises(scoped_locks.ScopedLockDirUnusable):
            status.acquire_scoped_lock("telegram-bot-token", "tok")
    finally:
        lock_dir.chmod(stat.S_IRWXU)
        stale.chmod(stat.S_IRUSR | stat.S_IWUSR)


class _StartupProbe:
    """Only the state the two real startup decisions read; their bodies run unchanged."""

    _startup_queue_transient_failure = GatewayStartupMixin._startup_queue_transient_failure
    _start_aggregate_connect_results = GatewayStartupMixin._start_aggregate_connect_results
    _start_handle_no_connections = GatewayStartupMixin._start_handle_no_connections
    _startup_retry_entry = GatewayStartupMixin._startup_retry_entry

    def __init__(self):
        self._failed_platforms, self.status_updates, self.fatal = {}, [], []
        self._startup_parked_platforms = False

    async def _safe_adapter_disconnect(self, adapter, platform):
        pass

    def _update_platform_runtime_status(self, name, **kw):
        self.status_updates.append((name, kw))

    def _reconnect_queue_entry(self, platform, adapter, platform_config, **kw):
        return {"adapter": adapter, **kw}

    def _startup_fail_fatal_config(self, reason):
        self.fatal.append(reason)


def test_a_sole_token_locked_platform_on_an_unusable_lock_dir_keeps_the_gateway_up(locked_out_dir, monkeypatch):
    """Review 2.1 on #140490: a Telegram-only gateway must not exit 78 over the lock dir."""
    import gateway.run as run_mod
    monkeypatch.setattr(run_mod, "_write_runtime_status_quiet", lambda **k: None, raising=False)
    adapter = _adapter()
    assert adapter._acquire_platform_lock("telegram-bot-token", "tok", "Telegram bot token") is False
    platform = MagicMock(value="telegram")
    runner = _StartupProbe()
    retryable, nonretryable = [], []
    connected = asyncio.run(runner._start_aggregate_connect_results(
        [(platform, adapter, MagicMock(), "failed", None)], retryable, nonretryable))
    must_exit = runner._start_handle_no_connections(connected, 1, retryable, nonretryable)

    assert (connected, must_exit, nonretryable, runner.fatal) == (0, False, [], [])
    assert platform in runner._failed_platforms  # the reconnect watcher picks the repair up
    assert retryable and "not usable" in retryable[0]
