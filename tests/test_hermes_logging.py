"""Tests for hermes_logging — centralized logging setup."""
import io
import logging
import os
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_logging
# Use whatever RotatingFileHandler class hermes_logging actually resolved so
# the autouse fixture's isinstance checks (which strip rotating handlers
# between tests) match the real handlers on every platform. hermes_logging
# aliases concurrent-log-handler's ConcurrentRotatingFileHandler on Windows
# (the #44873 fix) but keeps stdlib RotatingFileHandler on POSIX, so importing
# the name from the module under test keeps the two in lockstep.
from hermes_logging import RotatingFileHandler


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level sentinel and clean up root logger handlers
    added by setup_logging() so tests don't leak state.

    Under xdist (-n auto) other test modules may have called setup_logging()
    in the same worker process, leaving RotatingFileHandlers on the root
    logger.  We strip ALL RotatingFileHandlers before each test so the count
    assertions are stable regardless of test ordering.
    """
    hermes_logging._logging_initialized = False
    # File handlers now live behind the async QueueListener, not on the root
    # logger; tear down any leaked from other xdist tests in this worker.
    hermes_logging._reset_queued_handlers()
    root = logging.getLogger()
    prev_root_level = root.level
    root.setLevel(logging.NOTSET)
    # Snapshot the remaining (non-file) handlers so we can strip whatever the
    # test adds.
    pre_existing = list(root.handlers)
    # Ensure the record factory is installed (it's idempotent).
    hermes_logging._install_session_record_factory()
    yield
    # Restore — tear down async file logging + remove handlers added by the test.
    hermes_logging._reset_queued_handlers()
    for h in list(root.handlers):
        if h not in pre_existing:
            root.removeHandler(h)
            h.close()
    root.setLevel(prev_root_level)
    hermes_logging._logging_initialized = False
    hermes_logging.clear_session_context()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Provide an isolated HERMES_HOME for logging tests.

    Uses the same tmp_path as the autouse _isolate_hermes_home from conftest,
    reading it back from the env var to avoid double-mkdir conflicts.
    """
    home = Path(os.environ["HERMES_HOME"])
    return home


class TestSetupLogging:
    """setup_logging() creates agent.log + errors.log with RotatingFileHandler."""

    def test_creates_log_directory(self, hermes_home):
        log_dir = hermes_logging.setup_logging(hermes_home=hermes_home)
        assert log_dir == hermes_home / "logs"
        assert log_dir.is_dir()

    def test_creates_agent_log_handler(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        root = logging.getLogger()

        agent_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert len(agent_handlers) == 1
        assert agent_handlers[0].level == logging.INFO


    def test_idempotent_no_duplicate_handlers(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_logging(hermes_home=hermes_home)  # second call — should be no-op

        root = logging.getLogger()
        agent_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert len(agent_handlers) == 1





    def test_writes_to_agent_log(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        test_logger = logging.getLogger("test_hermes_logging.write_test")
        test_logger.info("test message for agent.log")

        # Flush handlers
        hermes_logging.flush_log_queue()

        agent_log = hermes_home / "logs" / "agent.log"
        assert agent_log.exists()
        content = agent_log.read_text()
        assert "test message for agent.log" in content

    def test_profile_routing_follows_context_home(self, hermes_home, tmp_path):
        """Desktop multiplex cron records are written to their owning profile."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        hermes_logging.setup_logging(hermes_home=hermes_home)
        assert hermes_logging.enable_profile_log_routing(
            [hermes_home, profile_home]
        ) is True

        logger = logging.getLogger("cron.scheduler.profile-routing-test")
        token = set_hermes_home_override(profile_home)
        try:
            logger.info("profile-routed cron record")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        assert "profile-routed cron record" in (
            profile_home / "logs" / "agent.log"
        ).read_text()
        assert "profile-routed cron record" not in (
            hermes_home / "logs" / "agent.log"
        ).read_text()




    def test_explicit_params_override_config(self, hermes_home):
        """Explicit function params take precedence over config.yaml."""
        import yaml
        config = {"logging": {"level": "DEBUG"}}
        (hermes_home / "config.yaml").write_text(yaml.dump(config))

        hermes_logging.setup_logging(hermes_home=hermes_home, log_level="WARNING")

        root = logging.getLogger()
        agent_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert agent_handlers[0].level == logging.WARNING



class TestGatewayMode:
    """setup_logging(mode='gateway') creates a filtered gateway.log."""

    def test_gateway_log_created(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")
        root = logging.getLogger()

        gw_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "gateway.log" in getattr(h, "baseFilename", "")
        ]
        assert len(gw_handlers) == 1

    def test_gateway_log_not_created_in_cli_mode(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="cli")
        root = logging.getLogger()

        gw_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "gateway.log" in getattr(h, "baseFilename", "")
        ]
        assert len(gw_handlers) == 0



    def test_gateway_log_receives_gateway_records(self, hermes_home):
        """gateway.log captures records from gateway.* loggers."""
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        gw_logger = logging.getLogger("plugins.platforms.telegram.adapter")
        gw_logger.info("telegram connected")

        hermes_logging.flush_log_queue()

        gw_log = hermes_home / "logs" / "gateway.log"
        assert gw_log.exists()
        assert "telegram connected" in gw_log.read_text()

    def test_gateway_log_rejects_non_gateway_records(self, hermes_home):
        """gateway.log does NOT capture records from tools.*, agent.*, etc."""
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        tool_logger = logging.getLogger("tools.terminal_tool")
        tool_logger.info("running command")

        agent_logger = logging.getLogger("agent.context_compressor")
        agent_logger.info("compressing context")

        hermes_logging.flush_log_queue()

        gw_log = hermes_home / "logs" / "gateway.log"
        if gw_log.exists():
            content = gw_log.read_text()
            assert "running command" not in content
            assert "compressing context" not in content



class TestGuiMode:
    """setup_logging(mode='gui') creates a filtered gui.log."""

    def test_gui_log_created(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gui")
        root = logging.getLogger()

        gui_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
            and "gui.log" in getattr(h, "baseFilename", "")
        ]
        assert len(gui_handlers) == 1


    def test_gui_log_receives_only_gui_components(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gui")

        logging.getLogger("hermes_cli.web_server").info("dashboard online")
        logging.getLogger("tui_gateway.ws").info("ws connected")
        logging.getLogger("gateway.run").info("gateway event")

        hermes_logging.flush_log_queue()

        gui_log = hermes_home / "logs" / "gui.log"
        assert gui_log.exists()
        content = gui_log.read_text()
        assert "dashboard online" in content
        assert "ws connected" in content
        assert "gateway event" not in content


class TestSessionContext:
    """set_session_context / clear_session_context + _SessionFilter."""

    def test_session_tag_in_log_output(self, hermes_home):
        """When session context is set, log lines include [session_id]."""
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.set_session_context("abc123")

        test_logger = logging.getLogger("test.session_tag")
        test_logger.info("tagged message")

        hermes_logging.flush_log_queue()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()
        assert "[abc123]" in content
        assert "tagged message" in content







class TestComponentFilter:
    """Unit tests for _ComponentFilter."""

    def test_passes_matching_prefix(self):
        f = hermes_logging._ComponentFilter(("gateway",))
        record = logging.LogRecord(
            "gateway.run", logging.INFO, "", 0, "msg", (), None
        )
        assert f.filter(record) is True


    def test_blocks_non_matching(self):
        f = hermes_logging._ComponentFilter(("gateway",))
        record = logging.LogRecord(
            "tools.terminal_tool", logging.INFO, "", 0, "msg", (), None
        )
        assert f.filter(record) is False





class TestSetupVerboseLogging:
    """setup_verbose_logging() adds a DEBUG-level console handler."""

    def test_adds_stream_handler(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_verbose_logging()

        root = logging.getLogger()
        verbose_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, RotatingFileHandler)
            and getattr(h, "_hermes_verbose", False)
        ]
        assert len(verbose_handlers) == 1
        assert verbose_handlers[0].level == logging.DEBUG



class TestAddRotatingHandler:
    """_add_rotating_handler() is idempotent and creates the directory."""


    def test_no_duplicate_for_same_path(self, tmp_path):
        log_path = tmp_path / "test.log"
        logger = logging.getLogger("_test_rotating_dup")
        formatter = logging.Formatter("%(message)s")

        hermes_logging._add_rotating_handler(
            logger, log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )
        hermes_logging._add_rotating_handler(
            logger, log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )

        rotating_handlers = [
            h for h in hermes_logging.rotating_file_handlers()
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating_handlers) == 1
        # Clean up
        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()


    def test_no_session_filter_on_handler(self, tmp_path):
        """Handlers rely on record factory, not per-handler _SessionFilter."""
        log_path = tmp_path / "no_session_filter.log"
        logger = logging.getLogger("_test_no_session_filter")
        formatter = logging.Formatter("%(session_tag)s%(message)s")

        hermes_logging._add_rotating_handler(
            logger, log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )

        handlers = [h for h in hermes_logging.rotating_file_handlers() if isinstance(h, RotatingFileHandler)]
        assert len(handlers) == 1
        # No _SessionFilter on the handler — record factory handles it
        assert len(handlers[0].filters) == 0

        # But session_tag still works (via record factory)
        hermes_logging.set_session_context("factory_test")
        logger.info("test msg")
        hermes_logging.flush_log_queue()
        content = log_path.read_text()
        assert "[factory_test]" in content

        # Clean up
        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()

    def test_managed_mode_initial_open_sets_group_writable(self, tmp_path):
        log_path = tmp_path / "managed-open.log"
        logger = logging.getLogger("_test_rotating_managed_open")
        formatter = logging.Formatter("%(message)s")

        old_umask = os.umask(0o022)
        try:
            with patch("hermes_cli.config.is_managed", return_value=True):
                hermes_logging._add_rotating_handler(
                    logger, log_path,
                    level=logging.INFO, max_bytes=1024, backup_count=1,
                    formatter=formatter,
                )
        finally:
            os.umask(old_umask)

        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o660

        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()



class TestWindowsConcurrentLogLockTimeout:
    """Windows concurrent-log-handler lock timeouts stay inside logging."""

    def _make_logger_and_handler(self, log_path: Path):
        logger = logging.getLogger(f"_test_concurrent_lock_timeout_{log_path.stem}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)

        handler = hermes_logging._ManagedRotatingFileHandler(
            str(log_path), maxBytes=1, backupCount=1, encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger, handler

    @pytest.mark.windows_only
    def test_helper_only_matches_windows_concurrent_lock_timeout(self):
        # Windows-only: concurrent-log-handler (and therefore its cross-process
        # lock timeout) is only installed on Windows — faking sys.platform
        # exercised the string check without the handler that raises it.
        assert hermes_logging._is_windows_concurrent_log_lock_timeout(
            RuntimeError("Cannot acquire lock after 20 attempts")
        )
        assert not hermes_logging._is_windows_concurrent_log_lock_timeout(
            RuntimeError("some other logging failure")
        )

    @pytest.mark.linux_only
    def test_helper_never_matches_off_windows(self):
        # On POSIX the suppression must stay inert: stdlib RotatingFileHandler
        # is in use, so this RuntimeError text is never a CLH lock timeout.
        assert not hermes_logging._is_windows_concurrent_log_lock_timeout(
            RuntimeError("Cannot acquire lock after 20 attempts")
        )

    @pytest.mark.windows_only
    def test_lock_timeout_routed_to_handle_error_is_suppressed(self, tmp_path, capsys):
        """Mirror CLH's real control flow.

        ``concurrent-log-handler``'s ``emit()`` wraps its whole body in
        ``try/except Exception: self.handleError(record)``, so the lock
        RuntimeError raised in ``_do_lock()`` is caught *inside* CLH and routed
        to ``handleError`` with the exception live in ``sys.exc_info()``.  We
        invoke ``handleError`` the same way CLH would and assert no traceback
        reaches stderr (the slash-worker surface).

        Windows-only: the suppression is keyed on the real host, and only on
        Windows is the base handler CLH at all — the fake platform gave us the
        branch without the handler that raises."""
        logger, handler = self._make_logger_and_handler(tmp_path / "agent.log")
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, "force rollover", (), None,
        )
        try:
            try:
                raise RuntimeError("Cannot acquire lock after 20 attempts")
            except RuntimeError:
                handler.handleError(record)

            captured = capsys.readouterr()
            assert "Cannot acquire lock after 20 attempts" not in captured.err
            assert "--- Logging error ---" not in captured.err
        finally:
            logger.removeHandler(handler)
            handler.close()



class TestReadLoggingConfig:
    """_read_logging_config() reads from config.yaml."""

    def test_returns_none_when_no_config(self, hermes_home):
        level, max_size, backup = hermes_logging._read_logging_config()
        assert level is None
        assert max_size is None
        assert backup is None

    def test_reads_logging_section(self, hermes_home):
        import yaml
        config = {"logging": {"level": "DEBUG", "max_size_mb": 10, "backup_count": 5}}
        (hermes_home / "config.yaml").write_text(yaml.dump(config))

        level, max_size, backup = hermes_logging._read_logging_config()
        assert level == "DEBUG"
        assert max_size == 10
        assert backup == 5



class TestExternalRotationRecovery:
    """_ManagedRotatingFileHandler recovers from external rotation.

    External rotation = anything that renames, unlinks, or replaces the
    log file without going through ``doRollover()``: logrotate, manual
    ``mv``, another process rotating under us, or a transient ``rm``.
    Before this fix the open file descriptor stayed pinned to the old
    inode forever, so every subsequent write went to the rotated backup
    instead of the file the operator expects to read.
    """

    def _make_handler(self, log_path: Path) -> hermes_logging._ManagedRotatingFileHandler:
        handler = hermes_logging._ManagedRotatingFileHandler(
            str(log_path), maxBytes=10 * 1024 * 1024, backupCount=3,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def _emit(self, handler: logging.Handler, msg: str) -> None:
        record = logging.LogRecord(
            name="gateway.run", level=logging.INFO, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None,
        )
        # Match the record factory that hermes_logging installs at import time.
        record.session_tag = ""
        handler.emit(record)
        hermes_logging.flush_log_queue()

    def test_recovers_after_external_rename(self, tmp_path):
        """logrotate-style external rename: ``mv gateway.log gateway.log.1``.

        Handler's fd was pinned to the renamed inode; new writes used to
        go to ``gateway.log.1`` forever.  After fix, the handler reopens
        ``gateway.log`` at the original path.
        """
        log_path = tmp_path / "gateway.log"
        rotated = tmp_path / "gateway.log.1"
        handler = self._make_handler(log_path)
        try:
            self._emit(handler, "before rotation")
            assert log_path.read_text() == "before rotation\n"

            # External rotation (NOT via handler.doRollover()).
            os.rename(log_path, rotated)
            assert not log_path.exists()

            self._emit(handler, "after rotation")

            # The new write should land in a freshly recreated gateway.log,
            # not appended to the rotated backup.
            assert log_path.exists(), "handler did not recreate gateway.log"
            assert log_path.read_text() == "after rotation\n"
            assert rotated.read_text() == "before rotation\n"
        finally:
            handler.close()


    def test_external_truncate_does_not_force_reopen(self, tmp_path):
        """``: > gateway.log`` keeps the same inode — no reopen needed.

        Truncation in place preserves the inode, so subsequent writes
        continue to the same file descriptor.  We assert the post-truncate
        content reflects the truncate (size shrinks) and then grows with
        new writes — i.e. the handler correctly does NOT detect this as
        an inode change.
        """
        log_path = tmp_path / "gateway.log"
        handler = self._make_handler(log_path)
        try:
            self._emit(handler, "AAAA" * 32)
            assert log_path.stat().st_size > 0

            with open(log_path, "w"):
                pass  # truncate to zero
            assert log_path.stat().st_size == 0

            self._emit(handler, "after truncate")
            assert log_path.read_text() == "after truncate\n"
        finally:
            handler.close()


    def test_gateway_log_attached_after_external_rotation_then_re_setup(
        self, hermes_home,
    ):
        """End-to-end Allen-reproduction: gateway.log gets externally rotated,
        ``setup_logging(mode='gateway')`` is re-called, the handler keeps
        working.

        Reproduces Allen's symptom (gateway.log frozen mid-write, all gateway
        records leaking to agent.log) when something external rotates the
        file between setup_logging() calls.
        """
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")
        gw_path = hermes_home / "logs" / "gateway.log"
        rotated = hermes_home / "logs" / "gateway.log.1"

        logging.getLogger("gateway.run").info("line BEFORE rotation")
        hermes_logging.flush_log_queue()
        assert "BEFORE rotation" in gw_path.read_text()

        # External actor renames the file out from under us.
        os.rename(gw_path, rotated)
        assert not gw_path.exists()

        # Caller (or some restart path) re-enters setup_logging.  This used
        # to silently no-op due to the per-path dedup check, leaving the
        # stale fd in place.
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        logging.getLogger("gateway.run").info("line AFTER rotation")
        hermes_logging.flush_log_queue()

        # The new record must reach the live gateway.log, not the rotated
        # backup.  Allen's logs had everything past the rotation point
        # going into agent.log only, never gateway.log.
        assert gw_path.exists(), "gateway.log was never recreated"
        assert "AFTER rotation" in gw_path.read_text()
        assert "AFTER rotation" not in rotated.read_text()


class TestSafeStderr:
    """Tests for _safe_stderr() — Unicode tolerance on Windows console."""


    def test_wraps_non_utf8_stderr(self, monkeypatch):
        """On non-UTF-8 systems (e.g. Windows cp949), wraps stderr with UTF-8."""
        import io

        class FakeStderr:
            """Simulates a Windows stderr with legacy encoding."""
            encoding = "cp949"
            buffer = io.BytesIO()

            def write(self, s):
                pass

            def flush(self):
                pass

        fake = FakeStderr()
        monkeypatch.setattr(sys, "stderr", fake)
        result = hermes_logging._safe_stderr()
        # Should be a TextIOWrapper, not the original FakeStderr
        assert isinstance(result, io.TextIOWrapper)
        assert result.encoding == "utf-8"
        assert result.errors == "replace"

    def test_handler_emits_unicode_without_crash(self, tmp_path):
        """StreamHandler with _safe_stderr can emit Unicode messages."""
        import io

        # Create a stderr-like stream with ASCII encoding
        class AsciiStream:
            encoding = "ascii"
            buffer = io.BytesIO()

            def write(self, s):
                self.buffer.write(s.encode("ascii", errors="replace"))

            def flush(self):
                pass

        # Without the fix, this would crash on cp949/ASCII stderr.
        # With the wrapper, the em-dash is replaced with '?'
        handler = logging.StreamHandler(
            io.TextIOWrapper(
                io.BytesIO(),
                encoding="utf-8",
                errors="replace",
            )
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("_test_unicode")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            # Em-dash U+2014 — the exact character from the bug report
            logger.info("Session hygiene: 400 messages — auto-compressing")
        finally:
            logger.removeHandler(handler)


class TestAsyncQueueLogging:
    """File logging runs through a QueueListener so emits never block on the
    cross-process rotation lock (Windows event-loop-stall fix)."""

    def test_file_handlers_not_on_root(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        root = logging.getLogger()
        # Rotating file handlers live on the async listener, never on root.
        assert not any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        # Exactly one queue handler funnels records to the listener.
        queue_handlers = [
            h for h in root.handlers if getattr(h, "_hermes_queue", False)
        ]
        assert len(queue_handlers) == 1
        # The real file handlers are discoverable via the accessor.
        assert any(
            "agent.log" in getattr(h, "baseFilename", "")
            for h in hermes_logging.rotating_file_handlers()
        )

