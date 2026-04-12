"""Tests for hermes_logging — centralized logging setup."""

import logging
import os
import stat
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_logging


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
    root = logging.getLogger()
    # Strip ALL RotatingFileHandlers — not just the ones we added — so that
    # handlers leaked from other test modules in the same xdist worker don't
    # pollute our counts.
    pre_existing = []
    for h in list(root.handlers):
        if isinstance(h, RotatingFileHandler):
            root.removeHandler(h)
            h.close()
        else:
            pre_existing.append(h)
    # Ensure the record factory is installed (it's idempotent).
    hermes_logging._install_session_record_factory()
    yield
    # Restore — remove any handlers added during the test.
    for h in list(root.handlers):
        if h not in pre_existing:
            root.removeHandler(h)
            h.close()
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
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert len(agent_handlers) == 1
        assert agent_handlers[0].level == logging.INFO

    def test_creates_errors_log_handler(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        root = logging.getLogger()

        error_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "errors.log" in getattr(h, "baseFilename", "")
        ]
        assert len(error_handlers) == 1
        assert error_handlers[0].level == logging.WARNING

    def test_idempotent_no_duplicate_handlers(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_logging(hermes_home=hermes_home)  # second call — should be no-op

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert len(agent_handlers) == 1

    def test_force_reinitializes(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        # Force still won't add duplicate handlers because _add_rotating_handler
        # checks by resolved path.
        hermes_logging.setup_logging(hermes_home=hermes_home, force=True)

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert len(agent_handlers) == 1

    def test_custom_log_level(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, log_level="DEBUG")

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert agent_handlers[0].level == logging.DEBUG

    def test_custom_max_size_and_backup(self, hermes_home):
        hermes_logging.setup_logging(
            hermes_home=hermes_home, max_size_mb=10, backup_count=5
        )

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert agent_handlers[0].maxBytes == 10 * 1024 * 1024
        assert agent_handlers[0].backupCount == 5

    def test_suppresses_noisy_loggers(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        assert logging.getLogger("openai").level >= logging.WARNING
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_writes_to_agent_log(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        test_logger = logging.getLogger("test_hermes_logging.write_test")
        test_logger.info("test message for agent.log")

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        assert agent_log.exists()
        content = agent_log.read_text()
        assert "test message for agent.log" in content

    def test_warnings_appear_in_both_logs(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        test_logger = logging.getLogger("test_hermes_logging.warning_test")
        test_logger.warning("this is a warning")

        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        errors_log = hermes_home / "logs" / "errors.log"
        assert "this is a warning" in agent_log.read_text()
        assert "this is a warning" in errors_log.read_text()

    def test_info_not_in_errors_log(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)

        test_logger = logging.getLogger("test_hermes_logging.info_test")
        test_logger.info("info only message")

        for h in logging.getLogger().handlers:
            h.flush()

        errors_log = hermes_home / "logs" / "errors.log"
        if errors_log.exists():
            assert "info only message" not in errors_log.read_text()

    def test_reads_config_yaml(self, hermes_home):
        """setup_logging reads logging.level from config.yaml."""
        import yaml
        config = {"logging": {"level": "DEBUG", "max_size_mb": 2, "backup_count": 1}}
        (hermes_home / "config.yaml").write_text(yaml.dump(config))

        hermes_logging.setup_logging(hermes_home=hermes_home)

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert agent_handlers[0].level == logging.DEBUG
        assert agent_handlers[0].maxBytes == 2 * 1024 * 1024
        assert agent_handlers[0].backupCount == 1

    def test_explicit_params_override_config(self, hermes_home):
        """Explicit function params take precedence over config.yaml."""
        import yaml
        config = {"logging": {"level": "DEBUG"}}
        (hermes_home / "config.yaml").write_text(yaml.dump(config))

        hermes_logging.setup_logging(hermes_home=hermes_home, log_level="WARNING")

        root = logging.getLogger()
        agent_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "agent.log" in getattr(h, "baseFilename", "")
        ]
        assert agent_handlers[0].level == logging.WARNING

    def test_record_factory_installed(self, hermes_home):
        """The custom record factory injects session_tag on all records."""
        hermes_logging.setup_logging(hermes_home=hermes_home)
        factory = logging.getLogRecordFactory()
        assert getattr(factory, "_hermes_session_injector", False), (
            "Record factory should have _hermes_session_injector marker"
        )
        # Verify session_tag exists on a fresh record
        record = factory("test", logging.INFO, "", 0, "msg", (), None)
        assert hasattr(record, "session_tag")


class TestGatewayMode:
    """setup_logging(mode='gateway') creates a filtered gateway.log."""

    def test_gateway_log_created(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")
        root = logging.getLogger()

        gw_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "gateway.log" in getattr(h, "baseFilename", "")
        ]
        assert len(gw_handlers) == 1

    def test_gateway_log_not_created_in_cli_mode(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="cli")
        root = logging.getLogger()

        gw_handlers = [
            h for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and "gateway.log" in getattr(h, "baseFilename", "")
        ]
        assert len(gw_handlers) == 0

    def test_gateway_log_receives_gateway_records(self, hermes_home):
        """gateway.log captures records from gateway.* loggers."""
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        gw_logger = logging.getLogger("gateway.platforms.telegram")
        gw_logger.info("telegram connected")

        for h in logging.getLogger().handlers:
            h.flush()

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

        for h in logging.getLogger().handlers:
            h.flush()

        gw_log = hermes_home / "logs" / "gateway.log"
        if gw_log.exists():
            content = gw_log.read_text()
            assert "running command" not in content
            assert "compressing context" not in content

    def test_agent_log_still_receives_all(self, hermes_home):
        """agent.log (catch-all) still receives gateway AND tool records."""
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        logging.getLogger("gateway.run").info("gateway msg")
        logging.getLogger("tools.file_tools").info("file msg")

        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()
        assert "gateway msg" in content
        assert "file msg" in content


class TestSessionContext:
    """set_session_context / clear_session_context + _SessionFilter."""

    def test_session_tag_in_log_output(self, hermes_home):
        """When session context is set, log lines include [session_id]."""
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.set_session_context("abc123")

        test_logger = logging.getLogger("test.session_tag")
        test_logger.info("tagged message")

        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()
        assert "[abc123]" in content
        assert "tagged message" in content

    def test_no_session_tag_without_context(self, hermes_home):
        """Without session context, log lines have no session tag."""
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.clear_session_context()

        test_logger = logging.getLogger("test.no_session")
        test_logger.info("untagged message")

        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()
        assert "untagged message" in content
        # Should not have any [xxx] session tag
        import re
        for line in content.splitlines():
            if "untagged message" in line:
                assert not re.search(r"\[.+?\]", line.split("INFO")[1].split("test.no_session")[0])

    def test_clear_session_context(self, hermes_home):
        """After clearing, session tag disappears."""
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.set_session_context("xyz789")
        hermes_logging.clear_session_context()

        test_logger = logging.getLogger("test.cleared")
        test_logger.info("after clear")

        for h in logging.getLogger().handlers:
            h.flush()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()
        assert "[xyz789]" not in content

    def test_session_context_thread_isolated(self, hermes_home):
        """Session context is per-thread — one thread's context doesn't leak."""
        hermes_logging.setup_logging(hermes_home=hermes_home)

        results = {}

        def thread_a():
            hermes_logging.set_session_context("thread_a_session")
            logging.getLogger("test.thread_a").info("from thread A")
            for h in logging.getLogger().handlers:
                h.flush()

        def thread_b():
            hermes_logging.set_session_context("thread_b_session")
            logging.getLogger("test.thread_b").info("from thread B")
            for h in logging.getLogger().handlers:
                h.flush()

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        ta.join()
        tb.start()
        tb.join()

        agent_log = hermes_home / "logs" / "agent.log"
        content = agent_log.read_text()

        # Each thread's message should have its own session tag
        for line in content.splitlines():
            if "from thread A" in line:
                assert "[thread_a_session]" in line
                assert "[thread_b_session]" not in line
            if "from thread B" in line:
                assert "[thread_b_session]" in line
                assert "[thread_a_session]" not in line


class TestRecordFactory:
    """Unit tests for the custom LogRecord factory."""

    def test_record_has_session_tag(self):
        """Every record gets a session_tag attribute."""
        factory = logging.getLogRecordFactory()
        record = factory("test", logging.INFO, "", 0, "msg", (), None)
        assert hasattr(record, "session_tag")

    def test_empty_tag_without_context(self):
        hermes_logging.clear_session_context()
        factory = logging.getLogRecordFactory()
        record = factory("test", logging.INFO, "", 0, "msg", (), None)
        assert record.session_tag == ""

    def test_tag_with_context(self):
        hermes_logging.set_session_context("sess_42")
        factory = logging.getLogRecordFactory()
        record = factory("test", logging.INFO, "", 0, "msg", (), None)
        assert record.session_tag == " [sess_42]"

    def test_idempotent_install(self):
        """Calling _install_session_record_factory() twice doesn't double-wrap."""
        hermes_logging._install_session_record_factory()
        factory_a = logging.getLogRecordFactory()
        hermes_logging._install_session_record_factory()
        factory_b = logging.getLogRecordFactory()
        assert factory_a is factory_b

    def test_works_with_any_handler(self):
        """A handler using %(session_tag)s works even without _SessionFilter."""
        hermes_logging.set_session_context("any_handler_test")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(session_tag)s %(message)s"))

        logger = logging.getLogger("_test_any_handler")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            # Should not raise KeyError
            logger.info("hello")
        finally:
            logger.removeHandler(handler)


class TestComponentFilter:
    """Unit tests for _ComponentFilter."""

    def test_passes_matching_prefix(self):
        f = hermes_logging._ComponentFilter(("gateway",))
        record = logging.LogRecord(
            "gateway.run", logging.INFO, "", 0, "msg", (), None
        )
        assert f.filter(record) is True

    def test_passes_nested_matching_prefix(self):
        f = hermes_logging._ComponentFilter(("gateway",))
        record = logging.LogRecord(
            "gateway.platforms.telegram", logging.INFO, "", 0, "msg", (), None
        )
        assert f.filter(record) is True

    def test_blocks_non_matching(self):
        f = hermes_logging._ComponentFilter(("gateway",))
        record = logging.LogRecord(
            "tools.terminal_tool", logging.INFO, "", 0, "msg", (), None
        )
        assert f.filter(record) is False

    def test_multiple_prefixes(self):
        f = hermes_logging._ComponentFilter(("agent", "run_agent", "model_tools"))
        assert f.filter(logging.LogRecord(
            "agent.compressor", logging.INFO, "", 0, "", (), None
        ))
        assert f.filter(logging.LogRecord(
            "run_agent", logging.INFO, "", 0, "", (), None
        ))
        assert f.filter(logging.LogRecord(
            "model_tools", logging.INFO, "", 0, "", (), None
        ))
        assert not f.filter(logging.LogRecord(
            "tools.browser", logging.INFO, "", 0, "", (), None
        ))


class TestComponentPrefixes:
    """COMPONENT_PREFIXES covers the expected components."""

    def test_gateway_prefix(self):
        assert "gateway" in hermes_logging.COMPONENT_PREFIXES
        assert ("gateway",) == hermes_logging.COMPONENT_PREFIXES["gateway"]

    def test_agent_prefix(self):
        prefixes = hermes_logging.COMPONENT_PREFIXES["agent"]
        assert "agent" in prefixes
        assert "run_agent" in prefixes
        assert "model_tools" in prefixes

    def test_tools_prefix(self):
        assert ("tools",) == hermes_logging.COMPONENT_PREFIXES["tools"]

    def test_cli_prefix(self):
        prefixes = hermes_logging.COMPONENT_PREFIXES["cli"]
        assert "hermes_cli" in prefixes
        assert "cli" in prefixes

    def test_cron_prefix(self):
        assert ("cron",) == hermes_logging.COMPONENT_PREFIXES["cron"]


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

    def test_idempotent(self, hermes_home):
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_verbose_logging()
        hermes_logging.setup_verbose_logging()  # second call

        root = logging.getLogger()
        verbose_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, RotatingFileHandler)
            and getattr(h, "_hermes_verbose", False)
        ]
        assert len(verbose_handlers) == 1


class TestAddRotatingHandler:
    """_add_rotating_handler() is idempotent and creates the directory."""

    def test_creates_directory(self, tmp_path):
        log_path = tmp_path / "subdir" / "test.log"
        logger = logging.getLogger("_test_rotating")
        formatter = logging.Formatter("%(message)s")

        hermes_logging._add_rotating_handler(
            logger, log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )

        assert log_path.parent.is_dir()
        # Clean up
        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()

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
            h for h in logger.handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating_handlers) == 1
        # Clean up
        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()

    def test_log_filter_attached(self, tmp_path):
        """Optional log_filter is attached to the handler."""
        log_path = tmp_path / "filtered.log"
        logger = logging.getLogger("_test_rotating_filter")
        formatter = logging.Formatter("%(message)s")
        component_filter = hermes_logging._ComponentFilter(("test",))

        hermes_logging._add_rotating_handler(
            logger, log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
            log_filter=component_filter,
        )

        handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(handlers) == 1
        assert component_filter in handlers[0].filters
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

        handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(handlers) == 1
        # No _SessionFilter on the handler — record factory handles it
        assert len(handlers[0].filters) == 0

        # But session_tag still works (via record factory)
        hermes_logging.set_session_context("factory_test")
        logger.info("test msg")
        handlers[0].flush()
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

    def test_managed_mode_rollover_sets_group_writable(self, tmp_path):
        log_path = tmp_path / "managed-rollover.log"
        logger = logging.getLogger("_test_rotating_managed_rollover")
        formatter = logging.Formatter("%(message)s")

        old_umask = os.umask(0o022)
        try:
            with patch("hermes_cli.config.is_managed", return_value=True):
                hermes_logging._add_rotating_handler(
                    logger, log_path,
                    level=logging.INFO, max_bytes=1, backup_count=1,
                    formatter=formatter,
                )
                handler = next(
                    h for h in logger.handlers if isinstance(h, RotatingFileHandler)
                )
                logger.info("a" * 256)
                handler.flush()
        finally:
            os.umask(old_umask)

        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o660

        for h in list(logger.handlers):
            if isinstance(h, RotatingFileHandler):
                logger.removeHandler(h)
                h.close()


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

    def test_handles_missing_logging_section(self, hermes_home):
        import yaml
        config = {"model": "test"}
        (hermes_home / "config.yaml").write_text(yaml.dump(config))

        level, max_size, backup = hermes_logging._read_logging_config()
        assert level is None
