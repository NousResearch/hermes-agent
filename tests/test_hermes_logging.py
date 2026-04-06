"""Tests for hermes_logging — centralized logging setup."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from unittest.mock import patch

import pytest

import hermes_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Reset the module-level sentinel and clean up root logger handlers
    added by setup_logging() so tests don't leak state."""
    hermes_logging._logging_initialized = False
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    yield
    # Restore — remove any handlers added during the test.
    for h in list(root.handlers):
        if h not in original_handlers:
            root.removeHandler(h)
            h.close()
    hermes_logging._logging_initialized = False


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
