"""Tests for hermes_logging — centralized logging setup."""
import importlib.util
import io
import logging
import os
import stat
import subprocess
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

    Under a shared-process run, other test modules may have called
    setup_logging() in the same process, leaving RotatingFileHandlers on the
    root logger.  We strip ALL RotatingFileHandlers before each test so the
    count assertions are stable regardless of test ordering.
    """
    hermes_logging._logging_initialized = False
    # File handlers now live behind the async QueueListener, not on the root
    # logger; tear down any leaked from other tests in this process.
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


@pytest.mark.parametrize("mode,component", [("cli", None), ("gateway", "gateway.log"), ("gui", "gui.log")])
@pytest.mark.parametrize("configured,explicit,minimum", [(None, None, logging.INFO), ("DEBUG", "WARNING", logging.WARNING), ("DEBUG", None, logging.DEBUG)])
def test_repeated_setup_routes_records_once(hermes_home, mode, component, configured, explicit, minimum):
    if configured:
        (hermes_home / "config.yaml").write_text(f"logging:\n  level: {configured}\n", encoding="utf-8")
    for _ in range(2):
        assert hermes_logging.setup_logging(hermes_home=hermes_home, mode=mode, log_level=explicit) == hermes_home / "logs"
    hermes_logging.set_session_context("routing-session")
    sources = ["tools.terminal_tool", "agent.context_compressor", "gateway.run",
               "plugins.platforms.telegram.adapter", "hermes_cli.web_server", "tui_gateway.ws"]
    for index, source in enumerate(sources):
        for level in (logging.DEBUG, logging.INFO, logging.WARNING):
            logging.getLogger(source).log(level, "routing-witness-%s-%s", index, level)
    hermes_logging.flush_log_queue()
    outputs = {path.name: path.read_text(encoding="utf-8-sig") for path in (hermes_home / "logs").glob("*.log")}
    assert set(outputs) == {"agent.log", "errors.log"} | ({component} if component else set())
    for filename, content in outputs.items():
        for index, source in enumerate(sources):
            for level in (logging.DEBUG, logging.INFO, logging.WARNING):
                accepted = {
                    "agent.log": level >= minimum,
                    "errors.log": level >= logging.WARNING,
                    "gateway.log": index in (2, 3) and level >= max(logging.INFO, minimum),
                    "gui.log": index in (4, 5) and level >= max(logging.INFO, minimum),
                }[filename]
                witness = f"routing-witness-{index}-{level}"
                assert content.count(witness) == int(accepted), (mode, filename, witness, content)
        assert "[routing-session]" in content


class TestSetupLogging:
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
        ).read_text(encoding="utf-8-sig")
        default_log = hermes_home / "logs" / "agent.log"
        assert not default_log.exists() or "profile-routed cron record" not in default_log.read_text(encoding="utf-8-sig")

    @pytest.mark.parametrize("launch_redacts, routed_opt_out, routed_redacted", [
        (False, None, True),      # the launch profile opted out, the routed one did not
        (True, "env", False),     # the routed profile opted out in its own .env
    ], ids=["launch-opt-out", "routed-env-opt-out"])
    def test_routed_records_follow_their_own_profiles_redaction_policy(
            self, hermes_home, tmp_path, monkeypatch, launch_redacts, routed_opt_out, routed_redacted):
        """The listener thread formats every record after its profile scope is gone, so a routed profile's own
        agent.log was redacted by the LAUNCH profile's policy: raw credentials if only the launch opted out."""
        from agent import redact
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        monkeypatch.setattr(redact, "_REDACT_ENABLED", launch_redacts)
        monkeypatch.setattr(redact, "_REDACT_ENABLED_BY_HOME", {})
        routed = tmp_path / "profile-b"
        routed.mkdir()
        if routed_opt_out == "config":
            (routed / "config.yaml").write_text("security:\n  redact_secrets: false\n", encoding="utf-8")
        elif routed_opt_out == "env":
            (routed / ".env").write_text("HERMES_REDACT_SECRETS=false\n", encoding="utf-8")
        hermes_logging.setup_logging(hermes_home=hermes_home)
        assert hermes_logging.enable_profile_log_routing([hermes_home, routed]) is True
        routed_secret = "sk-proj-ROUTEDPROFILE" + "b" * 24
        launch_secret = "sk-proj-LAUNCHPROFILE" + "a" * 24

        logger = logging.getLogger("gateway.redaction-routing-test")
        token = set_hermes_home_override(routed)
        try:
            logger.warning("provider rejected key %s", routed_secret)
        finally:
            reset_hermes_home_override(token)
        logger.warning("launch key %s", launch_secret)
        hermes_logging.flush_log_queue()

        assert (routed_secret not in (routed / "logs" / "agent.log").read_text()) is routed_redacted
        assert (launch_secret not in (hermes_home / "logs" / "agent.log").read_text()) is launch_redacts

    def test_release_profile_log_handlers_closes_only_deleted_profile(self, hermes_home, tmp_path):
        """Profile deletion releases its routed log files without disturbing another profile."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        deleted_home = tmp_path / "profile-deleted"
        other_home = tmp_path / "profile-other"
        deleted_home.mkdir()
        other_home.mkdir()
        hermes_logging.setup_logging(hermes_home=hermes_home)
        assert hermes_logging.enable_profile_log_routing(
            [hermes_home, deleted_home, other_home]
        ) is True

        logger = logging.getLogger("agent.profile-delete-log-release")
        token = set_hermes_home_override(deleted_home)
        try:
            logger.warning("deleted profile log handles")
        finally:
            reset_hermes_home_override(token)
        token = set_hermes_home_override(other_home)
        try:
            logger.warning("other profile log handles")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        routers = [
            handler for handler in hermes_logging._queued_file_handlers
            if isinstance(handler, hermes_logging._ProfileRoutingFileHandler)
        ]
        assert len(routers) == 2  # agent.log and errors.log
        assert all(deleted_home.resolve() in handler._profile_handlers for handler in routers)
        assert all(other_home.resolve() in handler._profile_handlers for handler in routers)

        assert hermes_logging.release_profile_log_handlers(deleted_home) == 2

        assert all(deleted_home.resolve() not in handler._profile_handlers for handler in routers)
        assert all(deleted_home.resolve() not in handler._profile_homes for handler in routers)
        assert all(other_home.resolve() in handler._profile_handlers for handler in routers)
        assert "other profile log handles" in (other_home / "logs" / "agent.log").read_text()
        assert "other profile log handles" in (other_home / "logs" / "errors.log").read_text()

    def test_a_second_home_routes_instead_of_stacking_an_unfiltered_handler(self, hermes_home, tmp_path):
        """A dashboard or serve backend builds agents for several profiles in ONE process, and each
        one calls setup_logging for its own home. The second home must get a router — a bare file
        handler beside the first home's would receive every profile's records."""
        from logging.handlers import RotatingFileHandler

        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_logging(hermes_home=profile_home)

        assert not [h for h in hermes_logging._queued_file_handlers if isinstance(h, RotatingFileHandler)], (
            "the second home must not add an unfiltered file handler")

        logger = logging.getLogger("agent.conversation_loop.second-home-test")
        token = set_hermes_home_override(profile_home)
        try:
            logger.info("turn of profile b")
        finally:
            reset_hermes_home_override(token)
        logger.info("turn of the launch profile")
        hermes_logging.flush_log_queue()

        a_log = (hermes_home / "logs" / "agent.log").read_text()
        b_log = (profile_home / "logs" / "agent.log").read_text()
        assert "turn of profile b" in b_log and "turn of profile b" not in a_log
        assert "turn of the launch profile" in a_log and "turn of the launch profile" not in b_log

    def test_setup_for_an_already_routed_home_adds_no_duplicate_writer(self, hermes_home, tmp_path):
        """Routing already on (multiplexed gateway, Desktop cron ticker): a profile's agent starting
        up must not add a second writer for its home on top of the router."""
        from logging.handlers import RotatingFileHandler

        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        hermes_logging.setup_logging(hermes_home=hermes_home)
        assert hermes_logging.enable_profile_log_routing([hermes_home, profile_home]) is True
        hermes_logging.setup_logging(hermes_home=profile_home)

        assert not [h for h in hermes_logging._queued_file_handlers if isinstance(h, RotatingFileHandler)]
        token = set_hermes_home_override(profile_home)
        try:
            logging.getLogger("agent.conversation_loop.routed-home-test").info("once please")
        finally:
            reset_hermes_home_override(token)
        hermes_logging.flush_log_queue()

        assert (profile_home / "logs" / "agent.log").read_text().count("once please") == 1
        assert "once please" not in (hermes_home / "logs" / "agent.log").read_text()

    def test_a_component_log_added_after_routing_is_routed_too(self, hermes_home, tmp_path):
        """setup_logging(mode="gateway") for an already-known home AFTER a second home turned
        routing on: gateway.log must be a routed writer, not a bare handler taking every home."""
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = tmp_path / "profile-b"
        profile_home.mkdir()
        hermes_logging.setup_logging(hermes_home=hermes_home)
        hermes_logging.setup_logging(hermes_home=profile_home)
        hermes_logging.setup_logging(hermes_home=hermes_home, mode="gateway")

        logger = logging.getLogger("gateway.run.routed-component-test")
        token = set_hermes_home_override(profile_home)
        try:
            logger.info("gw-b")
        finally:
            reset_hermes_home_override(token)
        logger.info("gw-a")
        hermes_logging.flush_log_queue()

        a_log = (hermes_home / "logs" / "gateway.log").read_text()
        assert "gw-a" in a_log and "gw-b" not in a_log
        assert "gw-b" in (profile_home / "logs" / "gateway.log").read_text()




    def test_explicit_params_override_config(self, hermes_home):
        """Explicit function params take precedence over config.yaml."""
        import hermes_yaml as yaml
        config = {"logging": {"level": "DEBUG"}}
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(config))

        hermes_logging.setup_logging(hermes_home=hermes_home, log_level="WARNING")

        root = logging.getLogger()
        agent_handlers = [
            h for h in hermes_logging._queued_file_handlers
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



class TestAddRotatingHandler:
    """_add_rotating_handler() is idempotent and creates the directory."""


    def test_no_duplicate_for_same_path(self, tmp_path):
        log_path = tmp_path / "test.log"
        formatter = logging.Formatter("%(message)s")

        hermes_logging._add_rotating_handler(
            log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )
        hermes_logging._add_rotating_handler(
            log_path,
            level=logging.INFO, max_bytes=1024, backup_count=1,
            formatter=formatter,
        )

        rotating_handlers = [
            h for h in hermes_logging._queued_file_handlers
            if isinstance(h, RotatingFileHandler)
        ]
        assert len(rotating_handlers) == 1
        # Clean up


    @pytest.mark.platforms("linux")
    def test_managed_mode_initial_open_sets_group_writable(self, tmp_path):
        log_path = tmp_path / "managed-open.log"
        formatter = logging.Formatter("%(message)s")

        old_umask = os.umask(0o022)
        try:
            with patch("hermes_cli.config.is_managed", return_value=True):
                hermes_logging._add_rotating_handler(
                    log_path,
                    level=logging.INFO, max_bytes=1024, backup_count=1,
                    formatter=formatter,
                )
        finally:
            os.umask(old_umask)

        assert log_path.exists()
        assert stat.S_IMODE(log_path.stat().st_mode) == 0o660



@pytest.mark.linux_only
def test_profile_rollover_keeps_previous_file_access(tmp_path):
    """The replacement must not inherit the gateway's tighter umask."""
    path = tmp_path / "agent.log"
    handler = hermes_logging._ManagedRotatingFileHandler(
        str(path), maxBytes=10, backupCount=1, encoding="utf-8",
    )
    try:
        os.chmod(path, 0o660)
        original = path.stat()
        old_umask = os.umask(0o077)
        try:
            handler.emit(logging.LogRecord("agent", logging.INFO, __file__, 0, "rollover", (), None))
            handler.emit(logging.LogRecord("agent", logging.INFO, __file__, 0, "next", (), None))
        finally:
            os.umask(old_umask)
        replacement = path.stat()
        assert (replacement.st_uid, replacement.st_gid) == (original.st_uid, original.st_gid)
        assert stat.S_IMODE(replacement.st_mode) == stat.S_IMODE(original.st_mode)
    finally:
        handler.close()


@pytest.mark.linux_only
def test_profile_rollover_keeps_shared_group_access_across_uids(tmp_path):
    """An unprivileged rotator must not lock out another shared-group worker."""
    if os.geteuid() != 0:
        pytest.skip("requires root to exercise distinct real Linux UIDs")

    path = tmp_path / "agent.log"
    path.touch()
    os.chown(path, 2000, 4)
    os.chmod(path, 0o660)
    os.chown(tmp_path, -1, 4)
    os.chmod(tmp_path, 0o770)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        rotator = """
import logging, os, sys
from unittest.mock import patch
import hermes_cli.config
from hermes_logging import _ManagedRotatingFileHandler
directory_fd = int(sys.argv[1])
os.setgroups([4])
os.setgid(1000)
os.setuid(1000)
path = f'/proc/self/fd/{directory_fd}/agent.log'
with patch('hermes_cli.config.is_managed', return_value=False):
    handler = _ManagedRotatingFileHandler(path, maxBytes=10, backupCount=1, encoding='utf-8')
try:
    for message in ('rollover', 'next'):
        handler.emit(logging.LogRecord('agent', logging.INFO, '<rollover>', 0, message, (), None))
finally:
    handler.close()
"""
        subprocess.run(
            [sys.executable, "-c", rotator, str(directory_fd)],
            pass_fds=(directory_fd,), check=True,
        )
        assert path.stat().st_gid == 4
        assert stat.S_IMODE(path.stat().st_mode) == 0o660

        worker = """
import os, sys
directory_fd = int(sys.argv[1])
os.setgroups([4])
os.setgid(2001)
os.setuid(2001)
fd = os.open('agent.log', os.O_WRONLY | os.O_APPEND, dir_fd=directory_fd)
try:
    os.write(fd, b'worker append\\n')
finally:
    os.close(fd)
"""
        subprocess.run(
            [sys.executable, "-c", worker, str(directory_fd)],
            pass_fds=(directory_fd,), check=True,
        )
        assert "worker append" in path.read_text()
    finally:
        os.close(directory_fd)


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

    @pytest.mark.platforms("windows")
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

    @pytest.mark.platforms("linux")
    def test_helper_never_matches_off_windows(self):
        # On POSIX the suppression must stay inert: stdlib RotatingFileHandler
        # is in use, so this RuntimeError text is never a CLH lock timeout.
        assert not hermes_logging._is_windows_concurrent_log_lock_timeout(
            RuntimeError("Cannot acquire lock after 20 attempts")
        )

    @pytest.mark.platforms("windows")
    def test_lock_timeout_routed_to_handle_error_is_suppressed(self, tmp_path, capsys):
        """Mirror CLH's real control flow.

        ``concurrent-log-handler``'s ``emit()`` wraps its whole body in
        ``try/except Exception: self.handleError(record)``, so the lock
        RuntimeError raised in ``_do_lock()`` is caught *inside* CLH and routed
        to ``handleError`` with the exception live in ``sys.exc_info()``.  We
        invoke ``handleError`` the same way CLH would and assert no traceback
        reaches stderr (the slash-worker surface) — but the suppression must
        still surface once through the logging system, not stay a black hole.

        Windows-only: the suppression is keyed on the real host, and only on
        Windows is the base handler CLH at all — the fake platform gave us the
        branch without the handler that raises."""
        logger, handler = self._make_logger_and_handler(tmp_path / "agent.log")
        record = logger.makeRecord(
            logger.name, logging.INFO, __file__, 0, "force rollover", (), None,
        )
        captured_warnings: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_warnings.append(record)

        listener = _Capture()
        logging.getLogger("hermes_logging").addHandler(listener)
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(hermes_logging, "_windows_lock_timeout_warned", False)
        try:
            try:
                raise RuntimeError("Cannot acquire lock after 20 attempts")
            except RuntimeError:
                handler.handleError(record)
            try:
                raise RuntimeError("Cannot acquire lock after 20 attempts")
            except RuntimeError:
                handler.handleError(record)

            captured = capsys.readouterr()
            assert "Cannot acquire lock after 20 attempts" not in captured.err
            assert "--- Logging error ---" not in captured.err
            # One-shot warning: the second suppressed emit must not re-warn.
            assert len(captured_warnings) == 1
            assert "concurrent-log-handler" in captured_warnings[0].getMessage()
        finally:
            monkeypatch.undo()
            logging.getLogger("hermes_logging").removeHandler(listener)
            logger.removeHandler(handler)
            handler.close()

    def test_lock_timeout_warning_is_one_shot(self, caplog):
        """The suppressed-timeout warning is exactly-once per process.

        Every emit after the first CLH lock failure raises the same
        RuntimeError, so warn-once is what keeps errors.log from being spammed
        as badly as the stderr noise the suppression replaces."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(hermes_logging, "_windows_lock_timeout_warned", False)
        try:
            with caplog.at_level(logging.WARNING, logger="hermes_logging"):
                hermes_logging._warn_windows_lock_timeout_once()
                hermes_logging._warn_windows_lock_timeout_once()
        finally:
            monkeypatch.undo()
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "concurrent-log-handler" in warnings[0].getMessage()

    @pytest.fixture
    def fresh_logging(self):
        def load():
            spec = importlib.util.spec_from_file_location(
                "_hermes_logging_import_test", hermes_logging.__file__,
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return load

    @pytest.mark.platforms("windows")
    def test_fresh_import_selects_working_concurrent_handler(self, tmp_path, fresh_logging):
        """Native locking must work at import, before the handler class is defined."""
        from concurrent_log_handler import ConcurrentRotatingFileHandler

        module = fresh_logging()
        assert module.RotatingFileHandler is ConcurrentRotatingFileHandler
        assert module._WINDOWS_CLH_FALLBACK is False
        assert module._WINDOWS_CLH_FALLBACK_REASON == ""

        log_path = tmp_path / "agent.log"
        handler = module._new_file_handler(
            log_path, level=logging.INFO, max_bytes=1, backup_count=1,
            formatter=logging.Formatter("%(message)s"),
        )
        try:
            for message in ("before rollover", "after rollover"):
                handler.handle(logging.LogRecord("test", logging.INFO, "", 0, message, (), None))
            handler.flush()
            assert log_path.read_text(encoding="utf-8-sig").splitlines() == ["after rollover"]
            assert (tmp_path / "agent.log.1").read_text(encoding="utf-8-sig").splitlines() == ["before rollover"]
        finally:
            handler.close()

    @pytest.mark.platforms("windows")
    def test_fresh_import_retains_lock_failure_and_writes_without_rollover(
        self, tmp_path, monkeypatch, fresh_logging,
    ):
        """A post-import platform fake misses both handler selection and fallback resets."""
        import portalocker
        from logging.handlers import RotatingFileHandler as StdlibRotatingFileHandler

        failure = ImportError("pywintypes is required for Win32Locker but not found")

        def fail_lock(stream, flags):
            raise failure

        monkeypatch.setattr(portalocker, "lock", fail_lock)
        module = fresh_logging()
        assert module.RotatingFileHandler is StdlibRotatingFileHandler
        assert module._WINDOWS_CLH_FALLBACK is True
        assert module._WINDOWS_CLH_FALLBACK_REASON == repr(failure)

        log_path = tmp_path / "agent.log"
        handler = module._new_file_handler(
            log_path, level=logging.INFO, max_bytes=1, backup_count=1,
            formatter=logging.Formatter("%(message)s"),
        )
        try:
            assert handler.maxBytes == 0
            assert handler.backupCount == 0
            for message in ("first message", "second message"):
                handler.handle(logging.LogRecord("test", logging.INFO, "", 0, message, (), None))
            handler.flush()
            assert log_path.read_text(encoding="utf-8-sig").splitlines() == ["first message", "second message"]
            assert not list(tmp_path.glob("agent.log.*"))
        finally:
            handler.close()



class TestReadLoggingConfig:
    """_read_logging_config() reads from config.yaml."""

    def test_returns_none_when_no_config(self, hermes_home):
        level, max_size, backup = hermes_logging._read_logging_config()
        assert level is None
        assert max_size is None
        assert backup is None

    def test_reads_logging_section(self, hermes_home):
        import hermes_yaml as yaml
        config = {"logging": {"level": "DEBUG", "max_size_mb": 10, "backup_count": 5}}
        (hermes_home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

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
            assert log_path.read_text(encoding="utf-8-sig") == "before rotation\n"

            # External rotation (NOT via handler.doRollover()).
            os.rename(log_path, rotated)
            assert not log_path.exists()

            self._emit(handler, "after rotation")

            # The new write should land in a freshly recreated gateway.log,
            # not appended to the rotated backup.
            assert log_path.exists(), "handler did not recreate gateway.log"
            assert log_path.read_text(encoding="utf-8-sig") == "after rotation\n"
            assert rotated.read_text(encoding="utf-8-sig") == "before rotation\n"
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

            with open(log_path, "w", encoding="utf-8"):
                pass  # truncate to zero
            assert log_path.stat().st_size == 0

            self._emit(handler, "after truncate")
            assert log_path.read_text(encoding="utf-8-sig") == "after truncate\n"
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
        assert "BEFORE rotation" in gw_path.read_text(encoding="utf-8-sig")

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
        assert "AFTER rotation" in gw_path.read_text(encoding="utf-8-sig")
        assert "AFTER rotation" not in rotated.read_text(encoding="utf-8-sig")


def test_eio_from_file_handler_names_the_path_once_then_recovers(tmp_path, capsys):
    """A failing log destination is named once (no per-record traceback) and writes resume
    once the file is reachable again."""

    class _SickStream(io.TextIOBase):
        def writable(self):
            return True

        def write(self, *_a):
            raise OSError(5, "Input/output error")

        seek = tell = flush = write

    path = tmp_path / "agent.log"
    handler = hermes_logging._ManagedRotatingFileHandler(
        str(path), maxBytes=1024, backupCount=1, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler.stream.close()
        handler.stream = _SickStream()
        for i in range(5):
            handler.handle(logging.LogRecord("t", logging.INFO, __file__, 0, f"sick {i}", (), None))
        err = capsys.readouterr().err
        assert "--- Logging error ---" not in err
        assert err.count(str(path)) == 1 and "Input/output error" in err

        # Stream dropped, so the next emit reopens the real file and logging resumes.
        handler.handle(logging.LogRecord("t", logging.INFO, __file__, 0, "recovered", (), None))
        assert "recovered" in path.read_text(encoding="utf-8")
    finally:
        handler.close()


def test_eio_after_successful_reopen_still_names_the_path_once(tmp_path, capsys):
    """The reported case: open() succeeds but every write/seek/flush raises EIO. Reopening must
    not re-arm the notice, or a stuck device prints the path once per record."""

    class _SickStream(io.TextIOBase):
        def writable(self):
            return True

        def write(self, *_a):
            raise OSError(5, "Input/output error")

        seek = tell = flush = write

    path = tmp_path / "agent.log"
    handler = hermes_logging._ManagedRotatingFileHandler(
        str(path), maxBytes=1024, backupCount=1, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        handler._builtin_open = lambda *_a, **_kw: _SickStream()
        handler.stream.close()
        handler.stream = _SickStream()
        for i in range(25):
            handler.handle(logging.LogRecord("t", logging.INFO, __file__, 0, f"sick {i}", (), None))
        err = capsys.readouterr().err
        assert "--- Logging error ---" not in err
        assert err.count(str(path)) == 1
    finally:
        handler.close()


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
        handler = logging.StreamHandler(result)
        handler.handle(logging.LogRecord("unicode", logging.INFO, "", 0, "Session — 日本", (), None))
        handler.flush()
        assert fake.buffer.getvalue() == "Session — 日本\n".encode("utf-8")



class TestLineBufferPipedStdout:
    """A piped stdout is line-buffered so headless log streams track the
    agent loop incrementally (#92281); a TTY stdout is left alone."""

    def _fake_stdout(self, isatty: bool):
        from unittest.mock import MagicMock

        stream = MagicMock()
        stream.isatty.return_value = isatty
        stream.reconfigure = MagicMock()
        return stream

    def test_tty_none_or_reconfigure_less_stdout_left_alone(self, monkeypatch):
        from types import SimpleNamespace

        tty = self._fake_stdout(isatty=True)
        monkeypatch.setattr(sys, "stdout", tty)
        hermes_logging._line_buffer_piped_stdout()
        tty.reconfigure.assert_not_called()

        monkeypatch.setattr(sys, "stdout", None)
        hermes_logging._line_buffer_piped_stdout()  # must not raise
        # A stream without reconfigure() (e.g. a print-redirect shim).
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: False))
        hermes_logging._line_buffer_piped_stdout()

    def test_setup_logging_applies_it_to_piped_stdout(self, tmp_path, monkeypatch):
        stream = self._fake_stdout(isatty=False)
        monkeypatch.setattr(sys, "stdout", stream)
        hermes_logging.setup_logging(hermes_home=tmp_path, force=True)
        # setup_logging runs per AIAgent build: a second call must not re-flush/reconfigure.
        hermes_logging.setup_logging(hermes_home=tmp_path, force=True)
        stream.reconfigure.assert_called_once_with(line_buffering=True)
