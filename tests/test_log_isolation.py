"""Test-log isolation guard (2026-06-27).

Regression test for the live-log pollution that tripped the compaction-stats-watch
cron with false pages: when tests exercise modules that call
``hermes_logging.setup_logging()`` (directly or via building a real AIAgent), the
root logger gets a ``RotatingFileHandler`` pointing at the REAL ``~/.hermes/logs/
agent.log`` — so WARNING records emitted by test code (e.g.
``COMPACTION_STATS_RECONCILE_FAILED``) land in the production log file and a
watcher pages on them.

The conftest autouse ``_isolate_log_handlers`` fixture must ensure the root logger
has NO file handler writing outside the per-test tempdir.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _root_file_handler_paths():
    paths = []
    for h in logging.getLogger().handlers:
        base = getattr(h, "baseFilename", None)
        if base:
            paths.append(base)
    return paths


def test_no_root_file_handler_outside_tmp(tmp_path):
    """No root-logger file handler may point at a path outside the test sandbox.
    The real ~/.hermes/logs/agent.log is the canonical leak target."""
    real_logs = str(Path.home() / ".hermes" / "logs")
    for p in _root_file_handler_paths():
        assert real_logs not in p, (
            f"root logger has a file handler at {p} (real ~/.hermes/logs) — "
            "test logging is leaking into the production log"
        )


def test_setup_logging_during_test_does_not_attach_real_handler(tmp_path, monkeypatch):
    """Even if a module calls setup_logging() mid-test, the guard keeps the real
    log path off the root logger (records can't reach ~/.hermes/logs)."""
    import hermes_logging

    # HERMES_HOME is already the per-test tempdir (via _hermetic_environment),
    # so setup_logging writes into tmp — assert it never targets the real home.
    hermes_logging._logging_initialized = False  # force re-attach
    hermes_logging.setup_logging()

    real_logs = str(Path.home() / ".hermes" / "logs")
    for p in _root_file_handler_paths():
        assert real_logs not in p, f"setup_logging attached a real-home handler: {p}"


def test_warning_during_test_does_not_write_real_agent_log(tmp_path):
    """A WARNING logged by 'agent.conversation_compression' during a test must not
    append to the real ~/.hermes/logs/agent.log (the exact watcher-tripping leak)."""
    real_agent_log = Path.home() / ".hermes" / "logs" / "agent.log"
    before = real_agent_log.stat().st_size if real_agent_log.exists() else None

    logging.getLogger("agent.conversation_compression").warning(
        "COMPACTION_STATS_RECONCILE_FAILED in-turn TEST-ISOLATION-PROBE should-not-leak"
    )

    if before is not None and real_agent_log.exists():
        after = real_agent_log.stat().st_size
        # the probe string must not be in the real log
        tail = real_agent_log.read_bytes()[-4096:]
        assert b"TEST-ISOLATION-PROBE" not in tail, (
            "a test WARNING leaked into the real ~/.hermes/logs/agent.log"
        )


def test_sibling_sandbox_dir_handler_is_stripped(tmp_path):
    """Greptile #114: a handler in a SIBLING sandbox dir (path that shares a string
    prefix but is not actually inside) must be stripped, not kept."""
    import sys as _sys
    from logging.handlers import RotatingFileHandler as _RFH

    _conftest = _sys.modules.get("conftest") or _sys.modules.get("tests.conftest")
    assert _conftest is not None

    # sandbox = .../tX ; sibling = .../tX1 (str-prefix match, but NOT inside)
    sandbox = tmp_path / "t0"
    sandbox.mkdir()
    sibling = tmp_path / "t01"
    (sibling / "logs").mkdir(parents=True)
    sib_log = sibling / "logs" / "agent.log"

    root = logging.getLogger()
    h = _RFH(str(sib_log), maxBytes=1024, backupCount=0, delay=True)
    root.addHandler(h)
    try:
        _conftest._strip_nonsandbox_file_handlers(str(sandbox))
        assert h not in root.handlers, "sibling-dir handler wrongly kept (str-prefix bug)"
    finally:
        if h in root.handlers:
            root.removeHandler(h)
        h.close()


def test_in_sandbox_handler_is_kept(tmp_path):
    """A handler genuinely INSIDE the sandbox is kept (the guard must not nuke
    legitimate per-test log handlers)."""
    import sys as _sys
    from logging.handlers import RotatingFileHandler as _RFH

    _conftest = _sys.modules.get("conftest") or _sys.modules.get("tests.conftest")
    assert _conftest is not None

    (tmp_path / "logs").mkdir()
    in_log = tmp_path / "logs" / "agent.log"
    root = logging.getLogger()
    h = _RFH(str(in_log), maxBytes=1024, backupCount=0, delay=True)
    root.addHandler(h)
    try:
        _conftest._strip_nonsandbox_file_handlers(str(tmp_path))
        assert h in root.handlers, "in-sandbox handler wrongly stripped"
    finally:
        if h in root.handlers:
            root.removeHandler(h)
        h.close()


def test_guard_strips_a_real_home_handler_attached_mid_session():
    """RED-proof of the guard: even if something attaches a RotatingFileHandler at
    the REAL ~/.hermes/logs (the intermittent import-order leak), the autouse
    _isolate_log_handlers fixture must have removed it before this test body runs,
    so a WARNING here cannot reach the real file.

    We simulate the leaked state by attaching a handler at the real path, then
    assert the guard (which runs at the START of every test via autouse) would
    catch it — by re-invoking the strip and confirming it removes our handler.
    """
    from logging.handlers import RotatingFileHandler as _RFH

    real_log = Path.home() / ".hermes" / "logs" / "agent.log"
    root = logging.getLogger()
    leaked = None
    try:
        # only attach if the real dir exists (don't create it in CI sandboxes)
        if real_log.parent.exists():
            leaked = _RFH(str(real_log), maxBytes=1024, backupCount=0, delay=True)
            root.addHandler(leaked)
            # the guard helper must classify + strip it. conftest is loaded by
            # pytest as a plugin; reach its module object via sys.modules.
            import sys as _sys
            _conftest = _sys.modules.get("conftest") or _sys.modules.get("tests.conftest")
            assert _conftest is not None, "conftest module not importable"
            _conftest._strip_nonsandbox_file_handlers()
            assert leaked not in root.handlers, "guard failed to strip the real-home handler"
    finally:
        if leaked is not None and leaked in root.handlers:
            root.removeHandler(leaked)
        if leaked is not None:
            leaked.close()
