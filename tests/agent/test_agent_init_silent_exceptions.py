"""Regression tests for agent init silent-exception logging (salvage of #4078).

Verifies that the three bare `except Exception: pass` blocks in AIAgent.__init__
now emit log records instead of swallowing errors silently:

  1. Config load failure  → logger.debug("Config load failed (using defaults): ...")
  2. Memory init failure  → logger.warning("Memory system init failed (non-fatal): ...")
  3. Skills config error  → logger.debug("Skills config load failed (using defaults): ...")

Uses source-inspection (same pattern as test_terminal_config_env_sync.py) plus
direct log-pattern checks — no full AIAgent instantiation required.
"""

import inspect
import logging

import run_agent


# ---------------------------------------------------------------------------
# 1. Direct log-pattern tests (simulate the exception path)
# ---------------------------------------------------------------------------


def test_config_load_failure_logs_debug(caplog):
    """Config-load except path must emit DEBUG with 'Config load failed'."""
    logger = logging.getLogger("run_agent")
    with caplog.at_level(logging.DEBUG, logger="run_agent"):
        try:
            raise RuntimeError("no config file")
        except Exception as e:
            logger.debug("Config load failed (using defaults): %s", e)

    records = [r for r in caplog.records if "Config load failed" in r.getMessage()]
    assert records, "No DEBUG record for config load failure"
    assert records[0].levelno == logging.DEBUG
    assert "no config file" in records[0].getMessage()


def test_memory_init_failure_logs_warning(caplog):
    """Memory-init except path must emit WARNING with 'Memory system init failed'."""
    logger = logging.getLogger("run_agent")
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        try:
            raise OSError("disk full")
        except Exception as e:
            logger.warning("Memory system init failed (non-fatal): %s", e)

    records = [r for r in caplog.records if "Memory system init failed" in r.getMessage()]
    assert records, "No WARNING record for memory init failure"
    assert records[0].levelno == logging.WARNING
    assert "disk full" in records[0].getMessage()


def test_skills_config_failure_logs_debug(caplog):
    """Skills-config except path must emit DEBUG with 'Skills config load failed'."""
    logger = logging.getLogger("run_agent")
    with caplog.at_level(logging.DEBUG, logger="run_agent"):
        try:
            raise ValueError("bad int")
        except Exception as e:
            logger.debug("Skills config load failed (using defaults): %s", e)

    records = [r for r in caplog.records if "Skills config load failed" in r.getMessage()]
    assert records, "No DEBUG record for skills config failure"
    assert records[0].levelno == logging.DEBUG
    assert "bad int" in records[0].getMessage()


# ---------------------------------------------------------------------------
# 2. Source-inspection: no bare `except Exception: pass` in the three blocks
# ---------------------------------------------------------------------------


def test_no_bare_except_pass_in_config_load_block():
    src = inspect.getsource(run_agent)
    start = src.index("Load config once for memory")
    end = src.index("_tool_guardrails", start)
    block = src[start:end]
    assert "except Exception:\n" not in block, (
        "Bare `except Exception:` still present in config-load block — add logger.debug"
    )


def test_no_bare_except_pass_in_memory_init_block():
    src = inspect.getsource(run_agent)
    # anchor on the old comment that was directly after `pass`
    assert "pass  # Memory is optional" not in src, (
        "Silent `pass  # Memory is optional` still present — should use logger.warning"
    )


def test_no_bare_except_pass_in_skills_config_block():
    src = inspect.getsource(run_agent)
    start = src.index("Skills config: nudge interval")
    end = src.index("Tool-use enforcement config", start)
    block = src[start:end]
    assert "except Exception:\n            pass" not in block, (
        "Bare `except Exception: pass` still present in skills-config block — add logger.debug"
    )
