"""Fallback-chain skip paths must leave a trace at the default log level (#87815).

A multi-hop chain can silently reorder itself at runtime; every skip reason needs to be
visible in agent.log. Regression pins:

  1. An entry memoized unavailable earlier in the session logs at WARNING (was logger.debug —
     invisible at the default level, the only untraceable skip in the walk).
  2. A malformed entry (missing provider or model) logs a WARNING instead of returning silently.
"""

import logging

from agent.chat_completion_helpers import _should_skip_fallback_candidate

_LOGGER_NAME = "agent.chat_completion_helpers"


def _records(caplog, level: int) -> list:
    return [r for r in caplog.records if r.name == _LOGGER_NAME and r.levelno >= level]


class TestFallbackSkipLogging:
    def test_previously_unavailable_skip_logs_warning(self, caplog):
        agent = object()  # the memoized branch touches nothing else on the agent
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            skipped = _should_skip_fallback_candidate(
                agent, {}, ("openrouter", "deepseek-v4-flash", ""),
                "openrouter", "deepseek-v4-flash",
                {("openrouter", "deepseek-v4-flash", "")})
        assert skipped is True
        warnings = _records(caplog, logging.WARNING)
        assert any("previously marked unavailable" in r.getMessage() for r in warnings)

    def test_malformed_entry_skip_logs_warning(self, caplog):
        agent = object()
        fb = {"model": "deepseek-v4-flash"}  # provider missing
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            skipped = _should_skip_fallback_candidate(
                agent, fb, ("", "deepseek-v4-flash", ""), "", "deepseek-v4-flash", set())
        assert skipped is True
        warnings = _records(caplog, logging.WARNING)
        assert any("malformed" in r.getMessage() for r in warnings)

    def test_empty_model_entry_skip_logs_warning(self, caplog):
        agent = object()
        fb = {"provider": "openrouter"}  # model missing
        with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
            skipped = _should_skip_fallback_candidate(
                agent, fb, ("openrouter", "", ""), "openrouter", "", set())
        assert skipped is True
        warnings = _records(caplog, logging.WARNING)
        assert any("malformed" in r.getMessage() for r in warnings)
