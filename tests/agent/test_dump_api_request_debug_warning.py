"""A failed request dump logs a warning even when verbose_logging is off."""

import logging
from types import SimpleNamespace

from agent.agent_runtime_helpers import dump_api_request_debug


def test_dump_failure_warns_without_verbose_logging(caplog):
    agent = SimpleNamespace(verbose_logging=False)
    with caplog.at_level(logging.WARNING):
        assert dump_api_request_debug(agent, None, reason="test") is None
    assert "Failed to dump API request debug payload" in caplog.text
    assert caplog.records[-1].exc_info, "the warning must keep the traceback"
