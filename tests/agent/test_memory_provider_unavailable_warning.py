"""Regression tests for NousResearch/hermes-agent#2765.

A memory provider configured via ``memory.provider`` but reporting
``is_available() == False`` (e.g. missing credentials, or a systemd/gateway
service that didn't inherit ``~/.hermes/.env``) used to be dropped silently.
``agent_init`` now emits a one-time, deduped warning instead.
"""

import logging

from agent import agent_init


def test_warns_once_and_dedupes(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("hindsight")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1, "should warn exactly once per provider (gateway dedup)"
    msg = warnings[0].getMessage()
    assert "hindsight" in msg
    assert "hermes memory status" in msg
    assert ".env" in msg  # surfaces the systemd/gateway root cause


def test_distinct_providers_each_warn(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("mem0")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 2
