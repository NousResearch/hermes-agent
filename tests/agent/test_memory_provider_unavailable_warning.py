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


def test_provider_reason_is_appended(caplog):
    # A provider's unavailable_reason() (e.g. the local_embedded install hint,
    # #7718) reaches the user through this warning — the only path that runs
    # when the provider is unavailable and thus never initialized.
    agent_init._warned_unavailable_providers.clear()
    hint = "Install the embedded runtime with: uv pip install hindsight-all."
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight", hint)

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1
    assert hint in warnings[0].getMessage()


def test_empty_reason_adds_no_trailing_noise(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight", "")

    msg = next(r.getMessage() for r in caplog.records if "unavailable" in r.getMessage())
    # No dangling separator when there's no provider-specific hint.
    assert msg.rstrip().endswith("service environment.")
