"""Regression tests for NousResearch/hermes-agent#2765.

A memory provider configured via ``memory.provider`` but reporting
``is_available() == False`` (e.g. missing credentials, or a systemd/gateway
service that didn't inherit ``~/.hermes/.env``) used to be dropped silently.
``agent_init`` now emits a one-time, deduped warning instead.
"""

import logging

import pytest

from agent import agent_init_memory_provider as agent_init


def test_warns_once_and_dedupes(caplog):
    agent_init._warned_unavailable_providers.clear()
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent_init._warn_memory_provider_unavailable("hindsight")
        agent_init._warn_memory_provider_unavailable("hindsight")

    warnings = [r for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(warnings) == 1, "should warn exactly once per provider (gateway dedup)"
    assert "hindsight" in warnings[0].getMessage()


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


def test_parked_provider_is_reported_and_never_reinstalled(monkeypatch):
    """A provider that plugins.disabled parks (by the user, or by an update that could not install it)
    is reported once with how to turn it back on and the update's reason; it is never re-installed."""
    from hermes_cli import memory_provider_migration
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    plugin = home / "plugins" / "parkedmem"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text("from agent.memory_provider import MemoryProvider\n", encoding="utf-8")
    (home / "config.yaml").write_text(
        "memory:\n  provider: parkedmem\nplugins:\n  disabled: [parkedmem]\n  disabled_by_update:\n"
        "    parkedmem: {enabled: true, reason: requires Python <3.14}\n", encoding="utf-8")
    monkeypatch.setattr(memory_provider_migration, "recover_at_startup",
                        lambda *args, **kwargs: pytest.fail("a parked provider must not be re-installed"))
    agent_init._warned_unavailable_providers.clear()
    said: list[str] = []

    assert agent_init.resolve_memory_provider("parkedmem", say=said.append) is None
    assert agent_init.resolve_memory_provider("parkedmem", say=said.append) is None
    assert len(said) == 1
    assert "hermes plugins enable parkedmem" in said[0] and "requires Python <3.14" in said[0]


def test_provider_that_still_fails_after_recovery_is_reported(monkeypatch):
    """An install notice from startup recovery does not explain a provider that still cannot load:
    the user is told external memory is off."""
    from hermes_cli import memory_provider_migration
    from hermes_constants import get_hermes_home

    plugin = get_hermes_home() / "plugins" / "brokenmem"

    def recover(name, say=None):
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text(
            "from agent.memory_provider import MemoryProvider\nimport hermes_missing_package_for_test\n", encoding="utf-8")
        say(f"Installed memory provider '{name}'.")
        return True

    monkeypatch.setattr(memory_provider_migration, "recover_at_startup", recover)
    agent_init._warned_unavailable_providers.clear()
    said: list[str] = []

    assert agent_init.resolve_memory_provider("brokenmem", say=said.append) is None
    assert len(said) == 2 and "failed to load" in said[1]

