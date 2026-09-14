"""The install namespace is resolved ONCE per process, not once per dispatch tick.

``install_namespaces()`` sits on the dispatcher's per-tick path, but its inputs
are deploy-time facts: ``HERMES_KANBAN_NAMESPACE``, ``kanban.namespace`` in
``config.yaml``, or the OS user. Re-parsing ``config.yaml`` every tick bought
nothing, so the resolution is memoised for the life of the process — the gateway
reads it at start (changing the namespace is a deliberate act that takes a
restart) and a dispatched worker is a fresh process, so it reads at spawn.

These tests pin the observable behaviour, not the implementation: the config read
happens once, the documented reset hook makes it happen again, and the
``HERMES_KANBAN_NAMESPACE`` override stays first in the resolution order and
stays effective in-process.
"""
from __future__ import annotations

import pytest

from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture()
def config_reads(monkeypatch):
    """Count ``config.yaml`` loads, with the env override cleared.

    Hands back the list the fake appends to, and guarantees every case starts
    from a clean memo (the process-wide lifetime is the thing under test).
    """
    calls: list[int] = []

    def fake_load():
        calls.append(1)
        return {"kanban": {"namespace": "em"}}

    monkeypatch.delenv("HERMES_KANBAN_NAMESPACE", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", fake_load)
    kbd._reset_namespace_cache()
    yield calls
    kbd._reset_namespace_cache()


def test_config_is_read_once_per_process(config_reads):
    """50 ticks, one parse — the point of the change."""
    assert kbd.install_namespaces() == ("em", frozenset({"em"}))
    for _ in range(50):
        assert kbd.install_namespaces() == ("em", frozenset({"em"}))
    assert kbd.canonical_install_namespace() == "em"
    assert len(config_reads) == 1


def test_reset_hook_re_reads_config(config_reads):
    """``_reset_namespace_cache()`` is the documented escape hatch for tests."""
    assert kbd.install_namespaces() == ("em", frozenset({"em"}))
    assert len(config_reads) == 1
    kbd._reset_namespace_cache()
    assert kbd.install_namespaces() == ("em", frozenset({"em"}))
    assert len(config_reads) == 2


def test_env_override_stays_first_and_effective_in_process(config_reads, monkeypatch):
    """The override is part of the memo key: still first, still live, still no
    config parse while it is set."""
    assert kbd.install_namespaces() == ("em", frozenset({"em"}))

    monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", "yummi, em")
    assert kbd.install_namespaces() == ("yummi", frozenset({"yummi", "em"}))
    assert kbd.canonical_install_namespace() == "yummi"
    assert len(config_reads) == 1, "the override path must not touch config.yaml"

    monkeypatch.delenv("HERMES_KANBAN_NAMESPACE")
    kbd._reset_namespace_cache()
    assert kbd.install_namespaces() == ("em", frozenset({"em"}))
    assert len(config_reads) == 2


def test_memoised_refusal_is_still_a_refusal(config_reads, monkeypatch):
    """An unresolvable override memoises as the refusal — caching must not turn a
    visible ``namespace_unresolved`` into a guess."""
    monkeypatch.setenv("HERMES_KANBAN_NAMESPACE", "not a token!")
    assert kbd.install_namespaces() == (None, frozenset())
    assert kbd.install_namespaces() == (None, frozenset())
    assert kbd.canonical_install_namespace() is None
    assert config_reads == []
