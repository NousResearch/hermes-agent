"""Closing an agent returns freed pages without a forced full collection.

In a gateway every agent close (idle-cache eviction, /new, session end) trims memory, and a full
``gc.collect()`` there walks the whole shared heap with the GIL held: 250-750 ms on a live
multi-million-object gateway, freezing every other session once a minute.
"""

from unittest.mock import Mock

from hermes_agent.hermes_cli import mem_trim


def test_agent_close_trim_returns_pages_without_a_full_collection(monkeypatch):
    from hermes_agent.run_agent import AIAgent

    collect = Mock()
    trim = Mock(return_value=1)
    monkeypatch.setattr(mem_trim.gc, "collect", collect)
    monkeypatch.setattr(mem_trim, "_malloc_trim", trim)
    monkeypatch.setattr(mem_trim, "_probe_done", True)
    monkeypatch.setattr(mem_trim, "_last_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_last_alloc_trim_monotonic", 0.0)
    monkeypatch.setattr(mem_trim, "_config_settings", lambda: (True, 60.0, 1, 0.0))

    AIAgent._trim_process_memory()

    collect.assert_not_called()
    trim.assert_called_once_with(0)
