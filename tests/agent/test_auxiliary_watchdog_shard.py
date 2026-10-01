"""Regression tests for the auxiliary watchdog shard (Part of #125186).

The stream-watchdog / forward-progress / cancellation cluster moved from
``agent/auxiliary_client.py`` to ``agent/auxiliary_watchdog.py``. These tests pin the
binding contract of that move (they are deliberately NOT snapshots of line counts or
symbol inventories):

1. the sibling is importable WITHOUT the facade (facade stays the late-imported
   dependency, so the module graph stays acyclic — agent/AGENTS.md facade+siblings rule);
2. the facade re-exports every moved name at module level, so callers, gateway code, and
   monkeypatch seams that bind ``agent.auxiliary_client.<name>`` keep working;
3. both modules resolve to the SAME objects (identity, not just equality).
"""

import sys
import types
from unittest.mock import patch

import pytest


def _purge(mod_name: str) -> None:
    """Drop a module (and the facade) from sys.modules so the import fires for real."""
    sys.modules.pop(mod_name, None)
    sys.modules.pop("agent.auxiliary_client", None)


@pytest.fixture
def fresh_watchdog():
    sys.modules.pop("agent.auxiliary_watchdog", None)
    sys.modules.pop("agent.auxiliary_client", None)
    yield
    # leaving both purged is fine; the next import rebuilds them


def test_watchdog_imports_without_facade(fresh_watchdog):
    """Facade+siblings rule: a sibling must be importable standalone (no import cycle)."""
    _purge("agent.auxiliary_watchdog")
    mod = __import__("agent.auxiliary_watchdog", fromlist=["x"])
    # module-level import of the facade would be the cycle; late imports inside functions are the pattern
    mod_level = [n for n in vars(mod).values() if isinstance(n, types.ModuleType)]
    assert not any(m.__name__ == "agent.auxiliary_client" for m in mod_level)
    # and the public surface works: default no-progress timeout must exist and be positive
    assert mod._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS > 0


@pytest.mark.parametrize(
    "name",
    [
        "_CodexStreamGuard",
        "_close_quietly",
        "AuxiliaryExplicitCancellation",
        "aux_progress_hook",
        "aux_stream_deadline",
        "aux_interrupt_protection",
        "_notify_aux_progress",
        "_notify_aux_dispatch",
        "_notify_aux_provider_response",
        "_notify_aux_timing_response",
        "_aux_progress_active",
        "_current_aux_stream_deadline",
        "_anthropic_aux_stream_event_hook",
        "_get_task_no_progress_timeout",
        "_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS",
        "_aux_progress",
        "_aux_dispatch",
        "_aux_provider_response",
        "_aux_stream_deadline",
        "_aux_interrupt_protection",
    ],
)
def test_facade_reexports_moved_names(fresh_watchdog, name):
    """Callers and tests keep binding the facade names (patch seams survive the shard)."""
    _purge("agent.auxiliary_client")
    import agent.auxiliary_client as facade

    assert hasattr(facade, name), f"facade lost re-export: {name}"
    from agent import auxiliary_watchdog as wd

    assert getattr(facade, name) is getattr(wd, name), f"{name} is not the same object"


def test_facade_timeout_patch_reaches_moved_guard(fresh_watchdog):
    """The facade's historical timeout patch seam still controls the moved consumer."""
    _purge("agent.auxiliary_client")
    import agent.auxiliary_client as facade
    from agent.auxiliary_watchdog import _CodexStreamGuard

    with patch.object(facade, "_AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS", 0.3):
        guard = _CodexStreamGuard(object(), total_timeout=None)

    assert guard.no_progress_timeout == 0.3
