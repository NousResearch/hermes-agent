"""Invariant tests for the declarative busy_policy on CommandDef.

Guards the contract introduced by the Guard-2 refactor (gateway/run.py):
every command's mid-run behavior is declared on its CommandDef via
``busy_policy`` / ``busy_handler`` and the historical
``ACTIVE_SESSION_BYPASS_COMMANDS`` frozenset is DERIVED from the registry
rather than hand-maintained.
"""

from hermes_cli.commands import (
    ACTIVE_SESSION_BYPASS_COMMANDS,
    COMMAND_REGISTRY,
    is_interrupt_then_dispatch,
)

def test_bypass_set_is_derived_from_registry():
    expected = frozenset(
        cmd.name for cmd in COMMAND_REGISTRY if cmd.busy_policy != "reject"
    )
    assert ACTIVE_SESSION_BYPASS_COMMANDS == expected


def test_interrupt_then_dispatch_class():
    # The cancel-handoff class (Guard 1, gateway/platforms/base.py) must
    # contain exactly the /stop and /new (alias /reset) commands today.
    assert is_interrupt_then_dispatch("stop")
    assert is_interrupt_then_dispatch("new")
    assert is_interrupt_then_dispatch("reset")  # alias of /new
    assert not is_interrupt_then_dispatch("model")
    assert not is_interrupt_then_dispatch("status")
    assert not is_interrupt_then_dispatch(None)
    assert not is_interrupt_then_dispatch("not-a-command")


