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
    VALID_BUSY_POLICIES,
    is_interrupt_then_dispatch,
    should_bypass_active_session,
)

# The hand-written frozenset as it existed before the busy_policy refactor.
# This is a behavior contract, not a snapshot: the derived set must remain a
# SUPERSET of these names (each had an explicit mid-run handler in the old
# Guard-2 if-chain and must keep bypassing the busy-reject catch-all).
_HISTORICAL_BYPASS_NAMES = frozenset(
    {
        "agents",
        "approve",
        "background",
        "commands",
        "deny",
        "help",
        "new",
        "profile",
        "queue",
        "restart",
        "status",
        "steer",
        "stop",
        "update",
        "version",
    }
)


def test_every_command_has_valid_busy_policy():
    bad = [
        (cmd.name, cmd.busy_policy)
        for cmd in COMMAND_REGISTRY
        if cmd.busy_policy not in VALID_BUSY_POLICIES
    ]
    assert not bad, f"Commands with invalid busy_policy: {bad}"


def test_derived_bypass_set_covers_historical_names():
    missing = _HISTORICAL_BYPASS_NAMES - ACTIVE_SESSION_BYPASS_COMMANDS
    assert not missing, (
        "Commands lost their mid-run bypass (busy_policy regressed to "
        f"'reject'): {sorted(missing)}"
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


def test_bypass_names_resolve_and_bypass_guard1():
    # Every derived bypass name must be a resolvable command (Guard 1's
    # should_bypass_active_session admits all resolvable commands).
    for name in ACTIVE_SESSION_BYPASS_COMMANDS:
        assert should_bypass_active_session(name), name
