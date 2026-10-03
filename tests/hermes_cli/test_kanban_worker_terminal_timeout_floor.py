"""Worker terminal-timeout overrides must never silently LOWER the child's timeout.

Same failure class as #85809 ("TERMINAL_TIMEOUT silently accepted, then every command reports a
misleading 'timed out after Ns'"): the dispatcher compared ``max_runtime_seconds`` against the
RAW child env value, where "unset" parsed as 0 instead of the terminal tool's own default — so a
task with a 30s runtime cap dispatched ``TERMINAL_TIMEOUT=1`` and every command in that worker
failed after one second. The override is documented to *raise only* so a long command is not
killed by the generic default first; lowering is never intended.
"""

import logging

from hermes_cli.kanban_db_dispatch import (
    KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS,
    _worker_terminal_timeout_env,
)


def test_short_runtime_does_not_lower_the_child_default():
    """An unset TERMINAL_TIMEOUT means the tool's 180s default, not zero."""
    assert _worker_terminal_timeout_env(60, None) is None
    assert _worker_terminal_timeout_env(200, None) is None


def test_runtime_at_the_grace_floor_never_dispatches_a_one_second_timeout(caplog):
    """A runtime cap at/below the 30s grace cannot host a usable timeout: keep the default and
    say so instead of dispatching TIMEOUT=1."""
    with caplog.at_level(logging.WARNING):
        assert _worker_terminal_timeout_env(10, None) is None
        assert _worker_terminal_timeout_env(KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS, None) is None
    messages = [r.getMessage() for r in caplog.records]
    assert any("max_runtime_seconds=10" in m and "terminal timeout" in m.lower() for m in messages), messages


def test_runtime_above_the_default_still_raises_the_override():
    """The documented behaviour is preserved: raise when the runtime cap exceeds the default."""
    assert _worker_terminal_timeout_env(600, None) == "570"
    assert _worker_terminal_timeout_env(600, "300") == "570"  # explicit smaller value is raised


def test_explicit_values_at_or_above_the_target_are_left_alone():
    assert _worker_terminal_timeout_env(600, "700") is None
    assert _worker_terminal_timeout_env(600, "570") is None


def test_no_runtime_cap_or_non_positive_cap_yields_no_override():
    assert _worker_terminal_timeout_env(None, None) is None
    assert _worker_terminal_timeout_env(0, None) is None
    assert _worker_terminal_timeout_env(-5, None) is None


def test_foreground_var_uses_its_own_default_as_the_baseline():
    """TERMINAL_MAX_FOREGROUND_TIMEOUT defaults to 600s, not 180s: a 600s runtime cap must not
    *lower* it to 570s."""
    assert _worker_terminal_timeout_env(600, None, default_seconds=600) is None
    assert _worker_terminal_timeout_env(1000, None, default_seconds=600) == "970"
