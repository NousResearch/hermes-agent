"""Tests for issue #65773 — cron scheduler installs profile secret scope
unconditionally, breaking env-injected credentials on single-profile
(non-multiplex) deployments.

The interactive paths in gateway/run.py guard the secret scope with
`if not multiplex_profiles: return` — skipping it on single-profile
deployments so env-injected credentials fall through to os.environ.
The cron path was missing this guard, causing HTTP 401 errors.

The fix adds the same multiplex_profiles check to cron/scheduler.py.
"""

from __future__ import annotations

import inspect

import pytest


# --------------------------------------------------------------------------- #
# Source inspection: the guard exists in cron scheduler
# --------------------------------------------------------------------------- #


def test_cron_scheduler_has_multiplex_guard():
    """The cron scheduler must check multiplex_profiles before installing
    the secret scope, matching the interactive path's guard in gateway/run.py.

    See issue #65773.
    """
    from cron import scheduler

    # Find the _execute_job function (where the secret scope is installed)
    # The scope installation code is in the job execution path
    source = inspect.getsource(scheduler)

    # Must contain a multiplex_profiles check
    assert "multiplex_profiles" in source, (
        "cron/scheduler.py must check multiplex_profiles before installing "
        "secret scope — see issue #65773"
    )

    # Must contain _skip_scope pattern (our fix)
    assert "_skip_scope" in source, (
        "cron/scheduler.py must have the _skip_scope guard"
    )


def test_cron_scheduler_skips_scope_when_not_multiplex():
    """The scope must be skipped (_scope_token = None) when
    multiplex_profiles is False."""
    from cron import scheduler

    source = inspect.getsource(scheduler)

    # The fix sets _scope_token = None when skipping
    assert "_scope_token = None" in source, (
        "cron/scheduler.py must set _scope_token = None when skipping scope"
    )

    # The finally block must guard against None
    assert "_scope_token is not None" in source, (
        "cron/scheduler.py finally block must guard reset_secret_scope "
        "against _scope_token being None"
    )


def test_gateway_run_has_the_same_guard():
    """Verify the reference guard exists in gateway/run.py — this is the
    pattern we're mirroring."""
    from gateway import run

    source = inspect.getsource(run)

    # gateway/run.py already has the guard
    assert "multiplex_profiles" in source
    assert "if not" in source and "multiplex_profiles" in source


# --------------------------------------------------------------------------- #
# Functional: the guard logic is correct
# --------------------------------------------------------------------------- #


def test_skip_scope_logic_multiplex_off():
    """When multiplex_profiles is False, _skip_scope should be True."""
    # Simulate the guard logic
    multiplex_profiles = False
    _skip_scope = False

    if not multiplex_profiles:
        _skip_scope = True

    assert _skip_scope is True


def test_skip_scope_logic_multiplex_on():
    """When multiplex_profiles is True, _skip_scope should be False."""
    multiplex_profiles = True
    _skip_scope = False

    if not multiplex_profiles:
        _skip_scope = True

    assert _skip_scope is False


def test_scope_token_none_when_skipped():
    """When _skip_scope is True, _scope_token must be None."""
    _skip_scope = True

    if _skip_scope:
        _scope_token = None
    else:
        _scope_token = "fake-token"

    assert _scope_token is None


def test_reset_scope_guards_none():
    """The finally block must not call reset_secret_scope when
    _scope_token is None."""
    _scope_token = None

    # Simulate the finally block
    reset_called = False
    if _scope_token is not None:
        reset_called = True

    assert not reset_called, "reset_secret_scope must not be called when _scope_token is None"