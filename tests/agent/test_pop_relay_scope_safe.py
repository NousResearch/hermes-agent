"""Regression: agent.relay_runtime.safe_pop_relay_scope tolerates the vendor
RuntimeError 'scope handle is not at the top of the stack', while the
strict pop_relay_scope preserves it so the drain path can fire.

Background
----------

`agent/relay_runtime.py::pop_relay_scope` calls `relay.scope.pop(handle)`
which delegates to `_native_pop_scope` (a compiled C extension from
`nemo-relay` 0.8.3). When the caller passes a stale handle - one that was
already popped by an earlier interrupt or drain path - the native call
raises `RuntimeError("invalid argument: scope handle is not at the top
of the stack")`.

The right behaviour depends on the call site:

* `_pop_with_drain` (drain-aware): it tries the pop, expects the
  RuntimeError as the signal to drain orphan scopes above the target,
  then retries. It must keep using `pop_relay_scope` so the RuntimeError
  still propagates.
* `_finish_task` in `hermes_cli/observability/relay_shared_metrics.py`
  (finalization/cleanup): the scope is already gone, raising only costs
  one observability-loss log line and skips metrics export. It must use
  `safe_pop_relay_scope` so the stale-handle pop is treated as a no-op
  success.

A previous attempt (PR #99302, now closed) tried to add this tolerance
inside `hermes_cli/observability/relay_shared_metrics.py` itself; that
was judged moot because that module already routes the call through a
log-and-swallow `_guarded()`. THIS commit hardens the runtime helper
itself (as an opt-in sibling) and wires the finish-task site to use it,
keeping the strict helper unchanged so `_pop_with_drain` still works.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Resolve the repo root from this test file's location (tests/agent/ -> ../../)
# so the test runs wherever the repo is checked out.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import relay_runtime


def _make_relay(raises: Exception | None):
    """Build a mock relay whose scope.pop raises the given exception (or not)."""
    relay = MagicMock()
    if raises is None:
        relay.scope.pop.return_value = None
    else:
        relay.scope.pop.side_effect = raises
    return relay


# ---- strict pop_relay_scope: drain contract preserved -----------------------


def test_pop_relay_scope_returns_none_on_clean_pop():
    relay = _make_relay(None)
    result = relay_runtime.pop_relay_scope(relay, handle="h-1")
    assert result is None


def test_pop_relay_scope_propagates_vendor_runtime_error():
    """The strict helper MUST still raise the vendor RuntimeError so that
    _pop_with_drain can detect it as the drain signal."""
    relay = _make_relay(
        RuntimeError("invalid argument: scope handle is not at the top of the stack")
    )
    with pytest.raises(RuntimeError, match="scope handle is not at the top of the stack"):
        relay_runtime.pop_relay_scope(relay, handle="h-stale")


def test_pop_relay_scope_propagates_unrelated_errors():
    """Bugs that are NOT the known vendor scope-stack symptom must still surface."""
    relay = _make_relay(ValueError("something completely different"))
    with pytest.raises(ValueError, match="something completely different"):
        relay_runtime.pop_relay_scope(relay, handle="h-x")


# ---- safe_pop_relay_scope: finalization/cleanup path ----------------------


def test_safe_pop_relay_scope_returns_none_on_clean_pop():
    relay = _make_relay(None)
    result = relay_runtime.safe_pop_relay_scope(relay, handle="h-1")
    assert result is None


def test_safe_pop_relay_scope_swallows_vendor_runtime_error():
    """Finalization/cleanup callers use the safe helper so a stale handle
    is treated as a no-op success."""
    relay = _make_relay(
        RuntimeError("invalid argument: scope handle is not at the top of the stack")
    )
    result = relay_runtime.safe_pop_relay_scope(relay, handle="h-stale")
    assert result is None


def test_safe_pop_relay_scope_propagates_unrelated_errors():
    relay = _make_relay(ValueError("something completely different"))
    with pytest.raises(ValueError, match="something completely different"):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-x")


def test_safe_pop_relay_scope_propagates_keyerror():
    relay = _make_relay(KeyError("nope"))
    with pytest.raises(KeyError):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-x")
