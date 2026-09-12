"""Regression: agent.relay_runtime.pop_relay_scope tolerates the vendor
RuntimeError 'scope handle is not at the top of the stack'.

Background
----------

`agent/relay_runtime.py::pop_relay_scope` calls `relay.scope.pop(handle)`
which delegates to `_native_pop_scope` (a compiled C extension from
`nemo-relay` 0.8.3). When the caller passes a stale handle - one that was
already popped by an earlier interrupt or drain path - the native call
raises `RuntimeError("invalid argument: scope handle is not at the top
of the stack")`.

Symptom-wise the operation is **already complete**: the handle was
popped, no further action is needed. But the unhandled exception
propagates back through `agent/relay_runtime._close_scope_handle` (where
`_pop_with_drain` swallows it via `contextlib.suppress(Exception)`)
to the `_run_in_session` boundary, returning a `RuntimeError` instance
that the caller then propagates as an observability-loss log line:

    WARNING Hermes shared-metrics task close failed ...
    WARNING Hermes Relay logical LLM finalization failed ...

No data loss, no crash, but every occurrence costs one log line + the
metrics for that task don't get exported.

The right fix: treat the native RuntimeError as a no-op success in
`pop_relay_scope` (it matches our "drained 1 orphan" semantics) so the
return value semantics align with the already-popped state.

See kanban card `t_4d63c02c` for the original diagnostic and stack trace.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(r"C:\Users\bbask\AppData\Local\hermes\hermes-agent")
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


def test_pop_relay_scope_returns_none_on_clean_pop():
    relay = _make_relay(None)
    result = relay_runtime.pop_relay_scope(relay, handle="h-1")
    assert result is None


def test_pop_relay_scope_swallows_vendor_runtime_error():
    relay = _make_relay(
        RuntimeError("invalid argument: scope handle is not at the top of the stack")
    )
    result = relay_runtime.pop_relay_scope(relay, handle="h-stale")
    assert result is None


def test_pop_relay_scope_propagates_unrelated_errors():
    """Bugs that are NOT the known vendor scope-stack symptom must still surface."""
    relay = _make_relay(ValueError("something completely different"))
    with pytest.raises(ValueError, match="something completely different"):
        relay_runtime.pop_relay_scope(relay, handle="h-x")


def test_pop_relay_scope_propagates_keyerror():
    relay = _make_relay(KeyError("nope"))
    with pytest.raises(KeyError):
        relay_runtime.pop_relay_scope(relay, handle="h-x")
