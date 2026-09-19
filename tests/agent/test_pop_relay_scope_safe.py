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


def _make_relay(raises: Exception | None, *, stack=None, stack_raises: Exception | None = None):
    """Build a mock relay whose scope.pop raises the given exception (or not).

    ``stack`` is what ``get_scope_stack()`` reports, so tests can express the two situations the
    vendor message conflates: our handle is genuinely gone, or it is still live but not on top.
    """
    relay = MagicMock()
    if raises is None:
        relay.scope.pop.return_value = None
    else:
        relay.scope.pop.side_effect = raises
    if stack_raises is not None:
        relay.get_scope_stack.side_effect = stack_raises
    else:
        relay.get_scope_stack.return_value = stack
    return relay


_VENDOR_ERROR = RuntimeError("invalid argument: scope handle is not at the top of the stack")


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


def test_safe_pop_relay_scope_swallows_when_handle_is_provably_gone():
    """Finalization/cleanup callers use the safe helper, and the handle is absent from the stack.

    Absence is what makes this a stale-handle case. The vendor message alone does not establish it —
    see the live-but-buried case below.
    """
    relay = _make_relay(_VENDOR_ERROR, stack=["h-other"])
    result = relay_runtime.safe_pop_relay_scope(relay, handle="h-stale")
    assert result is None


def test_safe_pop_relay_scope_swallows_when_the_stack_is_empty():
    relay = _make_relay(_VENDOR_ERROR, stack=[])
    assert relay_runtime.safe_pop_relay_scope(relay, handle="h-stale") is None


def test_safe_pop_relay_scope_reraises_when_handle_is_live_but_not_top():
    """The reviewers' case, and the one that used to be silently mishandled.

    'not at the top of the stack' also fires when our handle is still live beneath a nested
    model/tool scope. Swallowing that makes the caller forget a task whose scope is still on the
    stack, so it must propagate instead.
    """
    relay = _make_relay(_VENDOR_ERROR, stack=["h-child", "h-ours"])
    with pytest.raises(RuntimeError, match="scope handle is not at the top of the stack"):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-ours")


def test_safe_pop_relay_scope_reraises_when_absence_cannot_be_proven():
    """An uninspectable stack must not read as 'already popped'."""
    relay = _make_relay(_VENDOR_ERROR, stack_raises=RuntimeError("stack unavailable"))
    with pytest.raises(RuntimeError, match="scope handle is not at the top of the stack"):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-ours")


def test_safe_pop_relay_scope_reraises_when_only_the_top_handle_is_exposed():
    """Single-handle builds cannot distinguish 'gone' from 'buried', so they must fail closed."""
    relay = _make_relay(_VENDOR_ERROR, stack="h-someone-else")
    with pytest.raises(RuntimeError, match="scope handle is not at the top of the stack"):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-ours")


def test_safe_pop_relay_scope_tolerates_same_handle_matched_by_uuid():
    """Stack entries may be distinct Python objects wrapping the same native scope."""

    class _Handle:
        def __init__(self, uuid):
            self.uuid = uuid

    ours = _Handle("u-1")
    relay = _make_relay(_VENDOR_ERROR, stack=[_Handle("u-other"), _Handle("u-1")])
    with pytest.raises(RuntimeError, match="scope handle is not at the top of the stack"):
        relay_runtime.safe_pop_relay_scope(relay, handle=ours)


def test_safe_pop_relay_scope_propagates_unrelated_errors():
    relay = _make_relay(ValueError("something completely different"), stack=[])
    with pytest.raises(ValueError, match="something completely different"):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-x")


def test_safe_pop_relay_scope_propagates_keyerror():
    relay = _make_relay(KeyError("nope"), stack=[])
    with pytest.raises(KeyError):
        relay_runtime.safe_pop_relay_scope(relay, handle="h-x")


# ---- the finish-task wiring itself ---------------------------------------


def test_finish_task_call_site_uses_the_safe_helper():
    """The wiring is part of the contract, so a silent swap must fail a test.

    `_finish_task` finalizes a task whose scope was created elsewhere; routing it back through the
    strict helper would reintroduce the observability-loss it was changed to avoid. A fixture-driven
    integration test through `finish_task_run` needs the direct-runtime harness, so this pins the
    call site directly in the meantime — it is the cheap half of the same guarantee.
    """
    import inspect

    from hermes_cli.observability import relay_shared_metrics

    src = inspect.getsource(relay_shared_metrics)
    assert "relay_runtime.safe_pop_relay_scope" in src
    # A bare strict call here would be the regression.
    assert "relay_runtime.pop_relay_scope(" not in src


# ---- the native binding's SECOND message (found in review) ----------------
#
# The pinned binding (nemo-relay 0.8.3) reports the two LIFO states with two
# DIFFERENT messages:
#
#   already popped  -> "not found: scope handle not found"
#   live but buried -> "invalid argument: scope handle is not at the top of the stack"
#
# Modelling only the buried message is what let the original helper re-raise for
# the exact case it exists to tolerate: the already-popped path raised a string it
# did not recognise. These cases model the real one.

_NATIVE_NOT_FOUND = RuntimeError("not found: scope handle not found")


def test_pop_relay_scope_propagates_the_native_not_found_error():
    """The strict helper must still surface it, so `_pop_with_drain` keeps its signal."""
    relay = _make_relay(_NATIVE_NOT_FOUND)
    with pytest.raises(RuntimeError, match="scope handle not found"):
        relay_runtime.pop_relay_scope(relay, handle="h-x")


def test_safe_pop_relay_scope_tolerates_the_native_not_found_error():
    """Absence is asserted by the native layer, so no stack probe is required.

    On the real binding ``get_scope_stack()`` returns an opaque ``ScopeStack``, which makes
    :func:`_handle_still_on_stack` return ``None``. If tolerance demanded ``False`` for this
    message too, the already-popped case would re-raise on the very binding this PR targets.
    """
    relay = _make_relay(_NATIVE_NOT_FOUND, stack=object())
    assert relay_runtime.safe_pop_relay_scope(relay, handle="h-x") is None


def test_safe_pop_relay_scope_tolerates_the_native_not_found_error_when_the_probe_raises():
    """An uninspectable stack must not turn a native "not found" back into a raise."""
    relay = _make_relay(_NATIVE_NOT_FOUND, stack_raises=AttributeError("opaque stack"))
    assert relay_runtime.safe_pop_relay_scope(relay, handle="h-x") is None


# ---- end-to-end against the pinned native binding -------------------------
#
# The message-level cases above model what the binding says; this one drives the binding itself,
# because modelling it wrongly is exactly how the original helper shipped broken. The fixture
# mirrors the one in `tests/hermes_cli/test_relay_shared_metrics_runtime.py`.


@pytest.fixture
def real_binding(tmp_path, monkeypatch):
    relay = pytest.importorskip("nemo_relay")
    if getattr(relay, "_native", None) is None:
        pytest.skip("NeMo Relay native binding is unavailable on this platform")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    relay_runtime._reset_for_tests()
    yield relay
    relay_runtime._reset_for_tests()


def test_safe_pop_relay_scope_tolerates_a_real_already_popped_handle(real_binding):
    """push -> pop -> safe-pop the same handle must not raise.

    The observed native messages, both reproduced against the pinned binding:

        already popped  -> RuntimeError: not found: scope handle not found
        live but buried -> RuntimeError: invalid argument: scope handle is not at the top of the stack

    Before the fix the tolerant helper recognised only the second, so it re-raised the first —
    the very case it exists to tolerate.
    """
    runtime = relay_runtime.get_runtime()
    session = runtime.ensure_session({"session_id": "native-already-popped"})
    handle = runtime.run_in_session(
        session,
        real_binding.scope.push,
        "native-already-popped",
        real_binding.ScopeType.Function,
        handle=session.handle,
    )
    runtime.run_in_session(session, real_binding.scope.pop, handle)

    # The strict helper keeps the native error so `_pop_with_drain` still gets its signal.
    with pytest.raises(RuntimeError, match="scope handle not found"):
        runtime.run_in_session(session, relay_runtime.pop_relay_scope, real_binding, handle)

    # The tolerant helper treats the native "not found" as a completed pop.
    assert runtime.run_in_session(
        session, relay_runtime.safe_pop_relay_scope, real_binding, handle
    ) is None


def test_safe_pop_relay_scope_still_reraises_a_real_buried_handle(real_binding):
    """A live-but-buried handle must keep failing closed, on the real binding too."""
    runtime = relay_runtime.get_runtime()
    session = runtime.ensure_session({"session_id": "native-buried"})
    buried = runtime.run_in_session(
        session, real_binding.scope.push, "native-buried",
        real_binding.ScopeType.Function, handle=session.handle,
    )
    above = runtime.run_in_session(
        session, real_binding.scope.push, "native-nested",
        real_binding.ScopeType.Function, handle=session.handle,
    )
    try:
        # `not at the top` is generic: it also fires when our handle is still live beneath a
        # nested scope, so tolerance must not swallow it here.
        with pytest.raises(RuntimeError, match="not at the top of the stack"):
            runtime.run_in_session(
                session, relay_runtime.safe_pop_relay_scope, real_binding, buried
            )
    finally:
        runtime.run_in_session(session, real_binding.scope.pop, above)
