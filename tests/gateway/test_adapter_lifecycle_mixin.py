"""Mixin contract tests: GatewayRunner inherits adapter-lifecycle helpers via MRO.

After the adapter-lifecycle mixin extraction (god-file decomposition Phase 4),
the five adapter lifecycle helpers live on ``GatewayAdapterLifecycleMixin``:

  - ``_safe_adapter_disconnect``
  - ``_bounded_adapter_teardown``
  - ``_adapter_disconnect_timeout_secs``
  - ``_platform_connect_timeout_secs``
  - ``_connect_adapter_with_timeout``

The existing ``GatewayRunner.__init__`` is unchanged at the call sites, so
runtime behavior is preserved. These tests assert behavior contracts
(invariants), not snapshots:

  - The methods must resolve on a ``GatewayRunner`` instance via the MRO
    (mixin wired into the class bases), even without a real ``__init__``.
  - The mixin itself must expose the same five methods (so direct subclassing
    or ``object.__new__(GatewayAdapterLifecycleMixin)`` shells — like the
    existing regression tests do with ``GatewayRunner`` — keep working).
  - The mixin must NOT depend on any ``self.*`` state for these five methods
    (they read ``os.getenv`` + the module-level timeout constants only). That
    is the invariant that makes this a safe pure-move: there is nothing in
    ``__init__`` the methods need that a bare ``object.__new__`` shell lacks.

Per AGENTS.md, this is a behavior-contract test (invariant), not a
change-detector test (snapshot of a current value).
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.adapter_lifecycle_mixin import GatewayAdapterLifecycleMixin
from gateway.config import Platform
from gateway.run import GatewayRunner


_METHOD_NAMES = (
    "_safe_adapter_disconnect",
    "_bounded_adapter_teardown",
    "_adapter_disconnect_timeout_secs",
    "_platform_connect_timeout_secs",
    "_connect_adapter_with_timeout",
)


def test_mixin_exposes_all_five_methods():
    """Each named method must exist as an attribute on the mixin."""
    missing = [n for n in _METHOD_NAMES if not hasattr(GatewayAdapterLifecycleMixin, n)]
    assert not missing, f"Mixin missing methods: {missing}"


def test_runner_resolves_methods_via_mro():
    """A bare ``object.__new__(GatewayRunner)`` shell (no ``__init__``) must
    still resolve each method through the MRO — proves the mixin is wired into
    ``GatewayRunner``'s bases and the helpers do not depend on ``__init__``."""
    shell = object.__new__(GatewayRunner)
    missing = [n for n in _METHOD_NAMES if not hasattr(shell, n)]
    assert not missing, f"GatewayRunner shell missing methods via MRO: {missing}"


def test_methods_resolve_to_mixin_not_runner():
    """The methods must resolve to the *mixin*'s function objects, not stale
    copies left on ``GatewayRunner`` itself. After extraction, run.py must
    not re-declare them."""
    for name in _METHOD_NAMES:
        runner_attr = getattr(GatewayRunner, name, None)
        mixin_attr = getattr(GatewayAdapterLifecycleMixin, name, None)
        assert runner_attr is not None, f"{name} not reachable on GatewayRunner"
        assert mixin_attr is not None, f"{name} not defined on the mixin"
        # The function object on GatewayRunner must BE the mixin's function
        # (MRO resolution), not a separate def left in run.py.
        assert runner_attr is mixin_attr, (
            f"{name} on GatewayRunner is not the mixin's implementation — "
            f"either a stale duplicate was left in run.py or the mixin is "
            f"not in the MRO. Expected identical function objects."
        )


def test_timeout_secs_methods_dont_require_init_state():
    """The two ``_..._timeout_secs`` methods read only ``os.getenv`` and the
    module-level constants, so a bare shell (no config, no adapters dict)
    must return the default values with no AttributeError."""
    shell = object.__new__(GatewayRunner)
    # Defaults match the module constants in run.py:
    #   _ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT = 5.0
    #   _PLATFORM_CONNECT_TIMEOUT_SECS_DEFAULT   = 30.0
    # We assert the contract (env-var override wins; absent env-var returns
    # a positive float), not the exact default literal — that would freeze a
    # current value and become a change-detector test.
    assert isinstance(shell._adapter_disconnect_timeout_secs(), float)
    assert shell._adapter_disconnect_timeout_secs() > 0
    assert isinstance(shell._platform_connect_timeout_secs(), float)
    assert shell._platform_connect_timeout_secs() > 0


@pytest.mark.asyncio
async def test_connect_helper_forwards_is_reconnect_kwarg():
    """``_connect_adapter_with_timeout`` must forward ``is_reconnect`` through
    to ``adapter.connect`` unchanged. This is the behavior that preserves
    messages sent during an outage (#46621) — a regression here silently
    drops messages on reconnect."""
    shell = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.connect = AsyncMock(return_value=True)

    # Use a generous timeout so the test itself is not racy on slow CI.
    os.environ.pop("HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT", None)
    result = await shell._connect_adapter_with_timeout(
        adapter, Platform.TELEGRAM, is_reconnect=True
    )
    assert result is True
    adapter.connect.assert_awaited_once_with(is_reconnect=True)


@pytest.mark.asyncio
async def test_safe_disconnect_forward_progress_on_partial_init():
    """The defensive disconnect must never raise, even when adapter.disconnect
    raises — the caller is on a failure path and cannot propagate."""
    shell = object.__new__(GatewayRunner)
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("partial init"))

    # Must NOT raise.
    await shell._safe_adapter_disconnect(adapter, Platform.TELEGRAM)
    adapter.disconnect.assert_awaited_once()
