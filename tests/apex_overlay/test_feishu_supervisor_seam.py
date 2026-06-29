"""Seam-test + behavior test for the apex_overlay hc-384 Feishu supervisor.

This pins the upstream symbols that ``apex_overlay.feishu_supervisor``
monkey-patches/depends on, so an upstream rename/move turns a *silently
reverted-to-SDK-reconnect* Feishu adapter into a *loud CI failure* — the
prerequisite for trusting the monkey-patch (see ``apex_overlay/README.md``).

What the seam guards
====================
``gateway/platforms/feishu.py`` keeps two lifecycle entry points as
upstream-faithful no-op stubs — ``FeishuAdapter._start_ws_supervisor`` (called
at the end of ``connect()``) and ``FeishuAdapter._cancel_ws_supervisor``
(awaited in ``disconnect()``). ``feishu_supervisor.apply()`` swaps them for the
real supervisor (a watcher task + backoff ladder + ``/bot/v3/info`` liveness
probe — hc-384) and binds four ladder helpers. With the stubs alone (overlay
absent) the adapter behaves like stock Hermes: no supervisor, and the lark SDK
keeps its own (broken-but-present) auto-reconnect.

If upstream removes the extraction points (``connect``/``disconnect`` no longer
call the two methods), renames any method the ladder depends on, or drops the
``CONNECT_IN_BACKGROUND`` convention, the patch silently degrades to the SDK's
single-shot retry that left bots dead for hours in prod — these tests fail
loudly instead.

This file ALSO proves the seam wiring: the in-tree stubs are no-ops, ``apply()``
installs the real methods + marker, the inline websocket-override hook disables
the SDK retry only when the overlay is active, and the apex-overlay plugin's
``register()`` applies the seam.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter), not a
single in-process pytest — a process-wide monkey-patch behaves differently
under single-process isolation.
"""

from __future__ import annotations

import asyncio
import inspect
import types

from apex_overlay import feishu_supervisor


# ---------------------------------------------------------------------------
# Seam assertions — the two lifecycle extraction points feishu.py keeps
# ---------------------------------------------------------------------------

def test_seam_lifecycle_stub_methods_exist_with_compatible_shape():
    """feishu.py must keep the two no-op lifecycle stubs apply() swaps.

    connect() calls self._start_ws_supervisor() (sync) and disconnect() awaits
    self._cancel_ws_supervisor() (async). If upstream/a refactor drops or
    renames either, the supervisor seam has nothing to patch and Feishu
    silently reverts to the lark SDK's single-shot reconnect. Fail here instead.
    """
    from gateway.platforms.feishu import FeishuAdapter

    start = FeishuAdapter.__dict__.get("_start_ws_supervisor")
    assert start is not None, (
        "FeishuAdapter._start_ws_supervisor is gone — apex_overlay "
        "feishu_supervisor can no longer install the self-reconnect seam. "
        "Re-add the no-op stub in feishu.py (or update _LIFECYCLE_METHODS)."
    )
    assert not inspect.iscoroutinefunction(start), (
        "_start_ws_supervisor must be sync — connect() calls it without await."
    )
    start_params = [p for p in inspect.signature(start).parameters if p != "self"]
    assert start_params == [], (
        f"_start_ws_supervisor grew parameters {start_params!r}; connect() "
        f"calls it as self._start_ws_supervisor() with none."
    )

    cancel = FeishuAdapter.__dict__.get("_cancel_ws_supervisor")
    assert cancel is not None, (
        "FeishuAdapter._cancel_ws_supervisor is gone — disconnect()'s "
        "supervisor teardown hook was removed. Re-add the no-op stub."
    )
    assert inspect.iscoroutinefunction(cancel), (
        "_cancel_ws_supervisor must be async — disconnect() awaits it."
    )
    cancel_params = [p for p in inspect.signature(cancel).parameters if p != "self"]
    assert cancel_params == [], (
        f"_cancel_ws_supervisor grew parameters {cancel_params!r}; disconnect() "
        f"awaits self._cancel_ws_supervisor() with none."
    )


def test_lifecycle_methods_are_noop_when_not_connectable():
    """Calling the lifecycle hooks on a non-connectable adapter is harmless.

    connect()/disconnect() rely on _start_ws_supervisor()/_cancel_ws_supervisor()
    being crash-free even when there's no live socket. This invariant holds for
    BOTH the in-tree no-op stub (overlay absent → SDK owns reconnect) AND the
    overlay version (its early-return branches: webhook mode / no loop / no
    task). Order-independent: works whether or not apply() ran earlier in this
    process (CI runs each test file in a fresh interpreter; a combined pytest
    run may have patched the class via the cross-file import).

    For the pristine in-tree stub, even a bare object is a no-op; for the
    overlay version, a webhook-mode adapter with no loop/task hits its
    early-returns and likewise starts nothing.
    """
    from gateway.platforms.feishu import FeishuAdapter

    start = FeishuAdapter.__dict__.get("_start_ws_supervisor")
    cancel = FeishuAdapter.__dict__.get("_cancel_ws_supervisor")
    assert start is not None and cancel is not None, (
        "the two lifecycle stub methods must exist on FeishuAdapter."
    )

    # A non-connectable adapter: webhook mode + no loop + no in-flight task.
    # Both the stub (ignores everything) and the overlay (early-returns on
    # _connection_mode != 'websocket' / loop is None) must do nothing.
    stub_self = types.SimpleNamespace(
        _connection_mode="webhook",
        _ws_self_reconnect=True,
        _loop=None,
        _ws_supervisor_task=None,
    )
    assert start(stub_self) is None
    assert stub_self._ws_supervisor_task is None, (
        "no supervisor task may be started for a non-connectable adapter."
    )
    assert asyncio.run(cancel(stub_self)) is None


def test_seam_connect_and_disconnect_call_the_lifecycle_hooks():
    """feishu.py's connect()/disconnect() must call the swapped methods.

    The seam only works if connect() starts the supervisor and disconnect()
    cancels it. If a refactor drops these call sites, apply() still swaps the
    methods but they're never invoked — a silent no-supervisor revert.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    src = (repo / "gateway" / "platforms" / "feishu.py").read_text(encoding="utf-8")

    assert "self._start_ws_supervisor()" in src, (
        "feishu.py connect() no longer calls self._start_ws_supervisor() — the "
        "hc-384 supervisor would never launch even with the overlay applied."
    )
    assert "self._cancel_ws_supervisor()" in src, (
        "feishu.py disconnect() no longer calls self._cancel_ws_supervisor() — "
        "the supervisor task would leak on shutdown."
    )


# ---------------------------------------------------------------------------
# Seam assertions — every upstream method/attr the ladder depends on
# ---------------------------------------------------------------------------

def test_seam_ladder_dependencies_exist_with_compatible_signatures():
    """Pin the FeishuAdapter methods the overlay ladder/helpers call by name.

    The reconnect ladder rebuilds the socket and probes liveness through these.
    An upstream rename would blow up mid-reconnect (exactly when reliability
    matters most); pin them so it's a CI failure instead.
    """
    from gateway.platforms.feishu import FeishuAdapter

    expected = {
        "_teardown_ws_thread": ["self"],
        "_connect_websocket": ["self"],
        "_hydrate_bot_identity": ["self"],
    }
    for name, params in expected.items():
        fn = getattr(FeishuAdapter, name, None)
        assert fn is not None, (
            f"FeishuAdapter.{name} is gone — apex_overlay feishu_supervisor's "
            f"reconnect ladder depends on it. Update the overlay and seam-test "
            f"together."
        )
        got = list(inspect.signature(fn).parameters)
        assert got == params, (
            f"FeishuAdapter.{name} signature changed to {got!r}; the ladder "
            f"calls it as {name}({', '.join(params[1:])})."
        )
        assert inspect.iscoroutinefunction(fn), (
            f"FeishuAdapter.{name} must stay async — the ladder awaits it."
        )

    # _hydrate_bot_identity is reused as the liveness probe; the ladder relies
    # on its truthy return meaning "endpoint reachable" (hc-384 added the bool).
    hydrate_ret = inspect.signature(FeishuAdapter._hydrate_bot_identity).return_annotation
    assert hydrate_ret in (bool, "bool"), (
        "FeishuAdapter._hydrate_bot_identity must return bool — _verify_ws_alive "
        f"uses it as the /bot/v3/info liveness probe; got return {hydrate_ret!r}."
    )


def test_seam_fatal_error_api_is_compatible():
    """Pin the base-adapter fatal-error API the ladder uses on exhaustion.

    On ladder exhaustion the overlay calls _set_fatal_error(code, msg,
    retryable=True) then awaits _notify_fatal_error() so the gateway's reconnect
    watcher recreates the adapter. Pin both.
    """
    from gateway.platforms.base import BasePlatformAdapter

    set_fatal = getattr(BasePlatformAdapter, "_set_fatal_error", None)
    assert set_fatal is not None, "BasePlatformAdapter._set_fatal_error is gone."
    params = inspect.signature(set_fatal).parameters
    for needed in ("code", "message", "retryable"):
        assert needed in params, (
            f"_set_fatal_error lost the {needed!r} parameter the overlay passes."
        )

    notify = getattr(BasePlatformAdapter, "_notify_fatal_error", None)
    assert notify is not None and inspect.iscoroutinefunction(notify), (
        "BasePlatformAdapter._notify_fatal_error must exist and be async — the "
        "ladder awaits it after escalating to a fatal error."
    )


def test_seam_connect_in_background_marker_present():
    """Pin CONNECT_IN_BACKGROUND on FeishuAdapter (the gateway_bootstrap seam
    keys off it; supervisor + background startup are the two halves of the same
    Feishu reliability fix). Defined on the adapter class, falsy on the base."""
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.platforms.base import BasePlatformAdapter

    assert getattr(FeishuAdapter, "CONNECT_IN_BACKGROUND", False), (
        "FeishuAdapter.CONNECT_IN_BACKGROUND is no longer truthy — the gateway "
        "background-startup seam would stop deferring Feishu's slow attach."
    )
    assert not getattr(BasePlatformAdapter, "CONNECT_IN_BACKGROUND", False), (
        "BasePlatformAdapter now defaults CONNECT_IN_BACKGROUND truthy."
    )


def test_seam_ws_client_conn_convention_for_death_detection():
    """_websocket_appears_dead reads adapter._ws_client / ._ws_future and the
    lark client's ``_conn``. Pin that the adapter still tracks these so death
    detection isn't silently disarmed (it would think the socket is alive)."""
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.platforms.base import PlatformConfig

    adapter = FeishuAdapter(PlatformConfig(extra={}))
    for attr in ("_ws_client", "_ws_future", "_ws_supervisor_task",
                 "_ws_reconnecting", "_intentional_disconnect", "_ws_self_reconnect"):
        assert hasattr(adapter, attr), (
            f"FeishuAdapter no longer initializes {attr!r} in __init__ — the "
            f"hc-384 supervisor seam reads/writes it. Re-add the state line."
        )


# ---------------------------------------------------------------------------
# Seam assertions — the inline SDK-reconnect-disable hook integration
# ---------------------------------------------------------------------------

def test_seam_runtime_override_hook_present_and_gated_on_overlay():
    """feishu.py keeps _apply_feishu_ws_runtime_overrides + _feishu_supervisor_active.

    The override hook disables the lark SDK's broken auto-reconnect ONLY when
    this overlay is active (so an overlay-absent box keeps the SDK's own
    reconnect). Pin both the hook and the active-check bridge.
    """
    import gateway.platforms.feishu as feishu

    assert hasattr(feishu, "_apply_feishu_ws_runtime_overrides"), (
        "feishu.py lost _apply_feishu_ws_runtime_overrides — the WS thread "
        "launcher relies on it to push ws tuning + the SDK-reconnect disable."
    )
    assert hasattr(feishu, "_feishu_supervisor_active"), (
        "feishu.py lost _feishu_supervisor_active — the override hook needs it "
        "to gate the SDK-reconnect-disable on the overlay being installed."
    )
    # The overlay exposes the bridge the inline hook imports.
    assert hasattr(feishu_supervisor, "apex_overlay_active"), (
        "apex_overlay.feishu_supervisor.apex_overlay_active is gone — feishu.py "
        "imports it to decide whether to disable the SDK's auto-reconnect."
    )


def test_override_disables_sdk_reconnect_only_when_overlay_active():
    """Behavior of the inline hook + overlay marker, end to end.

    Without the overlay marker: SDK auto-reconnect is left ON (upstream).
    After apply() sets the marker: the hook disables it (hc-384 takes over).
    The ws-tuning overrides apply in both cases (they're upstream behavior).
    """
    import gateway.platforms.feishu as feishu
    from gateway.platforms.base import PlatformConfig

    # Fresh class state: ensure the marker is not set yet.
    if hasattr(feishu.FeishuAdapter, feishu_supervisor._ACTIVE_FLAG):
        delattr(feishu.FeishuAdapter, feishu_supervisor._ACTIVE_FLAG)
    feishu_supervisor._APPLIED = False

    adapter = feishu.FeishuAdapter(PlatformConfig(extra={}))  # self_reconnect defaults on

    ws_off = types.SimpleNamespace()
    feishu._apply_feishu_ws_runtime_overrides(ws_off, adapter)
    assert ws_off._reconnect_nonce == adapter._ws_reconnect_nonce
    assert not hasattr(ws_off, "_auto_reconnect"), (
        "with the overlay inactive the lark SDK's auto-reconnect must stay ON "
        "(upstream behavior) — otherwise an overlay-less box has NO reconnect."
    )

    # Now activate the overlay and re-run the hook.
    assert feishu_supervisor.apply() is True
    ws_on = types.SimpleNamespace()
    feishu._apply_feishu_ws_runtime_overrides(ws_on, adapter)
    assert ws_on._auto_reconnect is False, (
        "with the overlay active the hook must disable the SDK's broken "
        "single-shot reconnect so the supervisor owns reconnection (hc-384)."
    )


def test_override_leaves_sdk_reconnect_when_reverted_even_with_overlay():
    """The ws_self_reconnect=False revert flag wins even with the overlay on."""
    import gateway.platforms.feishu as feishu
    from gateway.platforms.base import PlatformConfig

    feishu_supervisor._APPLIED = False
    assert feishu_supervisor.apply() is True

    adapter = feishu.FeishuAdapter(PlatformConfig(extra={"ws_self_reconnect": False}))
    ws = types.SimpleNamespace()
    feishu._apply_feishu_ws_runtime_overrides(ws, adapter)
    assert not hasattr(ws, "_auto_reconnect"), (
        "ws_self_reconnect=False (revert) must leave the SDK's auto-reconnect "
        "ON even when the overlay is loaded — a clean fall-back to upstream."
    )


# ---------------------------------------------------------------------------
# apply() — swaps the stubs, binds the ladder helpers, idempotent
# ---------------------------------------------------------------------------

def test_apply_swaps_stubs_binds_helpers_and_is_idempotent():
    """apply() must swap the 2 stubs, bind the 4 ladder helpers, set the marker,
    and no-op on repeat."""
    from gateway.platforms.feishu import FeishuAdapter

    feishu_supervisor._APPLIED = False
    if hasattr(FeishuAdapter, feishu_supervisor._ACTIVE_FLAG):
        delattr(FeishuAdapter, feishu_supervisor._ACTIVE_FLAG)
    assert feishu_supervisor.apply() is True

    # The two lifecycle methods are now the overlay versions (overlay module).
    for name in feishu_supervisor._LIFECYCLE_METHODS:
        fn = getattr(FeishuAdapter, name)
        assert getattr(fn, "__module__", "").endswith("feishu_supervisor"), (
            f"after apply() FeishuAdapter.{name} must be the overlay version "
            f"(module apex_overlay.feishu_supervisor); got {fn.__module__!r}."
        )

    # Every ladder helper the supervisor calls is bound onto the class.
    for name in feishu_supervisor._LADDER_METHODS:
        assert hasattr(FeishuAdapter, name), (
            f"apply() did not bind ladder helper {name!r} onto FeishuAdapter."
        )

    # The active marker is set so the inline override hook engages.
    assert getattr(FeishuAdapter, feishu_supervisor._ACTIVE_FLAG, False) is True

    # Idempotent: second apply is a no-op and must not error or rebind.
    start_before = getattr(FeishuAdapter, "_start_ws_supervisor")
    assert feishu_supervisor.apply() is True
    assert getattr(FeishuAdapter, "_start_ws_supervisor") is start_before


def test_apply_real_start_supervisor_creates_task_when_enabled():
    """After apply(), the real _start_ws_supervisor launches a watcher task.

    Proves apply() installed the *behaving* method, not just any callable: in
    websocket mode with self-reconnect on, it creates the supervisor task.
    """
    from gateway.platforms.feishu import FeishuAdapter
    from gateway.platforms.base import PlatformConfig

    feishu_supervisor._APPLIED = False
    if hasattr(FeishuAdapter, feishu_supervisor._ACTIVE_FLAG):
        delattr(FeishuAdapter, feishu_supervisor._ACTIVE_FLAG)
    assert feishu_supervisor.apply() is True

    adapter = FeishuAdapter(PlatformConfig(extra={}))
    adapter._connection_mode = "websocket"
    adapter._running = True
    loop = asyncio.new_event_loop()
    adapter._loop = loop
    try:
        adapter._start_ws_supervisor()
        task = adapter._ws_supervisor_task
        assert task is not None, (
            "the overlay _start_ws_supervisor must create a watcher task in "
            "websocket mode with self-reconnect enabled."
        )
        task.cancel()
        loop.run_until_complete(asyncio.gather(task, return_exceptions=True))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Wiring — the seam loads via the apex-overlay plugin
# ---------------------------------------------------------------------------

def test_plugin_register_applies_feishu_supervisor_seam():
    """The bundled apex-overlay plugin's register() applies this seam too."""
    import importlib.util
    from pathlib import Path

    plugin_init = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "apex-overlay" / "__init__.py"
    )
    assert plugin_init.exists(), "apex-overlay plugin __init__.py missing"

    spec = importlib.util.spec_from_file_location(
        "_apex_overlay_plugin_under_test_feishu", plugin_init
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "register"), "plugin must expose register(ctx)"

    from unittest.mock import patch

    called = {}
    with patch.object(
        feishu_supervisor, "apply",
        lambda: called.setdefault("applied", True) or True,
    ):
        # The other seams' apply() also run inside register(); let them run for
        # real (they're idempotent) — we only assert our seam was invoked.
        mod.register(ctx=None)
    assert called.get("applied") is True, (
        "plugin.register() must call feishu_supervisor.apply()"
    )


def test_apex_overlay_enabled_in_config():
    """cli-config.yaml.example enables the apex-overlay plugin (config tier)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = (repo / "cli-config.yaml.example").read_text(encoding="utf-8")
    assert "apex-overlay" in cfg, (
        "cli-config.yaml.example must list apex-overlay under plugins.enabled "
        "or the Feishu supervisor seam never loads in production."
    )
