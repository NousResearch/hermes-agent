"""Seam-test + behavior test for the apex_overlay hc-384/385 gateway bootstrap.

This pins the upstream symbols that ``apex_overlay.gateway_bootstrap``
monkey-patches/depends on, so an upstream rename/move turns a *silently
reverted-to-blocking* gateway startup into a *loud CI failure* — the
prerequisite for trusting the monkey-patch (see ``apex_overlay/README.md``).

What the seam guards
====================
``gateway/run.py`` keeps the per-platform connect loop as one extracted method,
``GatewayRunner._connect_configured_platforms`` (in-tree body = upstream's
original sequential, *blocking* loop). ``gateway_bootstrap.apply()`` replaces
that method with the background-connect version (Feishu attaches off the
critical path so the API conversation surface is ready first — hc-384/385) and
binds eight helper methods. If upstream removes the extraction point or renames
any helper dependency, the patch can't bind and we'd silently fall back to the
blocking loop — these tests fail loudly instead.

This file ALSO proves the behavior the original +276 in-place lines provided:
a ``CONNECT_IN_BACKGROUND`` adapter (Feishu) is scheduled in the background and
does NOT block the loop returning, while an ordinary adapter connects inline.

Run via ``scripts/run_tests_parallel.py`` (per-file fresh interpreter), not a
single in-process pytest — a process-wide monkey-patch behaves differently
under single-process isolation.
"""

from __future__ import annotations

import asyncio
import inspect
import types

import pytest

from apex_overlay import gateway_bootstrap


# ---------------------------------------------------------------------------
# Seam assertions — pin the patched method + every upstream dependency
# ---------------------------------------------------------------------------

def test_seam_target_connect_configured_platforms_exists():
    """apply() replaces GatewayRunner._connect_configured_platforms.

    This is the single extraction point in run.py the overlay swaps. If
    upstream (or a careless refactor) drops/renames it, the background-startup
    seam has nothing to patch and the gateway silently reverts to the blocking
    loop. Fail here instead.
    """
    from gateway.run import GatewayRunner

    fn = getattr(GatewayRunner, gateway_bootstrap._TARGET_LOOP_METHOD, None)
    assert fn is not None, (
        "GatewayRunner._connect_configured_platforms is gone — apex_overlay "
        "gateway_bootstrap can no longer install the background-startup seam. "
        "Update _TARGET_LOOP_METHOD and re-extract the loop in run.py."
    )
    # It must be an async method (start() awaits it).
    assert inspect.iscoroutinefunction(fn), (
        "_connect_configured_platforms must be async — start() awaits it and "
        "the overlay replacement is a coroutine."
    )
    # Zero-arg (besides self): start() calls it with no arguments.
    params = [p for p in inspect.signature(fn).parameters if p != "self"]
    assert params == [], (
        f"_connect_configured_platforms grew parameters {params!r}; start() "
        f"calls it as self._connect_configured_platforms() with none."
    )


def test_intree_loop_returns_six_tuple():
    """The in-tree (pre-patch) loop returns the 6-tuple start() unpacks.

    Even if the overlay never applies (plugin disabled / apply() fails), the
    upstream-faithful body in run.py must still return
    (aborted, connected, background, enabled, nonretryable_errors,
    retryable_errors) so ``start()``'s unpacking never breaks. Drive it with an
    empty platform set so no adapter machinery is needed.
    """
    from gateway.run import GatewayRunner

    fn = GatewayRunner.__dict__.get(gateway_bootstrap._TARGET_LOOP_METHOD)
    assert fn is not None and not getattr(fn, gateway_bootstrap._MARK, False), (
        "expected the in-tree run.py method here, not the overlay replacement "
        "(this test must observe the pre-patch contract)."
    )

    runner = types.SimpleNamespace()
    runner.config = types.SimpleNamespace(platforms={})

    async def _no_abort(*_a, **_k):
        return False

    runner._abort_startup_if_shutdown_requested = _no_abort
    result = asyncio.run(fn(runner))
    assert isinstance(result, tuple) and len(result) == 6, (
        f"in-tree _connect_configured_platforms must return a 6-tuple; got "
        f"{result!r}. start() unpacks (aborted + 5 counters)."
    )
    aborted, connected, background, enabled, nonretryable, retryable = result
    assert aborted is False
    assert connected == 0 and background == 0 and enabled == 0
    assert nonretryable == [] and retryable == []


def test_seam_helper_dependencies_exist_with_compatible_signatures():
    """Pin the upstream GatewayRunner methods the overlay loop/helpers call.

    The background loop and its helpers call these by name. An upstream rename
    would make the patched loop blow up at gateway startup; pin them so it's a
    CI failure instead.
    """
    from gateway.run import GatewayRunner

    expected = {
        "_create_adapter": ["self", "platform", "config"],
        "_connect_adapter_with_timeout": ["self", "adapter", "platform"],
        "_safe_adapter_disconnect": ["self", "adapter", "platform"],
        "_sync_voice_mode_state_to_adapter": ["self", "adapter"],
        "_wire_teams_pipeline_runtime": ["self"],
        # v0.18: per-adapter fail-closed authz factory (upstream bb304b4914) —
        # the overlay's _prepare_adapter wires it like upstream's inline block.
        "_make_adapter_auth_check": ["self", "platform"],
        # v0.18: shutdown-during-startup abort — the overlay loop mirrors
        # upstream's three checkpoints.
        "_abort_startup_if_shutdown_requested": ["self", "adapter", "platform"],
    }
    for name, params in expected.items():
        fn = getattr(GatewayRunner, name, None)
        assert fn is not None, (
            f"GatewayRunner.{name} is gone — apex_overlay gateway_bootstrap "
            f"depends on it. Update the overlay and seam-test together."
        )
        sig = inspect.signature(fn)
        got = list(sig.parameters)
        # The overlay calls these with the pinned positional shape; upstream
        # may append OPTIONAL params (e.g. v0.18 added is_reconnect to
        # _connect_adapter_with_timeout) without breaking the call.
        assert got[: len(params)] == params, (
            f"GatewayRunner.{name} signature changed to {got!r}; the overlay "
            f"calls it as {name}({', '.join(params[1:])})."
        )
        for extra in got[len(params):]:
            p = sig.parameters[extra]
            assert p.default is not inspect.Parameter.empty or p.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ), (
                f"GatewayRunner.{name} grew a REQUIRED param {extra!r}; the "
                f"overlay's {name}({', '.join(params[1:])}) call would break."
            )

    # _update_platform_runtime_status is called with these keyword args.
    ups = inspect.signature(GatewayRunner._update_platform_runtime_status).parameters
    for kw in ("platform_state", "error_code", "error_message"):
        assert kw in ups, (
            f"_update_platform_runtime_status lost the {kw!r} kwarg the overlay "
            f"passes."
        )

    # _schedule_resume_pending_sessions(platform=...) is called by the
    # background connect helper.
    rps = inspect.signature(GatewayRunner._schedule_resume_pending_sessions).parameters
    assert "platform" in rps, (
        "_schedule_resume_pending_sessions lost its platform kwarg — the "
        "background connect helper calls it with platform=platform."
    )


def test_seam_platform_feishu_and_background_marker():
    """Pin the two upstream surfaces that *select* background behavior.

    1. Platform.FEISHU (the create-in-background platform).
    2. The CONNECT_IN_BACKGROUND adapter-class attribute convention — an
       adapter advertises background connect by setting it truthy.
    """
    from gateway.config import Platform
    from gateway.platforms.base import BasePlatformAdapter

    assert hasattr(Platform, "FEISHU"), (
        "Platform.FEISHU is gone — _platform_creation_connects_in_background "
        "keys off it. Re-point the overlay."
    )
    # BasePlatformAdapter must tolerate the marker; default is falsy/absent so
    # ordinary adapters connect inline.
    assert not getattr(BasePlatformAdapter, "CONNECT_IN_BACKGROUND", False), (
        "BasePlatformAdapter now defaults CONNECT_IN_BACKGROUND truthy — every "
        "adapter would background-connect. The overlay assumes opt-in."
    )


def test_seam_build_channel_directory_importable():
    """The background connect helper refreshes the channel directory."""
    from gateway.channel_directory import build_channel_directory

    params = list(inspect.signature(build_channel_directory).parameters)
    assert params and params[0] == "adapters", (
        f"build_channel_directory first param changed to {params!r}; the "
        f"overlay calls build_channel_directory(self.adapters)."
    )


# ---------------------------------------------------------------------------
# apply() — binds the method + 8 helpers, idempotent
# ---------------------------------------------------------------------------

def test_apply_binds_loop_and_helpers_and_is_idempotent():
    """apply() must replace the loop, bind all 8 helpers, and no-op on repeat."""
    from gateway.run import GatewayRunner

    gateway_bootstrap._APPLIED = False
    assert gateway_bootstrap.apply() is True

    # The loop method now carries our marker (it was swapped).
    patched = getattr(GatewayRunner, gateway_bootstrap._TARGET_LOOP_METHOD)
    assert getattr(patched, gateway_bootstrap._MARK, False), (
        "after apply() _connect_configured_platforms must be the overlay "
        "version (marked)."
    )

    # Every helper the loop calls is bound onto the class.
    for name in gateway_bootstrap._HELPER_NAMES:
        assert hasattr(GatewayRunner, name), (
            f"apply() did not bind helper {name!r} onto GatewayRunner."
        )

    # Idempotent: second apply is a no-op and must not error or double-wrap.
    assert gateway_bootstrap.apply() is True
    still = getattr(GatewayRunner, gateway_bootstrap._TARGET_LOOP_METHOD)
    assert still is patched


# ---------------------------------------------------------------------------
# Behavior — background adapter is scheduled off the loop; inline otherwise
# ---------------------------------------------------------------------------

class _FakePlatform:
    """Stand-in for a Platform enum member with a .value."""

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, _FakePlatform) and other.value == self.value


class _FakeAdapter:
    def __init__(self, *, background: bool):
        if background:
            self.CONNECT_IN_BACKGROUND = True
        self.connected = False
        self.has_fatal_error = False
        self.platform = None  # read by _prepare_adapter for the authz factory

    # Wired by _prepare_adapter — accept and ignore.
    def set_message_handler(self, *_a, **_k):
        pass

    def set_fatal_error_handler(self, *_a, **_k):
        pass

    def set_session_store(self, *_a, **_k):
        pass

    def set_busy_session_handler(self, *_a, **_k):
        pass

    def set_topic_recovery_fn(self, *_a, **_k):
        pass

    def set_authorization_check(self, *_a, **_k):
        pass


def _make_stub_runner(monkeypatch):
    """A minimal object carrying just what the overlay loop touches.

    We bind the overlay's freshly-applied class methods onto a SimpleNamespace
    so we exercise the real loop + helpers without standing up a full
    GatewayRunner (and its heavy __init__).
    """
    from gateway.run import GatewayRunner

    gateway_bootstrap._APPLIED = False
    assert gateway_bootstrap.apply() is True

    stub = types.SimpleNamespace()
    stub.adapters = {}
    stub._failed_platforms = {}
    stub._background_tasks = set()
    stub._busy_text_mode = False
    stub.delivery_router = types.SimpleNamespace(adapters={})
    stub.session_store = object()
    stub.config = types.SimpleNamespace(platforms={})

    # No-op the runner state/refresh calls the helpers make.
    stub._update_platform_runtime_status = lambda *a, **k: None
    stub._sync_voice_mode_state_to_adapter = lambda *a, **k: None
    stub._wire_teams_pipeline_runtime = lambda *a, **k: None
    stub._schedule_resume_pending_sessions = lambda *a, **k: 0
    stub._handle_message = lambda *a, **k: None
    stub._handle_adapter_fatal_error = lambda *a, **k: None
    stub._handle_active_session_busy_message = lambda *a, **k: None
    stub._recover_telegram_topic_thread_id = lambda *a, **k: None
    stub._make_adapter_auth_check = lambda *a, **k: (lambda *_a, **_k: True)

    async def _no_abort(*_a, **_k):
        return False

    stub._abort_startup_if_shutdown_requested = _no_abort

    async def _safe_disc(adapter, platform):
        return None

    stub._safe_adapter_disconnect = _safe_disc

    async def _connect_with_timeout(adapter, platform):
        adapter.connected = True
        return True

    stub._connect_adapter_with_timeout = _connect_with_timeout

    # Bind the (now-applied) overlay methods from the class onto the stub.
    for name in gateway_bootstrap._HELPER_NAMES + (gateway_bootstrap._TARGET_LOOP_METHOD,):
        fn = getattr(GatewayRunner, name)
        setattr(stub, name, types.MethodType(fn, stub))

    # _platform_creation_connects_in_background keys off Platform.FEISHU; our
    # fake platforms aren't enum members, so force the test's intent explicitly.
    monkeypatch.setattr(
        stub, "_platform_creation_connects_in_background", lambda platform: False
    )
    return stub


@pytest.mark.asyncio
async def test_background_adapter_scheduled_not_inline(monkeypatch):
    """A CONNECT_IN_BACKGROUND adapter is scheduled; the loop returns at once.

    Proves the hc-384/385 contract: Feishu (background) does not block the loop
    body — it is counted under background_connect_count and connected via a
    tracked asyncio task, while a normal adapter connects inline and lands in
    adapters immediately.
    """
    stub = _make_stub_runner(monkeypatch)

    bg_platform = _FakePlatform("feishu")
    inline_platform = _FakePlatform("telegram")
    bg_adapter = _FakeAdapter(background=True)
    inline_adapter = _FakeAdapter(background=False)

    cfg = types.SimpleNamespace(enabled=True)
    stub.config.platforms = {bg_platform: cfg, inline_platform: cfg}

    def _create_adapter(platform, platform_config):
        return {bg_platform: bg_adapter, inline_platform: inline_adapter}[platform]

    stub._create_adapter = _create_adapter

    (
        aborted,
        connected,
        background,
        enabled,
        nonretryable,
        retryable,
    ) = await stub._connect_configured_platforms()

    assert aborted is False
    assert enabled == 2
    assert background == 1, "the CONNECT_IN_BACKGROUND adapter must be deferred"
    assert connected == 1, "the ordinary adapter must connect inline"

    # Inline adapter is already registered; background one is not yet (its task
    # is scheduled and tracked).
    assert inline_platform in stub.adapters
    assert bg_platform not in stub.adapters
    assert len(stub._background_tasks) == 1

    # Let the background task finish, then it too should register.
    await asyncio.sleep(0)
    for _ in range(50):
        if bg_platform in stub.adapters:
            break
        await asyncio.sleep(0.01)
    assert bg_platform in stub.adapters, "background connect never registered"


@pytest.mark.asyncio
async def test_all_inline_when_nothing_backgrounds(monkeypatch):
    """With no background adapter, every platform connects inline (upstream-equiv)."""
    stub = _make_stub_runner(monkeypatch)

    p1 = _FakePlatform("telegram")
    p2 = _FakePlatform("discord")
    cfg = types.SimpleNamespace(enabled=True)
    stub.config.platforms = {p1: cfg, p2: cfg}
    stub._create_adapter = lambda platform, pc: _FakeAdapter(background=False)

    aborted, connected, background, enabled, _nonr, _retr = await stub._connect_configured_platforms()
    assert aborted is False
    assert (connected, background, enabled) == (2, 0, 2)
    assert p1 in stub.adapters and p2 in stub.adapters
    assert stub._background_tasks == set()


# ---------------------------------------------------------------------------
# Wiring — the seam loads via the apex-overlay plugin, before the connect call
# ---------------------------------------------------------------------------

def test_plugin_register_applies_gateway_bootstrap_seam():
    """The bundled apex-overlay plugin's register() applies this seam too."""
    from tests.apex_overlay.conftest import run_plugin_register_with_stubbed_seams

    called = run_plugin_register_with_stubbed_seams(
        "_apex_overlay_plugin_under_test_gw"
    )
    assert "gateway_bootstrap" in called, (
        "plugin.register() must call gateway_bootstrap.apply()"
    )


def test_seam_loads_before_platform_connect_in_run_py():
    """In run.py's start(), discover_plugins() runs BEFORE the connect call.

    discover_plugins() is what applies the overlay (via the apex-overlay
    plugin). It must run before self._connect_configured_platforms() so the
    loop is already the background-connect version at call time — otherwise the
    seam is a no-op for that boot.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    src = (repo / "gateway" / "run.py").read_text(encoding="utf-8")

    discover_idx = src.find("discover_plugins()")
    call_idx = src.find("await self._connect_configured_platforms()")
    assert discover_idx != -1, "discover_plugins() call missing in run.py start()"
    assert call_idx != -1, (
        "the _connect_configured_platforms() call hook is missing in run.py — "
        "the extraction the overlay depends on was reverted."
    )
    assert discover_idx < call_idx, (
        "discover_plugins() must run before _connect_configured_platforms() so "
        "the apex_overlay background-startup seam is installed in time."
    )


def test_apex_overlay_enabled_in_config():
    """cli-config.yaml.example enables the apex-overlay plugin (config tier)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    cfg = (repo / "cli-config.yaml.example").read_text(encoding="utf-8")
    assert "apex-overlay" in cfg, (
        "cli-config.yaml.example must list apex-overlay under plugins.enabled "
        "or the gateway bootstrap seam never loads in production."
    )
