"""Cross-session ContextVar *inheritance* leak guard.

Companion to ``tests/tools/test_local_env_session_leak.py``. That file covers
the ``os.environ``-mirror leak (a subprocess inheriting a foreign *global* when
this task's ContextVar is ``_UNSET``). THIS file covers a distinct, subtler
variant that the ``_UNSET``-strip guard does NOT catch:

    Each gateway message is processed in its own asyncio task, created via
    ``create_task`` — which snapshots the spawning context with
    ``copy_context()``. If message B's task is created from a context where a
    *concurrent* message A had ALREADY called ``set_session_vars``, B inherits
    A's **set** ContextVars. Between B's task start and B's own
    ``set_session_vars`` call, any subprocess B spawns reads A's
    ``HERMES_SESSION_*`` identity through the subprocess-env bridge. The bridge's
    strip-on-``_UNSET`` rule is no help: the inherited vars are set-to-A, not
    ``_UNSET``.

Verified in production 2026-06-21: a ``/bug`` turn ran ``bug_thread.py whoami``
and read a concurrent session's ticket (``cursor-captive-modals``) instead of
its own, because its task inherited that session's bound ContextVars.

The fix: ``gateway.session_context.reset_session_vars`` resets every session var
to ``_UNSET`` at the top of the per-message handler (``GatewayRunner._handle_message``),
*before* any work, so an inherited identity is dropped and the pre-bind window
strips safe instead of leaking the sibling's. The handler then binds its own
session a few steps later.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace
from typing import Any, Dict, cast

import pytest

import gateway.session_context as sc
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import (
    _SESSION_ASYNC_DELIVERY,
    _UNSET,
    _VAR_MAP,
    async_delivery_supported,
    reset_session_vars,
    set_session_vars,
)
from tools.environments.local import _make_run_env
from tools.thread_context import propagate_context_to_thread

SESSION_VARS = list(_VAR_MAP.keys())

MINE: Dict[str, Any] = dict(
    session_key="agent:main:discord:thread:MINE:MINE",
    platform="discord",
    chat_id="MINE_CHAT",
    thread_id="MINE_THREAD",
    user_id="MINE_USER",
    chat_name="mine",
    message_id="MINE_MSG",
)
FOREIGN: Dict[str, Any] = dict(
    session_key="agent:main:discord:thread:FOREIGN:FOREIGN",
    platform="discord",
    chat_id="FOREIGN_CHAT",
    thread_id="FOREIGN_THREAD",
    user_id="FOREIGN_USER",
    chat_name="foreign",
    message_id="FOREIGN_MSG",
)


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + engaged-latch slate per test, restored afterwards."""
    import os

    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_async = _SESSION_ASYNC_DELIVERY.get()
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    sc._session_context_engaged = True  # a concurrent multi-session host is engaged
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        _SESSION_ASYNC_DELIVERY.set(saved_async)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _spawn_view():
    """What a subprocess spawned right now would see for the session vars."""
    env = _make_run_env({})
    return {
        "HERMES_SESSION_CHAT_ID": env.get("HERMES_SESSION_CHAT_ID"),
        "HERMES_SESSION_THREAD_ID": env.get("HERMES_SESSION_THREAD_ID"),
        "HERMES_SESSION_KEY": env.get("HERMES_SESSION_KEY"),
        "HERMES_SESSION_ID": env.get("HERMES_SESSION_ID"),
        "HERMES_SESSION_MESSAGE_ID": env.get("HERMES_SESSION_MESSAGE_ID"),
    }


def _discord_thread_source(label: str, *, message_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=label,
        chat_type="thread",
        user_id="owner",
        thread_id=label,
        message_id=message_id,
    )


def _runner_with_inner(inner):
    runner = cast(Any, GatewayRunner.__new__(GatewayRunner))
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.adapters = {
        Platform.DISCORD: SimpleNamespace(supports_async_delivery=True),
    }
    runner._executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="test-session-context",
    )
    runner._run_agent_inner = inner
    return runner


async def _worker_spawn_view(runner):
    """Read the tool/subprocess environment from the gateway worker thread."""
    return await GatewayRunner._run_in_executor_with_context(runner, _spawn_view)


def _shutdown_test_executor(runner) -> None:
    executor = getattr(runner, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


def _poison_worker_thread(executor: ThreadPoolExecutor) -> None:
    """Bind FOREIGN directly on the reusable worker thread."""
    future = executor.submit(lambda: set_session_vars(**FOREIGN, session_id="FOREIGN_SID"))
    future.result(timeout=5)


@pytest.mark.asyncio
async def test_run_agent_rebinds_full_turn_context_before_inner_dispatch():
    """The _run_agent choke point must not trust its caller's current context."""
    set_session_vars(**FOREIGN, session_id="FOREIGN_SID")
    source = _discord_thread_source("MINE_THREAD", message_id="SOURCE_MSG")
    observed = {}
    runner = _runner_with_inner(None)

    async def inner(*args, **kwargs):
        observed.update(await _worker_spawn_view(runner))
        return {"final_response": "ok"}

    runner._run_agent_inner = inner
    try:
        await GatewayRunner._run_agent(
            runner,
            message="mine",
            context_prompt="",
            history=[],
            source=source,
            session_id="MINE_SID",
            session_key=MINE["session_key"],
            event_message_id="MINE_MSG",
        )
    finally:
        _shutdown_test_executor(runner)

    assert observed == {
        "HERMES_SESSION_CHAT_ID": "MINE_THREAD",
        "HERMES_SESSION_THREAD_ID": "MINE_THREAD",
        "HERMES_SESSION_KEY": MINE["session_key"],
        "HERMES_SESSION_ID": "MINE_SID",
        "HERMES_SESSION_MESSAGE_ID": "MINE_MSG",
    }


@pytest.mark.asyncio
async def test_gateway_executor_context_run_overrides_reused_thread_residue_bidirectionally():
    """A recycled gateway worker thread must not leak its previous ContextVars."""
    runner = _runner_with_inner(None)
    try:
        _poison_worker_thread(runner._executor)

        source_a = _discord_thread_source("THREAD_A", message_id="SOURCE_A_MSG")
        source_b = _discord_thread_source("THREAD_B", message_id="SOURCE_B_MSG")
        key_a = "agent:main:discord:thread:THREAD_A:THREAD_A"
        key_b = "agent:main:discord:thread:THREAD_B:THREAD_B"

        async def run_bound(source, key, sid, mid):
            tokens = GatewayRunner._set_session_vars_for_source(
                runner,
                source=source,
                session_key=key,
                session_id=sid,
                message_id=mid,
            )
            try:
                return await _worker_spawn_view(runner)
            finally:
                from gateway.session_context import restore_session_vars
                restore_session_vars(tokens)

        observed_a = await run_bound(source_a, key_a, "SID_A", "MSG_A")
        observed_b = await run_bound(source_b, key_b, "SID_B", "MSG_B")
    finally:
        _shutdown_test_executor(runner)

    assert observed_a == {
        "HERMES_SESSION_CHAT_ID": "THREAD_A",
        "HERMES_SESSION_THREAD_ID": "THREAD_A",
        "HERMES_SESSION_KEY": key_a,
        "HERMES_SESSION_ID": "SID_A",
        "HERMES_SESSION_MESSAGE_ID": "MSG_A",
    }
    assert observed_b == {
        "HERMES_SESSION_CHAT_ID": "THREAD_B",
        "HERMES_SESSION_THREAD_ID": "THREAD_B",
        "HERMES_SESSION_KEY": key_b,
        "HERMES_SESSION_ID": "SID_B",
        "HERMES_SESSION_MESSAGE_ID": "MSG_B",
    }
    assert observed_a["HERMES_SESSION_KEY"] != key_b
    assert observed_b["HERMES_SESSION_KEY"] != key_a


def test_tool_thread_context_wrapper_overrides_reused_thread_residue_bidirectionally():
    """The tool worker wrapper must snapshot the agent-thread context, not residue."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test-tool-context")
    try:
        _poison_worker_thread(executor)

        def submit_bound(binding: Dict[str, Any], sid: str):
            tokens = set_session_vars(**binding, session_id=sid)
            try:
                return executor.submit(
                    propagate_context_to_thread(_spawn_view)
                ).result(timeout=5)
            finally:
                from gateway.session_context import restore_session_vars
                restore_session_vars(tokens)

        mine = submit_bound(MINE, "MINE_SID")
        foreign = submit_bound(FOREIGN, "FOREIGN_SID")
    finally:
        executor.shutdown(wait=True)

    assert mine["HERMES_SESSION_KEY"] == MINE["session_key"]
    assert mine["HERMES_SESSION_CHAT_ID"] == MINE["chat_id"]
    assert mine["HERMES_SESSION_ID"] == "MINE_SID"
    assert foreign["HERMES_SESSION_KEY"] == FOREIGN["session_key"]
    assert foreign["HERMES_SESSION_CHAT_ID"] == FOREIGN["chat_id"]
    assert foreign["HERMES_SESSION_ID"] == "FOREIGN_SID"
    assert mine["HERMES_SESSION_KEY"] != FOREIGN["session_key"]
    assert foreign["HERMES_SESSION_KEY"] != MINE["session_key"]


def test_tool_submit_warns_on_agent_context_mismatch(caplog):
    """The tool-submit net reports a stale bound session before snapshotting it."""
    from agent.tool_executor import _warn_on_tool_submit_session_mismatch

    set_session_vars(**FOREIGN, session_id="FOREIGN_SID")
    agent = SimpleNamespace(_gateway_session_key=MINE["session_key"])

    with caplog.at_level("WARNING", logger="agent.tool_executor"):
        _warn_on_tool_submit_session_mismatch(agent)

    assert "Tool executor context mismatch" in caplog.text
    assert FOREIGN["session_key"] in caplog.text
    assert MINE["session_key"] in caplog.text


@pytest.mark.asyncio
async def test_run_agent_composes_session_binding_with_profile_scope(tmp_path):
    """The worker must inherit both the turn identity and multiplex profile."""
    from hermes_constants import get_hermes_home

    set_session_vars(**FOREIGN, session_id="FOREIGN_SID")
    source = _discord_thread_source("MINE_THREAD", message_id="SOURCE_MSG")
    source.profile = "coder"
    runner = _runner_with_inner(None)
    runner.config.multiplex_profiles = True
    runner._resolve_profile_home_for_source = lambda _source: tmp_path
    observed = {}

    def worker_view():
        return {"session": _spawn_view(), "home": str(get_hermes_home())}

    async def inner(*args, **kwargs):
        observed.update(
            await GatewayRunner._run_in_executor_with_context(runner, worker_view)
        )
        return {"final_response": "ok"}

    runner._run_agent_inner = inner
    try:
        await GatewayRunner._run_agent(
            runner,
            message="mine",
            context_prompt="",
            history=[],
            source=source,
            session_id="MINE_SID",
            session_key=MINE["session_key"],
            event_message_id="MINE_MSG",
        )
    finally:
        _shutdown_test_executor(runner)

    assert observed["session"]["HERMES_SESSION_KEY"] == MINE["session_key"]
    assert observed["home"] == str(tmp_path)


@pytest.mark.asyncio
async def test_run_agent_inner_warns_before_dispatch_on_context_mismatch(caplog):
    """The runtime net reports a turn whose bound key disagrees with its call."""
    set_session_vars(**FOREIGN, session_id="FOREIGN_SID")
    source = _discord_thread_source("MINE_THREAD", message_id="MINE_MSG")
    expected_key = "agent:main:discord:thread:MINE_THREAD:MINE_THREAD"
    runner = _runner_with_inner(None)
    runner._get_proxy_url = lambda: "http://proxy.invalid"

    async def proxy_call(**kwargs):
        return {"final_response": "ok"}

    runner._run_agent_via_proxy = proxy_call
    try:
        with caplog.at_level("WARNING", logger="gateway.run"):
            await GatewayRunner._run_agent_inner(
                runner,
                message="mine",
                context_prompt="",
                history=[],
                source=source,
                session_id="MINE_SID",
                session_key=expected_key,
                event_message_id="MINE_MSG",
            )
    finally:
        _shutdown_test_executor(runner)

    assert "Agent executor context mismatch" in caplog.text
    assert FOREIGN["session_key"] in caplog.text
    assert expected_key in caplog.text


@pytest.mark.asyncio
async def test_recursive_run_agent_rebinds_queued_cross_session_followup():
    """A queued B turn recursing from A must execute with B's full identity."""
    source_a = _discord_thread_source("THREAD_A", message_id="SOURCE_A_MSG")
    source_b = _discord_thread_source("THREAD_B", message_id="SOURCE_B_MSG")
    key_a = "agent:main:discord:thread:THREAD_A:THREAD_A"
    key_b = "agent:main:discord:thread:THREAD_B:THREAD_B"
    set_session_vars(
        platform="discord",
        chat_id="THREAD_A",
        thread_id="THREAD_A",
        session_key=key_a,
        session_id="SID_A",
        message_id="MSG_A",
    )
    observed = {}
    runner = _runner_with_inner(None)

    async def inner(message, context_prompt, history, source, session_id, **kwargs):
        if message == "turn-a":
            observed["a_before"] = await _worker_spawn_view(runner)
            result = await GatewayRunner._run_agent(
                runner,
                message="queued-turn-b",
                context_prompt=context_prompt,
                history=history,
                source=source_b,
                session_id="SID_B",
                session_key=key_b,
                event_message_id="MSG_B",
                _interrupt_depth=1,
            )
            observed["a_after"] = await _worker_spawn_view(runner)
            return result
        observed["b"] = await _worker_spawn_view(runner)
        return {"final_response": "ok"}

    runner._run_agent_inner = inner
    try:
        await GatewayRunner._run_agent(
            runner,
            message="turn-a",
            context_prompt="",
            history=[],
            source=source_a,
            session_id="SID_A",
            session_key=key_a,
            event_message_id="MSG_A",
        )
    finally:
        _shutdown_test_executor(runner)

    expected_a = {
        "HERMES_SESSION_CHAT_ID": "THREAD_A",
        "HERMES_SESSION_THREAD_ID": "THREAD_A",
        "HERMES_SESSION_KEY": key_a,
        "HERMES_SESSION_ID": "SID_A",
        "HERMES_SESSION_MESSAGE_ID": "MSG_A",
    }
    expected_b = {
        "HERMES_SESSION_CHAT_ID": "THREAD_B",
        "HERMES_SESSION_THREAD_ID": "THREAD_B",
        "HERMES_SESSION_KEY": key_b,
        "HERMES_SESSION_ID": "SID_B",
        "HERMES_SESSION_MESSAGE_ID": "MSG_B",
    }
    assert observed == {
        "a_before": expected_a,
        "b": expected_b,
        "a_after": expected_a,
    }


@pytest.mark.asyncio
async def test_recursive_run_agent_rebinds_steer_fallback_message_anchor():
    """A same-session /steer fallback must replace the outer turn's message id."""
    source = _discord_thread_source("THREAD_A", message_id="SOURCE_A_MSG")
    key = "agent:main:discord:thread:THREAD_A:THREAD_A"
    set_session_vars(
        platform="discord",
        chat_id="THREAD_A",
        thread_id="THREAD_A",
        session_key=key,
        session_id="SID_A",
        message_id="OUTER_MSG",
    )
    observed = {}
    runner = _runner_with_inner(None)

    async def inner(message, context_prompt, history, source, session_id, **kwargs):
        if message == "outer-turn":
            return await GatewayRunner._run_agent(
                runner,
                message="steer-fallback-turn",
                context_prompt=context_prompt,
                history=history,
                source=source,
                session_id=session_id,
                session_key=key,
                event_message_id="STEER_MSG",
                _interrupt_depth=1,
            )
        observed.update(await _worker_spawn_view(runner))
        return {"final_response": "ok"}

    runner._run_agent_inner = inner
    try:
        await GatewayRunner._run_agent(
            runner,
            message="outer-turn",
            context_prompt="",
            history=[],
            source=source,
            session_id="SID_A",
            session_key=key,
            event_message_id="OUTER_MSG",
        )
    finally:
        _shutdown_test_executor(runner)

    assert observed["HERMES_SESSION_MESSAGE_ID"] == "STEER_MSG"


async def _child_turn(reset_first: bool):
    """Simulate message B's processing task: created (copy_context) from a
    parent context where message A already bound its session.

    Returns the subprocess view from the *pre-bind window* — before B calls its
    own set_session_vars. With ``reset_first`` (the fix), B resets at entry.
    """
    captured = {}

    def _b_body():
        if reset_first:
            reset_session_vars()  # THE FIX: handler-entry reset
        captured["window"] = _spawn_view()  # pre-bind window
        set_session_vars(**FOREIGN)  # B binds its own session
        captured["bound"] = _spawn_view()

    # create_task snapshots the CURRENT (A-bound) context, exactly like the
    # gateway's per-message dispatch.
    await asyncio.create_task(_async_noop(_b_body))
    return captured


async def _async_noop(fn):
    fn()


def test_child_task_inherits_foreign_session_without_reset():
    """REPRODUCER: without the entry reset, B's pre-bind window leaks A's id.

    This is the production hijack. Asserting the leak EXISTS documents the bug
    the fix closes; the next test proves the fix.
    """
    set_session_vars(**MINE)  # parent A binds in the current context

    captured = asyncio.run(_child_turn(reset_first=False))

    # The pre-bind window inherited A's (MINE) identity — the leak.
    assert captured["window"]["HERMES_SESSION_CHAT_ID"] == "MINE_CHAT", (
        "Expected to reproduce the inheritance leak (window sees parent's "
        f"MINE_CHAT); got {captured['window']!r}"
    )


def test_reset_session_vars_closes_inheritance_leak():
    """THE FIX: resetting at handler entry strips the inherited identity.

    After reset_session_vars(), the pre-bind window must see NO session vars
    (stripped, because they are _UNSET in this context and the process is
    engaged) — NOT the parent's MINE_*. B's own bind then takes effect normally.
    """
    set_session_vars(**MINE)  # parent A binds in the current context

    captured = asyncio.run(_child_turn(reset_first=True))

    window = captured["window"]
    for var in ("HERMES_SESSION_CHAT_ID", "HERMES_SESSION_THREAD_ID", "HERMES_SESSION_KEY"):
        assert window[var] is None, (
            f"{var} leaked the parent session after reset: {window[var]!r}"
        )

    # B's own session still binds correctly after the reset window.
    assert captured["bound"]["HERMES_SESSION_CHAT_ID"] == "FOREIGN_CHAT"
    assert captured["bound"]["HERMES_SESSION_KEY"] == FOREIGN["session_key"]


def test_reset_session_vars_restores_unset_not_empty():
    """reset_session_vars sets _UNSET (not "" like clear_session_vars).

    The distinction matters: "" is 'explicitly cleared' (suppresses os.environ
    fallback, used when a handler finishes); _UNSET is 'never bound here' (lets
    the bridge strip and a CLI fallback resolve). Entry-reset must use _UNSET.
    """
    set_session_vars(**MINE)
    reset_session_vars()
    for name, var in _VAR_MAP.items():
        assert var.get() is _UNSET, f"{name} is {var.get()!r}, expected _UNSET"


# ---------------------------------------------------------------------------
# Async-delivery capability inheritance (the sibling var outside _VAR_MAP)
# ---------------------------------------------------------------------------
#
# ``_SESSION_ASYNC_DELIVERY`` is NOT in ``_VAR_MAP`` — it is a bool capability
# flag read via ``async_delivery_supported()``, not a string ``HERMES_SESSION_*``
# var read via ``get_session_env``. So the ``for var in _VAR_MAP.values()`` loop
# in ``reset_session_vars`` does not touch it; it must be reset explicitly.
#
# Without that explicit reset, a task created (copy_context) from a context where
# a *concurrent* sibling A had bound ``async_delivery=False`` (the stateless API
# server) inherits A's ``False``. In B's pre-bind window
# ``async_delivery_supported()`` then wrongly reports B's channel as unable to
# route a background completion — even though B is e.g. a real gateway turn that
# CAN. Tools (terminal notify_on_complete / watch_patterns, delegate_task
# background=True) would refuse a promise the channel could actually keep.


async def _child_async_delivery(reset_first: bool):
    """Simulate message B's task created from a parent context where a stateless
    sibling A bound ``async_delivery=False``.

    Returns ``async_delivery_supported()`` as seen in B's pre-bind window.
    """
    captured = {}

    def _b_body():
        if reset_first:
            reset_session_vars()  # THE FIX: handler-entry reset
        captured["window"] = async_delivery_supported()  # pre-bind window

    await asyncio.create_task(_async_noop(_b_body))
    return captured


def test_child_task_inherits_foreign_async_delivery_without_reset():
    """REPRODUCER: without the entry reset, B inherits A's async_delivery=False.

    A stateless adapter (API server) opts out with async_delivery=False. A task
    spawned from that context sees the inherited False in its pre-bind window —
    the leak the explicit reset closes.
    """
    set_session_vars(**FOREIGN, async_delivery=False)  # stateless sibling A

    captured = asyncio.run(_child_async_delivery(reset_first=False))

    assert captured["window"] is False, (
        "Expected to reproduce the async-delivery inheritance leak (window "
        f"inherits A's async_delivery=False); got {captured['window']!r}"
    )


def test_reset_session_vars_closes_async_delivery_leak():
    """THE FIX: resetting at handler entry drops the inherited async_delivery.

    After reset_session_vars(), the pre-bind window must fall back to the
    default-supported behavior (True) — NOT the stateless sibling's False — so a
    real gateway turn isn't wrongly told its channel can't route async delivery.
    """
    set_session_vars(**FOREIGN, async_delivery=False)  # stateless sibling A

    captured = asyncio.run(_child_async_delivery(reset_first=True))

    assert captured["window"] is True, (
        "After reset, async delivery must default to supported; "
        f"got {captured['window']!r}"
    )


def test_reset_session_vars_restores_async_delivery_unset():
    """reset_session_vars restores _SESSION_ASYNC_DELIVERY to the _UNSET sentinel.

    The capability flag must read 'never bound here' (_UNSET), not a falsy value,
    so async_delivery_supported() resolves to the default-supported path rather
    than being mistaken for an opted-out stateless adapter.
    """
    set_session_vars(**FOREIGN, async_delivery=False)
    reset_session_vars()
    assert _SESSION_ASYNC_DELIVERY.get() is _UNSET, (
        f"_SESSION_ASYNC_DELIVERY is {_SESSION_ASYNC_DELIVERY.get()!r}, expected _UNSET"
    )
    assert async_delivery_supported() is True
