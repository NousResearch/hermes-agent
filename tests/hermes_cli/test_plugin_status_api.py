"""Plugin status API (ctx.emit_status / ctx.platform_actions.send_status).

Regressions pinned here:
* capability gate (``gateway.plugin_status``) default-off; audit trail per action
* existing verbs stay gated by ``gateway.platform_actions`` (unchanged behaviour)
* target resolution priority: explicit session_id (FAIL-CLOSED) > ContextVars (strict, no
  os.environ fallback) > explicit platform+chat_id > invalid_argument (never a default chat)
* concurrency safety: sync emit never waits (no Future.result); same-gateway-loop and
  cross-thread scheduling both deliver without deadlock; async primitive awaits via
  wrap_future across loops
* failure isolation: delivery failure never raises, never changes hook/tool results
* namespace: status key is plugin-prefixed from the facade identity, not caller text;
  per-chat throttle independence; bounded throttle map
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from hermes_cli import platform_actions as pa
from hermes_cli.platform_actions import CAPABILITY_STATUS_ID, PlatformActions

# -- fakes --------------------------------------------------------------------

SR = lambda ok=True, mid="m1": SimpleNamespace(success=ok, message_id=mid, error="" if ok else "nope")


def _status_adapter(connected=True):
    a = SimpleNamespace(platform=Platform.TELEGRAM, is_connected=connected)
    a.status_calls = []
    a.send_calls = []
    async def _sus(chat_id, status_key, content, metadata=None):
        a.status_calls.append({"chat_id": chat_id, "status_key": status_key, "content": content, "metadata": metadata})
        return SR(mid=f"s{len(a.status_calls)}")
    async def _send(chat_id, content, reply_to=None, metadata=None):
        a.send_calls.append({"chat_id": chat_id, "content": content})
        return SR(mid=f"p{len(a.send_calls)}")
    a.send_or_update_status = _sus
    a.send = _send
    return a


def _plain_adapter(connected=True):
    """No send_or_update_status → plain-send fallback (e.g. discord-shaped)."""
    a = SimpleNamespace(platform=Platform.DISCORD, is_connected=connected)
    a.send_calls = []
    async def _send(chat_id, content, reply_to=None, metadata=None):
        a.send_calls.append({"chat_id": chat_id, "content": content})
        return SR(mid=f"p{len(a.send_calls)}")
    a.send = _send
    return a


def _failing_adapter():
    a = SimpleNamespace(platform=Platform.TELEGRAM, is_connected=True)
    async def _sus(chat_id, status_key, content, metadata=None):
        raise RuntimeError("telegram exploded")
    async def _send(chat_id, content, reply_to=None, metadata=None):
        raise RuntimeError("telegram exploded")
    a.send_or_update_status = _sus
    a.send = _send
    return a


def _entry(platform, chat_id, thread_id=None, profile=None):
    return SimpleNamespace(
        origin=SimpleNamespace(platform=platform, chat_id=chat_id, thread_id=thread_id, profile=profile),
        platform=platform,
    )


class FakeStore:
    def __init__(self, entries=None):
        self.entries = entries or {}
    def lookup_by_session_id(self, session_id):
        return self.entries.get(session_id)


class FakeRunner:
    """Runner exposing the same lazy-read surface as GatewayRunner."""
    def __init__(self, adapters, loop=None, store=None, profile_adapters=None):
        self.adapters = adapters
        self._gateway_loop = loop
        self.session_store = store if store is not None else FakeStore()
        self._profile_adapters = profile_adapters or {}
        self.resolutions = []
    def _authorization_adapter(self, platform_enum, profile_name):
        self.resolutions.append((platform_enum, profile_name))
        return self._profile_adapters.get((platform_enum, profile_name), self.adapters.get(platform_enum))


def _grant(*ids, **kw):
    """Patch the capability check the facade performs (keyed by capability id)."""
    granted = set(ids) | {k for k, v in kw.items() if v}
    return patch(
        "hermes_cli.plugin_capabilities.plugin_capability_granted",
        side_effect=lambda plugin_id, capability, config=None: capability in granted,
    )


def _runner_patch(runner):
    return patch("gateway.run._gateway_runner_ref", lambda: runner)


class _GatewayLoopThread:
    """A real, running asyncio loop on a non-test thread (stands in for the gateway loop)."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
    def __enter__(self):
        self.thread.start()
        return self.loop
    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=3)
        self.loop.close()


@pytest.fixture(autouse=True)
def _clean_throttle():
    with pa._STATUS_THROTTLE_LOCK:
        pa._STATUS_THROTTLE.clear()
    yield
    with pa._STATUS_THROTTLE_LOCK:
        pa._STATUS_THROTTLE.clear()


def _facade(plugin_id="demo"):
    return PlatformActions(plugin_id)


def _wait_until(predicate, timeout=3.0):
    """Event/poll based sync with a generous bound (flake policy: no quiet-runner assumptions)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# -- API / capability -----------------------------------------------------------

def test_capability_not_granted_structured_deny():
    runner = FakeRunner({Platform.TELEGRAM: _status_adapter()})
    with _grant(), _runner_patch(runner):
        res = _facade().emit_status("hi", "k1", platform="telegram", chat_id="42")
        assert res["accepted"] is False and res["error"] == "capability_not_granted"


def test_capability_granted_dispatches():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            res = _facade().emit_status("hi", "k1", platform="telegram", chat_id="42")
            assert res == {"accepted": True, "error": None}
            assert _wait_until(lambda: adapter.status_calls)


def test_audit_log_for_status_actions():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            res = _facade(plugin_id="audited").emit_status("hi", "k", platform="telegram", chat_id="42")
            assert res["accepted"] is True
    # _audit ran on the scheduling thread: the facade logs every action (#64176 rule)


def test_existing_verbs_gated_by_platform_actions_not_status():
    adapter = _status_adapter()
    adapter._set_reaction = AsyncMock(return_value=True)
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    f = _facade()
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):  # status granted, platform_actions NOT
        out = asyncio.run(f.add_reaction("telegram", "42", "7", "👍"))
        assert out["ok"] is False and out["error"] == "capability_not_granted"
    with _grant("gateway.platform_actions"), _runner_patch(runner):  # the inverse still works
        out = asyncio.run(f.add_reaction("telegram", "42", "7", "👍"))
        assert out["ok"] is True


# -- identity -------------------------------------------------------------------

def test_session_id_resolves_origin_profile_thread():
    adapter = _status_adapter()
    store = FakeStore({"sess-a": _entry(Platform.TELEGRAM, "42", thread_id="7", profile="worker")})
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store,
                        profile_adapters={(Platform.TELEGRAM, "worker"): adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            res = _facade().emit_status("hi", "k", session_id="sess-a")
            assert res["accepted"] is True
            assert _wait_until(lambda: adapter.status_calls)
            call = adapter.status_calls[0]
            assert call["chat_id"] == "42"
            assert (Platform.TELEGRAM, "worker") in runner.resolutions  # session-scoped profile honored


def test_unknown_session_id_fail_closed_no_contextvar_reroute():
    """The contamination case: an ambient ContextVar chat must NOT silently become the target."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=FakeStore({}))
    tokens = _bind_context(platform="telegram", chat_id="999")  # different chat is ambient
    try:
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            res = _facade().emit_status("hi", "k", session_id="ghost-session")
            assert res["accepted"] is False and res["error"] == "invalid_argument"
    finally:
        _clear_context(tokens)
    assert not adapter.status_calls and not adapter.send_calls


def test_contextvars_used_when_no_session_id():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    tokens = _bind_context(platform="telegram", chat_id="77")
    try:
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            with _GatewayLoopThread() as loop:
                runner._gateway_loop = loop
                res = _facade().emit_status("hi", "k")
                assert res["accepted"] is True
                assert _wait_until(lambda: adapter.status_calls)
                assert adapter.status_calls[0]["chat_id"] == "77"
    finally:
        _clear_context(tokens)


def test_identity_never_falls_back_to_os_environ():
    """os.environ HERMES_SESSION_* must not route a status (documented contamination path)."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    seen = {}
    def _worker():  # fresh thread → fresh context (no bound vars), while env holds a stale identity
        import os
        os.environ["HERMES_SESSION_PLATFORM"] = "telegram"
        os.environ["HERMES_SESSION_CHAT_ID"] = "42"
        try:
            with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
                seen["res"] = _facade().emit_status("hi", "k")
        finally:
            del os.environ["HERMES_SESSION_PLATFORM"]
            del os.environ["HERMES_SESSION_CHAT_ID"]
    t = threading.Thread(target=_worker)
    t.start(); t.join(timeout=5)
    assert seen["res"]["accepted"] is False and seen["res"]["error"] == "invalid_argument"
    assert not adapter.status_calls and not adapter.send_calls


def test_explicit_platform_chat_escape_hatch():
    plain = _plain_adapter()
    runner = FakeRunner({Platform.DISCORD: plain})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            res = _facade().emit_status("hi", "k", platform="discord", chat_id="ch1")
            assert res["accepted"] is True
            assert _wait_until(lambda: plain.send_calls)


def _bind_context(**kw):
    from gateway.session_context import set_session_vars
    return set_session_vars(**kw)


def _clear_context(tokens):
    from gateway.session_context import clear_session_vars
    clear_session_vars(tokens)


# -- concurrency ----------------------------------------------------------------

def test_sync_emit_from_gateway_loop_thread_no_deadlock():
    """emit_status called while executing ON the gateway loop must not self-deadlock."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            box = {}
            async def _on_loop():
                t0 = time.monotonic()
                box["res"] = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
                box["elapsed"] = time.monotonic() - t0
            fut = asyncio.run_coroutine_threadsafe(_on_loop(), loop)
            fut.result(timeout=3)  # test thread waiting on the OTHER loop: allowed
            assert box["res"]["accepted"] is True
            assert box["elapsed"] < 1.0
            assert _wait_until(lambda: adapter.status_calls)


def test_sync_emit_from_worker_thread_schedules():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    box = {}
    def _worker():  # plain thread, no loop at all (bounded-hook shape)
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            box["res"] = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
    with _GatewayLoopThread() as loop:
        runner._gateway_loop = loop
        t = threading.Thread(target=_worker)
        t.start(); t.join(timeout=5)
        assert box["res"] == {"accepted": True, "error": None}
        assert _wait_until(lambda: adapter.status_calls)


def test_async_primitive_same_loop_awaits():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            box = {}
            async def _on_loop():
                box["res"] = await _facade().send_status("hi", "k", platform="telegram", chat_id="42")
            asyncio.run_coroutine_threadsafe(_on_loop(), loop).result(timeout=3)
            assert box["res"]["ok"] is True and box["res"]["mode"] == "edit"
            assert box["res"]["message_id"] == "s1"


def test_async_primitive_cross_loop():
    """asyncio.run creates a foreign loop; delivery still lands on the gateway loop via wrap_future."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            out = asyncio.run(_facade().send_status("hi", "k", platform="telegram", chat_id="42"))
            assert out["ok"] is True and out["mode"] == "edit"
            assert adapter.status_calls and adapter.status_calls[0]["chat_id"] == "42"


def test_emit_status_never_waits_even_on_unstarted_loop():
    """A loop that accepts scheduling but never runs must not block the caller: proof there is
    no Future.result() anywhere on the sync path (also protects pre_tool_call fail-closed)."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    parked = asyncio.new_event_loop()  # NOT run_forever: future will never complete
    runner._gateway_loop = parked
    try:
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            t0 = time.monotonic()
            res = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
            elapsed = time.monotonic() - t0
        assert res["accepted"] is True
        assert elapsed < 0.5
    finally:
        parked.close()


def test_delivery_failure_does_not_raise_or_change_hook_result():
    """The post_tool_call contract: invoke_hook already isolates callbacks; our surface must not
    raise either — delivery failure lands as a structured result / swallowed exception only."""
    failing = _failing_adapter()
    runner = FakeRunner({Platform.TELEGRAM: failing})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            # sync: accepted (scheduling succeeded), delivery failure swallowed
            assert _facade().emit_status("hi", "k", platform="telegram", chat_id="42")["accepted"] is True
            # async: structured failure, no exception
            out = asyncio.run(_facade().send_status("hi2", "k2", platform="telegram", chat_id="42"))
            assert out["ok"] is False and out["error"] == "action_failed"


def test_closed_gateway_loop_structured_failure():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    closed = asyncio.new_event_loop(); closed.close()
    runner._gateway_loop = closed
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        res = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
        assert res["accepted"] is False and res["error"] == "gateway_unavailable"
        out = asyncio.run(_facade().send_status("hi", "k2", platform="telegram", chat_id="42"))
        assert out["ok"] is False and out["error"] == "gateway_unavailable"


# -- isolation --------------------------------------------------------------------

def test_two_chats_same_key_never_cross():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            f = _facade()
            assert f.emit_status("A text", "work", platform="telegram", chat_id="chatA")["accepted"]
            assert f.emit_status("B text", "work", platform="telegram", chat_id="chatB")["accepted"]
            assert _wait_until(lambda: len(adapter.status_calls) == 2)
            assert {c["chat_id"] for c in adapter.status_calls} == {"chatA", "chatB"}


def test_same_key_different_plugins_namespaced():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            assert _facade("pA").emit_status("t1", "work", platform="telegram", chat_id="42")["accepted"]
            assert _facade("pB").emit_status("t1", "work", platform="telegram", chat_id="42")["accepted"]
            # identical text+key from two plugins must NOT throttle each other (different namespace)
            assert _wait_until(lambda: len(adapter.status_calls) == 2)
            keys = sorted(c["status_key"] for c in adapter.status_calls)
            assert keys == ["plugin:pA:work", "plugin:pB:work"]


# -- presentation -----------------------------------------------------------------

def test_native_adapter_edit_path():
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        out = asyncio.run(_facade().send_status("hi", "core-ish", platform="telegram", chat_id="42"))
        assert out["ok"] and out["mode"] == "edit"
        # core status lines key by raw event_type; the plugin's identical key must be namespaced
        assert adapter.status_calls[0]["status_key"] == "plugin:demo:core-ish"
        # and the delivered content never reaches SessionDB/turn machinery (no turn objects exist here)


def test_plain_send_fallback_for_unsupported_adapter():
    plain = _plain_adapter()
    runner = FakeRunner({Platform.DISCORD: plain})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        out = asyncio.run(_facade().send_status("hi", "k", platform="discord", chat_id="ch"))
        assert out["ok"] and out["mode"] == "send"
        assert plain.send_calls and plain.send_calls[0]["content"] == "hi"


def test_unchanged_text_suppressed_changed_resent():
    adapter = _status_adapter()  # native send_or_update_status path
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            f = _facade()
            r1 = f.emit_status("same", "k", platform="telegram", chat_id="42")
            assert r1["accepted"]
            # Wait for the FIRST delivery to settle (throttle is recorded inside the delivery
            # coroutine) — asserting the throttle before it lands would race the real loop thread.
            assert _wait_until(lambda: adapter.status_calls and
                               any(c["content"] == "same" for c in adapter.status_calls))
            r2 = f.emit_status("same", "k", platform="telegram", chat_id="42")
            assert r2["accepted"]  # acceptance semantics; delivery throttled before scheduling
            assert len(adapter.status_calls) == 1  # skip is decided synchronously (reservation)
            r3 = f.emit_status("changed", "k", platform="telegram", chat_id="42")
            assert r3["accepted"] and _wait_until(lambda: len(adapter.status_calls) == 2)


def test_throttle_map_bounded(monkeypatch):
    plain = _plain_adapter()
    runner = FakeRunner({Platform.DISCORD: plain})  # loop unpublished → async runs directly
    monkeypatch.setattr(pa, "_MAX_STATUS_THROTTLE", 6)
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        f = _facade()
        for i in range(20):
            out = asyncio.run(f.send_status(f"text {i}", f"k{i}", platform="discord", chat_id="ch"))
            assert out["ok"]
    with pa._STATUS_THROTTLE_LOCK:
        assert len(pa._STATUS_THROTTLE) <= 6


def test_invalid_text_and_key_structured():
    runner = FakeRunner({Platform.TELEGRAM: _status_adapter()})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        f = _facade()
        assert f.emit_status("  ", "k", platform="telegram", chat_id="42")["error"] == "invalid_argument"
        assert f.emit_status("ok", "", platform="telegram", chat_id="42")["error"] == "invalid_argument"


def test_disconnected_adapter_structured_failure():
    dead = _status_adapter(connected=False)
    runner = FakeRunner({Platform.TELEGRAM: dead})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        res = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
        assert res["accepted"] is False and res["error"] == "adapter_disconnected"


# -- migration-shape compile check (step 13) --------------------------------------

def test_post_tool_call_migration_shape():
    """A status plugin needs only the payload session_id + ctx.emit_status — no runner, no
    session_store capture, no adapter lookup, no loop hop, no send_or_update_status."""
    adapter = _status_adapter()
    store = FakeStore({"s1": _entry(Platform.TELEGRAM, "42")})
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    facade = PlatformActions("collab-like")
    ctx = SimpleNamespace(emit_status=facade.emit_status)  # PluginContext-shaped stand-in

    def post_tool_call(**payload):  # the whole plugin callback, in the target shape
        ctx.emit_status(
            "🤝 done",
            key=f"collab:{payload['tool_name']}",
            session_id=payload["session_id"],
        )
        return None

    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            post_tool_call(session_id="s1", tool_name="delegate_task")
            assert _wait_until(lambda: adapter.status_calls)
            assert adapter.status_calls[0]["status_key"] == "plugin:collab-like:collab:delegate_task"
            assert adapter.status_calls[0]["chat_id"] == "42"


# -- F-2 revisions: audit + denial zero-side-effect --------------------------------

def test_audit_entry_logged_with_structured_fields(caplog):
    import logging

    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            with caplog.at_level(logging.INFO, logger="hermes_cli.platform_actions"):
                assert _facade(plugin_id="audited").emit_status(
                    "hi", "k", platform="telegram", chat_id="42")["accepted"]
            lines = [r.getMessage() for r in caplog.records if "platform_action" in r.getMessage()]
            assert any("plugin=audited" in m and "verb=emit_status" in m
                       and "platform=telegram" in m and "ok=True" in m for m in lines)
            assert _wait_until(lambda: adapter.status_calls)


def test_capability_denied_zero_side_effects():
    """Deny must not resolve identity, reserve throttle state, touch the adapter, or schedule."""
    adapter = _status_adapter()
    store = FakeStore({"s1": _entry(Platform.TELEGRAM, "42")})
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    with _grant(), _runner_patch(runner):  # nothing granted
        res = _facade().emit_status("hi", "k", session_id="s1")
        assert res["accepted"] is False and res["error"] == "capability_not_granted"
    assert not adapter.status_calls and not adapter.send_calls
    with pa._STATUS_THROTTLE_LOCK:
        assert dict(pa._STATUS_THROTTLE) == {}  # denied BEFORE reserve; nothing scheduled


# -- F-1: thread/topic identity reaches the actual delivery ------------------------

def test_telegram_topics_independent_and_metadata_carries_thread():
    adapter = _status_adapter()
    store = FakeStore({
        "sA": _entry(Platform.TELEGRAM, "group1", thread_id="topicA"),
        "sB": _entry(Platform.TELEGRAM, "group1", thread_id="topicB"),
    })
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            f = _facade()
            assert f.emit_status("same text", "k", session_id="sA")["accepted"]
            # identical text+key through the OTHER topic must NOT be throttled (F-1 regression):
            assert f.emit_status("same text", "k", session_id="sB")["accepted"]
            assert _wait_until(lambda: len(adapter.status_calls) == 2)
            threads = sorted(c["metadata"]["thread_id"] for c in adapter.status_calls)
            assert threads == ["topicA", "topicB"]


def test_caller_metadata_thread_wins_over_resolved():
    adapter = _status_adapter()
    store = FakeStore({"s1": _entry(Platform.TELEGRAM, "g", thread_id="origin-thread")})
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            assert _facade().emit_status(
                "hi", "k", session_id="s1", metadata={"thread_id": "caller-thread"})["accepted"]
            assert _wait_until(lambda: adapter.status_calls)
            assert adapter.status_calls[0]["metadata"]["thread_id"] == "caller-thread"


def test_plain_dm_has_no_thread_metadata_injected():
    adapter = _status_adapter()
    store = FakeStore({"s1": _entry(Platform.TELEGRAM, "42")})  # thread_id None
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            assert _facade().emit_status("hi", "k", session_id="s1")["accepted"]
            assert _wait_until(lambda: adapter.status_calls)
            assert adapter.status_calls[0]["metadata"] is None


def test_slack_threads_independent_plain_send_fallback():
    """Slack-shaped adapter without native status support: threads stay independent and the
    thread identity still reaches the plain-send metadata (slack _resolve_thread_ts reads it)."""
    plain = _plain_adapter()
    plain.platform = Platform.SLACK
    store = FakeStore({
        "sA": _entry(Platform.SLACK, "chan", thread_id="tsA"),
        "sB": _entry(Platform.SLACK, "chan", thread_id="tsB"),
    })
    runner = FakeRunner({Platform.SLACK: plain}, store=store)
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            f = _facade()
            assert f.emit_status("same", "k", session_id="sA")["accepted"]
            assert f.emit_status("same", "k", session_id="sB")["accepted"]  # not throttled
            assert _wait_until(lambda: len(plain.send_calls) == 2)


# -- F-A: scheduling failure never leaks the coroutine -----------------------------

def test_emit_status_create_task_failure_closes_coroutine():
    """Shutdown race: loop passed the is_closed check but create_task still raises — the
    coroutine must be closed (no 'never awaited' RuntimeWarning) and failure is structured."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _GatewayLoopThread() as loop:
        runner._gateway_loop = loop

        async def _emit_on_loop():
            import gc
            import warnings
            original = loop.create_task

            def _boom(coro):
                raise RuntimeError("Event loop is closed")  # deliberately do NOT close

            # Patch only for the emit itself — the injection of THIS coroutine must have used
            # the original create_task (patching earlier would break the test harness itself).
            loop.create_task = _boom
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    res = _facade().emit_status("hi", "k", platform="telegram", chat_id="42")
                    gc.collect()
            finally:
                loop.create_task = original
            return res, [str(w.message) for w in caught]

        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            # Call from INSIDE the loop thread so the same-loop create_task branch is exercised.
            res, warns = asyncio.run_coroutine_threadsafe(_emit_on_loop(), loop).result(timeout=3)
        assert res["accepted"] is False and res["error"] == "gateway_unavailable"
        assert not any("never awaited" in w for w in warns)
        assert not adapter.status_calls and not adapter.send_calls
        with pa._STATUS_THROTTLE_LOCK:
            assert dict(pa._STATUS_THROTTLE) == {}  # reservation released


# -- F-B: async contract — timeout, cancellation, structured failures --------------

def test_send_status_cross_loop_timeout(monkeypatch):
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    parked = asyncio.new_event_loop()  # scheduled callbacks will never run
    runner._gateway_loop = parked
    monkeypatch.setattr(pa, "_STATUS_DELIVERY_TIMEOUT_SECS", 0.2)
    try:
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            out = asyncio.run(_facade().send_status("hi", "k", platform="telegram", chat_id="42"))
        assert out["ok"] is False and out["error"] == "status_timeout"
        assert not adapter.status_calls
        with pa._STATUS_THROTTLE_LOCK:
            assert dict(pa._STATUS_THROTTLE) == {}  # timeout released the reservation
    finally:
        parked.close()


def test_send_status_cancellation_propagates_and_releases():
    """Caller cancels while delivery is running/queued on the gateway loop: CancelledError
    propagates (not swallowed as failure) and the throttle reservation is released either way."""
    adapter = _status_adapter()

    async def _slow(chat_id, status_key, content, metadata=None):
        await asyncio.sleep(60)  # cancellable, keeps the gateway loop responsive

    adapter.send_or_update_status = _slow
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop

            async def _caller():
                task = asyncio.create_task(
                    _facade().send_status("hi", "k", platform="telegram", chat_id="42"))
                await asyncio.sleep(0.05)  # let it reach the gateway loop (either stage)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            asyncio.run(_caller())
            with pa._STATUS_THROTTLE_LOCK:
                assert dict(pa._STATUS_THROTTLE) == {}


def test_send_status_structured_on_internal_failure():
    """A programmer error inside resolution surfaces structured, audited, never raises."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with patch("hermes_cli.platform_actions.PlatformActions._resolve_status_target",
                   side_effect=TypeError("boom in resolution")):
            out = asyncio.run(_facade().send_status("hi", "k", platform="telegram", chat_id="42"))
        assert out["ok"] is False and out["error"] == "action_failed"
        assert not adapter.status_calls


# -- concurrency: genuinely simultaneous A/B ----------------------------------------

def test_concurrent_cross_chat_emits_are_independent():
    """Two real threads racing on one facade, same key+text, different chats: both delivered.
    A throttle key missing the chat dimension would fail this (second thread skips)."""
    adapter = _status_adapter()
    runner = FakeRunner({Platform.TELEGRAM: adapter})
    barrier = threading.Barrier(2)
    results = {}

    def _worker(chat):
        with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
            barrier.wait(timeout=5)
            results[chat] = _facade().emit_status("same text", "work",
                                                  platform="telegram", chat_id=chat)

    with _GatewayLoopThread() as loop:
        runner._gateway_loop = loop
        threads = [threading.Thread(target=_worker, args=(f"chat{c}",)) for c in "AB"]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert all(r["accepted"] for r in results.values())
        assert _wait_until(lambda: len(adapter.status_calls) == 2)
        assert {c["chat_id"] for c in adapter.status_calls} == {"chatA", "chatB"}


# -- throttle hygiene: values are digests, not retained text ------------------------

def test_throttle_stores_digest_not_text():
    plain = _plain_adapter()
    runner = FakeRunner({Platform.DISCORD: plain})
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        out = asyncio.run(_facade().send_status(
            "super-sensitive status text " * 40, "k", platform="discord", chat_id="ch"))
        assert out["ok"]
        with pa._STATUS_THROTTLE_LOCK:
            values = list(pa._STATUS_THROTTLE.values())
        assert values and all(len(v) == 64 for v in values)  # sha256 hex, no full-text retention
        assert not any("sensitive" in v for v in values)


# -- PluginContext delegation (no SimpleNamespace bypass) ----------------------------

def test_plugin_context_emit_status_uses_facade_path_and_gate():
    from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager

    adapter = _status_adapter()
    store = FakeStore({"s1": _entry(Platform.TELEGRAM, "chat-ctx")})
    runner = FakeRunner({Platform.TELEGRAM: adapter}, store=store)
    manifest = PluginManifest(name="plugctx", source="user", key="plugctx")
    ctx = PluginContext(manifest, PluginManager())
    with _grant(CAPABILITY_STATUS_ID), _runner_patch(runner):
        with _GatewayLoopThread() as loop:
            runner._gateway_loop = loop
            res = ctx.emit_status("hi", "k", session_id="s1")
            assert res["accepted"] is True
            assert _wait_until(lambda: adapter.status_calls)
            # namespaced with the CONTEXT's plugin identity, not caller text:
            assert adapter.status_calls[0]["status_key"].startswith("plugin:plugctx:")
    # the sync convenience has NO bypass: ungranted capability denies through the same gate
    with _grant(), _runner_patch(runner):
        denied = ctx.emit_status("hi", "k2", session_id="s1")
        assert denied["accepted"] is False and denied["error"] == "capability_not_granted"
