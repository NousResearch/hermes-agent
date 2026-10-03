"""#123188: hook-suppression keys must not collide across callback lifetimes.

id(cb) is recycled by CPython after a callback is collected, and mid-list
removals shift slots: either way a healthy callback could inherit a dead
one's back-off window and abandoned-worker budget (a fail-closed skip for
pre_tool_call). Registration tokens bind the key to the callback's lease.
Direct ``_hooks[...]`` writes (tests, legacy callers) carry no token and
fall back to id()-keying.
"""

import threading

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _named(name, body=lambda **kw: None):
    def cb(**kw):
        return body(**kw)
    cb.__name__ = name
    return cb


@pytest.fixture()
def mgr():
    return PluginManager()


def _ctx(mgr):
    return PluginContext(PluginManifest(name="test-plugin", source="user"), mgr)


def test_register_hook_mints_distinct_tokens_per_callback(mgr):
    ctx = _ctx(mgr)
    cb_a, cb_b = lambda **kw: None, lambda **kw: None

    ctx.register_hook("post_tool_call", cb_a)
    ctx.register_hook("post_tool_call", cb_b)

    tokens = [mgr._hook_registration_tokens.get(id(cb)) for cb in (cb_a, cb_b)]
    assert None not in tokens
    assert tokens[0] != tokens[1]


def test_removed_callback_loses_its_token(mgr):
    ctx = _ctx(mgr)
    cb = lambda **kw: None  # noqa: E731
    handle = ctx.register_hook("post_tool_call", cb)

    assert mgr._hook_registration_tokens.get(id(cb)) is not None
    handle.dispose()

    assert cb not in mgr._hooks.get("post_tool_call", [])
    assert id(cb) not in mgr._hook_registration_tokens


def test_fresh_same_named_callback_never_inherits_dead_key(mgr):
    """A('guard') unloads, B('guard') registers later. Even when CPython
    hands B the recycled address, the suppression key must differ."""
    ctx = _ctx(mgr)
    guard_a = _named("guard")
    handle = ctx.register_hook("pre_tool_call", guard_a)
    key_a = ("pre_tool_call", mgr._hook_registration_tokens[id(guard_a)])
    handle.dispose()

    guard_b = _named("guard")
    ctx.register_hook("pre_tool_call", guard_b)
    key_b = ("pre_tool_call", mgr._hook_registration_tokens[id(guard_b)])

    assert key_a != key_b
    assert mgr._hooks["pre_tool_call"].count(guard_b) == 1


def test_slot_shift_after_mid_list_removal_keeps_keys_distinct(mgr):
    """[A(guard), B(guard), C(guard)]: A unloads, B shifts into slot 0.
    B's key must not equal A's key (slot-based keying would collide)."""
    ctx = _ctx(mgr)
    a, b, c = _named("guard"), _named("guard"), _named("guard")
    handle_a = ctx.register_hook("pre_tool_call", a)
    ctx.register_hook("pre_tool_call", b)
    ctx.register_hook("pre_tool_call", c)

    keys = [("pre_tool_call", mgr._hook_registration_tokens[id(cb)]) for cb in (a, b, c)]
    assert len(set(keys)) == 3

    handle_a.dispose()  # b and c shift down
    assert mgr._hooks["pre_tool_call"] == [b, c]
    assert keys[1] != keys[0]  # token keying: the shift cannot collide


def test_tokenless_callback_falls_back_to_id_key_and_runs(mgr):
    """Direct _hooks writes carry no token: dispatch uses the legacy id(cb)
    key instead of crashing."""
    survivor = lambda **kw: {"ok": True}  # noqa: E731
    mgr._hooks["post_tool_call"] = [survivor]

    results = mgr.invoke_hook("post_tool_call", tool_name="terminal", args={}, result="{}")
    assert results == [{"ok": True}]


def test_survivor_still_runs_after_sibling_removal(mgr):
    """The negative case from review: two same-named callbacks, one removed,
    the survivor must still run (no inherited suppression)."""
    ctx = _ctx(mgr)
    guard_a = _named("guard", body=lambda **kw: {"who": "a"})
    guard_b = _named("guard", body=lambda **kw: {"who": "b"})

    handle_a = ctx.register_hook("pre_tool_call", guard_a)
    ctx.register_hook("pre_tool_call", guard_b)
    handle_a.dispose()

    results = mgr.invoke_hook("pre_tool_call", tool_name="terminal", args={})
    assert results == [{"who": "b"}]


def test_suppression_window_sticks_to_the_same_callback(mgr, monkeypatch):
    """Suppression is per-callback: after a timeout, the same callback's
    token key must be in the suppression map (the window is not lost)."""
    import time as time_mod

    import hermes_cli.plugins as plugins_mod

    monkeypatch.setattr(plugins_mod, "_resolve_hook_callback_timeout", lambda: 0.05)

    hold = threading.Event()
    started = threading.Event()

    def blocker(**_kwargs):
        started.set()
        hold.wait(timeout=10.0)
        return "late"

    ctx = _ctx(mgr)
    ctx.register_hook("post_tool_call", blocker)

    t0 = time_mod.monotonic()
    results = mgr.invoke_hook("post_tool_call", tool_name="terminal", args={}, result="{}")
    try:
        assert started.wait(timeout=1.0)
        assert results == []
        assert time_mod.monotonic() - t0 < 5.0

        key = ("post_tool_call", mgr._hook_registration_tokens[id(blocker)])
        assert key in mgr._hook_timeout_suppressed_until
    finally:
        hold.set()
