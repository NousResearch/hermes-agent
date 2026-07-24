"""Tests for the AG-UI resume shim's core-safety invariants.

The shim monkeypatches Hermes core (`conversation_loop.build_turn_context`).
The load-bearing promise is that with the resume flag OFF (every non-AG-UI
code path) the wrapper is a pure pass-through, so core behavior is unchanged.
"""
from types import SimpleNamespace

from agui_adapter import resume_shim


def _fresh_ctx():
    # A turn context whose messages end with a trailing user turn right after a
    # tool message — the exact shape the shim rewrites on resume.
    return SimpleNamespace(
        messages=[{"role": "user"}, {"role": "tool"}, {"role": "user"}],
        current_turn_user_idx=None,
    )


def test_resume_off_is_pure_passthrough_and_install_idempotent(monkeypatch):
    import agent.conversation_loop as cl

    def _orig(agent, *a, **k):
        return _fresh_ctx()

    # Force a clean wrap around our stub (the process may already have installed
    # the shim at import time; monkeypatch restores everything afterwards).
    monkeypatch.setattr(cl, "build_turn_context", _orig)
    monkeypatch.setattr(resume_shim, "_installed", False)

    resume_shim.install()
    wrapped = cl.build_turn_context
    assert wrapped is not _orig  # it wrapped the original

    # Flag OFF (default): messages are returned untouched — core behavior for
    # every non-AG-UI caller is unchanged.
    res_off = wrapped(SimpleNamespace())
    assert [m["role"] for m in res_off.messages] == ["user", "tool", "user"]

    # install() again is a no-op (the _installed guard); no double-wrap.
    resume_shim.install()
    assert cl.build_turn_context is wrapped


def test_resume_on_drops_trailing_user_after_tool(monkeypatch):
    import agent.conversation_loop as cl

    monkeypatch.setattr(cl, "build_turn_context", lambda agent, *a, **k: _fresh_ctx())
    monkeypatch.setattr(resume_shim, "_installed", False)
    resume_shim.install()
    wrapped = cl.build_turn_context

    agent = SimpleNamespace()
    token = resume_shim.set_resume(True)
    try:
        res_on = wrapped(agent)
    finally:
        resume_shim.reset_resume(token)

    # The trailing user turn immediately following the tool message is popped,
    # and current_turn_user_idx is recomputed to the last remaining user turn.
    assert [m["role"] for m in res_on.messages] == ["user", "tool"]
    assert res_on.current_turn_user_idx == 0


def test_resume_on_non_matching_tail_is_untouched(monkeypatch):
    # Flag ON but the tail is NOT user-after-tool: the shim must leave the
    # context untouched (it only rewrites the exact user-after-tool shape), so
    # it can never corrupt a context it shouldn't touch.
    import agent.conversation_loop as cl

    def _orig(agent, *a, **k):
        return SimpleNamespace(
            messages=[{"role": "user"}, {"role": "assistant"}, {"role": "user"}],
            current_turn_user_idx=99,
        )

    monkeypatch.setattr(cl, "build_turn_context", _orig)
    monkeypatch.setattr(resume_shim, "_installed", False)
    resume_shim.install()
    wrapped = cl.build_turn_context

    token = resume_shim.set_resume(True)
    try:
        res = wrapped(SimpleNamespace())
    finally:
        resume_shim.reset_resume(token)

    # Tail is user-after-assistant (not user-after-tool) → no pop, idx untouched.
    assert [m["role"] for m in res.messages] == ["user", "assistant", "user"]
    assert res.current_turn_user_idx == 99
