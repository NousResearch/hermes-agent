"""Coverage for the ``mutates=True`` hook metadata (PR-102481 follow-up).

Plugin behavior contracts:

  - ``register_hook(name, cb)`` without ``mutates=`` is observational by
    default. Streaming surfaces must not pay the suppress cost.
  - ``register_hook(name, cb, mutates=True)`` declares that the callback's
    return value REPLACES the streamed payload — a surface that streams
    deltas must buffer instead of repaint.
  - Unloading the only mutating callback restores streaming (bookkeeping
    must leave no stale mutator entry behind).

These tests target the plugin manager surface directly. For the actual
gating decision consumed by the CLI see
``tests/cli/test_transform_stream_buffered_102203.py``.
"""

from __future__ import annotations

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _noop(**_kw):
    return None


def _ctx(mgr: PluginManager, name: str = "test-plugin") -> PluginContext:
    manifest = PluginManifest(name=name, source="user")
    return PluginContext(manifest, mgr)


def test_register_hook_default_is_observational():
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", _noop)

    assert mgr.has_hook("transform_llm_output") is True
    assert mgr.has_mutating_hook("transform_llm_output") is False


def test_register_hook_with_mutates_true_marks_mutating():
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", _noop, mutates=True)

    assert mgr.has_hook("transform_llm_output") is True
    assert mgr.has_mutating_hook("transform_llm_output") is True


def test_multiple_hooks_some_observational_still_mutating():
    """One mutator in the pool is enough — the surface must suppress."""
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", lambda **kw: None)  # observer
    ctx.register_hook("transform_llm_output", _noop, mutates=True)

    assert mgr.has_mutating_hook("transform_llm_output") is True


def test_unloading_mutating_hook_clears_mutating_index():
    mgr = PluginManager()
    ctx = _ctx(mgr)
    handle = ctx.register_hook("transform_llm_output", _noop, mutates=True)

    assert mgr.has_mutating_hook("transform_llm_output") is True

    handle.release()

    assert mgr.has_hook("transform_llm_output") is False
    assert mgr.has_mutating_hook("transform_llm_output") is False


def test_unloading_one_of_two_mutating_hooks_keeps_flag():
    mgr = PluginManager()
    ctx = _ctx(mgr)
    handle_a = ctx.register_hook("transform_llm_output", _noop, mutates=True)
    ctx.register_hook("transform_llm_output", _noop, mutates=True)

    handle_a.release()

    assert mgr.has_mutating_hook("transform_llm_output") is True
