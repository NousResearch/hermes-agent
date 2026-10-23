"""Coverage for the ``mutates=`` hook metadata (PR-102481 follow-up).

Plugin behavior contracts:

  - A hook class whose return value REPLACES the payload the caller already
    showed (``transform_llm_output``) is mutating BY DEFAULT. A plugin that
    mutates it but forgets the flag would otherwise garble the terminal
    (#102203), so the safe direction is the default and the plugin opts out
    with an explicit ``mutates=False``.
  - An observer hook class is observational by default: streaming surfaces
    must not pay the suppress cost for a hook that only reads.
  - ``register_hook(name, cb, mutates=True)`` declares that the callback's
    return value REPLACES the streamed payload — a surface that streams
    deltas must buffer instead of repaint.
  - Unloading the only mutating callback restores streaming (bookkeeping
    must leave no stale mutator entry behind).

These tests target the plugin manager surface directly. For the actual
gating decision consumed by the CLI see
``tests/hermes_cli/test_transform_stream_buffered_102203.py``.
"""

from __future__ import annotations

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _noop(**_kw):
    return None


def _ctx(mgr: PluginManager, name: str = "test-plugin") -> PluginContext:
    manifest = PluginManifest(name=name, source="user")
    return PluginContext(manifest, mgr)


def test_transform_llm_output_defaults_to_mutating():
    """A payload-replacing hook class is mutating without an explicit flag.

    Fail-safe: a plugin that mutates and forgets the flag must still be
    detected, or the streamed bytes and the final text disagree (#102203).
    """
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", _noop)

    assert mgr.has_hook("transform_llm_output") is True
    assert mgr.has_mutating_hook("transform_llm_output") is True


def test_observer_hook_class_defaults_to_observational():
    """An observer hook class does not pay the suppress cost."""
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("post_tool_call", _noop)

    assert mgr.has_hook("post_tool_call") is True
    assert mgr.has_mutating_hook("post_tool_call") is False


def test_explicit_opt_out_marks_transform_observational():
    """A plugin that only observes the response opts out explicitly."""
    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", _noop, mutates=False)

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
    ctx.register_hook("transform_llm_output", lambda **kw: None, mutates=False)
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


def test_memory_provider_proxy_forwards_mutates():
    """The memory-provider ctx proxy must not swallow the flag.

    It is a different object than PluginContext, so it needs the kwarg
    explicitly or a memory plugin declaring ``mutates=True`` raises
    TypeError at registration.
    """
    from plugins.memory import _ProviderCollector

    assert "mutates" in _ProviderCollector.register_hook.__code__.co_varnames


def test_plugin_validate_stub_accepts_mutates():
    """The validation probe runs a plugin's register() against a stub.

    A stub without the kwarg raises TypeError for a plugin that declares
    ``mutates=True``, failing validation for a plugin that is valid.
    """
    from hermes_cli.plugin_validate import _PROBE_SCRIPT

    assert "def register_hook(self, hook_name, callback, *, mutates=None)" in _PROBE_SCRIPT


def test_safe_worker_does_not_report_a_mutating_hook():
    """A safe worker never runs plugin hooks, so it never has a mutator.

    ``invoke_hook`` returns empty there and ``has_hook`` says False; a
    mutating hook that reports True would silence the stream for a
    replacement that can never happen.
    """
    from agent.safe_worker_policy import _bind_safe_worker_policy

    mgr = PluginManager()
    ctx = _ctx(mgr)
    ctx.register_hook("transform_llm_output", _noop, mutates=True)
    assert mgr.has_mutating_hook("transform_llm_output") is True

    import subprocess
    import sys
    import textwrap

    probe = textwrap.dedent(
        """
        from agent.safe_worker_policy import _bind_safe_worker_policy
        _bind_safe_worker_policy(safe_mode=True, ignore_user_config=True, config={})
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

        mgr = PluginManager()
        ctx = PluginContext(PluginManifest(name="p", source="user"), mgr)
        ctx.register_hook("transform_llm_output", lambda **kw: None, mutates=True)
        print("MUTATING", mgr.has_mutating_hook("transform_llm_output"))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    ).stdout
    assert "MUTATING False" in out
