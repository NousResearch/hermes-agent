"""Mutating-hook metadata and the hook-registration surface of ``PluginContext``.

A plugin hook can REPLACE the payload a surface already showed (a ``transform_*``
hook returning a replacement string) or only observe it. A surface that streamed
that payload must suppress it when a mutator is live, or the bytes on screen and
the final text disagree (#102203).

The declaration lives on ``register_hook(..., mutates=)``; the query lives on
``PluginManager.has_mutating_hook``. Both live here so the rule that decides a
hook class's default has one home, and so ``hermes_cli/plugins.py`` — already
over its line cap — does not grow for them.
"""

from __future__ import annotations

from typing import Any

# Hook classes whose callback return value REPLACES a payload the host already
# produced, so a surface that streamed that payload must suppress it (#102203).
# A plugin that registers one of these without saying so would garble the
# terminal, so these classes default to ``mutates=True`` and the plugin must
# opt out explicitly with ``mutates=False``.
#
# Only the classes whose replacement is visible in a STREAMED surface belong
# here: ``transform_tool_result`` and ``transform_terminal_output`` also
# replace a payload, but they do so before the host renders it, so a
# mutating flag on them has no consumer to suppress.
MUTATING_HOOKS: set[str] = {"transform_llm_output"}


def hook_mutates_by_default(hook_name: str) -> bool:
    """True when ``hook_name`` is a payload-replacing hook class."""
    return hook_name in MUTATING_HOOKS


def has_mutating_hook(hook_name: str) -> bool:
    """True when a loaded plugin handles a hook and marked it ``mutates=True`` at registration.

    Lazy-discovers first — same gate-before-invoke rationale as ``has_hook`` (tracking #64178).
    """
    from hermes_cli.plugins import _delivery_manager

    return _delivery_manager().has_mutating_hook(hook_name)


class PluginHookMixin:
    """The hook-registration surface of ``PluginContext``.

    Mixed into ``PluginContext`` so the registration surface and the rule that
    decides a hook class's default stay together, away from the plugins facade.
    """

    def register_hook(self, hook_name: str, callback, *, mutates: bool | None = None):
        """Register a lifecycle hook callback (unknown names warn but are still stored).

        ``mutates`` declares whether the callback's return value REPLACES the
        streamed payload; it defaults to the hook class's own contract (see
        :func:`hook_mutates_by_default`).
        """
        from hermes_cli.plugins import VALID_HOOKS

        handle = self._track_callback("hook", hook_name, callback, self._manager._hooks, VALID_HOOKS)
        # Parallel to ``_hooks`` and index-aligned with it: same index = one registration
        # (an id-keyed set would collapse two registrations of the same callable).
        flags = self._manager._hooks_mutating_flags.setdefault(hook_name, [])
        flags.append(hook_mutates_by_default(hook_name) if mutates is None else bool(mutates))
        return handle


def invoke_hook(hook_name: str, **kwargs: Any) -> list[Any]:
    """Invoke a lifecycle hook (lazy-discovers first); return non-``None`` callback results.

    Hot-path / observer hooks in ``_HOOK_TIMEOUT_BOUNDED_HOOKS`` and the policy hook ``pre_tool_call`` are
    bounded by ``plugins.hook_callback_timeout`` (default 30s). On timeout the worker is abandoned (not
    joined) so we do not reintroduce the #6622 hang. Timed-out or still-running ``pre_tool_call`` callbacks
    fail closed with a block directive; other bounded hooks fail open (skip).
    Ensures plugins are discovered on first invocation so callers in processes that never explicitly call
    ``discover_plugins()`` (gateway platform events, TUI slash workers, query mode, cron) still fire
    callbacks registered by user plugins (tracking #64178).
    """
    from hermes_cli.plugins import _delivery_manager

    return _delivery_manager().invoke_hook(hook_name, **kwargs)


async def ainvoke_hook(hook_name: str, **kwargs: Any) -> list[Any]:
    """``invoke_hook`` for callers on an event loop: ``async def`` callbacks are awaited
    there instead of bridged through a helper thread (see ``PluginManager.ainvoke_hook``)."""
    from hermes_cli.plugins import _delivery_manager

    return await _delivery_manager().ainvoke_hook(hook_name, **kwargs)
