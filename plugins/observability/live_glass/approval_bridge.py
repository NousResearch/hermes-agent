"""Approval bridge — wraps computer_use approval callback to fire PluginManager hooks.

This module bridges the gap documented in the AVA-14 findings note:
``computer_use`` has its own approval callback path and does not currently
emit the shared ``tools/approval.py`` approval hooks.  This bridge wraps
the existing callback so that ``pre_approval_request`` and
``post_approval_response`` PluginManager hooks fire on every computer_use
approval decision WITHOUT changing who makes the decision.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_WRAP_MARKER = "__live_glass_wrapped__"

ApprovalCallback = Callable[..., str]


def wrap_approval_callback(
    callback: Optional[ApprovalCallback],
) -> Optional[ApprovalCallback]:
    """Wrap *callback* so PluginManager approval hooks fire on every call.

    Returns *callback* unchanged if it is ``None`` or already wrapped
    (idempotent).  The wrapper:

    1. Calls ``invoke_hook("pre_approval_request", ...)``
    2. Calls the original *callback*
    3. Calls ``invoke_hook("post_approval_response", ..., choice=…)``
    4. Returns the original verdict unchanged
    """
    if callback is None:
        return None
    if getattr(callback, _WRAP_MARKER, False):
        return callback

    original = callback

    def _wrapped(action: str, args: dict, summary: str) -> str:
        # ── pre_approval_request ────────────────────────────────────
        _safe_invoke_hook(
            "pre_approval_request",
            command=summary,
            description=f"computer_use {action}",
            pattern_key=action,
            pattern_keys=[action],
            surface="cli",
        )

        # ── call original ───────────────────────────────────────────
        try:
            verdict = original(action, args, summary)
        except Exception:
            logger.debug("computer_use approval callback failed", exc_info=True)
            verdict = "deny"

        # ── post_approval_response ──────────────────────────────────
        _safe_invoke_hook(
            "post_approval_response",
            command=summary,
            description=f"computer_use {action}",
            pattern_key=action,
            pattern_keys=[action],
            surface="cli",
            choice=verdict,
        )

        return verdict

    setattr(_wrapped, _WRAP_MARKER, True)
    return _wrapped


def install_bridge(
    *,
    callback_getter: Callable[[], Optional[ApprovalCallback]],
    callback_setter: Callable[[ApprovalCallback], None],
) -> None:
    """Read the current computer_use approval callback, wrap it, and re-install.

    Intended to be called once during plugin registration.  Idempotent:
    if the callback is already wrapped this is a no-op.
    """
    current = callback_getter()
    if current is None:
        logger.debug("live-glass approval bridge: no callback set, skipping")
        return
    if getattr(current, _WRAP_MARKER, False):
        logger.debug("live-glass approval bridge: callback already wrapped")
        return
    wrapped = wrap_approval_callback(current)
    if wrapped is not current and wrapped is not None:
        callback_setter(wrapped)
        logger.debug("live-glass approval bridge: installed")


def _safe_invoke_hook(name: str, **kwargs: Any) -> None:
    """Invoke a PluginManager hook, swallowing all exceptions."""
    try:
        from hermes_cli.plugins import invoke_hook
        invoke_hook(name, **kwargs)
    except Exception:
        logger.debug("live-glass approval bridge: hook %s failed", name, exc_info=True)


def on_pre_approval_request(**kwargs: Any) -> None:
    """Hook callback registered by the live-glass plugin.

    Kept intentionally thin — the real observability is in the live-glass
    event bus subscriber. This function exists so the plugin can satisfy
    the PluginManager registration contract.
    """


def on_post_approval_response(**kwargs: Any) -> None:
    """Hook callback registered by the live-glass plugin.

    Kept intentionally thin — see ``on_pre_approval_request``.
    """


def register_approval_bridge(ctx) -> None:
    """Register approval hooks and install the computer_use callback wrapper."""
    # Register the hook callbacks so PluginManager knows about them.
    ctx.register_hook("pre_approval_request", on_pre_approval_request)
    ctx.register_hook("post_approval_response", on_post_approval_response)

    # Wrap the computer_use approval callback.
    try:
        import tools.computer_use.tool as cu_tool
    except Exception:
        logger.debug("live-glass approval bridge: cannot import computer_use tool")
        return

    install_bridge(
        callback_getter=lambda: cu_tool._approval_callback,
        callback_setter=cu_tool.set_approval_callback,
    )
