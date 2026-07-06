"""KarinAI request-scoped session ContextVars (gateway session bridge).

These vars used to live inline in ``gateway/session_context.py``; they moved
here to shrink the recurring upstream-sync conflict surface (see
``docs/karinai-gateway-bridge-design.md``). ``gateway/session_context.py``
keeps a thin registration hook: it calls
:func:`register_karinai_session_vars` with its ``_UNSET`` sentinel and merges
:data:`KARINAI_SESSION_VARS` into its ``_VAR_MAP``, so ``get_session_env()``,
``reset_session_vars()`` and ``clear_session_vars()`` cover these vars exactly
as before (including the thread-reuse leak safety: the api_server's
``_run_sync`` executes on a ThreadPoolExecutor thread, and without the
``clear_session_vars`` coverage the next run on that thread would inherit a
stale product_run_id / gateway token).

IMPORTANT: this module MUST NOT import ``gateway.*`` at module level —
``gateway.session_context`` imports this module at import time, so the
module-level dependency edge points gateway → karinai only. (Lazy in-function
gateway imports are fine.)
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

# name → ContextVar registry, merged into ``gateway.session_context._VAR_MAP``
# by the registration hook there. Populated by
# :func:`register_karinai_session_vars`; defined at module level so the dict
# OBJECT session_context imports (before registration runs) is the same one
# the hook and every later reader see.
KARINAI_SESSION_VARS: Dict[str, ContextVar] = {}


def register_karinai_session_vars(unset: Any) -> Dict[str, ContextVar]:
    """Create (once) the KarinAI ContextVars and return the registry.

    ``unset`` MUST be ``gateway.session_context._UNSET``: ``get_session_env``
    compares var values against that exact sentinel object to decide whether
    to fall back to ``os.environ`` (CLI/cron/dev compat), and
    ``reset_session_vars`` restores it. A var minted with any other default
    would leak the sentinel object out of ``get_session_env`` on never-bound
    reads instead of falling back. Idempotent — repeated calls return the
    already-populated registry.
    """
    if KARINAI_SESSION_VARS:
        return KARINAI_SESSION_VARS

    # Backend-assigned PRODUCT run id for this turn — the value of the
    # ``X-KarinAI-Run-Id`` HTTP header the KarinAI backend sends to the
    # api_server on POST /v1/runs. This is NOT the agent's own
    # internally-minted run id (``run_<uuid>``); it is the backend's run id,
    # and its post-run sweep collects durable artifacts from
    # ``<workspace>/outputs/<product_run_id>/``. The ``register_artifact``
    # tool reads this to stage deliverables under that exact directory.
    # Default ``_UNSET`` => empty for CLI/cron/dev, which the tool treats as
    # "no managed run" and degrades gracefully.
    KARINAI_SESSION_VARS["HERMES_PRODUCT_RUN_ID"] = ContextVar(
        "HERMES_PRODUCT_RUN_ID", default=unset
    )

    # Backend-owned app/tool gateway credentials for a single KarinAI managed
    # run. These are deliberately request-scoped and are never read from
    # product metadata or model-visible messages.
    for name in (
        "KARINAI_APP_TOOL_GATEWAY_URL",
        "KARINAI_APP_TOOL_GATEWAY_TOKEN",
        "KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT",
    ):
        KARINAI_SESSION_VARS[name] = ContextVar(name, default=unset)
    return KARINAI_SESSION_VARS


def mask_karinai_run_context() -> List:
    """Pin every KarinAI request-scoped var to ``""`` for the current turn.

    ``gateway.session_context.set_session_vars`` calls this for EVERY host
    (Telegram/Discord/cron/TUI/ACP/api_server), restoring the pre-bridge
    invariant that a bound session turn NEVER falls back to ``os.environ``
    for these names: without the mask, a process whose environment happens to
    contain e.g. ``KARINAI_APP_TOOL_GATEWAY_TOKEN`` (dev export, .env leak)
    would advertise and execute app tools with those credentials in ordinary
    messaging/cron sessions. The api_server rebinds real values AFTER the mask
    via :func:`bind_karinai_run_context` (its tokens ride the same list).

    Returns reset tokens for the caller's token list.
    """
    return [var.set("") for var in KARINAI_SESSION_VARS.values()]


def bind_karinai_run_context(
    product_run_id: str = "",
    app_tool_gateway: Optional[Dict[str, Any]] = None,
) -> List:
    """Bind the KarinAI request-scoped vars for one managed /v1/runs turn.

    Returns reset tokens to be concatenated onto the caller's
    ``set_session_vars()`` token list (the api_server's
    ``_bind_api_server_session`` does exactly that), so the KarinAI vars share
    the session vars' lifecycle: cleared by ``clear_session_vars`` in the
    caller's ``finally`` block when the turn ends.

    Value coercion matches what the api_server session binding always did:
    ``product_run_id`` is passed through as-is (empty when absent — dev /
    non-managed callers) and the ``app_tool_gateway`` dict fields are
    stringified with ``""`` fallbacks; a non-dict is treated as absent.
    """
    if not KARINAI_SESSION_VARS:
        # Registration happens when gateway.session_context is imported (it
        # supplies the shared _UNSET sentinel). Lazy import only — see the
        # module docstring for the dependency-direction rule.
        import gateway.session_context  # noqa: F401

    app_gateway = app_tool_gateway if isinstance(app_tool_gateway, dict) else {}
    return [
        KARINAI_SESSION_VARS["HERMES_PRODUCT_RUN_ID"].set(product_run_id),
        KARINAI_SESSION_VARS["KARINAI_APP_TOOL_GATEWAY_URL"].set(str(app_gateway.get("url") or "")),
        KARINAI_SESSION_VARS["KARINAI_APP_TOOL_GATEWAY_TOKEN"].set(str(app_gateway.get("token") or "")),
        KARINAI_SESSION_VARS["KARINAI_APP_TOOL_GATEWAY_EXPIRES_AT"].set(
            str(app_gateway.get("expires_at") or "")
        ),
    ]
