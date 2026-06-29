"""immediate_ack plugin.

Bridges the ``platforms.slack.extra.immediate_ack`` config flag to actual
behaviour: when the gateway starts processing an incoming Slack message,
post a short text ack (``On it…`` by default) to the same channel/thread,
then on session end replace the ack with a status marker (✅ on success,
❌ on failure/cancellation) using the same Slack adapter.

Why a plugin (and not a core change):

- The Slack adapter already does 👀→✅/❌ *reaction* swap automatically via
  ``BasePlatformAdapter.on_processing_complete`` (slack.py:1385-1400), but
  it does **not** post a visible *text* ack. Operators reading a quiet
  channel want a textual "On it…" so the user knows the bot saw them.
- All work is isolated under ``~/.hermes/plugins/immediate_ack/``; no
  edits to ``gateway/platforms/slack.py`` or any other core file.
- The plugin never raises — every entry point wraps its body in
  ``try/except`` so a failure here cannot crash the gateway dispatch loop.

Lifecycle
---------

``pre_gateway_dispatch`` (gateway/run.py:5649-5688) — fires once per
incoming MessageEvent before auth/pairing. The plugin reads the config
flag and posts the ack text only when enabled and only on Slack. The
incoming message's ``message_id`` is used as the tracker key so the
session-end cleanup can locate the ack we posted.

``on_session_end`` (run_agent.py:15246-15260 / cli.py:12905-12913) — fires
when the session terminates. We pop the tracker entry and post a single
follow-up ``✅`` or ``❌`` text. We do NOT delete the original ack because
the Slack adapter does not expose a public message-delete API; the
user-visible benefit (the user sees a clear "done" signal) outweighs the
minor channel clutter. The original ``On it…`` message remains visible
as a record of receipt.

Config (read at runtime, never written by the plugin)
-----------------------------------------------------

``platforms.slack.extra.immediate_ack`` — bool, default false. When true,
the plugin posts acks.

``platforms.slack.extra.immediate_ack_text`` — string, default "On it…".
The text body posted as the ack.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config access (read-only)
# ---------------------------------------------------------------------------

def _read_immediate_ack_config() -> Tuple[bool, str]:
    """Return ``(enabled, ack_text)`` from the live config.

    Reads the YAML directly so this works in any environment (gateway,
    CLI, test harness) without depending on a loaded config singleton.
    Failure to read returns ``(False, "On it…")`` — the plugin is
    opt-in and the default-off behaviour is the safe fallback.
    """
    try:
        import yaml  # type: ignore[import-untyped]

        # ``HERMES_HOME`` is the env var the gateway sets; fall back to
        # ``~/.hermes`` which is the canonical runtime root.
        hermes_home = os.getenv("HERMES_HOME") or os.path.expanduser("~/.hermes")
        config_path = os.path.join(hermes_home, "config.yaml")
        if not os.path.isfile(config_path):
            return False, "On it…"

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        extra = (
            cfg.get("platforms", {})
            .get("slack", {})
            .get("extra", {})
        )
        enabled = bool(extra.get("immediate_ack", False))
        ack_text = str(extra.get("immediate_ack_text", "On it…"))
        return enabled, ack_text
    except Exception as exc:
        logger.debug("immediate_ack: config read failed (%s); defaulting off", exc)
        return False, "On it…"


# ---------------------------------------------------------------------------
# State tracking — keyed by incoming MessageEvent.message_id
# ---------------------------------------------------------------------------

# Map: incoming_msg_id -> (channel_id, ack_msg_id, thread_ts)
_pending_acks: Dict[str, Tuple[str, str, Optional[str]]] = {}
_state_lock = threading.Lock()

_DONE_TEXT = "✅"
_FAILED_TEXT = "❌"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_slack_adapter(gateway: Any) -> Any:
    """Return the Slack adapter from ``gateway.adapters`` or None.

    Uses ``getattr`` defensively so the plugin doesn't crash on a
    partially-initialised gateway (e.g. mid-bootstrap tests).
    """
    try:
        from gateway.config import Platform  # type: ignore[import-not-found]
    except Exception:
        # If the gateway module isn't on the import path (CLI test), the
        # caller can still pass an adapter via the legacy ``gateway.adapters``
        # dict whose keys are the enum members. Try string match fallback.
        Platform = None  # type: ignore[assignment]

    adapters = getattr(gateway, "adapters", None)
    if not adapters:
        return None

    if Platform is not None:
        adapter = adapters.get(Platform.SLACK)
        if adapter is not None:
            return adapter

    # Fallback: scan by class name or platform value
    for key, adapter in adapters.items():
        key_name = getattr(key, "value", str(key)).lower()
        cls_name = type(adapter).__name__.lower()
        if key_name == "slack" or "slack" in cls_name:
            return adapter
    return None


def _tracker_key(event: Any) -> str:
    """Stable key for tracking the ack we posted for ``event``."""
    msg_id = getattr(event, "message_id", None) or getattr(
        getattr(event, "source", None), "message_id", None
    )
    if msg_id:
        return str(msg_id)
    # Fall back to a synthetic key so we never collide
    return f"evt:{id(event)}"


def _find_runtime_adapter() -> Any:
    """Best-effort: locate the live Slack adapter from the gateway runtime.

    The plugin manager singleton holds hooks but not a gateway ref. Try
    the well-known global module references the gateway publishes at
    startup. Return None on any failure — cleanup is best-effort.
    """
    try:
        import sys
        # Current upstream stores the active runner as a weakref named
        # ``_gateway_runner_ref``; older forks exposed direct module globals.
        candidates = [
            ("gateway.run", "gateway_runner"),
            ("gateway.run", "runner"),
            ("gateway.run", "_active_runner"),
            ("gateway.run", "_gateway_runner_ref"),
        ]
        for mod_name, attr in candidates:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            runner = getattr(mod, attr, None)
            if runner is None:
                continue
            if callable(runner) and attr.endswith("_ref"):
                runner = runner()
                if runner is None:
                    continue
            adapter = _resolve_slack_adapter(runner)
            if adapter is not None:
                return adapter
    except Exception as exc:
        logger.debug("immediate_ack: runtime adapter lookup failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Hook: pre_gateway_dispatch
# ---------------------------------------------------------------------------

async def _on_pre_gateway_dispatch(  # noqa: F811
    event: Any,
    gateway: Any,
    session_store: Any,
) -> Optional[Dict[str, Any]]:
    """DEPRECATED async shim — kept for backwards compat, defers to sync."""
    return _on_pre_gateway_dispatch_sync(event, gateway, session_store)


def _on_pre_gateway_dispatch_sync(
    event: Any,
    gateway: Any,
    session_store: Any,
) -> Optional[Dict[str, Any]]:
    """Post text ack if config flag is on and platform is Slack.

    PluginManager.invoke_hook calls hooks synchronously (does not await).
    The hook must therefore be a sync ``def``; async Slack adapter calls
    are bridged via ``_run_coro_in_fresh_loop`` which creates a dedicated
    event loop for the HTTP request.

    Returns ``None`` (allow normal dispatch) in all cases — the plugin
    must never gate the gateway dispatch loop on plugin-internal state.
    """
    try:
        enabled, ack_text = _read_immediate_ack_config()
        if not enabled:
            return None

        adapter = _resolve_slack_adapter(gateway)
        if adapter is None:
            return None

        source = getattr(event, "source", None)
        if source is None:
            return None

        # Bot-id check: a Slack user-token post that uses the workspace's
        # app config will still get `bot_id` set on the event (because the
        # message goes through the workspace's app). To avoid the
        # CLI-posted-test case where the user-token post is treated as a
        # bot event, we check `event.bot_id` directly (not `user_id`).
        # The original `user_id.startswith("B")` check was wrong — Slack
        # user ids start with "U" and bot ids start with "B" but
        # `source.user_id` is the AUTHOR's user, not the bot.
        bot_id = (
            getattr(event, "bot_id", None)
            or (getattr(event, "raw_message", None) or {}).get("bot_id")
            or (getattr(source, "bot_id", None) if source is not None else None)
        )
        if bot_id:
            # Bot-originated message — skip the plugin's ack (the gateway's
            # own Slack adapter already filtered these at the dispatcher,
            # but we double-check here in case the dispatcher logic was
            # changed to allow them).
            return None

        chat_id = getattr(source, "chat_id", None)
        if not chat_id:
            return None

        # Threading: the gateway's source object stores the thread anchor
        # under `thread_id` (per `_thread_metadata_for_source`), NOT `message_id`.
        # Using `source.message_id` returns None for top-level Slack messages
        # because the gateway uses a different field for that. For Slack
        # top-level channel messages, `source.thread_id` is set to the message
        # ts as a synthetic-thread fallback so replies thread under the
        # user's own message.
        thread_ts = getattr(source, "thread_id", None) or getattr(source, "message_id", None)

        result = _run_coro_in_fresh_loop(adapter.send(
            chat_id=chat_id,
            content=ack_text or "On it…",
            reply_to=thread_ts,
            metadata={"thread_id": thread_ts, "thread_ts": thread_ts},
        ))

        if getattr(result, "success", False) and getattr(result, "message_id", None):
            key = _tracker_key(event)
            with _state_lock:
                _pending_acks[key] = (chat_id, str(result.message_id), thread_ts)
            logger.debug(
                "immediate_ack: posted ts=%s for chat=%s thread=%s key=%s",
                result.message_id,
                chat_id,
                thread_ts,
                key,
            )
        else:
            err = getattr(result, "error", "unknown")
            logger.debug("immediate_ack: send returned not-ok (%s)", err)

    except Exception as exc:
        # NEVER crash the gateway dispatch loop
        logger.debug("immediate_ack: pre_gateway_dispatch failed: %s", exc)

    return None


# ---------------------------------------------------------------------------
# Hook: on_session_end
# ---------------------------------------------------------------------------

async def _on_session_end(**kwargs: Any) -> None:  # noqa: F811
    """DEPRECATED async shim — kept for backwards compat, defers to sync."""
    _on_session_end_sync(**kwargs)


def _on_session_end_sync(**kwargs: Any) -> None:
    """Replace the ack text with a status marker when the session ends.

    ``on_session_end`` receives ``session_id`` (and optionally other
    metadata depending on caller) — but we don't know the
    ``message_id`` here directly. We use a *first-in / first-out* pop
    strategy scoped to the session: at most one ack per session, which
    matches the gateway's per-message-session semantics.

    If multiple acks accumulated (shouldn't normally happen), the
    earliest one is closed first to keep ordering predictable.
    """
    try:
        session_id = kwargs.get("session_id") or kwargs.get("session_key") or ""

        with _state_lock:
            target_key = None
            for key, value in list(_pending_acks.items()):
                if session_id and key in session_id:
                    target_key = key
                    break
            if target_key is None and _pending_acks:
                target_key = next(iter(_pending_acks), None)

            if target_key is None:
                return

            pending = _pending_acks.pop(target_key)

        adapter = _find_runtime_adapter()
        if adapter is None:
            return

        chat_id, _ack_ts, thread_ts = pending
        outcome = kwargs.get("outcome") or kwargs.get("status") or "success"
        marker = _FAILED_TEXT if str(outcome).lower() in (
            "failure", "failed", "error", "cancelled", "timeout"
        ) else _DONE_TEXT

        _run_coro_in_fresh_loop(adapter.send(
            chat_id=chat_id,
            content=marker,
            reply_to=thread_ts,
            metadata={"thread_id": thread_ts, "thread_ts": thread_ts},
        ))
        logger.debug(
            "immediate_ack: posted marker %s for chat=%s thread=%s",
            marker, chat_id, thread_ts,
        )

    except Exception as exc:
        logger.debug("immediate_ack: on_session_end failed: %s", exc)


def _run_coro_in_fresh_loop(coro: Any) -> Any:
    """Run an async coroutine to completion from a sync context that may be
    inside a running event loop.

    The plugin hook is invoked synchronously by PluginManager.invoke_hook
    (which does not await). To run a Slack adapter call (which is
    ``async def``) we need a real event loop. BUT — the gateway's main
    loop is already running by the time the hook fires, so a naive
    ``loop.run_until_complete(coro)`` fails with
    "Cannot run the event loop while another loop is running".

    Strategy: run the coroutine in a SEPARATE THREAD with its own event
    loop. The Slack adapter's HTTP calls (chat.postMessage, etc.) are
    independent and don't share state with the gateway's main loop, so
    a separate thread is safe.

    This pattern is the standard asyncio "run sync code that needs async"
    idiom and works whether or not we're inside a running loop.
    """
    import asyncio
    import concurrent.futures

    def _runner() -> Any:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_runner)
        return future.result()


# ---------------------------------------------------------------------------
# Plugin registration (called by hermes_cli.plugins)
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Register this plugin's hooks with the host (SYNC).

    Per F3v2 verification 2026-06-28: PluginManager.invoke_hook calls
    hooks SYNCHRONOUSLY (does not await). If the registered callbacks
    are ``async def``, the returned coroutine is appended to
    ``_hook_results`` and discarded — the plugin body never runs.
    We therefore register the SYNC wrappers explicitly.
    """
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch_sync)
    ctx.register_hook("on_session_end", _on_session_end_sync)
    logger.debug("immediate_ack: registered hooks (sync)")
